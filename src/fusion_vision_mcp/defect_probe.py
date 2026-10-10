"""A learned check for AI-generation defects on hands and faces.

The part count in `inspection._check_anatomy` can only say "too many": a fused, melted or
misshapen hand with the right number of parts passes it. On VisionBench's user-reviewed round 7
(149 images, 49 defective) it flagged 38.8% of defective images and 30.0% of clean ones, which
is close to chance.

This probe reads the crops instead. Grounding DINO finds every `hand` and `face`; each crop
(padded 20 px) is embedded by the backbone the probe file names, and a logistic regression
trained on round 7's reviewed hand and face crops scores it: the defect boxes, plus the user's
own call on each crop those boxes did not cover (261 defect and 772 clean crops). The image is
flagged when its highest crop score passes the threshold, chosen on round 7 alone to leave 85%
of clean images unflagged.

**Since 2026-10-09 the backbone is DINOv2-large** (`dinov2.py`; CLS token + mean patch token,
2,048 features). It replaced CLIP ViT-L/14 on VisionBench's round 10, 299 user-graded images
from six generators no probe trained on (DALL-E 2, Midjourney, RealVisXL, SDXL base, FLUX.2
Klein, SANA), by a rule fixed before the run: image AUC 0.686 against CLIP's 0.612 (+0.074,
95% interval 0.013-0.135); at each probe's own threshold, 44.8% of defective images caught and
84.1% of clean left alone, against 47.4% / 68.3%. Even so, an AUC near 0.7 means a flag is a
lead to check, not a verdict.

Two earlier findings still stand. More of the same labels did not help: adding round 9 (246
Stable Diffusion 1.x images) made the CLIP probe worse on unseen images. And DINOv2-base scored
as well as large on round 10 (0.709) at a quarter of the size, but it was not the candidate named
before the run; it waits for a fresh test set (`Future_Ideas/frontier_capability_gaps.md`).

The weights are `data/defect_probe.json`, written by VisionBench's `export_defect_probe_dinov2.py`
with the training set, the backbone and every measurement recorded beside them. A probe file with
`"backbone": "clip_image_embeds"` (or no backbone field) still runs on CLIP, so rolling back is a
data-file change. This module is numpy- and torch-free so it can sit on the eager import path;
the features come from the backbone wrapper at call time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from typing import Any


@cache
def load_probe() -> dict[str, Any]:
    probe: dict[str, Any] = json.loads(files("fusion_vision_mcp").joinpath("data/defect_probe.json").read_text("utf-8"))
    return probe


def padded_box(box: Sequence[float], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    """The crop the probe was trained on: the detector box widened by `pad` pixels, clipped to the image."""
    return (
        max(0, int(box[0]) - pad),
        max(0, int(box[1]) - pad),
        min(width, int(box[2]) + pad),
        min(height, int(box[3]) + pad),
    )


def score(embedding: Sequence[float], probe: dict[str, Any]) -> float:
    """Probability that a crop shows a generation defect, from the backbone features the probe was trained on."""
    logit = probe["intercept"] + sum(w * x for w, x in zip(probe["coef"], embedding, strict=True))
    return 1.0 / (1.0 + math.exp(-logit))
