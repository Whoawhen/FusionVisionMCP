"""A learned check for AI-generation defects on hands and faces.

The part count in `inspection._check_anatomy` can only say "too many": a fused, melted or
misshapen hand with the right number of parts passes it. On VisionBench's user-reviewed round 7
(149 images, 49 defective) it flagged 38.8% of defective images and 30.0% of clean ones, which
is close to chance.

This probe reads the crops instead. Grounding DINO finds every `hand` and `face`; each crop
(padded 20 px) goes through CLIP ViT-L/14, the backbone `score_aesthetics` already loads, and a
logistic regression trained on round 7's reviewed hand and face crops scores it: the defect
boxes, plus the user's own call on each crop those boxes did not cover (2026-10-08; 261 defect
and 772 clean crops). The image is flagged when its highest crop score passes the threshold,
which was chosen on round 7 alone to leave 85% of clean images unflagged. Measured (each image
scored by a probe that never saw it): 67.3% of defective images flagged at 85.0% specificity on
round 7, and on the held-out round 8 (the user's own DreamShaperXL images, never tuned on) 73.1%
at 80.0% (5 clean images only). Through `structured_analysis` on VisionBench's 94 unseen images
(round 8, phase 10, phase 11): 72.3% of defective images flagged, 63.8% of clean left alone.

More training data did not help: adding round 9 (246 Stable Diffusion 1.x images) dropped that to
51.1% / 76.6% (AUC 0.741 -> 0.699). The CLIP crop probe has plateaued on these labels.

The weights are `data/defect_probe.json`, written by VisionBench's `export_defect_probe.py`
with the training set and measurements recorded beside them. This module is numpy- and
torch-free so it can sit on the eager import path; the embeddings come from the aesthetic
wrapper at call time.
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
    """Probability that a crop shows a generation defect, from its unnormalised CLIP image embedding."""
    logit = probe["intercept"] + sum(w * x for w, x in zip(probe["coef"], embedding, strict=True))
    return 1.0 / (1.0 + math.exp(-logit))
