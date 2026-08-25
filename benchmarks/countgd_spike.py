#  countgd_spike.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""A REJECTED spike: CountGD as a drop-in counting backend.

**This does not ship. It is kept so the result is reproducible rather than asserted.**
Do not import it from `src/`.

CountGD (MIT, ungated, `nikigoli/CountGD`, 938MB safetensors) is a counting model
derived from Grounding DINO, and was the counting plan's recommended candidate. The
spike answered two questions: can it be ported here at all, and is it better?

## Porting: yes, and more easily than expected

The published repository requires GCC 11.3+, Python 3.9.19 under Anaconda, compiled
GroundingDINO CUDA ops and three separate checkpoints -- all disqualifying. None of that
turned out to be necessary, because the *checkpoint* is close to stock Grounding DINO:

- Its tensor names are the original GroundingDINO scheme, and HF publishes an official
  original-to-HF rename mapping (`convert_grounding_dino_to_hf.py`), reused here.
- The backbone is Swin-B (`patch_embed.proj.weight` is [128, 3, 4, 4]), 900 queries,
  bert-base-uncased -- matching HF's `grounding-dino-base` config exactly.
- The only architectural addition is **one** 1x1 conv, `feature_map_proj` (1792->256,
  i.e. Swin-B's last three stage widths concatenated). That is the visual-exemplar ROI
  encoder, and it is unused in text-only mode.

So the weights load onto stock `GroundingDinoForObjectDetection` with **zero substantive
missing keys**, and run on CPU with no compiled ops in ~2.9s.

One trap worth recording: safetensors deduplicates shared tensors, so the six
`bbox_embed` heads are stored once with every alias listed in `__metadata__`. Load the
file naively and 66 decoder box-head tensors are silently missing, leaving those heads
at random initialisation. HF happens to tie those weights so the outputs looked correct
anyway -- which is exactly how this kind of bug survives review. Restore the aliases
from the metadata before converting.

## Verdict: worse than what is already shipped

Measured on the same fixtures, with the same post-processing and envelope filter, so the
checkpoint is the only variable:

| | grounding-dino-tiny (shipped) | CountGD (ported, text-only) |
| --- | --- | --- |
| Positives exact | **9/10** | 8/10 |
| Negative controls held | **8/8** | **4/8** |
| Mean absolute count error | **0.10** | 0.75 |
| Warm CPU latency | **2.11s** | 2.87s |
| Weights | **690MB** | 938MB |

It breaks four controls that the shipped detector holds: a four-colour logo counts as 2,
a finely textured blob as 2, a scene of three discs among square distractors as 5, and a
rough-edged rod as **0** -- it misses the object altogether.

The flower is the clearest disqualification. The *same image* at three resolutions gives
11, 4 and 5. A stable count cannot depend on resampling; that spread is noise, not a
measurement, and the 11 at 128px is a coincidence rather than a near-miss on the true
count.

## What this does and does not rule out

This tested **text-only** mode. CountGD's headline capability is *visual exemplars* --
point at one instance, find the rest -- which HF's graph has no path for, since the
forward pass fuses exemplar tokens alongside text tokens and `feature_map_proj` is never
called. Exercising that would mean implementing the fusion path, a genuine research port
rather than a weight remap.

Two things temper the appeal of doing so. Text-only CountGD is *worse* than stock
Grounding DINO on these controls, which is weak evidence about the checkpoint generally.
And CountGD is fine-tuned on FSC-147, photographs of many small real objects, so some of
the degradation here is plausibly domain shift onto synthetic shapes -- a caveat that
cuts both ways, since it also means its published accuracy may not transfer to this
server's inputs.

Run `python -m benchmarks.countgd_spike` from the repository root to reproduce.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

COUNTGD_REPO = "nikigoli/CountGD"
#: The conversion mapping lives in the transformers repository rather than the wheel.
#: Pinned to a tag rather than `main`: the mapping itself is stable, but `main`'s copy
#: has since grown an `httpx` import this project does not carry.
CONVERT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/v4.46.0/"
    "src/transformers/models/grounding_dino/convert_grounding_dino_to_hf.py"
)


def _load_conversion_module() -> Any:
    """Fetch HF's official conversion script, which is not shipped in the wheel."""
    import importlib.util

    import requests

    cache = Path(__file__).resolve().parent / "_convert_grounding_dino_to_hf.py"
    if not cache.exists():
        response = requests.get(CONVERT_SCRIPT_URL, timeout=120)
        response.raise_for_status()
        cache.write_bytes(response.content)

    spec = importlib.util.spec_from_file_location("_convert_gd", cache)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_ported_countgd() -> tuple[Any, Any]:
    """Load CountGD's weights onto HF's Grounding DINO graph, CPU-only."""
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    convert = _load_conversion_module()
    path = hf_hub_download(COUNTGD_REPO, "model.safetensors")
    state = load_file(path)

    # Restore deduplicated shared tensors, or the decoder's box heads stay randomly
    # initialised while appearing to load cleanly.
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    for alias, canonical in header.get("__metadata__", {}).items():
        if alias not in state and canonical in state:
            state[alias] = state[canonical].clone()

    config = convert.get_grounding_dino_config("grounding-dino-base")
    renamed = dict(state)
    for src, dest in convert.create_rename_keys(state, config):
        convert.rename_key(renamed, src, dest)
    convert.read_in_q_k_v_encoder(renamed, config)
    convert.read_in_q_k_v_text_enhancer(renamed, config)
    convert.read_in_q_k_v_decoder(renamed, config)

    model = GroundingDinoForObjectDetection(config)
    model.eval()
    missing, _ = model.load_state_dict(renamed, strict=False)
    substantive = [k for k in missing if "position_ids" not in k and "relative_position_index" not in k]
    if substantive:
        raise RuntimeError(f"port incomplete: {len(substantive)} substantive tensors missing, e.g. {substantive[:5]}")

    return model, AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")


class CountGDBackend:
    """Matches the shipped detector's contract, filter and thresholds exactly."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model, self.processor = model, processor

    def detect_objects(self, images: list[Any], object_name: str) -> list[dict[str, Any]]:
        from fusion_vision_mcp.grounding_dino import (
            DEFAULT_BOX_THRESHOLD,
            DEFAULT_TEXT_THRESHOLD,
            GroundingDino,
        )

        prompt = GroundingDino._as_prompt(object_name)
        results = []
        for image in images:
            rgb = image.convert("RGB")
            inputs = self.processor(images=rgb, text=prompt, return_tensors="pt")
            with torch.inference_mode():
                raw = self.model(**inputs)
            processed = self.processor.post_process_grounded_object_detection(
                raw,
                inputs["input_ids"],
                threshold=DEFAULT_BOX_THRESHOLD,
                text_threshold=DEFAULT_TEXT_THRESHOLD,
                target_sizes=[(rgb.height, rgb.width)],
            )[0]
            bboxes = [[float(v) for v in box] for box in processed["boxes"].tolist()]
            scores = [float(s) for s in processed["scores"].tolist()]
            envelopes = GroundingDino._envelope_indices(bboxes)
            keep = [i for i in range(len(bboxes)) if i not in envelopes]
            bboxes = [bboxes[i] for i in keep]
            results.append(
                {
                    "count": len(bboxes),
                    "bboxes": bboxes,
                    "points": [[(a + c) / 2, (b + d) / 2] for a, b, c, d in bboxes],
                    "labels": [object_name] * len(bboxes),
                    "scores": [scores[i] for i in keep],
                    "group_boxes_dropped": len(envelopes),
                }
            )
        return results


def main() -> int:
    from benchmarks.fixtures import all_fixtures
    from benchmarks.harness import render, run
    from fusion_vision_mcp.grounding_dino import GroundingDino

    fixtures = all_fixtures()

    print("loading ported CountGD...")
    started = time.time()
    model, processor = load_ported_countgd()
    print(f"  cold load {time.time() - started:.1f}s")

    ported = run(CountGDBackend(model, processor), fixtures, label="CountGD (ported)", include_equivalents=False)
    print(render(ported))

    del model
    shipped = run(GroundingDino(), fixtures, label="grounding-dino-tiny (shipped)", include_equivalents=False)

    print("\n=== side by side ===")
    print(f"{'fixture':<24}{'truth':>6}{'shipped':>9}{'countgd':>9}")
    print("-" * 50)
    by_name = {r["fixture"]: r for r in shipped["results"]}
    for r in ported["results"]:
        base = by_name[r["fixture"]]
        truth = "?" if r["truth"] is None else r["truth"]
        print(f"{r['fixture']:<24}{truth:>6}{base['predicted']:>9}{r['predicted']:>9}")

    for tag, report in (("shipped", shipped), ("countgd", ported)):
        s = report["summary"]
        print(
            f"\n{tag:<8} positives {s['positives_exact']}/{s['positives_total']}  "
            f"negatives {s['negatives_held']}/{s['negatives_total']}  MAE {s['mean_abs_error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
