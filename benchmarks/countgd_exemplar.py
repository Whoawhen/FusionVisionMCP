#  countgd_exemplar.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""A REJECTED spike: CountGD's visual-exemplar mode on HF's Grounding DINO graph.

**This does not ship.** Kept so the result is reproducible; not imported from `src/`.

CountGD's headline capability is counting from a visual exemplar -- point at one
instance, find the rest -- which the text-only port in `countgd_spike.py` could not
exercise. This implements the fusion path: concatenate the backbone's last three feature
maps at stride 8 (256+512+1024 = 1792 channels), project to 256 with CountGD's
`feature_map_proj`, ROI-align at the exemplar box to a single 1x1 token, and insert that
token into the BERT text sequence, where it behaves as an extra phrase.

HF computes text features inside `GroundingDinoModel.forward`, so rather than rewrite
that, this reserves placeholder phrases in the prompt and overwrites their projected
features -- which keeps every derived mask the right length. CountGD does the same thing
with a placeholder token id.

## It works, and it is wildly imprecise

The exemplar tokens are demonstrably in use: detection scores rise from ~0.35 text-only
to ~0.95 with an exemplar, which a silent no-op could not produce. But precision
collapses. Measured with one exemplar box per image:

| case | truth | text-only | with exemplar |
| --- | --- | --- | --- |
| ring, 67% overlap | 8 | 12 | 9 |
| ring, 40% overlap | 8 | **8** | 9 |
| flower @384 | ? | 4 | **18-20** |
| NEG solid disc | 1 | 1 | 2 |
| NEG striped ball | 1 | 1 | **12** |
| NEG spotted ball | 1 | 1 | **31** |
| NEG four-colour logo | 1 | 2 | 7 |
| NEG distractors | 3 | 5 | 7 |

Five of six negative controls break, the spotted ball worst of all at 31 objects. The
flower's 18-20 is not a petal count by any reading. Exemplar mode buys recall and spends
all of its precision doing so.

## An honest caveat about what this proves

This is a *reimplementation* of the fusion path, not CountGD's own code. The uniformly
saturated ~0.95 scores are consistent with exemplar tokens dominating the text
attention, which could mean the phrase-isolation masking here is not faithful to theirs.
So this is weaker evidence than the text-only comparison in `countgd_spike.py`, which
used HF's official conversion and changed nothing but the checkpoint.

What it does establish is that the cheap route to exemplar counting does not work.
Reaching a fair verdict on CountGD's exemplar mode would mean running their repository
as published, with its compiled operators and pinned Python -- which is what this
project's constraints rule out in the first place.

Run this module's `ExemplarCountGD` from a script; there is no CLI entry point.
"""

import json
import struct
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.ops import roi_align

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, r"c:\AI\MCP\FusionVisionMCP")
sys.path.insert(0, r"c:\AI\MCP\FusionVisionMCP\src")
sys.path.insert(0, str(SCRATCH))


#: CountGD concatenates at stride 8 and ROI-aligns with this spatial scale.
SPATIAL_SCALE = 1.0 / 8.0


def load_feature_map_proj() -> torch.nn.Conv2d:
    """The one CountGD-specific layer, which the HF port leaves unloaded."""
    path = hf_hub_download("nikigoli/CountGD", "model.safetensors")
    state = load_file(path)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        json.loads(fh.read(n))
    weight = state["feature_map_proj.weight"]
    bias = state["feature_map_proj.bias"]
    conv = torch.nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=1)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    conv.eval()
    return conv


class ExemplarCountGD:
    """Text-plus-exemplar counting on the ported graph."""

    def __init__(self, model: Any, processor: Any, feature_map_proj: torch.nn.Conv2d) -> None:
        self.model, self.processor, self.feature_map_proj = model, processor, feature_map_proj

    def _exemplar_tokens(self, pixel_values: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """One 256-d token per exemplar box, by ROI-align on the projected feature map."""
        inner = self.model.model
        with torch.inference_mode():
            vision_features, _ = inner.backbone(pixel_values, torch.ones_like(pixel_values[:, 0], dtype=torch.long))
            maps = [source for source, _mask in vision_features]
            target_hw = maps[0].shape[-2:]
            stacked = torch.cat(
                [m if m.shape[-2:] == target_hw else F.interpolate(m, size=target_hw, mode="bilinear") for m in maps],
                dim=1,
            )
            projected = self.feature_map_proj(stacked)
            rois = [boxes_xyxy.to(projected.dtype)]
            tokens = roi_align(projected, rois, output_size=(1, 1), spatial_scale=SPATIAL_SCALE, aligned=True)
        return tokens.squeeze(-1).squeeze(-1)  # (num_exemplars, 256)

    def detect(
        self,
        image: Any,
        object_name: str,
        exemplar_boxes: list[list[float]],
        threshold: float = 0.15,
        text_threshold: float = 0.25,
    ) -> dict[str, Any]:
        rgb = image.convert("RGB")
        n = len(exemplar_boxes)
        # Reserve one placeholder phrase per exemplar; each becomes its own text phrase,
        # so the model's phrase-separated attention mask already isolates them.
        prompt = object_name.strip().lower().rstrip(".") + "." + " thing." * n
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt")

        scale_x = inputs["pixel_values"].shape[-1] / rgb.width
        scale_y = inputs["pixel_values"].shape[-2] / rgb.height
        boxes = torch.tensor(
            [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y] for b in exemplar_boxes],
            dtype=torch.float32,
        )
        tokens = self._exemplar_tokens(inputs["pixel_values"], boxes)

        # Overwrite the last n text positions with the exemplar tokens.
        original = self.model.model.text_projection
        seq_len = int(inputs["attention_mask"].sum())

        class Patched(torch.nn.Module):
            def __init__(self, inner_mod):
                super().__init__()
                self.inner = inner_mod

            def forward(self, x):
                out = self.inner(x)
                for i in range(n):
                    pos = seq_len - 2 - i  # skip the trailing separator
                    if 0 <= pos < out.shape[1]:
                        out[:, pos, :] = tokens[n - 1 - i].to(out.dtype)
                return out

        self.model.model.text_projection = Patched(original)
        try:
            with torch.inference_mode():
                raw = self.model(**inputs)
        finally:
            self.model.model.text_projection = original

        processed = self.processor.post_process_grounded_object_detection(
            raw,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(rgb.height, rgb.width)],
        )[0]
        return {
            "count": len(processed["boxes"]),
            "bboxes": [[float(v) for v in b] for b in processed["boxes"].tolist()],
            "scores": [float(s) for s in processed["scores"].tolist()],
        }
