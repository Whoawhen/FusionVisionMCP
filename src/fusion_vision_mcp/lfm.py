"""LiquidAI LFM2.5-VL-3B, the long captioner behind `caption(deep=true)` since 2026-10-22.

Runs as VisionBench `caption_track4.py` measured it (variant `lfm3_t4`): see DEFAULT_DEEP_CAPTION_MODEL in
`constants` for the numbers and why each setting is what it is.
"""

from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .constants import (
    DEEP_CAPTION_MAX_NEW_TOKENS,
    DEEP_CAPTION_MAX_TILES,
    DEEP_CAPTION_PROMPT,
    DEFAULT_DEEP_CAPTION_MODEL,
    DEFAULT_DEEP_CAPTION_REVISION,
)
from .device import resolve_device

#: The model card's recommended text decoding.
_DECODING: dict[str, Any] = {"do_sample": True, "temperature": 0.1, "min_p": 0.15, "repetition_penalty": 1.05}


class LfmCaptioner:
    """Long image captions from LFM2.5-VL-3B (Liquid AI: LFM2.5 language model, SigLIP2 vision encoder)."""

    device: str
    model: Any
    processor: Any

    def __init__(
        self,
        model_id: str = DEFAULT_DEEP_CAPTION_MODEL,
        revision: str = DEFAULT_DEEP_CAPTION_REVISION,
        device: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        # Up to 4 tiles of 512 px plus a thumbnail: about 780 image tokens instead of 1,800 at the default 10.
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, max_tiles=DEEP_CAPTION_MAX_TILES)
        # `transformers` wraps the auto classes in a decorator that mypy reads as taking a model instance first.
        model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
        self.model = model.to(self.device).eval()  # type: ignore[arg-type]

    def caption(self, images: list[Image]) -> list[str]:
        res = []
        for img in images:
            with img.convert("RGB") as rgb_img:
                conv = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": rgb_img}, {"type": "text", "text": DEEP_CAPTION_PROMPT}],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=DEEP_CAPTION_MAX_NEW_TOKENS, **_DECODING)
            new = out[0, inputs["input_ids"].shape[1] :]
            res.append(self.processor.decode(new, skip_special_tokens=True).strip())
        return res
