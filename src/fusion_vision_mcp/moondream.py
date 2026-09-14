#  moondream.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from .constants import DEFAULT_MOONDREAM_MODEL, DEFAULT_MOONDREAM_REVISION
from .device import resolve_device


class Moondream:
    """Wraps Moondream2 for free-form visual question answering (VQA).

    Florence-2 has no open-ended VQA task token, so this second, smaller model is
    kept loaded alongside Florence2 specifically for `query`.

    This model's own detection head used to back `count_objects` too. It was replaced
    by Grounding DINO after a measured head-to-head: the two tie on cleanly separated
    instances, but this one collapses to a single region when the same shapes are
    named differently (8 as `petal`, 1 as `pink circle`), and is roughly four times
    slower. See README_DETAILED.md for the full comparison.
    """

    device: str
    model: Any
    revision: str

    def __init__(
        self,
        model_id: str = DEFAULT_MOONDREAM_MODEL,
        revision: str = DEFAULT_MOONDREAM_REVISION,
        device: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.revision = revision
        torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

        # `transformers` wraps the auto classes in a decorator that mypy reads as taking a
        # model instance in the first slot, so the model id below looks like the wrong type.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device)  # type: ignore[arg-type]

    def query(self, images: list[Image], question: str) -> list[str]:
        res = []
        for img in images:
            with img.convert("RGB") as rgb_img:
                answer = self.model.query(rgb_img, question)["answer"]
                res.append(answer)
        return res
