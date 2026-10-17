from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from .constants import DEFAULT_MOONDREAM_MODEL, DEFAULT_MOONDREAM_REVISION, THEME_PROMPT
from .device import resolve_device
from .repetition import cut_loop, loop_start

#: Tokens between loop checks while streaming a caption; a check rescans the text so far.
_LOOP_CHECK_EVERY = 16


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
            # HfMoondream has no SDPA path: transformers raises ValueError on "sdpa"
            # and names "eager" as the supported alternative. Verified against 4.57.6.
            attn_implementation="eager",
        ).to(self.device)  # type: ignore[arg-type]

    def query(self, images: list[Image], question: str) -> list[str]:
        res = []
        for img in images:
            with img.convert("RGB") as rgb_img:
                answer = self.model.query(rgb_img, question)["answer"]
                res.append(answer)
        return res

    def caption(self, images: list[Image], length: str = "long", theme: bool = False) -> list[str]:
        """Moondream2's own captioner; `length` is "short", "normal" or "long".

        Streamed, and stopped once the caption loops (`repetition.loop_start`): the long caption
        otherwise repeats a phrase until the 768-token cap on ~7% of images, which is also its slowest
        tail. The repeated text is cut off.

        Each image is encoded once (`encode_image`: vision features plus the text model's prefilled
        cache) and the caption runs on that encoding. With `theme`, THEME_PROMPT is asked on the same
        encoding and its answer added as `Theme: ...`; reusing it makes the question ~2 s instead of
        ~13 s, with identical output (VisionBench `caption_c4_1.py`, 20 of 20 under greedy decoding).
        """
        res = []
        for img in images:
            with img.convert("RGB") as rgb_img:
                encoded = self.model.encode_image(rgb_img)
            text = self._caption_until_loop(encoded, length)
            if theme:
                answer = " ".join(self.model.query(encoded, THEME_PROMPT)["answer"].split()).rstrip(".")
                if answer:
                    text = f"{text.rstrip()} Theme: {answer}."
            res.append(text)
        return res

    def _caption_until_loop(self, image: Any, length: str) -> str:
        text = ""
        for n, token in enumerate(self.model.caption(image, length=length, stream=True)["caption"], 1):
            text += token
            if n % _LOOP_CHECK_EVERY == 0 and loop_start(text) is not None:
                break
        return cut_loop(text)[0]
