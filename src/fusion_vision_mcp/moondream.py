import sys
from collections.abc import Callable
from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from .constants import DEFAULT_MOONDREAM_MODEL, DEFAULT_MOONDREAM_REVISION, THEME_PROMPT
from .device import resolve_device
from .repetition import cut_loop, loop_start

#: Tokens between loop checks while streaming a caption; a check rescans the text so far.
_LOOP_CHECK_EVERY = 16


class _SlicedCache:
    """Stands in for one block's `KVCache` during a single-token decode step: writes the new key and
    value as the real cache does, then returns only the filled `:n` prefix instead of all 2,048 slots."""

    def __init__(self, cache: Any, n: int) -> None:
        self.cache, self.n = cache, n

    def update(self, pos_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kout, vout = self.cache.update(pos_ids, k, v)
        return kout[:, :, : self.n, :], vout[:, :, : self.n, :]


def sliced_decode_attn(attn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Wrap the pinned Moondream2's `text.attn` so a one-token decode step attends over the filled cache only.

    Upstream hands SDPA the whole 2,048-slot cache and a boolean mask that is true on `0..pos` and false after
    it, on every generated token in all 24 layers. Slicing keys, values and the mask to `:pos+1` attends over
    the same keys without reading the empty slots. Any step whose mask is not false past `pos` (prefill, or a
    caller's own mask) runs unchanged.

    The mask is kept on purpose: dropping it moves CPU SDPA to another kernel whose bf16 rounding changed
    every one of 20 greedy captions (VisionBench `caption_1c_a.py`, 2026-10-21); with it, they match.
    """

    def wrapped(
        x: torch.Tensor,
        w: Any,
        freqs_cis: torch.Tensor,
        kv_cache: Any,
        attn_mask: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        position_ids: torch.Tensor,
        lora: dict | None,
    ) -> torch.Tensor:
        if kv_cache is not None and x.shape[1] == 1 and attn_mask is not None and position_ids.numel() == 1:
            n = int(position_ids[0]) + 1
            if not bool(attn_mask[..., n:].any()):
                kv_cache, attn_mask = _SlicedCache(kv_cache, n), attn_mask[..., :n]
        return attn(x, w, freqs_cis, kv_cache, attn_mask, n_heads, n_kv_heads, position_ids, lora)

    wrapped.__wrapped__ = attn  # type: ignore[attr-defined]
    return wrapped


def _slice_decode_kv(model: Any) -> None:
    """Install `sliced_decode_attn` in the loaded remote code's `text` module, once.

    Only for the pinned revision: the patch depends on that code's `attn` signature and decode mask.
    """
    # The remote code is loaded at runtime, so its module is only known as a ModuleType.
    text: Any = sys.modules.get(type(model).__module__.rsplit(".", 1)[0] + ".text")
    if text is not None and not hasattr(text.attn, "__wrapped__"):
        text.attn = sliced_decode_attn(text.attn)


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
        if revision == DEFAULT_MOONDREAM_REVISION:
            _slice_decode_kv(self.model)

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
