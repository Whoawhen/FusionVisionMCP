"""LiquidAI LFM2.5-VL-3B on ONNX Runtime, the default `caption` since 2026-10-23 and `query_image` since 2026-10-24.

Runs as VisionBench `lfm_onnx.py` measured it (arm `single`): see DEFAULT_CAPTION_ONNX_MODEL in `constants` for the
numbers and why each setting is what it is. The export has no generation loop, so this is one: the image tokens from
the vision encoder (projector included, padding already removed) replace the prompt's <image> placeholders, as
transformers' `masked_scatter` does, and the q8 decoder runs on its own conv and KV state.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL.Image import Image

from .constants import (
    CAPTION_MAX_NEW_TOKENS,
    CAPTION_ONNX_FILES,
    CAPTION_ONNX_THREADS,
    CAPTION_PROMPT,
    CAPTION_SEED,
    DEFAULT_CAPTION_ONNX_MODEL,
    DEFAULT_CAPTION_ONNX_REVISION,
    DEFAULT_DEEP_CAPTION_MODEL,
    DEFAULT_DEEP_CAPTION_REVISION,
    VQA_MAX_NEW_TOKENS,
    VQA_SEED,
)

#: The model card's text decoding, with llama.cpp's names as the gate ran it.
DECODING: dict[str, float] = {"temperature": 0.1, "min_p": 0.15, "repeat_penalty": 1.05}


def pick_token(logits: np.ndarray, seen: np.ndarray, decoding: dict[str, float], rng: np.random.Generator) -> int:
    """One sampling step, in transformers' order: repetition penalty over every id so far, temperature, min_p."""
    logits = logits.astype(np.float64)
    pen = decoding.get("repeat_penalty", 1.0)
    if pen != 1.0:
        s = logits[seen]
        logits[seen] = np.where(s > 0, s / pen, s * pen)
    temp = decoding.get("temperature", 1.0)
    if temp == 0:
        return int(logits.argmax())
    logits = logits / temp
    p = np.exp(logits - logits.max())
    p /= p.sum()
    min_p = decoding.get("min_p", 0.0)
    if min_p:
        p = np.where(p >= min_p * p.max(), p, 0.0)
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


class LfmOnnxCaptioner:
    """Prose captions from LFM2.5-VL-3B (Liquid AI) on ONNX Runtime, CPU, one 256-token image each."""

    processor: Any

    def __init__(
        self,
        model_id: str = DEFAULT_CAPTION_ONNX_MODEL,
        revision: str = DEFAULT_CAPTION_ONNX_REVISION,
        seed: int = CAPTION_SEED,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor

        root = Path(
            snapshot_download(
                model_id, revision=revision, allow_patterns=[f + "*" for f in CAPTION_ONNX_FILES.values()]
            )
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(CAPTION_ONNX_THREADS, os.cpu_count() or CAPTION_ONNX_THREADS)
        self.sessions = {
            k: ort.InferenceSession(str(root / f), opts, providers=["CPUExecutionProvider"])
            for k, f in CAPTION_ONNX_FILES.items()
        }
        # The base repo's processor and chat template (the ONNX repo is a transformers.js export), without tiling.
        self.processor = AutoProcessor.from_pretrained(
            DEFAULT_DEEP_CAPTION_MODEL, revision=DEFAULT_DEEP_CAPTION_REVISION, do_image_splitting=False
        )
        tok = self.processor.tokenizer
        self.image_token_id = tok.convert_tokens_to_ids("<image>")
        self.eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.seed = seed
        self._embed_input = self.sessions["embed"].get_inputs()[0].name
        self._dec_inputs = [i.name for i in self.sessions["decoder"].get_inputs()]
        self._dec_outputs = [o.name for o in self.sessions["decoder"].get_outputs()]

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.sessions["embed"].run(None, {self._embed_input: ids.astype(np.int64)})[0]
        return out

    def _empty_state(self) -> dict[str, np.ndarray]:
        state = {}
        for name in self._dec_inputs:
            if name.startswith("past_conv."):
                state[name] = np.zeros((1, 2048, 3), np.float32)
            elif name.startswith("past_key_values."):
                state[name] = np.zeros((1, 8, 0, 64), np.float32)
        return state

    def generate(self, image: Image, prompt: str, seed: int, max_new_tokens: int = CAPTION_MAX_NEW_TOKENS) -> str:
        """The reply to `prompt` about `image`, sampled with `seed`."""
        conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inp = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
        )
        ids = inp["input_ids"][0].astype(np.int64)
        feats = self.sessions["vision"].run(
            None,
            {
                "pixel_values": inp["pixel_values"].astype(np.float32),
                "pixel_attention_mask": inp["pixel_attention_mask"].astype(np.int64),
                "spatial_shapes": inp["spatial_shapes"].astype(np.int64),
            },
        )[0]
        embeds = self._embed(ids[None])
        slots = ids == self.image_token_id
        if int(slots.sum()) != feats.shape[0]:
            raise RuntimeError(f"{int(slots.sum())} image placeholders but {feats.shape[0]} image tokens")
        embeds[0, slots] = feats
        state = self._empty_state()
        rng = np.random.default_rng(seed)
        seen = list(ids)
        new: list[int] = []
        for _ in range(max_new_tokens):
            mask = np.ones((1, len(seen)), np.int64)
            out = self.sessions["decoder"].run(None, {"inputs_embeds": embeds, "attention_mask": mask, **state})
            res = dict(zip(self._dec_outputs, out, strict=True))
            for name in state:
                state[name] = res[name.replace("past_key_values.", "present.").replace("past_conv.", "present_conv.")]
            nxt = pick_token(res["logits"][0, -1], np.unique(np.array(seen)), DECODING, rng)
            if nxt in self.eos:
                break
            new.append(nxt)
            seen.append(nxt)
            embeds = self._embed(np.array([[nxt]]))
        return str(self.processor.decode(new, skip_special_tokens=True)).strip()

    def caption(self, images: list[Image]) -> list[str]:
        res = []
        for img in images:
            with img.convert("RGB") as rgb_img:
                res.append(self.generate(rgb_img, CAPTION_PROMPT, self.seed))
        return res

    def answer(self, images: list[Image], prompts: list[str], seed: int = VQA_SEED) -> list[str]:
        """`query_image`'s answers: one reply per image to its prompt (`analysis.lfm_vqa_prompt` builds them)."""
        res = []
        for img, prompt in zip(images, prompts, strict=True):
            with img.convert("RGB") as rgb_img:
                res.append(self.generate(rgb_img, prompt, seed, VQA_MAX_NEW_TOKENS))
        return res
