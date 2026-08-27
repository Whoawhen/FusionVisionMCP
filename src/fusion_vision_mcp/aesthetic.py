#  aesthetic.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""CLIP-backed aesthetic scoring: rates how pleasing an image is to look at, independent of content."""

import hashlib
from pathlib import Path
from typing import Any, Final

import requests
import torch
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

from .device import resolve_device

#: The aesthetic head below was trained on embeddings from this exact CLIP checkpoint;
#: swapping the backbone silently invalidates the head's weights.
DEFAULT_AESTHETIC_MODEL: Final[str] = "openai/clip-vit-large-patch14"

#: LAION's "improved aesthetic predictor" checkpoint, pinned at a specific commit so the
#: URL keeps working even if the upstream repo's default branch moves on.
_HEAD_URL: Final[str] = (
    "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/"
    "6934dd81792f086e613a121dbce43082cb8be85e/sac+logos+ava1-l14-linearMSE.pth"
)
#: Verified against the file actually served at `_HEAD_URL` above; a mismatch means the
#: pin no longer points at the file this code was written against.
_HEAD_SHA256: Final[str] = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"


def _cache_path() -> Path:
    cache_dir = Path.home() / ".cache" / "fusion_vision_mcp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "aesthetic_head.pth"


def _load_head_state_dict() -> dict[str, torch.Tensor]:
    """Downloads (and locally caches) the aesthetic head's weights, verifying their checksum.

    `torch.load(..., weights_only=True)` restricts unpickling to tensor data, since this
    file is fetched from the network rather than shipped with the package.
    """
    path = _cache_path()
    if not path.exists():
        response = requests.get(_HEAD_URL)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != _HEAD_SHA256:
            raise ValueError(f"aesthetic head checksum mismatch: expected {_HEAD_SHA256}, got {digest}")
        path.write_bytes(response.content)

    state_dict: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return state_dict


class _AestheticHead(torch.nn.Module):
    """LAION's "improved aesthetic predictor" MLP: a normalized 768-dim CLIP embedding to a scalar score.

    Architecture and layer names match the upstream checkpoint's state dict exactly
    (`layers.0` through `layers.7`) so its weights load with no key remapping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.layers(x)
        return result


class Aesthetic:
    """Wraps CLIP ViT-L/14 plus a small trained head to score an image's aesthetic quality.

    Florence-2 and Moondream describe what is in an image; neither has any notion of how
    good it looks. The head is a tiny (~3MB) linear network, but producing the embedding it
    scores still requires the full CLIP backbone (~1.7GB) — a real model, on the same order
    as SAM2, kept idle-released the same way since most sessions won't touch it.
    """

    device: str
    torch_dtype: torch.dtype
    model: Any
    processor: Any
    head: _AestheticHead

    def __init__(self, model_id: str = DEFAULT_AESTHETIC_MODEL, device: str | None = None) -> None:
        self.device = resolve_device(device)
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, dtype=self.torch_dtype).to(self.device)
        self.model.eval()

        # CLIP's learned temperature. Standard zero-shot classification scales the cosine
        # similarities by `logit_scale.exp()`; capturing it once here avoids re-reading the
        # parameter on every `classify_style` call.
        self._logit_scale: float = self.model.logit_scale.exp().item()

        self.head = _AestheticHead().to(self.device)
        self.head.load_state_dict(_load_head_state_dict())
        self.head.eval()

    def score(self, images: list[Image]) -> list[dict[str, Any]]:
        """Returns one {"score": float, "rating": str} dict per image, score roughly 1-10."""
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            # The head is shallow and was trained in float32; casting here avoids fp16
            # underflow that a network this shallow has no depth to absorb.
            predictions = self.head(features.float()).squeeze(-1).tolist()

        if isinstance(predictions, float):
            predictions = [predictions]

        return [
            {
                "score": round(value, 2),
                "rating": "excellent" if value >= 7 else "average" if value >= 4.5 else "poor",
            }
            for value in predictions
        ]

    def classify_style(self, images: list[Image]) -> list[dict[str, Any]]:
        """Zero-shot medium/genre classification, reusing the already-loaded CLIP backbone.

        Encodes a fixed palette of style prompts ("a photograph", "an oil painting", ...)
        and ranks them by cosine similarity to each image's embedding. No new weights —
        this is the CLIP zero-shot classification the backbone already supports, kept on
        the same `Aesthetic` instance so a `score_aesthetics`/`critique_composition`
        call that also asks for style context pays for one backbone load, not two.

        Returns one dict per image: `{style, distribution}` where `distribution` is a
        sorted list of `{style, score}` (softmax-normalized probabilities). `style` is the
        top-ranked prompt with its leading article stripped, e.g. "oil painting".
        """
        # "a photograph" is the medium the aesthetic head was trained on; the rest are the
        # common alternatives a caller might want to read the score in the context of.
        prompts = [
            "a photograph",
            "a digital photograph",
            "an oil painting",
            "a watercolor painting",
            "an acrylic painting",
            "a digital illustration",
            "an anime drawing",
            "a manga panel",
            "a pencil sketch",
            "an ink drawing",
            "a charcoal drawing",
            "a 3D render",
            "a vector graphic",
            "a pixel art image",
            "a collage",
            "a screenshot",
        ]
        with torch.no_grad():
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.torch_dtype)
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity (features are already unit-normalized) → softmax over styles.
            # Cast to float32 so the softmax is stable under fp16.
            logits = (image_features.float() @ text_features.float().T) * self._logit_scale
            probs = torch.softmax(logits, dim=-1)

        results = []
        for row in probs:
            ranked = sorted(zip(prompts, row.tolist()), key=lambda kv: kv[1], reverse=True)
            top_style, _ = ranked[0]
            # Strip the leading article ("a "/"an ") for a cleaner reported style.
            style = top_style[2:] if top_style.startswith(("a ", "an ")) else top_style
            results.append(
                {
                    "style": style,
                    "distribution": [
                        {
                            "style": p[2:] if p.startswith(("a ", "an ")) else p,
                            "score": round(float(s), 4),
                        }
                        for p, s in ranked
                    ],
                }
            )
        return results
