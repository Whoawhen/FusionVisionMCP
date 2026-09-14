#  domain_router.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Domain routing using SigLIP2 zero-shot classification.

Routes images to the appropriate vision backend based on their visual domain:
photograph -> Grounding DINO, clip-art/vector -> Florence-2 grounding, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL.Image import Image

from .device import resolve_device
from .idle import IdleReleased

# Domain labels matching the spec doc's DOMAIN_LABELS
DOMAIN_LABELS = [
    "a photograph",
    "a photorealistic image",
    "flat vector clip art",
    "vector artwork",
    "cartoon illustration",
    "digital illustration",
    "oil painting",
    "watercolor painting",
    "technical diagram",
    "document or scanned page",
    "computer screenshot",
]

# Domain groups for routing decisions
PHOTO_DOMAINS = {"a photograph", "a photorealistic image"}
CLIP_ART_DOMAINS = {"flat vector clip art", "vector artwork", "cartoon illustration", "digital illustration"}
PAINTING_DOMAINS = {"oil painting", "watercolor painting"}
DOCUMENT_DOMAINS = {"technical diagram", "document or scanned page", "computer screenshot"}


@dataclass(frozen=True)
class DomainResult:
    """Result of domain classification.

    `domain` is the top-scoring label. `confidence` and `scores` are softmax
    probabilities over `DOMAIN_LABELS`, not SigLIP2's native sigmoid outputs --
    checked empirically, raw sigmoid rounds to ~0.000 for every label on real
    fixtures here (SigLIP2's sigmoid head is calibrated for an independent
    per-pair match/no-match decision with a large negative bias, not a
    normalized distribution across many candidate labels), which makes it
    useless for a margin-based ambiguity check. Softmax over the same logits
    preserves the identical top-1 ranking (sigmoid is a monotonic transform of
    the logits) while giving a margin that actually varies with confidence.
    `ambiguous` is true when the softmax margin between top-1 and top-2 is
    below the threshold.
    """

    domain: str
    confidence: float
    scores: dict[str, float]
    ambiguous: bool


# Model ID for SigLIP2
_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"

# Margin threshold for ambiguous classification - will be tuned empirically
# This is the minimum difference between top-1 and top-2 sigmoid scores
_AMBIGUOUS_MARGIN_THRESHOLD = 0.15

# Global lazy-loaded model cache
_SIGLIP2_CACHE: IdleReleased[Any] | None = None


def _load_siglip2() -> tuple[Any, Any]:
    """Factory for IdleReleased: loads model and processor.

    No `trust_remote_code` -- checked directly, `AutoModel`/`AutoProcessor`
    resolve `google/siglip2-base-patch16-224` to the native `SiglipModel`/
    `SiglipProcessor` classes in the pinned transformers version without it.
    Unlike `moondream.py`'s model, this one needs no remote code at all.
    """
    from transformers import AutoModel, AutoProcessor

    device = resolve_device()
    torch_dtype = torch.float16 if device.startswith(("mps", "cuda")) else torch.float32

    model = AutoModel.from_pretrained(_SIGLIP2_MODEL_ID, dtype=torch_dtype).to(device)
    processor = AutoProcessor.from_pretrained(_SIGLIP2_MODEL_ID, clean_up_tokenization_spaces=True)
    return model, processor


def _get_siglip2_cache() -> IdleReleased[tuple[Any, Any]]:
    """Get or create the global SigLIP2 cache."""
    global _SIGLIP2_CACHE
    if _SIGLIP2_CACHE is None:
        _SIGLIP2_CACHE = IdleReleased(_load_siglip2, timeout=300.0, name="SigLIP2")
    return _SIGLIP2_CACHE


def release_siglip2() -> None:
    """Explicitly release the SigLIP2 model from memory."""
    global _SIGLIP2_CACHE
    if _SIGLIP2_CACHE is not None:
        _SIGLIP2_CACHE.release()
        _SIGLIP2_CACHE = None


def classify_domain(image: Image) -> DomainResult:
    """Classify an image's visual domain using SigLIP2 zero-shot, against `DOMAIN_LABELS`."""
    model, processor = _get_siglip2_cache().get()

    # Prepare inputs
    with image.convert("RGB") as rgb:
        inputs = processor(
            images=rgb,
            text=DOMAIN_LABELS,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Use softmax for zero-shot classification (standard practice).
    # SigLIP's sigmoid loss produces per-label logits, but softmax gives
    # meaningful relative probabilities for classification and margin calculation.
    logits = outputs.logits_per_image[0]  # shape: [num_labels]
    probs = torch.softmax(logits, dim=-1).cpu().tolist()

    # Map labels to scores
    scores = dict(zip(DOMAIN_LABELS, probs))

    # Find top-1 and top-2
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_score = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0

    # Determine if ambiguous based on margin (in softmax probability space)
    margin = top_score - second_score
    ambiguous = margin < _AMBIGUOUS_MARGIN_THRESHOLD

    return DomainResult(
        domain=top_label,
        confidence=top_score,
        scores=scores,
        ambiguous=ambiguous,
    )


def set_ambiguous_margin_threshold(threshold: float) -> None:
    """Set the margin threshold for ambiguous classification.

    Used during empirical tuning against fixtures.
    """
    global _AMBIGUOUS_MARGIN_THRESHOLD
    _AMBIGUOUS_MARGIN_THRESHOLD = threshold


def get_ambiguous_margin_threshold() -> float:
    """Get the current ambiguous margin threshold."""
    return _AMBIGUOUS_MARGIN_THRESHOLD


__all__ = [
    "CLIP_ART_DOMAINS",
    "DOCUMENT_DOMAINS",
    "DOMAIN_LABELS",
    "PAINTING_DOMAINS",
    "PHOTO_DOMAINS",
    "DomainResult",
    "classify_domain",
    "get_ambiguous_margin_threshold",
    "release_siglip2",
    "set_ambiguous_margin_threshold",
]
