"""OCR Fusion: checks the text a caption quotes against Florence-2's OCR regions (`verify_text`).

Measures agreement between each token the caption quotes and the closest OCR span, warns when
they differ, and produces an auditable corrected caption.

`caption(auto_verify_text)` used to add a third reading here: the `ocr` engine re-reading each
matched region on an upscaled crop. It was retired 2026-10-15: since every caption ends with the
`ocr` engine's `Visible text` line, it added 0.0 points of key text on 60 images (VisionBench
`caption_text_recheck.py`) at twice the time. `SpecialistOCR` stays: it types the `ocr` engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from PIL.Image import Image as PILImage

from .textmatch import _MIN_SIMILARITY, Correction, _ratio, extract_candidates, should_correct


class SpecialistOCR(Protocol):
    """Protocol for specialist OCR models."""

    def ocr(self, image: PILImage, max_new_tokens: int = 256) -> str: ...

    def readtext(self, image: PILImage) -> list[dict[str, Any]]: ...

    """Spans with their boxes and confidences: {text, confidence, box}[].

    Part of the contract, not an extra: the `ocr` tool's `detail=true` payload and
    `query_image`'s text-hallucination check both call this, and typing the field as
    a protocol without it made those two call sites fail type-checking.
    """


@dataclass(frozen=True)
class TextConsensus:
    """Agreement between a token the caption quotes and Florence-2's OCR of the matching region.

    Attributes:
        caption: The candidate token as quoted/written in the caption prose.
        florence_ocr: The verbatim text transcribed by Florence-2's OCR head.
        agreeing_sources: 2 when the caption and the OCR agree, else 1.
    """

    caption: str
    florence_ocr: str
    agreeing_sources: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "florence_ocr": self.florence_ocr,
            "agreeing_sources": self.agreeing_sources,
        }


def fuse_caption_ocr(caption: str, text_regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the text a caption quotes against Florence-2's OCR regions.

    Returns a dict with:
        caption: Original caption prose (never overwritten).
        caption_text_warning: True if any quoted/embedded text differs from the OCR.
        text_consensus: Primary TextConsensus dict (or None if no text was evaluated).
        caption_corrected: Corrected copy of the caption with the OCR text substituted.
        text_regions: Raw OCR text regions with bounding boxes.
        corrections: List of Correction dicts for auditing changes.
    """
    if not text_regions:
        return {
            "caption": caption,
            "caption_text_warning": False,
            "text_consensus": None,
            "caption_corrected": caption,
            "text_regions": [],
            "corrections": [],
        }

    ocr_texts = [str(r.get("text", "")) for r in text_regions]
    ocr_boxes = [list(r.get("box", [])) for r in text_regions]

    corrections: list[Correction] = []
    consensuses: list[TextConsensus] = []
    corrected = caption
    used_ocr: set[int] = set()

    for candidate in extract_candidates(caption):
        best_idx: int | None = None
        best_ratio = _MIN_SIMILARITY
        for i, ocr in enumerate(ocr_texts):
            if i in used_ocr:
                continue
            ratio = _ratio(candidate, ocr)
            if ratio >= _MIN_SIMILARITY and ratio > best_ratio:
                best_idx = i
                best_ratio = ratio

        if best_idx is None:
            continue

        target_text = ocr_texts[best_idx]
        box = [int(v) for v in ocr_boxes[best_idx]]
        used_ocr.add(best_idx)
        consensuses.append(
            TextConsensus(
                caption=candidate,
                florence_ocr=target_text,
                agreeing_sources=2 if candidate.lower() == target_text.lower() else 1,
            )
        )

        if candidate != target_text and should_correct(
            candidate, target_text, _ratio(candidate, target_text), ocr_texts
        ):
            corrections.append(Correction(candidate, target_text, box, best_ratio))
            replacement = target_text.replace("\\", "\\\\")
            corrected = re.sub(
                r"\b" + re.escape(candidate) + r"\b",
                replacement,
                corrected,
                count=1,
            )

    has_warning = any(c.caption != c.florence_ocr for c in consensuses)

    return {
        "caption": caption,
        "caption_text_warning": has_warning,
        "text_consensus": consensuses[0].as_dict() if consensuses else None,
        "caption_corrected": corrected,
        "text_regions": text_regions,
        "corrections": [c.as_dict() for c in corrections],
    }
