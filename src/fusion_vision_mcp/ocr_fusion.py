"""OCR Fusion: combines Florence-2 captioning and OCR with EasyOCR specialist OCR.

Implements:
- Automatic caption text verification (Spec 13): detects likely text in captions
  and triggers specialist OCR cross-checking.
- Small-text OCR crops (Spec 14): crops text regions from the full image,
  upscales small text (2x/3x bicubic), and transcribes with EasyOCR.
- Text consensus: measures agreement between caption, Florence OCR, and specialist OCR,
  warning when the caption misreads embedded text and producing an actionable corrected caption.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image
from PIL.Image import Image as PILImage

from .textmatch import _MIN_SIMILARITY, Correction, _ratio, extract_candidates

logger = logging.getLogger(__name__)


class SpecialistOCR(Protocol):
    """Protocol for specialist OCR models."""

    def ocr(self, image: PILImage, max_new_tokens: int = 256) -> str: ...


@dataclass(frozen=True)
class TextConsensus:
    """Consensus measurement across caption, Florence OCR, and specialist OCR.

    Attributes:
        caption: The candidate token as quoted/written in the caption prose.
        florence_ocr: The verbatim text transcribed by Florence-2's OCR head.
        specialist_ocr: The verbatim text transcribed by EasyOCR.
        agreeing_sources: Number of sources agreeing on the correct reading (1 to 3).
    """

    caption: str
    florence_ocr: str
    specialist_ocr: str
    agreeing_sources: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "florence_ocr": self.florence_ocr,
            "specialist_ocr": self.specialist_ocr,
            "agreeing_sources": self.agreeing_sources,
        }


def crop_and_upscale_text_region(
    image: PILImage,
    box: list[int],
    padding: int = 4,
) -> PILImage:
    """Crop a text region from the full image and upscale if small (Spec 14).

    Small text (<35px tall) is upscaled 3x using bicubic interpolation.
    Medium text (<100px tall) is upscaled 2x.
    """
    img_w, img_h = image.size
    x1 = max(0, int(box[0]) - padding)
    y1 = max(0, int(box[1]) - padding)
    x2 = min(img_w, int(box[2]) + padding)
    y2 = min(img_h, int(box[3]) + padding)

    if x2 <= x1 or y2 <= y1:
        return image

    crop = image.crop((x1, y1, x2, y2))
    h = crop.height
    scale = 3 if h < 35 else (2 if h < 100 else 1)
    if scale > 1:
        new_size = (crop.width * scale, crop.height * scale)
        crop = crop.resize(new_size, Image.Resampling.BICUBIC)

    return crop


def fuse_caption_ocr(
    caption: str,
    text_regions: list[dict[str, Any]],
    image: PILImage,
    specialist: SpecialistOCR | None = None,
) -> dict[str, Any]:
    """Fuse caption text with Florence-2 OCR and specialist OCR.

    Returns a dict with:
        caption: Original caption prose (never overwritten).
        caption_text_warning: True if any quoted/embedded text is disputed or corrected.
        text_consensus: Primary TextConsensus dict (or None if no text was evaluated).
        caption_corrected: Corrected copy of the caption with consensus text substituted.
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

    # Run specialist OCR on cropped, upscaled regions where available
    specialist_texts: list[str] = []
    for box in ocr_boxes:
        if specialist is not None and len(box) == 4:
            try:
                crop = crop_and_upscale_text_region(image, box)
                read_text = specialist.ocr(crop).strip()
                specialist_texts.append(read_text if read_text else "")
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.warning("Specialist OCR crop failed: %s", e)
                specialist_texts.append("")
        else:
            specialist_texts.append("")

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

        florence_val = ocr_texts[best_idx]
        specialist_val = specialist_texts[best_idx] if best_idx < len(specialist_texts) else ""
        if not specialist_val:
            specialist_val = florence_val

        box = [int(v) for v in ocr_boxes[best_idx]]
        used_ocr.add(best_idx)

        # Consensus counting
        sources_agreeing = 1
        if florence_val.lower() == specialist_val.lower():
            target_text = specialist_val
            sources_agreeing = 3 if candidate.lower() == target_text.lower() else 2
        elif candidate.lower() == specialist_val.lower():
            target_text = specialist_val
            sources_agreeing = 2
        elif candidate.lower() == florence_val.lower():
            target_text = florence_val
            sources_agreeing = 2
        else:
            target_text = specialist_val if specialist_val else florence_val
            sources_agreeing = 1

        consensus = TextConsensus(
            caption=candidate,
            florence_ocr=florence_val,
            specialist_ocr=specialist_val,
            agreeing_sources=sources_agreeing,
        )
        consensuses.append(consensus)

        if candidate != target_text:
            corrections.append(Correction(candidate, target_text, box, best_ratio))
            replacement = target_text.replace("\\", "\\\\")
            corrected = re.sub(
                r"\b" + re.escape(candidate) + r"\b",
                replacement,
                corrected,
                count=1,
            )

    has_warning = any(c.agreeing_sources < 3 or c.caption != c.florence_ocr for c in consensuses)

    return {
        "caption": caption,
        "caption_text_warning": has_warning,
        "text_consensus": consensuses[0].as_dict() if consensuses else None,
        "caption_corrected": corrected,
        "text_regions": text_regions,
        "corrections": [c.as_dict() for c in corrections],
    }
