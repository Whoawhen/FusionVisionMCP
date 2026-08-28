"""Cross-check and correct text a Florence-2 caption misread against OCR spans.

Florence-2's caption head paraphrases embedded text and misspells names/brands
(it rendered this project's own ``FusionVisionMCP`` logo as ``FusionVisionMP``).
The OCR-with-region head transcribes the same text verbatim. This module combines
the two -- already produced by the ``caption`` tool when ``verify_text=true`` --
into an actionable correction: for every token the caption quoted that is close
to, but not identical to, a verbatim OCR span, it records the discrepancy and
substitutes the verbatim text into a corrected copy of the caption.

This is a pure-Python, model-free layer over outputs the two Florence-2 heads
already produce, so it adds no model load and is unit-testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import cast

#: A candidate and an OCR span must be at least this similar to be considered the
#: same piece of text reproduced imperfectly. Below it, they are unrelated.
_MIN_SIMILARITY: float = 0.6

#: An exact match (ratio == 1.0) is *not* a correction -- the caption read it
#: right -- so only the half-open band ``[_MIN_SIMILARITY, 1.0)`` corrects.
_EXACT: float = 1.0

# --- Candidate-token extraction ------------------------------------------------
# A "candidate" is a word in the caption plausibly reproducing embedded text, and
# so at risk of the caption head's paraphrase/misspell failure. Signals, strongest
# first: quoted strings; CamelCase/internal caps (a name); all-caps >= 3 (an
# acronym); capitalized words >= 5 (a proper noun/label). Sentence-initial words
# are not excluded: a near-match against an OCR span is what gates a correction.
_QUOTED: re.Pattern[str] = re.compile(r'"([^"]+)"|' + r"'([^']+)'")
_CAMEL: re.Pattern[str] = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b")
_ALLCAPS: re.Pattern[str] = re.compile(r"\b[A-Z]{3,}\b")
_CAP_LONG: re.Pattern[str] = re.compile(r"\b[A-Z][a-z]{4,}\b")


@dataclass(frozen=True)
class Correction:
    """One discrepancy between a caption token and its verbatim OCR source.

    Attributes:
        quoted_in_caption: The token as the caption head wrote it (imperfect).
        verbatim_from_ocr: The same text as the OCR head transcribed it (verbatim).
        box: The OCR span's bounding box, ``[x1, y1, x2, y2]`` in image pixels.
        similarity: difflib ratio in ``[_MIN_SIMILARITY, 1.0)``; higher = more confident.
    """

    quoted_in_caption: str
    verbatim_from_ocr: str
    box: list[int]
    similarity: float

    def as_dict(self) -> dict[str, object]:
        return {
            "quoted_in_caption": self.quoted_in_caption,
            "verbatim_from_ocr": self.verbatim_from_ocr,
            "box": self.box,
            "similarity": round(self.similarity, 4),
        }


def extract_candidates(caption: str) -> list[str]:
    """Return caption tokens plausibly reproducing embedded text, de-duplicated.

    Order is preserved by first appearance; a token is returned once even if it
    matches several extractor patterns.
    """
    seen: dict[str, None] = {}
    for pattern in (_QUOTED, _CAMEL, _ALLCAPS, _CAP_LONG):
        for match in pattern.finditer(caption):
            # Only the quoted pattern has capture groups; the others match whole.
            if pattern is _QUOTED:
                token = match.group(1) or match.group(2) or ""
            else:
                token = match.group(0)
            token = token.strip().strip("\"'")
            if token and token not in seen:
                seen[token] = None
    return list(seen)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def correct_text(caption: str, text_regions: list[dict[str, object]]) -> tuple[list[Correction], str]:
    """Cross-check a caption against verbatim OCR spans and correct the close misses.

    Args:
        caption: The caption head's prose.
        text_regions: ``[{text, box}, ...]`` from the OCR-with-region head, where
            ``box`` is ``[x1, y1, x2, y2]`` in image pixels.

    Returns:
        A ``(corrections, caption_corrected)`` pair. ``corrections`` is one
        :class:`Correction` per caption token that was close to (but not identical
        to) an OCR span; ``caption_corrected`` is ``caption`` with each such token
        replaced by its verbatim OCR text. With no OCR spans, or when every quoted
        token already matches exactly, ``corrections`` is empty and
        ``caption_corrected`` equals ``caption`` (the negative controls).
    """
    if not text_regions:
        return [], caption

    ocr_texts = [str(region.get("text", "")) for region in text_regions]
    ocr_boxes = [list(cast(list[int], region.get("box", []))) for region in text_regions]

    corrections: list[Correction] = []
    corrected = caption
    # Each OCR span can back at most one correction (closest match wins); a span
    # already consumed is skipped so two caption tokens don't both snap to it.
    used_ocr: set[int] = set()
    for candidate in extract_candidates(caption):
        best_idx: int | None = None
        best_ratio = _MIN_SIMILARITY
        for i, ocr in enumerate(ocr_texts):
            if i in used_ocr:
                continue
            ratio = _ratio(candidate, ocr)
            if _MIN_SIMILARITY <= ratio < _EXACT and ratio > best_ratio:
                best_idx = i
                best_ratio = ratio
        if best_idx is None:
            continue
        verbatim = ocr_texts[best_idx]
        box = [int(v) for v in ocr_boxes[best_idx]]
        corrections.append(Correction(candidate, verbatim, box, best_ratio))
        used_ocr.add(best_idx)
        # Whole-word, escaped replace so a short token can't edit a substring of a
        # longer word; only the first occurrence is replaced to stay conservative.
        # Backslashes in the verbatim OCR text are doubled so re.sub treats them
        # literally rather than as backreference escapes.
        replacement = verbatim.replace("\\", "\\\\")
        corrected = re.sub(
            r"\b" + re.escape(candidate) + r"\b",
            replacement,
            corrected,
            count=1,
        )

    return corrections, corrected


__all__ = ["Correction", "correct_text", "extract_candidates"]
