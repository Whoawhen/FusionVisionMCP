"""English OCR with PaddleOCR's PP-OCRv6 models, run through RapidOCR on ONNX Runtime.

Backs the `ocr` tool and caption text verification. On 200 TextVQA scene photos it found
the answer's text on 58.6% of text questions against EasyOCR's 40.3%, and read a synthetic
invoice exactly where EasyOCR produced "812.50" for $12.50. EasyOCR stays loaded for
`query_image(structured_analysis=true)`, whose gibberish threshold is calibrated to
EasyOCR's confidence scores.

The English recognizer is chosen explicitly: RapidOCR's default PP-OCRv4 model is
Chinese/English and drops the spaces between English words ("Thisisasingleparagraph").

No torch: RapidOCR runs on ONNX Runtime and is imported only on first use.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL.Image import Image

from fusion_vision_mcp.constants import OCR_DET_BOX_THRESHOLD, OCR_MIN_CONFIDENCE


class PPOCREngine:
    def __init__(self, min_confidence: float = OCR_MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        self._engine: Any = None

    def release(self) -> None:
        self._engine = None

    def _load(self) -> Any:
        if self._engine is None:
            from rapidocr import LangRec, OCRVersion, RapidOCR

            self._engine = RapidOCR(params=engine_params(LangRec.EN, OCRVersion.PPOCRV6))
        return self._engine

    def ocr(self, image: Image, max_new_tokens: int = 256) -> str:
        """Verbatim text as one string, for the `SpecialistOCR` protocol."""
        return "\n".join(region["text"] for region in self.readtext(image))

    def readtext(self, image: Image) -> list[dict[str, Any]]:
        """Spans as `{text, confidence, box}`, box = [x1, y1, x2, y2] in image pixels.

        RapidOCR rounds the detector's input to multiples of 32 and raises when a side
        rounds to 0, which a sliver zone from `layout.reading_blocks` does (a 2048x11 top
        strip of a round 18 screenshot crashed `caption`). Nothing that thin is legible,
        so it reads as no text.
        """
        from rapidocr.utils.process_img import ResizeImgError

        engine = self._load()
        try:
            result = engine(np.array(image.convert("RGB")))
        except ResizeImgError:
            return []
        return spans(result.boxes, result.txts, result.scores, self.min_confidence)


def engine_params(lang: Any, version: Any) -> dict[str, Any]:
    """RapidOCR settings: the English PP-OCRv6 recognizer, and a detector tuned for photos.

    Split out so a test can pin them without loading the ONNX models.
    """
    return {"Rec.lang_type": lang, "Rec.ocr_version": version, "Det.box_thresh": OCR_DET_BOX_THRESHOLD}


def spans(
    boxes: Sequence[Any] | None,
    texts: Sequence[str] | None,
    scores: Sequence[float] | None,
    min_confidence: float,
) -> list[dict[str, Any]]:
    """Converts RapidOCR's parallel arrays to spans, dropping those below `min_confidence`.

    RapidOCR returns None for all three when it finds no text at all.
    """
    if boxes is None or texts is None or scores is None:
        return []
    out = []
    for box, text, score in zip(boxes, texts, scores, strict=True):
        if float(score) < min_confidence:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        out.append(
            {
                "text": str(text),
                "confidence": float(score),
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            }
        )
    return out
