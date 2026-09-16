from typing import Any

import numpy as np
import torch
from PIL.Image import Image

from fusion_vision_mcp.device import resolve_device


class EasyOCREngine:
    """
    Specialist OCR model using EasyOCR (CRAFT + CRNN).

    This model is optimized for extracting text in the wild (scene text,
    watermarks, stylized fonts, logos) and returns explicit confidence scores.
    It is loaded lazily and respects the central idle-release mechanism.
    """

    def __init__(self, device: str | None = None, languages: list[str] | None = None) -> None:
        # A device *string*, like every other wrapper in this package. Taking a
        # torch.device here used to force the package's entry point to import torch
        # just to build one, which broke the torch-free-import invariant.
        self.device = resolve_device(device)
        # One language, one CRNN recognition model. Each extra language is another
        # model downloaded and held in memory, so widen this deliberately via
        # --ocr-languages rather than paying for four by default.
        self.languages = languages or ["en"]
        self._reader = None

    def release(self) -> None:
        if self._reader is not None:
            del self._reader
            self._reader = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _load(self) -> "Any":
        if self._reader is None:
            import easyocr

            use_gpu = self.device.startswith("cuda")
            self._reader = easyocr.Reader(self.languages, gpu=use_gpu)
        return self._reader

    def ocr(self, image: Image, max_new_tokens: int = 256) -> str:
        """
        Extracts verbatim text from an image.
        Returns a flat string to satisfy the SpecialistOCR protocol for fusion.
        """
        results = self.readtext(image)
        return "\n".join(r["text"] for r in results)

    def readtext(self, image: Image) -> list[dict[str, Any]]:
        """
        Extracts text, bounding boxes, and confidence scores.
        """
        reader = self._load()
        img_np = np.array(image.convert("RGB"))
        # We pass batch_size=2 to enforce a strict memory cap on CPU execution
        results = reader.readtext(img_np, batch_size=2)

        output = []
        for bbox, text, prob in results:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            # Convert numpy types to native Python types for JSON serialization
            x1 = int(min(pt[0] for pt in bbox))
            y1 = int(min(pt[1] for pt in bbox))
            x2 = int(max(pt[0] for pt in bbox))
            y2 = int(max(pt[1] for pt in bbox))

            output.append({"text": text, "confidence": float(prob), "box": [x1, y1, x2, y2]})
        return output
