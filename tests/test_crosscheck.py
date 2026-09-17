"""Unit tests for the query_image cross-check execution (no models).

The routing decision (classify + names_for) is covered by test_question.py; these
pin the execution -- `_vqa_cross_check` / `_spatial_measurement` -- against stub
app objects and synthetic masks, so no model has to download to run them.
"""

import numpy as np
from PIL import Image

from fusion_vision_mcp.analysis import _dispatch, _vqa_cross_check, ocr_pages


class _StubCounter:
    def __init__(self, result: dict) -> None:
        self.result = result

    def detect_objects(self, images: list, object_name: str) -> list[dict]:
        return [self.result]


class _StubSegmenter:
    def __init__(self, masks: list) -> None:
        self.masks = masks

    def segment(self, image, boxes: list) -> list:
        return self.masks


class _StubOcrSpecialist:
    """Stands in for EasyOCR, which now backs every OCR path including this one.

    The cross-check used to call `app.florence2.ocr`; it goes through
    `analysis.ocr_pages` now so that the `ocr` tool, `batch_analyze_images` and this
    route cannot return different text for the same image.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def readtext(self, image) -> list[dict]:
        if not self._text:
            return []
        return [{"text": self._text, "confidence": 0.99, "box": [0, 0, 10, 10]}]


class _StubApp:
    def __init__(self, counter, segmenter=None, florence2=None, ocr_text="") -> None:
        self.counter = counter
        self.segmenter = segmenter
        self.florence2 = florence2
        self.ocr_specialist = _StubOcrSpecialist(ocr_text)


def _image() -> Image.Image:
    return Image.new("RGB", (20, 20))


class TestVqaCrossCheckCount:
    def test_count_route_returns_count_and_separable(self) -> None:
        app = _StubApp(counter=_StubCounter({"count": 3, "bboxes": [[1, 1, 5, 5]], "scores": [0.9]}))
        cross = _vqa_cross_check(app, _image(), "How many dogs are in this image?")
        assert cross is not None
        assert cross["tool"] == "count_objects"
        assert cross["object"] == "dogs"
        assert cross["count"] == 3
        assert cross["separable"] == "yes"  # count > 1 -> yes


class TestVqaCrossCheckOcr:
    def test_ocr_route_returns_transcription(self) -> None:
        app = _StubApp(counter=_StubCounter({}), ocr_text="Hello")
        cross = _vqa_cross_check(app, _image(), "What does the watermark say, exactly?")
        assert cross == {"tool": "ocr", "text": "Hello"}


class TestOcrPathsAgree:
    """Every OCR path must go through `ocr_pages`, or they return different text.

    They genuinely did: the `ocr` tool was switched to EasyOCR in the
    Granite-Docling swap, but `batch_analyze_images` and the VQA cross-check kept
    calling Florence-2's `<OCR>` head. Nothing covered either, so the divergence
    was invisible -- including the part that matters most, which is that the
    Florence-2 head invents text for a text-free image where EasyOCR returns none.
    """

    def test_batch_ocr_operation_matches_the_ocr_tool(self) -> None:
        app = _StubApp(counter=_StubCounter({}), ocr_text="Hello")
        assert _dispatch(app, "ocr", [_image()], question="", object_name="") == [
            page["text"] for page in ocr_pages(app, [_image()])
        ]

    def test_a_text_free_image_stays_empty_on_every_path(self) -> None:
        app = _StubApp(counter=_StubCounter({}), ocr_text="")
        assert _dispatch(app, "ocr", [_image()], question="", object_name="") == [""]
        assert ocr_pages(app, [_image()])[0]["text_regions"] == []


class TestVqaCrossCheckSpatial:
    def test_spatial_route_returns_relation(self) -> None:
        # Two overlapping 20x20 masks: a square on the left and one shifted right.
        a = np.zeros((20, 20), dtype=bool)
        a[2:10, 2:10] = True
        b = np.zeros((20, 20), dtype=bool)
        b[6:14, 6:14] = True  # overlaps a
        app = _StubApp(
            counter=_StubCounter({"bboxes": [[1, 1, 12, 12]], "scores": [0.9]}),
            segmenter=_StubSegmenter([a, b]),
        )
        cross = _vqa_cross_check(app, _image(), "Does the hand touch the shield?")
        assert cross is not None
        assert cross["tool"] == "spatial_relations"
        assert cross["objects"] == ["hand", "shield"]
        assert "relation" in cross

    def test_spatial_one_object_missing_returns_none(self) -> None:
        """Negative control: only one object located -> omit (need two)."""
        app = _StubApp(
            counter=_StubCounter({"bboxes": [], "scores": []}),  # nothing detected
            segmenter=_StubSegmenter([]),
        )
        cross = _vqa_cross_check(app, _image(), "Does the hand touch the shield?")
        assert cross is None


class TestVqaCrossCheckNoCategory:
    def test_open_ended_question_returns_none(self) -> None:
        """Negative control: a judgment question with no measurable fallback."""
        app = _StubApp(counter=_StubCounter({}))
        assert _vqa_cross_check(app, _image(), "Describe the mood of this image.") is None

    def test_count_question_without_a_noun_returns_none(self) -> None:
        app = _StubApp(counter=_StubCounter({}))
        assert _vqa_cross_check(app, _image(), "How many are there?") is None
