#  test_crosscheck.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for the query_image cross-check execution (no models).

The routing decision (classify + names_for) is covered by test_question.py; these
pin the execution -- `_vqa_cross_check` / `_spatial_measurement` -- against stub
app objects and synthetic masks, so no model has to download to run them.
"""

import numpy as np
from PIL import Image

from fusion_vision_mcp import _vqa_cross_check


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


class _StubProcessor:
    def __init__(self, ocr_text: str = "transcribed text") -> None:
        self.ocr_text = ocr_text

    def ocr(self, images: list) -> list[str]:
        return [self.ocr_text]


class _StubApp:
    def __init__(self, counter, segmenter=None, processor=None) -> None:
        self.counter = counter
        self.segmenter = segmenter
        self.processor = processor


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
        app = _StubApp(counter=_StubCounter({}), processor=_StubProcessor(ocr_text="Hello"))
        cross = _vqa_cross_check(app, _image(), "What does the watermark say, exactly?")
        assert cross == {"tool": "ocr", "text": "Hello"}


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
