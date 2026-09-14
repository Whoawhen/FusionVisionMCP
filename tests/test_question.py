#  test_question.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for the VQA question classifier and name extractor.

Pure-Python, no models: pins ``classify`` and ``names_for`` against the
question wordings the cross-check routes to a measurement, plus the negative
controls (no category, or a category whose names can't be parsed -> None).
"""

from fusion_vision_mcp.question import COUNT, OCR, SPATIAL, classify, names_for


class TestClassify:
    def test_count_question(self) -> None:
        assert classify("How many people are in this image?") == COUNT

    def test_spatial_touch_question(self) -> None:
        assert classify("Does the hand touch the shield?") == SPATIAL

    def test_spatial_contain_question(self) -> None:
        assert classify("Is the cup inside the box?") == SPATIAL

    def test_ocr_watermark_question(self) -> None:
        assert classify("What does the watermark say, exactly?") == OCR

    def test_open_ended_judgment_has_no_category(self) -> None:
        """Negative control: 'describe the mood' is a judgment, not measurable."""
        assert classify("Describe the mood of this image.") is None

    def test_count_takes_precedence_over_spatial_keyword(self) -> None:
        """'how many ... between' must classify as count, not spatial."""
        assert classify("How many fingers are between the hands?") == COUNT


class TestNamesFor:
    def test_spatial_quoted_names(self) -> None:
        names = names_for(SPATIAL, 'Does "hand" touch "shield"?')
        assert names == ["hand", "shield"]

    def test_spatial_plain_pattern_names(self) -> None:
        names = names_for(SPATIAL, "Does the hand touch the shield?")
        assert names == ["hand", "shield"]

    def test_spatial_single_name_returns_none(self) -> None:
        """Negative control: only one name parseable -> omit (need two)."""
        assert names_for(SPATIAL, "Does the hand touch something?") is None

    def test_count_quoted_name(self) -> None:
        assert names_for(COUNT, 'How many "cats"?') == ["cats"]

    def test_count_plain_pattern_name(self) -> None:
        assert names_for(COUNT, "How many dogs are in this image?") == ["dogs"]

    def test_count_no_name_returns_none(self) -> None:
        """Negative control: 'how many are there' with no noun -> omit."""
        assert names_for(COUNT, "How many are there?") is None

    def test_ocr_needs_no_names(self) -> None:
        assert names_for(OCR, "What does the text say?") == []

    def test_no_category_returns_none(self) -> None:
        assert names_for("nonexistent", "anything") is None
