#  test_textmatch.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for the caption/OCR cross-check and correction layer.

These exercise ``correct_text`` directly with synthetic captions and OCR spans,
so no model has to be downloaded to run them. Each test includes the negative
control (no correction) alongside the positive one.
"""

from fusion_vision_mcp.textmatch import correct_text, extract_candidates


def _region(text: str, box: list[int]) -> dict[str, object]:
    return {"text": text, "box": box}


class TestExtractCandidates:
    def test_camel_case_is_extracted(self) -> None:
        assert "FusionVisionMP" in extract_candidates("A banner reading FusionVisionMP is shown.")

    def test_quoted_strings_are_extracted(self) -> None:
        names = extract_candidates('The sign says "Hello World" near a logo.')
        assert "Hello World" in names

    def test_all_caps_acronyms_are_extracted(self) -> None:
        names = extract_candidates("An image labeled NASA and IBM.")
        assert "NASA" in names
        assert "IBM" in names  # length-3 all-caps meets the >= 3 minimum

    def test_no_candidates_in_plain_sentence(self) -> None:
        # No quoted, no internal caps, no all-caps>=3, no capitalized>=5.
        assert extract_candidates("a small red mug on a wooden table") == []


class TestCorrectText:
    def test_no_ocr_spans_returns_no_corrections(self) -> None:
        """Negative control: an image with no text -> nothing to correct."""
        caption = "A banner reading FusionVisionMP is displayed."
        corrections, corrected = correct_text(caption, [])
        assert corrections == []
        assert corrected == caption

    def test_exact_match_is_not_a_correction(self) -> None:
        """Negative control: caption read it right -> no correction, caption unchanged."""
        caption = "A banner reading FusionVisionMCP is displayed."
        regions = [_region("FusionVisionMCP", [0, 0, 100, 20])]
        corrections, corrected = correct_text(caption, regions)
        assert corrections == []
        assert corrected == caption

    def test_misspelled_camel_case_is_corrected(self) -> None:
        """The documented case: caption 'FusionVisionMP', OCR 'FusionVisionMCP'."""
        caption = "A banner reading FusionVisionMP is displayed."
        regions = [_region("FusionVisionMCP", [0, 0, 100, 20])]
        corrections, corrected = correct_text(caption, regions)
        assert len(corrections) == 1
        assert corrections[0].quoted_in_caption == "FusionVisionMP"
        assert corrections[0].verbatim_from_ocr == "FusionVisionMCP"
        assert corrections[0].box == [0, 0, 100, 20]
        assert 0.6 <= corrections[0].similarity < 1.0
        assert "FusionVisionMCP" in corrected
        assert "FusionVisionMP" not in corrected

    def test_unrelated_ocr_does_not_correct(self) -> None:
        """A caption token unrelated to any OCR span is left alone."""
        caption = "A banner reading FusionVisionMP is displayed."
        regions = [_region("completely different words", [0, 0, 10, 10])]
        corrections, corrected = correct_text(caption, regions)
        assert corrections == []
        assert corrected == caption

    def test_best_match_chosen_across_spans(self) -> None:
        caption = "The logo reads FusionVisionMP here."
        regions = [
            _region("FusionVision", [0, 0, 50, 20]),
            _region("FusionVisionMCP", [0, 0, 100, 20]),
        ]
        corrections, corrected = correct_text(caption, regions)
        assert len(corrections) == 1
        assert corrections[0].verbatim_from_ocr == "FusionVisionMCP"
        assert "FusionVisionMCP" in corrected

    def test_one_ocr_span_backs_one_correction(self) -> None:
        """Two caption tokens near the same span: closest wins, other skips it."""
        caption = "FusionVisionMP and FusionVisionMCP both appear."
        regions = [_region("FusionVisionMCP", [0, 0, 100, 20])]
        corrections, corrected = correct_text(caption, regions)
        # Only the misspelled one (FusionVisionMP) is in the correction band;
        # the exact one (FusionVisionMCP) is excluded by the < 1.0 guard.
        assert len(corrections) == 1
        assert corrections[0].quoted_in_caption == "FusionVisionMP"
        # The exact-match token is left untouched; only the misspelled one is fixed.
        assert corrected.count("FusionVisionMCP") == 2
        assert "FusionVisionMP" not in corrected
