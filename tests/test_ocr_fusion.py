from PIL import Image

from fusion_vision_mcp.ocr_fusion import (
    crop_and_upscale_text_region,
    fuse_caption_ocr,
)
from fusion_vision_mcp.textmatch import contains_likely_text


class MockSpecialist:
    def __init__(self, output: str = "FusionVisionMCP") -> None:
        self.output = output

    def ocr(self, image: Image, max_new_tokens: int = 256) -> str:
        return self.output


def test_contains_likely_text_detection():
    # Quotes
    assert contains_likely_text('A poster saying "SALE" in red')
    # Signage words
    assert contains_likely_text("A banner reading FusionVisionMP on a wall")
    assert contains_likely_text("A sign labeled Exit near the door")
    # CamelCase
    assert contains_likely_text("The company logo is FusionVisionMCP")
    # Acronyms
    assert contains_likely_text("The MCP server runs locally")
    # Capitalized long words
    assert contains_likely_text("Welcome to California today")
    # Negative control: plain scene description with no text/names
    assert not contains_likely_text("a brown dog running across the grass under trees")


def test_crop_and_upscale_small_text():
    img = Image.new("RGB", (300, 300), color="white")
    # Tiny box (< 35px) should upscale 3x
    small_box = [10, 10, 60, 30]  # height = 20
    crop_small = crop_and_upscale_text_region(img, small_box, padding=2)
    # (60 - 10 + 4) * 3 = 162, (30 - 10 + 4) * 3 = 72
    assert crop_small.height == 72
    assert crop_small.width == 162

    # Medium box (35 <= height < 100) should upscale 2x
    med_box = [10, 10, 100, 60]  # height = 50, width = 90 + 4 = 94
    crop_med = crop_and_upscale_text_region(img, med_box, padding=2)
    assert crop_med.height == 108
    assert crop_med.width == 188


def test_fuse_caption_ocr_banner_disagreement_and_consensus():
    """Regression test: FusionVisionMCP vs FusionVisionMP banner case."""
    img = Image.new("RGB", (500, 150), color="black")
    caption = 'A futuristic banner reading "FusionVisionMP" in glowing letters.'
    text_regions = [{"text": "FusionVisionMCP", "box": [50, 40, 350, 110]}]

    mock_specialist = MockSpecialist(output="FusionVisionMCP")
    res = fuse_caption_ocr(caption, text_regions, img, mock_specialist)

    # Must warn about the disagreement
    assert res["caption_text_warning"] is True
    # Must report the consensus
    consensus = res["text_consensus"]
    assert consensus is not None
    assert consensus["caption"] == "FusionVisionMP"
    assert consensus["florence_ocr"] == "FusionVisionMCP"
    assert consensus["specialist_ocr"] == "FusionVisionMCP"
    # Florence and Specialist agree against caption -> 2 sources
    assert consensus["agreeing_sources"] == 2

    # Must correct the caption with the fused consensus text
    assert 'reading "FusionVisionMCP"' in res["caption_corrected"]
    assert "FusionVisionMP" not in res["caption_corrected"]

    # Corrections list must be populated
    assert len(res["corrections"]) == 1
    assert res["corrections"][0]["quoted_in_caption"] == "FusionVisionMP"
    assert res["corrections"][0]["verbatim_from_ocr"] == "FusionVisionMCP"


def test_fuse_caption_ocr_all_agree():
    """When all sources agree, warning is False and sources == 3."""
    img = Image.new("RGB", (200, 100), color="white")
    caption = 'A sign reading "CAUTION" on the floor.'
    text_regions = [{"text": "CAUTION", "box": [10, 10, 90, 40]}]

    mock_specialist = MockSpecialist(output="CAUTION")
    res = fuse_caption_ocr(caption, text_regions, img, mock_specialist)

    assert res["caption_text_warning"] is False
    assert res["text_consensus"]["agreeing_sources"] == 3
    assert res["caption_corrected"] == caption
    assert len(res["corrections"]) == 0


def test_fuse_caption_ocr_no_text_negative_control():
    """When no text regions exist, gracefully return original caption."""
    img = Image.new("RGB", (200, 200), color="green")
    caption = "A lush green lawn with a garden hose."

    res = fuse_caption_ocr(caption, [], img, None)
    assert res["caption_text_warning"] is False
    assert res["text_consensus"] is None
    assert res["caption_corrected"] == caption
    assert res["corrections"] == []
