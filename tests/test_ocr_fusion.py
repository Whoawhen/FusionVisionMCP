from fusion_vision_mcp.ocr_fusion import fuse_caption_ocr


def test_fuse_caption_ocr_banner_disagreement_and_consensus():
    """Regression test: FusionVisionMCP vs FusionVisionMP banner case."""
    caption = 'A futuristic banner reading "FusionVisionMP" in glowing letters.'
    text_regions = [{"text": "FusionVisionMCP", "box": [50, 40, 350, 110]}]

    res = fuse_caption_ocr(caption, text_regions)

    # Must warn about the disagreement
    assert res["caption_text_warning"] is True
    # Must report the consensus
    consensus = res["text_consensus"]
    assert consensus is not None
    assert consensus["caption"] == "FusionVisionMP"
    assert consensus["florence_ocr"] == "FusionVisionMCP"
    # Caption and OCR disagree -> 1 source
    assert consensus["agreeing_sources"] == 1

    # Must correct the caption with the OCR text
    assert 'reading "FusionVisionMCP"' in res["caption_corrected"]
    assert "FusionVisionMP" not in res["caption_corrected"]

    # Corrections list must be populated
    assert len(res["corrections"]) == 1
    assert res["corrections"][0]["quoted_in_caption"] == "FusionVisionMP"
    assert res["corrections"][0]["verbatim_from_ocr"] == "FusionVisionMCP"


def test_fuse_caption_ocr_all_agree():
    """When caption and OCR agree, warning is False and sources == 2."""
    caption = 'A sign reading "CAUTION" on the floor.'
    text_regions = [{"text": "CAUTION", "box": [10, 10, 90, 40]}]

    res = fuse_caption_ocr(caption, text_regions)

    assert res["caption_text_warning"] is False
    assert res["text_consensus"]["agreeing_sources"] == 2
    assert res["caption_corrected"] == caption
    assert len(res["corrections"]) == 0


def test_fuse_caption_ocr_no_text_negative_control():
    """When no text regions exist, gracefully return original caption."""
    caption = "A lush green lawn with a garden hose."

    res = fuse_caption_ocr(caption, [])
    assert res["caption_text_warning"] is False
    assert res["text_consensus"] is None
    assert res["caption_corrected"] == caption
    assert res["corrections"] == []
