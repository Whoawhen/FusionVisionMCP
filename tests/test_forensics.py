"""Unit tests for forensics.py: the metadata check behind structured_analysis's AI verdict.

Contract under test: a hit is strong evidence, an empty result is "no evidence either way".
Images are round-tripped through a real PNG/JPEG encode so the tests see what a file load
produces, not a hand-built `info` dict.
"""

import io

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from fusion_vision_mcp.forensics import (
    SUSPICIOUS_SOFTWARE_KEYWORDS,
    analyze_metadata_anomalies,
    extract_metadata,
)

_SOFTWARE_TAG = 0x0131


def _png(software: str | None = None) -> Image.Image:
    """PNG round-trip; `Software` arrives as a text chunk in `image.info`."""
    info = PngInfo()
    if software is not None:
        info.add_text("Software", software)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG", pnginfo=info)
    buf.seek(0)
    return Image.open(buf)


def _jpeg(exif_items: dict[int, object]) -> Image.Image:
    """JPEG round-trip with the given EXIF tags."""
    exif = Image.Exif()
    for tag, value in exif_items.items():
        exif[tag] = value
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


@pytest.mark.parametrize("keyword", SUSPICIOUS_SOFTWARE_KEYWORDS)
def test_exif_software_keyword_is_flagged(keyword):
    """Every listed generator keyword must trip the check when it is in EXIF Software."""
    anomalies = analyze_metadata_anomalies(_jpeg({_SOFTWARE_TAG: f"tool {keyword} 1.0"}))
    assert len(anomalies) == 1
    assert anomalies[0]["source"] == "EXIF Forensics"


@pytest.mark.parametrize("software", ["Stable Diffusion", "MIDJOURNEY v6", "DALL-E 3", "ComfyUI"])
def test_match_is_case_insensitive(software):
    """Generators capitalise their own name however they like."""
    assert len(analyze_metadata_anomalies(_jpeg({_SOFTWARE_TAG: software}))) == 1


@pytest.mark.parametrize("keyword", SUSPICIOUS_SOFTWARE_KEYWORDS)
def test_png_text_chunk_software_is_flagged(keyword):
    """PNG generators write Software as a text chunk, which lands in image.info."""
    assert len(analyze_metadata_anomalies(_png(keyword.upper()))) == 1


def test_only_one_anomaly_when_two_keywords_match():
    """The loop breaks on the first hit: one piece of evidence, not one per keyword."""
    anomalies = analyze_metadata_anomalies(_png("midjourney via stable diffusion"))
    assert len(anomalies) == 1
    assert "midjourney via stable diffusion" in anomalies[0]["evidence"]


def test_plain_image_has_no_evidence():
    """No metadata at all means no evidence, not a verdict."""
    assert analyze_metadata_anomalies(_png()) == []


def test_camera_software_tag_is_not_flagged():
    """Ordinary editing software must not be mistaken for a generator."""
    assert analyze_metadata_anomalies(_jpeg({_SOFTWARE_TAG: "Adobe Lightroom"})) == []
    assert analyze_metadata_anomalies(_png("Adobe Lightroom")) == []


def test_extract_metadata_skips_bytes_values():
    """Raw byte blobs would crash JSON serialisation downstream, so they are dropped."""
    meta = extract_metadata(_jpeg({_SOFTWARE_TAG: "Adobe Lightroom", 0x927C: b"\x00\x01\x02"}))
    assert meta["Software"] == "Adobe Lightroom"
    assert "MakerNote" not in meta
    assert not any(isinstance(v, bytes) for v in meta.values())


def test_non_string_software_value_does_not_raise():
    """A malformed Software value is ignored rather than failing the tool."""
    image = _png()
    image.info["Software"] = 12345
    assert analyze_metadata_anomalies(image) == []
    image.info["Software"] = b"midjourney"
    assert analyze_metadata_anomalies(image) == []
