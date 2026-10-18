"""PP-OCRv6 span conversion and engine routing, without loading any model."""

import pytest

from fusion_vision_mcp import app_lifespan
from fusion_vision_mcp.constants import OCR_MIN_CONFIDENCE
from fusion_vision_mcp.ppocr_engine import spans

QUAD = [[10.0, 20.0], [110.0, 22.0], [109.0, 40.0], [9.0, 38.0]]


def test_quad_becomes_an_axis_aligned_box() -> None:
    assert spans([QUAD], ["Hello"], [0.98], OCR_MIN_CONFIDENCE) == [
        {"text": "Hello", "confidence": 0.98, "box": [9, 20, 110, 40]}
    ]


def test_no_text_is_an_empty_list() -> None:
    assert spans(None, None, None, OCR_MIN_CONFIDENCE) == []


@pytest.mark.parametrize(
    ("text", "score"),
    [("S", 0.521), ("0", 0.565), ("chee   o  ee s    s ns s eg ge.", 0.70)],
)
def test_measured_garbage_is_below_the_floor(text: str, score: float) -> None:
    """The spans PP-OCRv6 invented on text-free fixtures, and its one garbled line."""
    assert spans([QUAD], [text], [score], OCR_MIN_CONFIDENCE) == []


def test_real_text_survives_the_floor() -> None:
    kept = spans([QUAD, QUAD], ["Invoice #4471", "S"], [0.99, 0.52], OCR_MIN_CONFIDENCE)
    assert [s["text"] for s in kept] == ["Invoice #4471"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(("languages", "shared"), [(("en",), False), (("en", "de"), True)])
async def test_non_english_ocr_falls_back_to_easyocr(languages: tuple[str, ...], shared: bool) -> None:
    """The PP-OCRv6 recognizer is English-only; other languages keep EasyOCR for `ocr`."""
    async with app_lifespan(
        None,  # type: ignore[arg-type]
        model_id="unused",
        moondream_model_id="unused",
        moondream_revision="unused",
        ocr_languages=languages,
    ) as app:
        assert (app.ocr_engine is app.ocr_specialist) is shared


def test_detector_uses_the_photo_tuned_box_threshold() -> None:
    # 0.5 (RapidOCR's default) read 55.0% of TextVQA answers; 0.4 read 58.5% with the
    # rendered-page and text-free controls unchanged.
    from fusion_vision_mcp.ppocr_engine import engine_params

    params = engine_params("en", "v6")
    assert params["Det.box_thresh"] == 0.4
    assert params["Rec.lang_type"] == "en" and params["Rec.ocr_version"] == "v6"


def test_sliver_crop_reads_as_no_text() -> None:
    """A 2048x11 zone rounds to a 0-high detector input; RapidOCR raised instead of reading nothing."""
    from PIL import Image

    from fusion_vision_mcp.ppocr_engine import PPOCREngine

    assert PPOCREngine().readtext(Image.new("RGB", (2048, 11), "white")) == []
