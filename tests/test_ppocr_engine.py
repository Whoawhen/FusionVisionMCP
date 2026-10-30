"""PP-OCRv6 span conversion, engine routing and the sharpened second reading."""

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


class _FakeEngine:
    """Returns the same spans for any reading (the tests call `_sharpened_extra` directly)."""

    sharpened_pass = True

    def __init__(self, second: list[str]) -> None:
        self.second = second

    def readtext(self, image: object) -> list[dict]:
        return [{"text": t, "confidence": 0.9, "box": [0, 10 * i, 50, 10 * i + 8]} for i, t in enumerate(self.second)]


def test_sharpened_pass_adds_only_new_spans_of_three_characters() -> None:
    """The B2b rule: lone letters ("A", "7", "03") were what made text-free photos non-empty."""
    from types import SimpleNamespace

    from PIL import Image

    from fusion_vision_mcp.analysis import _sharpened_extra

    app = SimpleNamespace(ocr_engine=_FakeEngine(["A", "03", "Exit", "OPEN 24h", "abc"]))
    extra = _sharpened_extra(app, Image.new("RGB", (64, 64)), "Open 24H\nsomething")  # type: ignore[arg-type]
    assert [r["text"] for r in extra] == ["Exit", "abc"]


def test_sharpened_pass_is_skipped_for_engines_without_the_flag() -> None:
    from types import SimpleNamespace

    from PIL import Image

    from fusion_vision_mcp.analysis import _sharpened_extra

    engine = _FakeEngine(["Exit"])
    engine.sharpened_pass = False
    assert _sharpened_extra(SimpleNamespace(ocr_engine=engine), Image.new("RGB", (64, 64)), "") == []  # type: ignore[arg-type]


def _blurred_fixture(name: str, sigma: int) -> "object":
    from pathlib import Path

    from PIL import Image, ImageFilter

    with Image.open(Path(__file__).parent / name) as im:
        return im.convert("RGB").filter(ImageFilter.GaussianBlur(sigma))


def test_sharpened_pass_recovers_words_on_a_blurred_paragraph() -> None:
    """At Gaussian 2 the first reading keeps 18 of the paragraph's 40 words; the sharpened read adds a line."""
    from types import SimpleNamespace

    from fusion_vision_mcp.analysis import ocr_pages
    from fusion_vision_mcp.ppocr_engine import PPOCREngine

    text = ocr_pages(SimpleNamespace(ocr_engine=PPOCREngine()), [_blurred_fixture("layout_paragraph.png", 2)])[0][
        "text"
    ]  # type: ignore[arg-type,list-item]
    assert "no second column" in text


@pytest.mark.parametrize("sigma", [2, 3])
def test_sharpened_pass_keeps_a_blurred_text_free_photo_empty(sigma: int) -> None:
    from types import SimpleNamespace

    from fusion_vision_mcp.analysis import ocr_pages
    from fusion_vision_mcp.ppocr_engine import PPOCREngine

    page = ocr_pages(SimpleNamespace(ocr_engine=PPOCREngine()), [_blurred_fixture("sample.jpg", sigma)])[0]  # type: ignore[arg-type,list-item]
    assert page["text"] == "" and page["text_regions"] == []


def test_caption_and_vqa_reading_skips_the_sharpened_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sharpened=False` (caption's text line, query_image's OCR context) returns the first reading alone."""
    from types import SimpleNamespace

    from PIL import Image

    from fusion_vision_mcp import analysis

    monkeypatch.setattr(analysis, "_read_upright", lambda app, image: [])
    app = SimpleNamespace(ocr_engine=_FakeEngine(["Exit"]))
    page = Image.new("RGB", (64, 64))
    assert analysis.ocr_pages(app, [page])[0]["text"] == "Exit"  # type: ignore[arg-type]
    assert analysis.ocr_pages(app, [page], sharpened=False)[0]["text"] == ""  # type: ignore[arg-type]
