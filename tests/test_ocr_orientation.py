"""`ocr_pages` turns a sideways page upright before reading it (no models).

VisionBench phase 13 measured a 90-degree turn costing 26 points of key-phrase recall.
These pin the decision (read the regions already in hand, re-read only when they run
sideways) and the box mapping back onto the page as sent.
"""

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Transpose

from fusion_vision_mcp.analysis import _looks_sideways, _unrotate_box, ocr_pages


def _region(text: str, box: list[int]) -> dict:
    return {"text": text, "confidence": 0.99, "box": box}


@pytest.mark.parametrize("turn", [Transpose.ROTATE_90, Transpose.ROTATE_270])
def test_unrotate_box_maps_a_turned_box_back_onto_the_page(turn: Transpose) -> None:
    page = Image.new("L", (40, 30))
    page.paste(255, (5, 7, 18, 12))  # inclusive box [5, 7, 17, 11]
    ys, xs = np.nonzero(np.asarray(page.transpose(turn)))
    turned_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    assert _unrotate_box(turned_box, turn, page.width, page.height) == [5, 7, 17, 11]


def test_short_tall_regions_do_not_make_a_page_sideways() -> None:
    digits = [_region("1", [10, y, 14, y + 20]) for y in range(0, 200, 25)]
    assert not _looks_sideways(digits)
    lines = [_region("a sideways line", [x, 10, x + 14, 200]) for x in range(0, 100, 20)]
    assert _looks_sideways(lines)


class _OrientedOcr:
    """Reads well only when a marker pixel, top-left on the upright page, is top-left again."""

    def __init__(self) -> None:
        self.calls = 0

    def readtext(self, image: Image.Image) -> list[dict]:
        self.calls += 1
        if image.getpixel((0, 0)) == (255, 255, 255):
            return [_region("an upright line of text", [10, 10, 200, 24])]
        # Read sideways: tall boxes, fewer characters survive.
        return [_region("side", [10, 10, 24, 120]), _region("ways", [30, 10, 44, 120]), _region("txt", [50, 5, 60, 90])]


class _App:
    def __init__(self) -> None:
        self.ocr_engine = _OrientedOcr()


def _page(upright_marker_at: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    page = Image.new("RGB", size)
    page.putpixel(upright_marker_at, (255, 255, 255))
    return page


def test_an_upright_page_is_read_once() -> None:
    app = _App()
    page = _page((0, 0), (300, 100))
    out = ocr_pages(app, [page])[0]
    assert out["text"] == "an upright line of text"
    assert app.ocr_engine.calls == 1


@pytest.mark.parametrize("turn", [Transpose.ROTATE_90, Transpose.ROTATE_270])
def test_a_sideways_page_is_read_turned_upright(turn: Transpose) -> None:
    upright = _page((0, 0), (300, 100))
    # Send the page turned the opposite way; reading it means turning it back.
    sent = upright.transpose(Transpose.ROTATE_270 if turn is Transpose.ROTATE_90 else Transpose.ROTATE_90)
    app = _App()
    out = ocr_pages(app, [sent])[0]
    assert out["text"] == "an upright line of text"
    x1, y1, x2, y2 = out["text_regions"][0]["box"]
    assert 0 <= x1 < x2 < sent.width and 0 <= y1 < y2 < sent.height
    assert y2 - y1 > x2 - x1  # the line runs sideways on the page as sent
