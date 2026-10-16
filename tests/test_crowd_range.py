"""`count_objects`' `crowd_range`, without loading SAM 3."""

from types import SimpleNamespace
from typing import Any, cast

from PIL import Image

from fusion_vision_mcp.analysis import crowd_range
from fusion_vision_mcp.constants import CROWD_RANGE_SAM3_FLOOR


class FakeSam3:
    def __init__(self, count: int | None = None, error: Exception | None = None) -> None:
        self.count, self.error, self.calls = count, error, []

    def detect_objects(self, images: list[Any], object_name: str, threshold: float) -> list[dict[str, Any]]:
        self.calls.append((object_name, threshold))
        if self.error:
            raise self.error
        return [{"count": self.count}]


IMG = Image.new("RGB", (8, 8))


def app(locator: Any) -> Any:
    return cast(Any, SimpleNamespace(locator=locator))


def test_range_spans_both_counts_at_the_gated_floor() -> None:
    sam3 = FakeSam3(count=13)
    out = crowd_range(app(sam3), IMG, "banana", 1)
    assert out is not None
    assert (out["low"], out["high"], out["sam3_count"]) == (1, 13, 13)
    assert sam3.calls == [("banana", CROWD_RANGE_SAM3_FLOOR)]


def test_order_does_not_matter() -> None:
    out = crowd_range(app(FakeSam3(count=5)), IMG, "car", 9)
    assert out is not None and (out["low"], out["high"]) == (5, 9)


def test_not_shown_below_four() -> None:
    assert crowd_range(app(FakeSam3(count=3)), IMG, "cup", 2) is None


def test_shown_when_only_one_reaches_four() -> None:
    out = crowd_range(app(FakeSam3(count=4)), IMG, "cup", 1)
    assert out is not None and (out["low"], out["high"]) == (1, 4)


def test_absent_without_sam3() -> None:
    assert crowd_range(app(None), IMG, "cup", 7) is None
    assert crowd_range(app(FakeSam3(error=OSError("gated"))), IMG, "cup", 7) is None
