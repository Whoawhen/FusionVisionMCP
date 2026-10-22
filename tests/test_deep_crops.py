"""`caption(deep=true)` caps Moondream2's crops by how much text the page has (plan step 1C-b)."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from PIL import Image

from fusion_vision_mcp.analysis import deep_crops
from fusion_vision_mcp.constants import DEEP_CROPS, DEEP_CROPS_TEXT, DEEP_TEXT_WORDS


@dataclass(frozen=True)
class _VisionConfig:
    max_crops: int = 12


class _Model:
    """Records the crop cap in force each time an image is encoded."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(vision=_VisionConfig()))
        self.crops_at_encode: list[int] = []

    def encode_image(self, image: Any) -> object:
        self.crops_at_encode.append(self.model.config.vision.max_crops)
        return object()

    def caption(self, image: Any, length: str, stream: bool = False) -> dict[str, Any]:
        return {"caption": iter(["A page."])}


def _moondream() -> Any:
    from fusion_vision_mcp.moondream import Moondream

    md = object.__new__(Moondream)
    md.model = _Model()
    return md


def test_text_heavy_pages_keep_twelve_crops() -> None:
    assert (DEEP_CROPS, DEEP_CROPS_TEXT, DEEP_TEXT_WORDS) == (6, 12, 30)
    assert deep_crops("") == 6
    assert deep_crops(" ".join(["word"] * 29)) == 6
    assert deep_crops("\n".join(["two words"] * 15)) == 12


def test_caption_sets_the_cap_per_image_and_restores_it() -> None:
    md = _moondream()
    image = Image.new("RGB", (8, 8))
    md.caption([image, image], max_crops=[6, 12])
    assert md.model.crops_at_encode == [6, 12]
    assert md.model.model.config.vision.max_crops == 12


def test_caption_without_a_cap_leaves_the_model_setting() -> None:
    md = _moondream()
    md.caption([Image.new("RGB", (8, 8))])
    assert md.model.crops_at_encode == [12]
