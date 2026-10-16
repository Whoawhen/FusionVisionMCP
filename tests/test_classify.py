"""`classify_image`'s label handling and ranking, without loading CLIP."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from fusion_vision_mcp.analysis import classify_pages, clean_labels
from fusion_vision_mcp.constants import CLASSIFY_MAX_LABELS


def test_clean_labels_strips_and_dedupes_in_order() -> None:
    assert clean_labels([" coyote", "Red Fox", "coyote ", "red fox", "", "wolf"]) == ["coyote", "Red Fox", "wolf"]


@pytest.mark.parametrize("labels", [[], ["one"], ["one", " ", "ONE"]])
def test_clean_labels_needs_two(labels: list[str]) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        clean_labels(labels)


def test_clean_labels_caps_the_count() -> None:
    with pytest.raises(ValueError, match="at most"):
        clean_labels([f"label {i}" for i in range(CLASSIFY_MAX_LABELS + 1)])


def test_classify_pages_ranks_and_templates() -> None:
    seen: dict[str, Any] = {}

    def zero_shot(images: list[Any], prompts: list[str]) -> list[list[float]]:
        seen["prompts"] = prompts
        return [[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]][: len(images)]

    app = SimpleNamespace(aesthetic=SimpleNamespace(zero_shot=zero_shot))
    pages = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    out = classify_pages(cast(Any, app), pages, ["coyote", "red fox", "wolf"])

    assert seen["prompts"] == ["a photo of coyote", "a photo of red fox", "a photo of wolf"]
    assert out[0]["label"] == "red fox" and out[0]["probability"] == 0.7
    assert [r["label"] for r in out[0]["ranking"]] == ["red fox", "wolf", "coyote"]
    assert out[1]["label"] == "coyote"
