"""`caption(region=...)`'s crop: the caller's box widened as VisionBench gate R measured it."""

import pytest
from PIL import Image

from fusion_vision_mcp.images import crop_region


def test_the_box_is_widened_by_fifteen_percent_each_side() -> None:
    crop = crop_region(Image.new("RGB", (1000, 1000)), [100, 200, 300, 600])
    # 200 x 400 box: +30 px each side horizontally, +60 vertically.
    assert crop.size == (260, 520)


def test_a_small_box_grows_to_the_minimum_side() -> None:
    assert crop_region(Image.new("RGB", (500, 500)), [100, 100, 110, 104]).size == (32, 32)


def test_the_crop_is_clipped_to_the_image() -> None:
    # Widened to x -71..111 and y -18.5..228.5, then clipped to the 100 x 80 image.
    assert crop_region(Image.new("RGB", (100, 80)), [-50, 10, 90, 200]).size == (100, 80)
    assert crop_region(Image.new("RGB", (100, 80)), [60, 40, 99, 79]).size == (46, 46)


@pytest.mark.parametrize("region", [[10, 10, 5, 20], [1, 2, 3], [200, 200, 300, 300]])
def test_bad_regions_are_refused(region: list[float]) -> None:
    with pytest.raises(ValueError):
        crop_region(Image.new("RGB", (100, 100)), region)
