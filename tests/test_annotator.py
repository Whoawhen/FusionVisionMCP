"""Unit tests for the bounding-box annotator.

Pure PIL: no model is downloaded and nothing is inferred, so these run in the fast
suite alongside the geometry and layout tests.
"""

from pathlib import Path

import pytest
from PIL import Image

from fusion_vision_mcp.annotator import save_annotated_image


def _image(mode: str = "RGB", size: tuple[int, int] = (80, 60)) -> Image.Image:
    fill: object = "white"
    if mode == "L":
        fill = 255
    elif mode == "P":
        fill = 0
    return Image.new(mode, size, fill)  # type: ignore[arg-type]


def test_returns_an_absolute_path_to_a_readable_jpeg() -> None:
    path = save_annotated_image(_image(), [[10, 10, 40, 40]])

    assert Path(path).is_absolute()
    assert Path(path).exists()
    with Image.open(path) as written:
        assert written.format == "JPEG"


def test_boxes_are_actually_drawn() -> None:
    """A blank page and an annotated one must not be the same pixels."""
    blank = _image()
    path = save_annotated_image(blank, [[10, 10, 40, 40]])

    with Image.open(path) as written:
        assert written.convert("RGB").tobytes() != blank.convert("RGB").tobytes()


def test_source_image_is_not_mutated() -> None:
    """The annotator draws on a copy; the caller's image must be untouched."""
    original = _image()
    before = original.tobytes()

    save_annotated_image(original, [[10, 10, 40, 40]], ["thing"])

    assert original.tobytes() == before


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_every_input_mode_survives_the_jpeg_encode(mode: str) -> None:
    """JPEG cannot encode an alpha channel.

    `get_images` does not normalise mode, so any transparent PNG reached this function
    as RGBA and raised `OSError: cannot write mode RGBA as JPEG` -- a crash on
    `detect_objects(return_annotated=true)` for a perfectly ordinary input. No test
    fixture in this repo has an alpha channel, which is why it went unnoticed.
    """
    path = save_annotated_image(_image(mode), [[5, 5, 30, 30]], ["label"])

    assert Path(path).exists()


def test_no_boxes_still_produces_an_image() -> None:
    path = save_annotated_image(_image(), [])

    assert Path(path).exists()


def test_more_boxes_than_labels_does_not_raise() -> None:
    """Labels are index-matched and may be shorter; the extra boxes go unlabelled."""
    path = save_annotated_image(_image(), [[5, 5, 20, 20], [30, 30, 50, 50]], ["only one"])

    assert Path(path).exists()


def test_each_call_writes_a_distinct_file() -> None:
    """Filenames carry a uuid suffix, so concurrent calls cannot clobber each other."""
    first = save_annotated_image(_image(), [[5, 5, 20, 20]])
    second = save_annotated_image(_image(), [[5, 5, 20, 20]])

    assert first != second
