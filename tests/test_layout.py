#  test_layout.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
from pathlib import Path

from PIL import Image, ImageDraw

from fusion_vision_mcp.layout import find_column_splits, split_columns

TEST_DIR = Path(__file__).resolve().parent


def _blank(width: int = 800, height: int = 400) -> Image.Image:
    return Image.new("L", (width, height), "white")


def _fill(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    """Fills a rectangle with solid ink -- a stand-in for a block of text.

    Column detection only looks at ink density per x-column, so a solid block
    exercises the same code path as real text without depending on a font
    being installed, which keeps these fast unit tests portable across CI.
    """
    ImageDraw.Draw(image).rectangle(box, fill="black")


# --- Fine-grained synthetic edge cases -------------------------------------


def test_blank_image_has_no_splits() -> None:
    assert find_column_splits(_blank()) == []


def test_single_wide_block_has_no_splits() -> None:
    image = _blank()
    _fill(image, (30, 100, 770, 300))
    assert find_column_splits(image) == []


def test_two_blocks_with_wide_gutter_split_once() -> None:
    image = _blank()
    _fill(image, (30, 100, 300, 300))
    _fill(image, (500, 100, 770, 300))
    splits = find_column_splits(image)
    assert len(splits) == 1
    assert 300 < splits[0] < 500


def test_narrow_gap_is_not_a_gutter() -> None:
    """Ordinary word spacing must not be mistaken for a column boundary."""
    image = _blank()
    _fill(image, (30, 100, 395, 300))
    _fill(image, (400, 100, 770, 300))
    assert find_column_splits(image) == []


def test_thin_rule_line_does_not_defeat_detection() -> None:
    """A divider line down the middle of a real gutter is bridged, not treated as content."""
    image = _blank()
    _fill(image, (30, 100, 300, 300))
    _fill(image, (500, 100, 770, 300))
    ImageDraw.Draw(image).line((400, 60, 400, 340), fill="black", width=1)
    splits = find_column_splits(image)
    assert len(splits) == 1
    assert 300 < splits[0] < 500


def test_full_width_heading_does_not_mask_the_gutter_below_it() -> None:
    image = _blank()
    _fill(image, (30, 10, 770, 50))  # heading spans the full width
    _fill(image, (30, 100, 300, 300))
    _fill(image, (500, 100, 770, 300))
    splits = find_column_splits(image)
    assert len(splits) == 1
    assert 300 < splits[0] < 500


def test_gap_near_edge_is_not_a_column_boundary() -> None:
    """A stray sliver of content near the edge should not read as a second column."""
    image = _blank()
    _fill(image, (5, 100, 40, 300))
    _fill(image, (100, 100, 770, 300))
    assert find_column_splits(image) == []


def test_three_blocks_split_twice() -> None:
    image = _blank(width=900)
    _fill(image, (30, 100, 260, 300))
    _fill(image, (330, 100, 560, 300))
    _fill(image, (630, 100, 860, 300))
    splits = find_column_splits(image)
    assert len(splits) == 2
    assert 260 < splits[0] < 330
    assert 560 < splits[1] < 630


def test_split_columns_returns_original_image_when_no_gutter() -> None:
    image = _blank()
    _fill(image, (30, 100, 770, 300))
    assert split_columns(image) == [image]


def test_split_columns_crops_span_the_full_width_in_order() -> None:
    image = _blank(width=800)
    _fill(image, (30, 100, 300, 300))
    _fill(image, (500, 100, 770, 300))
    columns = split_columns(image)
    assert len(columns) == 2
    assert columns[0].width + columns[1].width == image.width
    assert columns[0].width < columns[1].width or columns[0].width == columns[1].width
    split_x = find_column_splits(image)[0]
    assert columns[0].width == split_x


# --- Regression fixtures: real rendered documents ---------------------------
# These reproduce the exact case that motivated this module: a synthetic
# meeting-notes form whose two columns, run through raster-order OCR, came
# back interleaved ("Attendee: Alice, Location: Room 4B, Attende: Ben, ...")
# with two names mangled in the process. Splitting at the detected gutter and
# OCR-ing each column independently fixed both the ordering and a field
# (`Deadline: Sept 10`) that four different `query_image` re-prompts never
# once produced.


def test_table_fixture_is_not_split() -> None:
    """A table has its own column gaps, but every row carries ink in each one."""
    image = Image.open(TEST_DIR / "layout_table.png")
    assert find_column_splits(image) == []


def test_paragraph_fixture_is_not_split() -> None:
    image = Image.open(TEST_DIR / "layout_paragraph.png")
    assert find_column_splits(image) == []


def test_two_column_fixture_splits_once() -> None:
    image = Image.open(TEST_DIR / "layout_two_column.png")
    splits = find_column_splits(image)
    assert len(splits) == 1
    assert 200 < splits[0] < 420


def test_two_column_ruled_fixture_splits_at_the_same_point() -> None:
    """The divider line drawn down this fixture's gutter must not change the result."""
    plain = find_column_splits(Image.open(TEST_DIR / "layout_two_column.png"))
    ruled = find_column_splits(Image.open(TEST_DIR / "layout_two_column_ruled.png"))
    assert plain == ruled


def test_three_column_fixture_splits_twice() -> None:
    image = Image.open(TEST_DIR / "layout_three_column.png")
    splits = find_column_splits(image)
    assert len(splits) == 2
