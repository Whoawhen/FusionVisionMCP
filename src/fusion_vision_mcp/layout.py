#  layout.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Column detection for OCR.

A document laid out in side-by-side columns (a two-column form, meeting notes,
a resume) breaks naive raster-order OCR: reading strictly left-to-right within
each horizontal band interleaves unrelated fields from different columns
instead of reading one column fully before the next. Measured on a synthetic
two-column fixture, this scrambled `Attendee: Alice` / `Location: Room 4B` /
`Attendee: Ben` / `Duration: 45 min` into one interleaved stream, and mangled
two names in the process.

The fix is not a better prompt or a different model -- re-prompting Moondream2
(`query_image`) to read the columns separately recovered the right order but
silently dropped a field across every phrasing tried, which matches its known
tendency to paraphrase rather than exhaustively transcribe. Detecting the
column boundary and OCR-ing each column independently recovered every field.

Pure numpy/PIL -- no model loading -- so this is cheap to run before every OCR
call and cheap to test against synthetic fixtures in isolation from Florence-2.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image

#: A pixel darker than this (0-255 grayscale) counts as ink.
_INK_THRESHOLD: Final[int] = 200

#: A column of pixels counts as part of a gutter (blank vertical strip) when its
#: ink density is at or below this fraction of the body height. Not exactly
#: zero: anti-aliased or compressed edges can leave a stray faint pixel in an
#: otherwise blank column.
_GUTTER_DENSITY_FRAC: Final[float] = 0.002

#: A gutter narrower than this fraction of the image width is ordinary
#: inter-word spacing, not a column boundary.
_MIN_GUTTER_WIDTH_FRAC: Final[float] = 0.01

#: An ink run at most this many pixels wide, flanked by gutter on both sides,
#: is treated as part of the gutter rather than a column boundary -- a ruled
#: divider line down the middle of a real gutter should not defeat detection.
#: Ordinary text strokes at any normal reading size are wider than this.
_RULE_LINE_MAX_WIDTH_PX: Final[int] = 3

#: Fraction of the image height, at the top, excluded from gutter detection.
#: A page title or heading commonly spans the full width above the columns
#: and would otherwise mask a real gutter that only starts below it.
_TOP_BAND_FRAC: Final[float] = 0.15

#: A candidate gutter within this fraction of either edge is discarded --
#: ordinary page margins are not column boundaries.
_EDGE_MARGIN_FRAC: Final[float] = 0.1


def _close_thin_ink(is_gap: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Bridges thin ink runs (e.g. a ruled divider line) flanked by gutter on both sides."""
    closed = is_gap.copy()
    width = len(is_gap)
    x = 0
    while x < width:
        if closed[x]:
            x += 1
            continue
        start = x
        while x < width and not closed[x]:
            x += 1
        if start > 0 and x < width and x - start <= _RULE_LINE_MAX_WIDTH_PX:
            closed[start:x] = True
    return closed


def find_column_splits(image: Image) -> list[int]:
    """Finds x-coordinates that split `image` into left-to-right reading columns.

    Returns split points in left-to-right order, e.g. `[310]` for a two-column
    page, `[]` for a page that reads as a single column. Each split point is
    the horizontal midpoint of a detected gutter: a contiguous vertical strip
    with (almost) no ink, wide enough and far enough from the page edges to be
    a real column boundary rather than ordinary word spacing or a margin.

    A table's own column gaps do not trigger a split here: unlike a form's
    columns, a table's columns each carry ink in most rows, so there is
    usually no vertical strip that is blank across the whole body height.
    """
    arr = np.asarray(image.convert("L"))
    height, width = arr.shape
    body = arr[int(height * _TOP_BAND_FRAC) :, :]

    ink = body < _INK_THRESHOLD
    density = ink.sum(axis=0)
    is_gap = _close_thin_ink(density <= max(1, int(_GUTTER_DENSITY_FRAC * body.shape[0])))

    min_gutter_width = max(4, int(_MIN_GUTTER_WIDTH_FRAC * width))
    edge_margin = _EDGE_MARGIN_FRAC * width

    splits = []
    start: int | None = None
    for x in range(width + 1):
        gap = x < width and is_gap[x]
        if gap and start is None:
            start = x
        elif not gap and start is not None:
            end = x
            if end - start >= min_gutter_width and start > edge_margin and end < width - edge_margin:
                splits.append((start + end) // 2)
            start = None
    return splits


def split_columns(image: Image) -> list[Image]:
    """Crops `image` into left-to-right reading columns at each detected gutter.

    Returns `[image]` unchanged when no gutter is found.
    """
    splits = find_column_splits(image)
    if not splits:
        return [image]

    bounds = [0, *splits, image.width]
    return [image.crop((left, 0, right, image.height)) for left, right in pairwise(bounds)]
