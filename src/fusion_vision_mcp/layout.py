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

#: A pixel darker than this (0-255 grayscale) counts as ink on an ordinary
#: dark-text-on-light-page scan.
_INK_THRESHOLD: Final[int] = 200

#: Plausible share of a text page that is ink. Outside this band the fixed
#: threshold above has clearly mis-read the page -- an inverted (light-on-dark)
#: page measures ~0.99 under it, and a washed-out or very low-contrast scan
#: measures ~0 -- and `_ink_mask` re-derives the threshold from the image itself.
#: Measured against this repo's own fixtures, which sit between 0.014 and 0.025.
_INK_FRACTION_BAND: Final[tuple[float, float]] = (0.0005, 0.5)

#: A column of pixels counts as part of a gutter (blank vertical strip) when its
#: ink density is at or below this fraction of the body height. Not exactly
#: zero: anti-aliased or compressed edges can leave a stray faint pixel in an
#: otherwise blank column.
_GUTTER_DENSITY_FRAC: Final[float] = 0.002

#: A gutter narrower than this fraction of the image width is ordinary
#: inter-word spacing, not a column boundary.
_MIN_GUTTER_WIDTH_FRAC: Final[float] = 0.01

#: A gutter must also be at least as wide as a typical text line is tall, up to
#: this fraction of the image width. A word space is a fraction of the text height
#: and a column gutter is usually wider than it: on VisionBench's rendered pages a
#: 10px word gap passed the 1% floor on a 1024px dashboard and split its lines
#: between words. The cap keeps the floor sane where "lines" are tall blocks.
_MAX_LINE_HEIGHT_GUTTER_FRAC: Final[float] = 0.03

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

#: A row counts as a text line (rather than inter-line gap or noise) when its ink
#: density is above this fraction of the image width.
_LINE_ROW_DENSITY_FRAC: Final[float] = 0.005

#: Splitting is skipped when the body has more than one text line but fewer than
#: this many. A single short paragraph of small text can have word-gaps that,
#: by chance, line up across its few lines closely enough to pass the gutter
#: test above -- measured on an 11pt three-line paragraph, whose two lines
#: falling within the body (the heading-exclusion band above absorbs part of
#: the first) produced six spurious splits, none of them real column
#: boundaries. With three or more independent lines, the same coincidence
#: becomes implausible -- every real multi-line fixture this module is tested
#: against (five and seven-line forms) clears this floor easily. A single
#: contiguous ink region (no internal line gaps at all, `_count_line_bands`
#: returns 1) is exempt: there is only one line, so there is nothing for word
#: gaps to coincidentally align across, and this is also what every
#: single-block synthetic test in `test_layout.py` looks like.
_MIN_LINES_FOR_SPLIT: Final[int] = 3

#: A text line crosses a gutter when more than this many of its ink pixels fall
#: inside the gutter strip. A gutter's per-column tolerance (`_GUTTER_DENSITY_FRAC`)
#: scales with the page height, so on a sparse page -- a slide, a dashboard -- a
#: strip can pass it while running through the rounded edges of letters: measured
#: on VisionBench's rendered pages (phase 14), 34 of 60 pages were split through a
#: line of text, cutting "Bicycles" into "Bicycl" + "es". A real gutter holds no
#: glyph at all, so any line with more than stray-pixel ink there is read whole.
_CROSSING_MAX_INK_PX: Final[int] = 2

#: The outermost columns of a gutter are not checked for crossing lines. They can
#: legitimately hold a pixel or two -- the gutter test tolerates that -- and the
#: anti-aliased edge of an opening quote mark at a column's left margin made a
#: whole newspaper column zone cross. Insetting by a quarter of the gutter instead
#: let dashboard lines through at word gaps, so this is a fixed, small margin.
_CROSSING_INSET_PX: Final[int] = 2


def _otsu_threshold(gray: NDArray[np.uint8]) -> int:
    """Otsu's method: the grey level that best separates the image into two classes."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return _INK_THRESHOLD

    probability = hist / total
    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(probability)
    mean_bg = np.cumsum(probability * levels)
    mean_total = mean_bg[-1]

    denominator = weight_bg * (1.0 - weight_bg)
    with np.errstate(divide="ignore", invalid="ignore"):
        between_class = (mean_total * weight_bg - mean_bg) ** 2 / denominator
    between_class[~np.isfinite(between_class)] = 0.0
    return int(np.argmax(between_class))


def _ink_mask(body: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Boolean ink mask for a page body, tolerant of inverted and low-contrast pages.

    The fixed `_INK_THRESHOLD` stays the primary path so ordinary dark-on-light scans
    binarize exactly as they always have -- this is bit-identical to
    `body < _INK_THRESHOLD` for every fixture in `tests/`. It only re-derives a
    threshold when the fixed one produces an implausible amount of ink, which is what a
    white-on-dark page (~0.99) or a washed-out scan (~0) looks like. Otsu then splits
    the histogram and the *minority* class is taken as ink, since background dominates
    a text page whichever way round it is printed.
    """
    ink = body < _INK_THRESHOLD
    low, high = _INK_FRACTION_BAND
    if low <= ink.mean() <= high:
        return ink

    threshold = _otsu_threshold(body)
    dark = body <= threshold
    light = ~dark
    candidate = dark if dark.mean() <= light.mean() else light
    # If re-deriving does not land in the plausible band either, there is no text-like
    # structure here; keep the original reading rather than inventing one.
    return candidate if low <= candidate.mean() <= high else ink


def _line_bands(ink: NDArray[np.bool_], floor: int | None = None) -> list[tuple[int, int]]:
    """Contiguous ink-bearing row runs (text lines) in an ink mask, as `(top, bottom)`.

    A row counts as text when it holds more than `floor` ink pixels, by default a
    small fraction of the width. A thin ruled divider running down a real gutter
    contributes at most a couple of ink pixels to any row it crosses, which stays
    under that default -- a fully blank inter-line row does not gain a false line
    just because a rule line passes through it.
    """
    density = ink.sum(axis=1)
    if floor is None:
        floor = max(1, int(_LINE_ROW_DENSITY_FRAC * ink.shape[1]))
    has_text = density > floor
    padded = np.concatenate(([False], has_text, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _count_line_bands(body: NDArray[np.uint8]) -> int:
    """Counts the text lines in a page body; see `_line_bands`."""
    return len(_line_bands(_ink_mask(body)))


def _close_thin_ink(is_gap: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Bridges thin ink runs (e.g. a ruled divider line) flanked by gutter on both sides."""
    closed = is_gap.copy()
    width = len(is_gap)
    if width == 0:
        return closed

    # Find each maximal run of ink (False) by its edges, then fill the ones that are
    # narrow enough to be a rule line *and* flanked by gutter on both sides. Same rule
    # as the index-walking loop this replaces, expressed as two edge arrays.
    padded = np.concatenate(([True], is_gap, [True]))
    starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    ends = np.flatnonzero(~padded[:-1] & padded[1:])
    for start, end in zip(starts, ends, strict=True):
        if start > 0 and end < width and end - start <= _RULE_LINE_MAX_WIDTH_PX:
            closed[start:end] = True
    return closed


def _find_gutters(image: Image) -> tuple[list[tuple[int, int]], NDArray[np.bool_]]:
    """Gutters as `(left, right)` x-ranges, plus which x-columns hold a bridged rule line.

    See `find_column_splits` for what counts as a gutter.
    """
    arr = np.asarray(image.convert("L"))
    height, width = arr.shape
    body = arr[int(height * _TOP_BAND_FRAC) :, :]
    no_rules = np.zeros(width, dtype=bool)

    line_bands = _count_line_bands(body)
    if 1 < line_bands < _MIN_LINES_FOR_SPLIT:
        return [], no_rules

    ink = _ink_mask(body)
    density = ink.sum(axis=0)
    raw_gap = density <= max(1, int(_GUTTER_DENSITY_FRAC * body.shape[0]))
    is_gap = _close_thin_ink(raw_gap)

    heights = [bottom - top for top, bottom in _line_bands(ink)]
    line_height = int(np.median(heights)) if heights else 0
    min_gutter_width = max(
        4,
        int(_MIN_GUTTER_WIDTH_FRAC * width),
        min(line_height, int(_MAX_LINE_HEIGHT_GUTTER_FRAC * width)),
    )
    edge_margin = _EDGE_MARGIN_FRAC * width

    # Same edge-pair scan as above, over gutter runs this time.
    padded = np.concatenate(([False], is_gap, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    gutters = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= min_gutter_width and start > edge_margin and end < width - edge_margin
    ]
    return gutters, is_gap & ~raw_gap


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

    These are candidates over the page body. `reading_blocks` decides, line by
    line, where they actually apply.
    """
    return [(left + right) // 2 for left, right in _find_gutters(image)[0]]


def reading_blocks(image: Image) -> list[tuple[Image, int, int]]:
    """Crops `image` into blocks in reading order, as `(crop, left, top)`.

    A page with no gutter is one block, the image itself. Otherwise the page is
    cut into horizontal zones between text lines. A line that runs through any
    gutter -- a title over the columns, an intro paragraph, a table heading, a
    word that happens to straddle a gap -- is read across the full width, so it
    is never cut in half. A run of lines clear of every gutter is split into
    columns and read one column at a time, unless no column in it holds
    `_MIN_LINES_FOR_SPLIT` lines: a gap lining up across one or two lines is
    most likely a coincidence of word spacing. A page that is one ink region is
    exempt, as in `find_column_splits`.
    """
    gutters, rule_columns = _find_gutters(image)
    if not gutters:
        return [(image, 0, 0)]

    ink = _ink_mask(np.asarray(image.convert("L")))
    height, width = ink.shape
    keep = ~rule_columns

    def crosses(top: int, bottom: int) -> bool:
        for left, right in gutters:
            inset = min(_CROSSING_INSET_PX, (right - left) // 4)
            mid = slice(left + inset, right - inset)
            if int(ink[top:bottom, mid][:, keep[mid]].sum()) > _CROSSING_MAX_INK_PX:
                return True
        return False

    def column_lines(top: int, bottom: int, left: int, right: int) -> int:
        return len(_line_bands(ink[top:bottom, left:right]))

    # Group consecutive lines by whether they cross, then give each zone the page
    # rows halfway to its neighbours so the zones tile the page with nothing lost.
    # Any ink makes a row part of a line here, rule lines aside, so a zone boundary
    # always falls on a blank row: the default floor leaves a line's descenders and
    # accents outside its band, and a boundary there cut lines in half.
    bands = _line_bands(ink[:, keep], floor=0)
    zones: list[tuple[bool, list[tuple[int, int]]]] = []
    for band in bands:
        crossing = crosses(*band)
        if zones and zones[-1][0] == crossing:
            zones[-1][1].append(band)
        else:
            zones.append((crossing, [band]))

    bounds = [0, *((prev[1][-1][1] + nxt[1][0][0]) // 2 for prev, nxt in pairwise(zones)), height]
    cuts = [0, *((left + right) // 2 for left, right in gutters), width]
    blocks: list[tuple[Image, int, int]] = []
    for (crossing, _), (top, bottom) in zip(zones, pairwise(bounds), strict=True):
        # Lines are counted per column: columns set at offset baselines leave no
        # row blank across the page, so the full-width bands above merge them.
        most_lines = max(column_lines(top, bottom, left, right) for left, right in pairwise(cuts))
        if crossing or (most_lines < _MIN_LINES_FOR_SPLIT and len(bands) > 1):
            blocks.append((image.crop((0, top, width, bottom)), 0, top))
            continue
        blocks.extend(
            (image.crop((left, top, right, bottom)), left, top)
            for left, right in pairwise(cuts)
            if ink[top:bottom, left:right].any()
        )
    return blocks or [(image, 0, 0)]
