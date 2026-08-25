#  geometry.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Geometric measurements over segmentation masks.

Language models judge images semantically but estimate quantities poorly: whether
two things *touch*, how far apart they are, or how deeply one is buried inside
another are exactly the questions they answer unreliably by eye. These functions
turn a pair of masks into numbers so the caller can apply its own judgement to a
measurement rather than to an impression.

Pure numpy/scipy — no model loading — so every function here is cheap to test in
isolation from Florence-2, Moondream or SAM2.

A note on what is measurable: the metrics below are aggregate statistics over a
mask (principal axes, band centroids, overlap fractions, distance transforms).
They deliberately avoid the medial axis / skeleton, which is unstable on the
rough silhouettes typical of photographic subjects: a single bump on the contour
spawns a spurious branch, and that noise swamps the signal being measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    generate_binary_structure,
    label,
)
from scipy.spatial import ConvexHull, QhullError

Mask = NDArray[np.bool_]

#: Bands used when profiling an object along its long axis.
_PROFILE_BINS: Final[int] = 20

#: Fraction of the long axis averaged together when measuring width at one end.
_END_BAND: Final[float] = 0.08

#: Masks closer than this (in image pixels) are reported as touching rather than
#: separate. SAM2 decodes at a fixed 256x256 grid, so boundaries carry roughly a
#: pixel of slop once upscaled; treating a sub-pixel gap as contact avoids
#: reporting a "gap" that is really just resampling error.
_TOUCH_TOLERANCE_PX: Final[float] = 2.0


def _principal_frame(mask: Mask) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Project a mask's pixels onto its own long and short axes.

    Returns ``(along, across, eigenvalues)`` where ``along`` runs down the
    object's longest extent and ``across`` is perpendicular to it.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    perp = np.array([-axis[1], axis[0]])
    return pts @ axis, pts @ perp, eigvals


def elongation(mask: Mask) -> float:
    """How long-and-thin an object is: 1.0 is circular, higher is more elongated."""
    if mask.sum() < 10:
        return float("nan")
    _, _, eigvals = _principal_frame(mask)
    return float(np.sqrt(max(eigvals) / max(float(min(eigvals)), 1e-9)))


def width_profile(mask: Mask) -> dict[str, float]:
    """Width at each end of the long axis, and at the middle.

    Distinguishes shapes that taper once from shapes that taper at both ends.
    A blade tapers to a point at the tip and stops at a wider hilt; something
    pointed at *both* ends has a different signature even when it is a single
    connected region that no object detector would split in two.

    ``end_symmetry`` is ``min(end_a, end_b) / max(end_a, end_b)``: near 1.0 means
    both ends are equally wide, near 0 means one end is far narrower.
    """
    if mask.sum() < 50:
        return {
            "end_a_width": float("nan"),
            "mid_width": float("nan"),
            "end_b_width": float("nan"),
            "end_symmetry": float("nan"),
        }

    along, across, _ = _principal_frame(mask)
    lo, hi = float(along.min()), float(along.max())
    span = hi - lo

    def width_at(position: float) -> float:
        band = (along >= position - span * _END_BAND) & (along <= position + span * _END_BAND)
        return float(across[band].max() - across[band].min()) if band.any() else 0.0

    end_a, end_b = width_at(lo), width_at(hi)
    widest = max(end_a, end_b)
    return {
        "end_a_width": end_a,
        "mid_width": width_at((lo + hi) / 2),
        "end_b_width": end_b,
        "end_symmetry": (min(end_a, end_b) / widest) if widest > 0 else float("nan"),
    }


def straightness(mask: Mask, bins: int = _PROFILE_BINS) -> dict[str, float]:
    """How far an object's centreline strays from straight, and from a smooth curve.

    The centreline is built from the centroid of each band across the long axis,
    so every point averages the object's full width there and surface texture
    cancels out.

    ``max_deviation`` is the largest offset from a straight line, as a fraction of
    the object's length: a straight rod is near 0, a banana is large. ``kink`` is
    the largest offset from a fitted *quadratic*, which a smoothly bent object
    still fits well — so a high ``kink`` with a low ``max_deviation`` points at an
    abrupt local direction change rather than an overall bow.
    """
    empty = {"max_deviation": float("nan"), "kink": float("nan")}
    if mask.sum() < 50:
        return empty

    along, across, _ = _principal_frame(mask)
    edges = np.linspace(along.min(), along.max(), bins + 1)
    centres_a, centres_c = [], []
    for i in range(bins):
        band = (along >= edges[i]) & (along <= edges[i + 1])
        if band.sum() > 5:
            centres_a.append(float(along[band].mean()))
            centres_c.append(float(across[band].mean()))
    if len(centres_a) < 5:
        return empty

    a = np.asarray(centres_a)
    c = np.asarray(centres_c)
    length = float(a.max() - a.min())
    if length <= 0:
        return empty

    linear = np.polyval(np.polyfit(a, c, 1), a)
    quadratic = np.polyval(np.polyfit(a, c, 2), a)
    return {
        "max_deviation": float(np.abs(c - linear).max()) / length,
        "kink": float(np.abs(c - quadratic).max()) / length,
    }


def relation(a: Mask, b: Mask) -> dict[str, Any]:
    """Measure how two masks sit relative to each other.

    ``state`` is one of ``separate``, ``touching`` or ``overlapping``.

    ``gap`` is the shortest distance between them in pixels, and is ``0.0`` once
    they overlap. Two things that ought to be in contact — a hand and the grip it
    holds — but report a gap of tens of pixels are not in contact.

    ``a_inside_b`` / ``b_inside_a`` give the share of one mask's area falling
    within the other. ``embed_depth`` is how far inside the overlapping pixels
    reach, measured to the nearest boundary of ``b``: a hand curled around a
    shield's rim overlaps it shallowly, while a hand fused into the middle of the
    shield face overlaps it deeply. Depth is what separates the two cases; the
    overlap fraction alone does not.
    """
    area_a, area_b = int(a.sum()), int(b.sum())
    if area_a == 0 or area_b == 0:
        return {
            "state": "empty",
            "gap": float("nan"),
            "a_inside_b": float("nan"),
            "b_inside_a": float("nan"),
            "embed_depth": float("nan"),
        }

    both = a & b
    if not both.any():
        gap = float(distance_transform_edt(~b)[a].min())
        return {
            "state": "touching" if gap <= _TOUCH_TOLERANCE_PX else "separate",
            "gap": gap,
            "a_inside_b": 0.0,
            "b_inside_a": 0.0,
            "embed_depth": 0.0,
        }

    return {
        "state": "overlapping",
        "gap": 0.0,
        "a_inside_b": float(both.sum()) / area_a,
        "b_inside_a": float(both.sum()) / area_b,
        "embed_depth": float(distance_transform_edt(b)[both].mean()),
    }


#: Below this many pixels a mask carries no shape to measure.
_MIN_MEASURABLE_AREA: Final[int] = 100

#: Working grid the silhouette is subsampled to before any lobe measurement. Matches
#: SAM2's own decode resolution: a mask upscaled from a 256 grid carries no finer
#: detail, so measuring above it costs time without adding information, and fixing it
#: is what makes the answer independent of the source image's size.
_WORK_RESOLUTION: Final[int] = 256

#: Holes smaller than this share of the silhouette are decode noise and get filled.
#: Anything larger is structure -- petals in a ring enclose a real central hole, and
#: filling it manufactures a hub that then counts as an extra lobe.
_MAX_FILLED_HOLE_FRACTION: Final[float] = 0.02

#: Distance transform smoothing, as a fraction of the object's *own* inradius rather
#: than a pixel count -- which is what keeps the measure scale-free. A contour bump of
#: amplitude `a` perturbs the transform by about `a`, and a segmented boundary is only
#: accurate to about a pixel, so structure narrower than a tenth of the inradius is
#: surface texture. This is the parameter separating this measure from the skeleton
#: approaches that produced inverted results on real photographs.
_EDT_SMOOTH: Final[float] = 0.10

#: Floor under that smoothing, in working-grid pixels. A segmented boundary carries a
#: pixel or two of slop no matter how small the object is, so a thin shape must still
#: be smoothed by more than that slop or its own roughness reads as a row of lobes.
#: Measured, not chosen: a rod with +/-2px edge jitter splits into 8 spurious lobes at
#: a floor of 1.0 and holds at 1 from 2.0 upward, at every sigma tried. The floor is
#: what defends against surface texture; the scale-relative term above cannot, because
#: on a thin object a tenth of its inradius is under a pixel. Raising sigma past 0.20
#: instead starts merging real lobes -- 8 discs at 40% overlap collapse to 1 at 0.25.
_SMOOTH_FLOOR_PX: Final[float] = 2.0

#: Inradius below which the shape is a thread on the working grid and has no lobes.
_MIN_INRADIUS_PX: Final[float] = 3.0

#: Superlevel range swept, as fractions of the smoothed peak. Above `_LEVEL_HI`
#: discretisation jitter dominates; below `_LEVEL_LO` the level set is essentially the
#: silhouette itself and carries no lobe information.
_LEVEL_HI: Final[float] = 0.95
_LEVEL_LO: Final[float] = 0.20
_LEVEL_STEP: Final[float] = 0.01

#: A component smaller than a 3x3 patch on the working grid is below the resolution
#: the mask actually carries.
_MIN_SEED_AREA_PX: Final[int] = 9

#: Share of the swept range a count must hold *contiguously* to be believed. A count
#: that flickers on and off down the sweep is a wiggle in the transform, not a saddle
#: between two lobes; requiring a run is persistence without a merge tree.
_MIN_SUPPORT: Final[float] = 0.06

#: Ceiling on lobes. Above this the rosette hypothesis is untestable at the angular
#: resolution used below, and a sweep producing more cores is far likelier to be
#: shattering on a rough silhouette than counting.
_MAX_LOBES: Final[int] = 24

#: Widest level runs reported back as raw evidence.
_MAX_REPORTED_RUNS: Final[int] = 6

#: Above this elongation a shape is a row rather than a rosette, and the angular
#: harmonic is meaningless -- five discs in a row produce notches at uneven angles.
_MAX_ROSETTE_ELONGATION: Final[float] = 1.6

#: Angular bins for the radial profile, and the circular smoothing applied to it.
#: 3 bins is 3 degrees: enough to kill single-pixel outline spikes while leaving a
#: 24-fold harmonic (15 degrees per lobe) resolvable.
_ANGLE_BINS: Final[int] = 360
_ANGLE_SMOOTH: Final[float] = 3.0

#: Share of angular bins allowed to be empty before the radial estimator gives up --
#: a crescent does not surround its own centroid.
_MAX_EMPTY_BINS: Final[float] = 0.05

#: Below this hull-residual spread the outline is convex and has exactly one lobe;
#: searching its noise for a harmonic would return an arbitrary number.
_MIN_NOTCH: Final[float] = 0.02


def _unmeasurable_lobes(lobes: int) -> dict[str, Any]:
    """The all-`nan` shape, for masks too small or too thin to carry lobe structure."""
    return {
        "lobes": lobes,
        "by_distance": lobes,
        "distance_support": float("nan"),
        "by_radial": 0,
        "radial_strength": float("nan"),
        "notch_depth": float("nan"),
        "solidity": float("nan"),
        "agreement": False,
        "inradius": 0.0,
        "level_runs": [],
        "shattered": False,
        "clipped": False,
    }


def _to_working_grid(mask: Mask) -> tuple[Mask, int, bool]:
    """Crop to the silhouette, subsample to a fixed grid, and fill only small holes.

    Returns ``(work, stride, clipped)``. ``clipped`` records that the silhouette runs
    off the array edge, where the frame boundary is a false contour and any lobe count
    is a lower bound.
    """
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    clipped = y0 == 0 or x0 == 0 or y1 == mask.shape[0] or x1 == mask.shape[1]

    crop = mask[y0:y1, x0:x1]
    stride = max(1, int(np.ceil(max(crop.shape) / _WORK_RESOLUTION)))
    # Pad so the distance transform sees a real boundary on every side.
    work = np.pad(crop[::stride, ::stride], 1, constant_values=False)

    filled = binary_fill_holes(work)
    if filled is not None and filled.any():
        holes, count = label(filled & ~work)
        if count:
            areas = np.bincount(holes.ravel())[1:]
            small = np.flatnonzero(areas < _MAX_FILLED_HOLE_FRACTION * work.sum()) + 1
            if small.size:
                work = work | np.isin(holes, small)

    return work, stride, clipped


def _distance_lobes(work: Mask) -> dict[str, Any]:
    """Count separable cores by sweeping superlevel sets of the distance transform.

    Each level is a slice through the object's own thickness. Two lobes joined by a
    narrow neck separate into two components as soon as the level rises above the
    neck, and stay separate for a range of levels; a bump on a rough contour separates
    for one or two levels and rejoins. Requiring a *contiguous run* of levels is what
    distinguishes the two without building a merge tree over the transform.
    """
    distance = distance_transform_edt(work)
    assert isinstance(distance, np.ndarray)
    raw_peak = float(distance.max())
    if raw_peak < _MIN_INRADIUS_PX:
        return {"by_distance": 1, "distance_support": float("nan"), "runs": [], "shattered": False, "peak": raw_peak}

    smoothed = gaussian_filter(distance, sigma=max(_SMOOTH_FLOOR_PX, _EDT_SMOOTH * raw_peak))
    smoothed[~work] = 0.0
    peak = float(smoothed.max())
    if peak <= 0:
        return {"by_distance": 1, "distance_support": float("nan"), "runs": [], "shattered": False, "peak": raw_peak}

    neighbourhood = generate_binary_structure(2, 2)
    levels = np.arange(_LEVEL_HI, _LEVEL_LO - 1e-9, -_LEVEL_STEP)
    counts = []
    for level in levels:
        parts, found = label(smoothed > level * peak, structure=neighbourhood)
        if not found:
            counts.append(0)
            continue
        sizes = np.bincount(parts.ravel())[1:]
        counts.append(int((sizes >= _MIN_SEED_AREA_PX).sum()))

    # Widest contiguous run per distinct count, as a share of the swept range.
    runs: list[tuple[int, float, float]] = []
    start = 0
    for index in range(1, len(counts) + 1):
        if index == len(counts) or counts[index] != counts[start]:
            runs.append((counts[start], float(levels[start]), float(levels[index - 1])))
            start = index

    support: dict[int, float] = {}
    span = float(levels[0] - levels[-1]) or 1.0
    for count, hi, lo in runs:
        support[count] = max(support.get(count, 0.0), (hi - lo) / span)

    # `max`, not best-supported: the count 1 always owns the entire low end of the
    # sweep, so "most supported" would answer 1 on every input. The support width is
    # reported alongside so thin evidence stays visible instead of being hidden.
    candidates = [c for c, s in support.items() if 1 < c <= _MAX_LOBES and s >= _MIN_SUPPORT]
    by_distance = max(candidates) if candidates else 1

    widest = sorted(runs, key=lambda r: r[1] - r[2], reverse=True)[:_MAX_REPORTED_RUNS]
    return {
        "by_distance": by_distance,
        "distance_support": support.get(by_distance, 0.0),
        "runs": [[c, round(hi, 2), round(lo, 2)] for c, hi, lo in sorted(widest, key=lambda r: -r[1])],
        "shattered": any(c > _MAX_LOBES for c in counts),
        "peak": raw_peak,
    }


def _hull_radii(points: NDArray[np.float64], centre: NDArray[np.float64]) -> tuple[NDArray[np.float64], float] | None:
    """Distance from ``centre`` to the convex hull, per angular bin, plus hull area.

    Dividing the silhouette's own radius by this turns the profile into a measure of
    *concavity*. That matters: a rectangle's raw radius profile is a strong 4-fold
    signal, so a raw-radius harmonic reports a square as four lobes. Every convex
    shape has zero hull residual by construction, so the negative controls pass
    structurally rather than by tuning a threshold.
    """
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None

    angles = (np.arange(_ANGLE_BINS) + 0.5) * (2 * np.pi / _ANGLE_BINS)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Facets are `A . x + b <= 0`; along direction u the hull is hit at -(A.c + b)/(A.u).
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    slack = -(normals @ centre + offsets)
    projection = directions @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = np.where(projection > 1e-12, slack / projection, np.inf)
    radii = distances.min(axis=1)
    if not np.isfinite(radii).all():
        return None
    return radii, float(hull.volume)


def _radial_lobes(work: Mask) -> dict[str, Any]:
    """Count repeats around the centroid, for rosette arrangements only.

    Returns ``by_radial = 0`` -- meaning *not measured*, never zero lobes -- whenever
    the arrangement is not one this can speak to: anything elongated enough to be a
    row rather than a ring, or a shape whose centroid falls outside itself.
    """
    unmeasured = {
        "by_radial": 0,
        "radial_strength": float("nan"),
        "notch_depth": float("nan"),
        "solidity": float("nan"),
    }

    ys, xs = np.nonzero(work)
    centre = np.array([xs.mean(), ys.mean()])
    if elongation(work) > _MAX_ROSETTE_ELONGATION:
        return unmeasured
    # Deliberately no "centroid must lie inside the mask" test: a ring of petals has a
    # genuine hole at its centre, which is exactly the arrangement this estimator is
    # for. Shapes that fail to surround their centroid are rejected by the empty-bin
    # check below instead, which is what actually distinguishes a crescent.

    boundary = work & ~binary_erosion(work)
    by, bx = np.nonzero(boundary)
    if by.size < 8:
        return unmeasured

    # The hull only needs the outline, but the angular profile needs every pixel: a
    # silhouette's boundary carries roughly as many pixels as there are bins, which
    # leaves a tenth of them empty by chance and trips the guard below on a plain disc.
    hull = _hull_radii(np.stack([bx, by], axis=1).astype(np.float64), centre)
    if hull is None:
        return unmeasured
    hull_radii, hull_area = hull
    solidity = float(work.sum()) / hull_area if hull_area > 0 else float("nan")

    offsets = np.stack([xs, ys], axis=1).astype(np.float64) - centre
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    bins = ((np.arctan2(offsets[:, 1], offsets[:, 0]) % (2 * np.pi)) / (2 * np.pi) * _ANGLE_BINS).astype(int)
    bins = np.clip(bins, 0, _ANGLE_BINS - 1)

    profile = np.zeros(_ANGLE_BINS)
    np.maximum.at(profile, bins, radii)
    if (np.bincount(bins, minlength=_ANGLE_BINS) == 0).mean() > _MAX_EMPTY_BINS:
        return {**unmeasured, "solidity": solidity}

    notch = gaussian_filter1d(1.0 - profile / hull_radii, sigma=_ANGLE_SMOOTH, mode="wrap")
    notch_depth = float(notch.std())
    if notch_depth < _MIN_NOTCH:
        # Convex outline: one lobe, and no harmonic worth reading out of the noise.
        return {"by_radial": 1, "radial_strength": float("nan"), "notch_depth": notch_depth, "solidity": solidity}

    spectrum = np.abs(np.fft.rfft(notch - notch.mean()))
    band = spectrum[2 : _MAX_LOBES + 1]
    if not band.size or band.sum() <= 0:
        return {**unmeasured, "notch_depth": notch_depth, "solidity": solidity}

    # k=1 is a single dent -- a comma is one lobe with a bite out of it, not a repeat.
    best = int(np.argmax(band)) + 2
    return {
        "by_radial": best,
        "radial_strength": float(band.max() / spectrum[1 : _MAX_LOBES + 1].sum()),
        "notch_depth": notch_depth,
        "solidity": solidity,
    }


def count_lobes(mask: Mask) -> dict[str, Any]:
    """Estimate how many repeated parts compose one silhouette.

    Detectors collapse instances that visually merge: eight overlapping shapes come
    back as a single region, and so does a flower's worth of overlapping petals. The
    silhouette still carries the evidence, because an outline stays notched where the
    interiors have merged. This measures that, and deliberately reports two
    independent estimates rather than arbitrating between them.

    ``by_distance`` sweeps superlevel sets of the distance transform and counts cores
    that persist across a contiguous run of levels; it handles rows and rings alike.
    ``by_radial`` looks for angular repeats in the outline's departure from its own
    convex hull, and applies only to rosettes -- it reports ``0`` for *not measured*
    rather than guessing on a row. ``agreement`` is true only when both saw the same
    number, and ``distance_support`` says how much of the sweep backed the headline
    figure, so thin evidence is visible rather than hidden.

    Two limits are structural. Lobes overlapping past roughly 60% of their diameter
    leave no saddle to find and come back as 1 -- at that point the silhouette really
    is one blob. Unequal lobes are normalised against the largest, so a small one
    merges early: this undercounts, never overcounts. ``shattered`` and ``clipped``
    mark results that should not be used at all.
    """
    if mask.sum() < _MIN_MEASURABLE_AREA:
        return _unmeasurable_lobes(int(mask.any()))

    work, stride, clipped = _to_working_grid(mask)
    if work.sum() < _MIN_SEED_AREA_PX:
        return {**_unmeasurable_lobes(1), "clipped": clipped}

    distance = _distance_lobes(work)
    if not distance["runs"]:
        return {**_unmeasurable_lobes(1), "clipped": clipped, "inradius": distance["peak"] * stride}

    radial = _radial_lobes(work)
    by_distance = int(distance["by_distance"])
    by_radial = int(radial["by_radial"])

    return {
        "lobes": by_distance,
        "by_distance": by_distance,
        "distance_support": float(distance["distance_support"]),
        "by_radial": by_radial,
        "radial_strength": radial["radial_strength"],
        "notch_depth": radial["notch_depth"],
        "solidity": radial["solidity"],
        "agreement": by_radial > 0 and by_radial == by_distance,
        "inradius": distance["peak"] * stride,
        "level_runs": distance["runs"],
        "shattered": bool(distance["shattered"]),
        "clipped": clipped,
    }


def describe(mask: Mask) -> dict[str, Any]:
    """Shape metrics for a single object."""
    return {
        "area": int(mask.sum()),
        "elongation": elongation(mask),
        "straightness": straightness(mask),
        "width_profile": width_profile(mask),
        "lobes": count_lobes(mask),
    }


#: The two gridlines rule-of-thirds composition splits each axis into.
_THIRDS: Final[tuple[float, float]] = (1 / 3, 2 / 3)


def rule_of_thirds(box: Sequence[float], image_size: tuple[int, int]) -> dict[str, Any]:
    """How closely a bounding box's centroid sits to a rule-of-thirds gridline intersection.

    Unlike the metrics above, this is frame-relative rather than mask-relative: it
    only needs a box and the frame's dimensions, not a silhouette.

    ``thirds_offset`` near 0 means the subject sits on a rule-of-thirds power
    point; ``center_offset`` near 0 means it sits in the middle of the frame
    instead. Both are normalized by the frame's diagonal, so they are comparable
    across image sizes.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = image_size
    diagonal = float(np.hypot(width, height))

    empty = {
        "centroid": [cx, cy],
        "thirds_offset": float("nan"),
        "center_offset": float("nan"),
        "nearest_gridpoint": [cx, cy],
    }
    if diagonal <= 0:
        return empty

    gridpoints = [(width * gx, height * gy) for gx in _THIRDS for gy in _THIRDS]
    nearest = min(gridpoints, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    return {
        "centroid": [cx, cy],
        "thirds_offset": float(np.hypot(nearest[0] - cx, nearest[1] - cy)) / diagonal,
        "center_offset": float(np.hypot(width / 2 - cx, height / 2 - cy)) / diagonal,
        "nearest_gridpoint": [nearest[0], nearest[1]],
    }
