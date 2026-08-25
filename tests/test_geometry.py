#  test_geometry.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for the mask measurements.

These build masks directly rather than segmenting a photograph, so the expected
answer is known exactly and no model has to be downloaded to run them.
"""

import numpy as np
import pytest

from fusion_vision_mcp import geometry

SIZE = 200


def blank() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=bool)


def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = blank()
    mask[y0:y1, x0:x1] = True
    return mask


def disc(cx: int, cy: int, radius: int, size: int = SIZE) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius**2


def ring(n: int, ring_radius: float, disc_radius: int, size: int = SIZE) -> np.ndarray:
    """`n` discs arranged evenly around a circle -- the rosette arrangement."""
    mask = np.zeros((size, size), dtype=bool)
    for i in range(n):
        angle = 2 * np.pi * i / n
        cx = size / 2 + ring_radius * np.cos(angle)
        cy = size / 2 + ring_radius * np.sin(angle)
        mask |= disc(round(cx), round(cy), disc_radius, size=size)
    return mask


def row(n: int, spacing: int, disc_radius: int, size: int = SIZE) -> np.ndarray:
    """`n` discs in a straight line -- the arrangement the radial estimator refuses."""
    mask = np.zeros((size, size), dtype=bool)
    start = size / 2 - spacing * (n - 1) / 2
    for i in range(n):
        mask |= disc(round(start + i * spacing), size // 2, disc_radius, size=size)
    return mask


def test_separate_masks_report_the_distance_between_them() -> None:
    left = rect(10, 90, 30, 110)
    right = rect(70, 90, 90, 110)
    result = geometry.relation(left, right)

    assert result["state"] == "separate"
    assert result["gap"] == pytest.approx(40, abs=1.5)
    assert result["a_inside_b"] == 0.0


def test_masks_a_pixel_apart_count_as_touching() -> None:
    left = rect(10, 90, 50, 110)
    right = rect(51, 90, 90, 110)

    assert geometry.relation(left, right)["state"] == "touching"


def test_a_mask_fully_inside_another_reports_full_containment_and_depth() -> None:
    small = disc(100, 100, 10)
    large = disc(100, 100, 50)
    result = geometry.relation(small, large)

    assert result["state"] == "overlapping"
    assert result["a_inside_b"] == pytest.approx(1.0)
    assert result["b_inside_a"] < 0.1
    # Every pixel of the small disc is at least 40px from the large disc's edge.
    assert result["embed_depth"] > 35


def test_a_mask_overlapping_only_at_the_rim_is_shallow() -> None:
    """The signal that separates gripping a rim from being buried in a face."""
    large = disc(100, 100, 50)
    straddling = disc(148, 100, 10)
    deep = disc(100, 100, 10)

    shallow_result = geometry.relation(straddling, large)
    deep_result = geometry.relation(deep, large)

    assert shallow_result["state"] == "overlapping"
    assert shallow_result["embed_depth"] < deep_result["embed_depth"] / 3


def test_elongation_separates_a_bar_from_a_square() -> None:
    assert geometry.elongation(rect(10, 95, 190, 105)) > 8
    assert geometry.elongation(rect(80, 80, 120, 120)) == pytest.approx(1.0, abs=0.15)


def test_a_straight_bar_has_a_near_zero_deviation() -> None:
    result = geometry.straightness(rect(10, 95, 190, 105))

    assert result["max_deviation"] < 0.02
    assert result["kink"] < 0.02


def test_a_bent_bar_deviates_from_straight() -> None:
    mask = blank()
    for x in range(10, 100):
        mask[95 + (x - 10) // 3 : 105 + (x - 10) // 3, x] = True
    for x in range(100, 190):
        mask[125 - (x - 100) // 3 : 135 - (x - 100) // 3, x] = True

    assert geometry.straightness(mask)["max_deviation"] > 0.05


def test_width_profile_distinguishes_one_taper_from_two() -> None:
    """A blade tapers at the tip only; something pointed at both ends does not."""
    one_taper = blank()
    two_tapers = blank()
    for i, x in enumerate(range(20, 180)):
        half_a = max(1, (160 - i) // 12)
        one_taper[100 - half_a : 100 + half_a, x] = True
        half_b = max(1, int(12 - abs(i - 80) * 0.14))
        two_tapers[100 - half_b : 100 + half_b, x] = True

    single = geometry.width_profile(one_taper)
    double = geometry.width_profile(two_tapers)

    assert single["end_symmetry"] < 0.5
    assert double["end_symmetry"] > 0.7


def test_an_empty_mask_does_not_raise() -> None:
    result = geometry.relation(blank(), disc(100, 100, 10))

    assert result["state"] == "empty"
    assert np.isnan(result["gap"])


def test_describe_reports_every_metric() -> None:
    result = geometry.describe(rect(10, 95, 190, 105))

    assert result["area"] == 180 * 10
    assert set(result) == {"area", "elongation", "straightness", "width_profile", "lobes"}


# --- count_lobes: negative controls -------------------------------------------------
#
# Written first and deliberately: a lobe count that splits a single object is worse
# than useless. Per CLAUDE.md, a metric whose known-good case scores worse than its
# known-bad one gets dropped rather than tuned, so these decide whether the measure
# ships at all.


def test_a_single_disc_is_one_lobe() -> None:
    result = geometry.count_lobes(disc(100, 100, 60))

    assert result["lobes"] == 1
    assert result["by_distance"] == 1
    assert result["by_radial"] == 1
    assert result["notch_depth"] < 0.02
    assert result["solidity"] > 0.95


def test_a_long_thin_rod_is_one_lobe() -> None:
    """A rod's raw radial profile is violently non-circular; it must still not split."""
    result = geometry.count_lobes(rect(10, 95, 190, 105))

    assert result["lobes"] == 1
    # Not measured, rather than the 2 or 4 a raw-radius harmonic would report.
    assert result["by_radial"] == 0


def test_a_rough_edged_rod_is_still_one_lobe() -> None:
    """Surface texture must not become lobes -- the failure skeletons produced here."""
    rng = np.random.default_rng(0)
    mask = blank()
    for x in range(10, 190):
        top = 95 + int(rng.integers(-2, 3))
        bottom = 105 + int(rng.integers(-2, 3))
        mask[top:bottom, x] = True

    result = geometry.count_lobes(mask)

    assert result["lobes"] == 1
    assert result["shattered"] is False


def test_an_ellipse_is_one_lobe_not_two() -> None:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    ellipse = (((xs - 100) / 75.0) ** 2 + ((ys - 100) / 25.0) ** 2) <= 1.0

    result = geometry.count_lobes(ellipse)

    assert result["by_distance"] == 1
    assert result["by_radial"] == 0


def test_a_square_is_one_lobe_not_four() -> None:
    """The hull-residual control: a raw-radius harmonic reports a square as 4."""
    result = geometry.count_lobes(rect(60, 60, 140, 140))

    assert result["lobes"] == 1
    assert result["by_radial"] == 1


def test_an_empty_or_tiny_mask_does_not_raise() -> None:
    empty = geometry.count_lobes(blank())
    tiny = geometry.count_lobes(disc(100, 100, 2))

    assert empty["lobes"] == 0
    assert np.isnan(empty["distance_support"])
    assert tiny["lobes"] == 1
    assert np.isnan(tiny["distance_support"])


def test_a_convex_blob_scores_lower_than_a_lobed_one() -> None:
    """The comparative control: assert ordering, not absolute thresholds."""
    solid = geometry.count_lobes(disc(100, 100, 60))
    lobed = geometry.count_lobes(ring(8, 42.0, 22))

    assert solid["solidity"] > lobed["solidity"]
    assert solid["notch_depth"] < lobed["notch_depth"]


# --- count_lobes: positives ---------------------------------------------------------


def test_eight_overlapping_discs_in_a_ring_count_as_eight() -> None:
    """The exact measured failure: detectors return 1 for this arrangement."""
    result = geometry.count_lobes(ring(8, 42.0, 22))

    assert result["lobes"] == 8
    assert result["by_distance"] == 8
    assert result["by_radial"] == 8
    assert result["agreement"] is True


@pytest.mark.parametrize("overlap", [0.10, 0.25, 0.40])
def test_moderate_overlap_still_counts_eight(overlap: float) -> None:
    radius = 22
    spacing = 2 * radius * (1 - overlap)
    result = geometry.count_lobes(ring(8, spacing / (2 * np.sin(np.pi / 8)), radius))

    assert result["by_distance"] == 8


@pytest.mark.parametrize("overlap", [0.10, 0.25, 0.40, 0.55, 0.70])
def test_heavy_overlap_undercounts_rather_than_overcounting(overlap: float) -> None:
    """The safety property: past the resolvable limit it falls back toward 1.

    Support is deliberately not compared across overlaps -- it describes whichever
    count was reported, so a confident 1 and a confident 8 both score high and the
    comparison would be meaningless. What must hold is that no arrangement of eight
    lobes is ever read as more than eight.
    """
    spacing = 2 * 22 * (1 - overlap)
    result = geometry.count_lobes(ring(8, spacing / (2 * np.sin(np.pi / 8)), 22))

    assert 1 <= result["by_distance"] <= 8


def test_two_discs_in_a_row_count_as_two() -> None:
    # `by_radial` is 0 here by design -- a row is not a rosette, so the distance
    # estimator carries this case alone.
    result = geometry.count_lobes(row(2, 34, 22))

    assert result["by_distance"] == 2


def test_five_discs_in_a_row_count_as_five() -> None:
    assert geometry.count_lobes(row(5, 34, 22))["by_distance"] == 5


def test_three_separated_discs_count_as_three() -> None:
    mask = disc(40, 100, 18) | disc(100, 100, 18) | disc(160, 100, 18)
    result = geometry.count_lobes(mask)

    assert result["lobes"] == 3
    assert result["distance_support"] > 0.8


def test_the_count_does_not_depend_on_image_scale() -> None:
    """Pins the "subsample to a fixed grid, thresholds as fractions" design."""
    small = geometry.count_lobes(ring(8, 42.0, 22, size=200))
    large = geometry.count_lobes(ring(8, 168.0, 88, size=800))

    assert small["lobes"] == large["lobes"]
    assert large["inradius"] == pytest.approx(small["inradius"] * 4, rel=0.25)


def test_a_clipped_silhouette_is_flagged() -> None:
    assert geometry.count_lobes(disc(0, 100, 40))["clipped"] is True


def test_rule_of_thirds_centered_box_is_far_from_a_gridpoint() -> None:
    result = geometry.rule_of_thirds([90.0, 90.0, 110.0, 110.0], (SIZE, SIZE))

    assert result["center_offset"] < 0.02
    assert result["thirds_offset"] > 0.1


def test_rule_of_thirds_box_on_a_gridpoint_scores_near_zero() -> None:
    gx = SIZE / 3
    result = geometry.rule_of_thirds([gx - 5, gx - 5, gx + 5, gx + 5], (SIZE, SIZE))

    assert result["thirds_offset"] < 0.02


def test_rule_of_thirds_prefers_the_gridpoint_over_the_center() -> None:
    """Negative control: a well-composed subject scores lower than a dead-centered one."""
    gx = SIZE / 3
    centered = geometry.rule_of_thirds([90.0, 90.0, 110.0, 110.0], (SIZE, SIZE))
    on_gridpoint = geometry.rule_of_thirds([gx - 5, gx - 5, gx + 5, gx + 5], (SIZE, SIZE))

    assert on_gridpoint["thirds_offset"] < centered["thirds_offset"]
