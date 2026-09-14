"""Unit tests for curated COCO val2017 benchmark fixtures and license invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from coco_fixtures import ALLOWED_LICENSE_IDS, CocoFixture, coco_counting_fixtures, coco_fixtures
from measure_coco_colors import count_unique_rgb_colors


def test_coco_fixtures_count_and_types() -> None:
    """Ensure exactly 35 fixtures are curated and match expected types."""
    fixtures = coco_fixtures()
    assert len(fixtures) == 35
    for fx in fixtures:
        assert isinstance(fx, CocoFixture)
        assert fx.name.startswith("coco_")
        assert fx.truth > 0
        assert fx.width > 0
        assert fx.height > 0
        assert fx.kind == "positive"
        assert len(fx.primary_category) > 0


def test_coco_fixtures_images_exist_and_readable() -> None:
    """Ensure all fixture image files exist on disk and open with matching dimensions."""
    fixtures = coco_fixtures()
    for fx in fixtures:
        assert fx.image_path.exists(), f"Missing image file: {fx.image_path}"
        img = fx.image
        assert isinstance(img, Image.Image)
        assert img.size == (fx.width, fx.height)


def test_coco_fixtures_licenses_strictly_permitted() -> None:
    """Enforce Sprint 15 license invariant: strictly open-source redistribution licenses.

    Must be License 4 (CC-BY 2.0), License 7 (Flickr Commons / No Known Restrictions),
    or License 8 (US Government Work). All No-Derivatives (ND) or ambiguous licenses
    must be strictly excluded.
    """
    fixtures = coco_fixtures()
    for fx in fixtures:
        assert fx.license_id in ALLOWED_LICENSE_IDS, (
            f"Fixture {fx.name} has disallowed license ID {fx.license_id} ({fx.license_name})"
        )


def test_coco_fixtures_tier_distribution() -> None:
    """Ensure balanced distribution across instance-density tiers."""
    fixtures = coco_fixtures()
    tiers = {fx.tier for fx in fixtures}
    assert tiers == {"low", "med", "high"}

    low = [fx for fx in fixtures if fx.tier == "low"]
    med = [fx for fx in fixtures if fx.tier == "med"]
    high = [fx for fx in fixtures if fx.tier == "high"]

    assert len(low) == 10
    assert len(med) == 13
    assert len(high) == 12

    for fx in low:
        assert 1 <= fx.truth <= 2
    for fx in med:
        assert 3 <= fx.truth <= 5
    for fx in high:
        assert fx.truth >= 6


def test_coco_fixtures_boxes_match_truth_count() -> None:
    """Ensure each fixture's bounding boxes count matches its truth instance count."""
    fixtures = coco_fixtures()
    for fx in fixtures:
        assert len(fx.boxes) == fx.truth
        for box in fx.boxes:
            norm = box["norm_ymin_xmin_ymax_xmax"]
            ymin, xmin, ymax, xmax = norm
            assert 0.0 <= ymin <= 1.0
            assert 0.0 <= xmin <= 1.0
            assert 0.0 <= ymax <= 1.0
            assert 0.0 <= xmax <= 1.0
            assert ymin <= ymax
            assert xmin <= xmax


def test_coco_counting_fixtures_conversion() -> None:
    """Ensure COCO fixtures convert cleanly to `benchmarks.fixtures.Fixture` instances."""
    counting_fxs = coco_counting_fixtures()
    assert len(counting_fxs) == 35
    for fx in counting_fxs:
        assert fx.truth is not None and fx.truth > 0
        assert fx.kind == "positive"
        assert isinstance(fx.image, Image.Image)


def test_unique_rgb_colors_calculation() -> None:
    """Verify unique color metric produces valid positive counts and rates."""
    fixtures = coco_fixtures()
    sample = fixtures[0]
    unique_count, total_pixels, per_1k = count_unique_rgb_colors(sample.image)

    assert unique_count > 0
    assert total_pixels == sample.width * sample.height
    assert per_1k == pytest.approx((unique_count * 1000.0) / total_pixels, rel=1e-4)
