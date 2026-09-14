#  test_open_images_fixtures.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for Open Images visual relationship benchmark fixtures and agreement logic."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from evaluate_spatial_relations import check_agreement
from open_images_relation_fixtures import ALLOWED_LICENSE, SpatialRelationFixture, relation_fixtures


def test_open_images_fixtures_count_and_types() -> None:
    """Ensure exactly 36 fixtures are loaded and match expected types."""
    fixtures = relation_fixtures()
    assert len(fixtures) == 36
    for fx in fixtures:
        assert isinstance(fx, SpatialRelationFixture)
        assert fx.name.startswith("oi_")
        assert len(fx.subject) > 0
        assert len(fx.object) > 0
        assert fx.analogue_geometry_state in ("containment", "contact", "separate")


def test_open_images_fixtures_images_exist_and_readable() -> None:
    """Ensure all fixture image files exist on disk and can be read by PIL."""
    fixtures = relation_fixtures()
    for fx in fixtures:
        assert fx.image_path.exists(), f"Missing image file: {fx.image_path}"
        img = fx.image
        assert isinstance(img, Image.Image)
        assert img.width > 0 and img.height > 0


def test_open_images_fixtures_license_strictly_cc_by() -> None:
    """Enforce Sprint 16 license invariant: 100% verified Creative Commons Attribution 2.0."""
    fixtures = relation_fixtures()
    for fx in fixtures:
        assert fx.license == ALLOWED_LICENSE, f"Fixture {fx.name} has non-CC-BY license: {fx.license}"


def test_open_images_fixtures_distribution() -> None:
    """Ensure balanced distribution: 12 containment, 12 contact, 12 separation."""
    fixtures = relation_fixtures()
    containment = [fx for fx in fixtures if fx.analogue_geometry_state == "containment"]
    contact = [fx for fx in fixtures if fx.analogue_geometry_state == "contact"]
    separate = [fx for fx in fixtures if fx.analogue_geometry_state == "separate"]

    assert len(containment) == 12
    assert len(contact) == 12
    assert len(separate) == 12


def test_open_images_fixtures_boxes_valid_normalized_coords() -> None:
    """Ensure ground-truth subject and object boxes are valid normalized coordinates."""
    fixtures = relation_fixtures()
    for fx in fixtures:
        for box in (fx.box_subject, fx.box_object):
            assert len(box) == 4
            ymin, xmin, ymax, xmax = box
            assert 0.0 <= ymin <= 1.0
            assert 0.0 <= xmin <= 1.0
            assert 0.0 <= ymax <= 1.0
            assert 0.0 <= xmax <= 1.0
            assert ymin <= ymax
            assert xmin <= xmax


def test_check_agreement_containment_logic() -> None:
    """Verify agreement function correctly evaluates containment relationships."""
    # Deep containment -> agrees
    assert check_agreement("containment", "overlapping", gap=0.0, a_inside_b=0.85, b_inside_a=0.15)
    # Reverse containment -> agrees
    assert check_agreement("containment", "overlapping", gap=0.0, a_inside_b=0.10, b_inside_a=0.60)
    # Shallow overlap below 0.20 -> does not agree
    assert not check_agreement("containment", "overlapping", gap=0.0, a_inside_b=0.05, b_inside_a=0.08)
    # Separate -> does not agree
    assert not check_agreement("containment", "separate", gap=15.0, a_inside_b=0.0, b_inside_a=0.0)


def test_check_agreement_contact_and_separation_logic() -> None:
    """Verify agreement function correctly evaluates contact and separation relationships."""
    # Touching within tolerance -> contact agrees
    assert check_agreement("contact", "touching", gap=1.5, a_inside_b=0.0, b_inside_a=0.0)
    # Overlapping -> contact agrees
    assert check_agreement("contact", "overlapping", gap=0.0, a_inside_b=0.1, b_inside_a=0.1)
    # Separate with large gap -> contact disagrees
    assert not check_agreement("contact", "separate", gap=20.0, a_inside_b=0.0, b_inside_a=0.0)

    # Separate with gap > 2.0 -> separation agrees
    assert check_agreement("separate", "separate", gap=25.0, a_inside_b=0.0, b_inside_a=0.0)
    # Overlapping -> separation disagrees
    assert not check_agreement("separate", "overlapping", gap=0.0, a_inside_b=0.4, b_inside_a=0.2)
    # Touching within tolerance -> separation disagrees
    assert not check_agreement("separate", "touching", gap=1.0, a_inside_b=0.0, b_inside_a=0.0)
