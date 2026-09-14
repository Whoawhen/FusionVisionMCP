#  open_images_relation_fixtures.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Open Images visual relationship detection (VRD) fixtures for spatial relationship benchmarking.

Prior to this module, `spatial_relations` and the underlying geometric measurements in
`geometry.py` were validated against exactly two synthetic fixtures (`tests/spatial_touch_separate.png`
and `tests/spatial_containment.png`).

This module provides a curated subset of 36 photographic scenes from the official Open Images V6
Visual Relationship Detection validation split. Every selected image has been mechanically verified
as Creative Commons Attribution 2.0 (CC-BY 2.0) from official Open Images metadata.

The fixtures cover three core spatial relationship analogues defined in `geometry.py`:
- Containment / Inside (12 fixtures): official relationships `contain` and `inside_of` (e.g. coffee cup
  containing coffee, person inside car, flowerpot containing plant).
- Contact / Touching (12 fixtures): official relationships `on` and `holds` (e.g. strawberry on cake,
  man on horse, musician holding instrument, rider on bicycle).
- Separation (12 fixtures): official relationship `at` with disjoint bounding boxes and positive spatial
  clearance between the two objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

BENCHMARKS_DIR = Path(__file__).resolve().parent
OPEN_IMAGES_DIR = BENCHMARKS_DIR / "open_images"
ANNOTATIONS_FILE = BENCHMARKS_DIR / "open_images_annotations.json"

ALLOWED_LICENSE = "https://creativecommons.org/licenses/by/2.0/"


@dataclass
class SpatialRelationFixture:
    """One spatial relationship benchmark test case with Open Images ground-truth annotations."""

    name: str
    image_path: Path
    image_id: str
    file_name: str
    subject: str
    object: str
    relationship: str
    analogue_geometry_state: Literal["containment", "contact", "separate"]
    box_subject: list[float]  # [ymin, xmin, ymax, xmax] normalized
    box_object: list[float]  # [ymin, xmin, ymax, xmax] normalized
    license: str
    notes: str = ""
    _cached_image: Image.Image | None = field(default=None, repr=False, init=False)

    @property
    def image(self) -> Image.Image:
        """Lazily load the RGB image to conserve memory."""
        if self._cached_image is None:
            self._cached_image = Image.open(self.image_path).convert("RGB")
        return self._cached_image


def _load_raw_annotations() -> list[dict[str, Any]]:
    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def relation_fixtures() -> list[SpatialRelationFixture]:
    """Load all 36 curated Open Images visual relationship fixtures."""
    raw_data = _load_raw_annotations()
    fixtures: list[SpatialRelationFixture] = []
    for entry in raw_data:
        img_path = OPEN_IMAGES_DIR / entry["file_name"]
        fx = SpatialRelationFixture(
            name=entry["name"],
            image_path=img_path,
            image_id=entry["image_id"],
            file_name=entry["file_name"],
            subject=entry["subject"],
            object=entry["object"],
            relationship=entry["relationship"],
            analogue_geometry_state=entry["analogue_geometry_state"],
            box_subject=entry["box_subject_norm_ymin_xmin_ymax_xmax"],
            box_object=entry["box_object_norm_ymin_xmin_ymax_xmax"],
            license=entry["license"],
            notes=f"Open Images VRD {entry['image_id']}: {entry['subject']} --[{entry['relationship']}]--> {entry['object']}",
        )
        fixtures.append(fx)
    return fixtures
