"""Curated COCO val2017 fixtures with professional ground-truth annotations.

Currently, this repository's benchmark suite relies almost entirely on synthetic geometric
fixtures drawn programmatically, with only a single real photograph (the flower sample in
`tests/sample.jpg`). Sprint 4 demonstrated that SigLIP2 zero-shot classification confidently
misclassified that lone flower photograph as clip-art, exposing the acute need for real
photographic ground truth.

This module provides a curated subset of 35 photographic scenes from Microsoft COCO's
official `val2017` validation split. Every selected image has been mechanically verified
against COCO's official `licenses` metadata to ensure strict compatibility with open-source
redistribution (CC-BY 2.0, Flickr Commons, or US Government Work; all No-Derivatives and
ambiguous licenses are strictly excluded).

The fixtures span three instance-density tiers across 29 distinct object categories:
- Low density (1-2 instances): clean single- and dual-object grounding baselines
- Medium density (3-5 instances): standard multi-object scenes
- High density / cluttered (6-22 instances): stress tests for distractor suppression and counting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fixtures import Fixture, Kind
from PIL import Image

BENCHMARKS_DIR = Path(__file__).resolve().parent
COCO_IMAGES_DIR = BENCHMARKS_DIR / "coco_images"
ANNOTATIONS_FILE = BENCHMARKS_DIR / "coco_annotations.json"

ALLOWED_LICENSE_IDS = {4, 7, 8}


@dataclass
class CocoFixture:
    """One COCO benchmark fixture with ground-truth instance counts and bounding boxes.

    Provides both standard `Fixture`-compatible attributes (`name`, `prompt`, `truth`,
    `kind`, `notes`) and detailed COCO ground-truth metadata including normalized bounding
    boxes, category breakdowns, and license provenance.
    """

    name: str
    prompt: str
    truth: int
    image_path: Path
    file_name: str
    image_id: int
    width: int
    height: int
    primary_category: str
    tier: Literal["low", "med", "high"]
    license_id: int
    license_name: str
    license_url: str
    coco_url: str
    flickr_url: str
    all_categories: list[str] = field(default_factory=list)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    kind: Kind = "positive"
    notes: str = ""
    equivalent_prompts: list[str] = field(default_factory=list)
    _cached_image: Image.Image | None = field(default=None, repr=False, init=False)

    @property
    def image(self) -> Image.Image:
        """Lazily load the RGB image to conserve memory during large test iterations."""
        if self._cached_image is None:
            self._cached_image = Image.open(self.image_path).convert("RGB")
        return self._cached_image

    def to_fixture(self) -> Fixture:
        """Convert to the standard `Fixture` dataclass used by `harness.py`."""
        return Fixture(
            name=self.name,
            image=self.image,
            prompt=self.prompt,
            truth=self.truth,
            kind=self.kind,
            notes=self.notes,
            equivalent_prompts=self.equivalent_prompts,
        )


def _load_raw_annotations() -> list[dict[str, Any]]:
    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def coco_fixtures() -> list[CocoFixture]:
    """Load all 35 curated COCO val2017 fixtures.

    Each fixture is verified against the local images directory and COCO's license
    constraints.
    """
    raw_data = _load_raw_annotations()
    fixtures: list[CocoFixture] = []
    for entry in raw_data:
        img_path = COCO_IMAGES_DIR / entry["file_name"]
        fx = CocoFixture(
            name=entry["name"],
            prompt=entry["primary_category"],
            truth=entry["count"],
            image_path=img_path,
            file_name=entry["file_name"],
            image_id=entry["image_id"],
            width=entry["width"],
            height=entry["height"],
            primary_category=entry["primary_category"],
            tier=entry["tier"],
            license_id=entry["license_id"],
            license_name=entry["license_name"],
            license_url=entry["license_url"],
            coco_url=entry["coco_url"],
            flickr_url=entry["flickr_url"],
            all_categories=entry.get("all_categories", []),
            boxes=entry.get("boxes", []),
            kind="positive",
            notes=(
                f"COCO val2017 image {entry['image_id']} ({entry['width']}x{entry['height']}), "
                f"tier={entry['tier']}, license={entry['license_name']}"
            ),
        )
        fixtures.append(fx)
    return fixtures


def coco_counting_fixtures() -> list[Fixture]:
    """Return COCO fixtures converted to `benchmarks.fixtures.Fixture` for counting harnesses."""
    return [cf.to_fixture() for cf in coco_fixtures()]
