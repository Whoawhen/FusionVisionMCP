#  inspection_coco_negative_control.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Negative control for `inspection._check_anatomy`, against real photographs of people.

Sprint 17 of the v0.8.1 plan ("The missing negative control"). The third vision pass found that
`structured_analysis` reported `corroborated: true` anomalies ("10 arms for 2 people") on
`tests/defect_test6.jpg` that do not correspond to anything real in the image -- the anomaly
heuristic in `_check_anatomy` tallies Grounding DINO's part detections globally across the frame,
with no association to a specific person box and no dedup between overlapping part detections
(a bent arm can yield upper-arm + forearm + whole-arm boxes). `tests/test_inspection.py` covers
only stubbed integers on a blank 10x10 image, and `tests/test_server.py`'s only real-image
`structured_analysis` test uses the flower fixture, which has zero people and returns before the
anatomy check ever runs. Neither test could ever have caught this.

This script is `neg_spotted_ball` applied to inspection: the cheapest possible disproof. A real
photograph of one or more ordinary people, with bent arms and normal occlusion, must produce zero
anomalies from `_check_anatomy` -- the heuristic is self-referential (it compares the detector's own
`person` count against the detector's own `arm`/`leg`/`head`/`finger` counts), so the assertion holds
regardless of whether an exact independent ground-truth person count is available for a given image.

Fixture selection and an honest gap in this repo's own ground truth: `benchmarks/coco_annotations.json`
curates 35 COCO val2017 images, and 24 of them carry "person" somewhere in `all_categories`. But
COCO's own box-level ground truth (`boxes`, and the derived `truth`/`count` field) is populated only
for each image's `primary_category`. Only 2 of the 24 person-bearing images have `primary_category
== "person"` with an exact COCO ground-truth person count in this repo's curated data:
`coco_person_3_000000461751` (3 people) and `coco_person_10_000000247917` (10 people). For the other
22, a person is present in the scene but this repo holds no exact box-level truth for how many --
recording a made-up number for those would be fabricating evidence, which is exactly the failure this
sprint exists to catch elsewhere. Those 22 are recorded as `"n/a (secondary category, no box-level
truth)"` rather than guessed, and all 24 are used for the negative control regardless, since the
control does not depend on knowing the true headcount.

Run directly: `uv run python benchmarks/inspection_coco_negative_control.py`. This calls live
Grounding DINO inference for each of the six part queries (`person`, `arm`, `leg`, `hand`, `head`,
`finger`) on each of the 24 fixtures -- up to 144 detection calls -- so it takes a while on CPU.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from PIL import Image

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from coco_fixtures import CocoFixture, coco_fixtures

from fusion_vision_mcp.grounding_dino import GroundingDino
from fusion_vision_mcp.inspection import analyze_inspection

PARTS = ("person", "arm", "leg", "hand", "head", "finger")


class _RecordingDetector:
    """Wraps a real `GroundingDino` instance, recording each call's count by object name.

    `_check_anatomy` calls `app.counter.detect_objects(...)` once per part and never returns the
    raw counts -- only the derived observations/anomalies. Wrapping the detector lets this script
    read the exact same counts `_check_anatomy` saw, without invoking the detector a second time
    (which would double the already-expensive live inference this script performs).
    """

    def __init__(self, inner: GroundingDino) -> None:
        self._inner = inner
        self.last_counts: dict[str, int] = {}

    def detect_objects(self, images: list[Image.Image], object_name: str, **kwargs: Any) -> list[dict[str, Any]]:
        results = self._inner.detect_objects(images, object_name, **kwargs)
        self.last_counts[object_name] = int(results[0].get("count", 0))
        return results


class _MinimalApp:
    """The minimal `AppContext` stand-in `analyze_inspection` actually needs: a `.counter`."""

    def __init__(self, counter: _RecordingDetector) -> None:
        self.counter = counter


def _gt_person_count(fx: CocoFixture) -> str:
    """COCO's own exact ground truth, or an honest "n/a" -- never a fabricated estimate.

    `fx.truth` is only ever a ground-truth count for `fx.primary_category`. For the 22 fixtures
    where a person is present but is not the primary category, this repo has no box-level truth
    for the person count, and recording one would be exactly the kind of invented evidence this
    sprint exists to stop.
    """
    if fx.primary_category == "person":
        return str(fx.truth)
    return "n/a (secondary category, no box-level truth)"


def _anomaly_type_breakdown(descriptions: list[str]) -> dict[str, int]:
    """Classify anomaly descriptions by the part they concern, for the summary printout."""
    breakdown = {"arm": 0, "leg": 0, "head": 0, "finger": 0}
    for desc in descriptions:
        lowered = desc.lower()
        if "arms" in lowered:
            breakdown["arm"] += 1
        elif "legs" in lowered:
            breakdown["leg"] += 1
        elif "heads" in lowered:
            breakdown["head"] += 1
        elif "fingers" in lowered:
            breakdown["finger"] += 1
    return breakdown


def run_evaluation() -> None:
    """Run `_check_anatomy` (via `analyze_inspection`) on every person-bearing COCO fixture."""
    fixtures = [fx for fx in coco_fixtures() if "person" in fx.all_categories]
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "inspection_coco_negative_control.csv"

    print(f"Loading Grounding DINO for {len(fixtures)} person-bearing COCO fixtures...")
    real_detector = GroundingDino()

    records: list[dict[str, Any]] = []
    all_descriptions: list[str] = []
    images_with_anomalies = 0

    print("\nRunning structured anatomical inspection (negative control)...")
    print("-" * 100)
    print(f"{'Fixture':38s} {'GT person':26s} {'det.person':10s} {'anomalies':10s}")
    print("-" * 100)

    for fx in fixtures:
        recording_detector = _RecordingDetector(real_detector)
        app = _MinimalApp(recording_detector)

        _observations, anomalies = analyze_inspection(app, fx.image, [])
        counts = recording_detector.last_counts

        descriptions = [a.description for a in anomalies]
        all_descriptions.extend(descriptions)
        if anomalies:
            images_with_anomalies += 1

        gt_person = _gt_person_count(fx)
        record = {
            "fixture_name": fx.name,
            "image_id": fx.image_id,
            "coco_gt_person_count": gt_person,
            "detected_person": counts.get("person", 0),
            "detected_arm": counts.get("arm", 0),
            "detected_leg": counts.get("leg", 0),
            "detected_hand": counts.get("hand", 0),
            "detected_head": counts.get("head", 0),
            "detected_finger": counts.get("finger", 0),
            "anomaly_count": len(anomalies),
            "anomaly_descriptions": "; ".join(descriptions),
            "license": fx.license_url,
        }
        records.append(record)

        print(f"{fx.name:38s} {gt_person:26s} {counts.get('person', 0):<10d} {len(anomalies):<10d}")

    fieldnames = list(records[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("-" * 100)
    print(f"Results written to: {out_csv}")

    total = len(records)
    breakdown = _anomaly_type_breakdown(all_descriptions)
    print(f"\nOverall Summary ({total} person-bearing COCO fixtures):")
    print(f"  Images raising at least one anomaly: {images_with_anomalies}/{total}")
    print(f"  Total anomalies raised: {len(all_descriptions)}")
    print(
        f"  Breakdown by type: arm={breakdown['arm']}, leg={breakdown['leg']}, head={breakdown['head']}, finger={breakdown['finger']}"
    )
    print(
        "  Fixtures with exact COCO ground-truth person count: "
        "coco_person_3_000000461751 (3), coco_person_10_000000247917 (10)"
    )
    print(
        "  The other 22 person-bearing fixtures have no exact box-level GT person count in this "
        "repo's curated data (person is a secondary category); recorded as 'n/a', not estimated."
    )


if __name__ == "__main__":
    run_evaluation()
