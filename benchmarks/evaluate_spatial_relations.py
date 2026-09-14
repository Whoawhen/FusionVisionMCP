#  evaluate_spatial_relations.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Evaluate spatial_relations agreement against Open Images visual relationship ground truth.

Prior to Sprint 16, `spatial_relations` and the underlying geometric measurements in
`geometry.py` were validated against only two synthetic fixtures (`tests/spatial_touch_separate.png`
and `tests/spatial_containment.png`).

This benchmark script processes all 36 curated photographic scenes from Open Images V6 Visual
Relationship Detection (VRD) across three relationship analogues:
- Containment (`contain`, `inside_of`)
- Contact / Touching (`on`, `holds`)
- Separation (`at` with spatial clearance)

For each fixture, it executes the real detection and segmentation pipeline (Grounding DINO + SAM2 +
`geometry.relation`), recording:
- Whether both objects were successfully located by the detector
- The measured geometric relation state (`separate`, `touching`, `overlapping`)
- The numeric metrics (`gap`, `a_inside_b`, `b_inside_a`, `embed_depth`)
- Agreement with Open Images ground truth

Per Sprint 16's instructions, this script reports empirical measurements and agreement rates
arithmetically into a CSV without altering thresholds or inventing subjective adjustments.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from open_images_relation_fixtures import relation_fixtures

from fusion_vision_mcp import geometry, query_policy
from fusion_vision_mcp.grounding_dino import GroundingDino
from fusion_vision_mcp.sam2 import Sam2


def check_agreement(
    analogue: str,
    state: str,
    gap: float,
    a_inside_b: float,
    b_inside_a: float,
) -> bool:
    """Determine whether geometric measurement agrees with the official relationship label.

    - Containment: agrees if overlapping with substantial area containment (a_inside_b or b_inside_a > 0.20)
    - Contact: agrees if touching (gap <= 2.0 px) or overlapping (physical contact)
    - Separation: agrees if separate (gap > 2.0 px)
    """
    if state == "empty":
        return False
    if analogue == "containment":
        return state == "overlapping" and (a_inside_b > 0.20 or b_inside_a > 0.20)
    if analogue == "contact":
        return state in ("touching", "overlapping") and gap <= 2.0
    if analogue == "separate":
        return state == "separate" and gap > 2.0
    return False


def run_evaluation() -> None:
    """Run spatial relationship evaluation on all Open Images fixtures and save results."""
    fixtures = relation_fixtures()
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "open_images_spatial_agreement.csv"

    print(f"Loading Grounding DINO and SAM2 models for {len(fixtures)} fixtures...")
    counter = GroundingDino()
    segmenter = Sam2()

    records: list[dict[str, Any]] = []
    print("\nRunning spatial relations evaluation...")
    print("-" * 90)
    print(f"{'Fixture':36s} {'Relationship':14s} {'Detected':10s} {'State':12s} {'Gap':8s} {'Agree':6s}")
    print("-" * 90)

    for fx in fixtures:
        img = fx.image
        w, h = img.width, img.height

        # Step 1: Detect subject and object using Grounding DINO
        located: list[dict[str, Any]] = []
        for obj_name in (fx.subject, fx.object):
            det = counter.detect_objects([img], obj_name)[0]
            det = query_policy.suppress_generic_full_frame(obj_name, det, w, h)
            if det["bboxes"]:
                best_idx = max(range(len(det["bboxes"])), key=lambda i: det["scores"][i])
                box = det["bboxes"][best_idx]
                located.append({"name": obj_name, "box": [int(v) for v in box], "score": det["scores"][best_idx]})

        detected_both = len(located) == 2

        # Step 2: Segment and measure relation
        if detected_both:
            boxes = [loc["box"] for loc in located]
            masks = segmenter.segment(img, boxes)
            rel = geometry.relation(masks[0], masks[1])
            meas_state = str(rel["state"])
            meas_gap = float(rel["gap"])
            meas_a_in_b = float(rel["a_inside_b"])
            meas_b_in_a = float(rel["b_inside_a"])
            meas_depth = float(rel["embed_depth"])
        else:
            # Fallback to ground-truth boxes to evaluate segmentation & geometry in isolation
            gt_b_sub = [
                int(fx.box_subject[1] * w),
                int(fx.box_subject[0] * h),
                int(fx.box_subject[3] * w),
                int(fx.box_subject[2] * h),
            ]
            gt_b_obj = [
                int(fx.box_object[1] * w),
                int(fx.box_object[0] * h),
                int(fx.box_object[3] * w),
                int(fx.box_object[2] * h),
            ]
            masks = segmenter.segment(img, [gt_b_sub, gt_b_obj])
            rel = geometry.relation(masks[0], masks[1])
            meas_state = str(rel["state"])
            meas_gap = float(rel["gap"])
            meas_a_in_b = float(rel["a_inside_b"])
            meas_b_in_a = float(rel["b_inside_a"])
            meas_depth = float(rel["embed_depth"])

        agree = check_agreement(
            fx.analogue_geometry_state,
            meas_state,
            meas_gap,
            meas_a_in_b,
            meas_b_in_a,
        )

        record = {
            "fixture_name": fx.name,
            "image_id": fx.image_id,
            "subject": fx.subject,
            "object": fx.object,
            "official_relationship": fx.relationship,
            "analogue_geometry_state": fx.analogue_geometry_state,
            "detected_both": detected_both,
            "measured_state": meas_state,
            "gap_pixels": "NaN" if math.isnan(meas_gap) else round(meas_gap, 2),
            "a_inside_b": "NaN" if math.isnan(meas_a_in_b) else round(meas_a_in_b, 4),
            "b_inside_a": "NaN" if math.isnan(meas_b_in_a) else round(meas_b_in_a, 4),
            "embed_depth": "NaN" if math.isnan(meas_depth) else round(meas_depth, 2),
            "agreement": agree,
            "license": fx.license,
        }
        records.append(record)

        det_str = "both" if detected_both else "gt_box"
        agree_str = "YES" if agree else "NO"
        gap_str = "NaN" if math.isnan(meas_gap) else f"{meas_gap:.1f}"
        print(f"{fx.name:36s} {fx.relationship:14s} {det_str:10s} {meas_state:12s} {gap_str:8s} {agree_str:6s}")

    # Write CSV
    fieldnames = list(records[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("-" * 90)
    print(f"Results written to: {out_csv}")

    # Summary statistics
    total = len(records)
    det_count = sum(1 for r in records if r["detected_both"])
    agree_count = sum(1 for r in records if r["agreement"])

    print(f"\nOverall Summary ({total} fixtures):")
    print(f"  Detector located both objects: {det_count}/{total} ({det_count * 100.0 / total:.1f}%)")
    print(f"  Spatial relationship agreement: {agree_count}/{total} ({agree_count * 100.0 / total:.1f}%)")

    for analogue in ("containment", "contact", "separate"):
        sub = [r for r in records if r["analogue_geometry_state"] == analogue]
        sub_agree = sum(1 for r in sub if r["agreement"])
        print(f"  - {analogue:12s}: {sub_agree}/{len(sub)} ({sub_agree * 100.0 / len(sub):.1f}%) agreement")


if __name__ == "__main__":
    run_evaluation()
