#  tune_domain_threshold.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Empirically evaluate the domain router against `domain_fixtures.py`'s honest ground truth.

The margin threshold only gates the `ambiguous` flag -- it cannot change which label wins
(`classify_domain`'s top-1 pick is threshold-independent, since sigmoid/softmax is a
monotonic transform of the same logits at every threshold). So this script measures two
different things and does not conflate them: (1) raw top-1 accuracy per fixture, which no
threshold can improve, and (2) whether the `ambiguous` flag actually fires on the cases that
need it, which is what the threshold controls. See `CHANGELOG.md`'s Sprint 4 entry for why
this distinction turned out to matter -- the two most operationally important misses
(photograph and screenshot) are *confidently* wrong, so no threshold value can flag them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from domain_fixtures import domain_fixtures

from fusion_vision_mcp.domain_router import (
    CLIP_ART_DOMAINS,
    DOCUMENT_DOMAINS,
    PAINTING_DOMAINS,
    PHOTO_DOMAINS,
    classify_domain,
    release_siglip2,
    set_ambiguous_margin_threshold,
)


def _predicted_group(top_label: str) -> str:
    for group, labels in (
        ("photograph", PHOTO_DOMAINS),
        ("clip_art", CLIP_ART_DOMAINS),
        ("painting", PAINTING_DOMAINS),
        ("document", DOCUMENT_DOMAINS),
    ):
        if top_label in labels:
            return group
    return "other"


def run_sweep() -> int:
    fixtures = domain_fixtures()
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # Classify once per fixture -- margin doesn't affect the prediction, only the flag.
    classifications = {fx.name: classify_domain(fx.image) for fx in fixtures}

    print("=== Top-1 accuracy (threshold-independent) ===")
    correct = 0
    for fx in fixtures:
        result = classifications[fx.name]
        predicted = _predicted_group(result.domain)
        hit = predicted == fx.true_group
        correct += hit
        print(
            f"  {'Y' if hit else 'N'} {fx.name:22s} true={fx.true_group:10s} "
            f"pred={predicted:10s} ({result.domain}, conf={result.confidence:.3f})"
        )
    print(f"  {correct}/{len(fixtures)} correct\n")

    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domain_router_threshold_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "margin_threshold",
                "fixture",
                "true_group",
                "predicted_group",
                "predicted_label",
                "confidence",
                "second_score",
                "margin",
                "ambiguous",
            ]
        )
        print("=== Ambiguous-flag behaviour by margin threshold ===")
        for threshold in thresholds:
            set_ambiguous_margin_threshold(threshold)
            print(f"--- margin={threshold:.2f} ---")
            for fx in fixtures:
                result = classifications[fx.name]
                sorted_scores = sorted(result.scores.items(), key=lambda x: -x[1])
                second = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
                margin = result.confidence - second
                ambiguous = margin < threshold
                predicted = _predicted_group(result.domain)
                writer.writerow(
                    [
                        threshold,
                        fx.name,
                        fx.true_group,
                        predicted,
                        result.domain,
                        f"{result.confidence:.4f}",
                        f"{second:.4f}",
                        f"{margin:.4f}",
                        ambiguous,
                    ]
                )
                flag = "AMBIGUOUS" if ambiguous else ""
                hit = "correct" if predicted == fx.true_group else "WRONG"
                print(f"    {fx.name:22s} margin={margin:.3f} {hit:9s} {flag}")

    print(f"\nResults written to {out_path}")
    release_siglip2()
    return 0


if __name__ == "__main__":
    run_sweep()
