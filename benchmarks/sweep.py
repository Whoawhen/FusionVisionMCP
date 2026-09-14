#  sweep.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Phase 3: sweep the thresholds the shipped detector already exposes.

Every documented counting limit was measured at Grounding DINO's default
`threshold=0.25` / `text_threshold=0.25`, and neither has ever been swept. This runs the
whole fixture set across a grid, loading the model once, and reports the full curve --
not just the best point, since a configuration that wins on positives by breaking a
negative control is disqualified regardless of its score.

Run with `python -m benchmarks.sweep` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmarks.fixtures import all_fixtures
from benchmarks.harness import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--text-thresholds", type=float, nargs="+", default=[0.25])
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/sweep.json"))
    parser.add_argument("--no-equivalents", action="store_true")
    args = parser.parse_args(argv)

    from fusion_vision_mcp.grounding_dino import GroundingDino

    model = GroundingDino()  # loaded once for the whole sweep
    fixtures = all_fixtures()

    class Pinned:
        def __init__(self, threshold: float, text_threshold: float) -> None:
            self.threshold, self.text_threshold = threshold, text_threshold

        def detect_objects(self, images: list[Any], object_name: str) -> list[dict[str, Any]]:
            return model.detect_objects(
                images, object_name, threshold=self.threshold, text_threshold=self.text_threshold
            )

    reports = []
    header = f"{'thresh':>7}{'text':>7}{'pos':>8}{'neg':>8}{'MAE':>8}  negative-control breaks"
    print(header)
    print("-" * len(header))

    for text_threshold in args.text_thresholds:
        for threshold in args.thresholds:
            label = f"t={threshold} tt={text_threshold}"
            report = run(
                Pinned(threshold, text_threshold),
                fixtures,
                label=label,
                include_equivalents=not args.no_equivalents,
            )
            reports.append(report)
            s = report["summary"]
            breaks = [
                f"{r['fixture']}={r['predicted']}"
                for r in report["results"]
                if r["kind"] == "negative" and r["error"] not in (None, 0)
            ]
            print(
                f"{threshold:>7}{text_threshold:>7}"
                f"{s['positives_exact']:>4}/{s['positives_total']:<3}"
                f"{s['negatives_held']:>4}/{s['negatives_total']:<3}"
                f"{s['mean_abs_error']:>8}  {', '.join(breaks) if breaks else '-'}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    clean = [r for r in reports if r["summary"]["negatives_held"] == r["summary"]["negatives_total"]]
    if clean:
        best = max(clean, key=lambda r: (r["summary"]["positives_exact"], -r["summary"]["mean_abs_error"]))
        print(
            f"\nbest configuration holding every negative control: {best['label']} "
            f"({best['summary']['positives_exact']}/{best['summary']['positives_total']} positives exact, "
            f"MAE {best['summary']['mean_abs_error']})"
        )
    else:
        print("\nno configuration held every negative control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
