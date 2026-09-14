#  harness.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Model-independent evaluation harness for repeated-instance counting.

Scores anything matching the `InstanceDetector` shape -- `detect_objects(images,
object_name) -> [{"count", "bboxes", ...}]` -- so a new backend can be compared against
the shipped one without touching this file. Emits JSON and a readable table.

Run with `python -m benchmarks.harness --help` from the repository root.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmarks.fixtures import Fixture, all_fixtures


class Detector(Protocol):
    def detect_objects(self, images: list[Any], object_name: str) -> list[dict[str, Any]]: ...


def peak_rss_mb() -> float:
    """Process working set in MB. Windows-specific; NaN elsewhere rather than a lie."""
    if platform.system() != "Windows":
        try:
            import resource

            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
        except Exception:  # noqa: BLE001 - any failure here just means no number
            return float("nan")

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    return counters.PeakWorkingSetSize / 1e6


@dataclass
class Result:
    fixture: str
    kind: str
    prompt: str
    truth: int | None
    predicted: int
    error: int | None
    latency_s: float
    bboxes: list[list[float]]
    points: list[list[float]]
    scores: list[float]
    group_boxes_dropped: int
    notes: str
    failure: str = ""


def run(
    detector: Detector,
    fixtures: Sequence[Fixture],
    *,
    label: str,
    include_equivalents: bool = True,
) -> dict[str, Any]:
    """Score `detector` over `fixtures`, returning a JSON-serialisable report."""
    rss_before = peak_rss_mb()
    results: list[Result] = []

    for fixture in fixtures:
        prompts = [fixture.prompt]
        if include_equivalents:
            prompts += fixture.equivalent_prompts

        for prompt in prompts:
            started = time.time()
            failure = ""
            try:
                out = detector.detect_objects([fixture.image], prompt)[0]
            except Exception as exc:  # noqa: BLE001 - a failed backend is a datum, not a crash
                out, failure = {"count": -1, "bboxes": [], "points": [], "scores": []}, f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - started

            predicted = int(out.get("count", -1))
            results.append(
                Result(
                    fixture=fixture.name,
                    kind=fixture.kind,
                    prompt=prompt,
                    truth=fixture.truth,
                    predicted=predicted,
                    error=None if fixture.truth is None or predicted < 0 else abs(predicted - fixture.truth),
                    latency_s=round(elapsed, 3),
                    bboxes=[[round(v, 1) for v in b] for b in out.get("bboxes", [])],
                    points=[[round(v, 1) for v in p] for p in out.get("points", [])],
                    scores=[round(s, 3) for s in out.get("scores", [])],
                    group_boxes_dropped=int(out.get("group_boxes_dropped", 0)),
                    notes=fixture.notes,
                    failure=failure,
                )
            )

    scored = [r for r in results if r.error is not None]
    negatives = [r for r in scored if r.kind == "negative"]
    positives = [r for r in scored if r.kind == "positive"]

    return {
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "summary": {
            "scored": len(scored),
            "exact": sum(1 for r in scored if r.error == 0),
            "positives_exact": sum(1 for r in positives if r.error == 0),
            "positives_total": len(positives),
            "negatives_held": sum(1 for r in negatives if r.error == 0),
            "negatives_total": len(negatives),
            "mean_abs_error": round(statistics.fmean([r.error for r in scored]), 3) if scored else None,
            "mean_latency_s": round(statistics.fmean([r.latency_s for r in results]), 3) if results else None,
            "rss_delta_mb": round(peak_rss_mb() - rss_before, 1),
            "failures": sum(1 for r in results if r.failure),
        },
        "results": [asdict(r) for r in results],
    }


def render(report: dict[str, Any], *, verbose: bool = False) -> str:
    """A readable table. Negative-control breaks are called out, since they disqualify."""
    lines = [f"=== {report['label']} ===", report["platform"], ""]
    lines.append(f"{'fixture':<26}{'prompt':<20}{'truth':>6}{'pred':>6}{'err':>5}{'sec':>7}  flag")
    lines.append("-" * 82)
    for r in report["results"]:
        truth = "?" if r["truth"] is None else r["truth"]
        err = "-" if r["error"] is None else r["error"]
        flag = ""
        if r["failure"]:
            flag = "FAILED"
        elif r["kind"] == "negative" and r["error"] not in (None, 0):
            flag = "NEG-BREAK"
        elif r["error"] == 0:
            flag = "ok"
        lines.append(
            f"{r['fixture']:<26}{r['prompt'][:19]:<20}{truth:>6}{r['predicted']:>6}{err:>5}{r['latency_s']:>7.2f}  {flag}"
        )
        if verbose and r["scores"]:
            lines.append(f"{'':<26}scores={r['scores'][:10]} dropped={r['group_boxes_dropped']}")

    s = report["summary"]
    lines += [
        "",
        f"positives exact : {s['positives_exact']}/{s['positives_total']}",
        f"negatives held  : {s['negatives_held']}/{s['negatives_total']}",
        f"mean abs error  : {s['mean_abs_error']}",
        f"mean latency    : {s['mean_latency_s']}s      rss delta: {s['rss_delta_mb']}MB",
        f"failures        : {s['failures']}",
    ]
    return "\n".join(lines)


def grounding_dino_backend(threshold: float, text_threshold: float, drop_group_box: bool = True) -> Detector:
    """The shipped detector, with its thresholds pinned for a sweep."""
    from fusion_vision_mcp.grounding_dino import GroundingDino

    model = GroundingDino()

    class Pinned:
        def detect_objects(self, images: list[Any], object_name: str) -> list[dict[str, Any]]:
            return model.detect_objects(
                images,
                object_name,
                threshold=threshold,
                text_threshold=text_threshold,
                drop_group_box=drop_group_box,
            )

    return Pinned()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--no-group-filter", action="store_true")
    parser.add_argument("--only", help="substring filter on fixture name")
    parser.add_argument("--no-equivalents", action="store_true")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    fixtures = [f for f in all_fixtures() if not args.only or args.only in f.name]
    label = f"grounding-dino t={args.threshold} tt={args.text_threshold} group_filter={not args.no_group_filter}"
    report = run(
        grounding_dino_backend(args.threshold, args.text_threshold, not args.no_group_filter),
        fixtures,
        label=label,
        include_equivalents=not args.no_equivalents,
    )
    print(render(report, verbose=args.verbose))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
