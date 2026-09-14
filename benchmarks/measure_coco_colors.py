"""Compute the unique-RGB-colors-per-1000-pixels statistic across COCO val2017 fixtures.

Sprint 4 demonstrated that SigLIP2 zero-shot domain classification misclassified the
repository's lone photograph (`tests/sample.jpg`) as clip-art with high confidence (0.60),
and established that margin-based thresholds cannot catch confident classification errors.
To reframe Sprint 5's automatic clip-art routing, this project is evaluating whether a cheap,
model-free statistic -- unique RGB colors per 1,000 pixels -- can reliably distinguish real
photographs from synthetic/flat-art illustrations.

This script processes all 35 curated COCO `val2017` photographic scenes, computing:
  unique_colors = number of distinct (R, G, B) pixel values in the image
  total_pixels = width * height
  unique_colors_per_1000_pixels = (unique_colors * 1000.0) / total_pixels

Per Sprint 15's specification, this script performs pure measurement and reports empirical
distributions without asserting an arbitrary cutoff or interpreting the results, leaving
decision-making and thresholding for senior review.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(BENCHMARKS_DIR))

from coco_fixtures import coco_fixtures


def count_unique_rgb_colors(image: Image.Image) -> tuple[int, int, float]:
    """Calculate unique RGB colors, total pixels, and colors per 1,000 pixels.

    Packs 24-bit RGB values into 32-bit integers for vectorised unique extraction.
    """
    rgb = np.asarray(image.convert("RGB"))
    total_pixels = image.width * image.height
    flat_rgb = rgb.reshape(-1, 3)
    # Pack R, G, B into a single uint32 per pixel
    packed = (
        (flat_rgb[:, 0].astype(np.uint32) << 16)
        | (flat_rgb[:, 1].astype(np.uint32) << 8)
        | flat_rgb[:, 2].astype(np.uint32)
    )
    unique_count = len(np.unique(packed))
    colors_per_1k = (unique_count * 1000.0) / total_pixels if total_pixels > 0 else 0.0
    return unique_count, total_pixels, colors_per_1k


def run_measurement() -> None:
    """Measure the color statistic across all COCO fixtures and save to CSV."""
    fixtures = coco_fixtures()
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "coco_unique_colors.csv"

    records = []
    print(f"Measuring unique RGB colors across {len(fixtures)} COCO val2017 fixtures...")
    print("-" * 80)
    print(f"{'Fixture':36s} {'Dimensions':12s} {'Total Pix':10s} {'Unique':8s} {'Per 1k':8s} Tier")
    print("-" * 80)

    for fx in fixtures:
        unique_colors, total_pixels, per_1k = count_unique_rgb_colors(fx.image)
        records.append(
            {
                "fixture_name": fx.name,
                "image_id": fx.image_id,
                "file_name": fx.file_name,
                "width": fx.width,
                "height": fx.height,
                "total_pixels": total_pixels,
                "unique_colors": unique_colors,
                "unique_colors_per_1000_pixels": round(per_1k, 4),
                "license_id": fx.license_id,
                "license_name": fx.license_name,
                "primary_category": fx.primary_category,
                "count": fx.truth,
                "tier": fx.tier,
            }
        )
        dims = f"{fx.width}x{fx.height}"
        print(f"{fx.name:36s} {dims:12s} {total_pixels:<10d} {unique_colors:<8d} {per_1k:<8.2f} {fx.tier}")

    # Write CSV
    fieldnames = [
        "fixture_name",
        "image_id",
        "file_name",
        "width",
        "height",
        "total_pixels",
        "unique_colors",
        "unique_colors_per_1000_pixels",
        "license_id",
        "license_name",
        "primary_category",
        "count",
        "tier",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("-" * 80)
    print(f"Results written to: {out_csv}")

    # Summary statistics
    per_1k_values = [r["unique_colors_per_1000_pixels"] for r in records]
    print("\nSummary Statistics for unique_colors_per_1000_pixels across COCO photographs:")
    print(f"  Count:  {len(per_1k_values)}")
    print(f"  Min:    {min(per_1k_values):.2f}")
    print(f"  Max:    {max(per_1k_values):.2f}")
    print(f"  Mean:   {statistics.mean(per_1k_values):.2f}")
    print(f"  Median: {statistics.median(per_1k_values):.2f}")
    print(f"  StdDev: {statistics.stdev(per_1k_values):.2f}")


if __name__ == "__main__":
    run_measurement()
