#  adaptive_threshold_sweep.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Sweep the adaptive threshold danger zone prevalence.

Companion to ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md. Measures how often real
two-instance synthetic scenes produce (s0, s1) score pairs where
choose_threshold would actually drop the second instance.

The danger zone is defined operationally: choose_threshold(scores, base_threshold=0.15)
returns used=True with adapted threshold > s1 (the second-highest score), meaning
the real second instance would be filtered out.

Outputs a CSV with one row per scene:
(scenario, param, s0, s1, in_danger_zone, reason, visible_fraction, top_scores_json)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.fixtures import CANVAS, PETAL_FILL, _canvas, _disc
from fusion_vision_mcp.adaptive_threshold import choose_threshold
from fusion_vision_mcp.grounding_dino import GroundingDino


def get_raw_scores(model: GroundingDino, img: Image.Image, object_name: str) -> list[float]:
    """Run Grounding DINO with threshold=0.0 to get all raw detection scores."""
    prompt = model._as_prompt(object_name)
    with img.convert("RGB") as rgb:
        inputs = model.processor(images=rgb, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.model(**inputs)

        processed = model.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.0,
            text_threshold=0.25,
            target_sizes=[(rgb.height, rgb.width)],
        )[0]

    raw_scores = [float(s) for s in processed["scores"].tolist()]
    return sorted(raw_scores, reverse=True)


def check_danger_zone(scores: list[float]) -> tuple[bool, float, float, str]:
    """
    Operational danger-zone check: call choose_threshold directly and see if it would
    drop the second-highest score.

    Returns: (in_danger_zone, s0, s1, reason)
    """
    if len(scores) < 2:
        return False, 0.0, 0.0, "fewer than 2 scores"

    s0 = scores[0]
    s1 = scores[1]

    # Use the actual function under test, not a theoretical proxy
    result = choose_threshold(scores, base_threshold=0.15)

    # Danger zone = adaptation was used AND the adapted threshold exceeds s1,
    # meaning the real second instance would be filtered out
    in_danger = result.used and result.threshold > s1

    if in_danger:
        reason = (
            f"adapted: threshold={result.threshold:.3f} > s1={s1:.3f} "
            f"(gap={s0 - s1:.3f}, ratio={s0 / s1:.2f}, top={s0:.2f})"
        )
    else:
        if not result.used:
            reason = f"no adaptation: {result.reason}"
        else:
            reason = f"adapted but safe: threshold={result.threshold:.3f} <= s1={s1:.3f}"

    return in_danger, s0, s1, reason


def visible_fraction(cx: float, cy: float, radius: int) -> float:
    """Fraction of a disc's bounding box inside the [0, CANVAS] frame."""
    left = max(0, cx - radius)
    right = min(CANVAS, cx + radius)
    top = max(0, cy - radius)
    bottom = min(CANVAS, cy + radius)
    visible_area = max(0, right - left) * max(0, bottom - top)
    total_area = (2 * radius) * (2 * radius)
    return visible_area / total_area if total_area > 0 else 0.0


def run_occlusion_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Vary occlusion fraction of instance 2 by drawing an overlapping rectangle."""
    print("\n=== Occlusion Sweep ===")
    for occ_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        img, draw = _canvas()
        # Instance 1: fixed, high-confidence
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)
        # Instance 2: partially occluded by a gray rectangle
        cx = CANVAS / 2 + 120
        cy = CANVAS / 2
        _disc(draw, cx, cy, base_radius)
        # Occlusion rectangle covers occ_frac of the disc's diameter from the left
        occ_width = int(2 * base_radius * occ_frac)
        draw.rectangle(
            [cx - base_radius, cy - base_radius, cx - base_radius + occ_width, cy + base_radius],
            fill=(180, 180, 180),
            outline=None,
        )

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, base_radius)

        writer.writerow(
            [
                "occlusion",
                f"{occ_frac:.1f}",
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  occ={occ_frac:.1f}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_blur_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Vary blur radius applied to instance 2."""
    print("\n=== Blur Sweep ===")
    cx = CANVAS / 2 + 120
    cy = CANVAS / 2
    for blur_radius in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        img, draw = _canvas()
        # Instance 1: sharp
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)
        # Instance 2: draw then blur just that region
        # Create a separate image for the blurred disc
        disc_img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        disc_draw = ImageDraw.Draw(disc_img)
        _disc(disc_draw, cx, cy, base_radius, fill=PETAL_FILL)
        if blur_radius > 0:
            disc_img = disc_img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = Image.alpha_composite(img.convert("RGBA"), disc_img).convert("RGB")

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, base_radius)

        writer.writerow(
            [
                "blur",
                str(blur_radius),
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  blur={blur_radius}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_size_ratio_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Vary size ratio of instance 2 (radius from base_radius down to 10)."""
    print("\n=== Size Ratio Sweep ===")
    cx = CANVAS / 2 + 120
    cy = CANVAS / 2
    for r2 in [70, 60, 50, 40, 30, 20, 15, 10]:
        img, draw = _canvas()
        # Instance 1: fixed at base_radius
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)
        # Instance 2: varying radius
        _disc(draw, cx, cy, r2)

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, r2)

        writer.writerow(
            [
                "size_ratio",
                f"r2={r2}",
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  r2={r2}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_contrast_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Vary contrast/desaturation of instance 2 against white canvas."""
    print("\n=== Contrast/Desaturation Sweep ===")
    # Interpolate between PETAL_FILL (235, 120, 150) and white (255, 255, 255)
    cx = CANVAS / 2 + 120
    cy = CANVAS / 2
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        r = int(235 * alpha + 255 * (1 - alpha))
        g = int(120 * alpha + 255 * (1 - alpha))
        b = int(150 * alpha + 255 * (1 - alpha))
        fill = (r, g, b)
        edge = (
            int(170 * alpha + 255 * (1 - alpha)),
            int(70 * alpha + 255 * (1 - alpha)),
            int(100 * alpha + 255 * (1 - alpha)),
        )

        img, draw = _canvas()
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)
        draw.ellipse(
            [cx - base_radius, cy - base_radius, cx + base_radius, cy + base_radius], fill=fill, outline=edge, width=3
        )

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, base_radius)

        writer.writerow(
            [
                "contrast",
                f"alpha={alpha:.1f}",
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  alpha={alpha:.1f}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_edge_crop_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Vary how much instance 2 is cropped at frame edge."""
    print("\n=== Edge Crop Sweep ===")
    # Move instance 2 progressively off the right edge
    # Original offsets + bisection points for the transition boundary (offset 100-120)
    for offset in [0, 20, 40, 60, 80, 100, 105, 110, 115, 120, 140]:
        img, draw = _canvas()
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)
        cx = CANVAS - base_radius + offset  # starts at right edge, moves outward
        cy = CANVAS / 2
        _disc(draw, cx, cy, base_radius)

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, base_radius)

        writer.writerow(
            [
                "edge_crop",
                f"offset={offset}",
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  offset={offset}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_combination_sweep(model: GroundingDino, writer: csv.writer, base_radius: int = 70):
    """Test a few realistic combinations: moderate occlusion + blur, etc."""
    print("\n=== Combination Sweep ===")
    combos = [
        ("occ0.3_blur2", 0.3, 2, None, None, base_radius, CANVAS / 2 + 120, CANVAS / 2),
        ("occ0.4_blur3", 0.4, 3, None, None, base_radius, CANVAS / 2 + 120, CANVAS / 2),
        ("occ0.5_blur4", 0.5, 4, None, None, base_radius, CANVAS / 2 + 120, CANVAS / 2),
        ("occ0.3_size50", 0.3, None, 50, None, 50, CANVAS / 2 + 120, CANVAS / 2),
        ("occ0.4_contrast0.5", 0.4, None, None, 0.5, base_radius, CANVAS / 2 + 120, CANVAS / 2),
    ]

    for name, occ, blur, r2, alpha, radius, cx, cy in combos:
        img, draw = _canvas()
        _disc(draw, CANVAS / 2 - 120, CANVAS / 2, base_radius)

        if blur:
            # Blur approach
            disc_img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
            disc_draw = ImageDraw.Draw(disc_img)
            _disc(disc_draw, cx, cy, radius, fill=PETAL_FILL)
            if blur > 0:
                disc_img = disc_img.filter(ImageFilter.GaussianBlur(blur))
            if occ > 0:
                # Draw occlusion on top
                occ_draw = ImageDraw.Draw(disc_img)
                occ_width = int(2 * radius * occ)
                occ_draw.rectangle(
                    [cx - radius, cy - radius, cx - radius + occ_width, cy + radius],
                    fill=(180, 180, 180, 255),
                    outline=None,
                )
            img = Image.alpha_composite(img.convert("RGBA"), disc_img).convert("RGB")
        elif occ > 0:
            # Simple occlusion
            _disc(draw, cx, cy, radius)
            occ_width = int(2 * radius * occ)
            draw.rectangle(
                [cx - radius, cy - radius, cx - radius + occ_width, cy + radius], fill=(180, 180, 180), outline=None
            )
        elif alpha is not None:
            # Contrast
            r = int(235 * alpha + 255 * (1 - alpha))
            g = int(120 * alpha + 255 * (1 - alpha))
            b = int(150 * alpha + 255 * (1 - alpha))
            fill = (r, g, b)
            edge = (
                int(170 * alpha + 255 * (1 - alpha)),
                int(70 * alpha + 255 * (1 - alpha)),
                int(100 * alpha + 255 * (1 - alpha)),
            )
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill, outline=edge, width=3)
        else:
            _disc(draw, cx, cy, radius)

        scores = get_raw_scores(model, img, "petal")
        in_danger, s0, s1, reason = check_danger_zone(scores)
        vf = visible_fraction(cx, cy, radius)

        writer.writerow(
            [
                "combination",
                name,
                f"{s0:.4f}",
                f"{s1:.4f}",
                "true" if in_danger else "false",
                reason,
                f"{vf:.3f}",
                json.dumps(scores[:20]),
            ]
        )
        print(f"  {name}: s0={s0:.3f} s1={s1:.3f} danger={in_danger} vf={vf:.3f} ({reason})")


def run_f9_distractor_logging(model: GroundingDino, writer: csv.writer):
    """Log F9's actual distractor scores for mitigation calibration."""
    print("\n=== F9 Distractor Fixture ===")
    from benchmarks.fixtures import distractors

    img = distractors()  # 3 target discs + 4 square distractors
    scores = get_raw_scores(model, img, "pink circle")

    # Log all scores for the distractor case
    # F9 has multiple real instances, so visible_fraction is N/A (1.0 for all)
    writer.writerow(
        [
            "f9_distractors",
            "all_scores",
            f"{scores[0]:.4f}",
            f"{scores[1]:.4f}",
            "false",
            "reference",
            "1.000",
            json.dumps(scores[:30]),
        ]
    )
    print(f"  F9 scores (top 10): {scores[:10]}")
    print(f"  F9 scores (10-20): {scores[10:20]}")
    print(f"  F9 scores (20-30): {scores[20:30]}")

    # Also run with the adaptive threshold to see what it does
    result = choose_threshold(scores, base_threshold=0.15)
    print(f"  choose_threshold result: used={result.used}, threshold={result.threshold:.3f}, reason={result.reason}")


def main():
    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adaptive_threshold_danger_zone_sweep.csv"

    print("Loading Grounding DINO model...")
    model = GroundingDino()
    print("Model loaded.")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "param", "s0", "s1", "in_danger_zone", "reason", "visible_fraction", "top_scores_json"]
        )

        # Run all sweeps
        run_occlusion_sweep(model, writer)
        run_blur_sweep(model, writer)
        run_size_ratio_sweep(model, writer)
        run_contrast_sweep(model, writer)
        run_edge_crop_sweep(model, writer)
        run_combination_sweep(model, writer)
        run_f9_distractor_logging(model, writer)

    print(f"\n=== Sweep complete. Results written to {out_path} ===")

    # Quick summary
    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Exclude F9 reference row
    non_f9 = [r for r in rows if r["scenario"] != "f9_distractors"]

    # Separate degenerate (not actually in frame) from valid rows
    valid_rows = [r for r in non_f9 if float(r["visible_fraction"]) > 0]
    degenerate_rows = [r for r in non_f9 if float(r["visible_fraction"]) <= 0]

    danger_count = sum(1 for r in valid_rows if r["in_danger_zone"] == "true")
    total = len(valid_rows)
    degenerate_count = len(degenerate_rows)
    degenerate_danger = sum(1 for r in degenerate_rows if r["in_danger_zone"] == "true")

    print(
        f"\nDanger zone prevalence (valid, visible_fraction > 0): {danger_count}/{total} = {danger_count / total * 100:.1f}%"
    )
    print(f"Degenerate rows (visible_fraction <= 0): {degenerate_count} (of which {degenerate_danger} in danger zone)")

    # Break down by scenario
    from collections import Counter

    by_scenario = Counter(r["scenario"] for r in valid_rows)
    danger_by_scenario = Counter(r["scenario"] for r in valid_rows if r["in_danger_zone"] == "true")

    print("\nBreakdown by scenario (valid rows only):")
    for scenario in sorted(by_scenario.keys()):
        total_s = by_scenario[scenario]
        danger_s = danger_by_scenario.get(scenario, 0)
        print(f"  {scenario}: {danger_s}/{total_s} = {danger_s / total_s * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
