#  interior_structure.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""A REJECTED experiment: counting repeated parts from interior colour.

**This does not ship, and this file is kept only so the negative result is
reproducible rather than merely asserted.** Do not import it from `src/`.

The idea was sound on its face. `geometry.count_lobes` reads the silhouette, which is
empty when parts overlap enough to leave a convex outline -- `tests/sample.jpg` is a
flower whose petals overlap into a 98%-convex blob. But the petals still differ from
one another in colour, so the evidence was thought to survive *inside* the region. This
looks for flat CIELAB patches at several smoothing scales and reports the patch count
that persists across them.

It failed on two independent counts, either of which is disqualifying:

1. **It cannot tell a pattern from a group of parts.** Measured against the fixture
   suite's negative controls, each of which is exactly ONE object:

   | control | reported parts | boundary strength |
   | --- | --- | --- |
   | solid disc | 1 (flat) | 0.46 |
   | gradient-lit disc | 1 (flat) | 0.58 |
   | finely textured blob | 1 (flat) | 0.52 |
   | spotted ball | 1 | 1.58 |
   | **striped ball** | **7** | 2.15 |
   | **four-colour logo** | **4** | 2.03 |

   The stripes and the logo segments are indistinguishable from parts by any measure
   of interior colour alone, which is the fundamental objection to this whole approach.

2. **It does not help the case that motivated it.** The flower reports 1 part, and
   *still* reports 1 with the validity gate disabled entirely -- so this is not a
   threshold rejecting a real signal. Its interior boundary strength is 0.87, barely
   above a plain textured blob's 0.52 and far below the striped ball's 2.15. The petals
   are pastel and low-contrast; after the smoothing needed to survive JPEG noise, there
   is no separable interior structure left to find at any scale.

   The synthetic overlap fixtures are worse still, at 0.23-0.29: their instances are
   drawn in an *identical* colour, so interior colour is definitionally useless there.

Anyone tempted to retry this should note that fixing (1) does not deliver (2). A
perfect pattern-versus-parts discriminator would still find nothing in the flower,
because the contrast is not there. The information that would settle that image is not
in its colours.

Run `python -m benchmarks.interior_structure` from the repository root to reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

Mask = NDArray[np.bool_]

_MIN_MEASURABLE_AREA: Final[int] = 100
_WORK_RESOLUTION: Final[int] = 256
_INTERIOR_SCALES: Final[tuple[float, ...]] = (0.02, 0.03, 0.05, 0.08, 0.12)
_INTERIOR_SMOOTH_FLOOR_PX: Final[float] = 1.5
_INTERIOR_EDGE_PERCENTILE: Final[float] = 70.0
_MIN_PART_AREA_FRACTION: Final[float] = 0.04
_MIN_BOUNDARY_STRENGTH: Final[float] = 1.5
_MAX_PARTS: Final[int] = 24


def srgb_to_lab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an HxWx3 sRGB array in 0-255 to CIELAB.

    Written out rather than pulled from scikit-image, whose dependency tree is not
    worth twenty lines. Lab is used because Euclidean distance in it approximates
    perceived difference, so one edge threshold means roughly the same thing on a pink
    petal and a blue stripe -- which is not true in RGB.
    """
    srgb = np.clip(rgb, 0, 255) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def interior_structure(image_rgb: NDArray[np.uint8], mask: Mask) -> dict[str, Any]:
    """Count flat colour patches inside a region, across several smoothing scales."""
    empty = {
        "parts": 1,
        "scale_support": float("nan"),
        "boundary_strength": float("nan"),
        "coverage": float("nan"),
        "size_uniformity": float("nan"),
        "valid": False,
        "note": "",
    }
    if mask.sum() < _MIN_MEASURABLE_AREA:
        return {**empty, "note": "region too small to measure"}
    if image_rgb.shape[:2] != mask.shape:
        return {**empty, "note": "image and mask shapes differ"}

    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    stride = max(1, int(np.ceil(max(y1 - y0, x1 - x0) / _WORK_RESOLUTION)))
    work_mask = mask[y0:y1:stride, x0:x1:stride]
    work_rgb = image_rgb[y0:y1:stride, x0:x1:stride].astype(np.float64)
    if work_mask.sum() < _MIN_MEASURABLE_AREA:
        return {**empty, "note": "region too small after subsampling"}

    lab = srgb_to_lab(work_rgb)
    short_side = float(min(work_mask.shape))
    total_area = float(work_mask.sum())

    counts: list[int] = []
    strengths: list[float] = []
    best: dict[str, Any] | None = None

    for scale in _INTERIOR_SCALES:
        sigma = max(_INTERIOR_SMOOTH_FLOOR_PX, scale * short_side)
        smoothed = np.stack([gaussian_filter(lab[..., c], sigma=sigma) for c in range(3)], axis=-1)
        gy, gx = np.gradient(smoothed, axis=(0, 1))
        gradient = np.sqrt((gy**2).sum(axis=-1) + (gx**2).sum(axis=-1))
        interior = gradient[work_mask]
        if interior.size == 0:
            continue
        strengths.append(float(interior.mean()))

        flat = work_mask & (gradient <= np.percentile(interior, _INTERIOR_EDGE_PERCENTILE))
        parts, found = label(flat, structure=generate_binary_structure(2, 2))
        if not found:
            counts.append(1)
            continue
        areas = np.bincount(parts.ravel())[1:]
        kept = areas[areas >= _MIN_PART_AREA_FRACTION * total_area]
        count = int(min(max(kept.size, 1), _MAX_PARTS))
        counts.append(count)
        if best is None or count > best["parts"]:
            best = {
                "parts": count,
                "coverage": float(kept.sum()) / total_area if kept.size else 0.0,
                "size_uniformity": float(kept.min() / kept.max()) if kept.size else float("nan"),
            }

    if not counts or best is None:
        return {**empty, "note": "no usable interior signal"}

    boundary_strength = float(np.mean(strengths))
    tally = {c: counts.count(c) for c in set(counts)}
    persistent = max(tally, key=lambda c: (tally[c], c))
    support = tally[persistent] / len(counts)

    if boundary_strength < _MIN_BOUNDARY_STRENGTH:
        return {
            **empty,
            "parts": 1,
            "boundary_strength": boundary_strength,
            "scale_support": support,
            "note": "interior is flat; no structure to report",
        }

    return {
        "parts": persistent,
        "scale_support": support,
        "boundary_strength": boundary_strength,
        "coverage": best["coverage"],
        "size_uniformity": best["size_uniformity"],
        "valid": True,
        "note": "",
    }


def main() -> int:
    """Reproduce the negative result: the controls that break, and the flower that doesn't help."""
    from benchmarks.fixtures import (
        flower,
        gradient_disc,
        multicolour_logo,
        ring,
        solid_disc,
        spotted_ball,
        striped_ball,
        textured_natural,
    )
    from fusion_vision_mcp.grounding_dino import GroundingDino
    from fusion_vision_mcp.sam2 import Sam2

    gd, sam = GroundingDino(), Sam2()
    cases = [
        ("flower@384", flower(384), "petal", None, False),
        ("flower@768", flower(768), "petal", None, False),
        ("ring8_overlap67", ring(8, 0.67), "petal", 8, False),
        ("NEG solid_disc", solid_disc(), "petal", 1, True),
        ("NEG gradient_disc", gradient_disc(), "petal", 1, True),
        ("NEG striped_ball", striped_ball(), "petal", 1, True),
        ("NEG spotted_ball", spotted_ball(), "petal", 1, True),
        ("NEG multicolour_logo", multicolour_logo(), "shape", 1, True),
        ("NEG textured_natural", textured_natural(), "blob", 1, True),
    ]

    print(f"{'case':<22}{'exp':>5}{'parts':>7}{'bstr':>7}{'valid':>7}  flag")
    print("-" * 60)
    breaks = 0
    for name, img, prompt, expected, is_neg in cases:
        det = gd.detect_objects([img], prompt)[0]
        if not det["bboxes"]:
            print(f"{name:<22}  no detection")
            continue
        boxes = np.array(det["bboxes"])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask = sam.segment(img, [[int(v) for v in boxes[int(np.argmax(areas))]]])[0]
        r = interior_structure(np.asarray(img.convert("RGB")), mask)
        flag = ""
        if is_neg and r["parts"] != 1:
            flag, breaks = "NEG-BREAK", breaks + 1
        elif expected is not None and r["parts"] == expected:
            flag = "ok"
        print(
            f"{name:<22}{str(expected) if expected is not None else '?':>5}{r['parts']:>7}"
            f"{r['boundary_strength']:>7.2f}{r['valid']!s:>7}  {flag} {r['note']}"
        )

    print(f"\nnegative-control breaks: {breaks}")
    print("GATE:", "PASSED" if breaks == 0 else "FAILED - rejected, does not ship")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
