#  tiling.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Experiment: does tiled inference recover heavily-overlapping instances?

The one positive the shipped detector still misses is eight shapes overlapping by
roughly two-thirds of their width, counted as 6. Tiling is the standard remedy for
dense and small objects: run the detector over overlapping crops, where each instance
occupies more pixels and fewer neighbours share the frame, then merge.

The risk is the mirror image of the hope. A crop boundary cuts instances in half, and
the same instance seen in two crops must be merged rather than counted twice, so tiling
can just as easily inflate a count as recover one.

## Measured: it does not help, and it is expensive

| | positives | negatives held | MAE | latency |
| --- | --- | --- | --- | --- |
| shipped, no tiling | **9/10** | **8/8** | **0.10** | **2.07s** |
| tiled 2x2 | 8/10 | 7/8 | 0.45 | 10.6s |
| tiled 3x3 | 8/10 | 5/8 | 0.95 | 24.9s |

It buys one point on the case it was aimed at -- eight shapes at two-thirds overlap go
from 6 to 7 at 3x3 -- and pays for it with a regression everywhere else at five to
twelve times the cost.

The instructive failure is the spotted ball, a single object covered in high-contrast
spots: 1 without tiling, **6** at 2x2, **15** at 3x3. Cropping in removes the context
that made those spots read as surface texture, and each one becomes an object in its own
right. Forty-percent-overlap rings also start overcounting at 9, because an instance
straddling a cut is seen whole in one tile and partially in another, and the merge
cannot always tell those apart. The flower stays noise: 7/2/1 at 2x2 and 4/5/5 at 3x3
across three resolutions of the same picture.

**Rejected.** Kept so the result is reproducible rather than asserted; not imported from
`src/`.

Run `python -m benchmarks.tiling` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Fraction of a tile shared with its neighbour. An instance straddling a cut is then
#: whole in at least one tile, provided it is smaller than the overlap band.
TILE_OVERLAP: float = 0.30

#: Two boxes overlapping more than this are treated as the same instance seen twice.
#: Deliberately lower than a detection-NMS threshold, because the duplicates here come
#: from different crops and their coordinates disagree by more than a within-crop pair.
MERGE_IOU: float = 0.45

#: A detection touching a tile edge this closely is probably a cut-off fragment. Kept
#: only if no whole detection elsewhere explains it.
EDGE_MARGIN_PX: int = 4


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    area_b = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge(detections: list[tuple[list[float], float]]) -> tuple[list[list[float]], list[float]]:
    """Greedy score-ordered merge: keep the best box, drop anything overlapping it."""
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    for box, score in sorted(detections, key=lambda d: -d[1]):
        if all(_iou(box, other) <= MERGE_IOU for other in kept_boxes):
            kept_boxes.append(box)
            kept_scores.append(score)
    return kept_boxes, kept_scores


class TiledBackend:
    """Runs the shipped detector over a grid of overlapping crops and merges the result."""

    def __init__(self, model: Any, grid: int = 2, include_full_frame: bool = True) -> None:
        self.model, self.grid, self.include_full_frame = model, grid, include_full_frame

    def _tiles(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        if self.grid <= 1:
            return [(0, 0, width, height)]
        step_x = width / self.grid
        step_y = height / self.grid
        pad_x = step_x * TILE_OVERLAP
        pad_y = step_y * TILE_OVERLAP
        boxes = []
        for row in range(self.grid):
            for col in range(self.grid):
                x0 = max(0, int(col * step_x - pad_x))
                y0 = max(0, int(row * step_y - pad_y))
                x1 = min(width, int((col + 1) * step_x + pad_x))
                y1 = min(height, int((row + 1) * step_y + pad_y))
                boxes.append((x0, y0, x1, y1))
        return boxes

    def detect_objects(self, images: list[Any], object_name: str) -> list[dict[str, Any]]:
        from fusion_vision_mcp.grounding_dino import GroundingDino

        results = []
        for image in images:
            rgb = image.convert("RGB")
            detections: list[tuple[list[float], float]] = []

            if self.include_full_frame:
                whole = self.model.detect_objects([rgb], object_name)[0]
                detections += list(zip(whole["bboxes"], whole["scores"], strict=True))

            for x0, y0, x1, y1 in self._tiles(rgb.width, rgb.height):
                crop = rgb.crop((x0, y0, x1, y1))
                found = self.model.detect_objects([crop], object_name, drop_group_box=False)[0]
                for box, score in zip(found["bboxes"], found["scores"], strict=True):
                    bx0, by0, bx1, by1 = box
                    # A detection spanning nearly the whole crop is the crop's own
                    # contents, not an instance within it.
                    if (bx1 - bx0) > 0.95 * (x1 - x0) and (by1 - by0) > 0.95 * (y1 - y0):
                        continue
                    touches_cut = (
                        (bx0 < EDGE_MARGIN_PX and x0 > 0)
                        or (by0 < EDGE_MARGIN_PX and y0 > 0)
                        or (bx1 > (x1 - x0) - EDGE_MARGIN_PX and x1 < rgb.width)
                        or (by1 > (y1 - y0) - EDGE_MARGIN_PX and y1 < rgb.height)
                    )
                    if touches_cut:
                        continue
                    detections.append(([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0], score))

            bboxes, scores = _merge(detections)
            envelopes = GroundingDino._envelope_indices(bboxes)
            keep = [i for i in range(len(bboxes)) if i not in envelopes]
            bboxes = [bboxes[i] for i in keep]
            results.append(
                {
                    "count": len(bboxes),
                    "bboxes": bboxes,
                    "points": [[(a + c) / 2, (b + d) / 2] for a, b, c, d in bboxes],
                    "labels": [object_name] * len(bboxes),
                    "scores": [scores[i] for i in keep],
                    "group_boxes_dropped": len(envelopes),
                }
            )
        return results


def main() -> int:
    from benchmarks.fixtures import all_fixtures
    from benchmarks.harness import run
    from fusion_vision_mcp.grounding_dino import GroundingDino

    fixtures = all_fixtures()
    model = GroundingDino()

    reports = {"shipped (no tiling)": run(model, fixtures, label="shipped", include_equivalents=False)}
    for grid in (2, 3):
        reports[f"tiled {grid}x{grid}"] = run(
            TiledBackend(model, grid=grid), fixtures, label=f"tiled {grid}x{grid}", include_equivalents=False
        )

    names = list(reports)
    print(f"{'fixture':<24}{'truth':>6}" + "".join(f"{n[:13]:>15}" for n in names))
    print("-" * (30 + 15 * len(names)))
    for i, r in enumerate(reports[names[0]]["results"]):
        truth = "?" if r["truth"] is None else r["truth"]
        cells = "".join(f"{reports[n]['results'][i]['predicted']:>15}" for n in names)
        print(f"{r['fixture']:<24}{truth:>6}{cells}")

    print()
    for name in names:
        s = reports[name]["summary"]
        print(
            f"{name:<22} positives {s['positives_exact']}/{s['positives_total']}  "
            f"negatives {s['negatives_held']}/{s['negatives_total']}  "
            f"MAE {s['mean_abs_error']}  latency {s['mean_latency_s']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
