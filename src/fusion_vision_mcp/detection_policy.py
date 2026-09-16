"""Central detection execution policy for the FastMCP application.

Implements the shared execution logic for detect_objects, count_objects, and
spatial_relations (spec A 11) to eliminate duplicated and inconsistent threshold/guard
logic across the three call sites.
"""

from __future__ import annotations

# We import AppContext only for type checking to avoid circular imports.
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from fusion_vision_mcp import query_policy

# From constants, not grounding_dino: that module imports torch at its top level, and
# this one is on the package's eager import path, which must stay torch-free.
from fusion_vision_mcp.constants import DEFAULT_BOX_THRESHOLD
from fusion_vision_mcp.routing import choose_count_backend

if TYPE_CHECKING:
    from fusion_vision_mcp import AppContext


def execute_detection(
    app: AppContext,
    images: list[Image.Image],
    object_name: str,
    backend_preference: Literal["florence2", "grounding_dino", "auto"],
    threshold: float = DEFAULT_BOX_THRESHOLD,
    adaptive_threshold: bool = True,
) -> list[dict[str, Any]]:
    """Shared detection execution.

    Handles backend routing, generic query full-frame guards, threshold passing,
    and formats the output to the standard {"count", "bboxes", "points", "labels",
    "scores", "detector"} shape.

    If `backend_preference` is `"auto"`, the backend is chosen dynamically via
    `choose_count_backend`, and `_routing` metadata is attached to the results.
    """
    image = images[0]
    exclude_full_frame = query_policy.is_generic_query(object_name)

    if backend_preference == "auto":
        routing = choose_count_backend(image)
        backend = routing.backend
        routing_metadata = {
            "backend": routing.backend,
            "unique_colors_per_1k_px": round(routing.unique_colors_per_1k_px, 2),
            "mean_saturation": round(routing.mean_saturation, 2),
            "luminance_stddev": round(routing.luminance_stddev, 2),
            "reason": routing.reason,
        }
    else:
        backend = backend_preference
        routing_metadata = None

    if backend == "florence2":
        # Florence-2's grounding head. We always pass exclude_full_frame based
        # on the query policy, fixing the bug where explicit clip_art=True
        # blindly dropped legitimate full-frame matches (like a close-up of wood).
        detected = app.florence2.detect_objects(images, object_name, exclude_full_frame=exclude_full_frame)
        results = [
            {
                "count": len(d.get("bboxes", [])),
                "bboxes": d.get("bboxes", []),
                "points": d.get("points", []),
                "labels": d.get("labels", []),
                "scores": None,
                "group_boxes_dropped": None,
                "detector": "florence2",
            }
            for d in detected
        ]
    else:
        # Grounding DINO. It uses the passed thresholds.
        results = app.counter.detect_objects(
            images, object_name, threshold=threshold, adaptive_threshold=adaptive_threshold
        )
        if exclude_full_frame:
            # Drop full frame boxes if the query was generic.
            results = [
                query_policy.suppress_generic_full_frame(object_name, result, img.width, img.height)
                for img, result in zip(images, results, strict=True)
            ]

    # Attach routing diagnostic metadata to every result if routing fired
    if routing_metadata:
        for r in results:
            r["_routing"] = routing_metadata

    return results


__all__ = ["execute_detection"]
