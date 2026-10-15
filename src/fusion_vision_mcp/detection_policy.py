"""Central detection execution policy for the FastMCP application.

Implements the shared execution logic for detect_objects, count_objects, and
spatial_relations (spec A 11) to eliminate duplicated and inconsistent threshold/guard
logic across the three call sites.
"""

from __future__ import annotations

# We import AppContext only for type checking to avoid circular imports.
import logging
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from fusion_vision_mcp import query_policy

# From constants, not grounding_dino: that module imports torch at its top level, and
# this one is on the package's eager import path, which must stay torch-free.
from fusion_vision_mcp.constants import DEFAULT_BOX_THRESHOLD, DETECT_BOX_THRESHOLD, LOCATE_BOX_THRESHOLD
from fusion_vision_mcp.routing import choose_count_backend

if TYPE_CHECKING:
    from fusion_vision_mcp import AppContext
    from fusion_vision_mcp.protocols import InstanceDetector

logger = logging.getLogger(__name__)


def execute_detection(
    app: AppContext,
    images: list[Image.Image],
    object_name: str,
    backend_preference: Literal["florence2", "grounding_dino", "auto"],
    threshold: float = DEFAULT_BOX_THRESHOLD,
    adaptive_threshold: bool = True,
    locate: bool = False,
    candidates_floor: float | None = None,
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

    flat_sam3 = _flat_locate(app, images, object_name) if backend == "florence2" and locate else None
    if flat_sam3 is not None:
        results = flat_sam3
        if exclude_full_frame:
            results = [
                query_policy.suppress_generic_full_frame(object_name, result, img.width, img.height)
                for img, result in zip(images, results, strict=True)
            ]
    elif backend == "florence2":
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
        results = _photo_detect(app, images, object_name, threshold, adaptive_threshold, locate, candidates_floor)
        if exclude_full_frame:
            # Drop full frame boxes if the query was generic.
            results = [
                query_policy.suppress_generic_full_frame(object_name, result, img.width, img.height)
                for img, result in zip(images, results, strict=True)
            ]

    # Attach routing diagnostic metadata to every result if routing fired. `backend` names the model that
    # actually ran: the photo branch is SAM 3 for `detect_objects`, or Grounding DINO when counting or as
    # SAM 3's fallback, so the routing decision's own "grounding_dino" would be wrong half the time.
    if routing_metadata:
        for r in results:
            r["_routing"] = {**routing_metadata, "backend": r.get("detector", routing_metadata["backend"])}

    return results


def _flat_locate(app: AppContext, images: list[Image.Image], object_name: str) -> list[dict[str, Any]] | None:
    """SAM 3 for `detect_objects` on flat images; None (use Florence-2) without it.

    VisionBench gate F (2026-10-14): on 88 fresh flat images SAM 3 boxed the absent object on 11.4% against
    Florence-2's 100%, and found 53.8% of labelled objects against 40.5% (`sam3_flat_confirm.py`; the screen on 63
    other images agreed). Counting keeps Florence-2 on flat images.
    """
    locator: InstanceDetector | None = getattr(app, "locator", None)
    if locator is None:
        return None
    try:
        results = locator.detect_objects(images, object_name, threshold=LOCATE_BOX_THRESHOLD)
    except OSError as exc:
        logger.warning(f"SAM 3 unavailable ({exc}); detect_objects on a flat image falls back to Florence-2")
        return None
    for result in results:
        result["detector"] = "sam3"
    return results


def _photo_detect(
    app: AppContext,
    images: list[Image.Image],
    object_name: str,
    threshold: float,
    adaptive_threshold: bool,
    locate: bool,
    candidates_floor: float | None = None,
) -> list[dict[str, Any]]:
    """SAM 3 when locating (`detect_objects`), otherwise Grounding DINO at the passed thresholds."""
    locator: InstanceDetector | None = getattr(app, "locator", None)
    if locate and locator is not None:
        try:
            results = locator.detect_objects(images, object_name, threshold=LOCATE_BOX_THRESHOLD)
        except OSError as exc:
            # Gated on Hugging Face: without accepted access or a login the download fails with an
            # OSError subclass. Fall back to what shipped before SAM 3, at its own threshold.
            logger.warning(f"SAM 3 unavailable ({exc}); detect_objects falls back to Grounding DINO")
            threshold, adaptive_threshold = DETECT_BOX_THRESHOLD, False
        else:
            for result in results:
                result["detector"] = "sam3"
            return results
    # Only counting asks for the candidates, and only then is the keyword passed, so detectors
    # written before it existed (and test doubles) keep working.
    extra = {"candidates_floor": candidates_floor} if candidates_floor is not None else {}
    results = app.counter.detect_objects(
        images, object_name, threshold=threshold, adaptive_threshold=adaptive_threshold, **extra
    )
    for result in results:
        result["detector"] = "grounding_dino"
    return results


__all__ = ["execute_detection"]
