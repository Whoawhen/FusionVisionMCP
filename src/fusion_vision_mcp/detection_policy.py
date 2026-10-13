"""Central detection execution policy for the FastMCP application.

Implements the shared execution logic for detect_objects, count_objects, and
spatial_relations (spec A 11) to eliminate duplicated and inconsistent threshold/guard
logic across the three call sites.
"""

from __future__ import annotations

# We import AppContext only for type checking to avoid circular imports.
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from PIL import Image

from fusion_vision_mcp import query_policy

# From constants, not grounding_dino: that module imports torch at its top level, and
# this one is on the package's eager import path, which must stay torch-free.
from fusion_vision_mcp.constants import DEFAULT_BOX_THRESHOLD, DETECT_BOX_THRESHOLD, LOCATE_BOX_THRESHOLD
from fusion_vision_mcp.routing import choose_count_backend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from fusion_vision_mcp import AppContext
    from fusion_vision_mcp.protocols import InstanceDetector, MaskLocator

logger = logging.getLogger(__name__)


def execute_detection(
    app: AppContext,
    images: list[Image.Image],
    object_name: str,
    backend_preference: Literal["florence2", "grounding_dino", "auto"],
    threshold: float = DEFAULT_BOX_THRESHOLD,
    adaptive_threshold: bool = True,
    locate: bool = False,
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
        results = _photo_detect(app, images, object_name, threshold, adaptive_threshold, locate)
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


def _photo_detect(
    app: AppContext,
    images: list[Image.Image],
    object_name: str,
    threshold: float,
    adaptive_threshold: bool,
    locate: bool,
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
    results = app.counter.detect_objects(
        images, object_name, threshold=threshold, adaptive_threshold=adaptive_threshold
    )
    for result in results:
        result["detector"] = "grounding_dino"
    return results


def locate_and_segment(
    app: AppContext, image: Image.Image, names: list[str], limit: int
) -> tuple[list[dict[str, Any]], list[NDArray[np.bool_]], str]:
    """The best-scoring instance of each name and its mask, for `spatial_relations`.

    SAM 3 locates and segments in one call (its own instance mask, no SAM2): VisionBench gate S
    (2026-10-13) scored it 25 of 36 Open Images relations against 25 for Grounding DINO + SAM2, and it
    passed both synthetic fixtures. Without SAM 3 (gated, or a hand-built context) this falls back to the
    old path at its old threshold. Returns `(located, masks, locator)`; `located` holds `id`, `label` and
    `box` per name found, in request order, at most `limit` of them.
    """
    locator: MaskLocator | None = getattr(app, "locator", None)
    if locator is not None:
        try:
            located, masks = _located(names, limit, lambda name: locator.locate_masks(image, name), image)
            return located, masks, "sam3"
        except OSError as exc:
            logger.warning(f"SAM 3 unavailable ({exc}); spatial_relations falls back to Grounding DINO + SAM2")
    located, _ = _located(names, limit, lambda name: app.counter.detect_objects([image], name)[0], image)
    masks = app.segmenter.segment(image, [cast(list[int], obj["box"]) for obj in located]) if located else []
    return located, masks, "grounding_dino"


def _located(
    names: list[str], limit: int, detect: Callable[[str], dict[str, Any]], image: Image.Image
) -> tuple[list[dict[str, Any]], list[NDArray[np.bool_]]]:
    """Each name's single best-scoring box after the generic-name guard, and its mask if `detect` gave masks."""
    located: list[dict[str, Any]] = []
    masks: list[NDArray[np.bool_]] = []
    for call_index, name in enumerate(names):
        if len(located) >= limit:
            break
        detected = query_policy.suppress_generic_full_frame(name, detect(name), image.width, image.height)
        if not detected["bboxes"]:
            continue
        best = max(range(len(detected["bboxes"])), key=lambda i: detected["scores"][i])
        located.append({"id": f"{name}#{call_index}", "label": name, "box": [int(v) for v in detected["bboxes"][best]]})
        if "masks" in detected:
            masks.append(detected["masks"][best])
    return located, masks


__all__ = ["execute_detection", "locate_and_segment"]
