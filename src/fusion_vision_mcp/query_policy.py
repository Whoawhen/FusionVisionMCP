"""Recognizes a maximally vague object name, for the blank-canvas false-detection guard.

`detect_objects("object")` on a blank canvas returns a box spanning almost the entire frame --
Florence-2's grounding head has no explicit "nothing here" output for a query with nothing
distinctive to point to, so it falls back to the whole image. `Florence2.detect_objects`'s
`exclude_full_frame` option already exists to drop that box, but applying it unconditionally
would also drop a *legitimate* full-frame match (a close-up of wood, asked for "wood"). The
fix has to be query-aware: only a maximally generic noun with nothing else to go on should
trigger it.

This is exact-match on the whole (stripped, lowercased) query, not a substring test -- "the
small object on the left" names something specific and must not be treated as generic just
because it contains the word "object".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

#: Nouns vague enough that "found the whole frame" is more likely a fallback than a real match.
GENERIC_OBJECT_QUERIES: Final[frozenset[str]] = frozenset(
    {
        "object",
        "objects",
        "thing",
        "things",
        "item",
        "items",
        "stuff",
        "something",
        "shape",
        "shapes",
        "entity",
        "entities",
    }
)


def is_generic_query(query: str) -> bool:
    """True when `query` is nothing but a maximally vague noun, exactly.

    Whitespace and case are normalized before the comparison; anything else about the query
    (an article, an adjective, a second word) takes it out of scope, since a phrase like
    "a red object" or "the object on the left" is specific enough that the blank-canvas
    failure mode this exists to catch does not apply.
    """
    return query.strip().lower() in GENERIC_OBJECT_QUERIES


#: Share of the image's own area at/above which a box is treated as spanning "the whole
#: frame" rather than one instance. Mirrors `Florence2.detect_objects`'s own
#: `exclude_full_frame` cutoff, so the two independent detection heads agree on what
#: "full frame" means.
FULL_FRAME_AREA_RATIO: Final[float] = 0.98


def is_full_frame_box(box: Sequence[float], image_width: int, image_height: int) -> bool:
    """True when `box` covers at least `FULL_FRAME_AREA_RATIO` of the image's area."""
    image_area = image_width * image_height
    if image_area <= 0:
        return False
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / image_area >= FULL_FRAME_AREA_RATIO


def suppress_generic_full_frame(
    object_name: str, result: dict[str, Any], image_width: int, image_height: int
) -> dict[str, Any]:
    """Drop near-full-frame boxes from a Grounding-DINO-shaped detection result.

    Grounding DINO has the same fallback-to-the-whole-image failure mode as Florence-2's
    grounding head (see `florence2.Florence2.detect_objects`'s `exclude_full_frame`):
    measured live on `tests/detect_blank_canvas.png`, `object_name="object"` returns a
    single box spanning the entire frame (score 0.46) rather than nothing. Unlike
    `exclude_full_frame`, this only ever applies when `object_name` is the exact generic
    vocabulary `is_generic_query` recognizes -- a specific noun that genuinely fills the
    frame (a close-up of wood, asked for "wood") is returned untouched, even from this
    same detector.

    `result` is expected in the `{"bboxes", "points", "labels", "scores", "count", ...}`
    shape `InstanceDetector.detect_objects` and `GroundingDino.detect_objects` share.
    Returns `result` unchanged (same object) when nothing was dropped, so a caller can
    tell whether the guard fired at all by identity or by the new `count`.
    """
    if not is_generic_query(object_name):
        return result

    bboxes = result.get("bboxes") or []
    keep = [i for i, box in enumerate(bboxes) if not is_full_frame_box(box, image_width, image_height)]
    if len(keep) == len(bboxes):
        return result

    filtered = dict(result)
    filtered["bboxes"] = [bboxes[i] for i in keep]
    for key in ("points", "labels", "scores"):
        values = result.get(key)
        if values is not None:
            filtered[key] = [values[i] for i in keep]
    filtered["count"] = len(filtered["bboxes"])
    filtered["generic_full_frame_dropped"] = len(bboxes) - len(keep)
    return filtered


__all__ = [
    "FULL_FRAME_AREA_RATIO",
    "GENERIC_OBJECT_QUERIES",
    "is_full_frame_box",
    "is_generic_query",
    "suppress_generic_full_frame",
]
