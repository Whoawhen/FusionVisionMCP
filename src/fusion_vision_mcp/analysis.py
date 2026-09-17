"""Cross-checks, measurements and enrichment shared by the tools.

Split out of the package entry point, which had grown past 1,900 lines with the tool
definitions and these helpers interleaved. Nothing here registers an MCP tool; each
function is called by one and takes the already-resolved `AppContext`.

Like `protocols`, this module is on the package's eager import path and must stay
torch-free -- take constants from `constants`, never from a model wrapper.
"""

import logging
import math
import re
from collections.abc import Sequence
from typing import Any, Final, cast

from PIL.Image import Image

from fusion_vision_mcp import geometry, layout, question
from fusion_vision_mcp.constants import CaptionLevel
from fusion_vision_mcp.protocols import AppContext, Processor

#: Ceiling on objects compared in one `spatial_relations` call. Relations grow as
#: n(n-1)/2, so an over-broad detection could otherwise return a huge payload.
_MAX_RELATED_OBJECTS: Final[int] = 12

logger = logging.getLogger(__name__)


#: Only a single detection triggers the silhouette check -- that is precisely the
#: answer `count_objects` documents as "could not separate them", so the segmentation
#: buys information where there currently is none. Any healthy count skips it, which
#: is what keeps SAM2 unloaded for sessions that never hit the collapse case.
_COLLAPSE_SUSPECT_COUNT: Final[int] = 1

#: A box smaller than this share of the frame is one small object, not a merged group.
_MIN_VERIFY_BOX_AREA: Final[float] = 0.01


def _add_silhouette(app: AppContext, image: Image, result: dict[str, Any]) -> None:
    """Attach a geometric second opinion when the detector collapsed to one region.

    Segments that single region and measures its outline, adding the result under
    `silhouette` without touching `count`. The two numbers are produced by different
    methods and the tool deliberately reports both rather than reconciling them.
    """
    if result.get("count") != _COLLAPSE_SUSPECT_COUNT or not result.get("bboxes"):
        return

    box = [int(v) for v in result["bboxes"][0]]
    width, height = max(box[2] - box[0], 0), max(box[3] - box[1], 0)
    if not image.width or not image.height:
        return
    if width * height < _MIN_VERIFY_BOX_AREA * image.width * image.height:
        return

    masks = app.segmenter.segment(image, [box])
    if not masks:
        return
    result["silhouette"] = {"box": box, **geometry.count_lobes(masks[0])}


def _region_label_consensus(processor: Processor, image: Image, object_name: str) -> int:
    """Second-opinion count: how many `dense_region_caption` labels match the object.

    Florence-2's dense region captioner emits a label per salient region ("a red car",
    "a person", "a yellow flower"). Tallying labels that contain the object name is an
    independent estimate of the instance count -- it comes from a different head than
    Grounding DINO, so agreement between the two is real evidence and disagreement is
    a visible warning. Returns 0 when the captioner finds nothing matching.
    """
    regions = processor.dense_region_caption([image])[0]
    needle = object_name.strip().lower()
    count = 0
    for label in regions.get("labels", []):
        if needle and needle in str(label).lower():
            count += 1
    return count


def _separability(result: dict[str, Any]) -> str:
    """Surfaces whether the reported `count` is a real tally or a structural collapse.

    ``"yes"`` -- the detector separated the instances (count > 1), or a single region
    whose silhouette both estimators (by_distance and the rosette-specific by_radial)
    agree is one lobe.
    ``"no"`` -- the detector collapsed while the silhouette's evidence indicates
    multiple lobes, or both estimators agree the count is > 1.
    ``"unknown"`` -- no silhouette check available, bad data (shattered/clipped), or
    the estimators disagree on a genuinely ambiguous shape with no strong signal either
    way (neither found multiple lobes).
    """
    count = result.get("count")
    if count is None:
        return "unknown"
    if count > 1:
        return "yes"
    # count == 1: only a silhouette block can tell a real singleton from a collapse.
    silhouette = result.get("silhouette")
    if not silhouette:
        return "unknown"
    if silhouette.get("shattered") or silhouette.get("clipped"):
        return "unknown"

    by_distance: int | None = silhouette.get("lobes")  # alias set by count_lobes
    by_radial: int = silhouette.get("by_radial") or 0  # rosette estimator, 0 = not measured
    agreement: bool = bool(silhouette.get("agreement"))

    if by_distance is None:
        return "unknown"

    # by_radial == 0: the rosette estimator was not applicable (not a rosette shape),
    # so by_distance is the only signal the geometry module can provide.
    if by_radial <= 0:
        return "yes" if by_distance <= 1 else "no"

    # by_radial > 0: the rosette estimator was applicable. Agreement is the key signal.
    if agreement:
        # Both estimators agree — the agreed count is trustworthy.
        return "yes" if by_distance <= 1 else "no"

    # Disagreement between the two estimators.
    # by_radial > 1 with by_distance == 1 is the canonical rosette collapse:
    # the distance estimator found no saddle (overlapping petals form one blob),
    # while the radial estimator caught the angular notch pattern (lobes < convex hull).
    # The documented flower case: by_distance=1, by_radial=8 → "no".
    if by_radial > 1:
        return "no"
    return "unknown"


# Short answers Moondream2 emits as a flat default regardless of the image -- the
# documented failure mode where it answers "None" to "describe anything wrong" on
# images that all had real visible defects. Seeing one of these agreed upon by two
# differently-phrased questions is the signature of a default, not an observation.
_DEFAULT_ANSWERS: Final[frozenset[str]] = frozenset(
    {"none", "nothing", "yes", "no", "n/a", "na", "i don't know", "unknown", "not sure", ""}
)


def _vqa_consistency(answer: str, control_answer: str) -> dict[str, Any]:
    """Compares an answer to its control-question answer and flags unreliable responses.

    `consistent` is true when the two answers agree -- exact (case/punctuation-stripped)
    match for short answers, or one containing the other for longer ones.

    `confidence` is `"low"` in two failure modes:

    1. *Agreed flat default* -- both answers reduce to a known default token ("none",
       "yes", "nothing", ...) and they agree: the model produced the same short answer
       to two differently-phrased questions without looking at the image. This is the
       documented signature from the six-image probe where Moondream answered "None"
       to "describe anything wrong" on images with real visible defects.

    2. *Self-contradiction* -- the two answers substantively disagree. One says
       something's wrong, the other says nothing's wrong, or they give opposite
       factual claims about the same image. A model contradicting itself across two
       rephrased questions is weaker evidence than either answer taken alone.
       (The documented "pette" probe: one answer said "missing a centerpiece" and the
       control said "Nothing is wrong" -- both substantive, both contradicting.)

    ``"normal"`` only when the answers are substantive AND not flat defaults AND
    they agree: real observations the consistency layer can confirm.
    """
    a = _normalize_answer(answer)
    c = _normalize_answer(control_answer)

    if a and c:
        consistent = a == c or a in c or c in a
    else:
        consistent = a == c

    both_default = a in _DEFAULT_ANSWERS and c in _DEFAULT_ANSWERS

    # "normal" only when the answers are substantive AND they agree.
    # Everything else is signal for a caller to distrust:
    #   - agreed flat defaults (both_default and consistent)  →  model is not looking
    #   - disagreement  (not consistent)                      →  model is contradicting itself
    confidence = "normal" if (consistent and not both_default) else "low"

    return {
        "answer": answer,
        "control_answer": control_answer,
        "consistent": consistent,
        "confidence": confidence,
    }


def _normalize_answer(text: str) -> str:
    """Lowercases, strips punctuation/whitespace, for default-token comparison."""
    return text.strip().lower().rstrip(".?!,;:")


def _first_int(text: str) -> int | None:
    """First integer appearing in a string, or None (Moondream may answer in prose)."""
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _vqa_cross_check(app: AppContext, image: Image, question_text: str) -> dict[str, Any] | None:
    """Route an unreliable VQA answer to the measurement that answers it.

    Classifies the question's wording and, when a measurable category applies and
    the object names can be parsed from that wording, runs the matching tool's
    underlying measurement and returns its result:

    - ``spatial`` -> ``spatial_relations`` (detect + segment both objects, measure
      their relation -- contact, gap, containment).
    - ``count`` -> ``count_objects`` (detect + count, with a separability flag).
    - ``ocr`` -> ``ocr`` (verbatim transcription, a second opinion from a different
      Florence-2 head).

    Returns None -- meaning "omit the cross-check" -- when no measurable category
    applies or the names can't be parsed to the required arity. It never guesses an
    object name, so a low-confidence "describe the mood" (no measurement) produces
    no cross-check rather than a fabricated one.
    """
    category = question.classify(question_text)
    if category is None:
        return None
    names = question.names_for(category, question_text)
    if names is None:
        return None

    if category == question.SPATIAL:
        return _spatial_measurement(app, image, cast(list[str], names))
    if category == question.COUNT:
        result = app.counter.detect_objects([image], names[0])[0]
        return {
            "tool": "count_objects",
            "object": names[0],
            "count": result.get("count"),
            "separable": _separability(result),
        }
    if category == question.OCR:
        return {"tool": "ocr", "text": ocr_pages(app, [image])[0]["text"]}
    if category == question.SIZE:
        return _size_measurement(app, image, names[0], question_text)
    return None


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _size_measurement(app: AppContext, image: Image, object_name: str, question_text: str) -> dict[str, Any] | None:
    """Resolve a "which is largest/smallest <name>" judgment by comparing detected boxes.

    Detects every instance of the named object and picks the extremum by bounding-box
    area -- a measurement, not a guess. Returns None (omit the cross-check) when
    nothing was detected, the same "never guess" rule the other categories follow.
    """
    detected = app.counter.detect_objects([image], object_name)[0]
    bboxes = detected.get("bboxes") or []
    if not bboxes:
        return None

    want_smallest = "smallest" in question_text.lower()
    best_index = (min if want_smallest else max)(range(len(bboxes)), key=lambda i: _box_area(bboxes[i]))
    scores = detected.get("scores")

    return {
        "tool": "detect_objects",
        "object": object_name,
        "extremum": "smallest" if want_smallest else "largest",
        "box": [int(v) for v in bboxes[best_index]],
        "area": _box_area(bboxes[best_index]),
        "instances_compared": len(bboxes),
        "score": scores[best_index] if scores else None,
    }


def _spatial_measurement(app: AppContext, image: Image, names: list[str]) -> dict[str, Any] | None:
    """Detect and segment two named objects and measure their relation.

    Mirrors ``spatial_relations`` for the two-object case: detect each name, keep
    the best-scoring box, segment both, and return ``geometry.relation`` between
    the two masks. Returns None when fewer than two objects could be located, so the
    cross-check is omitted rather than reporting a half-measurement.
    """
    located: list[dict[str, Any]] = []
    boxes: list[list[int]] = []
    for call_index, object_name in enumerate(names):
        if len(located) >= _MAX_RELATED_OBJECTS:
            break
        detected = app.counter.detect_objects([image], object_name)[0]
        if not detected["bboxes"]:
            continue
        best = max(range(len(detected["bboxes"])), key=lambda i: detected["scores"][i])
        box = [int(v) for v in detected["bboxes"][best]]
        located.append({"label": object_name, "box": box})
        boxes.append(box)
    if len(located) < 2:
        return None

    masks = app.segmenter.segment(image, boxes)
    return {
        "tool": "spatial_relations",
        "objects": [obj["label"] for obj in located],
        "relation": geometry.relation(masks[0], masks[1]),
    }


def _enrich_aesthetics(app: AppContext, image: Image, result: dict[str, Any], style: dict[str, Any] | None) -> None:
    """
    Conceptually splits technical_quality, photographic_aesthetic, and artistic_judgment.
    Modifies the `result` dictionary in-place.
    """
    result["photographic_aesthetic"] = result["score"]

    is_photo = True
    if style is not None and style.get("style") != "photograph":
        is_photo = False

    result["photographic_aesthetic_applicable"] = is_photo

    if app.iqa is not None:
        try:
            iqa_res = app.iqa.score(image)
            result["technical_quality"] = iqa_res.get("technical_quality")
        except Exception as exc:
            # Report the failure rather than omitting the field: an absent key is
            # indistinguishable from "IQA not configured", which hides a broken backend.
            logger.warning("Technical IQA scoring failed", exc_info=True)
            result["technical_quality_error"] = str(exc)

    if app.reasoner is not None:
        try:
            observations = [
                {"source": "aesthetic_scorer", "description": f"Photographic aesthetic score: {result['score']}"}
            ]
            measurements: dict[str, Any] = {"photographic_aesthetic": result["score"]}
            if "technical_quality" in result:
                observations.append(
                    {"source": "technical_iqa", "description": f"Technical quality: {result['technical_quality']}"}
                )
                measurements["technical_quality"] = result["technical_quality"]

            r_out = app.reasoner.analyze(
                question="Provide an artistic judgment of this image's aesthetics based on the measurements.",
                observations=observations,
                measurements=measurements,
            )
            if r_out:
                result["artistic_judgment"] = r_out.as_dict()
        except Exception as exc:
            # Same rationale as the IQA branch above: surface the failure.
            logger.warning("Artistic judgment via the reasoner failed", exc_info=True)
            result["artistic_judgment_error"] = str(exc)


#: Below this absolute delta, two scores are a tie (noise, not a preference).
_COMPARE_TIE: Final[float] = 0.05


def _attach_annotated_images(images: list[Image], results: list[dict[str, Any]], fallback_label: str = "") -> None:
    """Render boxes onto each image and record the path on its result, in place.

    `fallback_label` names the detections when the backend returned none -- counting
    reports a tally without per-box labels, so the boxes would otherwise be unlabelled.
    """
    from fusion_vision_mcp.annotator import save_annotated_image

    for img, res in zip(images, results, strict=False):
        boxes = res.get("bboxes")
        if not boxes:
            continue
        labels = res.get("labels")
        if not labels and fallback_label:
            labels = [f"{fallback_label} {i + 1}" for i in range(len(boxes))]
        res["annotated_image_path"] = save_annotated_image(img, boxes, labels)


def _aesthetic_comparison(
    app: AppContext, images: list[Image], ref_images: list[Image], style_context: bool
) -> list[dict[str, Any]]:
    """Score an image against a reference and report the relative preference.

    The predictor's documented valid use is like-with-like comparison, so this
    routes the caller to a calibrated relative answer instead of a single
    bias-affected absolute number. Each image page is compared to the first page of
    the reference. With ``style_context``, both media are classified and a
    ``cross_medium_warning`` is added when they differ (cross-medium comparison is
    out of calibrated scope). The absolute scores are not recalibrated.
    """
    img_styles: Sequence[dict[str, Any] | None]
    ref_style: dict[str, Any] | None
    if style_context:
        img_scores, img_styles = app.aesthetic.score_and_classify(images)
        ref_scores, ref_styles = app.aesthetic.score_and_classify(ref_images)
        ref_score, ref_style = ref_scores[0], ref_styles[0]
    else:
        img_scores = app.aesthetic.score(images)
        ref_score = app.aesthetic.score(ref_images)[0]
        img_styles = [None] * len(images)
        ref_style = None

    _enrich_aesthetics(app, ref_images[0], ref_score, ref_style)

    # The reference's style is the same for every page, so stamp it once here rather
    # than re-applying it per iteration below.
    if style_context and ref_style is not None:
        ref_score["style"] = ref_style["style"]
        ref_score["style_distribution"] = ref_style["distribution"]

    out: list[dict[str, Any]] = []
    for image, img, ist in zip(images, img_scores, img_styles, strict=True):
        _enrich_aesthetics(app, image, img, ist)
        delta = round(img["score"] - ref_score["score"], 4)
        if delta > _COMPARE_TIE:
            preferred = "image"
        elif delta < -_COMPARE_TIE:
            preferred = "reference"
        else:
            preferred = "tie"

        entry: dict[str, Any] = {
            "image": img,
            # A copy per entry: a shared dict would let a consumer mutating one
            # page's reference silently rewrite every other page's.
            "reference": dict(ref_score),
            "delta": delta,
            "preferred": preferred,
        }
        if style_context and ist is not None and ref_style is not None:
            img["style"] = ist["style"]
            img["style_distribution"] = ist["distribution"]
            if ist["style"] != ref_style["style"]:
                entry["cross_medium_warning"] = (
                    "the image and reference are different media; the score is calibrated "
                    "for like-with-like comparison, so this delta is out of scope"
                )
        out.append(entry)
    return out


def _critique_one(
    app: AppContext,
    image: Image,
    target_subject: str,
    low_score_threshold: float,
    style_context: bool,
) -> dict[str, Any]:
    """Single-image composition critique -- the body of `critique_composition`.

    Extracted so `critique_composition`'s relative mode can run it for both the
    image and the reference. See that tool's docstring for the return shape.
    """
    box: list[int] | None
    if target_subject:
        detected = app.florence2.detect_objects([image], target_subject)[0]
        box = [int(v) for v in detected["bboxes"][0]] if detected["bboxes"] else None
    else:
        box = _pick_primary_subject(app.florence2.dense_region_caption([image])[0], (image.width, image.height))

    style: dict[str, Any] | None = None
    if style_context:
        scores, styles = app.aesthetic.score_and_classify([image])
        aesthetics, style = scores[0], styles[0]
    else:
        aesthetics = app.aesthetic.score([image])[0]

    _enrich_aesthetics(app, image, aesthetics, style)

    if box is None:
        result: dict[str, Any] = {
            "image_size": [image.width, image.height],
            "aesthetics": aesthetics,
            "note": "Could not locate a subject to assess framing for.",
        }
        if style is not None:
            result["style"] = style["style"]
            result["style_distribution"] = style["distribution"]
        return result

    result = {
        "image_size": [image.width, image.height],
        "subject_box": box,
        "aesthetics": aesthetics,
        "framing": geometry.rule_of_thirds(box, (image.width, image.height)),
    }
    if style is not None:
        result["style"] = style["style"]
        result["style_distribution"] = style["distribution"]
    if aesthetics["score"] < low_score_threshold:
        result["critique"] = app.vqa.query(
            [image], "Critique this photo's composition and framing in 2 concise sentences."
        )[0]
    return result


def ocr_pages(app: AppContext, images: Sequence[Image]) -> list[dict[str, Any]]:
    """Transcribe each page with EasyOCR, splitting side-by-side columns first.

    This is the `ocr` tool's whole implementation, factored out so the other two
    places that transcribe text cannot drift from it. Both used to call
    `app.florence2.ocr` -- the head EasyOCR replaced -- so the same named operation
    returned different text depending on how it was reached, and on a text-free
    image the Florence-2 path invented some. `tests/test_server.py::
    test_ocr_textless_image_returns_no_text` pins the correct behaviour for the
    direct tool; routing everything through here extends it to the other two.

    Returns one `{"text", "text_regions"}` dict per page. Boxes are in page
    coordinates: a column crop's boxes are shifted right by the crop's origin,
    which is why the offset is accumulated here rather than inside `split_columns`.
    """
    pages: list[dict[str, Any]] = []
    for image in images:
        x_offset = 0
        page_texts: list[str] = []
        page_text_regions: list[dict[str, Any]] = []

        for crop in layout.split_columns(image):
            for region in app.ocr_specialist.readtext(crop):
                box = region["box"]
                box[0] += x_offset
                box[2] += x_offset
                page_texts.append(region["text"])
                page_text_regions.append({"text": region["text"], "confidence": region["confidence"], "box": box})
            x_offset += crop.width

        pages.append({"text": "\n".join(page_texts), "text_regions": page_text_regions})
    return pages


def _dispatch(app: AppContext, operation: str, images: list[Image], *, question: str, object_name: str) -> Any:
    """Routes a `batch_analyze_images` operation to the right processor call."""
    if operation == "caption":
        return app.florence2.caption(images, CaptionLevel.MORE_DETAILED)
    if operation == "ocr":
        # Must match the `ocr` tool exactly -- see `ocr_pages`.
        return [page["text"] for page in ocr_pages(app, images)]
    if operation == "detect":
        if not object_name:
            raise ValueError("object_name is required for the 'detect' operation")
        return app.florence2.detect_objects(images, object_name)
    if operation == "dense_caption":
        return app.florence2.dense_region_caption(images)
    if operation == "query":
        if not question:
            raise ValueError("question is required for the 'query' operation")
        return app.vqa.query(images, question)
    if operation == "count":
        if not object_name:
            raise ValueError("object_name is required for the 'count' operation")
        # No silhouette check here: batching is the throughput path, and the check
        # would pull SAM2 in behind the caller's back once per collapsed image.
        return app.counter.detect_objects(images, object_name)
    raise ValueError(f"Unknown operation: {operation!r}")


def _pick_primary_subject(regions: dict[str, Any], image_size: tuple[int, int]) -> list[int] | None:
    """Picks the most prominent region from `dense_region_caption`'s output.

    Scores each box by area weighted toward the image center, since the most
    prominent subject in a photo is usually both large and roughly centered.
    Returns None if no regions were found.
    """
    boxes = regions.get("bboxes") or []
    if not boxes:
        return None

    width, height = image_size
    diagonal = math.hypot(width, height)

    def score(box: list[float]) -> float:
        x0, y0, x1, y1 = box
        area = max(x1 - x0, 0) * max(y1 - y0, 0)
        distance = math.hypot((x0 + x1) / 2 - width / 2, (y0 + y1) / 2 - height / 2)
        centrality = 1 - min(distance / diagonal, 1.0) if diagonal else 1.0
        return area * centrality

    return [int(v) for v in max(boxes, key=score)]
