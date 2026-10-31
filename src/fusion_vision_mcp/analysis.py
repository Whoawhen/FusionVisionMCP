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

from PIL import ImageFilter
from PIL.Image import Image, Transpose

from fusion_vision_mcp import geometry, layout, question
from fusion_vision_mcp.constants import (
    BUSY_BACKGROUND_QUESTION,
    CAPTION_IMAGE_TYPES,
    CAPTION_TEXT_WORDS,
    CLASSIFY_MAX_LABELS,
    CLASSIFY_TEMPLATE,
    COUNT_BOX_THRESHOLD,
    COUNT_CHECK_COUNT,
    COUNT_CHECK_PRESENCE,
    COUNT_CHECK_WEAK,
    COUNT_RANGE_VETO,
    CROWD_RANGE_SAM3_FLOOR,
    CROWD_RANGE_SHOW,
    DEEP_CROPS,
    DEEP_CROPS_TEXT,
    DEEP_TEXT_WORDS,
    DETECT_BOX_THRESHOLD,
    MAIN_SUBJECT_QUESTION,
    OCR_LFM_MAX_NEW_TOKENS,
    OCR_LFM_MIN_CHARS,
    OCR_LFM_NO_TEXT,
    OCR_LFM_PROMPT,
    OCR_SHARPEN_MIN_CHARS,
    OCR_SHARPEN_PERCENT,
    OCR_SHARPEN_RADIUS,
    VQA_COUNT_QUESTION,
    VQA_OCR_WORDS,
    VQA_TEXT_QUESTION,
    VQA_TEXT_SUFFIX,
    CaptionLevel,
)
from fusion_vision_mcp.detection_policy import execute_detection
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


#: A silhouette split into several lobes counts as a collapse only when that reading held
#: over at least this share of the distance-transform level sweep (`distance_support`).
_MIN_SPLIT_SUPPORT: Final[float] = 0.5


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
        if by_distance <= 1:
            return "yes"
        # A split that held over only a sliver of the level sweep is a narrow point in
        # one outline -- a person's neck, a bird's wing -- not evidence of a collapse. On
        # the COCO holdout (2026-10-03) every "no" from this path was a true singleton
        # (8 of 8, support 0.08-0.24), while the real collapses all came through the
        # rosette path below. Weak evidence says neither yes nor no.
        support = silhouette.get("distance_support")
        if isinstance(support, (int, float)) and not math.isnan(support) and support < _MIN_SPLIT_SUPPORT:
            return "unknown"
        return "no"

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


def _inside_share(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box a's area that lies inside box b."""
    area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    if area <= 0:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / area


def drop_nested(result: dict[str, Any]) -> dict[str, Any]:
    """`result` without boxes that are a part or a duplicate of another counted box (`geometry.nested_indices`).

    Applied to every count, both paths, before `possibly_more`; `nested_boxes_dropped` says how many went.
    """
    bboxes = result.get("bboxes") or []
    drop = geometry.nested_indices(bboxes, result.get("scores"))
    keep = [i for i in range(len(bboxes)) if i not in drop]
    for field in ("bboxes", "points", "labels", "scores"):
        if isinstance(result.get(field), list) and len(result[field]) == len(bboxes):
            result[field] = [result[field][i] for i in keep]
    if bboxes:
        result["count"] = len(keep)
    result["nested_boxes_dropped"] = len(drop)
    return result


def distinct_candidates(
    counted: Sequence[Sequence[float]], candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The candidates `possibly_more` accepts, in order: each neither on (IoU) nor inside a counted or accepted box."""
    kept = [list(b) for b in counted]
    accepted = []
    for c in candidates:
        box = list(c["box"])
        if all(geometry.box_iou(box, k) < COUNT_RANGE_VETO and _inside_share(box, k) < COUNT_RANGE_VETO for k in kept):
            kept.append(box)
            accepted.append(c)
    return accepted


def possibly_more(counted: Sequence[Sequence[float]], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How many more instances the boxes just under the count threshold could add, as a hedged hint.

    Candidates come highest score first; each is accepted unless it overlaps (IoU) or sits inside a
    counted or already-accepted box at `COUNT_RANGE_VETO` or more, so a duplicate box or a part of
    an object is not read as another instance. `count` itself is never changed.
    """
    extra = len(distinct_candidates(counted, candidates))
    return {
        "upper": len(counted) + extra,
        "extra_candidates": extra,
        "reliability": (
            "a hint, not a tally: on 953 photo counts, when upper > count the true count was "
            "higher than count 38-58% of the time; when upper == count it was too low only 7-14% of "
            "the time (it can still be too high). Measured at the default threshold"
        ),
    }


def crowd_range(app: AppContext, image: Image, object_name: str, count: int) -> dict[str, Any] | None:
    """`[low, high]` from Grounding DINO's `count` (runs low on crowds) and SAM 3's (runs high), as a hedged hint.

    None when neither reaches `CROWD_RANGE_SHOW`, or when SAM 3 is not available (gated on Hugging Face).
    """
    locator = getattr(app, "locator", None)
    if locator is None:
        return None
    try:
        sam3_count = int(locator.detect_objects([image], object_name, threshold=CROWD_RANGE_SAM3_FLOOR)[0]["count"])
    except OSError as exc:
        logger.warning(f"SAM 3 unavailable ({exc}); count_objects has no crowd_range")
        return None
    low, high = min(count, sam3_count), max(count, sam3_count)
    if high < CROWD_RANGE_SHOW:
        return None
    return {
        "low": low,
        "high": high,
        "sam3_count": sam3_count,
        "reliability": (
            "a hint, not a tally: on 92 fresh photos of 4-30 objects the true count lay inside "
            "[low, high] 85.9% of the time (possibly_more's range: 65.2%); count itself tends to "
            "run low on crowds and sam3_count high"
        ),
    }


_COUNT_WORDS: Final[dict[str, int]] = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
])}  # fmt: skip


def _reply_count(reply: str) -> int | None:
    """The gate's reading of a count reply: the first digits, else the first number word ("no"/"none" are 0)."""
    match = re.search(r"\d+", reply)
    if match:
        return int(match.group())
    for word in re.findall(r"[a-z]+", reply.lower()):
        if word in _COUNT_WORDS:
            return _COUNT_WORDS[word]
        if word in ("no", "none"):
            return 0
    return None


def _reply_word(reply: str) -> str:
    """The first word of a yes/no reply, lower-cased with punctuation removed."""
    words = re.sub(r"[^a-z ]", "", reply.strip().lower()).split()
    return words[0] if words else ""


def lfm_count_check(
    app: AppContext, image: Image, object_name: str, result: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> None:
    """Let LFM2.5-VL change a photo count of 1 to 0, or 0 to 1, when both its answers say so (counting plan Gate 2).

    A count of 1 whose box scores under `COUNT_CHECK_WEAK` becomes 0 when LFM answers "no" to `COUNT_CHECK_PRESENCE`
    (the box moves to `vetoed_boxes`). A count of 0 with a distinct candidate under the threshold becomes 1 (the best
    such candidate) when `COUNT_CHECK_COUNT` gets 1 or more *and* the presence question "yes". Nothing else is ever
    changed. Sets `detector_count` and `count_basis`; `count_check` carries the replies, or why LFM was not asked.
    """
    count = int(result.get("count") or 0)
    result["detector_count"] = count
    result["count_basis"] = "detector"
    scores = result.get("scores") or []
    veto = count == 1 and len(scores) == 1 and scores[0] < COUNT_CHECK_WEAK
    extras = distinct_candidates([], candidates) if count == 0 else []
    if app.answerer is None or not (veto or extras):
        return
    answerer = app.answerer
    replies: dict[str, str] = {}

    def ask(kind: str, prompt: str) -> str:
        replies[kind] = answerer.answer([image], [prompt.format(name=object_name)])[0]
        return replies[kind]

    try:
        if veto:
            if _reply_word(ask("presence", COUNT_CHECK_PRESENCE)) == "no":
                result["vetoed_boxes"] = [{"box": result["bboxes"][0], "score": scores[0]}]
                for field in ("bboxes", "points", "labels", "scores"):
                    result[field] = []
                result["count"] = 0
                result["count_basis"] = "presence_veto"
        else:
            k = _reply_count(ask("count", COUNT_CHECK_COUNT))
            if k is not None and k >= 1 and _reply_word(ask("presence", COUNT_CHECK_PRESENCE)) == "yes":
                box, score = list(extras[0]["box"]), float(extras[0]["score"])
                result.update(
                    count=1,
                    bboxes=[box],
                    points=[[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]],
                    labels=[object_name],
                    scores=[score],
                    count_basis="presence_rescue",
                )
    except (OSError, ImportError, MemoryError) as exc:
        logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); count_objects keeps the detector's count")
        result["count_check"] = {"skipped": f"LFM2.5-VL unavailable ({exc}); the detector's count stands"}
        return
    result["count_check"] = {"replies": replies, "model": "LFM2.5-VL-3B"}


def lfm_flat_check(app: AppContext, image: Image, object_name: str, result: dict[str, Any]) -> None:
    """Let LFM2.5-VL zero a Florence-2 (flat-image) count when it says the object is not there (counting plan Gate 5).

    Any count of 1 or more becomes 0 when LFM answers "no" to `COUNT_CHECK_PRESENCE` (every box moves to
    `vetoed_boxes`). Nothing else is ever changed: a zero is never rescued and a count is never moved by less.
    Sets `detector_count` and `count_basis`; `count_check` carries the reply, or why LFM was not asked.
    """
    count = int(result.get("count") or 0)
    result["detector_count"] = count
    result["count_basis"] = "detector"
    if app.answerer is None or count < 1:
        return
    try:
        reply = app.answerer.answer([image], [COUNT_CHECK_PRESENCE.format(name=object_name)])[0]
    except (OSError, ImportError, MemoryError) as exc:
        logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); count_objects keeps the detector's count")
        result["count_check"] = {"skipped": f"LFM2.5-VL unavailable ({exc}); the detector's count stands"}
        return
    if _reply_word(reply) == "no":
        result["vetoed_boxes"] = [{"box": box} for box in result.get("bboxes") or []]
        for field in ("bboxes", "points", "labels"):
            result[field] = []
        result["count"] = 0
        result["count_basis"] = "presence_veto"
    result["count_check"] = {"replies": {"presence": reply}, "model": "LFM2.5-VL-3B"}


def vqa_count_estimate(app: AppContext, image: Image, object_name: str) -> dict[str, Any]:
    """`count_objects`' `estimates.vqa`: LFM2.5-VL-3B asked `COUNT_CHECK_COUNT` (counting plan step 2D); Moondream2 with
    `VQA_COUNT_QUESTION` without an answerer or when LFM cannot load. Parsed with `_first_int`, as the gate scored it."""
    if app.answerer is not None:
        try:
            reply = app.answerer.answer([image], [COUNT_CHECK_COUNT.format(name=object_name)])[0]
            return {"value": _first_int(reply), "raw": reply, "note": "an LFM2.5-VL-3B judgment estimate, not a tally"}
        except (OSError, ImportError, MemoryError) as exc:
            logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); estimating with Moondream2")
    reply = app.vqa.query([image], VQA_COUNT_QUESTION.format(name=object_name))[0]
    return {"value": _first_int(reply), "raw": reply, "note": "a Moondream2 judgment estimate, not a tally"}


def clean_labels(labels: Sequence[str]) -> list[str]:
    """The caller's labels, stripped and de-duplicated in order; at least 2 and at most `CLASSIFY_MAX_LABELS`."""
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        text = label.strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    if len(out) < 2:
        raise ValueError("classify_image needs at least 2 distinct, non-empty labels to choose between")
    if len(out) > CLASSIFY_MAX_LABELS:
        raise ValueError(f"classify_image takes at most {CLASSIFY_MAX_LABELS} labels, got {len(out)}")
    return out


def classify_pages(app: AppContext, images: list[Image], labels: list[str]) -> list[dict[str, Any]]:
    """CLIP's pick among `labels` for each page, with every label's probability, highest first."""
    rows = app.aesthetic.zero_shot(images, [CLASSIFY_TEMPLATE.format(label) for label in labels])
    results = []
    for row in rows:
        order = sorted(zip(labels, row, strict=True), key=lambda pair: -pair[1])
        ranking = [{"label": label, "probability": round(p, 4)} for label, p in order]
        results.append({"label": order[0][0], "probability": round(order[0][1], 4), "ranking": ranking})
    return results


def busy_background(app: AppContext, image: Image) -> dict[str, Any]:
    """Moondream2's yes/no on whether the background competes with the subject, as a hedged hint.

    `busy` is None when the answer is neither yes nor no. The reliability is stated in the
    result because a caller reads that, not this docstring.
    """
    answer = app.vqa.query([image], BUSY_BACKGROUND_QUESTION)[0].strip().lower()
    busy = True if answer.startswith("yes") else False if answer.startswith("no") else None
    return {
        "busy": busy,
        "reliability": (
            "a hint, not a measurement: Moondream2's yes/no agreed with a person's blind calls "
            "on 69% of 52 held-out photos"
        ),
    }


#: A named subject box covering at least this share of the frame defers to the region pick.
_WHOLE_FRAME: Final[float] = 0.95


def name_main_subject(app: AppContext, image: Image) -> str:
    """Moondream2's short name for the photo's main subject, or "" when it says there is none.

    `critique_composition` boxes this name with Florence-2 when the caller gives no
    `target_subject`. VisionBench `composition_critique.py`, against a labeller's boxes: IoU >= 0.5
    on 79.5% / 82.0% / 84.4% of three photo sets (50, 89, 96 photos) where Florence-2's largest,
    most central dense region managed 64.1% / 66.3% / 75.0%.
    """
    name = app.vqa.query([image], MAIN_SUBJECT_QUESTION)[0].strip().strip(".").strip()
    return "" if name.lower().startswith("none") else name


def _critique_one(
    app: AppContext,
    image: Image,
    target_subject: str,
    low_score_threshold: float,
    style_context: bool,
    check_background: bool = True,
) -> dict[str, Any]:
    """Single-image composition critique -- the body of `critique_composition`.

    Extracted so `critique_composition`'s relative mode can run it for both the
    image and the reference. See that tool's docstring for the return shape.
    """
    subject = target_subject or name_main_subject(app, image)
    box: list[int] | None = None
    if subject:
        detected = app.florence2.detect_objects([image], subject)[0]
        box = [int(v) for v in detected["bboxes"][0]] if detected["bboxes"] else None
    # A named box over (nearly) the whole frame usually means the "subject" is the scene itself (a
    # blank image comes back as "a plain gray background"), so the region pick decides instead. On
    # the three VisionBench sets this never cost a hit and gained two.
    whole_frame = box is not None and geometry.box_area(box) >= _WHOLE_FRAME * image.width * image.height
    if not target_subject and (box is None or whole_frame):
        box = _pick_primary_subject(app.florence2.dense_region_caption([image])[0], (image.width, image.height))
        subject = ""  # the box is no longer the named subject's

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
        if check_background:
            result["background"] = busy_background(app, image)
        return result

    result = {
        "image_size": [image.width, image.height],
        "subject": subject or None,
        "subject_box": box,
        "aesthetics": aesthetics,
        "framing": geometry.rule_of_thirds(box, (image.width, image.height)),
    }
    if style is not None:
        result["style"] = style["style"]
        result["style_distribution"] = style["distribution"]
    if check_background:
        result["background"] = busy_background(app, image)
    if aesthetics["score"] < low_score_threshold:
        result["critique"] = app.vqa.query(
            [image], "Critique this photo's composition and framing in 2 concise sentences."
        )[0]
    return result


#: A page reads as sideways when at least this many of its regions are tall...
_SIDEWAYS_MIN_REGIONS: Final[int] = 3
#: ...and they are at least this share of the regions counted.
_SIDEWAYS_SHARE: Final[float] = 0.6
#: A region is tall when its box is this many times taller than wide. RapidOCR reads a
#: crop this shape turned on its side, and reports the box as found, so the regions of
#: an ordinary read already say which way the page faces.
_TALL_RATIO: Final[float] = 1.5
#: Regions shorter than this are not counted: a lone "1" or "l" is tall upright too.
_SIDEWAYS_MIN_CHARS: Final[int] = 3


def _read_page(app: AppContext, image: Image) -> list[dict[str, Any]]:
    """One page's regions in reading order, boxes in page coordinates."""
    regions: list[dict[str, Any]] = []
    for crop, left, top in layout.reading_blocks(image):
        for region in app.ocr_engine.readtext(crop):
            x1, y1, x2, y2 = region["box"]
            box = [x1 + left, y1 + top, x2 + left, y2 + top]
            regions.append({"text": region["text"], "confidence": region["confidence"], "box": box})
    return regions


def _looks_sideways(regions: list[dict[str, Any]]) -> bool:
    counted = [r["box"] for r in regions if len(r["text"].strip()) >= _SIDEWAYS_MIN_CHARS]
    tall = sum(int((y2 - y1) > _TALL_RATIO * (x2 - x1)) for x1, y1, x2, y2 in counted)
    return tall >= _SIDEWAYS_MIN_REGIONS and tall >= _SIDEWAYS_SHARE * len(counted)


def _unrotate_box(box: list[int], turn: Transpose, width: int, height: int) -> list[int]:
    """Maps a box read on `image.transpose(turn)` back onto the `width` x `height` page."""
    x1, y1, x2, y2 = box
    if turn is Transpose.ROTATE_90:  # counter-clockwise: page (x, y) -> (y, width - 1 - x)
        return [width - 1 - y2, x1, width - 1 - y1, x2]
    # ROTATE_270, clockwise: page (x, y) -> (height - 1 - y, x)
    return [y1, height - 1 - x2, y2, height - 1 - x1]


def _read_upright(app: AppContext, image: Image) -> list[dict[str, Any]]:
    """Reads the page, and again turned a quarter each way if its text runs sideways.

    VisionBench phase 13 (2026-10-01): turning images 90 degrees cost 26 points of
    key-phrase recall. RapidOCR reads a sideways line crop by turning it, but a whole
    sideways page also defeats column detection, and on a turned article it kept 1,096
    of the 2,178 characters it read upright. The turn that reads the most confident
    characters wins, and its boxes are mapped back onto the page as sent. An upright
    page costs nothing extra: the check reads the regions it already has.
    """
    regions = _read_page(app, image)
    if not _looks_sideways(regions):
        return regions

    def chars(found: list[dict[str, Any]]) -> int:
        return sum(len(r["text"].strip()) for r in found)

    best = regions
    for turn in (Transpose.ROTATE_90, Transpose.ROTATE_270):
        turned = _read_page(app, image.transpose(turn))
        if chars(turned) > chars(best):
            best = [{**r, "box": _unrotate_box(r["box"], turn, image.width, image.height)} for r in turned]
    return best


def ocr_pages(
    app: AppContext, images: Sequence[Image], *, sharpened: bool = True, lfm_read: bool = True
) -> list[dict[str, Any]]:
    """Transcribe each page with `app.ocr_engine`, splitting side-by-side columns first.

    This is the `ocr` tool's whole implementation, factored out so the other two
    places that transcribe text cannot drift from it. Both used to call
    `app.florence2.ocr` -- the head EasyOCR replaced -- so the same named operation
    returned different text depending on how it was reached, and on a text-free
    image the Florence-2 path invented some. `tests/test_server.py::
    test_ocr_textless_image_returns_no_text` pins the correct behaviour for the
    direct tool; routing everything through here extends it to the other two.

    Returns one `{"text", "text_regions"}` dict per page. Boxes are in the
    coordinates of the page as sent, including a page read turned upright.

    `sharpened=False` skips the sharpened second reading (OCR_SHARPEN_RADIUS). `caption`'s text line and
    `query_image`'s OCR context pass it: it changes the text on ~60% of sharp photos, and their gates measured the
    first reading alone.

    `lfm_read=False` skips LFM2.5-VL's transcription (OCR_LFM_PROMPT); the same two callers pass it, for the same reason.
    """
    pages: list[dict[str, Any]] = []
    for image in images:
        regions = _read_upright(app, image)
        if sharpened:
            regions += _sharpened_extra(app, image, "\n".join(r["text"] for r in regions))
        if lfm_read:
            regions += _lfm_extra(app, image, "\n".join(r["text"] for r in regions))
        pages.append({"text": "\n".join(r["text"] for r in regions), "text_regions": regions})
    return pages


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _sharpened_extra(app: AppContext, image: Image, text: str) -> list[dict[str, Any]]:
    """Spans a whole-page read of the unsharp-masked page finds that `text` lacks, top to bottom (see
    OCR_SHARPEN_RADIUS). Only for an engine that sets `sharpened_pass` (PP-OCRv6): EasyOCR was not measured."""
    if not getattr(app.ocr_engine, "sharpened_pass", False):
        return []
    sharp = image.convert("RGB").filter(ImageFilter.UnsharpMask(OCR_SHARPEN_RADIUS, OCR_SHARPEN_PERCENT, 0))
    have = _alnum(text)
    extra = [
        r
        for r in app.ocr_engine.readtext(sharp)
        if len(key := _alnum(r["text"])) >= OCR_SHARPEN_MIN_CHARS and key not in have
    ]
    return sorted(extra, key=lambda r: (r["box"][1], r["box"][0]))


_LFM_NO_TEXT_RE: Final = re.compile(OCR_LFM_NO_TEXT)


def _lfm_extra(app: AppContext, image: Image, text: str) -> list[dict[str, Any]]:
    """Lines LFM2.5-VL's transcription of the page adds to `text`, in its order (see OCR_LFM_PROMPT). Only when `text`
    is non-empty, for an engine that sets `sharpened_pass` (PP-OCRv6), with an answerer. The lines carry no box or
    confidence (`box` and `confidence` are None, `source` "lfm")."""
    answerer = getattr(app, "answerer", None)
    if not text.strip() or answerer is None or not getattr(app.ocr_engine, "sharpened_pass", False):
        return []
    try:
        reply = answerer.answer([image], [OCR_LFM_PROMPT], max_new_tokens=OCR_LFM_MAX_NEW_TOKENS)[0]
    except (OSError, ImportError, MemoryError) as exc:
        logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); ocr returns PP-OCRv6's reading alone")
        return []
    have = _alnum(text)
    extra = []
    for line in reply.splitlines():
        key = re.sub(r"[^0-9a-z ]+", "", line.lower()).strip()
        if not key or _LFM_NO_TEXT_RE.match(key):
            continue
        if len(norm := _alnum(line)) >= OCR_LFM_MIN_CHARS and norm not in have:
            extra.append({"text": line.strip(), "confidence": None, "box": None, "source": "lfm"})
    return extra


def image_type_lines(app: AppContext, images: Sequence[Image]) -> list[str]:
    """`Image type: <x>.` per image: CLIP's top pick among CAPTION_IMAGE_TYPES (VisionBench `caption_type_line.py`)."""
    prompts = [("an " if t[0].lower() in "aeiou" else "a ") + t for t in CAPTION_IMAGE_TYPES]
    rows = app.aesthetic.zero_shot(list(images), prompts)
    return [f"Image type: {CAPTION_IMAGE_TYPES[max(range(len(row)), key=row.__getitem__)]}." for row in rows]


_TEXT_QUESTION_RE: Final = re.compile(VQA_TEXT_QUESTION, re.IGNORECASE)


def lfm_vqa_prompt(question: str, ocr_text: str) -> str:
    """The prompt LFM gets for a text question: the `ocr` reading in front, VQA_TEXT_SUFFIX after (VisionBench
    `vqa_premise.ocr_prompt`, which the gate measured, byte for byte)."""
    words = " / ".join(line.strip() for line in ocr_text.splitlines() if line.strip()).split()
    if not words:
        return "An OCR pass found no readable text in this image.\n" + question + VQA_TEXT_SUFFIX
    quoted = " ".join(words[:VQA_OCR_WORDS])
    return (
        f'Text read from the image by OCR (it may be incomplete or misread): "{quoted}"\n' + question + VQA_TEXT_SUFFIX
    )


def vqa_answers(app: AppContext, images: Sequence[Image], question: str) -> list[str]:
    """`query_image`'s answers: LFM2.5-VL-3B ONNX (`app.answerer`), a text question carrying the `ocr` reading; Moondream2
    (`app.vqa`) without an answerer or when LFM cannot load."""
    if app.answerer is not None:
        try:
            if _TEXT_QUESTION_RE.search(question):
                prompts = [
                    lfm_vqa_prompt(question, p["text"]) for p in ocr_pages(app, images, sharpened=False, lfm_read=False)
                ]
            else:
                prompts = [question] * len(images)
            return app.answerer.answer(list(images), prompts)
        except (OSError, ImportError) as exc:
            logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); answering with Moondream2")
    return app.vqa.query(list(images), question)


def _text_line_key(line: str) -> str:
    return re.sub(r"\W+", " ", line.lower()).strip()


def deep_crops(text: str) -> int:
    """Moondream2's crop cap for `caption(deep=true)` on a page whose `ocr` text is `text` (see DEEP_CROPS)."""
    return DEEP_CROPS_TEXT if len(text.split()) >= DEEP_TEXT_WORDS else DEEP_CROPS


def visible_text_line(text: str, max_words: int = CAPTION_TEXT_WORDS) -> str:
    """The `ocr` engine's text as one caption sentence, or "" for a text-free page.

    Lines keep their reading order, joined with " / " and cut at `max_words` words (each
    separator counts as one), and are quoted so a caller can tell verbatim text from the
    captioner's own words. A line is dropped if, lower-cased with punctuation as spaces, it
    is under 2 characters or repeats an earlier line. A cut line says how many words it left
    out, since `ocr` has them all.
    """
    lines, seen = [], set()
    for line in text.splitlines():
        key = _text_line_key(line)
        if len(key.replace(" ", "")) < 2 or key in seen:
            continue
        seen.add(key)
        lines.append(line.strip())
    if not lines:
        return ""
    words = " / ".join(lines).split()
    shown = " ".join(words[:max_words]).rstrip(" /")
    cut = sum(word != "/" for word in words[max_words:])
    if cut:
        return f'Visible text: "{shown}…" ({cut} more words not shown; ocr returns the full text).'
    return f'Visible text: "{shown}".'


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
        # Same policy as the `detect_objects` tool, or a batch returns different boxes.
        return execute_detection(
            app,
            images,
            object_name,
            backend_preference="auto",
            threshold=DETECT_BOX_THRESHOLD,
            adaptive_threshold=False,
            locate=True,
        )
    if operation == "dense_caption":
        return app.florence2.dense_region_caption(images)
    if operation == "query":
        if not question:
            raise ValueError("question is required for the 'query' operation")
        return vqa_answers(app, images, question)
    if operation == "count":
        if not object_name:
            raise ValueError("object_name is required for the 'count' operation")
        # No silhouette check here: batching is the throughput path, and the check
        # would pull SAM2 in behind the caller's back once per collapsed image.
        # Same floor as the `count_objects` tool, or a batch counts differently.
        return [drop_nested(r) for r in app.counter.detect_objects(images, object_name, threshold=COUNT_BOX_THRESHOLD)]
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
