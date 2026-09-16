from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from PIL import Image

from fusion_vision_mcp import geometry

if TYPE_CHECKING:
    from fusion_vision_mcp.protocols import AppContext


@dataclass
class Observation:
    #: True only where the claim survived the full structural chain: box dedup, strict
    #: association to one person, and a violated anatomical ratio. Bare global tallies
    #: (detection coverage) and raw VQA claims report False. The value is carried by the
    #: control flow that reaches each construction site rather than by a separate
    #: computation -- see `_check_anatomy`.
    claim: str
    source: str
    corroborated: bool
    evidence: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "source": self.source,
            "corroborated": self.corroborated,
            "evidence": self.evidence,
        }


@dataclass
class Anomaly:
    description: str
    observations: list[Observation]

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "observations": [obs.as_dict() for obs in self.observations],
        }


#: Every part queried from the detector. `person` is the frame of reference every other part is
#: attributed to. `hand` carries no ratio of its own -- it never did, in the pre-association check
#: either -- and is kept because it is reported as coverage evidence and because dropping a query
#: would change what the Sprint 17 benchmark records; a hand is in any case implied by an arm, so
#: giving it a ratio would count one limb twice.
_PARTS: Final[tuple[str, ...]] = ("person", "arm", "leg", "hand", "head", "finger")

#: Per part: how many one human body has, the margin allowed on top of it, and the severity of
#: exceeding that. The margin absorbs the detector's residual re-detection of one limb at a
#: second scale that survived dedup, and partial second bodies overlapping a person's own box.
#: These are unchanged from the pre-association heuristic (they were `persons * 2 + 1` and so on
#: for the whole frame); what changed is that they are now applied to one person's own tally
#: rather than to a frame-wide sum, so any improvement measured this sprint is attributable to
#: association and dedup rather than to quietly loosening the ratios.
_PART_ANATOMY: Final[dict[str, tuple[int, int]]] = {
    "arm": (2, 1),
    "leg": (2, 1),
    "head": (1, 1),
    "finger": (10, 2),
}

#: Share of a part box's own area that must fall inside a person box before that part can be
#: counted against that person. Not 1.0: a person box is drawn around the body the detector is
#: confident about, and a limb box frequently reaches past it -- an outstretched arm, a leg
#: crossing the box edge -- so requiring full containment would drop real limbs and make an extra
#: one invisible.
#:
#: Chosen by sweep over 0.3-0.95 against two controls at once (the full table is in the Sprint 18
#: commit message). The negative control is the 24 person-bearing COCO photographs in
#: `benchmarks/coco_annotations.json`, which must raise no anomalies. The positive control injects
#: four extra synthetic `arm` boxes into one detected person box per fixture and requires that
#: person to still be flagged -- in four variants: disjoint boxes, boxes overlapping each other
#: at IoU 0.2 and 0.3 (an extra limb lying against a real one, which an over-eager dedup erases),
#: and boxes only 65% inside the person box (a limb reaching past a tight person box).
#:
#: 0.5 sits in the middle of the flat optimum. 0.45-0.55 all measure identically (1 false image,
#: 24/24 disjoint, 24/24 at overlap 0.2, 22/24 reaching); 0.4 adds a second false image, and 0.6
#: collapses the reaching-limb control from 22/24 to 8/24 -- the cliff that fixes the upper bound.
_PART_CONTAINMENT: Final[float] = 0.5

#: Above this IoU, two same-category part boxes are one physical part detected twice rather than
#: two parts. A bent arm yields upper-arm, forearm and whole-arm detections; nothing merged them
#: before, so one arm counted as three and an ordinary photograph cleared the anomaly threshold.
#:
#: This is deliberately a duplicate-instance merge and not `grounding_dino._envelope_indices`,
#: which solves the neighbouring but different problem of one box drawn around a *group* of real
#: instances. It follows that function's pattern -- a small, documented helper over `_iou` with one
#: swept constant -- rather than reusing it.
#:
#: 0.3 is the *least aggressive* merge (the highest cutoff, merging the fewest boxes, so the least
#: able to hide a real extra limb) that still holds the negative control at its floor of one false
#: image: 0.35 adds a second and 0.4 a third and fourth. Below it, sensitivity falls away with
#: nothing gained -- the positive control's extra limbs overlapping a real one at IoU 0.2 go from
#: 24/24 caught at this cutoff to 7/24 at 0.2 -- so this is a ceiling set by the negative control
#: and a floor set by the positive one, not a single-sided pick.
_DUPLICATE_PART_IOU: Final[float] = 0.3


#: Shared with `grounding_dino` via `geometry`, which is pure numpy and therefore safe to
#: import from this eagerly-loaded module. These were two separate copies until they were
#: consolidated; see `geometry.box_iou`.
_box_area = geometry.box_area
_intersection_area = geometry.box_intersection_area
_iou = geometry.box_iou


def _containment(part: list[float], person: list[float]) -> float:
    """Share of `part`'s own area lying inside `person` -- the box-native form of `a_inside_b`.

    `geometry.relation` already measures containment for `spatial_relations`, and this is that
    same quantity rather than a second notion of it: for two axis-aligned rectangles,
    `relation(a, b)["a_inside_b"]` reduces exactly to intersection area over `a`'s area, which is
    what this returns. `tests/test_inspection.py` pins that equivalence by rasterizing boxes into
    masks and comparing the two, so the claim is checked rather than asserted.

    What is avoided by computing it in closed form is the rasterization: `relation` takes boolean
    masks, so reusing it literally would mean filling a full-image array per box and running a
    distance transform per pair. Measured on a 640x480 frame, one `relation` call on two
    rasterized rectangles costs 25.4 ms against 0.3 us for this arithmetic, and the pairing is
    every part box against every person box -- one COCO fixture here detects 39 person boxes and
    13 part boxes, 507 pairs, which is 13 s of array work per image for numbers identical to these.
    """
    area = _box_area(part)
    return _intersection_area(part, person) / area if area > 0 else 0.0


def _merge_duplicate_boxes(bboxes: list[list[float]], scores: list[float]) -> list[list[float]]:
    """Collapse same-category boxes that are re-detections of one physical part.

    Greedy, highest-scoring box first, keeping a box only when it overlaps every box already kept
    by less than `_DUPLICATE_PART_IOU` -- the survivor is the detector's own most confident view of
    that part, so the merged box is a real detection rather than an average of several.
    """
    order = sorted(range(len(bboxes)), key=lambda i: -(scores[i] if i < len(scores) else 0.0))
    kept: list[list[float]] = []
    for index in order:
        box = bboxes[index]
        if all(_iou(box, other) < _DUPLICATE_PART_IOU for other in kept):
            kept.append(box)
    return kept


def _associate(part_boxes: list[list[float]], person_boxes: list[list[float]]) -> tuple[list[int], int]:
    """Attribute each part box to one person box, returning per-person tallies and the leftovers.

    A part goes to the *smallest* person box holding enough of it, and to exactly one. Smallest,
    rather than the box holding the largest share, because in a crowd every person box is nested
    inside some looser one: a limb belongs to the tightest body that can plausibly own it, and the
    loose box around the pair behind it is not making a claim about anyone's anatomy. Measured, the
    difference is not cosmetic -- on the 24 COCO photographs the largest-share rule leaves 2 false
    images to the smallest-box rule's 1, while the positive control's injected extra limbs are
    caught on 24/24 fixtures instead of 21/24, so this rule is better on both controls at once
    rather than trading one against the other.

    Exactly one, because counting a limb against every box that contains it would reinvent the
    frame-wide over-count this function exists to remove.

    A part matching no person box is not evidence about anatomy -- it is a limb whose owner the
    detector did not find, or a leg that belongs to the horse. It is returned separately as a
    count so the caller can see the detector's coverage gap, and it never enters a ratio.
    """
    tallies = [0] * len(person_boxes)
    unassociated = 0
    for part in part_boxes:
        best_index = -1
        best_area = 0.0
        for index, person in enumerate(person_boxes):
            if _containment(part, person) < _PART_CONTAINMENT:
                continue
            area = _box_area(person)
            if best_index < 0 or area < best_area:
                best_index, best_area = index, area
        if best_index < 0:
            unassociated += 1
        else:
            tallies[best_index] += 1
    return tallies, unassociated


def _check_anatomy(app: AppContext, image: Image.Image) -> tuple[list[Observation], list[Anomaly]]:
    """Count body parts against the person each one belongs to, and flag anyone carrying too many.

    A generative artifact looks like *this person has three arms*. The check used to ask a
    question that cannot express that: it summed every `arm` box in the frame and compared the sum
    against the number of detected people. Two things went wrong with it at once on ordinary
    photographs. The numerator collected parts belonging to nobody the detector found -- background
    figures, a body cropped at the frame edge, a horse's legs -- while the denominator counted only
    whole people it was confident about; and one bent arm detected as upper-arm, forearm and whole
    arm counted as three arms. Measured on 24 real COCO photographs of ordinary people, that raised
    9 anomalies across 7 images, none of them corresponding to anything wrong in the image.

    So each part box is now deduplicated, attributed to one person box, and tallied against that
    person's own body. Parts attributed to nobody are reported as a coverage observation and kept
    out of every ratio, because "the detector found a leg and no owner for it" is a statement about
    detection, not about anatomy -- and quietly folding it into a ratio is exactly how the old
    check invented its evidence. Re-measured on the same 24 photographs, that takes 9 anomalies
    across 7 images down to 1 across 1.

    That last one does not go away, and it is worth knowing what it is rather than reading the
    check as clean. On `coco_sheep_4_000000094871` -- a woman petting sheep -- Grounding DINO
    answers the query `leg` with sheep legs, several of which lie inside her person box, so she
    tallies more legs than a person has. No association or dedup can reach it: the boxes really are
    inside her box, and nothing measurable here distinguishes a sheep's leg from hers. Asking the
    detector for `human leg` does fix that image, but it was measured and rejected for this sprint:
    it raises limb recall enough to create *new* false anomalies in crowds (3 new ones across the
    same set with every part qualified), and the one configuration that reaches zero needs the
    containment cutoff dropped to 0.4 as well -- a setting picked by searching the negative control
    alone, with no positive control able to constrain it. See the Sprint 18 commit message.

    `corroborated=True` on the anomaly observations below is not a constant standing in for an
    unwritten check, despite reading like one at the call site: this line is reachable only after a
    part box has survived IoU dedup, been strictly associated with one person box, and exceeded that
    person's anatomical ratio. That is precisely the condition Sprint 19 defined for the flag, and
    it is enforced by the branches above rather than recomputed here. Observations that do *not*
    clear that chain -- coverage tallies, raw VQA claims -- construct with `corroborated=False`.

    One query per part, deliberately -- batching them was measured and rejected. Grounding DINO
    accepts a multi-phrase prompt ("person. arm. leg. hand. head. finger.") in a single forward
    pass and it is ~5x faster (3.2s vs 16.4s on a COCO frame), but the results are not the same
    detections and cannot back a per-part tally:

    - It returns boxes whose text span maps to no single part, reported with an empty label --
      22 of 31 boxes on `000000008021.jpg`, 8 of 12 on `000000017029.jpg`. There is nothing to
      associate those with.
    - Phrases compete inside one prompt, so weak parts vanish: `000000013659.jpg` goes from 35
      detections across six parts to 4, all `person`, losing every limb.

    The containment and dedup constants below were swept against the per-part behaviour, so
    switching would invalidate that tuning as well as the tallies. Measured 2026-09-16.
    """
    detections = {}
    for part in _PARTS:
        result = app.counter.detect_objects([image], part, adaptive_threshold=True)[0]
        detections[part] = (
            [[float(v) for v in box] for box in result.get("bboxes", [])],
            [float(s) for s in result.get("scores", [])],
        )

    person_boxes_raw, person_scores = detections["person"]
    person_boxes = _merge_duplicate_boxes(person_boxes_raw, person_scores)
    if not person_boxes:
        return [], []

    observations: list[Observation] = []
    anomalies: list[Anomaly] = []
    coverage_lines: list[str] = []

    unassociated_count = 0

    for part, (expected, margin) in _PART_ANATOMY.items():
        raw_boxes, scores = detections[part]
        merged = _merge_duplicate_boxes(raw_boxes, scores)
        tallies, unassociated = _associate(merged, person_boxes)
        unassociated_count += unassociated

        if unassociated:
            coverage_lines.append(
                f"{unassociated} {part} detection(s) matched no detected person box "
                f"(of {len(merged)} after dedup, {len(raw_boxes)} raw)"
            )

        for index, count in enumerate(tallies):
            if count <= expected + margin:
                continue
            box = [round(v) for v in person_boxes[index]]
            observation = Observation(
                claim=f"Person {index + 1} of {len(person_boxes)} has {count} {part}s associated with them",
                source="detector",
                corroborated=True,
                evidence=[
                    f"Person box: {box}",
                    f"{part.capitalize()} detections inside that box (>= {_PART_CONTAINMENT:.0%} of each box's area): {count}",
                    f"Expected at most {expected} per person, plus a margin of {margin}",
                    f"{part.capitalize()} detections in frame: {len(merged)} after IoU dedup at {_DUPLICATE_PART_IOU}, {len(raw_boxes)} raw",
                    f"{part.capitalize()} detections associated with no person: {unassociated} (excluded from this ratio)",
                ],
            )
            observations.append(observation)
            anomalies.append(
                Anomaly(
                    description=(
                        f"Structural anomaly: person {index + 1} of {len(person_boxes)} at {box} has "
                        f"{count} {part}s associated with them (expected at most {expected})."
                    ),
                    observations=[observation],
                )
            )

    hand_raw, hand_scores = detections["hand"]
    hand_merged = _merge_duplicate_boxes(hand_raw, hand_scores)
    _hand_tallies, hand_unassociated = _associate(hand_merged, person_boxes)
    unassociated_count += hand_unassociated
    if hand_unassociated:
        coverage_lines.append(
            f"{hand_unassociated} hand detection(s) matched no detected person box "
            f"(of {len(hand_merged)} after dedup, {len(hand_raw)} raw)"
        )

    if coverage_lines:
        observations.append(
            Observation(
                claim=(
                    f"unassociated_count: {unassociated_count} part detection(s) belong to no person the "
                    f"detector found ({len(person_boxes)} person box(es) after dedup, "
                    f"from {len(person_boxes_raw)} raw). Detection coverage, not anatomy: excluded from "
                    f"every ratio above."
                ),
                source="detector",
                corroborated=False,
                evidence=coverage_lines,
            )
        )

    return observations, anomalies


def analyze_inspection(
    app: AppContext, image: Image.Image, vqa_answers: list[str]
) -> tuple[list[Observation], list[Anomaly]]:
    """Perform a structured visual inspection for anomalies (like AI artifacts)."""
    observations, anomalies = _check_anatomy(app, image)

    # We could parse vqa_answers for other claims and corroborate them here.
    # For now, we rely on the structural anatomy checks which catch AI artifacts
    # that the VLM is typically blind to.
    for ans in vqa_answers:
        if ans and ans.lower() not in ("none", "none.", "no", "yes", "nothing", "nothing."):
            observations.append(
                Observation(
                    claim=ans,
                    source="vqa",
                    corroborated=False,
                    evidence=["VLM provided this claim, but it lacks structural corroboration."],
                )
            )

    return observations, anomalies
