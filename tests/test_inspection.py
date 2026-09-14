"""Synthetic unit tests for `_check_anatomy`'s association, dedup and threshold logic.

These tests exercise the per-person association, the duplicate-box merge and the anomaly
arithmetic in isolation, using a `_StubDetector` that returns hand-placed synthetic bounding
boxes rather than running any real detection. They can never see a real image and therefore can
never catch a failure that only shows up on real detector output -- see
`tests/test_inspection_coco.py` for that (a live, unmocked Grounding DINO regression test against
real COCO photographs), added in Sprint 17 of `FusionVisionMCP_v0.8.1_Sprint_Plan.md` after this
exact gap let a hardcoded `corroborated=True` and an untested global-tally heuristic ship in
Sprint 10.

Before Sprint 18 the stub returned bare integers with no boxes at all, so the tests asserted only
that `10 > 2 * 2 + 1` evaluates true. Synthetic boxes are what let them say something about the
logic that actually decides an anomaly: that a leg outside every person box is not somebody's
third leg, and that one limb detected three times is one limb.
"""

import numpy as np
from PIL import Image

from fusion_vision_mcp import geometry
from fusion_vision_mcp.inspection import _containment, analyze_inspection


class _StubDetector:
    """Returns pre-placed synthetic boxes per part name, in `detect_objects`' real result shape."""

    def __init__(self, boxes: dict[str, list[list[float]]], scores: dict[str, list[float]] | None = None) -> None:
        self.boxes = boxes
        self.scores = scores or {}

    def detect_objects(self, images: list, object_name: str, **kwargs) -> list[dict]:
        boxes = self.boxes.get(object_name, [])
        scores = self.scores.get(object_name) or [0.5] * len(boxes)
        return [
            {
                "count": len(boxes),
                "bboxes": boxes,
                "points": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes],
                "labels": [object_name] * len(boxes),
                "scores": scores,
            }
        ]


class _StubApp:
    def __init__(self, counter) -> None:
        self.counter = counter


def _image() -> Image.Image:
    return Image.new("RGB", (400, 300))


#: One person, standing in the left third of a 400x300 frame.
PERSON = [0.0, 0.0, 100.0, 200.0]


def _inside(n: int, part_width: float = 20.0) -> list[list[float]]:
    """`n` disjoint boxes wholly inside `PERSON`, spaced evenly down its height."""
    step = 190.0 / max(n, 1)
    height = min(step * 0.6, 20.0)
    return [[10.0, 5.0 + step * i, 10.0 + part_width, 5.0 + step * i + height] for i in range(n)]


def _outside(n: int) -> list[list[float]]:
    """`n` disjoint boxes wholly outside `PERSON` -- the horse's legs, the cropped bystander."""
    return [[200.0 + 30.0 * i, 10.0, 220.0 + 30.0 * i, 60.0] for i in range(n)]


def test_stub_arithmetic_analyze_inspection_no_person():
    app = _StubApp(_StubDetector({"person": [], "arm": _inside(10)}))
    obs, anomalies = analyze_inspection(app, _image(), [])
    assert len(obs) == 0
    assert len(anomalies) == 0


def test_stub_arithmetic_analyze_inspection_normal_person():
    app = _StubApp(
        _StubDetector(
            {
                "person": [PERSON],
                "arm": _inside(2),
                "leg": _inside(2),
                "head": _inside(1),
                "hand": _inside(2),
                "finger": _inside(10, part_width=4.0),
            }
        )
    )
    obs, anomalies = analyze_inspection(app, _image(), ["A person."])
    assert len(anomalies) == 0
    # There should be 1 uncorroborated VLM claim and no coverage observation (nothing was orphaned).
    assert len(obs) == 1
    assert obs[0].source == "vqa"


def test_stub_arithmetic_parts_outside_every_person_box_are_coverage_evidence_not_anatomy():
    """The horse-leg case: four legs belonging to nobody detected must not become a third leg.

    This is the single largest source of the 9 false anomalies the pre-association check raised on
    real COCO photographs (`benchmarks/results/inspection_coco_negative_control.csv`): a frame-wide
    tally of 6 legs against 1 detected person cleared `persons * 2 + 1` on a photograph of a person
    next to a horse.
    """
    app = _StubApp(_StubDetector({"person": [PERSON], "leg": _inside(2) + _outside(4)}))
    obs, anomalies = analyze_inspection(app, _image(), [])
    assert anomalies == []

    coverage = [o for o in obs if o.source == "detector"]
    assert len(coverage) == 1
    assert coverage[0].corroborated is False
    assert "unassociated_count: 4" in coverage[0].claim
    assert any("4 leg detection(s) matched no detected person box" in line for line in coverage[0].evidence)


def test_stub_arithmetic_one_limb_detected_several_times_is_one_limb():
    """A bent arm yields upper-arm + forearm + whole-arm boxes; undeduped that is three arms."""
    real_arm, second_arm = _inside(2)
    # Four near-identical re-detections of `real_arm` at slightly different scales (IoU well above
    # the dedup cutoff), plus one genuinely separate arm: six raw boxes, two real arms.
    duplicates = [[real_arm[0] + d, real_arm[1] + d, real_arm[2] + d, real_arm[3] + d] for d in (0.0, 1.0, 2.0, 3.0)]
    app = _StubApp(_StubDetector({"person": [PERSON], "arm": [*duplicates, second_arm]}))
    obs, anomalies = analyze_inspection(app, _image(), [])
    assert anomalies == [], "duplicate re-detections of one arm were counted as separate arms"
    assert obs == []


def test_stub_arithmetic_a_person_with_too_many_arms_is_still_flagged_against_that_person():
    """The positive control: over-correcting until nothing ever flags is the failure mode here."""
    app = _StubApp(_StubDetector({"person": [PERSON], "arm": _inside(4)}))
    obs, anomalies = analyze_inspection(app, _image(), ["None"])

    assert len(anomalies) == 1
    assert "person 1 of 1" in anomalies[0].description.lower()
    assert "4 arms" in anomalies[0].description

    det_obs = [o for o in obs if o.source == "detector"]
    assert len(det_obs) == 1
    assert det_obs[0].corroborated is True
    assert any("Person box: [0, 0, 100, 200]" == line for line in det_obs[0].evidence)

    # The "None" VQA claim is still filtered out as a default answer.
    assert [o for o in obs if o.source == "vqa"] == []


def test_stub_arithmetic_anomaly_names_the_offending_person_not_the_frame():
    """Two people, one ordinary and one with four arms: exactly one of them is named.

    A frame-wide tally is structurally incapable of this -- 6 arms for 2 people is under its own
    threshold, so the real artifact was invisible to it in exactly the images it fired on.
    """
    second_person = [200.0, 0.0, 300.0, 200.0]
    normal_arms = [[210.0, 10.0, 230.0, 30.0], [210.0, 60.0, 230.0, 80.0]]
    app = _StubApp(_StubDetector({"person": [PERSON, second_person], "arm": _inside(4) + normal_arms}))
    _obs, anomalies = analyze_inspection(app, _image(), [])

    assert len(anomalies) == 1
    assert "person 1 of 2" in anomalies[0].description.lower()


def test_stub_arithmetic_a_limb_reaching_past_its_person_box_is_still_that_person_s():
    """Association is a fraction, not full containment: person boxes clip outstretched limbs."""
    # 60% of this box's area lies inside PERSON (x from 88 to 100 of 88..108), above the cutoff.
    reaching = [[88.0, 40.0, 108.0, 60.0]]
    mostly_out = [[96.0, 100.0, 116.0, 120.0]]  # 20% inside, below the cutoff
    app = _StubApp(_StubDetector({"person": [PERSON], "arm": _inside(3) + reaching}))
    _obs, anomalies = analyze_inspection(app, _image(), [])
    assert len(anomalies) == 1, "a limb 60% inside its person box was not associated with them"

    app = _StubApp(_StubDetector({"person": [PERSON], "arm": _inside(3) + mostly_out}))
    _obs, anomalies = analyze_inspection(app, _image(), [])
    assert anomalies == [], "a box only 20% inside a person box was counted against them"


def test_stub_arithmetic_a_part_inside_two_person_boxes_is_counted_once_against_the_tightest():
    """A loose box around a pair of people must not collect both their limbs as one person's.

    The envelope is the whole frame rather than a snug box around the pair, because a snug one
    would overlap each person box past the dedup cutoff and be merged with it -- which is the
    right behaviour for two views of one body, and moot for a real group envelope, since
    `grounding_dino._envelope_indices` already drops those before `_check_anatomy` sees them.
    """
    pair_envelope = [0.0, 0.0, 400.0, 300.0]
    second_person = [200.0, 0.0, 300.0, 200.0]
    second_arms = [[210.0, 10.0, 230.0, 30.0], [210.0, 60.0, 230.0, 80.0]]
    app = _StubApp(_StubDetector({"person": [pair_envelope, PERSON, second_person], "arm": _inside(2) + second_arms}))
    _obs, anomalies = analyze_inspection(app, _image(), [])
    # Four arms all fall inside the envelope box; attributed to the tightest box containing each,
    # they are two arms apiece and nothing is anomalous.
    assert anomalies == []


def test_stub_arithmetic_box_containment_matches_geometry_relation_on_rasterized_masks():
    """Pins the claim in `_containment`'s docstring: this is `relation`'s `a_inside_b`, not a rival.

    `geometry.relation` is the project's one containment measurement, but it takes boolean masks.
    For two axis-aligned rectangles its `a_inside_b` has a closed form, which is what `_containment`
    computes; rasterizing here proves the two agree rather than asserting it.
    """
    person = [20.0, 30.0, 220.0, 230.0]
    cases = [
        [40.0, 50.0, 100.0, 110.0],  # wholly inside
        [180.0, 50.0, 260.0, 110.0],  # straddling the right edge
        [300.0, 50.0, 340.0, 90.0],  # wholly outside
        [0.0, 0.0, 400.0, 300.0],  # swallowing the person box
    ]

    def raster(box: list[float]) -> np.ndarray:
        mask = np.zeros((300, 400), dtype=bool)
        mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = True
        return mask

    person_mask = raster(person)
    for part in cases:
        measured = geometry.relation(raster(part), person_mask)["a_inside_b"]
        assert abs(measured - _containment(part, person)) < 1e-9


def test_stub_arithmetic_analyze_inspection_records_vqa_claims():
    app = _StubApp(_StubDetector({"person": [PERSON], "arm": _inside(2)}))
    obs, anomalies = analyze_inspection(app, _image(), ["The person is glowing."])

    # "The person is glowing" is not filtered out
    assert len(anomalies) == 0
    assert len(obs) == 1
    assert obs[0].claim == "The person is glowing."
    assert obs[0].corroborated is False
