"""SAM 3's post-processing, without weights: the steps the sprint R gates replayed."""

import pytest

pytest.importorskip("transformers", minversion="5")
from fusion_vision_mcp.sam3 import Sam3Detector

post = Sam3Detector.postprocess


def test_score_floor_drops_low_boxes() -> None:
    r = post([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.1], "mug", 0.125)
    assert r["bboxes"] == [[0, 0, 10, 10]]
    assert r["scores"] == [0.9]
    assert r["count"] == 1


def test_nms_keeps_the_higher_scoring_of_two_overlapping_boxes() -> None:
    r = post([[0, 0, 10, 10], [0, 0, 10, 11], [40, 40, 50, 50]], [0.9, 0.8, 0.7], "mug", 0.125)
    assert r["bboxes"] == [[0, 0, 10, 10], [40, 40, 50, 50]]
    assert r["scores"] == [0.9, 0.7]


def test_envelope_box_around_the_instances_is_dropped() -> None:
    inst = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    r = post([[0, 0, 50, 10], *inst], [0.95, 0.9, 0.9, 0.9], "mug", 0.125)
    assert r["group_boxes_dropped"] == 1
    assert sorted(r["bboxes"]) == inst
    assert r["points"] == [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in r["bboxes"]]


def test_nothing_above_the_floor_is_an_empty_answer() -> None:
    r = post([[0, 0, 10, 10]], [0.05], "unicorn", 0.125)
    assert r == {"count": 0, "bboxes": [], "points": [], "labels": [], "scores": [], "group_boxes_dropped": 0}
