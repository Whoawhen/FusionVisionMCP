"""`count_objects`' `possibly_more`: the boxes just under the count threshold, vetoed for duplicates and parts."""

from typing import Any

import torch
from PIL import Image

from fusion_vision_mcp.analysis import possibly_more
from fusion_vision_mcp.grounding_dino import GroundingDino


def _cand(box: list[float], score: float = 0.3) -> dict[str, Any]:
    return {"box": box, "score": score}


def test_a_box_beside_the_counted_one_is_another_possible_instance() -> None:
    out = possibly_more([[0, 0, 10, 10]], [_cand([20, 0, 30, 10])])
    assert out["upper"] == 2 and out["extra_candidates"] == 1
    assert "hint" in out["reliability"]


def test_a_part_inside_a_counted_box_is_not() -> None:
    # A spot on a ball: wholly inside the counted box.
    assert possibly_more([[0, 0, 100, 100]], [_cand([40, 40, 50, 50])])["extra_candidates"] == 0


def test_a_duplicate_of_a_counted_box_is_not() -> None:
    assert possibly_more([[0, 0, 10, 10]], [_cand([1, 0, 11, 10])])["extra_candidates"] == 0


def test_two_candidates_on_the_same_new_object_add_one() -> None:
    out = possibly_more([[0, 0, 10, 10]], [_cand([20, 0, 30, 10], 0.4), _cand([21, 0, 31, 10], 0.3)])
    assert out["upper"] == 2


def test_no_candidates_means_upper_equals_count() -> None:
    out = possibly_more([[0, 0, 10, 10], [20, 0, 30, 10]], [])
    assert out["upper"] == 2 and out["extra_candidates"] == 0


class _Inputs(dict[str, Any]):
    def to(self, _device: Any) -> "_Inputs":
        return self


class _Processor:
    def __init__(self, boxes: list[list[float]], scores: list[float]) -> None:
        self.boxes, self.scores = boxes, scores

    def __call__(self, **_: Any) -> _Inputs:
        return _Inputs(input_ids=torch.zeros(1, 3, dtype=torch.long))

    def post_process_grounded_object_detection(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return [
            {
                "boxes": torch.tensor(self.boxes),
                "scores": torch.tensor(self.scores),
                "text_labels": ["cup"] * len(self.scores),
            }
        ]


def _detector(boxes: list[list[float]], scores: list[float]) -> GroundingDino:
    gd = object.__new__(GroundingDino)
    gd.processor = _Processor(boxes, scores)  # type: ignore[assignment]
    gd.model = lambda **_: None  # type: ignore[assignment]
    gd.device = "cpu"  # type: ignore[assignment]
    return gd


def test_grounding_dino_lists_unreturned_boxes_between_the_floor_and_the_threshold() -> None:
    gd = _detector([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]], [0.9, 0.3, 0.1, 0.2])
    res = gd.detect_objects(
        [Image.new("RGB", (100, 20))], "cup", threshold=0.425, adaptive_threshold=False, candidates_floor=0.18
    )[0]
    assert res["count"] == 1
    assert [round(c["score"], 2) for c in res["candidates_below"]] == [0.3, 0.2]  # 0.1 is under the floor


def test_grounding_dino_omits_candidates_unless_asked() -> None:
    gd = _detector([[0, 0, 10, 10], [20, 0, 30, 10]], [0.9, 0.3])
    res = gd.detect_objects([Image.new("RGB", (100, 20))], "cup", threshold=0.425, adaptive_threshold=False)[0]
    assert "candidates_below" not in res
