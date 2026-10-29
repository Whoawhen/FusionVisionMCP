"""`count_objects`' LFM presence checks (`analysis.lfm_count_check`): a weak single box vetoed, a zero rescued."""

from types import SimpleNamespace
from typing import Any

from fusion_vision_mcp.analysis import lfm_count_check, vqa_count_estimate
from fusion_vision_mcp.constants import COUNT_CHECK_COUNT, VQA_COUNT_QUESTION


class _Answerer:
    def __init__(self, count: str = "", presence: str = "", fail: Exception | None = None) -> None:
        self.count, self.presence, self.fail, self.asked = count, presence, fail, []

    def answer(self, images: list[Any], prompts: list[str]) -> list[str]:
        if self.fail is not None:
            raise self.fail
        kind = "count" if prompts[0].startswith("How many") else "presence"
        self.asked.append(prompts[0])
        return [self.count if kind == "count" else self.presence]


def _one(score: float) -> dict[str, Any]:
    return {"count": 1, "bboxes": [[0, 0, 10, 10]], "points": [[5, 5]], "labels": ["cat"], "scores": [score]}


def _zero() -> dict[str, Any]:
    return {"count": 0, "bboxes": [], "points": [], "labels": [], "scores": []}


_CANDS = [{"box": [20, 0, 40, 10], "score": 0.3}, {"box": [60, 0, 70, 10], "score": 0.2}]


def _check(result: dict[str, Any], answerer: Any, cands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    lfm_count_check(SimpleNamespace(answerer=answerer), None, "cat", result, cands or [])
    return result


def test_a_weak_single_box_is_vetoed_when_the_model_says_no() -> None:
    out = _check(_one(0.5), _Answerer(presence="No."))
    assert out["count"] == 0 and out["bboxes"] == [] and out["scores"] == []
    assert out["detector_count"] == 1 and out["count_basis"] == "presence_veto"
    assert out["vetoed_boxes"] == [{"box": [0, 0, 10, 10], "score": 0.5}]
    assert out["count_check"]["replies"] == {"presence": "No."}


def test_a_weak_single_box_stays_when_the_model_says_yes() -> None:
    out = _check(_one(0.5), _Answerer(presence="Yes, there is."))
    assert out["count"] == 1 and out["count_basis"] == "detector"


def test_a_strong_single_box_is_never_questioned() -> None:
    answerer = _Answerer(presence="no")
    out = _check(_one(0.6), answerer)
    assert out["count"] == 1 and answerer.asked == [] and "count_check" not in out


def test_two_boxes_are_never_vetoed() -> None:
    result = _one(0.45)
    result.update(count=2, bboxes=[[0, 0, 10, 10], [20, 0, 30, 10]], scores=[0.45, 0.44])
    answerer = _Answerer(presence="no")
    assert _check(result, answerer)["count"] == 2 and answerer.asked == []


def test_a_zero_is_rescued_only_when_the_count_and_presence_agree() -> None:
    out = _check(_zero(), _Answerer(count="2", presence="yes"), _CANDS)
    assert out["count"] == 1 and out["count_basis"] == "presence_rescue"
    assert out["bboxes"] == [[20, 0, 40, 10]] and out["scores"] == [0.3] and out["points"] == [[30.0, 5.0]]
    assert out["detector_count"] == 0


def test_a_zero_stays_when_presence_says_no() -> None:
    assert _check(_zero(), _Answerer(count="one", presence="No"), _CANDS)["count"] == 0


def test_a_zero_count_answer_skips_the_presence_question() -> None:
    answerer = _Answerer(count="There are none.", presence="yes")
    assert _check(_zero(), answerer, _CANDS)["count"] == 0
    assert len(answerer.asked) == 1


def test_a_zero_without_candidates_asks_nothing() -> None:
    answerer = _Answerer(count="3", presence="yes")
    out = _check(_zero(), answerer)
    assert out["count"] == 0 and answerer.asked == [] and out["count_basis"] == "detector"


def test_an_unavailable_model_leaves_the_detector_count() -> None:
    out = _check(_one(0.5), _Answerer(fail=MemoryError("not enough commit")))
    assert out["count"] == 1 and out["count_basis"] == "detector"
    assert "not enough commit" in out["count_check"]["skipped"]


def test_no_answerer_leaves_the_count_alone() -> None:
    out = _check(_one(0.5), None)
    assert out["count"] == 1 and out["detector_count"] == 1 and "count_check" not in out


class _Moondream:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def query(self, images: list[Any], question: str) -> list[str]:
        self.asked.append(question)
        return ["4"]


def test_the_vqa_estimate_asks_lfm_the_gated_question() -> None:
    answerer, md = _Answerer(count="There are 3 cats."), _Moondream()
    out = vqa_count_estimate(SimpleNamespace(answerer=answerer, vqa=md), None, "cat")
    assert answerer.asked == [COUNT_CHECK_COUNT.format(name="cat")] and md.asked == []
    assert out["value"] == 3 and out["raw"] == "There are 3 cats." and "LFM2.5-VL" in out["note"]


def test_the_vqa_estimate_falls_back_to_moondream() -> None:
    for answerer in (None, _Answerer(fail=OSError("gated"))):
        md = _Moondream()
        out = vqa_count_estimate(SimpleNamespace(answerer=answerer, vqa=md), None, "cat")
        assert md.asked == [VQA_COUNT_QUESTION.format(name="cat")]
        assert out["value"] == 4 and "Moondream2" in out["note"]
