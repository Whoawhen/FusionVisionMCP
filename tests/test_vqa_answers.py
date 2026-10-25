"""`query_image`'s answering path (`analysis.vqa_answers`), checked without models: which questions carry the `ocr`
reading, the exact prompt the round 23 gate measured, and the fall back to Moondream2."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from fusion_vision_mcp import analysis
from fusion_vision_mcp.analysis import lfm_vqa_prompt, vqa_answers
from fusion_vision_mcp.protocols import AppContext

SUFFIX = "\nIf the question asks about text that is not in the image, answer that there is none."


class _Answerer:
    def __init__(self, fail: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.fail = fail

    def answer(self, images: list[Any], prompts: list[str]) -> list[str]:
        if self.fail:
            raise self.fail
        self.prompts += prompts
        return [f"lfm:{p}" for p in prompts]


class _Moondream:
    def query(self, images: list[Any], question: str) -> list[str]:
        return [f"moondream:{question}" for _ in images]


def _app(answerer: _Answerer | None) -> AppContext:
    return cast(AppContext, SimpleNamespace(answerer=answerer, vqa=_Moondream()))


@pytest.fixture
def ocr_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def fake_ocr(app: Any, images: Any) -> list[dict[str, Any]]:
        calls.append(len(images))
        return [{"text": "OPEN\n 24 HOURS \n\n"} for _ in images]

    monkeypatch.setattr(analysis, "ocr_pages", fake_ocr)
    return calls


def test_prompt_matches_the_gate_byte_for_byte() -> None:
    # VisionBench vqa_premise.ocr_prompt, the arm round 23 confirmed.
    assert lfm_vqa_prompt("What does the sign say?", "OPEN\n 24 HOURS \n") == (
        'Text read from the image by OCR (it may be incomplete or misread): "OPEN / 24 HOURS"\nWhat does the sign say?'
        + SUFFIX
    )
    assert lfm_vqa_prompt("What brand name is printed on it?", " \n") == (
        "An OCR pass found no readable text in this image.\nWhat brand name is printed on it?" + SUFFIX
    )


def test_prompt_keeps_at_most_160_words() -> None:
    prompt = lfm_vqa_prompt("What is written here?", " ".join(f"w{i}" for i in range(300)))
    quoted = prompt.split('"')[1].split()
    assert len(quoted) == 160 and quoted[-1] == "w159"


def test_text_question_carries_the_ocr_reading(ocr_calls: list[int]) -> None:
    ans = _Answerer()
    out = vqa_answers(_app(ans), [Image.new("RGB", (8, 8))] * 2, "What text is written on the door?")
    assert ocr_calls == [2]
    assert ans.prompts == [lfm_vqa_prompt("What text is written on the door?", "OPEN\n 24 HOURS")] * 2
    assert out[0].startswith("lfm:Text read from the image by OCR")


def test_other_questions_go_verbatim_without_ocr(ocr_calls: list[int]) -> None:
    ans = _Answerer()
    assert vqa_answers(_app(ans), [Image.new("RGB", (8, 8))], "What colour is the car?") == [
        "lfm:What colour is the car?"
    ]
    assert ocr_calls == []


def test_falls_back_to_moondream_when_lfm_cannot_load(ocr_calls: list[int]) -> None:
    out = vqa_answers(_app(_Answerer(fail=OSError("gated"))), [Image.new("RGB", (8, 8))], "What colour is the car?")
    assert out == ["moondream:What colour is the car?"]


def test_without_an_answerer_moondream_answers(ocr_calls: list[int]) -> None:
    assert vqa_answers(_app(None), [Image.new("RGB", (8, 8))], "What does it say?") == ["moondream:What does it say?"]
    assert ocr_calls == []


class _Clip:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def zero_shot(self, images: list[Any], prompts: list[str]) -> list[list[float]]:
        self.prompts = prompts
        # The second image's top pick is the last label.
        return [[0.9] + [0.0] * (len(prompts) - 1), [0.0] * (len(prompts) - 1) + [0.9]][: len(images)]


def test_image_type_lines_name_clips_top_label() -> None:
    from fusion_vision_mcp.analysis import image_type_lines
    from fusion_vision_mcp.constants import CAPTION_IMAGE_TYPES

    clip = _Clip()
    app = cast(AppContext, SimpleNamespace(aesthetic=clip))
    lines = image_type_lines(app, [Image.new("RGB", (8, 8))] * 2)
    assert lines == [f"Image type: {CAPTION_IMAGE_TYPES[0]}.", f"Image type: {CAPTION_IMAGE_TYPES[-1]}."]
    # The prompts the gate measured: "a"/"an" + the label.
    assert clip.prompts[0] == "a photograph" and "an oil painting" in clip.prompts
