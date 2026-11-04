"""`query_image`'s answering path (`analysis.vqa_answers`), checked without models: which questions carry the `ocr`
reading, the exact prompt Gate V5 measured on round 29, and the fall back to Moondream2."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from fusion_vision_mcp import analysis
from fusion_vision_mcp.analysis import lfm_vqa_prompt, vqa_answers
from fusion_vision_mcp.protocols import AppContext

SUFFIX = (
    "\nUse the exact spelling and full name from the OCR text when it answers the question. "
    "If the question asks about text or an item that is not in the image, answer that there is none."
)


class _Answerer:
    def __init__(self, fail: Exception | None = None, check_replies: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.tokens: list[int | None] = []
        self.fail = fail
        self.check_replies = check_replies

    def answer(self, images: list[Any], prompts: list[str], *, max_new_tokens: int | None = None) -> list[str]:
        if self.fail:
            raise self.fail
        self.prompts += prompts
        self.tokens += [max_new_tokens] * len(prompts)
        if self.check_replies is not None and prompts[0].startswith("Is the thing this question asks about"):
            return self.check_replies[: len(prompts)]
        return [f"lfm:{p}" for p in prompts]


class _Moondream:
    def query(self, images: list[Any], question: str) -> list[str]:
        return [f"moondream:{question}" for _ in images]


def _app(answerer: _Answerer | None) -> AppContext:
    return cast(AppContext, SimpleNamespace(answerer=answerer, vqa=_Moondream()))


@pytest.fixture
def ocr_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def fake_ocr(app: Any, images: Any, *, sharpened: bool = True, lfm_read: bool = True) -> list[dict[str, Any]]:
        assert not sharpened and not lfm_read, "query_image's OCR context is the first reading its gate measured"
        calls.append(len(images))
        return [{"text": "OPEN\n 24 HOURS \n\n"} for _ in images]

    monkeypatch.setattr(analysis, "ocr_pages", fake_ocr)
    return calls


def test_prompt_matches_the_gate_byte_for_byte() -> None:
    # VisionBench vqa_gate_v1.ocr_prompt with VERBATIM_PREFIX / VERBATIM_SUFFIX, the arm Gate V5 confirmed.
    assert lfm_vqa_prompt("What does the sign say?", "OPEN\n 24 HOURS \n") == (
        'Text read from the image by OCR: "OPEN / 24 HOURS"\nWhat does the sign say?' + SUFFIX
    )
    assert lfm_vqa_prompt("What brand name is printed on it?", " \n") == (
        "An OCR pass found no readable text in this image.\nWhat brand name is printed on it?" + SUFFIX
    )


@pytest.mark.parametrize(
    ("question", "routed"),
    [
        ("What brand of laptop is resting near the flower stems?", True),  # the lookup branch: What ... brand
        ("Which button is highlighted at the top?", True),
        ("Who is the author of the book?", True),  # the text-word branch
        ("What colour is the sign?", False),  # "what colour" is a visual question, never routed
        ("Is the price tag visible?", False),  # the lookup branch needs a Wh-word first
        ("How many people are there?", False),
    ],
)
def test_routing_takes_text_lookups_too(question: str, routed: bool) -> None:
    assert bool(analysis._TEXT_QUESTION_RE.search(question)) is routed


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


def test_memory_refusal_falls_back_to_moondream(ocr_calls: list[int]) -> None:
    from fusion_vision_mcp.memory import ModelMemoryError

    out = vqa_answers(
        _app(_Answerer(fail=ModelMemoryError("no commit"))), [Image.new("RGB", (8, 8))], "What colour is the car?"
    )
    assert out == ["moondream:What colour is the car?"]


def test_given_readings_replace_the_ocr_call(ocr_calls: list[int]) -> None:
    ans = _Answerer()
    vqa_answers(_app(ans), [Image.new("RGB", (8, 8))], "What does the sign say?", ocr_readings=[{"text": "STOP"}])
    assert ocr_calls == []
    assert ans.prompts == [lfm_vqa_prompt("What does the sign say?", "STOP")]


def _query_image_tool(app: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fusion_vision_mcp.tools import vqa

    # The cross-check's own OCR (the full `ocr` reading) is a different path, not under test here.
    monkeypatch.setattr(vqa, "_vqa_cross_check", lambda *a, **k: None)

    tools: dict[str, Any] = {}

    class _Mcp:
        def tool(self, *a: Any, **k: Any) -> Any:
            return lambda fn: tools.setdefault(fn.__name__, fn)

    vqa.register(cast(Any, _Mcp()))
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=app))
    return lambda src, question, **kw: tools["query_image"](ctx, src, question, **kw)


def _png(tmp_path: Any) -> str:
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(path)
    return str(path)


def test_consistency_shares_one_ocr_reading(
    ocr_calls: list[int], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ans = _Answerer()
    query = _query_image_tool(_app(ans), monkeypatch)
    question = "What does the sign say?"
    query(_png(tmp_path), question, check_consistency=True, structured_analysis=False)
    assert ocr_calls == [1]
    reading = "OPEN\n 24 HOURS"
    assert ans.prompts == [
        lfm_vqa_prompt(question, reading),
        lfm_vqa_prompt(f"Looking carefully at this image, answer precisely: {question}", reading),
    ]


def test_unrouted_question_makes_no_ocr_call_under_consistency(
    ocr_calls: list[int], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ans = _Answerer()
    query = _query_image_tool(_app(ans), monkeypatch)
    query(_png(tmp_path), "What colour is the car?", check_consistency=True, structured_analysis=False)
    assert ocr_calls == []
    # The presence check, then the bare question; the control question starts "Looking", so it is never checked.
    assert ans.prompts[1:] == [
        "What colour is the car?",
        "Looking carefully at this image, answer precisely: What colour is the car?",
    ]


# Track 8, Gate V7: the presence check before an unrouted What/Which/Who/Where question.
CHECK = "Is the thing this question asks about in the image? Question: {q} Answer yes or no."
ABSENT = "\nIf the question asks about something that is not in the image, answer that there is none."


@pytest.mark.parametrize(
    ("question", "checked"),
    [
        ("What colour is the passenger train?", True),
        ("  which dog is sleeping?", True),
        ("Who is holding the umbrella?", True),
        ("Where is the cat?", True),
        ("What does the sign say?", False),  # routed to the OCR context instead
        ("What brand of laptop is it?", False),  # routed by the lookup branch
        ("Is there a hookah on the table?", False),  # not a Wh question
        ("How many people are there?", False),
        ("Looking carefully at this image, answer precisely: What colour is it?", False),  # a control question
        ("Whatever is on the table?", False),  # a whole word only
    ],
)
def test_presence_checked_questions(question: str, checked: bool) -> None:
    assert analysis.presence_checked(question) is checked


@pytest.mark.parametrize(
    ("reply", "suffixed"),
    [("No.", True), ("no, it is not", True), ("Yes", False), ("Yes, there is no doubt", False), ("Maybe", False)],
)
def test_check_no_adds_the_absent_suffix(reply: str, suffixed: bool, ocr_calls: list[int]) -> None:
    question = "What colour is the passenger train?"
    ans = _Answerer(check_replies=[reply])
    out = vqa_answers(_app(ans), [Image.new("RGB", (8, 8))], question)
    assert ans.prompts[0] == CHECK.format(q=question)  # the gate's prompt, byte for byte
    assert ans.tokens[0] == 8
    assert ans.prompts[1] == question + (ABSENT if suffixed else "")
    assert ans.tokens[1] is None  # the answer keeps the answerer's own token cap
    assert out == ["lfm:" + ans.prompts[1]]
    assert ocr_calls == []


def test_check_decides_per_image(ocr_calls: list[int]) -> None:
    question = "Which dog is sleeping?"
    ans = _Answerer(check_replies=["No", "Yes"])
    vqa_answers(_app(ans), [Image.new("RGB", (8, 8))] * 2, question)
    assert ans.prompts[2:] == [question + ABSENT, question]


def test_presence_check_can_be_turned_off(ocr_calls: list[int]) -> None:
    ans = _Answerer(check_replies=["No"])
    assert vqa_answers(_app(ans), [Image.new("RGB", (8, 8))], "Where is the cat?", presence_check=False) == [
        "lfm:Where is the cat?"
    ]
    assert ans.prompts == ["Where is the cat?"]


def test_routed_question_skips_the_check(ocr_calls: list[int]) -> None:
    ans = _Answerer(check_replies=["No"])
    vqa_answers(_app(ans), [Image.new("RGB", (8, 8))], "What does the sign say?")
    assert ans.prompts == [lfm_vqa_prompt("What does the sign say?", "OPEN\n 24 HOURS")]


def test_moondream_fallback_has_no_check(ocr_calls: list[int]) -> None:
    out = vqa_answers(_app(_Answerer(fail=OSError("gated"))), [Image.new("RGB", (8, 8))], "Where is the cat?")
    assert out == ["moondream:Where is the cat?"]


def test_query_image_passes_presence_check(
    ocr_calls: list[int], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ans = _Answerer(check_replies=["No"])
    query = _query_image_tool(_app(ans), monkeypatch)
    query(_png(tmp_path), "Where is the cat?", structured_analysis=False, presence_check=False)
    assert ans.prompts == ["Where is the cat?"]


# -- clean_ocr_page / smart_ocr_excerpt (Gate V8 screen, arm NE) ----------------------------------------------------
# Questions marked "cached" are real ones from VisionBench's vqa_gate_v8_scored.jsonl; the rest are synthetic.
_RELIGIOUS = "What religious symbol stands at the peak of the chapel gable roof?"  # cached, r30 n=284: OCR read "十"
_ARTIKEL = "What person's name is the subject of 'Artikel des Tages'?"  # cached, r23 n=440, a 2,688-character page


def _quoted(prompt: str) -> str:
    return prompt.split("\n")[0]


def test_noise_reading_is_treated_as_empty() -> None:
    empty = "An OCR pass found no readable text in this image.\n" + _RELIGIOUS + SUFFIX
    assert lfm_vqa_prompt(_RELIGIOUS, "十") == empty
    # cached: r23 n=3 / r29 n=346 read a lone "8" / "A" on a brand and a street-name question
    assert lfm_vqa_prompt("What brand name is printed on the rim of the broth pot?", "8").startswith("An OCR pass")
    assert lfm_vqa_prompt("What street name is printed on the metal curbside sign?", "A").startswith("An OCR pass")


def test_single_character_is_kept_for_a_letter_question() -> None:
    # synthetic: a one-letter answer is the point of the question
    assert lfm_vqa_prompt("Which letter is printed on the sign?", "A") == (
        'Text read from the image by OCR: "A"\nWhich letter is printed on the sign?' + SUFFIX
    )
    # but not a non-ASCII one: that is PP-OCRv6 noise
    assert lfm_vqa_prompt("Which letter is printed on the sign?", "人").startswith("An OCR pass")


def test_cjk_reading_is_kept_when_the_question_mentions_cjk() -> None:
    # synthetic; r23 n=869 is the cached Chinese-characters question (its reading there was "111")
    question = "What English translation is written below the Chinese characters?"
    assert lfm_vqa_prompt(question, "人民") == f'Text read from the image by OCR: "人民"\n{question}{SUFFIX}'
    assert lfm_vqa_prompt("What does the sign say?", "人民").startswith("An OCR pass")  # no CJK in the question
    # a lone character is under the 2-character floor whatever the question (frozen arm N), so it is emptied too
    assert lfm_vqa_prompt(question, "人").startswith("An OCR pass")
    assert lfm_vqa_prompt(question, "出口\n中") == f'Text read from the image by OCR: "出口 / 中"\n{question}{SUFFIX}'


def test_multi_line_latin_reading_is_untouched() -> None:
    assert lfm_vqa_prompt("What is the total?", "Item\n2\nTotal") == (
        'Text read from the image by OCR: "Item / 2 / Total"\nWhat is the total?' + SUFFIX
    )


def test_long_cjk_reading_is_not_noise() -> None:
    assert lfm_vqa_prompt(_RELIGIOUS, "管家\n天文\n管家人").startswith('Text read from the image by OCR: "管家')


def test_cjk_noise_rule_is_for_the_english_engine_only() -> None:
    # synthetic: under EasyOCR (non-English --ocr-languages) a short CJK sign is a real reading
    assert lfm_vqa_prompt(_RELIGIOUS, "出口", english_engine=False).startswith(
        'Text read from the image by OCR: "出口"'
    )
    assert lfm_vqa_prompt(_RELIGIOUS, "出口").startswith("An OCR pass")
    assert lfm_vqa_prompt(_RELIGIOUS, "x", english_engine=False).startswith(
        "An OCR pass"
    )  # the short rule is engine-free


@pytest.mark.parametrize(("sharpened_pass", "empty"), [(True, True), (False, False)])
def test_vqa_answers_scopes_the_cjk_rule_to_the_engine(
    monkeypatch: pytest.MonkeyPatch, sharpened_pass: bool, empty: bool
) -> None:
    monkeypatch.setattr(analysis, "ocr_pages", lambda *a, **k: [{"text": "出口"}])
    ans = _Answerer()
    app = cast(
        AppContext,
        SimpleNamespace(answerer=ans, vqa=_Moondream(), ocr_engine=SimpleNamespace(sharpened_pass=sharpened_pass)),
    )
    vqa_answers(app, [Image.new("RGB", (8, 8))], "What text is written on the door?")
    assert ans.prompts[0].startswith("An OCR pass") is empty


def test_reading_within_the_limit_is_unchanged_byte_for_byte() -> None:
    words = " ".join(f"w{i}" for i in range(160))
    assert lfm_vqa_prompt(_ARTIKEL, words) == f'Text read from the image by OCR: "{words}"\n{_ARTIKEL}{SUFFIX}'


def test_anchor_past_the_cut_gets_a_window() -> None:
    page = [f"w{i}" for i in range(250)] + ["Artikel", "des", "Tages", "Zephyrine", "Quill"] + ["x"] * 40
    quoted = _quoted(lfm_vqa_prompt(_ARTIKEL, " ".join(page)))
    head, _, windows = quoted.partition(" / ... / ")
    assert head.endswith("w159")  # the 160-word head, as shipped
    assert windows.startswith("w") and "Artikel des Tages Zephyrine Quill" in windows


def test_windows_are_capped_at_two() -> None:
    page = []
    for anchor in ("Alphabet", "Bravo99", "Charlie77"):
        page += ["filler"] * 200 + [anchor]
    question = "Which of Alphabet, Bravo99 and Charlie77 is listed?"
    quoted = _quoted(lfm_vqa_prompt(question, " ".join(page)))
    assert quoted.count(" / ... / ") == 1
    tail = quoted.split(" / ... / ", 1)[1]
    assert "Alphabet" in tail and "Bravo99" in tail and "Charlie77" not in tail
    assert tail.count(" / ") == 1


def test_anchor_already_in_the_head_adds_nothing() -> None:
    page = ["Artikel", "des", "Tages"] + [f"w{i}" for i in range(250)]
    assert " / ... / " not in lfm_vqa_prompt(_ARTIKEL, " ".join(page))


def test_question_anchors() -> None:
    assert analysis.question_anchors(_ARTIKEL) == ["s name is the subject of", "person", "subject", "Artikel", "Tages"]
    assert analysis.question_anchors("What colour is it?") == ["colour"]
