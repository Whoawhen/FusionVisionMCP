"""Loop detection for Moondream2's streamed long caption (`repetition.py`, `Moondream.caption`)."""

from typing import Any

from PIL import Image

from fusion_vision_mcp.repetition import cut_loop, loop_start

PROSE = (
    "The image shows a small wooden boat tied to a dock on a calm lake at sunset. The sky is orange and pink, "
    "and the water reflects the colors. In the background there are pine trees and a low range of hills. "
    "A red flag hangs from the boat's mast, and a coil of rope lies on the dock beside a metal bucket."
)


def test_normal_description_is_left_alone() -> None:
    assert loop_start(PROSE) is None
    assert cut_loop(PROSE) == (PROSE, False)


def test_a_phrase_reused_once_is_not_a_loop() -> None:
    # A sentence restated once is redundant, not a loop: 8 repeated spans, below the run of 16.
    text = PROSE + " In the background there are pine trees and a low range of hills, under the evening sky."
    assert loop_start(text) is None


def test_single_word_loop_keeps_one_copy() -> None:
    # Seen on a tune image: an alphabet chart's caption ended in hundreds of "Í".
    text = "The chart lists letters A, B, C and Í" + " Í" * 300
    cut, found = cut_loop(text)
    assert found
    assert cut.startswith("The chart lists letters")
    assert cut.count("Í") < 10


def test_list_cycle_loop_keeps_one_cycle() -> None:
    # Seen on a tune image: the same seven language names repeated to the token cap.
    cycle = "Polish, Spanish, Italian, French, German, Dutch, Russian, "
    text = "The menu offers languages: " + cycle * 40
    cut, found = cut_loop(text)
    assert found
    for name in ("Polish", "Spanish", "Russian"):
        assert name in cut
    assert len(cut.split()) < 30


def test_loop_after_real_content_keeps_the_content() -> None:
    text = PROSE + " Labels include " + "effects, " * 200
    cut, found = cut_loop(text)
    assert found
    assert cut.startswith(PROSE)
    assert cut.count("effects") < 10
    assert not cut.endswith(",")


def test_punctuation_and_case_do_not_hide_a_loop() -> None:
    text = "Buttons: " + '"FX," "SX," "fx," "M," "S," ' * 60
    assert loop_start(text) is not None


def test_short_text_is_never_cut() -> None:
    assert loop_start("a a a a a a a a") is None
    assert cut_loop("") == ("", False)


class _StreamingModel:
    """Stands in for Moondream2: `caption(..., stream=True)` yields tokens and counts how many were taken."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.taken = 0

    def caption(self, image: Any, length: str, stream: bool = False) -> dict[str, Any]:
        assert stream

        def gen() -> Any:
            for t in self.tokens:
                self.taken += 1
                yield t

        return {"caption": gen()}


def _moondream(tokens: list[str]) -> Any:
    from fusion_vision_mcp.moondream import Moondream

    md = object.__new__(Moondream)
    md.model = _StreamingModel(tokens)
    return md


def test_caption_stops_generating_once_it_loops() -> None:
    tokens = [w + " " for w in PROSE.split()] + ["effects, "] * 700
    md = _moondream(tokens)
    [caption] = md.caption([Image.new("RGB", (8, 8))])
    assert caption.startswith(PROSE)
    assert caption.count("effects") < 10
    # Generation stopped soon after the loop began, not at the end of the 700 repeated tokens.
    assert md.model.taken < len(PROSE.split()) + 60


def test_caption_without_a_loop_is_unchanged() -> None:
    md = _moondream([w + " " for w in PROSE.split()])
    assert md.caption([Image.new("RGB", (8, 8))]) == [PROSE]
    assert md.model.taken == len(PROSE.split())
