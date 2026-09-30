"""Unit tests for the decision helpers in analysis.py (no models).

`_pick_primary_subject`, `_size_measurement` and `_aesthetic_comparison` are the logic
behind `critique_composition`, size questions and `compare_with`. They are pinned here
against stub apps, so a regression shows up in milliseconds rather than in a benchmark run.
"""

from PIL import Image

from fusion_vision_mcp.analysis import (
    _COMPARE_TIE,
    _aesthetic_comparison,
    _pick_primary_subject,
    _size_measurement,
)


class TestPickPrimarySubject:
    def test_no_boxes_returns_none(self) -> None:
        assert _pick_primary_subject({"bboxes": []}, (100, 100)) is None
        assert _pick_primary_subject({}, (100, 100)) is None

    def test_equal_area_prefers_the_centred_box(self) -> None:
        centred = [40, 40, 60, 60]
        corner = [0, 0, 20, 20]
        assert _pick_primary_subject({"bboxes": [corner, centred]}, (100, 100)) == centred
        assert _pick_primary_subject({"bboxes": [centred, corner]}, (100, 100)) == centred

    def test_much_larger_off_centre_box_beats_a_tiny_centred_one(self) -> None:
        tiny_centred = [48, 48, 52, 52]
        large_off_centre = [0, 0, 60, 60]
        assert _pick_primary_subject({"bboxes": [tiny_centred, large_off_centre]}, (100, 100)) == large_off_centre

    def test_returned_box_is_one_of_the_inputs(self) -> None:
        boxes = [[5, 5, 30, 30], [40, 40, 90, 70], [10, 60, 25, 95]]
        assert _pick_primary_subject({"bboxes": boxes}, (100, 100)) in boxes

    def test_float_boxes_come_back_as_ints(self) -> None:
        assert _pick_primary_subject({"bboxes": [[40.7, 40.2, 60.9, 60.1]]}, (100, 100)) == [40, 40, 60, 60]


class _StubCounter:
    def __init__(self, bboxes: list, scores: list | None = None) -> None:
        self.bboxes = bboxes
        self.scores = scores

    def detect_objects(self, images: list, object_name: str) -> list[dict]:
        return [{"bboxes": self.bboxes, "scores": self.scores}]


class _SizeApp:
    def __init__(self, counter: _StubCounter) -> None:
        self.counter = counter


_IMAGE = Image.new("RGB", (100, 100))
_SMALL, _MEDIUM, _BIG = [0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 50, 50]


class TestSizeMeasurement:
    def test_nothing_detected_returns_none(self) -> None:
        """The "never guess" rule: no detections means no cross-check at all."""
        assert _size_measurement(_SizeApp(_StubCounter([])), _IMAGE, "sword", "which sword is largest?") is None

    def test_largest_picks_the_biggest_box(self) -> None:
        app = _SizeApp(_StubCounter([_SMALL, _BIG, _MEDIUM], [0.5, 0.6, 0.7]))
        result = _size_measurement(app, _IMAGE, "sword", "Which sword is the largest?")
        assert result is not None
        assert result["extremum"] == "largest"
        assert result["box"] == _BIG
        assert result["area"] == 2500
        assert result["instances_compared"] == 3
        assert result["score"] == 0.6
        assert result["tool"] == "detect_objects"
        assert result["object"] == "sword"

    def test_smallest_picks_the_smallest_box(self) -> None:
        app = _SizeApp(_StubCounter([_MEDIUM, _BIG, _SMALL]))
        result = _size_measurement(app, _IMAGE, "sword", "which is the SMALLEST sword?")
        assert result is not None
        assert result["extremum"] == "smallest"
        assert result["box"] == _SMALL
        assert result["score"] is None

    def test_single_detection_still_returns_a_result(self) -> None:
        result = _size_measurement(_SizeApp(_StubCounter([_MEDIUM])), _IMAGE, "sword", "which is largest?")
        assert result is not None
        assert result["box"] == _MEDIUM
        assert result["instances_compared"] == 1


class _StubAesthetic:
    """Scores by image width, so a test controls each score by sizing its image.

    `styles` maps width -> medium for the classified path.
    """

    def __init__(self, styles: dict[int, str] | None = None) -> None:
        self.styles = styles or {}

    def score(self, images: list) -> list[dict]:
        return [{"score": float(i.width) / 10} for i in images]

    def score_and_classify(self, images: list) -> tuple[list[dict], list[dict]]:
        styles = [{"style": self.styles[i.width], "distribution": {self.styles[i.width]: 1.0}} for i in images]
        return self.score(images), styles


class _CompareApp:
    iqa = None
    reasoner = None

    def __init__(self, aesthetic: _StubAesthetic) -> None:
        self.aesthetic = aesthetic


def _img(score: float) -> Image.Image:
    return Image.new("RGB", (int(score * 10), 4))


class TestAestheticComparison:
    def test_higher_scored_image_is_preferred(self) -> None:
        app = _CompareApp(_StubAesthetic())
        (entry,) = _aesthetic_comparison(app, [_img(6.0)], [_img(4.0)], style_context=False)
        assert entry["preferred"] == "image"
        assert entry["delta"] == 2.0
        (entry,) = _aesthetic_comparison(app, [_img(4.0)], [_img(6.0)], style_context=False)
        assert entry["preferred"] == "reference"
        assert entry["delta"] == -2.0

    def test_equal_scores_do_not_claim_a_preference(self) -> None:
        app = _CompareApp(_StubAesthetic())
        (entry,) = _aesthetic_comparison(app, [_img(5.0)], [_img(5.0)], style_context=False)
        assert entry["preferred"] == "tie"
        assert entry["delta"] == 0.0

    def test_a_difference_inside_the_tie_band_is_a_tie(self) -> None:
        app = _CompareApp(_StubAesthetic())
        nudge = _COMPARE_TIE / 2
        image, reference = _img(5.0), _img(5.0)
        app.aesthetic.score = lambda images: [{"score": 5.0 + nudge} if i is image else {"score": 5.0} for i in images]  # type: ignore[method-assign]
        (entry,) = _aesthetic_comparison(app, [image], [reference], style_context=False)
        assert entry["preferred"] == "tie"

    def test_each_page_is_compared_to_the_first_reference_page(self) -> None:
        app = _CompareApp(_StubAesthetic())
        entries = _aesthetic_comparison(app, [_img(6.0), _img(3.0)], [_img(5.0), _img(9.0)], style_context=False)
        assert [e["preferred"] for e in entries] == ["image", "reference"]

    def test_different_media_add_a_cross_medium_warning(self) -> None:
        app = _CompareApp(_StubAesthetic({60: "photograph", 40: "illustration"}))
        (entry,) = _aesthetic_comparison(app, [_img(6.0)], [_img(4.0)], style_context=True)
        assert "cross_medium_warning" in entry
        assert entry["image"]["style"] == "photograph"
        assert entry["reference"]["style"] == "illustration"

    def test_same_medium_adds_no_warning(self) -> None:
        app = _CompareApp(_StubAesthetic({60: "photograph", 40: "photograph"}))
        (entry,) = _aesthetic_comparison(app, [_img(6.0)], [_img(4.0)], style_context=True)
        assert "cross_medium_warning" not in entry

    def test_no_style_context_means_no_warning_even_across_media(self) -> None:
        app = _CompareApp(_StubAesthetic({60: "photograph", 40: "illustration"}))
        (entry,) = _aesthetic_comparison(app, [_img(6.0)], [_img(4.0)], style_context=False)
        assert "cross_medium_warning" not in entry
        assert "style" not in entry["image"]

    def test_absolute_scores_come_back_unchanged(self) -> None:
        app = _CompareApp(_StubAesthetic({60: "photograph", 40: "illustration"}))
        (entry,) = _aesthetic_comparison(app, [_img(6.0)], [_img(4.0)], style_context=True)
        assert entry["image"]["score"] == 6.0
        assert entry["reference"]["score"] == 4.0
        assert entry["image"]["photographic_aesthetic"] == 6.0

    def test_reference_is_a_copy_per_entry(self) -> None:
        """A shared dict would let one page's consumer rewrite every other page's reference."""
        app = _CompareApp(_StubAesthetic())
        first, second = _aesthetic_comparison(app, [_img(6.0), _img(7.0)], [_img(4.0)], style_context=False)
        assert first["reference"] is not second["reference"]
