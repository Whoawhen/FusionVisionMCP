"""Unit tests for the generic-query guard. Pure logic, no model, runs in milliseconds."""

from fusion_vision_mcp.query_policy import (
    GENERIC_OBJECT_QUERIES,
    is_full_frame_box,
    is_generic_query,
    suppress_generic_full_frame,
)


def test_every_generic_noun_is_recognized() -> None:
    for noun in GENERIC_OBJECT_QUERIES:
        assert is_generic_query(noun)


def test_whitespace_and_case_are_normalized() -> None:
    assert is_generic_query("  Object  ")
    assert is_generic_query("THING")
    assert is_generic_query("\tItems\n")


def test_a_specific_noun_is_not_generic() -> None:
    assert not is_generic_query("tree")
    assert not is_generic_query("blue rectangle")
    assert not is_generic_query("wood")


def test_a_phrase_containing_a_generic_word_is_not_generic() -> None:
    """Substring matching would wrongly flag a specific description as vague."""
    assert not is_generic_query("the small object on the left")
    assert not is_generic_query("a red object")
    assert not is_generic_query("interesting things in the background")


def test_empty_string_is_not_generic() -> None:
    assert not is_generic_query("")
    assert not is_generic_query("   ")


# -- is_full_frame_box / suppress_generic_full_frame -----------------------------------
# Sprint 1: extends the generic-query guard already shipped for detect_objects into
# count_objects's plain (non-clip_art) Grounding DINO path and spatial_relations's
# per-object detection, per spec doc §8 ("Wire the guard into detect_objects,
# count_objects, and relevant spatial_relations paths").

_MEASURED_FULL_FRAME_BOX = [1.716238260269165, 0.25887489318847656, 598.6983642578125, 398.87591552734375]
_IMAGE_WIDTH, _IMAGE_HEIGHT = 600, 400


def test_measured_full_frame_box_is_recognized() -> None:
    """Pins the live measurement: GroundingDino.detect_objects("object") on
    tests/detect_blank_canvas.png returns exactly this box (score 0.46, count=1)."""
    assert is_full_frame_box(_MEASURED_FULL_FRAME_BOX, _IMAGE_WIDTH, _IMAGE_HEIGHT)


def test_a_small_box_is_not_full_frame() -> None:
    assert not is_full_frame_box([10, 10, 50, 50], _IMAGE_WIDTH, _IMAGE_HEIGHT)


def test_zero_area_image_is_never_full_frame() -> None:
    assert not is_full_frame_box([0, 0, 10, 10], 0, 0)


def test_suppress_drops_the_measured_false_positive() -> None:
    result = {
        "count": 1,
        "bboxes": [_MEASURED_FULL_FRAME_BOX],
        "points": [[300.2, 199.6]],
        "labels": ["object"],
        "scores": [0.46395546197891235],
        "group_boxes_dropped": 0,
    }
    filtered = suppress_generic_full_frame("object", result, _IMAGE_WIDTH, _IMAGE_HEIGHT)

    assert filtered["bboxes"] == []
    assert filtered["points"] == []
    assert filtered["labels"] == []
    assert filtered["scores"] == []
    assert filtered["count"] == 0
    assert filtered["generic_full_frame_dropped"] == 1
    # group_boxes_dropped is untouched -- a different, unrelated field.
    assert filtered["group_boxes_dropped"] == 0


def test_suppress_leaves_a_specific_noun_untouched_even_if_full_frame() -> None:
    """The "wood" case: a specific noun that genuinely fills the frame must survive,
    from this detector too, not just Florence-2's exclude_full_frame."""
    result = {
        "count": 1,
        "bboxes": [_MEASURED_FULL_FRAME_BOX],
        "points": [[300.2, 199.6]],
        "labels": ["wood"],
        "scores": [0.9],
        "group_boxes_dropped": 0,
    }
    filtered = suppress_generic_full_frame("wood", result, _IMAGE_WIDTH, _IMAGE_HEIGHT)

    assert filtered is result
    assert filtered["bboxes"] == [_MEASURED_FULL_FRAME_BOX]
    assert "generic_full_frame_dropped" not in filtered


def test_suppress_leaves_a_generic_query_untouched_when_the_box_is_not_full_frame() -> None:
    """A vague noun that happens to find a small, plausible box should not be discarded
    just because the query is generic -- only the full-frame shape is the tell."""
    result = {
        "count": 1,
        "bboxes": [[10, 10, 50, 50]],
        "points": [[30, 30]],
        "labels": ["object"],
        "scores": [0.5],
        "group_boxes_dropped": 0,
    }
    filtered = suppress_generic_full_frame("object", result, _IMAGE_WIDTH, _IMAGE_HEIGHT)

    assert filtered is result
    assert filtered["bboxes"] == [[10, 10, 50, 50]]


def test_suppress_keeps_a_genuine_instance_alongside_a_dropped_envelope() -> None:
    """Mixed case: one real small detection plus one spurious full-frame box for the
    same generic query -- only the full-frame entry should be dropped."""
    result = {
        "count": 2,
        "bboxes": [[10, 10, 50, 50], _MEASURED_FULL_FRAME_BOX],
        "points": [[30, 30], [300.2, 199.6]],
        "labels": ["object", "object"],
        "scores": [0.5, 0.46],
        "group_boxes_dropped": 0,
    }
    filtered = suppress_generic_full_frame("object", result, _IMAGE_WIDTH, _IMAGE_HEIGHT)

    assert filtered["bboxes"] == [[10, 10, 50, 50]]
    assert filtered["labels"] == ["object"]
    assert filtered["scores"] == [0.5]
    assert filtered["count"] == 1
    assert filtered["generic_full_frame_dropped"] == 1


def test_suppress_handles_a_result_with_no_scores_field() -> None:
    """count_objects's clip_art path reports scores=None (Florence-2's grounding head
    has no per-box confidence) -- the helper must not choke on that shape."""
    result = {
        "count": 1,
        "bboxes": [_MEASURED_FULL_FRAME_BOX],
        "points": [[300.2, 199.6]],
        "labels": ["object"],
        "scores": None,
        "group_boxes_dropped": None,
    }
    filtered = suppress_generic_full_frame("object", result, _IMAGE_WIDTH, _IMAGE_HEIGHT)

    assert filtered["bboxes"] == []
    assert filtered["scores"] is None
    assert filtered["count"] == 0
