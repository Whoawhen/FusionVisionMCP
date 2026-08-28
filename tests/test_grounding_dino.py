#  test_grounding_dino.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Unit tests for the counting detector's pure logic.

`_envelope_indices` and `_as_prompt` need no weights, so these run in milliseconds and
without a download -- unlike the tool-level tests in `test_server.py`.
"""

from fusion_vision_mcp.grounding_dino import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    GroundingDino,
)

envelope_indices = GroundingDino._envelope_indices
as_prompt = GroundingDino._as_prompt


def test_a_box_around_the_whole_group_is_identified() -> None:
    """The measured artifact: the model returns the instances plus one box round them."""
    instances = [[0, 0, 10, 10], [20, 0, 30, 10], [0, 20, 10, 30], [20, 20, 30, 30]]
    boxes = [[-5, -5, 35, 35], *instances]

    assert envelope_indices(boxes) == {0}


def test_two_instances_plus_an_envelope_is_still_caught() -> None:
    """Three boxes is the real two-instance case, and the floor must not exclude it.

    Measured on two touching shapes: the detector returns both instances and one box
    spanning the pair, so a floor above three left every two-instance count one too high.
    """
    boxes = [[0, 0, 30, 10], [0, 0, 14, 10], [16, 0, 30, 10]]

    assert envelope_indices(boxes) == {0}


def test_two_boxes_alone_are_never_treated_as_a_group() -> None:
    """At two boxes, containment cannot distinguish an envelope from genuine nesting."""
    nested = [[0, 0, 40, 40], [10, 10, 20, 20]]

    assert envelope_indices(nested) == set()


def test_instances_that_merely_overlap_are_not_envelopes() -> None:
    """Overlapping is not containing: none of these swallows the others' centres."""
    boxes = [[0, 0, 20, 10], [12, 0, 32, 10], [24, 0, 44, 10], [36, 0, 56, 10]]

    assert envelope_indices(boxes) == set()


def test_a_row_of_separate_instances_has_no_envelope() -> None:
    boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(5)]

    assert envelope_indices(boxes) == set()


def test_a_single_occluded_object_is_not_treated_as_a_group_envelope() -> None:
    """The measured occlusion bug: duplicate re-detections of ONE object, not a group.

    Exact boxes/scores measured on a rectangle ~60% hidden behind another shape,
    queried at the default threshold=0.15: the correct tight box (score 0.66) plus
    three looser, lower-confidence boxes describing the same object at wider
    extents. The old containment-only test misread the tight box (and the second
    box) as an envelope around a "group" of the looser ones and dropped both real
    detections, taking the count to zero even though the tight box scored well
    above threshold. What must survive is the tight box -- the correct detection --
    not being classified as an envelope; the two loosest, most redundant boxes
    (which mostly duplicate each other, mean pairwise IoU ~0.8) are still fair
    game to drop.
    """
    tight = [148.6, 148.3, 398.6, 351.6]  # score 0.66, the correct box
    loose_a = [319.1, 98.4, 581.8, 360.9]  # score 0.33
    loose_b = [148.7, 98.9, 582.2, 359.7]  # score 0.27
    loose_c = [148.3, 144.9, 579.4, 356.6]  # score 0.19

    envelopes = envelope_indices([tight, loose_a, loose_b, loose_c])
    assert 0 not in envelopes


def test_prompts_are_lowercased_and_period_terminated() -> None:
    """Grounding DINO expects this exact shape; a bare noun silently matches worse."""
    assert as_prompt("Petal") == "petal."
    assert as_prompt("  PINK Circle  ") == "pink circle."
    assert as_prompt("petal.") == "petal."


def test_the_default_thresholds_are_the_measured_ones() -> None:
    """Pins the sweep's outcome: 0.15 is the lowest box threshold holding every control.

    Lower scores better on positives alone and breaks the textured negative controls,
    so a well-meaning nudge downward should fail here and be argued for with fixtures.
    """
    assert DEFAULT_BOX_THRESHOLD == 0.15
    assert DEFAULT_TEXT_THRESHOLD == 0.25
