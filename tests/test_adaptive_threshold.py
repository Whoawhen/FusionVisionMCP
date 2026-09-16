"""Unit tests for adaptive_threshold.py. Pure logic, no model, runs in milliseconds."""

from fusion_vision_mcp.adaptive_threshold import ThresholdResult, choose_threshold


def test_empty_scores_returns_base() -> None:
    r = choose_threshold([])
    assert isinstance(r, ThresholdResult)
    assert r.threshold == 0.15
    assert r.used is False
    assert r.confidence == 0.0
    assert "no scores" in r.reason


def test_single_detection_returns_base() -> None:
    r = choose_threshold([0.9])
    assert r.threshold == 0.15
    assert r.used is False
    assert r.confidence == 0.0
    assert "single detection" in r.reason


def test_weak_top_score_suppresses_adaptation() -> None:
    """Top score below 0.50 should suppress adaptation even with a gap."""
    # Top score 0.45, gap 0.30 (would be a cliff if top were stronger)
    r = choose_threshold([0.45, 0.15])
    assert r.threshold == 0.15
    assert r.used is False
    assert "top score" in r.reason
    assert "adaptation suppressed" in r.reason


def test_clean_cliff_adapts() -> None:
    """Strong target (0.90) with clear gap (0.30) to distractors (0.19-0.32) --
    the star-among-18-shapes case from the spec doc (F9).
    """
    scores = [0.90, 0.32, 0.29, 0.25, 0.22, 0.19]
    r = choose_threshold(scores)
    assert r.used is True
    assert r.threshold > 0.15
    assert r.threshold < 0.90
    assert r.confidence > 0.0
    assert "cliff" in r.reason


def test_shallow_gradient_no_adaptation() -> None:
    """A shallow gradient (no dominant gap) should not trigger adaptation."""
    scores = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
    r = choose_threshold(scores)
    assert r.used is False
    assert r.threshold == 0.15
    assert "shallow gradient" in r.reason or "no confidence cliff" in r.reason


def test_adaptation_clamped_to_base() -> None:
    """A candidate at or below a (raised) base_threshold is clamped and marked unused.

    `adapted = max(base_threshold, min(candidate, top_score - 0.01))` can never go
    below `base_threshold` by construction, so this branch only fires when the
    cliff's own midpoint doesn't clear a caller-supplied `base_threshold` higher
    than the module's own default (0.15) would ever require. Scores [0.60, 0.20]
    have a genuine cliff (gap 0.40) whose midpoint is 0.40; raising base_threshold
    to 0.45 -- above that midpoint -- must clamp to 0.45 and report unused, not
    silently accept the lower candidate.
    """
    scores = [0.60, 0.20]
    r = choose_threshold(scores, base_threshold=0.45)
    assert r.used is False
    assert r.threshold == 0.45
    assert "clamped to base" in r.reason


def test_adaptation_not_clamped_when_candidate_clears_base() -> None:
    """The same cliff, with the default base_threshold, adapts normally -- the
    clamping in the test above is caused by the raised base, not the cliff itself."""
    scores = [0.60, 0.20]
    r = choose_threshold(scores, min_top_score=0.40)
    assert r.used is True
    assert r.threshold == 0.40


def test_two_scores_with_cliff_adapts() -> None:
    """Two scores with a clear gap should adapt."""
    scores = [0.80, 0.30]
    r = choose_threshold(scores)
    assert r.used is True
    assert r.threshold > 0.15
    assert r.threshold < 0.80
    # Midpoint would be 0.55, clamped below top
    assert r.threshold == 0.55


def test_three_scores_cliff_at_middle() -> None:
    """Cliff between middle scores should still be found."""
    scores = [0.85, 0.80, 0.30]
    r = choose_threshold(scores)
    assert r.used is True
    # Gap between 0.80 and 0.30 is 0.50
    assert r.threshold == 0.55


def test_adaptation_never_exceeds_top_score() -> None:
    """Adapted threshold should always be below the top score."""
    scores = [0.60, 0.20]
    r = choose_threshold(scores)
    assert r.threshold < 0.60


def test_confidence_scales_with_cliff_strength() -> None:
    """Confidence should be higher for stronger cliffs relative to top score."""
    r_strong = choose_threshold([0.90, 0.30])  # gap 0.60, top 0.90 -> confidence ~0.67
    r_weak = choose_threshold([0.55, 0.30])  # gap 0.25, top 0.55 -> confidence ~0.45
    assert r_strong.confidence > r_weak.confidence


def test_base_threshold_param_respected() -> None:
    """Custom base_threshold should be used when no adaptation."""
    r = choose_threshold([0.40, 0.35], base_threshold=0.25)
    assert r.threshold == 0.25
    assert r.used is False


def test_custom_min_gap_for_cliff() -> None:
    """Custom min_gap_for_cliff should affect whether adaptation triggers."""
    # Gap of 0.18 should adapt with min_gap=0.15 but not with default 0.20
    scores = [0.70, 0.52]
    r_default = choose_threshold(scores, min_gap_for_cliff=0.20)
    r_custom = choose_threshold(scores, min_gap_for_cliff=0.15)
    assert r_default.used is False
    assert r_custom.used is True


def test_custom_min_top_score() -> None:
    """Custom min_top_score should affect adaptation trigger."""
    scores = [0.45, 0.20]
    r_default = choose_threshold(scores, min_top_score=0.50)
    r_custom = choose_threshold(scores, min_top_score=0.40)
    assert r_default.used is False
    assert r_custom.used is True


def test_evenly_spaced_scores_are_a_shallow_gradient_not_a_cliff() -> None:
    """Tied maximal gaps must not read as a cliff.

    `[0.9, 0.5, 0.1]` has gaps `[0.4, 0.4]` -- no gap dominates, which is the
    definition of a shallow gradient. Filtering the "other" gaps by value rather than
    by index removed both, leaving an empty list that forced the ratio test to pass.
    """
    result = choose_threshold([0.9, 0.5, 0.1])

    assert not result.used
    assert result.threshold == 0.15
    assert "shallow gradient" in result.reason


def test_a_genuine_cliff_still_registers() -> None:
    """The fix must not suppress a real cliff: one dominant gap, the rest tight."""
    result = choose_threshold([0.92, 0.90, 0.88, 0.20, 0.18])

    assert result.used
    assert result.threshold > 0.15
