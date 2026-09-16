"""Adaptive detection threshold selection for Grounding DINO.

The fixed `threshold=0.15` default was chosen by sweeping the counting benchmark
fixtures: it is the *lowest* box-confidence floor at which every negative control
still holds. On a clean scene it works well. On a cluttered scene with a strong
target (score ~0.9) and many distractors (scores ~0.2-0.3), the fixed floor lets
the distractors through.

`choose_threshold` looks for a confidence cliff in the sorted detection scores.
If the gap between two consecutive scores is >= 0.20, it places a candidate
threshold midway between them -- but only if the top score is reasonably strong
(>= 0.5) and the gap is not just the tail of a shallow gradient. The adaptation
is conservative: the returned threshold is clamped to never go below the
`base_threshold` (0.15), and the adaptation is skipped if the evidence is weak.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdResult:
    """One adaptive-threshold decision.

    `threshold` is the box-confidence floor to actually use -- `base_threshold`
    unchanged unless `used` is true. `confidence` (0.0-1.0) is how strong the
    evidence for the adaptation was, and `reason` explains the decision in words,
    the same spirit as `separable`/`consensus`'s explanatory fields elsewhere in
    this project: a caller should be able to tell *why* a number came back the way
    it did, not just trust it.
    """

    threshold: float
    used: bool
    confidence: float
    reason: str


def _gaps(sorted_scores: list[float]) -> list[float]:
    """Differences between consecutive scores (descending order assumed)."""
    return [sorted_scores[i] - sorted_scores[i + 1] for i in range(len(sorted_scores) - 1)]


def _has_shallow_gradient(sorted_scores: list[float], min_gap_for_cliff: float = 0.20) -> bool:
    """True if the score distribution looks like a shallow gradient rather than
    a clear cliff -- i.e., no single gap dominates the others."""
    if len(sorted_scores) < 2:
        return True  # not enough evidence for a cliff
    if len(sorted_scores) == 2:
        # With exactly two scores, there's only one gap. If it meets the
        # min_gap_for_cliff threshold, it's a cliff, not a shallow gradient.
        return _gaps(sorted_scores)[0] < min_gap_for_cliff
    g = _gaps(sorted_scores)
    max_gap = max(g)
    # Drop the largest gap by INDEX, not by value. Filtering `x != max_gap` also removed
    # every gap that merely tied the maximum, so an evenly-spaced distribution -- the
    # textbook shallow gradient -- emptied this list, forced median_other to 0.0, and
    # made `max_gap / 0.01` clear the cliff test unconditionally.
    max_index = g.index(max_gap)
    other_gaps = [x for i, x in enumerate(g) if i != max_index] or [0.0]
    median_other = sorted(other_gaps)[len(other_gaps) // 2]
    return max_gap < min_gap_for_cliff * 1.5 or max_gap / max(median_other, 0.01) < 3.0


def choose_threshold(
    scores: list[float],
    base_threshold: float = 0.15,
    min_gap_for_cliff: float = 0.20,
    min_top_score: float = 0.50,
) -> ThresholdResult:
    """Picks an adaptive box-confidence threshold from a set of detection scores.

    `scores` are raw per-box confidences from Grounding DINO, in any order.
    `base_threshold` is the fixed floor (0.15) this falls back to whenever the
    evidence doesn't support raising it -- which is the common case, by design.

    Looks for a confidence cliff: a gap of at least `min_gap_for_cliff` between two
    consecutive sorted scores, with a top score of at least `min_top_score` (a weak
    top score means even the best detection is dubious, so the floor shouldn't be
    raised on its account) and a gap that isn't just the tail end of an otherwise
    shallow gradient rather than a real cliff. When a cliff qualifies, the candidate
    threshold sits midway between the two scores it separates, clamped so it never
    drops below `base_threshold` and never rises above the top score itself.
    """
    if not scores:
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.0,
            reason="no scores provided",
        )

    sorted_scores = sorted(scores, reverse=True)
    top_score = sorted_scores[0]

    if len(sorted_scores) == 1:
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.0,
            reason="single detection; no gap to measure",
        )

    if top_score < min_top_score:
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.2,
            reason=f"top score {top_score:.2f} < {min_top_score:.2f}; adaptation suppressed",
        )

    g = _gaps(sorted_scores)
    max_gap = max(g)
    max_gap_index = g.index(max_gap)

    if max_gap < min_gap_for_cliff:
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.3,
            reason=f"max gap {max_gap:.2f} < {min_gap_for_cliff:.2f}; no confidence cliff",
        )

    if _has_shallow_gradient(sorted_scores, min_gap_for_cliff):
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.3,
            reason="score distribution resembles shallow gradient; no clear cliff",
        )

    candidate = (sorted_scores[max_gap_index] + sorted_scores[max_gap_index + 1]) / 2.0
    adapted = max(base_threshold, min(candidate, top_score - 0.01))

    if adapted <= base_threshold:
        return ThresholdResult(
            threshold=base_threshold,
            used=False,
            confidence=0.2,
            reason=f"candidate {candidate:.2f} clamped to base {base_threshold:.2f}",
        )

    confidence = min(1.0, max_gap / top_score)
    return ThresholdResult(
        threshold=adapted,
        used=True,
        confidence=round(confidence, 2),
        reason=f"cliff at index {max_gap_index} (gap={max_gap:.2f}); candidate={candidate:.2f}, clamped={adapted:.2f}",
    )


__all__ = ["ThresholdResult", "choose_threshold"]
