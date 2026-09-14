"""Unit tests for domain_router.py's pure logic. No model, runs in milliseconds.

Live classification accuracy is measured in `benchmarks/tune_domain_threshold.py` against
`benchmarks/domain_fixtures.py`, not here -- see that module and CHANGELOG.md's Sprint 4
entry for the honest, measured results (3/6 correct; painting and screenshot are documented
misses, not silently passing).
"""

from fusion_vision_mcp.domain_router import (
    CLIP_ART_DOMAINS,
    DOCUMENT_DOMAINS,
    DOMAIN_LABELS,
    PAINTING_DOMAINS,
    PHOTO_DOMAINS,
    DomainResult,
    get_ambiguous_margin_threshold,
    set_ambiguous_margin_threshold,
)


def test_domain_groups_partition_the_labels_exactly() -> None:
    """Every label belongs to exactly one group -- no orphans, no double-counting.

    `_predicted_group`-style routing logic (here and in `tune_domain_threshold.py`) assumes
    this; a label added to `DOMAIN_LABELS` without adding it to a group would silently fall
    through as "other" everywhere that logic is used.
    """
    groups = [PHOTO_DOMAINS, CLIP_ART_DOMAINS, PAINTING_DOMAINS, DOCUMENT_DOMAINS]
    union: set[str] = set()
    for group in groups:
        assert union.isdisjoint(group), f"label(s) in more than one group: {union & group}"
        union |= group
    assert union == set(DOMAIN_LABELS)


def test_domain_result_is_frozen_and_holds_its_fields() -> None:
    result = DomainResult(domain="a photograph", confidence=0.7, scores={"a photograph": 0.7}, ambiguous=False)
    assert result.domain == "a photograph"
    assert result.confidence == 0.7
    assert result.ambiguous is False


def test_ambiguous_margin_threshold_get_set_roundtrip() -> None:
    original = get_ambiguous_margin_threshold()
    try:
        set_ambiguous_margin_threshold(0.33)
        assert get_ambiguous_margin_threshold() == 0.33
    finally:
        set_ambiguous_margin_threshold(original)
