"""Unit tests for semantic ambiguity reporting (Spec 15).

These test the diagnostic ambiguity classification logic directly using synthetic
detection, silhouette, and consensus results without invoking vision models.
"""

from typing import Any

from fusion_vision_mcp.ambiguity import AmbiguityResult, check_semantic_ambiguity


class TestAmbiguityResult:
    def test_as_dict(self) -> None:
        result = AmbiguityResult(
            ambiguous=True,
            reason="Detector collapsed instances",
            evidence=["count=1", "separable='no'"],
        )
        as_dict = result.as_dict()
        assert as_dict == {
            "ambiguous": True,
            "reason": "Detector collapsed instances",
            "evidence": ["count=1", "separable='no'"],
        }


class TestCheckSemanticAmbiguity:
    def test_canonical_collapse_separable_no(self) -> None:
        data: dict[str, Any] = {
            "count": 1,
            "separable": "no",
            "silhouette": {"by_radial": 8, "lobes": 8},
            "estimates": {"outline": 8},
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is True
        assert count_semantics == "minimum_visible_instances"
        assert res.reason is not None
        assert "collapsed" in res.reason.lower()
        assert any("separable='no'" in item for item in res.evidence)
        assert any("8" in item for item in res.evidence)

    def test_collapse_without_estimates_uses_radial(self) -> None:
        data: dict[str, Any] = {
            "count": 1,
            "separable": "no",
            "silhouette": {"by_radial": 6},
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is True
        assert count_semantics == "minimum_visible_instances"
        assert any("6" in item for item in res.evidence)

    def test_consensus_disagreement_on_singleton(self) -> None:
        data: dict[str, Any] = {
            "count": 1,
            "separable": "yes",
            "consensus": {
                "detector_count": 1,
                "region_label_count": 4,
                "agree": False,
            },
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is True
        assert count_semantics == "minimum_visible_instances"
        assert res.reason is not None
        assert "Dense region captioner found multiple instances" in res.reason
        assert any("4 instances" in item for item in res.evidence)

    def test_cleanly_separated_instances(self) -> None:
        data: dict[str, Any] = {
            "count": 5,
            "separable": "yes",
            "consensus": {
                "detector_count": 5,
                "region_label_count": 5,
                "agree": True,
            },
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is False
        assert count_semantics == "measured_tally"
        assert res.reason is None
        assert any("separated 5" in item for item in res.evidence)

    def test_singleton_verified_separated(self) -> None:
        data: dict[str, Any] = {
            "count": 1,
            "separable": "yes",
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is False
        assert count_semantics == "measured_tally"
        assert res.reason is None
        assert any("isolated instance" in item for item in res.evidence)

    def test_unverified_fallback(self) -> None:
        data: dict[str, Any] = {
            "count": 3,
            "separable": "unknown",
        }
        res, count_semantics = check_semantic_ambiguity(data)
        assert res.ambiguous is False
        assert count_semantics == "unverified_tally"
        assert res.reason is None

