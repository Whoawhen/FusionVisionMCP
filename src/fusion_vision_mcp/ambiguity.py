"""Semantic ambiguity detection for object counting (Spec 15).

When independent detectors collapse overlapping instances into a single connected
region (such as petals on a flower or blades of a multi-part tool), reporting a
count of 1 without context misrepresents the scene.

This module analyzes detection, silhouette, and consensus evidence to explicitly
report semantic ambiguity (semantic_ambiguity: true) and clarify count semantics
(count_semantics: "minimum_visible_instances"), preventing callers from fabricating
or trusting collapsed counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AmbiguityResult:
    """Diagnostic result reporting semantic ambiguity in detection and counting.

    Attributes:
        ambiguous: True if the detection is semantically ambiguous (collapsed structure).
        reason: Human-readable explanation of why ambiguity was detected, or None.
        evidence: Concrete evidence points supporting the determination.
    """

    ambiguous: bool
    reason: str | None
    evidence: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ambiguous": self.ambiguous,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def check_semantic_ambiguity(result: dict[str, Any]) -> tuple[AmbiguityResult, str]:
    """Evaluate detection, silhouette, and consensus fields for semantic ambiguity.

    Analyzes a `count_objects` result dictionary containing keys like `count`,
    `separable`, `silhouette`, `consensus`, and `estimates`.

    Returns a tuple of `(AmbiguityResult, count_semantics)`.
    `count_semantics` is `"minimum_visible_instances"` when ambiguous,
    or `"measured_tally"` / `"unverified_tally"` when unambiguous.
    """
    count = result.get("count", 0)
    separable = result.get("separable", "unknown")
    silhouette = result.get("silhouette") or {}
    consensus = result.get("consensus") or {}
    estimates = result.get("estimates") or {}

    evidence: list[str] = []

    # Case 1: Separability is explicitly 'no' (canonical collapse)
    if separable == "no":
        by_radial = silhouette.get("by_radial")
        lobes = silhouette.get("lobes")
        outline_est = estimates.get("outline")

        evidence.append(f"Detector returned count={count} with separable='no'")
        if outline_est:
            evidence.append(f"Silhouette outline indicates {outline_est} repeated lobes")
        elif by_radial and isinstance(by_radial, (int, float)) and by_radial > 1:
            evidence.append(f"Radial notch analysis indicates {int(by_radial)} lobes")
        elif lobes and isinstance(lobes, int) and lobes > 1:
            evidence.append(f"Distance transform indicates {lobes} lobes")

        return (
            AmbiguityResult(
                ambiguous=True,
                reason=(
                    "Detector collapsed overlapping instances into a single connected region, "
                    "but outline geometry indicates composite structure"
                ),
                evidence=evidence,
            ),
            "minimum_visible_instances",
        )

    # Case 2: Consensus disagreement on singleton detection
    if count == 1 and consensus.get("agree") is False:
        region_count = consensus.get("region_label_count", 0)
        if isinstance(region_count, int) and region_count > 1:
            evidence.append(f"Primary detector found {count} instance")
            evidence.append(f"Dense region captioner found {region_count} instances with matching labels")
            return (
                AmbiguityResult(
                    ambiguous=True,
                    reason="Dense region captioner found multiple instances while detector collapsed to one",
                    evidence=evidence,
                ),
                "minimum_visible_instances",
            )

    # Clean non-ambiguous cases
    if isinstance(count, int) and count > 1 and separable == "yes":
        evidence.append(f"Detector successfully separated {count} discrete instances")
        return (
            AmbiguityResult(
                ambiguous=False,
                reason=None,
                evidence=evidence,
            ),
            "measured_tally",
        )

    if count == 1 and separable == "yes":
        evidence.append("Single isolated instance confirmed by silhouette analysis")
        return (
            AmbiguityResult(
                ambiguous=False,
                reason=None,
                evidence=evidence,
            ),
            "measured_tally",
        )

    # Default fallback for unverified / unknown cases
    return (
        AmbiguityResult(
            ambiguous=False,
            reason=None,
            evidence=[f"Count of {count} with separable='{separable}'"],
        ),
        "unverified_tally" if separable == "unknown" else "measured_tally",
    )


__all__ = ["AmbiguityResult", "check_semantic_ambiguity"]
