#  test_inspection_coco.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Live-model negative control for `inspection._check_anatomy` (Sprint 17, v0.8.1 plan).

`tests/test_inspection.py` only ever exercises `_check_anatomy` against a `_StubDetector`
returning hand-placed synthetic boxes on a blank image, and `tests/test_server.py`'s only
real-image `structured_analysis` test uses the flower fixture (`tests/sample.jpg`), which has zero
people and returns before the anatomy check ever runs (`persons == 0`). Neither test could ever
fail on real detector output, which is exactly how a hardcoded `corroborated=True` and an
unassociated, undeduped part tally shipped in Sprint 10 without anything catching it (see "What
the third vision pass measured" in `FusionVisionMCP_v0.8.1_Sprint_Plan.md`).

This module runs the real `analyze_inspection` -> `_check_anatomy` path against real Grounding
DINO inference (no mocks, consistent with `tests/test_granite_docling.py`'s pattern of live model
inference in the normal suite) over all 24 of `benchmarks/coco_annotations.json`'s person-bearing
COCO val2017 fixtures. That inference is expensive -- six detection queries per fixture -- so it
runs once in a module-scoped fixture and both tests read the same result.

The two tests say different things on purpose. `test_check_anatomy_raises_no_anomalies_on_real_coco_photos`
is the standing goal and still fails, so it stays `xfail(strict=True)`; its reason records exactly
what is left and why. `test_check_anatomy_residual_false_anomalies_are_the_documented_one` passes,
and pins the measured state so that a regression -- anything beyond the one known case -- breaks
the suite loudly instead of hiding inside the expected failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from coco_fixtures import coco_fixtures

from fusion_vision_mcp.grounding_dino import GroundingDino
from fusion_vision_mcp.inspection import analyze_inspection

#: The one photograph in the set that `_check_anatomy` still calls anomalous, and the reason.
#: A woman petting sheep: Grounding DINO answers the query `leg` with the sheep's legs as well as
#: hers, and several of those boxes lie inside her person box, so she tallies more legs than a
#: person has. Association cannot reach it (the boxes really are inside her box) and neither can
#: dedup (they are separate legs, not re-detections of one). See Sprint 18's commit message for
#: the measured alternatives, including the species-qualified `human leg` prompt that fixes this
#: image and creates new false anomalies elsewhere.
KNOWN_RESIDUAL = {"coco_sheep_4_000000094871"}


@pytest.fixture(scope="module")
def anomalies_per_fixture() -> dict[str, list[str]]:
    """Run the real anatomy check once over every person-bearing COCO fixture."""
    fixtures = [fx for fx in coco_fixtures() if "person" in fx.all_categories]
    assert len(fixtures) == 24

    class _MinimalApp:
        """The minimal `AppContext` stand-in `analyze_inspection` actually needs: a `.counter`."""

        def __init__(self, counter: GroundingDino) -> None:
            self.counter = counter

    app = _MinimalApp(GroundingDino())
    results = {}
    for fx in fixtures:
        _observations, anomalies = analyze_inspection(app, fx.image, [])
        results[fx.name] = [a.description for a in anomalies]
    return results


@pytest.mark.xfail(
    reason=(
        "One fixture of 24 still raises one anomaly, measured after Sprint 18's association and "
        "dedup: coco_sheep_4_000000094871, a woman petting sheep, whose person box contains "
        "several of the sheep's legs -- Grounding DINO answers 'leg' for any species. Sprint 17 "
        "measured 7/24 images and 9 anomalies (arm=0, leg=5, head=4, finger=0) on the "
        "pre-association code; per-person association and IoU dedup take that to 1/24 and 1 "
        "(leg=1). The remainder is not reachable by either: the boxes really are inside her "
        "person box, and they are separate legs rather than re-detections of one. Asking the "
        "detector for 'human leg' does clear this image, and was measured and rejected -- it "
        "raises limb recall enough to create new false anomalies in crowd scenes, and the only "
        "configuration reaching zero also needs the containment cutoff dropped to 0.4, a setting "
        "chosen by searching this negative control alone with no positive control able to "
        "constrain it. See benchmarks/results/inspection_coco_negative_control.csv and Sprint "
        "18's commit message."
    ),
    strict=True,
)
def test_check_anatomy_raises_no_anomalies_on_real_coco_photos(anomalies_per_fixture) -> None:
    """The standing goal: zero anomalies from `_check_anatomy` on real photographs of people.

    This is `neg_spotted_ball` applied to inspection -- the cheapest possible disproof, run for
    real against live Grounding DINO inference rather than a stub.
    """
    offenders = {name: descriptions for name, descriptions in anomalies_per_fixture.items() if descriptions}
    assert offenders == {}, (
        f"{len(offenders)}/{len(anomalies_per_fixture)} real COCO photographs of ordinary people "
        f"raised anomalies: {offenders}"
    )


def test_check_anatomy_residual_false_anomalies_are_the_documented_one(anomalies_per_fixture) -> None:
    """Pins the measured state, so a regression fails loudly instead of hiding in the xfail above.

    An expected failure absorbs any failure, including a much worse one. This assertion is the
    part that has to keep holding while the one above is still red.
    """
    offenders = {name for name, descriptions in anomalies_per_fixture.items() if descriptions}
    assert offenders == KNOWN_RESIDUAL, (
        f"anomalies on {sorted(offenders)}, expected only {sorted(KNOWN_RESIDUAL)} -- "
        f"full output: { {k: v for k, v in anomalies_per_fixture.items() if v} }"
    )

    descriptions = anomalies_per_fixture["coco_sheep_4_000000094871"]
    assert len(descriptions) == 1
    assert "legs" in descriptions[0]
