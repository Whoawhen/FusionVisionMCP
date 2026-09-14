# Adaptive threshold: the real-but-weaker-instance risk, precisely stated

Companion to `FusionVisionMCP_v0.8.0_Sprint_Plan.md` (Sprint 3 verification). Written up per your
request for something to run deeper analysis against, rather than my two failed ad hoc
reproduction attempts from the verification pass.

**Correction to how I described this last time:** I called it "theoretical, unresolved." That
undersold it. Below is a closed-form proof that the failure mode exists — given the right score
pair, `choose_threshold` is *guaranteed* to drop a real second instance, no randomness or luck
involved. What's actually open is not "can this happen" but **"how often does a real photograph
of two genuine instances of one class produce that score pair"** — a prevalence question, not an
existence question. That reframing is the main point of this doc; it changes what's worth testing.

## 1. The mechanism

`choose_threshold` (`src/fusion_vision_mcp/adaptive_threshold.py`) sorts all raw detection scores
descending, finds the single largest gap between consecutive scores, and — if that gap clears two
tests — sets the threshold at the midpoint of that gap. Everything below the midpoint is dropped.

The algorithm has no notion of "real instance" vs. "distractor." It only sees numbers. So if a
genuine second instance happens to score far enough below the top instance, the algorithm cannot
distinguish "gap between target and distractor cluster" from "gap between strong instance and weak
but real instance" — both look identical to it: one big gap, then a lower value.

## 2. Sufficient condition — corrected after checking it against the actual code

**This section was wrong in an earlier draft of this doc and I caught it by running the code, not
just deriving it on paper.** My first pass claimed the shallow-gradient check couldn't veto a large
enough gap. It can: `_has_shallow_gradient` rejects unless the max gap is *also* at least 3x the
**median of the other gaps** — a condition that depends on how the rest of the score list is
shaped, not just on `s0` and `s1`. Running `choose_threshold([0.85, 0.30, 0.05, 0.03])` (a 4-score
list) returns `used=False, reason="shallow gradient"` — it does *not* misfire, contrary to my first
draft's claim. Recorded here so the correction is visible rather than silently edited away.

The real question, then, is what the *tail* looks like in practice. Two things settle it:

**a) The ratio test is easy to satisfy once the tail has more than a couple of low scores**, because
`median_other_gap` shrinks fast as more low-value scores get packed into the same `[0, s1]` range.
Confirmed directly: `choose_threshold([0.85, 0.30, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])`
— same top pair, a 7-value tail instead of 2 — returns `used=True, threshold=0.575`, dropping the
real `0.30` detection. The 4-score version above and this 10-score version differ *only* in tail
density; that's the whole effect.

**b) Real Grounding DINO output has a dense tail, not a sparse one — checked on a real image, not
assumed.** Calling `post_process_grounded_object_detection(threshold=0.0, ...)` on `tests/sample.jpg`
with prompt `"person"` returns **900 raw scores** (the model's fixed query-slot count), with **223
of them in `[0.01, 0.05)` alone**. That is a dense tail by construction, every single call, on every
image, because 900 is fixed regardless of scene content. So condition (a) above is essentially
*always* satisfied on real output — the tail-density gate is not a practical filter in this
system, even though it is a real part of the algorithm.

**That collapses the practical question back down to just `s0` and `s1`.** Given the tail is
reliably dense in real usage, the sufficient condition for `choose_threshold` to drop a genuine
second instance is, in practice:

1. `s0 >= 0.50` (clears `min_top_score`)
2. `s0 - s1 >= 0.30` (clears `min_gap_for_cliff` with the margin needed for the tail-density
   argument in (a) to apply — verified above, not assumed)
3. `s0 > 2 * s1` (guarantees the target↔instance-2 gap is the single largest gap in the list, since
   every other real score is `<= s1`)

Under those three — which is now a claim checked against real 900-slot model output, not a paper
derivation — `choose_threshold` sets `threshold = (s0+s1)/2 > s1` and the real second instance is
dropped. **This can still misfire; it just needs the tail-density argument stated honestly rather
than asserted.**

## 3. Why my two empirical attempts didn't reproduce it

Both attempts (`verify_sprint3_weak_instance.py`-style: a large disc + a much smaller disc, radius
70 vs 18, then 70 vs 9) tried to manufacture the failure by making the second instance *physically*
smaller, on the assumption that smaller → lower confidence → eventually crosses the danger zone.
Given §2's corrected analysis, the tail-density condition was never the blocker in those runs (real
raw output is always dense) — the runs simply never landed `s0 > 2*s1` with `s0-s1 >= 0.30`.
Grounding DINO's confidence score is not a direct, controllable function of object size; shrinking
a synthetic disc doesn't reliably push its score into that zone. **Confidence isn't proportional to
size, and there's no way to hit a target `(s0, s1)` pair by guessing geometry** — which is exactly
why §4 below proposes measuring the `(s0, s1)` relationship directly instead of guessing at it.

## 4. What's actually worth testing (the real open question)

The existence proof is closed. What's open is **prevalence**: across realistic two-instance scenes,
how often does the genuine `(s0, s1)` pair for "same class, one prominent, one less so" actually land
in the `s0 >= 0.5, s0 - s1 >= 0.30, s0 > 2*s1` danger zone? If it's rare, the risk is real but low-
priority. If it's common — e.g. any time one instance is meaningfully occluded, smaller, or at a
worse angle than the other — it's a live bug affecting real counts, not an edge case.

**Suggested method:** stop trying to hand-construct one adversarial image. Instead, sweep a
parameter that plausibly affects model confidence (occlusion fraction, blur, size ratio, contrast
against background, partial crop at frame edge) across many two-instance synthetic scenes, call
`GroundingDino.detect_objects(..., adaptive_threshold=False)` to get raw scores for each, and
directly log every `(s0, s1)` pair. Then check what fraction of pairs satisfy the danger-zone
condition from §2. That gives an actual rate instead of one more anecdote.

A reasonable starting sweep, reusing `benchmarks/fixtures.py`'s `_canvas`/`_disc` helpers already in
the worktree:
- Fix instance 1 at a size/contrast known to score high (e.g. radius 70, matching existing fixtures).
- Vary instance 2 across: occlusion (0-80% covered by another shape), blur radius, size ratio
  (radius 70 down to 10), and desaturation/contrast against the canvas — one axis at a time first,
  then a couple of combinations.
- Record `(s0, s1, danger_zone: bool)` per run into a CSV (or straight into DuckDB, per your
  data-file rule) and look at which axis actually moves `s1` down while `s0` stays anchored, since
  that's precisely what the danger zone needs.

## 5. Mitigation hypotheses (not implemented — evaluate before building)

**One originally-listed idea is now known not to work and is omitted below.** A "require the noise
below the cliff to look like a cluster, not one item" mitigation (minimum count of scores below the
candidate threshold) sounded plausible before §2's real-data check — it isn't, because real raw
output always has ~200+ scores in the low tail regardless of scene content (900 fixed query slots).
A count-based gate would almost never reject anything; it doesn't discriminate the danger zone from
normal operation. Dropped rather than left in as a false lead.

What's left, roughly in order of how much they preserve the feature's value on the F9-style case it
was built for:

1. **Floor on the absolute value the cliff drops to, not just the gap size.** Only trust a cliff if
   `s1` (the score just below the gap) is itself close to the noise floor in absolute terms — e.g.
   `< 0.30` — rather than judging purely on how far below the top score it sits. This directly
   blocks the `s0=0.85, s1=0.30` case in §2. The open tension: F9's own distractor cluster (the case
   this feature exists to fix) needs its own score checked against whatever floor gets picked —
   log F9's actual distractor scores as part of the §4 sweep before choosing a number, since a floor
   set too low neuters the feature on its own motivating case.
2. **Gate on the ratio `s0/s1` rather than (or in addition to) the absolute gap.** §2's condition 3
   (`s0 > 2*s1`) is already doing the real work of separating "distractor much weaker than target"
   from "second instance somewhat weaker than target" — making that ratio the primary signal (with
   a tunable cutoff, not necessarily exactly 2x) may separate the two populations more cleanly than
   an absolute gap in points. Needs the same F9-distractor-score logging as (1) to calibrate against
   a real positive case, not just the synthetic danger-zone construction in §2.
3. **Do nothing, document the rate.** If §4's sweep shows the danger zone is rare on realistic
   inputs (e.g. requires occlusion so severe the object is barely there anyway), the honest
   conclusion may be that this is an acceptable, documented trade-off — same posture as the
   flower/petal case elsewhere in this project: a known, named limitation rather than a silently
   hidden one.

Whatever the sweep finds, it belongs in `CHANGELOG.md`'s existing Sprint 3 "residual risk" note —
either upgraded from "unresolved" to a measured rate with a citation, or a mitigation implemented
and verified against `benchmarks/` the same way every other threshold in this project has been.

## 6. First sweep: results and what they actually show

`benchmarks/adaptive_threshold_sweep.py` was built and run against §4's plan — five perturbation
axes (occlusion, blur, size ratio, contrast, edge-crop) plus a combination sweep and an F9
reference row, all using real Grounding DINO calls, output to
`benchmarks/results/adaptive_threshold_danger_zone_sweep.csv`. Good build. Its own headline number
(1/49 ≈ 2%, "mostly degenerate") is not the full picture — checked directly against the real
`choose_threshold`, not the script's simplified proxy, two things turned up:

- **The one flagged row is invalid, not just low-severity.** `edge_crop offset=140` places the
  second disc at `[512, 652]` on a 512px canvas — **0% of it is inside the frame.** The model
  correctly scored something that isn't there as not there. That's not the risk this sweep exists
  to measure.
- **The script's proxy check (`gap>=0.20 AND ratio>2 AND top>=0.50`) undercounts.** Re-running the
  real `choose_threshold` on the full 900-score raw output (not the CSV's truncated top-20) found
  `edge_crop offset=120` — disc **86% cropped, only ~14% visible** — genuinely triggers
  (`threshold=0.526 > s1=0.358`), even though the proxy called it safe (`ratio=1.94`, just under
  its hard `>2` cutoff). The real algorithm's shallow-gradient test depends on tail density, not a
  fixed ratio, so a flat cutoff was always going to be an approximation — this is where it broke.
  `offset=100` (~29% visible), by contrast, is genuinely safe for real: the algorithm's cliff lands
  between the *second* and *third* scores, not between the first and second, so both real instances
  survive.

**So the corrected picture, from hand-verification of the edge-crop axis only:** occlusion (up to
80%), blur (up to 10px on a 70px object), size ratio (down to 1:7), and contrast (down to 10%
saturation) produced zero hits even on inspection. The one non-degenerate real hit found so far
needs the object almost entirely cropped out of frame (~14% visible) to occur — a materially
narrower and more specific risk than "a real second instance can be misclassified," and not the
kind of case F9 or ordinary occlusion/blur represent.

**This is not yet a closed investigation** — only the edge-crop axis got a real (non-proxy)
recheck; the other four axes' "zero hits" still rest on the flawed proxy and haven't been
individually re-verified the same way. §7 below is the concrete, ordered task list to finish it.

## 7. Instructions to run this down to a conclusion

Execute in order. Each step names what "done" looks like so the next step (and verification) isn't
guessing.

**1. Replace the proxy check with the real function.** In
`benchmarks/adaptive_threshold_sweep.py`, `check_danger_zone()` currently re-implements a 3-condition
approximation of `choose_threshold`. Delete the approximation; call the real thing instead:

```python
from fusion_vision_mcp.adaptive_threshold import choose_threshold

def check_danger_zone(scores: list[float]) -> tuple[bool, float, float, str]:
    if len(scores) < 2:
        return False, 0.0, 0.0, "fewer than 2 scores"
    s0, s1 = scores[0], scores[1]
    result = choose_threshold(scores, base_threshold=0.15)
    in_danger = result.used and result.threshold > s1
    return in_danger, s0, s1, result.reason
```

Note the `result.threshold > s1` check — `used=True` alone isn't enough (see `offset=100` above,
where adaptation triggers but on a *different* gap and both real instances survive). Feed the full
raw score list (all ~900), not a truncated slice — the CSV's top-20-only logging was fine for
human inspection but is not enough to reproduce `choose_threshold`'s tail-density-dependent
decision; keep logging only the top 20 for readability, but compute `in_danger_zone` from the full
list before truncating.

**2. Make degenerate rows auditable, not something a reviewer has to compute by hand.** Add a
`visible_fraction` column: `1.0` for every scenario except `edge_crop`, where it's the fraction of
the disc's bounding box actually inside `[0, CANVAS]` (the computation used to catch `offset=140`
above — turn it into a helper instead of one-off arithmetic). Exclude rows with
`visible_fraction <= 0` from the prevalence denominator in `main()`'s summary; report them in a
separate "degenerate (not actually in frame)" count instead of folding them into either bucket.

**3. Re-run the full sweep** with the corrected check and the new column. Overwrite
`benchmarks/results/adaptive_threshold_danger_zone_sweep.csv` in place.

**4. Bisect the edge-crop boundary.** Current data only brackets the transition between
`offset=100` (safe, ~29% visible) and `offset=120` (triggers, ~14% visible). Add
`offset ∈ {105, 110, 115}` (visible fractions ~25%, ~21%, ~18%) to the sweep, using the corrected
check from step 1, and report the approximate visibility fraction at which the risk actually
starts.

**5. Confirm — don't assume — the other four axes are clean under the corrected check.** The
"zero hits" conclusion for occlusion/blur/size/contrast in §6 above is from my own spot inspection
of the CSV values, not a re-run through the real function the way edge-crop got. Step 3's full
re-run covers this automatically — just don't skip reading those rows on the assumption they'll
obviously still be zero. If any of them flip to a real hit, that changes the conclusion in step 6
from Branch A to Branch B below.

**6. Decide and document, branching on step 5's outcome:**

- **Branch A — only edge-crop produces real hits** (expected, based on the spot check already
  done): close this out as *characterized, not a general risk*. Update `CHANGELOG.md`'s Sprint 3
  residual-risk note (currently "unresolved") with: the actual mechanism (real second instance
  cropped to a sliver at the frame edge, not "partially occluded/blurry/smaller"), the visibility
  fraction boundary from step 4, and a reference to the sweep CSV. No mitigation needed — same
  posture as the flower/petal limitation elsewhere in this project: a measured, accepted, named
  edge case rather than a hidden one.
- **Branch B — another axis also produces a real hit under the corrected check**: do not close
  this as accepted. Move to evaluating mitigation hypothesis 1 or 2 from §5 above (absolute floor
  on `s1`, or a ratio-based gate), and validate any change against the *full* `benchmarks/` suite —
  every existing positive **and** negative control, plus the F9 fixture — before shipping it. A
  mitigation that closes this gap but breaks F9 (the case Sprint 3 exists for) is not acceptable;
  this project has shipped that exact mistake before (see the CLAUDE.md counting section on why
  0.125/0.10 thresholds were rejected despite scoring better on positives alone).

**7. Commit.** `benchmarks/adaptive_threshold_sweep.py`, the regenerated CSV, the `CHANGELOG.md`
update, and (Branch B only) any mitigation code with its own tests — following this project's
existing norm of one commit whose message states exactly what was tested and against which
fixture, the same granularity Sprint 3's own commits used.

This will be verified the same way every other sprint has been: independently re-run the corrected
sweep, spot-check a handful of rows directly against `choose_threshold` (not trusted from the CSV
alone), and check the `CHANGELOG.md` update actually matches the data before this counts as closed.
