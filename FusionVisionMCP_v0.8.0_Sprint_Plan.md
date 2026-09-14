# FusionVisionMCP v0.8.0 — Sprint Plan

Companion to `FusionVisionMCP_v0.8.0_Agentic_Implementation.md`. That doc describes *what* to
build in 14 phases; this one re-sequences the same work into independently-completable sprints,
each sized to run in a few hours, each landing as its own commit on `v0.8.0-dev` with its own
passing test run — the same discipline the `detect_objects` blank-canvas increment already used.

All work happens in the isolated worktree at `C:\AI\MCP\FusionVisionMCP-v0.8-dev` (branch
`v0.8.0-dev`). The original checkout at `C:\AI\MCP\FusionVisionMCP` stays on `main`, running
v0.7.1, until a sprint (or a batch of them) is deliberately merged over.

A sprint is "**built**" when: the new/changed code has real unit tests (and a live model smoke
test where one applies), `mypy`/`ruff` are clean, the full suite passes, docs are updated, and
it's committed. A sprint is "**verified**" only after independent review — see below. No sprint
should be marked verified on the strength of "the code looks right" or "the tests pass" as
self-reported by whoever built it.

## Division of labor and verification

This plan is built by a different worker (human or another AI agent) than the one verifying it.
Claude (this project's established collaborator, with the full history of what's already shipped
and why) verifies each sprint independently before it counts as done. This split exists because
it already caught real problems once on this project: an earlier externally-authored "v7.1"
pass (see `CHANGELOG.md`'s v0.7.1 entry and the git history around it) reported six fixes as
complete and tested, and independent verification found one was a guaranteed crash never
actually run, one used the wrong model ID and drew a wrong conclusion, and three of the six
didn't work as claimed once checked against the real fixture. That is the standard this
verification step is holding every sprint to — not a rubber stamp.

**For whoever builds a sprint:**
- Work only inside `C:\AI\MCP\FusionVisionMCP-v0.8-dev`, on branch `v0.8.0-dev`. Never check out
  a different branch inside `C:\AI\MCP\FusionVisionMCP` — that directory is the live install the
  user runs daily, and changing its branch changes what's live immediately (see `CLAUDE.md`'s
  "editable install" section). Don't merge to `main` and don't re-run
  `uv tool install --editable . --force` yourself; that cutover happens deliberately, later.
- Follow this project's existing norms, documented in `CLAUDE.md` and visible throughout
  `CHANGELOG.md`: root-cause a failure rather than patching its symptom (the occlusion fix in
  v0.7.1 is the model example — the real bug was two layers away from where it first looked);
  never claim a fix or an improvement without a measured before/after; benchmark any new
  threshold or model choice against `benchmarks/fixtures.py`'s existing positive *and* negative
  controls, not positives alone; inspect a new model's actual card/API before writing loading
  code rather than guessing it.
- Commit each sprint on its own, with a message that states exactly what was tested and against
  which fixture — that's what makes verification fast rather than another full investigation.
- Match the existing docstring style: prose paragraphs explaining the *why*, like every other
  module (`geometry.py`, `layout.py`, `grounding_dino.py`, `query_policy.py`). Not Google-style
  `Args:`/`Returns:` blocks — `adaptive_threshold.py` used those and it was the one inconsistency
  verification flagged in Sprint 2.
- Don't mark a sprint's checkbox below as done. That's the verifier's call.

**What verification actually checks**, per sprint: `mypy`/`ruff`/the full test suite are re-run
independently (not trusted from the commit message); every specific behavioral claim in the
sprint's own description is live-tested against its named fixture, not assumed from the diff
reading correct; a case the sprint *doesn't* target is checked for a regression, not just the
target case itself; any new dependency or model claim is checked against the real thing, not the
model card's marketing. Findings — including partial fixes, unverified claims, or bugs — are
reported back plainly, the same way the v0.7.1 review above was.

## Junior-safe vs. senior-required

Every sprint below is tagged **junior** or **senior**, so routing work doesn't have to be
re-guessed each time and a junior builder isn't handed a task that quietly requires judgment
it can't supply. This came directly out of Sprint 4: the builder wasn't careless at execution —
the sweep ran, the numbers were real — the actual failure was being handed a *senior* problem
(invent a valid test methodology, be skeptical of your own result) shaped like a *junior* one
(fill in this data table), and defaulting to making the number look good rather than questioning
whether it measured anything. That is the specific failure mode this split exists to prevent.

**Junior-safe** requires all four:
1. **Ground truth already exists** — from a third party (a public dataset's own annotations), an
   established pattern elsewhere in this codebase, or fully pre-specified in the sprint's own
   text. The builder is never the one deciding what "correct" means.
2. **A close, existing pattern to follow** — not the first time this codebase has done this kind
   of thing. New-model integration is never junior-safe on the first pass: see Sprint 4's
   unnecessary `trust_remote_code` (copied from Moondream2's pattern without checking) and the
   earlier v0.7.1 review's wrong-model-ID finding.
3. **Success is mechanically checkable** — tests pass, numbers match a pre-stated target,
   byte-identical output — without interpretation.
4. **Contained blast radius** — doesn't become the shared default behavior of multiple other
   tools (Sprint 3's mistake), and never touches the `main` checkout / live-install boundary
   (Sprint 14 is senior for exactly this reason, every time).

**Senior-required** if any one of: inventing or validating what ground truth means for a new
kind of measurement; first integration of a new model whose real API isn't already known in this
codebase; choosing a threshold or tradeoff that isn't purely read off already-gathered data;
becoming a shared default across tools; a refactor whose "equivalent behavior" isn't trivially,
mechanically checkable end to end.

A junior-safe sprint still gets the exact same independent verification every sprint gets — the
tag changes who it's safe to hand the *building* to, never whether the result gets checked.

## Status

- [x] **Sprint 0** — `detect_objects` blank-canvas guard (`query_policy.py`). Built and verified
      (built by Claude directly this time). Committed `782df4b` on `v0.8.0-dev`.
      *Not yet merged to `main`.* **Retrospective tag: junior** — pre-specified exact-match
      vocabulary + area-ratio filter, no invented ground truth, contained to one tool.
- [x] **Sprint 1** — generic-query guard extended to `count_objects`'s plain path and
      `spatial_relations`. Built by another worker, verified by Claude: mypy/ruff/full suite
      re-run independently (189/189), the actual wiring read directly (correctly placed before
      the silhouette/consensus logic in `count_objects`, before best-box selection in
      `spatial_relations`), and the new tests' assertions checked against the code rather than
      trusted from the commit message. One gap noted, not blocking: no live-model fixture for the
      mixed "one real detection + one spurious full-frame box in the same call" case — covered by
      a synthetic unit test only. Committed `d075235` on `v0.8.0-dev`.
      *Neither Sprint 0 nor Sprint 1 is merged to `main` yet — still isolated in the worktree.*
      **Retrospective tag: junior** — same guard, two more call sites, exact pattern from
      Sprint 0. The one real gap (missing live-model fixture) is a coverage note, not a
      methodology failure — worth remembering junior-safe doesn't mean zero findings.
- [x] **Sprint 2** — `adaptive_threshold.py` pure logic module. Built by another worker, verified
      by Claude with two findings fixed rather than just flagged (`f06fb4b`):
      - The self-report claimed "mypy/ruff clean," but `ruff format --check` actually failed on
        two files — `ruff check` (linting) was clean, format wasn't. Fixed.
      - `test_adaptation_clamped_to_base` was named for the clamped-to-`base_threshold` branch in
        `choose_threshold`, but its body had three abandoned attempts left as comments and ended
        up asserting an unrelated, already-covered scenario — the clamping branch itself had zero
        test coverage despite 202 tests passing. Hand-traced the branch (logic is correct, only
        reachable when a caller raises `base_threshold` above the cliff's own midpoint) and
        replaced the test with one that actually forces it, plus a contrast test showing the same
        cliff adapting normally under the default base.
      - Also verified the module's core logic against the real F9 numbers (target 0.90 vs.
        distractors 0.19–0.32): correctly identifies the cliff and proposes threshold ≈0.61,
        comfortably separating the target from every distractor.
      - Style: this module's docstrings used Google-style `Args:`/`Returns:` blocks, which no
        other module in the codebase does — rewritten as prose (`913ce5d`), no behavior change.
      Commits `292eaa5` (build), `f06fb4b` (clamping test + formatting fix), `913ce5d` (docstring
      style), all on `v0.8.0-dev`. 203/203 tests pass, mypy/ruff (check *and* format) clean. Still
      pure logic only — not wired into any tool yet, as scoped; Sprint 3 benchmarks and wires it.
      **Retrospective tag: senior** — the algorithm itself was fully spec'd, but *writing correct
      tests for a new heuristic's boundary conditions* is a judgment call, and the one real gap
      found (a test that didn't test its own named branch) was exactly that kind of miss.
- [x] **Sprint 3** — Adaptive thresholding wired into `grounding_dino.py` and `count_objects`
      (`GroundingDino.detect_objects` gains `adaptive_threshold=True`, default on — meaning
      `count_objects`, `spatial_relations`, and the VQA cross-check all inherit it, since they
      share this one detector call; only `count_objects` exposes an explicit opt-out). Built by
      another worker, verified by Claude with one real gap closed rather than just flagged
      (`d18371e`, on top of the build commit):
      - Core claims hold, independently re-checked: the `threshold=0.0`-then-manual-filter
        approach is bit-identical to a direct call at the target threshold (checked on two real
        images); the star-among-18-distractors case, re-run against the *actual* fixture (not a
        reconstruction), goes from 6 (the documented F9 bug) to the correct 1; no bad interaction
        with the v0.7.1 occlusion/envelope fix — that case incidentally improves too (2 boxes
        including a loose duplicate → 1 clean box).
      - **Gap found and closed**: zero automated tests existed for any of this, despite the
        change becoming the *default* for the shared detection path four tools rely on — the
        "full benchmark suite" claim in the build commit was a real, independently-reproduced
        result, but nothing regression-tests it going forward. Added the real F9 fixture
        (`tests/count_cluttered_target.png`) and two `test_server.py` tests: the adaptive default
        isolates the target, and `adaptive_threshold=false` genuinely reproduces the old count.
      - **Documented, not resolved**: a theoretical risk that the algorithm could misread a
        genuinely real but less-confident second instance as a distractor. Proved algebraically
        that a cliff between the weakest real detection and the near-zero query-slot tail can
        never self-filter; two attempts to construct a cliff *between two real instances* didn't
        reproduce a failure, but nothing rules it out in general and nothing tests for it. Written
        up in `CHANGELOG.md` as an open risk to watch for, not swept under the rug.
      - Also noted the build's own checkbox had been pre-marked done before verification — this
        entry corrects that; per the "Division of labor" section above, that's the verifier's call.
      Commits `30e0436` (build) and `d18371e` (verification fixes), both on `v0.8.0-dev`. 205/205
      tests pass, mypy/ruff (check *and* format) clean.
      **Retrospective tag: senior** — this is the textbook case: shipping a default across four
      shared call sites and judging whether zero test coverage on that is acceptable is not a
      mechanically-checkable decision, and it was missed until verification caught it.
- [x] **Sprint 3b** — closed Sprint 3's residual-risk investigation (Branch A: characterize, no
      mitigation needed). Built by another worker following `ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md`
      §7, verified by Claude with three real gaps fixed rather than just flagged:
      - **Numbers independently reproduced**: F9's claimed `threshold=0.439` re-derived exactly
        via a direct real-model call; the benchmark harness re-run gave the same 9/10
        positives / 8/8 negatives; the `visible_fraction` values for the bisection points
        (offset=115 → 0.179, offset=120 → 0.143) matched hand-computed geometry. The sweep script
        itself was read line-by-line and does call the real `choose_threshold`, not a proxy, per
        §7 step 1's exact instructions — confirmed, not assumed.
      - **Gap found and fixed: nothing had been committed.** The self-report described the sweep
        as updated, re-run, and the CHANGELOG updated, but `git log` on `v0.8.0-dev` showed no new
        commit at all — everything was sitting untracked. Committed as `8d6a81f`.
      - **Gap found and fixed: the CHANGELOG edit landed in the wrong repository.** The Sprint 3b
        entry had been written into `C:\AI\MCP\FusionVisionMCP\CHANGELOG.md` — the **main
        checkout**, the live install this whole worktree workflow exists to keep undisturbed — and
        left uncommitted there, instead of the worktree's own `CHANGELOG.md`. Main's branch was
        never switched and nothing executable changed, so the live server itself wasn't affected,
        but it's exactly the kind of boundary slip the isolation setup exists to catch. Reverted
        from main (`git checkout -- CHANGELOG.md`), and two stray duplicate copies of the sweep
        script/CSV that had also been placed in main were removed (both untracked, both exact
        duplicates of what's now properly committed in the worktree). The entry was rewritten in
        place in the worktree's `CHANGELOG.md`, correctly positioned as a subsection of the
        existing Sprint 3 entry rather than before a heading that came earlier in the file.
      - **Gap found and fixed: lint wasn't clean.** `ruff check` had 3 real errors (an unsorted
        import block, two unused imports) and `ruff format --check` wanted one file reformatted —
        neither was mentioned in the self-report, and this project's own "built" bar requires both
        clean. Fixed with `ruff check --fix` + `ruff format`; re-verified clean after.
      - **Minor accuracy note**: the self-report's "56 unit tests pass" doesn't match anything —
        the actual full suite is 205 (unchanged from Sprint 3, since this sprint adds sweep
        tooling only and touches no test files, consistent with how other `benchmarks/` scripts in
        this project are kept outside pytest). Not a blocking issue, just recorded so it isn't
        repeated as fact.
      Commit `8d6a81f` on `v0.8.0-dev` (sweep script, CSV, CHANGELOG.md). 205/205 tests pass,
      ruff check + format clean (mypy unaffected — no `src/` changes). `main` confirmed still at
      `345ca5b`, untouched.
      **Retrospective tag: junior, but the process failures show why "junior-safe" needs a
      pre-flight/post-flight checklist, not just a clear spec** — the *content* was correct
      throughout (real function called correctly, geometry matched hand-computation) because the
      methodology had already been fully decided by the senior write-up (§7) before this was
      handed off; only mechanical execution hygiene (commit, repo boundary, lint) went wrong.
      Compare directly to Sprint 4 below, where the methodology itself was left for the builder
      to invent.
- [x] **Sprint 4** — `domain_router.py` (SigLIP2 zero-shot domain classification). Built by
      another worker; **first draft sent back, not verified**. Rather than send it back a second
      time for the substantive gap, Claude rebuilt the missing validation directly (the mechanical
      issues below were fixed the same way prior sprints' were; the fixture/ground-truth problem
      was real new work, done here rather than round-tripped):
      - Mechanical, same pattern as Sprint 3b: nothing committed in the worktree; a `CHANGELOG.md`
        edit landed in main again, left uncommitted; a stray copy of `domain_router.py` had also
        been placed directly in main's `src/` — the live install's actual source tree this time,
        not just docs/benchmarks — confirmed inert (nothing in main's `__init__.py` imports it, no
        dynamic/glob module loading exists to pick it up) so the live server was never affected,
        but a step past Sprint 3b's version of the same mistake. All reverted/removed from main.
        `ruff check` had 9 errors, `ruff format` wanted 3 files reformatted; `DomainResult`'s
        docstring claimed sigmoid while the code computed softmax; `classify_domain`'s docstring
        was Google-style (flagged once already in Sprint 2); `trust_remote_code=True` was carried
        over from the Moondream2 pattern without checking (model loads fine without it — confirmed
        directly). All fixed.
      - **The substantive gap**: the sweep behind "20/23 correct" tested zero painting/document/
        screenshot fixtures, and the one real photograph in the set (`flower()`, confirmed by
        reading `fixtures.py`: it's `Image.open(SAMPLE_IMAGE)`, the actual `tests/sample.jpg`) had
        its ground truth hardcoded to `clip_art` — not because that's true, but because that's
        what SigLIP2 already predicted. Ground truth was set to match the model's own output.
      - **Rebuilt as `benchmarks/domain_fixtures.py`** with honest labels and new minimal fixtures
        for the three missing domains. Real result: **3/6 correct**, and the misses are documented
        rather than hidden — `photo_flower` and `screenshot_ui` are *confidently* wrong (not
        borderline), and `painting_stylized` never got "oil painting" into the top 3 across six
        materially different synthetic constructions tried (posterize, saturation+blur+grain,
        colour quantization+blur, mode-filter smear, heavy blur, a synthetic landscape, brush-dab
        strokes) — a convergent negative result, same posture as this project's flower/petal
        counting finding, not one fixture that happened not to work.
      - **Margin threshold 0.15 kept, verified against real evidence rather than re-asserted**: the
        margin only gates the `ambiguous` flag and cannot change which label wins (softmax is a
        monotonic transform of the logits at every threshold — checked, not assumed). Swept
        `{0.05..0.50}`: `0.15`–`0.20` is the only band that flags both genuinely-uncertain cases
        without false-flagging the confidently-correct ones. **No threshold can ever flag the two
        confidently-wrong cases** — a structural limit of a margin-only signal, documented plainly
        for Sprint 5 rather than left implicit.
      - `benchmarks/test_domain_router.py` (print-only, no assertions, despite the `test_` name)
        deleted; real pure-logic tests added at `tests/test_domain_router.py`.
      Commit `0c8873d` on `v0.8.0-dev`. 208/208 tests pass, ruff check + format clean, mypy clean,
      counting benchmark suite unaffected (9/10 positives, 8/8 negatives). `main` confirmed still
      at `345ca5b`, untouched.

      **Open item for whoever picks up Sprint 5**: `photo_flower` (a real photo) currently routes
      to the clip-art path with 0.60 confidence and no ambiguity flag — the router cannot be
      trusted blindly for the photograph/clip-art split Sprint 5's routing decision depends on.
      Only one real photo was available to test here; that's a real limitation of this validation
      pass worth knowing before leaning on this router's output uncritically.

      **Process note, carried over from Sprint 3b and now a clear pattern across three sprints
      running (3, 3b, 4)**: self-reports have described work as committed when it wasn't, and
      CHANGELOG edits have twice landed in main instead of the worktree. Worth raising directly
      with whoever's building these rather than continuing to silently fix it each time.
      **Retrospective tag: mislabeled — this is the case study.** The model-loading half (follow
      Sprint 3's `grounding_dino.py` pattern, check the real model card) is junior-safe. The
      validation half — deciding what counts as ground truth for a domain the codebase has never
      classified before, and building fixtures nothing here previously needed — is squarely
      senior. Handed over as one undifferentiated task, it got a junior-shaped answer (ground
      truth quietly matched to the model's own output) to a senior-shaped question. Should have
      been split at the point where fixture *design* decisions started, with the split stated
      explicitly rather than left implicit in the sprint's own text.
- [ ] **Sprint 5** (senior) — depends on the domain router Sprint 4 just showed can't be trusted
      blindly on its most important split (photograph vs. clip_art). Routing design needs to
      account for that before this is junior-safe wiring; see the open item in Sprint 4 above and
      the reframing under Sprint 5's own entry below.
- [ ] Sprints 6–14 below: not started.

## Sprints

### Sprint 1 — Finish the generic-query guard (2 hrs) — junior
Extend `query_policy.is_generic_query` (already built) into `count_objects`'s plain (non
`clip_art`) path and `spatial_relations`'s per-object detection calls — the two other places
§8 of the spec doc names. No new modules, no new models. Regression test: a generic query on
the blank-canvas fixture returns nothing from all three tools, not just `detect_objects`.
*Junior-safe because*: exact existing pattern (Sprint 0), no invented ground truth, mechanically
checkable regression test.

### Sprint 2 — `adaptive_threshold.py`, pure logic only (2 hrs) — senior
Write `ThresholdResult` / `choose_threshold` (spec §9) as pure functions over a list of scores —
no wiring into any tool yet. Unit-test against synthetic score distributions: a clean gap (should
adapt), a shallow gradient (should not), a single detection (should not). This is safe to build
and test without touching a live model at all. *Senior because*: writing tests that actually
force a new heuristic's boundary conditions (not just any test that passes) is a judgment call —
this is exactly where Sprint 2 needed a real fix, see the Status entry above.

### Sprint 3 — Benchmark and wire adaptive thresholding (3–4 hrs) — senior
The part that actually needs rigor. Wire `choose_threshold` into `grounding_dino.py`'s call path,
sweep it against `benchmarks/fixtures.py` (the same suite `DEFAULT_BOX_THRESHOLD=0.15` was
originally tuned against) plus a new cluttered/distractor fixture (the star-among-18-shapes case,
F9). Ship only if every existing negative control still holds — mirror the rigor already in
`CLAUDE.md`'s counting section, not just "gap ≥ 0.20" from the spec doc's example. *Senior
because*: becomes the shared default across four tools — recognizing and owning that blast radius
is a judgment call, not a mechanical one (Sprint 3's actual gap was exactly this).

### Sprint 3b — Close the adaptive-threshold residual risk (2–3 hrs) — junior (methodology pre-decided)
Full instructions in `ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md` §7 — follow that doc directly rather
than re-deriving the task from this summary. In short: replace the sweep script's simplified
danger-zone proxy with a real `choose_threshold` call (the proxy is known to both over- and
under-count), add a `visible_fraction` column so degenerate off-canvas rows aren't silently folded
into the prevalence count, bisect the edge-crop boundary between the known safe (~29% visible) and
triggering (~14% visible) points, re-confirm the other four perturbation axes under the corrected
check rather than assuming they're still clean, then either document the characterized risk in
`CHANGELOG.md` (if only edge-crop reproduces it) or implement and fully re-validate a mitigation
against `benchmarks/`'s full positive-and-negative-control suite plus F9 (if it doesn't).
*Junior-safe because §7 already made every methodology decision* — the builder executes a
numbered procedure against pre-specified ground truth, doesn't design the test. If the sweep
surfaces a genuinely new, un-anticipated finding outside §7's branches, stop and flag rather than
improvise — that's the line back into senior territory.

### Sprint 4 — `domain_router.py` (3–4 hrs) — split: model loading is junior, validation design is senior
The first new-model sprint. Inspect the actual SigLIP2 model card and API before writing any
code (per the spec doc's own instruction — don't guess). Real smoke test loading
`google/siglip2-base-patch16-224`, lazy-loaded through the existing `idle.py`/`device.py`
conventions. Build `DomainResult` over `DOMAIN_LABELS`, and empirically choose the
confidence/margin cutoff for `ambiguous` — don't hardcode it, per the doc's explicit warning.
Test against photograph / clip-art / painting / document / screenshot fixtures (reuse
`benchmarks/fixtures.py`'s flower photo, the multi-object clip-art scene, and new minimal
fixtures for the rest).

This is where the project's history of new-model and new-threshold sprints has broken before, so
two things specifically:
- **Inspect the real `google/siglip2-base-patch16-224` model card and API before writing any
  loading code.** Don't guess the interface, the preprocessing, or the output shape from memory
  or from how a different CLIP-family model works — check the actual thing. (v0.7.1's F1 finding
  was reported "partial" at first purely because a test used the wrong model ID and drew a wrong
  conclusion from it.)
- **Don't hardcode the `ambiguous` confidence/margin cutoff.** Choose it empirically by sweeping
  it against real fixtures across all five domain classes, the same way `DEFAULT_BOX_THRESHOLD`
  and the adaptive-threshold constants were tuned — report the sweep numbers, not a guessed value,
  and hold every fixture (not just the ones that look good) the way Sprint 3b's edge-crop
  bisection did.

**Split explicitly, not left implicit — this is the sprint that taught this project why.**
`_load_siglip2()` (find the real model class, load it, following Sprint 3's model-loading
pattern) is junior-safe: a close existing template, mechanically checkable (does it load, does
inference run). *Everything about what the fixtures are and what ground truth means for them* is
senior: deciding what "photograph," "painting," "document," "screenshot" ground truth actually
looks like, recognizing when a measurement (like the first draft's) doesn't measure what it
claims to. Route those separately if this sprint (or one like it) is ever re-run or extended.

### Sprint 5 — Automatic clip-art routing (2 hrs) — senior (reframed after Sprint 4's finding)
`routing.py`'s `choose_count_backend`, wired into `count_objects`'s existing `clip_art` param:
explicit `clip_art` stays authoritative, `clip_art` omitted routes through the Sprint 4 domain
router, ambiguous domain falls back to today's consensus logic. Regression test: the
tree/house clip-art fixture, called *without* `clip_art=true`, now matches the already-verified
manual `clip_art=true` result.

**Reframe before building, given what Sprint 4 found**: the domain router got the one real photo
tested confidently wrong on exactly the photograph/clip-art split this sprint routes on, and no
margin threshold can flag a confident misclassification (see Sprint 4's Status entry). A cheap,
model-free statistic measured directly against this repo's fixtures — unique RGB colors per 1,000
pixels — separated every photograph (300+) from every synthetic/flat-art fixture (under 3) by
roughly two orders of magnitude, no exceptions, across a dozen-plus cases spanning clip-art,
painting, document, and screenshot fixtures. Consider deriving the photograph-vs-clip_art bit
from that kind of measured statistic rather than trusting SigLIP2's label outright, with SigLIP2's
5-way label demoted to descriptive metadata rather than the routing decision itself — same
posture `spatial_relations` and `count_objects` already take elsewhere in this project: measure,
don't blindly trust a single model's judgment call. *Senior because*: choosing the routing
architecture (which signal is authoritative, how they combine, where the ambiguous/fallback
boundary sits) is exactly the kind of tradeoff decision this project's split reserves for senior,
even though the wiring itself, once decided, is mechanical. **Known risk to test before trusting
the statistic**: a grayscale photograph would likely score low on unique-color-count and could
land in "synthetic" territory — add that as a negative control before relying on this, alongside
a heavily-posterized real photo and a screenshot with an embedded photo.

### Sprint 6 — Centralize the detection policy (3–4 hrs) — senior
Refactor `detect_objects` / `count_objects` / `spatial_relations` onto one shared
`prepare_detection_policy` helper (spec §11) so threshold/guard logic isn't duplicated three
ways. Pure refactor — the bar is the *full* existing test and benchmark suite passing with zero
behavior change outside the new auto-routing paths. This is the sprint most likely to hide a
subtle regression (compare to the ring/washer hole-fill trade-off and the "wood" full-frame case
found in v0.7.1) — budget time for a careful diff review, not just green tests. *Senior because*:
"equivalent behavior" across three call sites isn't trivially, mechanically checkable — it needs
a judgment-driven diff review, the plan's own text already flags this as the highest regression
risk in the whole release.

### Sprint 7 — `granite_docling.py` standalone (3–4 hrs) — senior (first pass), junior once a pattern exists
Second new-model sprint. Inspect `ibm-granite/granite-docling-258M`'s actual model card and the
currently-pinned `transformers` version before writing loading code. Lazy `load()`/`ocr()`, no
wiring into `caption`/`ocr` yet. Real smoke test on `tests/layout_small_text_paragraph.png`
(the exact fixture with known-exact ground truth from the v0.7.1 CHANGELOG), with measured
char/word accuracy against Florence-2's OCR head on the same image — numbers, not a guess. Add
as an optional `ocr-specialist` extra in `pyproject.toml`, never mandatory. *Senior because*: new
model, first integration — see the junior/senior framework's rule 2. Once this lands and there's
a working pattern to copy, a *later* new-model sprint that closely follows it could be junior-safe
the way Sprint 3b was — the model-loading half of Sprint 4 already showed that pattern-following
is the junior-safe part, not first discovery.

### Sprint 8 — OCR fusion (2–3 hrs) — junior
Wire Granite-Docling into `caption`'s new `auto_verify_text` flag (spec §13) and the small-text
upscale-crop pipeline (spec §14), building on `verify_text`/`corrections`/`textmatch.py` already
shipped. Regression test: the `FusionVisionMCP`/`FusionVisionMP` banner case must show the
disagreement and the correct fused text. *Junior-safe because*: wiring two already-built,
already-tested things together with the exact regression case pre-specified — assuming Sprint 7
lands clean first.

### Sprint 9 — Semantic ambiguity reporting (2 hrs) — junior
`ambiguity.py` (`AmbiguityResult`), wired into `count_objects`'s existing `separable`/silhouette
machinery — this is the smallest new-module sprint because it builds directly on logic already
well understood from v0.7.x. Regression test: the flower/petal fixture reports
`semantic_ambiguity: true` with `count_semantics: "minimum_visible_instances"`, never a fabricated
higher count. *Junior-safe because*: the regression case and its expected output are already
fully pre-specified, and the machinery being wired into is already well-understood in this
codebase.

### Sprint 10 — Structured visual inspection (3–4 hrs) — senior
`inspection.py` (`Observation`/`Anomaly`), wired into `query_image` behind `structured_analysis`.
Corroborate Moondream2's observations against Grounding DINO/SAM2/OCR/spatial/count evidence so an
uncorroborated VLM-only claim stays low-confidence. Reuse the six defect fixtures already
established for `check_consistency`'s testing. *Senior because*: deciding how much corroborating
evidence is "enough" to trust a VLM claim is a judgment call with no mechanical answer, the same
category of decision that went wrong in Sprint 4.

### Sprint 11 — Optional reasoning backend, scoped down (3–4 hrs) — senior
`reasoner.py`. Per the spec doc's own "only implement providers this project's architecture
actually supports" — Ollama is already installed on this machine (`C:\AI\Ollama\app`), which makes
`none` and `ollama` the two realistic provider modes for a first pass, not the full list in §19.
Structured input/output contract only (spec §19–20); don't chase reasoning quality yet. Direct
measurements must never be silently overridden — test that explicitly. *Senior because*:
architectural scoping decision (which providers are realistic given this project's actual
constraints) plus a correctness invariant ("never silently override a direct measurement") that
needs judgment to test properly, not just a fixture comparison.

### Sprint 12 — `image_quality.py`, one IQA model (2–4 hrs) — senior
Third new-model sprint. Evaluate exactly one candidate (MUSIQ / CLIP-IQA / a lightweight ONNX
model) on CPU RAM, latency, and accuracy — pick one, don't integrate several. Minimal `score()`
returning only what the chosen model actually provides (no invented submetrics, per spec §22).
*Senior because*: both a first-time model integration and a pick-one-of-several tradeoff decision
— two separate reasons this can't be junior-safe on its own.

### Sprint 13 — Aesthetic refactor (2–3 hrs) — senior
Conceptually split `technical_quality` / `photographic_aesthetic` / `artistic_judgment` in
`score_aesthetics`/`critique_composition`, using Sprint 12's IQA model and (if built) Sprint 11's
reasoner. Keep the existing `aesthetic_score` field byte-for-byte for compatibility — regression
test that historical scores are unchanged. Non-photographic media get
`photographic_aesthetic_applicable: false` rather than a misleading number. *Senior because*: a
conceptual split of an existing scoring scheme is a design decision, and "non-photographic" needs
a real definition — likely leaning on Sprint 4/5's domain routing, so inherits its caution too.

### Sprint 14 — Release checkpoint (2–3 hrs) — senior, always
The actual Phase 1 from the original doc, done last instead of first since this project is
building incrementally: version bump to `0.8.0` across `pyproject.toml`/`manifest.json`/
`server.json`, full `README.md`/`README_DETAILED.md`/`CLAUDE.md` documentation pass for
everything built in Sprints 0–13, full regression + benchmark run (spec §26–27's required
outcomes table), final `CHANGELOG.md` v0.8.0 entry with real fixture evidence. Merge
`v0.8.0-dev` into `main`, reinstall the live uv tool (new dependencies were added along the way),
confirm the live server through `mcp__fusionvision__*` calls same as Sprint 0's isolation proof.
*Senior, unconditionally*: this is the merge into `main`, the live install boundary — rule 4 of
the junior-safe framework rules this out categorically, independent of how mechanical the actual
steps are.

### Sprint 15 — Real-photograph ground truth from COCO `val2017` (3–4 hrs) — junior
Pull a curated subset (~30–40 images) from COCO's official `val2017` annotation set — a single
pre-packaged ~1GB download, not the full training set. Select for a spread of object counts and
scene complexity (not just easy cases), convert COCO's existing bounding-box/count annotations
into this project's `Fixture`-equivalent format as a new `benchmarks/coco_fixtures.py`, and
**check each selected image's license from COCO's own `annotations/instances_val2017.json`
`licenses` field before committing it** — keep only images whose license permits redistribution
in an open-source test repo (CC-BY, CC0, public domain); exclude any ND (No-Derivatives) variant,
since fixtures may get cropped/resized. Separately, using the same pulled images, compute and
report (do not interpret or threshold) the unique-RGB-colors-per-1000-pixels statistic from
Sprint 5's reframe across all of them, output as a CSV — the *measurement* is junior-safe because
the exact statistic and exact output format are fully pre-specified here; deciding whether it
holds up and picking a cutoff is explicitly left for senior review afterward, the same split that
worked cleanly in Sprint 3b. *Junior-safe because*: ground truth is COCO's own professional
annotation, not invented; the license check is a mechanical lookup against a provided field, not
a judgment call; the statistic computation follows an exact pre-specified formula with no
interpretation required.

### Sprint 16 — Real spatial-relationship ground truth from Open Images (3–4 hrs) — junior
`spatial_relations` is currently validated against exactly two synthetic fixtures
(`tests/spatial_touch_separate.png`, `tests/spatial_containment.png`) — this sprint gives it real
ground truth. Pull a small subset (~30–50 images) from Open Images' visual-relationship
annotations (`oidv6-train-annotations-vrd.csv` or the validation-split equivalent — use their
official per-class/per-relationship CSV filtering, not a bulk download) filtering specifically for
relationship types with an analogue in `geometry.py`'s vocabulary (contact/touching, containment/
"inside", clear separation) rather than pulling everything. Convert into a
`benchmarks/open_images_relation_fixtures.py` module recording each image's official relationship
triple as ground truth. **Check the license column Open Images provides per image before
committing any of them** — same posture as Sprint 15, exclude anything not clearly redistributable.
Run `spatial_relations` against each and report (not interpret) agreement with the official triple
as a CSV — same measure-don't-interpret split as Sprint 15. *Junior-safe because*: ground truth is
Open Images' own annotation, not invented; filtering criteria and license-check are both mechanical
and fully specified here; reporting agreement is arithmetic, not judgment. *If the agreement rate
comes back surprising (very high or very low), stop and flag rather than tuning anything* — that's
where this would cross back into senior territory, the same boundary Sprint 3b's instructions drew.

## Dependencies between sprints

```
Sprint 1 ── independent
Sprint 2 → Sprint 3 (needs the pure logic before it can be benchmarked)
Sprint 3 → Sprint 3b (closes out the risk Sprint 3 shipped documented-but-open)
Sprint 4 → Sprint 5 (routing needs the domain router)
Sprint 5 → Sprint 6 (centralizing the policy makes most sense once auto-routing exists)
Sprint 7 → Sprint 8 (fusion needs the specialist working standalone first)
Sprint 9 ── independent
Sprint 10 ── independent (benefits from Sprint 9's ambiguity vocabulary but doesn't require it)
Sprint 11 ── independent
Sprint 12 → Sprint 13 (refactor needs the IQA model to refactor around)
Sprint 14 ── last, depends on everything else that's in scope for this release
Sprint 15 ── independent (feeds Sprint 5's reframed routing decision, but doesn't block it)
Sprint 16 ── independent (feeds real validation for spatial_relations, no other dependency)
```

Sprints 1, 2, 9, 10, 11, 15, and 16 have no upstream dependency and can be done in any order or
picked up opportunistically — and 15/16 are junior-safe, so they're good default picks to keep a
junior builder moving on cheap, well-bounded work while a senior sprint is in progress elsewhere.
The three "new model" sprints (4, 7, 12) are the ones most likely to run long if the model's
actual API surprises — budget the high end of their range, and don't hand the first pass of any
of them to a junior builder (see the junior/senior framework above).

**Junior/senior tally, for quick reference when assigning work:** junior — 0, 1, 3b, 8, 9, 15, 16.
Senior — 2, 3, 4 (partially), 5, 6, 7 (first pass), 10, 11, 12, 13, 14. Senior clearly outnumbers
junior in what's left, which tracks: the sprints most amenable to junior execution (guard
extensions, data pulls with pre-existing ground truth) were mostly the ones already done or just
added. Most of what remains is genuinely design work.
