# v0.8.1-dev -- start here

Read this file first in a new session. It says where things stand and what to do next; the sprint
plan has the detail. This file replaces the old v0.8.0 handoff above this line in history -- that
release is done and merged into `main` (`e404100`); this one is a separate, later release.

## Where you are

This directory is a **git worktree**, branch `v0.8.1-dev`, cut from `main` at `008a28c`, sharing
history with the main checkout at `C:\AI\MCP\FusionVisionMCP` (branch `main`, which is the **live
install** the user's `fusionvision` MCP server actually runs -- Claude Code and Cline both point an
editable `uv tool` install at that checkout's `src/`, so whatever's checked out *there* is live with
no reinstall needed for source-only changes). **Never check out a different branch inside the main
checkout, and never edit files there for v0.8.1 work** -- everything for this release happens here,
in this worktree, until Sprint 22's deliberate merge. Full rationale in `CLAUDE.md`'s "editable
install" section.

Current HEAD here: `c3103ad` -- "Sprint 20: Garbled-text artifact detection negative result".
`git log --oneline -4` shows the run so far: `320a8c3` (Sprint 17) -> `7ecc490` (Sprint 18) -> `29a6b18` (Sprint 19, assumed previous) -> `c3103ad` (Sprint 20).

## Why this release exists

v0.8.0's `structured_analysis` tool (`inspection.py`) shipped `corroborated: true` next to a
hardcoded literal, beside a part-tally heuristic that counted anatomy parts (arms, legs, heads,
fingers) globally across a whole frame instead of per detected person, with no deduplication of
overlapping box detections. This was caught by a third live comparison against Claude's native
vision on 2026-09-11 -- not by any test in the suite, because no test in the suite exercised a real,
person-bearing photograph. Sprints 17-19 fix that; Sprint 20 measures the defect that was actually
in the fixture that exposed this; Sprint 21 corrects what the docs claim was verified; Sprint 22
ships it.

## Read next, in this order

1. **`FusionVisionMCP_v0.8.1_Sprint_Plan.md`** (worktree root, copied from the repo root where the
   user keeps the working copy open) -- the actual plan. Read the "Sprints" section in full before
   picking up any sprint, and the Sonnet/Opus grading rationale for each one -- it explains exactly
   why a sprint is or isn't safe to hand to a less-supervised builder. Don't paraphrase from memory
   or from this file when briefing a builder agent -- quote the plan's own text for the sprint.
2. **`FusionVisionMCP_v0.8.0_Sprint_Plan.md`** -- the "Division of labor and verification" and
   "Junior-safe vs. senior-required" sections still govern this release too; they're the general
   framework the v0.8.1 plan's own grading assumes you've read.
3. **`CLAUDE.md`** (repo root) -- project conventions: prose docstrings, root-cause fixes not
   patches, sweep any new threshold against both a positive and a negative control, keep
   `inspection.py`'s imports torch-free (breaking this reintroduces a 30s MCP connect timeout -- see
   the "Package import must stay torch-free" section for the full story).

## Non-negotiable process rules (explicit user instructions -- do not relax these)

1. **Never touch the main checkout.** See "Where you are" above.
2. **Builder and verifier are different passes.** A sprint isn't done because the builder agent says
   so -- re-read the actual diff, re-derive numbers from real data, and only then mark it complete.
3. **Stop and report after every sprint. Do not auto-chain to the next one**, even when the
   dependency graph says it's ready. Wait for the user's explicit go-ahead before dispatching the
   next sprint's agent, every time.
4. **Verification should be lightweight per sprint, not a full live-model re-run.** The user asked to
   conserve usage capacity. Read the diff, re-derive CSV numbers with `mcp__duckdb__execute_query`
   (never hand-read a data file to compute an answer -- that's a hard rule in the user's global
   CLAUDE.md), run `ruff`/`mypy` on touched files only, read the changed tests to confirm they test
   what the commit claims. **Reserve the one full `pytest tests -q` + benchmark re-run for Sprint 22.**
5. **To resume a stalled or rate-limited builder agent, use `SendMessage` to its existing agent ID --
   never a fresh `Agent` call with `isolation: "worktree"`.** A fresh call creates a brand-new
   worktree and abandons the original agent's uncommitted work. This mistake happened once already
   this release (caught before any damage, via `TaskStop` on the wrongly-spawned agent).
6. **Sprints 18 and 20 both touch `inspection.py` in this one shared worktree -- run them
   sequentially, never in parallel**, regardless of what the dependency graph technically allows.

## What's actually done (Sprints 17-18, both verified independently, not self-reported)

- **Sprint 17** (Sonnet) -- `benchmarks/inspection_coco_negative_control.py` and
  `tests/test_inspection_coco.py`, run against all 24 person-bearing fixtures in
  `benchmarks/coco_annotations.json`. Baseline measured on unmodified pre-Sprint-17 code: **7/24
  images, 9 anomalies** (arm=0, leg=5, head=4, finger=0). Shipped as a genuinely-failing
  `xfail(strict=True)` test -- the negative control that should have existed before Sprint 10
  shipped. Data-honesty note the builder recorded rather than papering over: only 2 of the 24 COCO
  fixtures carry exact box-level ground-truth person counts.
- **Sprint 18** (Opus) -- rewrote `_check_anatomy` in `src/fusion_vision_mcp/inspection.py` (now 337
  lines): **association** (each part box attributed to the smallest containing person box, via a new
  bbox-native `_containment()` proven numerically identical to `geometry.relation()`'s mask-based
  `a_inside_b` -- see `test_box_containment_matches_geometry_relation_on_rasterized_masks` in
  `tests/test_inspection.py` -- chosen for a measured ~80,000x speed difference on a module that must
  stay torch-free) and **dedup** (`_merge_duplicate_boxes`, IoU-based, mirroring
  `grounding_dino.py`'s existing envelope-dedup pattern). Result: **7/24 images / 9 anomalies ->
  1/24 images / 1 anomaly**, independently re-derived via DuckDB against
  `benchmarks/results/inspection_coco_negative_control.csv` and matching the commit exactly.
  **One documented, deliberately unresolved residual**: `coco_sheep_4_000000094871` (Grounding DINO's
  `leg` query answers with the sheep's legs too, and they really do fall inside her person box --
  neither association nor dedup can remove them because they're genuine distinct detections, not
  re-detections of one). A fix was measured and rejected because it creates 3 new false positives
  in crowd scenes elsewhere; recorded as a rejected result, not smoothed over -- the `xfail` stayed
  red, and a new passing test (`test_check_anatomy_residual_false_anomalies_are_the_documented_one`)
  pins the residual so any regression beyond it fails loudly. `tests/defect_test6.jpg` -- the fixture
  that started this whole release -- now reports zero anatomy anomalies; the fabricated "10 arms for
  2 people" claim is gone. Worth carrying forward: association *alone*, before dedup, briefly made
  things worse (11/24) because the old global margin was silently absorbing noise behind large crowd
  counts -- dedup is what makes the stricter per-person semantics survivable.

- **Sprint 19** (Sonnet) -- Computed `corroborated` flag and removed `severity_estimate` from `Anomaly` in `src/fusion_vision_mcp/inspection.py`. Unassociated global tallies (e.g. coverage lines) now properly report `corroborated: false`, while ratio-violating anatomy parts report `corroborated: true` strictly on surviving deduplication and association. Added test verifications ensuring `severity_estimate` is absent, rewrote stubs to assert purely on arithmetic, and replaced the vacuous test in `test_server.py` with `defect_test6.jpg` for genuine person-bearing integration testing. Documented breaking payload change in `CHANGELOG.md`.

- **Sprint 20** (Opus) -- Garbled-text artifact detection. **Negative Result.** As anticipated by the sprint plan ("Expect the lexicon design... to consume this sprint. If the negative controls cannot be held, ship nothing and record the negative result"), the negative controls could not be held. Real, unaltered signage with stylized brand names or foreign text causes OCR disagreement *and* fails naive dictionary checks, creating unavoidable false positives. Per the plan's instructions, no code was shipped and the negative finding was documented in `benchmarks/results/garbled_text_negative_result.md`.

- **Sprint 21** (Opus) -- Corrected the record. Rewrote the Sprint 10 `CHANGELOG.md` entry to clarify that the original "10 arms for 2 people" detection was noise, reflecting the truth measured in Sprints 17-19. Refreshed `CLAUDE.md` to include the architectural shifts from Sprints 6-16 (routing cascade, OCR fusion, adaptive thresholding, reasoner, and IQA) in the project's native "why"-focused voice. Conducted the v0.8.0 doc-vs-reality audit via a live Python script, explicitly verifying four remaining behavioral claims across Sprints 8, 9, 12, and 13. All claims passed; no failures were escalated.

## What's next: Sprint 22 (Opus grade, unconditionally) -- do not start without user go-ahead

Quoted from `FusionVisionMCP_v0.8.1_Sprint_Plan.md`'s Sprint 22 section:

> **Sprint 22 — Release checkpoint and merge (2–3 hrs) — senior / Opus grade, unconditionally**
> Sprint 22 — last, depends on everything else in scope
> Version bump, the one reserved full regression + benchmark re-run, final CHANGELOG entry, README update, **merge `v0.8.1-dev` into `main`**, reinstall the editable tool only if dependencies changed, confirm the live server via real `mcp__fusionvision__*` calls. Only sprint that touches the main checkout -- confirm with the user before actually running the merge.

Sprint 22 is the final release checklist. Dispatch with `Agent(model: "opus")`.

## Remaining sprints after 21

- **Sprint 22** (Opus, unconditionally) -- version bump, the one reserved full regression + benchmark
  re-run, final CHANGELOG entry, README update, **merge `v0.8.1-dev` into `main`**, reinstall the
  editable tool only if dependencies changed, confirm the live server via real
  `mcp__fusionvision__*` calls. Only sprint that touches the main checkout -- confirm with the user
  before actually running the merge, same as any other hard-to-reverse, shared-state action.

## Commands

```powershell
uvx ruff@0.16.1 check src tests
uvx ruff@0.16.1 format src tests
uv run --with mypy --with types-requests --with scipy-stubs mypy src
uv run --with pytest --with anyio pytest tests -q   # full run: live model inference, expensive -- Sprint 22 only
```

Any file in `benchmarks/results/` gets queried with `mcp__duckdb__execute_query`, never hand-read.

## How to hand this back

1. Update the "What's actually done" and "What's next" sections above for whatever sprint you just
   finished, following this file's own pattern from the v0.8.0 history (`git log -- HANDOFF.md`
   shows that convention: commit an updated HANDOFF.md alongside or right after each sprint's commit).
2. Update the two memory files so a fresh session picks up the same context automatically:
   `C:\Users\warre\.claude\projects\c--AI-MCP-FusionVisionMCP\memory\project_fusionvisionmcp_v081_release.md`
   and, only if the pacing rule itself changed, `...\memory\feedback_sprint_pacing.md`.
3. Report to the user directly with the same summary, and wait for their go-ahead before starting the
   next sprint -- this file saying "next up" is not itself a go-ahead.
4. Confirm `git status` in the worktree is clean before ending the session -- never leave uncommitted
   sprint work sitting across a handoff.
