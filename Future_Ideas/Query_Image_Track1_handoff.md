# Handoff: `query_image` Track 1 (bug fixes, no gate)

For: Sonnet 5.5, new session. Written 2026-10-31 by the Opus session that reviewed
`Future_Ideas/Query_Image_Improvement_Plan.md`. When this handoff and the plan's body disagree, this handoff wins.
The plan's **Review** section (R.1-R.6) is the source behind it; read R.2 and R.4 if anything here is unclear.

## Scope

Five changes in `src/fusion_vision_mcp/`, each with unit tests. **No change to any default prompt or to what
`query_image(src, question)` returns on a healthy call.** No benchmark gate: this is the plan's "Track 1 exception"
(bug fixes measured by unit tests and parity). Track 2 onward (regex, suffixes, consistency rule) is **out of scope**;
don't touch `VQA_TEXT_QUESTION`, `VQA_TEXT_SUFFIX`, `lfm_vqa_prompt`'s wording or `_vqa_consistency`.

Don't rename or change the signatures of `analysis._TEXT_QUESTION_RE` or `analysis.lfm_vqa_prompt`: VisionBench's
`benchmarks/vqa_control_cache.py` imports both (it runs tonight at 03:00).

Do them in this order. Line numbers were checked 2026-10-30; confirm before editing.

### 1A. `vqa_answers` falls back on `MemoryError` too

`analysis.py` `vqa_answers` (~:980), its `except (OSError, ImportError):` (~:992). Make it
`(OSError, ImportError, MemoryError)`, as `lfm_count_check`, `lfm_flat_check` and `vqa_count_estimate` already do
(`memory.ModelMemoryError` subclasses `MemoryError`, `memory.py:74`). Log which model answered (a `logger.warning`
on fallback is enough). If Moondream2 is then refused as well, that becomes a tool error; that's fine.

Test (`tests/test_vqa_answers.py`): an answerer whose `answer` raises `ModelMemoryError` falls back to `app.vqa.query`.

### 1B. One OCR reading shared between the answer and the control question

With `check_consistency=true`, `tools/vqa.py` `query_image` calls `vqa_answers` twice (question, then the control
question `"Looking carefully at this image, answer precisely: {question}"`). Each routed call runs
`ocr_pages(app, images, sharpened=False, lfm_read=False)` on the same images, giving the same reading twice.

- Add a keyword-only `ocr_readings: Sequence[dict[str, Any]] | None = None` to `vqa_answers`. When it is given, use it
  instead of calling `ocr_pages`.
- In `query_image`, compute the reading at most once, lazily: only if the question *or* the control question matches
  `_TEXT_QUESTION_RE`. Same arguments as today (`sharpened=False, lfm_read=False`). Pass it to both calls.
- **Do not** pass it to `_vqa_cross_check`. Its `ocr` branch calls `ocr_pages(app, [image])` with the defaults, which
  is the full `ocr` tool (sharpened pass plus LFM transcription), a different reading. That is the second opinion the
  cross-check exists for. Leave it as it is.

Tests:
- Under `check_consistency=true` on a text question, `ocr_pages` is called once with `sharpened=False,
  lfm_read=False` for the answer and the control together.
- The prompts sent to the answerer are byte-identical to what they are today, for both calls.
- An unrouted question still makes no OCR call.

### 1C(1). `question.py` count-name parsing

`names_for(COUNT, ...)` goes `_COUNT_PATTERN`, then `_trim_count_noun`, which cuts at the first `_COUNT_FILLER`
substring (`" are"`, `" is"`, ...), then a `_STOPWORDS` check on the first word. (The plan's `_COUNT_PATTERNS`,
`_clean_span` and `_STOP_WORDS` don't exist.) The bugs, checked live:

| Question | Today | Want |
| --- | --- | --- |
| How many matching pink vases stand on the mantelpiece? | `['matching pink vases stand']` | `['matching pink vases']` |
| How many heraldic figures appear inside the top shield? | `['heraldic figures appear']` | `['heraldic figures']` |
| How many image panels are shown side by side? | `None` | `['image panels']` |
| How many photo thumbnails are displayed along the right edge? | `None` | `['photo thumbnails']` |

Fix:
- Add finite verbs as **whole-word** boundaries: stand(s), appear(s), lead(s), flank(s), represent(s), make(s),
  surround(s), remain(s), rest(s), sit(s), lie(s), hang(s), hold(s), wear(s), carry/carries, form(s), touch(es),
  cross(es), overlap(s), border(s).
- A plain substring like `" stand"` would also cut "standing", the same way `" in"` already cuts "inside". Use word
  boundaries for the new terms. If you fix `" in"` too, add a test showing "inside" no longer cuts mid-word, and check
  that every existing test still passes.
- Let a first-word stopword ("image", "photo", "picture") through when another noun follows it ("image panels").
  Keep returning `None` when the stopword is the whole span ("How many are in this image?").

Tests in `tests/test_question.py`: the four rows above, "standing" not cut, and the existing cases unchanged.

### 1C(2). The cross-check's count uses the `count_objects` pipeline (user decision, R.6 Q3)

Today `_vqa_cross_check` (`analysis.py` ~:247) counts with `app.counter.detect_objects([image], names[0])[0]`: raw
Grounding DINO at 0.15, with none of the shipped pipeline. `count_objects` (`tools/detection.py` ~:391-425) does this:

1. `execute_detection(app, images, name, backend_preference="auto", threshold=COUNT_BOX_THRESHOLD,
   adaptive_threshold=True, candidates_floor=COUNT_RANGE_FLOOR, drop_group_box=True)`.
2. Per result: `drop_nested`, then `possibly_more` from `candidates_below`.
3. `crowd_range` only when `upper >= CROWD_RANGE_RUN_UPPER`.
4. `lfm_count_check` on the Grounding DINO path, or `lfm_flat_check` when `detector == "florence2"`.

Silhouette and consensus are separate options. Flat counts run on Florence-2, not SAM 3.

- Extract steps 1-4, the per-image core at the default threshold, into one shared helper. Put it in `analysis.py`,
  or wherever keeps the torch-free import rule (see Rules). Both `count_objects` and `_vqa_cross_check` call it.
- `count_objects` keeps its exact behaviour for every parameter combination: `clip_art`, a non-default threshold,
  `adaptive_threshold=False`, `lfm_check=False`, silhouette, consensus. The helper takes what it needs. The
  default-threshold guards (`threshold == COUNT_BOX_THRESHOLD and adaptive_threshold`) must stay exactly where they
  apply today.
- The cross-check returns `{"tool": "count_objects", "object", "count", "separable"}` as now. Add `crowd_range` /
  `possibly_more` only if that falls out naturally; don't redesign the payload.
- Tests (`tests/test_crosscheck.py`, `tests/test_count_check.py`):
  - The cross-check's count equals `count_objects`' count on the same fake detector output, including a nested-box
    case and a flat (Florence-2) case.
  - `count_objects`' existing tests pass unchanged.
- **Parity, live, after it:** run VisionBench `benchmarks/count_lfm_parity.py` and `benchmarks/count_flat_lfm_parity.py`
  (from `C:\AI\Projects\VisionBench`, with `C:\AI\MCP\FusionVisionMCP\.venv\Scripts\python.exe`). The previous
  results were 1,696/1,696 cached and 8/8 live, and 1,045/1,045 and 8/8. Before running them, read each script to
  check it calls the server's code path you changed. If it calls `lfm_count_check` directly, it won't exercise the
  helper. In that case say so in your report rather than claiming parity.

### 1C(3). `_size_measurement` uses `detect_objects`' detection: contract change, needs a fixture test

`_size_measurement` (`analysis.py` ~:294) uses `app.counter.detect_objects([image], name)[0]`. Mirror `detect_objects`
instead: `execute_detection(app, [image], name, backend_preference="auto", threshold=DETECT_BOX_THRESHOLD,
adaptive_threshold=False, locate=True)[0]`. Check `tools/detection.py`'s `detect_objects` for the exact call and copy
it.

This changes outputs. SAM 3 returns many candidates and only its first box is trusted (71-73%), so a max/min-by-area
pick across them may be noisier. Before you keep it:
- Add a server-level test on a fixture with known sizes (`tests/spatial_containment.png`, or a new synthetic PNG with
  three same-class objects of clearly different sizes) asserting the largest/smallest box is the right one.
- If it fails on SAM 3's candidates, stop: revert 1C(3) and report. Don't tune around it.

Leave `_spatial_measurement` alone. It mirrors `spatial_relations`, which stays on Grounding DINO + SAM2 by measurement.

### 1D. `structured_analysis`: reuse the `hand` detections (8 → 7 Grounding DINO passes)

**The plan's 1D is wrong; don't drop `head` or `finger`.** `inspection._PART_ANATOMY` turns arm, leg, head and finger
into observations. The real duplicate: `_check_anatomy` detects `hand` with `adaptive_threshold=True`, and
`_check_defect_probe` detects `hand` again with the same arguments (its probe parts are `["hand", "face"]`).

- Pass `_check_anatomy`'s `hand` result into `_check_defect_probe`, which then detects only `face`.
- If `_check_anatomy` didn't run or has no `hand` result, the probe detects `hand` as today.

Test (`tests/test_inspection.py`):
- A recording fake detector sees 7 calls, not 8. Assert the part list.
- `observations` and `anomalies` are identical to today on the same fake outputs. Compare against the output of the
  unmodified code path, captured in the test.

### Docs

Where a change alters what a caller sees, update the tool's MCP description (`tools/vqa.py`), `params.py` field text,
and this repo's `CLAUDE.md`, in its existing voice: dated, short, measured numbers only. This means the cross-check's
count and size now match `count_objects` / `detect_objects`, and `structured_analysis` is one pass faster. Don't add
numbers you didn't measure.

## Rules for this repo (the ones that bite)

- **Torch-free import.** `protocols`, `analysis`, `images`, `params`, `detection_policy` and `question` are on the
  eager import path. Take constants from `constants`, never from a model wrapper. Check after any import change:
  `.\.venv\Scripts\python.exe -c "import sys, fusion_vision_mcp; assert 'torch' not in sys.modules"`
- **Fast loop:** `.\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_server.py`. The full suite
  (`uv run --with pytest --with anyio pytest tests -q`) and the pre-commit hook load every model (~40 GB peak). Run
  them **only while no benchmark is running**. Check first:
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? CommandLine -match 'benchmarks\\|run_visionbench|pytest'`.
  **Tonight (2026-10-31) from 03:00, a VisionBench control-cache run (~75 min) uses the CPU.** Don't overlap it.
- Lint: `uvx ruff@0.16.1 check src tests` and `uvx ruff@0.16.1 format src tests`. Types:
  `uv run --with mypy --with types-requests --with scipy-stubs mypy src`.
- **Editable install:** source edits go live in the shared HTTP server (127.0.0.1:8765) on its next restart. Don't
  restart it. No `pyproject.toml` change is expected; if one happens, stop and report (it needs a reinstall).
- **Git:** stage by path, never `git commit -a` or `git add -A`; another chat edits this repo too.
  `Future_Ideas/Query_Image_Improvement_Plan.md` is untracked; leave it. Commit each step separately once its tests
  pass (the hook runs pytest, so only while no benchmark runs), ending the message with the co-author line.
  **Don't push.** The Opus session reviews the diff and pushes.
- Don't edit VisionBench files; running its two parity scripts is fine. Don't add Chinese-developed models. Don't use
  native vision on images; use the `fusionvision` MCP tools if you need to look at a fixture.
- If a step's test shows the plan's premise is wrong (as 1D's was), stop that step and report. Don't redesign it.

## Report back

Write the report in this file under a new `## Result` heading, then tell the user. Include:
- Per step: done, reverted, or skipped, with the commit hash.
- Test counts (fast loop and full suite) and the import-invariant check.
- Parity script results, with whether they exercise the new helper.
- The 1C(3) fixture result.
- Anything you found that contradicts this handoff.

## Result

Written 2026-10-31 by the Sonnet 5.5 session. All five steps done and committed on `main`; **not pushed**.

| Step | Status | Commit |
| --- | --- | --- |
| 1A `MemoryError` fallback | done | 6b46119 |
| 1B shared OCR reading | done (same commit as 1A: both touch `analysis.vqa_answers` and `tools/vqa.py`) | 6b46119 |
| 1C(1) count-name parsing | done | d9e8bc5 |
| 1C(2) cross-check count = `count_objects` pipeline | done | 16eb3b9 |
| 1C(3) size cross-check = `detect_objects` detection | done, kept (fixture passed) | e005b7f |
| 1D reuse `hand` detections (7 passes) | done, with the Docs changes (CLAUDE.md, `query_image` description) | 63bbf7f |

- **Tests.** Each commit's pre-commit hook ran the full suite (real models) with no benchmark running: 599 passed at
  1C(3); the 1D hook passed too (about 600 with the new test). Fast loop (`--ignore=tests/test_server.py`) 512 passed
  before the 1C(2) test fixes, all green after. Ruff and mypy clean. Torch-free import check: `ok`.
- **1B shape.** `analysis.text_question_readings(app, images, *questions)` returns the one reading (or None when no
  question is routed or there is no answerer; a failed read returns None so `vqa_answers` retries it inside its own
  fallback). Tests pin one `ocr_pages(sharpened=False, lfm_read=False)` call under `check_consistency`, byte-identical
  prompts for both calls, and no OCR call for an unrouted question.
- **1C(1).** All four table rows now parse as wanted. I made *every* filler a whole word (not only the new verbs), which
  also fixes `" do"` cutting "red dogs" and `" is"` cutting "island". Tests added in `tests/test_question.py`.
- **1C(2).** `analysis.finish_count` (per-image post-detection steps) and `count_image`; `count_objects` calls
  `finish_count` with its own threshold/adaptive/lfm_check, guards unchanged. Tests compare the cross-check's count and
  `separable` with the `count_objects` tool on the same fake detector output: nested boxes, plain boxes, a flat
  (Florence-2) case and a zero.
- **Parity scripts do not exercise the helper.** `count_lfm_parity.py` and `count_flat_lfm_parity.py` import and call
  `lfm_count_check` / `lfm_flat_check` / `drop_nested` and `execute_detection` directly, so they say nothing about
  `finish_count`. I ran `count_flat_lfm_parity.py logic`: 1,045 of 1,045 (as before; those functions were not changed).
  `count_lfm_parity.py logic` timed out at 500 s without output (it runs detection), so I have no fresh result for it. Live
  mode not run. The `finish_count` equivalence rests on the unit tests above, not on parity.
- **1C(3) fixture.** New `tests/size_three_circles.png` (red circles, radius 40/70/120). Live server test
  `test_size_cross_check_picks_the_extreme_circle_from_detect_objects_boxes` passed: with `detect_objects`' boxes (≥3),
  the max-area box is the big circle and the min-area box the small one. The test checks `detect_objects` boxes with the
  same max/min rule rather than calling `query_image`'s cross-check end to end.
- **1D.** Seven calls, order person, arm, leg, hand, head, finger, face. Observations and anomalies equal the unshared
  path's (computed in the test from `_check_anatomy` + `_check_defect_probe` run independently).

**Contradictions with the handoff / things to know**
- Pre-commit stashes unstaged files and takes ~11 minutes per commit; my first 1A/1B commit attempt and the first 1C(3)
  attempt passed the hooks but left no commit (cause unknown, retries worked). I edited `inspection.py` while one hook was
  running; the hook's tree differed from the commit by that unstaged file only.
- 1A and 1B are one commit, not two.
- The `finish_count` / `count_image` names and `text_question_readings` are new public helpers in `analysis.py`.
- VisionBench's `vqa_control_cache.py` imports (`_TEXT_QUESTION_RE`, `lfm_vqa_prompt`) are untouched.
