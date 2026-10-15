# FusionVisionMCP

Originally derived from [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2), renamed and
rebranded as **FusionVisionMCP** (and no longer a fork of it — see Remotes below) (Python package `fusion-vision-mcp`, module `fusion_vision_mcp`, binary
`fusion-vision-mcp.exe`), adding Moondream2 VQA and counting (`query_image`, `count_objects`), object grounding
(`detect_objects`, `dense_region_caption`), CLIP/LAION aesthetic scoring, batch analysis, and a `--memory-mode`
that controls how long models stay resident after use. See [README.md](README.md) for the tool and option
reference.

## Benchmarking and evaluation live in a separate project now

As of 2026-09-18, `benchmarks/` (the native Python harnesses, synthetic fixtures, JSON annotations,
COCO/Open Images directories, `run_visionbench.py`) moved to `C:\AI\Projects\VisionBench`, ported
verbatim (verified byte-for-byte identical at move time) and now developed independently there. This
repo keeps only what unit and smoke tests need: `tests/` still holds the fixture images
(`sample.jpg`, `defect_test6.jpg`, the `layout_*`/`spatial_*` synthetic PNGs, etc.) that the real unit
suite exercises. The three test files that depended on `benchmarks/` itself for ground-truth
evaluation (`test_coco_fixtures.py`, `test_inspection_coco.py`, `test_open_images_fixtures.py`) moved
with it — they live on in `VisionBench/tests` alongside the harness they test.

References to `benchmarks/` elsewhere in this file (constants tuning rationale, the counting and
spatial-relations sections below) describe *where a measurement was made*, not a path that still
exists in this checkout — treat them as historical citations pointing at VisionBench now.

## This is an editable install — checking out a branch changes the running server

`fusion-vision-mcp` is installed as an editable `uv` tool pointing at this checkout (`uv tool install --editable . --force ...`). The MCP server both Cline and Claude Code run **is whatever branch is checked out here**, live, with no reinstall needed to pick up source changes.

This bit us once already: checking out a branch without this fork's work silently removed the idle-release option and the Moondream tools from the running server, and it came back as `✘ Failed to connect` in both clients because that branch didn't accept the flag. The same trap applies to `--memory-mode`. Work lands on `main` now (see Remotes below), so the branch check that used to be needed no longer is — but the reinstall rule still holds: **re-run `uv tool install --editable . --force ...` after any change to `pyproject.toml`**. Source edits are live immediately; dependency and version changes are not until reinstalled.

Two practical notes from doing this in v0.8.2. The reinstall **deletes and recreates the tool's `Scripts`
directory, so it fails with `Access is denied` while any client still has a server running** — VS Code/Cline and
other agents hold `fusion-vision-mcp.exe` open. Stop those servers (or kill the `fusion-vision-mcp` processes)
first. And because `pyproject.toml` pins `torch>=2.13` with no upper bound, a reinstall can quietly move the
tool's torch (it went 2.13.0 → 2.14.0 in v0.8.2) while the project's own `.venv` — what the test suite runs
against — stays where it was. Run `uv sync --extra cpu --extra ocr-specialist --extra iqa` if you want the two environments on the same torch.

## `pyvips` is required transitively, not by this package directly

`moondream2`'s `trust_remote_code` module imports `pyvips`, which needs native libvips libraries that plain `pip install pyvips` does not provide. Without them the server fails at startup with `ModuleNotFoundError: No module named '_libvips'`, or `OSError: cannot load library 'libvips-42.dll'` if libvips isn't on `PATH`.

The fix already applied here: `pyproject.toml` declares `pyvips[binary]`, which ships the libvips shared libraries inside the wheel. No system libvips install and no `PATH` entry are needed. If you ever see either error, check that the dependency still reads `pyvips[binary]` rather than plain `pyvips` — this is exactly the failure that broke the upstream project's own test suite for anyone without a system-wide libvips.

## Commands

```powershell
# Reinstall after a dependency change (not needed for source-only edits)
# `[cpu]` is required: torch lives behind mutually exclusive cpu/cu130 extras, and naming
# neither installs no torch at all. The old --extra-index-url/--index-strategy flags are gone;
# pyproject's [tool.uv.sources] routes the index now, so every install path agrees.
uv tool install --editable ".[cpu,ocr-specialist,iqa]" --force

# Lint / format / type-check
uvx ruff@0.16.1 check src tests
uvx ruff@0.16.1 format src tests
uv run --with mypy --with types-requests --with scipy-stubs mypy src

# Tests. `testpaths` is set, so bare `pytest` collects only tests/.
# Integration tests spawn the real server and download Florence-2-base on first run.
uv run --with pytest --with anyio pytest tests -q

# Faster inner loop: skip the suite that loads real models (~ several minutes)
.\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_server.py

# The torch-free import invariant. This has broken twice; check it after touching imports.
.\.venv\Scripts\python.exe -c "import sys, fusion_vision_mcp; assert 'torch' not in sys.modules"
```

`uv run` and `uv tool install` can fail here in ways specific to the environment, not the code — see the OneDrive note below if this checkout is ever moved back under a synced folder.

## Remotes, and why attribution stays even though this is no longer a fork

`origin` is `https://github.com/Whoawhen/FusionVisionMCP` and `main` tracks `origin/main`. There is no
`upstream` remote and should not be: **this is no longer a fork.** The history was deliberately flattened (root
commit `44377fd`, "Flattened from v0.8.1") to sever the lineage from `jkawamoto/mcp-florence2`, which by then
meant nothing: 11 tools across five local models here against upstream's three on Florence-2 alone, and the
`mcp_florence2` package deleted outright early on and replaced by this repo's own `fusion_vision_mcp`.

**How that was done, because the obvious approach does not work.** Force-pushing flattened history does *not*
clear GitHub's fork flag — the fork relationship is repo metadata, not git history, so the "forked from" badge,
the fork network membership and the contribution-graph exclusion all survive it. What actually worked
(2026-09-16): rename the old repo to `Whoawhen/FusionVisionMCP-fork-archive`, create a fresh standalone repo
under the original name (so every URL in `manifest.json`, `server.json`, `README.md` and `_USER_AGENT` stays
valid), and push. The new repo verifies as `isFork: false` with no parent. Deleting rather than renaming would
work too but needs the `delete_repo` scope, which the local `gh` token does not carry.

The old 259-commit history survives in two independent places: the `FusionVisionMCP-fork-archive` repo, and a
verified bundle at `C:\AI\MCP\_backups\FusionVisionMCP-github-20260916.bundle`.

**Severing the fork relationship is not the same as dropping attribution, and the second one is not optional.**
Measured 2026-09-16 against upstream rather than assumed: `src/fusion_vision_mcp/florence2.py` still contains
**18 of upstream's 36 substantive lines byte-for-byte** — `ocr()` and `caption()` are character-identical, as is
the `AutoProcessor.from_pretrained(...)` call — and `src/fusion_vision_mcp/__main__.py` is a **verbatim copy**
apart from the import path. MIT grants everything else freely in exchange for exactly one condition: the
copyright notice travels with substantial portions of the software. Remove it while that code is present and
there is no license covering it at all.

So attribution is retained in four places, and none of them should be "cleaned up":

- `LICENSE` — byte-identical to upstream's, "Copyright (c) 2025 Junpei Kawamoto". **Do not let a repo-creation
  flow replace this with a fresh notice**; that is the single most common way this breaks.
- The MIT header at the top of `florence2.py` and `__main__.py` (restored 2026-09-16 after being lost in the
  flattening).
- The `authors` list in `pyproject.toml`.

`__init__.py` and `cli.py` share only two or three generic boilerplate lines with upstream (`import rich_click
as click`, a logger assignment) — not copyrightable expression, and they carry no header for that reason.

## `ocr` is PP-OCRv6 English, and it returns nothing rather than inventing text

`ocr` runs **PaddleOCR's PP-OCRv6 English models through RapidOCR** on ONNX Runtime (`ppocr_engine.py`, no
torch). It replaced EasyOCR on 2026-09-29, which had replaced Florence-2's `<OCR>` head before that. Measured
on VisionBench's 200-photo TextVQA set before shipping (`VisionBench/benchmarks/ocr_param_sweep.py`, results in
`benchmarks/results/ocr_engine_*.json`): answer-text recall 37.5% → **53.0%** end to end through the server,
58.6% vs 40.3% on text questions without column splitting. It also reads the invoice fixture exactly, where
EasyOCR produced `812.50` for $12.50 and `S11.00` for $11.00.

**2026-10-03: 58.5%, from the detector, not the recognizer.** Column zoning and the sideways re-read (see the
layout section) took TextVQA recall from 53.0% to 55.0%. Then `VisionBench/benchmarks/ppocr_det_sweep.py` swept
PP-OCR's *detector*, never tuned before, through the server's own `ocr_pages` with three controls: rendered pages
(exact lines), the 100 held-out sourced images by line, and the text-free fixtures. Lowering `Det.box_thresh`
(`OCR_DET_BOX_THRESHOLD`) from RapidOCR's document-tuned 0.5 climbs smoothly to a plateau — 55.0 → 57.0 (0.45) →
**58.5 (0.4)** → 58.0 (0.35, 0.3) — with every control held (rendered 97.5%, held-out lines 75.6 → 76.2%,
text-free empty). Rejected: a larger detector side length, `thresh` 0.2, `unclip` 2.0 and a 1.5x upscale gained
less or lost lines. The PP-OCRv5 server detector reads photos best (61%) but drops rendered pages to 87%; routing
it to photos only reached 62% while losing on the held-out images and rendered pages and doubling photo latency,
so it is a trade, not a gain, and was not shipped.

Four things about that swap are load-bearing:

- **No EasyOCR setting would have closed the gap.** Magnification (1.5x, 2x), beam-search decoding and turning
  off column splitting were all swept first: none beat EasyOCR's defaults on photos, and beam search dropped a
  third of a paragraph's words. The limit was the recognition model.
- **The recognizer must be the English one.** RapidOCR's default PP-OCRv4 model is Chinese/English and drops
  the spaces between English words ("Thisisasinglewideparagraph"). `ppocr_engine.py` pins `LangRec.EN` +
  `PP-OCRv6`; v4/v5 English are not valid combinations in rapidocr 3.9.
- **The 0.75 confidence floor (`OCR_MIN_CONFIDENCE`) is what keeps a text-free image empty.** Unfiltered,
  PP-OCRv6 read "S" (0.52) on `tests/sample.jpg` and "0" (0.57) on a shapes fixture, and garbled the last line
  of `layout_small_text_paragraph.png` at 0.70. Recall is flat from no floor to 0.70 and loses one item at 0.80.
  With the floor that last line is *missing* rather than garbled — the right failure, but still a gap.
- **EasyOCR is still installed, for two reasons.** `query_image(structured_analysis=true)` flags spans below
  0.20 as likely generated gibberish, and that threshold is calibrated to EasyOCR's scores (PP-OCRv6 gave real
  garbage 0.70), so it stays on `AppContext.ocr_specialist`. And the PP-OCRv6 recognizer is English-only, so
  any non-English `--ocr-languages` routes `ocr` back to EasyOCR (`AppContext.ocr_engine` is then the same proxy).

easyocr needs `opencv-python-headless` and rapidocr needs `opencv-python`; both install `cv2`, so
`[tool.uv] override-dependencies` drops `opencv-python`. Keep that override if either dependency moves.

`detail=true` returns `{text, text_regions}` per page, each region carrying `{text, confidence, box}` in page
coordinates. Because of the floor, low-confidence spans never reach it, so `detail` is not a gibberish detector.

**An image with no text returns empty, and that is correct.** This is worth stating because the repo's own test
suite asserted the opposite for four tests: they ran against `tests/sample.jpg` (a paper flower, no text at all)
and asserted `len(text) > 0`. That passed only because Florence-2's OCR head emitted *something* for a text-free
image — the assertions were pinning a hallucination. The `ocr` engine returns 0 regions there. `tests/test_server.py::test_ocr_textless_image_returns_no_text` now pins the
correct behaviour. Do not "fix" an empty OCR result by routing to a VLM; an empty result is an answer.

**`caption` describes text, it does not transcribe it.** Tested live on this repo's own banner (2026-08-25):
`caption` rendered the logo "FusionVisionMCP" as "FusionVisionMP" mid-sentence, while the OCR path read it
exactly right. A caption quoting a name, brand or label is not evidence of what it says — confirm it with `ocr`.
This is stated in `caption`'s MCP description, since a calling agent reads that at tool-selection time and not
this file. `caption(verify_text=true)` still uses Florence-2's `<OCR_WITH_REGION>` head for that cross-check —
the one remaining place Florence-2 reads text.

**2026-10-10: the caption now carries the `ocr` engine's text (`read_text`, on by default).** The captioners
paraphrase text and cannot read most of it, so `caption` runs `ocr_pages` and ends the caption with
`Visible text: "..."` (verbatim, reading order, ` / ` between lines, 40 words at most; `CAPTION_TEXT_WORDS`). A
text-free image gets no line, so its caption is unchanged. On VisionBench's 280 held-out images (variants fixed
before running, `caption_text_grounding.py`): default 53.3% -> 60.9% of key facts, `deep` 61.5% -> 66.3%, no image
worse, and none of the absent objects named came from the text. The line is appended *after* any `verify_text`
work, so `verify_text` checks only the prose's own quotes. Two things lost on the tune
split and were not shipped: asking Moondream2 for the image type first (-9.2 points; its answers are a third the
length of its long caption) and a language tag on the text (no recall gain, about 1 in 4 tags wrong).

**2026-10-12: `deep` stops when Moondream2 loops.** Its long caption is sampled (temperature 0.5, top_p 0.3) and
on ~7% of images ends in a loop ("effects, effects, ...", hundreds of "Í", a list of languages cycling) up to the
768-token cap. `Moondream.caption` now streams and stops once 16 consecutive 8-word spans repeat earlier text
(`repetition.py`, `LOOP_NGRAM`/`LOOP_RUN`), cutting the repeats off. Replayed on cached tune captions it cost 0.2
points (one group, hit only inside a loop). On 18 tune images that had looped, timed in one session against the
unstopped call: median 57.3 -> 40.7 s, 9 loops -> 0. Greedy decoding was tried first (VisionBench `caption_c4_0.py`)
and is *not* shipped: a tie on recall and speed, but it failed the loop guard fixed before the run. Seconds measured
in different sessions are not comparable here: the same captioner measured 39.3 s one day and 32.1 s the next.

## `critique_composition` is scored on a checklist now: a better subject box and a busy-background hint passed

Phase 7 scores the critique's prose by word overlap with frontier critiques, where even Claude, Gemini and GPT only
reach 29-34% against each other. VisionBench `composition_labels.py` built 150 COCO photos (50 tune / 100 confirm)
labelled with composition facts by the user's Gemini worker, and the user checked the worker blind twice: its subject
boxes (34 good / 4 loose / 0 wrong of 38) and background blur (89.7%) held up; "cut off", "awkward crop" and "busy"
did not. `composition_critique.py` then scored candidates with rules committed before each run (2026-10-11):

- **Shipped: `background: {busy, reliability}`** (`check_background`, default on, ~10 s). Moondream2 asked
  `BUSY_BACKGROUND_QUESTION` agreed with the user's own blind calls 88.1% (balanced) on 25 tune photos and **69.2%
  on 52 held-out** (CI 56.5-81.0), about as often as the worker agreed with the user. The user chose to state it as
  a hedged hint, so the reliability travels in the result.
- **Not shipped: background blur.** A Laplacian sharpness ratio (subject box vs surroundings) scored 77.6% on tune
  but its threshold was fitted on 7 blurred photos and fell to 61.4% held-out (CI 44.8-77.2). Moondream2's yes/no
  says "no" to nearly everything (54%).
- **Shipped after a fresh test: the subject box.** Without a `target_subject`, Moondream2 names the subject
  (`MAIN_SUBJECT_QUESTION`, returned as `subject`) and Florence-2 boxes the name, instead of the largest, most
  central dense region. IoU >= 0.5 against the worker's box: 79.5 vs 64.1% on tune (CI 0.0 to 30.8, short of the
  rule), 82.0 vs 66.3% held-out (CI 6.7 to 25.8), so the user asked for new photos first: on 120 fresh ones
  (`composition_box2`, tool boxes committed before the labels existed; the user found 0 of 40 boxes wrong, 18
  loose) **84.4 vs 75.0%, +9.4, CI 1.0 to 18.8**. The loose labels did not favour it (both methods' boxes match the
  labels' size; it led 20-18 on the good ones and tied 14-14 on the loose). About 10 s more per call. One guard
  was added *after* that test: a named box covering 95%+ of the frame (`_WHOLE_FRAME`) defers to the region pick,
  because a blank image came back as "a plain gray background" boxed edge to edge. On the cached boxes of all
  three sets it never cost a hit and gained two.
- **Untestable here:** subject in focus (3 "no" of 150) and awkward crop (no measurement matched the user's calls).

## Multi-column layouts are handled by geometry, not by asking the model harder

`ocr` reads strictly in raster order, so a document laid out in side-by-side columns (a two-column form, meeting
notes, a resume) gets its fields interleaved: a synthetic two-column fixture (`tests/layout_two_column.png`) came
back as `Attendee: Alice, Location: Room 4B, Attende: Ben, Duration: 45 min, ...` — alternating between two
unrelated columns line by line, with two names mangled in the process (2026-08-25).

**Re-prompting `query_image` to read the columns separately is not a reliable fix.** Four phrasings were tried —
asking for both columns in one call, asking for each column separately, insisting on an exact line count, asking
for a table — and `Deadline: Sept 10` never appeared in *any* of them, matching Moondream2's documented tendency
to paraphrase rather than exhaustively transcribe (see the OCR-routing note above). One phrasing even bled left-
column fields into the right-column answer.

**What ships instead: `layout.find_column_splits`/`reading_blocks` (`src/fusion_vision_mcp/layout.py`), pure
numpy/PIL, no model call.** It sums ink pixels per x-column, finds a vertical strip with near-zero ink density
that's wide enough and far enough from the edges to be a real gutter rather than word-spacing or a margin, and
crops there. `ocr` calls it automatically on every page: each column is OCR'd independently and the results
joined in reading order, so nothing has to reason about layout *and* transcribe exhaustively in the same call.
On the fixture above this recovers all ten fields in the correct order, including the one `query_image` never
produced.

Two details were load-bearing enough to test explicitly, both in `tests/test_layout.py`:

- **A ruled divider line down the middle of a real gutter must not defeat detection.** A thin (≤3px) ink run
  flanked by gutter on both sides is bridged and folded into the gutter, rather than being read as a second,
  narrower column boundary. `tests/layout_two_column_ruled.png` produces the identical split to the unruled
  version.
- **A page heading spans the full width above the columns** and would otherwise mask a real gutter that only
  starts below it — the top 15% of the image is excluded from gutter detection for exactly this reason.

**Negative controls, held before this shipped:** a table (`tests/layout_table.png`) has its own column gaps, but
every row carries ink in most columns, so there's no vertical strip blank across the whole body height — it is
correctly *not* split, and Test 1 in the original comparison (a clean invoice table) already showed both `ocr`
and `query_image` handle a proper table fine without this. A single wrapped paragraph
(`tests/layout_paragraph.png`) and a stray content sliver near the edge are also correctly left unsplit. The
approach generalizes past two columns without new code — `tests/layout_three_column.png` splits twice, at both
gutters.

**v0.7.1 found a second false-positive mode, and it was misdiagnosed at first.** A dense, small (11pt) single
paragraph came back scrambled into out-of-order word fragments, and a "Second Vision Pass" comparison against
Claude's native vision (2026-08-26) filed this as a Florence-2 *legibility* problem at small text scale.
Direct testing disproved that: calling Florence-2's OCR head on the un-upscaled original text, bypassing this
module entirely, reads it correctly. The real cause is the gutter detector above, misfiring on this specific
input: with only two of the paragraph's three lines landing inside the analyzed page body (the heading-exclusion
band at the top absorbs part of the first line), word-gaps in those two lines coincidentally lined up closely
enough to pass the gutter test, producing six spurious splits and slicing the paragraph into fragments as narrow
as 26px. Fixed by counting independent text lines first (`layout._count_line_bands`) and refusing to split when
there are more than one but fewer than three — too few for that kind of coincidental alignment to mean anything
real. Every existing multi-column fixture (five-plus lines) clears this floor easily; `layout_small_text_paragraph.png`
(committed from the exact fixture that exposed the bug) is the regression test.

**2026-10-02: a third mode, found by measuring rather than by a report — the split cut lines in half.** VisionBench
phase 14 renders 60 pages in a browser and takes every line's exact box from the DOM, so a split can be checked
against ground truth. Cropping the whole page height at each gutter cut **105 of 2,754 lines**: titles and intro
paragraphs above the columns, table headings, and on a slide `Bicycles` → `Bicycl` + `es`. Two causes. A gutter's
per-column tolerance scales with page height, so on a sparse page a strip passed it while running through the
rounded edges of letters; and a gutter only has to be blank below the top 15%, while plenty of full-width text
sits lower than that.

`reading_blocks` replaced `split_columns`. It cuts the page into horizontal zones on blank rows: a line with more
than two ink pixels in a gutter's interior is read across the full width, and a run of lines clear of every gutter
is split into columns, provided some column in it has three or more lines. A gutter must also be at least as wide
as the median line is tall (capped at 3% of the width), which stops a 10px word gap on a 1024px dashboard
qualifying. Cut lines went 105 → 3 (two invoice "Total due:" labels whose word gap lines up with a table gap, and
one accented word). Through the server: lines read verbatim 95.4% → 97.3%, character accuracy 98.8% → 99.6%,
regions sitting on a real line 96.4% → 99.3%, non-English verbatim 92.9% → 97.3%; real images unchanged at 75.3%.
Three things were load-bearing, each found by a regression on the rendered set: zone boundaries use *any* ink
(rule lines excepted), since the density floor leaves descenders outside a line and a boundary there cut it;
lines are counted per column, since columns at offset baselines leave no blank row across the page and merge
into one "line"; and the outer 2px of a gutter are not checked, since an opening quote mark's anti-aliasing at a
column's edge made a whole newspaper zone read full width.

**That fix eliminates the scrambling, but does not make Florence-2's OCR head perfect at small text.** Verified
end-to-end through the real MCP server (fresh subprocess, real protocol call, not a direct-Python import) on the
same fixture: the transcription now comes back in the correct order with no interleaving, but it still drops two
words ("font", "pixel"), duplicates one ("wide"), and merges two words across punctuation where a space should be
("scale,independent", "directlycheckable") against the fixture's known-exact source string. That is ordinary
small-text transcription error — a pre-existing, separate limitation of Florence-2's OCR head that this fix was
never intended to solve and does not claim to. Don't read "no longer scrambled" as "verbatim at small scale";
for text this small, a caller that needs the exact wording should still treat the output as approximate rather
than confirmed.

## `detect_objects` runs SAM 3 on photos, which needs transformers 5

**2026-10-12.** VisionBench sprint R (`Future_Ideas/detector_ocr_replacement_plan.md`) replaced Grounding DINO in
`detect_objects` on photographs with SAM 3 (`sam3.py`, `facebook/sam3`, threshold `LOCATE_BOX_THRESHOLD` 0.125).
On the held-out half it found 76.4% of labelled objects against 60.2%, and boxed a plausible-but-absent neighbour
on 9.0% of probes against 65.7%; its boxes are less precise (about a third land on a labelled instance on COCO).
**2026-10-14: flat images too.** VisionBench gate F (`sam3_flat_gate.py` screen on 63 images, then
`sam3_flat_confirm.py` on 88 fresh ones, rule fixed in `Future_Ideas/sam3_expansion_plan.md` before they existed):
on the confirm set SAM 3 boxed the absent object on 11.4% against Florence-2's 100%, and found 53.8% of labelled
objects against 40.5%; precision 22.4% vs 35.9%, ~29 s vs ~12 s per image, and 21 of 275 queries returned more than
20 boxes. "Flat" is the router's pixel test (under 10 colours per 1,000 pixels, not greyscale with tonal range),
which most JPEG screenshots fail, so the flat path is mostly clip art, logos, vector charts and lossless UI
captures. `detection_policy._flat_locate`; without SAM 3 it falls back to Florence-2. Counting on flat images
keeps Florence-2.

Every other Grounding DINO caller (`count_objects`, `spatial_relations`,
structured analysis's part crops and the defect probe) is unchanged.

**`spatial_relations` stays on Grounding DINO + SAM2: SAM 3 was shipped there and reverted the same day
(2026-10-13).** Gate S (VisionBench `sam3_spatial_gate.py`) tied it on 36 Open Images relations (25 each; SAM 3 12/12
on separate, 6/12 on contact). Round 12, 171 held-object pairs labelled by the worker and blind-checked by the user,
then showed the regression the 12 contact scenes were too few to see (`gd_sam2_held_compare.py`): on 106 held pairs
Grounding DINO + SAM2 agreed on 88, SAM 3 on 54 (+32 points, CI 23 to 42), 37 vs 22 of 39 with one of each object,
equal on separate pairs at a table, and twice as fast. SAM 3 fails both ways: its best-scoring match is a different
instance more often (labelled objects chosen 86 vs 59), and its instance masks stop a few pixels short of a hand.
Picking the closest near-best pair fixed held objects but broke separate scenes (`sam3_held_gate.py`); a 5 px contact
tolerance is waiting on a fresh set (round 12b). SAM 3's box with SAM2's mask (27/36 in gate S) was not shipped:
it removes no model, since SAM2 also backs `count_objects`' silhouette check and Grounding DINO its counts.

Four things are load-bearing:

- **`sam3.py`'s post-processing is what the gates measured**: the processor at 0.05, top 300, the 0.125 floor,
  class-agnostic NMS at IoU 0.5, then Grounding DINO's envelope filter. Change any step and re-run
  `replacement_gate2.py`.
- **SAM 3 does not count.** On the literal-name counting controls it returns no box for "ball", "rod", "shape" or
  "blob" at any threshold, and on real photos it counted worse than Grounding DINO at 0.425.
- **It is gated.** Without accepted access the load raises an `OSError` subclass; `detection_policy._photo_detect`
  then falls back to Grounding DINO at 0.35, and the result's `detector` says which ran.
- **It needs transformers 5**, and the old Moondream2 pin (2025-06-21) does not load there. Its commit `5d6c926`
  ships byte-identical weights, tokenizer and config and only fixes buffer initialisation, so
  `DEFAULT_MOONDREAM_REVISION` pins that commit. Meta's own `sam3` package was tried and not used: it is GPU-first
  (hardcoded CUDA, a module-level triton import, a missing vocabulary file) and its scores differ from the
  transformers port the gates measured.

**What transformers 5 changed, found by checking every model's outputs against 4.57** (VisionBench
`transformers5_equivalence.py`, 40 images, 2026-10-13). Moondream2, SAM2 and DINOv2 matched exactly as they were.
The rest needed fixes, each verified by a rerun:

- **Fast image processors are now the default.** CLIP's embeddings moved by up to 0.13 and Florence-2's captions
  changed. `aesthetic.py`, `florence2.py` and `grounding_dino.py` pass `use_fast=False`, as `dinov2.py` already did.
- **CLIP's `get_*_features` return an output object**, with the embedding in `pooler_output`; `score_aesthetics`
  crashed. `Aesthetic._projected` reads either.
- **Florence-2's tokenizer skips `clean_up_tokenization_spaces` for BPE**, so region text read "spa ., Nov". Forced
  back on in `florence2.py`.
- **Grounding DINO had two upstream bugs.** The phrase position ids give the closing "." position 0, and
  `encode_sinusoidal_position_embedding` casts its result to the integer ids' dtype, truncating the text position
  embeddings to 0s and 1s. Scores moved by up to 0.05 and box counts changed. `grounding_dino.py` restores the 4.57
  position ids and casts the ids to float (module-level patches, with tests).

After the fixes CLIP, Florence-2 and the three above match exactly. **Grounding DINO does not quite**: box counts are
identical on all 80 calls and boxes within 0.08 px, but scores differ by a median 2e-4 and at most 0.012, against
a limit of 1e-4 fixed before the run. That residue starts in transformers 5's rewritten Swin backbone (feature
differences around 50x float32 epsilon, eager attention barely changes them), so it is arithmetic order rather than
a logic change. Replayed on all 719 of gate 1's confirm queries (`gd_tf5_decision_replay.py`, 1,110 scored queries at 0.35 / 0.425), every metric was unchanged (exact counts 61.8%, detect recall 60.2%, controls 8/8 under both prompts) and one box count moved (a chandelier, 3 -> 2). `sam3.py` reproduces the gates'
replayed boxes on 20 of 20 confirm queries (`sam3_wrapper_parity.py`).

## Counting is a separate tool from detection, and it is still not solved

`detect_objects`' region count was never a tally (see the `wing` and `sword blade` cases in
[README_DETAILED.md](README_DETAILED.md#detect_objects-)), so `count_objects` routes "how many" elsewhere.

**Grounding DINO backs it, chosen by measurement rather than taste.** Moondream2's detect head held the job
first; both were built and benchmarked against the same cases before either shipped. They tie on eight shapes
separated (8) and touching (8), but Moondream collapses to **1** when those same *separated* shapes are asked
for as `pink circle` instead of `petal`, where Grounding DINO still returns 8 — the class-name sensitivity is
gone. It is also ~4x faster (2.1s vs 8.6s per call). Both pass the negative controls (one blob → 1, a rod → 1).

Grounding DINO returns one box around the whole arrangement *in addition to* the instances — 68% of the frame
when they are separated, and carrying the **highest** score, so a confidence cut removes the real instances
first. `grounding_dino.py` drops boxes that swallow most of the others' centres and reports
`group_boxes_dropped`. Without that filter every count is one too many.

**Tune counting against `benchmarks/`, never against one image.** The fixture suite draws its synthetic cases
programmatically, so their counts are exact by construction, and it includes eight negative controls that are
each exactly one object. Two settings were chosen from it and should not be nudged without re-running it:
the box threshold (0.25 → **0.15**) and the envelope filter's floor (4 → **3** boxes, since two instances plus
their envelope is only three). Together those took exact positives from 6/10 to 9/10 and mean absolute count
error from 1.07 to 0.10, holding all eight controls.

The sweep is also the clearest illustration of why this repo insists on negative controls. Thresholds of 0.125
and 0.10 score a **perfect 10/10 on positives** — and fragment a spotted ball into 3–4 objects and a rough-edged
rod into 2. Optimising on positives alone would have shipped exactly the failure mode the project exists to
avoid. 0.15 is the *lowest* threshold at which every control still holds, and that is the reason it is the
default.

**2026-10-12: `count_objects` now counts at 0.425 (`COUNT_BOX_THRESHOLD`), not 0.15.** 0.15 was right for the
fixtures and wrong for photos, where counts ran high. VisionBench sprint R picked 0.425 on the tune half of its
photo counts under the same rule (most exact counts with all 8 controls held). On 427 held-out counts it went
43.8% -> 61.8% exact (+18.0, CI 14.1 to 22.2), with the controls holding under both the "petal" and literal-name
prompts (positives 8/10 -> 7/10). `DEFAULT_BOX_THRESHOLD` stays 0.15 for every other Grounding DINO caller
(spatial relations, structured analysis's part crops, the defect probe's hand and face boxes), because those were
not re-measured. Gate 2 also showed SAM 3 cannot take over counting: on the literal-name controls it returns 0
boxes for "ball", "rod", "shape" and "blob" at every threshold.

**The honest negative result: the flower is not counted by anything here.** `tests/sample.jpg` — a paper flower
with overlapping petals — returns 1 from *every* approach tried: Florence-2 grounding, Moondream's detect head,
Grounding DINO (at every box threshold down to 0.10), the `count_lobes` outline measurement, SAM2 in
segment-everything mode (one mask for the whole flower), and a CIELAB interior-colour experiment.

Two measurements explain it, and together they close the case. Its silhouette has **solidity 0.984**, so the
outline is essentially a smooth disc — nothing there. And its interior colour boundary strength is **0.87**,
barely above a plain textured blob's 0.52, because the petals are pastel and low-contrast; the interior
experiment returns 1 even with its validity gate disabled entirely, so that is an absence of signal rather than
a threshold rejecting one. See `benchmarks/interior_structure.py`, which is kept *only* to make this
reproducible. Don't spend another pass on silhouettes, detectors, or colour for this image.

That experiment also failed the controls it had to pass: a striped ball came back as **7** objects and a
four-colour logo as **4**. Interior colour cannot distinguish a pattern from a group of parts, and fixing that
would still not deliver the flower, because the contrast is not there to begin with.

**CountGD was ported and measured, and is worse than what ships.** It was the counting plan's recommended
candidate, and the port turned out easy: its checkpoint is Grounding DINO Swin-B plus a single 1x1 conv for
visual exemplars, so HF's own conversion mapping loads it onto stock `GroundingDinoForObjectDetection` with zero
substantive missing keys, CPU-only, no compiled ops — none of the GCC/CUDA/Python-3.9 apparatus its repository
demands. On identical fixtures it scores 8/10 positives against 9/10, holds **4 of 8** negative controls against
8/8, and triples the mean count error. It misses a rough-edged rod entirely (0 objects), and the flower comes
back as 11, 4 and 5 at three resolutions of the *same image* — a spread that is noise, not a count. Rejected;
see `benchmarks/countgd_spike.py`, which is kept only to make that reproducible.

**Tiling and exemplar prompting were also tried, and fail the same way.** Tiled inference (2x2 and 3x3
overlapping crops, merged by IoU) buys one point on the heavy-overlap case and costs 5-12x the latency while
dropping to 7/8 and then 5/8 negative controls. CountGD's visual-exemplar path, reimplemented onto the HF graph,
breaks five of six controls. See `benchmarks/tiling.py` and `benchmarks/countgd_exemplar.py`.

**All four rejected approaches failed in exactly the same place, and it is worth naming.** Interior colour,
tiled crops, and exemplar matching each raise recall by attending to sub-object detail — and each one turns the
**spotted ball**, a single object covered in high-contrast spots, into many objects: 6 at 2x2 tiling, 15 at 3x3,
and 31 with an exemplar. Any future method that zooms in, matches appearance, or reads interior contrast will
meet the same wall. A spot, a stripe, a logo segment and a petal are the same thing to a detector; what
separates them is knowing what the object *is*, which is the calling model's job and not a measurement this
server can make. Test any new idea against `neg_spotted_ball` first — it is the cheapest possible disproof.

One trap the text-only spike recorded is worth knowing generally: **safetensors deduplicates shared tensors.** CountGD's
six `bbox_embed` heads are stored once with the aliases in `__metadata__`, so a naive `load_file` leaves 66
decoder box-head tensors missing and randomly initialised. HF ties those weights, so the outputs looked fine
anyway — restore aliases from the metadata before converting any checkpoint, and check *substantive* missing
keys rather than trusting that inference produced plausible numbers.

Two rules follow, both stated in the tool's own MCP description because a calling agent reads that and not this
file. A low count on something expected to be many means *"could not separate them"*, not a tally. And `count`
is never rewritten by the silhouette check — `count: 1` beside `silhouette.lobes: 8` reports two methods
disagreeing, which is information, rather than hiding one behind the other.

The Moondream pin still moved `2025-01-09` → `2025-06-21` (`src/fusion_vision_mcp/moondream.py`) and stays there
for `query_image`: it fixed a measured *self-consistency* failure, where asked to count the petals in
`tests/sample.jpg` the old pin answered 12 then listed 6 colors for them, and the new pin answers 10 and lists
exactly 10.

## `count_lobes` splits a silhouette, and the pixel floor is what makes it safe

`geometry.count_lobes` estimates how many repeated parts compose one mask, for when a detector collapses a group
into a single region. Two estimators, reported side by side and never reconciled: a distance-transform level
sweep that requires a count to hold across a *contiguous run* of levels (persistence without a merge tree), and
a hull-residual angular harmonic that only speaks for rosettes and returns `0` for *not measured* otherwise.

The hull step is not decoration. A raw radius profile reports a **square as four lobes** and a rod as two,
because those shapes are genuinely non-circular; dividing by the convex hull's radius measures *concavity*
instead, so every convex shape scores zero by construction and the negative controls pass structurally rather
than by tuning a threshold.

The constant that actually matters was found by sweep, not judgement: `_SMOOTH_FLOOR_PX`. A rod with ±2px edge
jitter — standing in for a photographic silhouette — **splits into 8 spurious lobes at a floor of 1.0px** and
holds at 1 from 2.0px upward, at every sigma tried. The scale-relative `_EDT_SMOOTH` term cannot defend against
this, because a tenth of a thin object's inradius is under a pixel. Raising sigma instead of the floor breaks
the positives: 8 discs at 40% overlap collapse to 1 at sigma 0.25. If you touch either constant, re-run the
whole case table, negative controls included.

## `spatial_relations` measures; it does not judge

`spatial_relations` (Grounding DINO boxes → SAM2 masks → `geometry.py`) reports numbers and deliberately stops
there. That split came out of testing, not taste: Moondream answered a plain "describe anything wrong in this
image" with a flat `"None"` on six different images that all contained a real, human-visible defect, and closed
yes/no questions gave the *same* answer across genuinely different images often enough that the answer was
clearly a default rather than an observation. A small VLM does not reliably supply that judgement. The calling
model does — and what it cannot do is measure, so that is what the tool provides.

**Locating the named objects doesn't reliably use color to discriminate, and this was found the hard way.** A
usage-quality pass (2026-08-26) built a synthetic scene with one red, one blue and one green circle and asked
`spatial_relations` for `['red circle', 'blue circle', 'green circle']`. Every query returned the *same three
boxes* — the detector (both Florence-2's grounding head and Grounding DINO were tested; both do this) found
"circle" and largely ignored the color word. The relations computed from that were nonsense: "red circle" vs
"blue circle" came back `overlapping` with `a_inside_b: 1.0`, because both labels pointed at the same physical
region compared against itself.

What made this fixable: the correctly-matching box scored highest for its own query, every time, in every test
run (5/5 on the color case, plus the pre-existing size-disparity case in `spatial_containment.png`). Switching
the detector from Florence-2's `detect_objects` (no per-box confidence available) to Grounding DINO (which
already backs `count_objects` and does carry scores) and keeping only the single best-scoring match per
requested name turned that ranking signal into a real filter. Verified end-to-end through a freshly spawned
server on both fixtures: three distinct, correctly-colored boxes with clean `touching`/`separate` relations, and
`b_inside_a: 1.0` on the containment case with no more self-comparison noise. Regression tests:
`test_spatial_relations_discriminates_same_shaped_objects_by_color` and `test_spatial_relations_measures_containment`
in `tests/test_server.py`, against `tests/spatial_touch_separate.png` and `tests/spatial_containment.png`.

The real cost of this fix: `spatial_relations` now assumes **one instance per requested name**. A scene with two
swords and you ask for `'sword'` twice gets you the same single best match twice, not two different swords —
give them distinguishing names, or use `count_objects` for an actual tally. This wasn't a real regression so
much as making an already-fuzzy assumption explicit: the tool never had a principled way to pair multiple
same-label instances (it just returned everything and computed every pairwise relation), so this trades that
loosely-defined behavior for one clear, documented rule.

**A companion finding from the same pass turned out not to be a bug.** `score_aesthetics` scored a crisp flat
vector-style graphic and a heavily blurred, noised version of it almost identically (4.19 vs 4.16). Re-run on an
actual photograph (`tests/sample.jpg`) at four blur levels, the score dropped monotonically (5.23 → 4.40 → 4.08
→ 3.95) — blur sensitivity works fine within the tool's already-documented scope ("rates photography, not fine
art"). The flat-graphic test was invalid, not the tool; see the `score_aesthetics` section of
`README_DETAILED.md` for the numbers. Recorded here so it isn't re-investigated as if it were still open.

Two things follow for anyone extending this:

- **Prefer aggregate statistics over a mask to fine topological derivatives of one.** Overlap fractions,
  distance transforms, principal axes and per-band centroids all average over many pixels and were stable on
  real photographs. Skeleton-based measures were tried first for curvature and for counting branch tips: both
  produced *inverted* results on real images, because `skeletonize` on a rough silhouette turns bark-level
  texture into spurious branches. `geometry.straightness` gets its centreline from band centroids for exactly
  this reason.
- **Validate any new metric against a negative control, not just a positive one.** Every check in `geometry.py`
  was measured on a known-good image as well as a known-bad one; two candidate checks were dropped precisely
  because the "good" case scored worse than the "bad" one, which a positive-only test would have hidden.

`geometry.py` is pure numpy/scipy and is unit-tested against synthetic masks in `tests/test_geometry.py`, so
its behaviour can be checked without downloading a model.

## v0.6.0: four tools got an opt-in flag that surfaces unreliability instead of hiding it

Four gaps against Claude's native vision were tracked as open: `caption` misreading embedded text (this
project's own banner logo came back "FusionVisionMP"), `query_image` on open-ended judgment (documented flat
"None" on six images with real defects), `count_objects`/`detect_objects` on overlapping cases (the flower — see
above), and `score_aesthetics`'s photography bias. None of these are fixable locally — a caption head that
paraphrases can't be made to transcribe, a small VQA model can't be made to reason reliably, a detector with no
notion of object identity can't separate a spot from a petal, and a CLIP head trained on photographs can't be
retrained into a fine-art critic without new weights. What shipped instead, on all four existing tools with no
new tools and no new models: an opt-in parameter that either cross-checks the unreliable output against a second
signal, or surfaces the disagreement instead of masking it. Same philosophy as the "`spatial_relations` measures;
it does not judge" section above — measure and flag, don't paper over.

- **`caption(verify_text=true)`** also runs Florence-2's `<OCR_WITH_REGION>` head and returns verbatim
  `text_regions` alongside the caption. Re-verified live against the actual `FusionVisionMCP-Dark.jpg` banner:
  the caption still reads "FusionVisionMP" (the head itself is unchanged and still wrong), but `text_regions`
  correctly returns `[{"text": "FusionVisionMCP", "box": [562, 211, 1493, 300]}]` in the same call. This closes
  the gap of having to guess or make a second blind call, not the gap of the caption head misreading text.
- **`query_image(check_consistency=true)`** asks a rephrased control question and returns
  `{answer, control_answer, consistent, confidence}`.
- **`count_objects(consensus=true)`** (default on) adds a `dense_region_caption`-based second opinion and a
  `separable` flag reading `count` against the silhouette's `by_distance`/`by_radial`/`agreement` fields.
- **`score_aesthetics`/`critique_composition`(`style_context=true`)** classifies the image's medium via
  zero-shot CLIP (16 style prompts, reusing the already-loaded aesthetic backbone) and returns it alongside the
  score, so a non-photographic result is read with the documented caveat instead of as an absolute verdict.

**Both of the first two shipped once already wrong, and both were caught by testing against the specific
documented case rather than a generic unit test.** `separable` originally read only `by_distance` (the
silhouette's `lobes` field), so on the flower/petal case — the canonical example the flag exists to catch — it
returned `separable: "yes"` sitting directly next to `consensus.agree: false` in the same response: two fields
in one payload contradicting each other. The fix makes `by_radial` (the rosette-specific angular-harmonic
estimator) and `agreement` load-bearing: `by_distance=1, by_radial=8, agreement=false` — a detector collapse the
outline's angular structure still catches — is now the one case documented to return `"no"`. Separately,
`_vqa_consistency` originally scored `confidence: "low"` only when both answers reduced to the *same* short
default token (e.g. "None"/"None"); asked to "describe anything wrong" on `tests/sample.jpg`, one answer invented
a missing centerpiece and the control answer said nothing was wrong — a direct, substantive self-contradiction —
and the original logic scored that `"normal"`. Fixed so `confidence` is `"low"` whenever the two answers
disagree, not only when they agree on a default. The lesson both cases share: a flag meant to catch a known
failure mode has to be tested against that exact failure mode, not just checked for producing *a* valid-shaped
value — the original `count_objects` test only asserted `separable in ("yes", "no", "unknown")`, which the buggy
version also satisfied.

**What this does and does not buy.** All four are opt-in (default `false` except `count_objects`'s `consensus`,
which is cheap and on by default) and none of them make the underlying model more capable. They convert a
silent wrong answer into a flagged one a calling agent can act on — decline to trust it, fall back to
`spatial_relations` or its own reasoning, or ask the user. That is the ceiling for what a local model stack can
do about a gap that is genuinely about semantic understanding rather than measurement.

## v0.8.0: The "Agentic Implementation" Refactor (Sprints 6–16)

Sprints 6 through 16 shifted the project from raw model wrappers to "agentic" capabilities—making the server smart enough to route, measure, and cross-check its own answers rather than just passing VLM hallucinations through to the user.

**The Routing Cascade (Sprint 5) & Adaptive Thresholding (Sprint 6)**
We needed `count_objects` to handle both real photographs and flat vector art without the caller explicitly passing `clip_art=True`. We originally tried zero-shot CLIP (SigLIP2, Sprint 4) for domain routing, but found it confidently misclassified real photographs as clip art. The fix was model-free (Sprint 15): a simple pixel-statistics check (unique RGB colors per 1,000 pixels) cleanly separates the two domains.
Similarly, `count_objects` using Grounding DINO on photographs originally used a static threshold (`0.30`). On dense scenes (like a flock of birds), it failed entirely. Sprint 6 introduced adaptive thresholding: it sweeps the threshold down to `0.10` if no boxes are found at `0.30`. This introduces a known risk (the "danger zone" of hallucinated boxes), so it is strictly gated: the lower thresholds are only accepted if `score_aesthetics` (or rather, its internal confidence) confirms the image is actually densely populated. We measure, we don't blindly lower the bar.

**Specialist OCR & Text Fusion (Sprints 7 & 8; specialist replaced in v0.8.1)**
Florence-2's caption head paraphrases and misspells text in the image, so `caption(auto_verify_text=true)` scanned
the generated prose for signage words and, if it found any, cropped and upscaled those regions, cross-checked them
against a specialist OCR model, and substituted the verbatim text back into the caption.

**Retired 2026-10-15.** Once every caption ended with the `ocr` engine's `Visible text` line (2026-10-10), the
reliable text was already in the caption. VisionBench `caption_text_recheck.py` re-measured it on 60 phase 11 text
images: key text phrases 52.9% -> 52.9% (+0.0), firing on 73% of them, median 31.6 s against 14.5 s. The parameter,
its caption-wording trigger (`textmatch.contains_likely_text`) and the crop-and-upscale re-read are gone;
`verify_text` (Florence-2's text regions, `corrections`, `caption_corrected`) is unchanged, its `text_consensus`
now comparing the caption with Florence-2 alone (`agreeing_sources` 1 or 2; it never had the third reading). Nothing
outside this machine called it: the server is not in the MCP registry.

The specialist was originally `ibm-granite/granite-docling-258M`, then **EasyOCR** (`easyocr_engine.py`), and
from 2026-09-29 the `ocr` engine, PP-OCRv6 English (see the `ocr` section above). The `granite_docling`
module is gone. If you find a reference to Granite-Docling anywhere, it is stale — two
test files importing the deleted module survived the swap and broke `pytest` collection outright until v0.8.2.

**Ground-Truth Suites (Sprints 15 & 16)**
We historically relied on synthetic, geometric fixtures. We realized this was a massive blind spot. We pulled 35 real photographs from COCO val2017 and 24 from Open Images v7 into `benchmarks/`. Every structural change is now regression-tested against real-world complexity, not just white backgrounds.

**Technical IQA & Reasoner (Sprints 11 & 12)**
We needed to split aesthetics into `photographic_aesthetic`, `technical_quality`, and `artistic_judgment`.
- `technical_quality` uses MUSIQ via ONNX. We chose MUSIQ over CLIP-IQA because it produces a native 0-100 score and uses 20% less memory, and we siloed it behind an `[iqa]` optional dependency so we don't force ONNX binaries on everyone.
- `artistic_judgment` uses `VisionReasoner` (`reasoner.py`), an optional Ollama-backed LLM backend. A strict invariant here: the LLM is allowed to interpret, but it is explicitly forbidden from silently overriding direct physical measurements (like object counts). The code discards LLM-fabricated measurements and re-attaches the raw deterministic ones.

**Structured Visual Inspection (Sprint 10)**
VLMs are notoriously blind to generative AI artifacts (e.g., humans with 3 arms). We built `structured_analysis` into `query_image` to physically measure anatomy using Grounding DINO (`person`, `arm`, `leg`, etc.) and evaluate against physiological ratios (e.g., `arm > persons * 2 + 1`). If the ratio fails, it raises an `Anomaly` backed by a physical `Observation`, overriding the VLM's hallucination. (Sprints 17-19 later refined this by enforcing strict part-to-person association and bounding-box deduplication to ensure the anomaly flags aren't just detector noise).

**2026-10-06: the part count no longer raises anomalies; a learned probe does.** Measured on VisionBench's
user-reviewed round 7 (149 images, 49 defective) the part count flagged 38.8% of defective images and 30.0% of
clean ones, near chance, and 47 of its 69 anomalies were leg counts. It can only say "too many", so a fused or
misshapen hand passes it. `defect_probe.py` replaces it as the anomaly signal: every Grounding DINO `hand` and
`face` crop (padded 20 px) is embedded with the CLIP ViT-L/14 `score_aesthetics` already loads, and a logistic
regression trained on round 7's reviewed crops scores it (`data/defect_probe.json`, written by VisionBench's
`probe_rounds_7_9.py --round7-only`, threshold set on round 7 alone at 85% specificity). The part count still
reports an over-count as a `detector` observation.

Three things were load-bearing, each measured:

- **Train on the reviewed rounds, not SynArtifact.** The published probe (AUC 0.88 on SynArtifact, one generator)
  scored round 7 at image AUC 0.65 and rated real photos as *more* defective than defects -- it had learned a
  generator's look. Trained on round 7 it reached 0.885 on the held-out round 8; adding SynArtifact made it worse.
- **The embeddings must be the unnormalised `image_embeds`**, which is what `Aesthetic.image_embeddings` returns
  (`get_image_features`), not the normalised features the aesthetic head uses. Parity with the training
  embeddings was checked to 4e-6.
- **Round 7 is training data now.** Measure the shipped tool on what the probe never saw: VisionBench
  `ai_defect_e2e.py` (round 8, phase 10, phase 11; 95 images). There, defective images flagged went 41.9% ->
  59.7% and clean images left alone 72.7% -> 66.7%. Specificity is the open gap.

**2026-10-08: retrained with the user's crop calls; more data stopped helping.** The user called every round 7
crop the defect boxes did not cover (216), and re-graded phase 11 themselves (the e2e set is now 94 images, 47
defective). Retrained on round 7's boxes plus those calls (261 defect, 772 clean crops): 72.3% of defective images
flagged and 63.8% of clean left alone, against 63.8% / 63.8% for the first probe on the same 94 images (image AUC
0.720 -> 0.741). VisionBench `ai_defect_rescore.py` re-scores the e2e set from cached boxes and embeddings, so a
candidate probe is checked in minutes rather than a two-hour server run. Adding round 9 (246 user-graded Stable
Diffusion 1.x images) made it *worse* on the unseen images (51.1% / 76.6%, AUC 0.699) while its own CV looked
fine, so the CLIP crop probe has plateaued: the next gain has to come from what the probe looks at (a different
backbone, landmarks), not from more of the same labels.

**2026-10-09: the probe reads DINOv2-large, not CLIP.** Same crops, labels and training rule; only the features
changed (`dinov2.py`: CLS token + mean patch token, 2,048 floats; `data/defect_probe.json` names its backbone).
Chosen on VisionBench round 10, 299 user-graded images from six generators no probe trained on (DALL-E 2,
Midjourney, RealVisXL, SDXL base, FLUX.2 Klein, SANA; 154 defective), by a rule committed before the run
(`defect_round10.py`, a5efb20): image AUC 0.686 vs CLIP's 0.612, +0.074 with a 95% interval of 0.013-0.135. At
each probe's own threshold: 44.8% of defective images caught, 84.1% of clean left alone, against 47.4% / 68.3%.
MediaPipe landmarks were tried too and gave nothing on hands (landmarked on 17% of crops, defective or not).

Four things to know:

- **Feature parity is load-bearing.** The probe was fitted to the slow `AutoImageProcessor` in float32; the server
  matches the research features to 7e-6 (probe scores to 3e-6). Don't switch to the fast processor or fp16
  without re-checking.
- **It is a new model in memory**, ~1.2GB, loaded only by `structured_analysis` when a hand or face is found and
  idle-released like the rest. CLIP is no longer loaded by `structured_analysis` at all.
- **Rollback is a data-file change.** `inspection._probe_features` reads the probe file's `backbone`: a CLIP probe
  file (or one without the field) still runs on CLIP.
- **DINOv2-base is the open question.** It scored as well on round 10 (0.709) at a quarter of the size, but it
  was not the candidate named before the run, so it waits for a fresh test set rather than being picked after
  seeing results.

## Where things live

`__init__.py` was 1,952 lines with all eleven tool definitions and their helpers inline. It is now ~250 and
holds only the entry point: `server()`, `app_lifespan`, and the per-model lazy factories.

- `tools/` -- the MCP tool definitions, grouped by domain (`text`, `detection`, `vqa`, `aesthetics`). Each
  module exposes `register(mcp)`; `tools.register_all` fixes the declaration order, which is the order a client
  sees the tools listed. Adding a tool means adding it to one of these, not to `__init__`.
- `protocols.py` -- the structural types the tools are written against (`Processor`, `VqaProcessor`,
  `Segmenter`, `AestheticScorer`, `InstanceDetector`) plus `AppContext`. They are `Protocol`s specifically so
  the tools can be type-checked without importing a torch-importing wrapper.
- `analysis.py` -- the cross-checks and measurements shared by tools (`_vqa_cross_check`, `_enrich_aesthetics`,
  `_separability`, the aesthetic comparison, the dispatch table).
- `images.py` -- `get_images`, the one place that knows a PDF becomes one image per page.
- `params.py` -- the `Annotated` parameter types. Their `Field(description=...)` text is what a calling agent
  reads at tool-selection time, so it is public contract, not decoration.

**`protocols`, `analysis`, `images`, `params` and `detection_policy` are all on the eager import path**, so the
torch-free rule below applies to them too: take constants from `constants`, never from a model wrapper.

## The idle timer measures completed calls, not attribute lookups

`IdleProxy.__getattr__` runs on every *attribute lookup*, and `get()` used to cancel and recreate a
`threading.Timer` — a fresh OS thread — on each one. A loop touching a proxy per page or per part paid that many
times over; `inspection._check_anatomy` alone does six lookups.

Worse, the countdown measured time since the last lookup, not since the call *returned*. Any inference longer
than the timeout was released while still running: `_value = None`, then `gc.collect()` and a Windows
`SetProcessWorkingSetSize` trim executing while torch was mid-forward-pass. The live local reference prevented a
use-after-free, but the working-set trim during inference is a real stall.

What ships now (`idle.py`): one self-rearming timer instead of one per lookup, `_last_used` updated when a call
*completes*, and an in-flight counter so `release()` defers while any call is running. `gc.collect()` moved
inside the lock — it was outside, so a concurrent `get()` could rebuild the model while the old one was still
being collected, briefly doubling resident memory on the exact path that exists to reduce it. `IdleProxy` also
grew its own `release()`: `proxy.release()` previously went through `__getattr__` and therefore *loaded* the
model in order to release it.

Regression tests in `tests/test_idle.py` cover all three: a call outlasting the timeout, thread count across 25
lookups, and `proxy.release()` on an unloaded proxy.

## Package import must stay torch-free, or MCP clients time out on connect

**This invariant has now been broken twice and restored twice. Check it before assuming it holds** — there is a
regression test (`'torch' not in sys.modules` after `import fusion_vision_mcp`) precisely because a prose note
here was not enough to hold it. The second break was found in the v0.8.2 review, with two causes beyond the
obvious one: `detection_policy.py` (on the eager path) imported `DEFAULT_BOX_THRESHOLD` from `grounding_dino`
rather than `constants`, and `EasyOCREngine.__init__` took a `torch.device`, forcing `__init__.py` to import
torch just to construct one. It now takes a device *string*, like every other wrapper.

`__init__.py` used to import `Aesthetic`/`Florence2`/`GroundingDino`/`Moondream`/`Sam2` at module
level, purely to get their names for use inside lazy `IdleProxy(IdleReleased(lambda: ...))` wrappers in
`app_lifespan`. Every one of those five wrapper modules does `import torch` (and `transformers`) at its own top
level, so the package import paid that cost regardless of whether any model was ever actually used — 4.5s warm,
and a measured 14.1s cold under memory pressure (5.2GB free of 31.5GB, 62 other `python.exe` processes). Real
successful connects in this project's own history ranged from 4.2s to 21.2s; several genuinely timed out against
Claude Code's 30s `CONNECT_TIMEOUT`, logged as zero `Server stderr:` output in 30 seconds — not a hang, just an
import that hadn't finished.

Fixed by moving every constant/enum `__init__.py` needs as a function-signature default (`DEFAULT_*`,
`MASK_DECODE_RESOLUTION`, `CaptionLevel`) into a new `constants.py` with no torch dependency, and deferring the
six heavy imports entirely. Each wrapper module still imports its own constant(s) back from `constants.py` and
re-exports them, so `from fusion_vision_mcp.grounding_dino import DEFAULT_BOX_THRESHOLD` (used by
`tests/test_grounding_dino.py`) keeps working unchanged. **Anything on the eager import path must take that
constant from `constants` instead** — importing it from `grounding_dino` drags torch in, which is exactly how
the invariant broke the second time.

**The first attempt at deferring the six class imports was wrong, and it is worth recording why.** Hoisting them
from module level to the *top of `app_lifespan`* measurably made no difference — `initialize` still took 15s.
`app_lifespan` runs as part of server startup, before the MCP handshake is answered, so anything unconditional
there is exactly as blocking as a module-level import; only the log line's timing moved. The fix that actually
worked: move each import *inside its own lazy factory function* (`_florence2`, `_moondream`, etc.), so it only
runs when `IdleReleased.get()` calls the factory on first real tool use (see `idle.py`) — matching the laziness
the `IdleProxy` wrapper already claimed to provide. Verified end-to-end against a real subprocess driving the
actual MCP wire protocol (not a direct Python import): `initialize` now returns in ~1.05s warm (down from 4.5s;
14.1s cold), with no `torch`/`transformers` appearing anywhere in `python -X importtime -c "import
fusion_vision_mcp"`'s trace. A subsequent real tool call (`caption`) still logs `Loading Florence-2` and pays the
full model-load cost, just deferred to when it's actually needed rather than at connect time — nothing about
model behavior changed, only when the cost lands.

Since return-type annotations on those per-model factory functions (e.g. `def _florence2() -> Florence2:`) are
evaluated at `def` time (this file has no `from __future__ import annotations`), and the six class names only
exist under a `TYPE_CHECKING` guard now, the annotations have to be quoted forward references (`-> "Florence2"`)
— ruff's `TC004` catches the unquoted form as "used at runtime" and it would otherwise be a `NameError` the
first time `app_lifespan` runs.

Historical note, since it explains a mode that no longer exists: there used to be a `Florence2SP` subprocess
path (`--cache-model` / `subprocess=True`) in which every model call ran in a spawned child process that
reimported `fusion_vision_mcp` from scratch and reloaded the full model *per call*. It was selected only when
`idle_timeout == 0`, i.e. `--memory-mode persistent` — so the mode documented as fastest was the one reloading
Florence-2 on every request. `Florence2SP`, `subprocess.py`, the `dill` dependency and the `--cache-model` flag
were all removed in v0.8.2; `IdleReleased` with a timeout of 0 already gives a persistent in-process model.
Use `--memory-mode persistent` where `--cache-model` used to appear.

## A note on where this lives

This checkout used to live under OneDrive. `uv tool install` failed there with a hardlink error, and `uv run`/`uv sync` separately failed removing a `.dist-info/licenses` directory that OneDrive had turned into a cloud placeholder — neither error message mentions OneDrive. Moving the checkout to `C:\AI\MCP\FusionVisionMCP` (a plain local path, renamed from `C:\AI\MCP\mcp-florence2` on 2026-08-20) resolved both. Keep it out of any synced folder.
