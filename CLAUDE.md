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

References to `benchmarks/` here and in [docs/history.md](docs/history.md) (constants tuning rationale,
the counting and spatial-relations records) describe *where a measurement was made*, not a path that still
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
# Since 2026-10-23 they also load both LFM2.5-VL-3B copies (the ONNX default, ~8 GB at peak, and bf16 `deep`, ~6.4 GB),
# and the test server keeps every model loaded. Run the suite (and the pre-commit hook) only while no benchmark holds a
# model: on 2026-10-23 test_caption and test_caption_deep failed with is_error during VisionBench runs, then passed 502/502 alone.
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

## Tools: what ships and the rules that hold it

What each tool runs today and what must not be changed without a fresh measurement. The dated record behind every
line (gate scripts, figures, what was tried and failed) is in [docs/history.md](docs/history.md); read the matching
section there before changing a tool. Two habits hold everywhere: a change that alters a measured output gets a
VisionBench gate with its rules fixed before the run, and a shipped wrapper is checked for parity with the gate it
passed (same prompts, same decisions on cached replies).

### `ocr`

- **PP-OCRv6 English through RapidOCR** (`ppocr_engine.py`, ONNX, no torch). The recognizer must be the English
  one (`LangRec.EN` + `PP-OCRv6`): RapidOCR's default Chinese/English model drops the spaces between English words.
  Detector `Det.box_thresh` is 0.4 (`OCR_DET_BOX_THRESHOLD`).
- **An image with no text returns empty, and that is correct.** The 0.75 floor (`OCR_MIN_CONFIDENCE`) is what keeps
  it empty; `tests/test_server.py::test_ocr_textless_image_returns_no_text` pins it. Never "fix" an empty result by
  asking a VLM. Because of the floor, `detail=true` is not a gibberish detector.
- **Every page is read twice more:** a sharpened reading (`analysis._sharpened_extra`, `OCR_SHARPEN_*`, spans of 3+
  characters not already read) and, on pages with text, LFM2.5-VL's transcription (`lfm_read`, `_lfm_extra`,
  `source: "lfm"`, no box). An empty page is never sent to LFM: that is what keeps text-free images empty.
  **`caption`'s text line and `query_image`'s OCR context pass `sharpened=False, lfm_read=False`**; using either
  there needs its own gate.
- **EasyOCR stays installed.** `query_image(structured_analysis=true)`'s 0.20 gibberish threshold is calibrated to
  EasyOCR's scores, and non-English `--ocr-languages` route `ocr` to EasyOCR (no second passes there). easyocr
  needs `opencv-python-headless` and rapidocr `opencv-python`, so `[tool.uv] override-dependencies` drops
  `opencv-python`; keep that override.
- **Columns are split by geometry, not by prompting** (`layout.reading_blocks`, pure numpy/PIL). The page is cut
  into zones on blank rows; a line with more than two ink pixels inside a gutter is read full width; a zone is split
  only if some column in it has 3+ lines; a gutter must be at least the median line height wide (capped at 3% of
  the width). Load-bearing details, each from a regression: zone boundaries use any ink (rule lines excepted); lines
  are counted per column; a gutter's outer 2 px are not checked; a ruled divider of 3 px or less is bridged; a page
  with 2 lines is never split (`_count_line_bands`). `tests/test_layout.py` holds the fixtures, negative controls
  included. Small text still reads approximately.
- Cropping before OCR (`region`) made it worse (it cuts words at the edge); it was not shipped on `ocr`.

### `caption`

- **Default: LFM2.5-VL-3B on ONNX Runtime** (`lfm_onnx.LfmOnnxCaptioner`, transformers.js export at `020f931`, CPU,
  one image of at most 256 tokens, "Describe only what is clearly visible in this image.", `CAPTION_SEED`).
  `--caption-model florence2` or a failed load falls back to Florence-2.
- **`Image type: <x>.`** comes first (`image_type`, CLIP ViT-L/14 over `CAPTION_IMAGE_TYPES`): a best guess, on the
  LFM default only (`deep`, `region` and the Florence-2 fallback were not measured). A CLIP that can't load drops the
  line, never the caption.
- **`Visible text: "..."`** comes last (`read_text`, `visible_text_line`): the `ocr` engine's first reading, cleaned
  (repeated and one-character lines dropped), at most 400 words (`CAPTION_TEXT_WORDS`), a cut line saying how many
  words it left out. A text-free image gets no line. It is appended after any `verify_text` work.
- **The prose describes text, it does not transcribe it** (the banner came back "FusionVisionMP"). `verify_text`
  cross-checks with Florence-2's `<OCR_WITH_REGION>`, the one place Florence-2 still reads text.
- **`deep=true`** is LFM2.5-VL-3B in bf16 with 4 tiles (`lfm.LfmCaptioner`, revision `35a118d`, ~35 s): it adds
  detail, not accuracy per word. Its Moondream2 fallback keeps the loop stop (`repetition.py`) and the crop rule.
- **`region=[x1, y1, x2, y2]`** crops before captioning (`images.crop_region`: 15% pad each side, at least 32 px).
  Changing the pad means re-running gate R on fresh questions.
- Florence-2 runs in bf16 on CPU for every consumer and still backs `verify_text`, `dense_region_caption`, flat-image
  detection and counting, the composition subject box and `process`. Anything re-measured against an fp32 figure has
  to be re-measured in bf16.
- LFM weights are under the LFM Open License v1.0, downloaded on first use, not redistributed.
- **llama-server runs on the GPU unless told not to.** The winget llama.cpp build defaults to Vulkan offload; any
  llama.cpp run here sets `LLAMA_ARG_N_GPU_LAYERS=0`, `LLAMA_ARG_DEVICE=none` and `LLAMA_ARG_MMPROJ_OFFLOAD=0` and
  checks `-v` for "offloaded 0/31 layers".

### `query_image`

Live plan: [Future_Ideas/Query_Image_Improvement_Plan.md](Future_Ideas/Query_Image_Improvement_Plan.md).

- **LFM2.5-VL-3B ONNX answers** (`analysis.vqa_answers`, `AppContext.answerer`, the default caption's proxy, so one
  loaded copy serves both). `--vqa-model moondream`, or a load failing with `OSError`, `ImportError` or
  `MemoryError`, falls back to Moondream2.
- **Text and name lookups get the `ocr` reading** in front: `VQA_TEXT_QUESTION` (text words, plus What/Which/Who/Where
  questions naming a name, number, price, sign, logo or UI element, not "what colour"), `VQA_OCR_PREFIX`, at most 160
  words (`VQA_OCR_WORDS`), `VQA_TEXT_SUFFIX`.
- **Other What/Which/Who/Where questions get a presence check first** (`presence_check`, default on;
  `VQA_PRESENCE_CHECK`, a "no" adds `VQA_ABSENT_SUFFIX`). It failed one gate rule (-1.7 on other changed questions,
  limit -1.0) and shipped on the user's decision: quote -1.7 as the measured figure. Control questions start
  "Looking", so `check_consistency` never checks them.
- **`check_consistency`** asks a rephrased control question, sharing one OCR reading with the answer, and judges
  agreement by question kind (`analysis._answers_agree`, `VQA_AGREE_COVER`). It carries no signal on absent-object
  questions, and an unflagged answer is not a checked one.
- The cross-check counts with `analysis.count_image`, the same steps as `count_objects` at its defaults.
  `question.py` extracts count nouns (whole-word fillers; a verb that is also a noun ends the noun only before a
  preposition, article, quantifier or the end).
- Changing a routing regex, prompt, suffix or agreement rule means a fresh round. The prompts each gate measured are
  frozen in VisionBench (`vqa_shipped_20261024.py` for the pre-V5 ones), so older gates still measure their own
  baseline.
- Moondream2 stays loaded only for `critique_composition`'s two questions.

### `classify_image`

CLIP zero-shot over the caller's labels (`CLASSIFY_TEMPLATE` "a photo of {}", `Aesthetic.zero_shot`, the
`score_aesthetics` model). Gate Z2's 98.5% is an upper bound (famous Commons images); gate Z's everyday photos are
the realistic estimate. It always picks one of the labels. Routing `query_image` questions that list options to it
would need its own gate.

### `critique_composition`

Live plan: [Future_Ideas/Critique_Composition_Improvement_Plan.md](Future_Ideas/Critique_Composition_Improvement_Plan.md).
`background: {busy, reliability}` is Moondream2's answer to `BUSY_BACKGROUND_QUESTION`, a hedged hint (69% agreement
with the user's blind calls). Without `target_subject`, Moondream2 names the subject (`MAIN_SUBJECT_QUESTION`) and
Florence-2 boxes it; a box covering 95%+ of the frame (`_WHOLE_FRAME`) defers to the largest central dense region.
Background blur, subject focus and awkward crops are not shipped: nothing measured matched the user's calls.

### `detect_objects`

- **SAM 3 on photos and flat images** (`sam3.py`, `facebook/sam3`). Its post-processing is what the gates measured:
  processor at 0.05, top 300, floor 0.125 (`LOCATE_BOX_THRESHOLD`), class-agnostic NMS at IoU 0.5, then Grounding
  DINO's envelope filter. Change any step and re-run VisionBench `replacement_gate2.py`.
- **The first box is the one to trust** (right 71-73%); the rest are candidates. Dropping boxes CLIP rejects cost
  too much recall.
- **SAM 3 is gated** on Hugging Face: without access the load raises an `OSError` subclass and
  `detection_policy._photo_detect` falls back to Grounding DINO at 0.35 (Florence-2 on flat images). The result's
  `detector` says which ran.
- "Flat" is the router's pixel test (under 10 colours per 1,000 pixels, not greyscale with tonal range), so most
  JPEG screenshots take the photo path.
- **SAM 3 needs transformers 5.** Moondream2 is pinned to `5d6c926` (the 2025-06-21 weights). The transformers 5
  fixes are load-bearing: `use_fast=False` in `aesthetic.py`, `florence2.py`, `grounding_dino.py` and `dinov2.py`;
  CLIP's `pooler_output` (`Aesthetic._projected`); Florence-2's `clean_up_tokenization_spaces` forced on; and two
  module-level Grounding DINO patches (phrase position ids, float position embeddings). Grounding DINO's scores
  still differ from 4.57 by up to 0.012 (Swin's arithmetic order); every replayed decision held. Meta's own `sam3`
  package is GPU-first and was not used.

### `count_objects`

The plan is closed (archived). What ships, in order:

- **Grounding DINO on photos at 0.425** (`COUNT_BOX_THRESHOLD`); `DEFAULT_BOX_THRESHOLD` 0.15 stays for every other
  Grounding DINO caller, which were not re-measured. **Florence-2 on flat images** (`drop_group_box=True`).
- Boxes dropped on both paths: the whole-group envelope (`geometry.envelope_indices`), then parts and
  near-duplicates (`geometry.nested_indices`: IoU >= 0.55 with a kept box; or 90%+ inside a larger kept box and
  under 0.85 of its area, unless that box holds 2+ barely-overlapping nested boxes, which makes it the group box).
- **`possibly_more`**: unreturned boxes in [`COUNT_RANGE_FLOOR` 0.18, threshold) that are neither on nor inside a
  counted box. It only ever points upward.
- **`crowd_range`**: SAM 3 at `CROWD_RANGE_SAM3_FLOOR` 0.3 against Grounding DINO, shown when either is 4+
  (`CROWD_RANGE_SHOW`), run only when `possibly_more.upper` >= 3 (`CROWD_RANGE_RUN_UPPER`). On generalist images it
  mostly marks an overcount of 1-3 objects, not a crowd. The 0.3 was picked by looking; changing it needs a fresh set.
- **`lfm_check`** (default on). Photos: a count of 1 whose box scores under `COUNT_CHECK_WEAK` 0.6 becomes 0 when
  LFM answers "no" to `COUNT_CHECK_PRESENCE`; a 0 with a candidate in [0.18, threshold) becomes 1 when LFM's count
  (`COUNT_CHECK_COUNT`) is 1+ and presence says "yes". Flat: any count becomes 0 on a presence "no"; a zero is never
  raised. The prompts, 0.6 and the reply parsing were fixed before the gates; changing any of them needs a fresh set.
  `estimates.vqa` asks LFM too.
- `consensus` is opt-in; `separable` is always reported. **`count` is never rewritten by the silhouette check**:
  `count: 1` beside `silhouette.lobes: 8` is two methods disagreeing, which is information. A low count on something
  expected to be many means "could not separate them".
- **Tune against VisionBench's counting fixtures, never one image, and hold all 8 negative controls.** Any idea that
  zooms in, matches appearance or reads interior contrast turns the spotted ball into many objects (interior colour,
  tiling, CountGD exemplars, LFM's own flat count all did). Test a new idea against `neg_spotted_ball` first.
- **Don't spend another pass on `tests/sample.jpg`'s flower**: every approach returns 1, and the contrast is not
  there (solidity 0.984, interior boundary strength 0.87). Also closed: CountGD (worse on every measure), SAM 3 as a
  counter (returns 0 for "ball", "rod"), tiling, exemplars.
- Converting a checkpoint: safetensors deduplicates shared tensors, so restore the aliases from `__metadata__` and
  check substantive missing keys rather than trusting plausible outputs.
- Open gaps: hard negatives (a verified-absent object counted about 7 times in 10) and present objects on flat images
  (34.8% exact on round 28).

### `spatial_relations`

It measures; the calling model judges (a small VLM's "is anything wrong" answers were defaults, not observations).
Grounding DINO boxes (best-scoring box per requested name, so **one instance per name**: give two same-class objects
distinguishing names), SAM2 masks, `geometry.py` measurements. It stays on Grounding DINO + SAM2: SAM 3 was shipped
here and reverted the same day (it matched the wrong instance and its masks stop short of a hand). Prefer aggregate
statistics over a mask (overlap fractions, distance transforms, band centroids) to skeleton-based ones, which
inverted on real photos, and validate every new metric against a negative control as well as a positive one.

### `query_image(structured_analysis=true)` and the defect probe

- The anomaly signal is a learned probe (`defect_probe.py`, `data/defect_probe.json`) on Grounding DINO `hand` and
  `face` crops (padded 20 px), embedded with **DINOv2-large** (`dinov2.py`: CLS + mean patch token). The part count
  only reports an over-count as a `detector` observation.
- **Feature parity is load-bearing:** the slow `AutoImageProcessor` in float32. Don't switch to the fast processor or
  fp16 without re-checking. A CLIP probe file uses the unnormalised `image_embeds`.
- Rollback is a data-file change: `inspection._probe_features` reads the probe file's `backbone`.
- Round 7 (and the user's crop calls) is training data; measure the shipped tool on VisionBench `ai_defect_e2e.py`'s
  unseen images. More labels of the same kind made it worse; specificity is the open gap. DINOv2-base waits on a
  fresh test set.

### Other parts that carry rules

- **`score_aesthetics` is photography-biased**: blur lowers the score on a photograph, not on a flat graphic (that
  test was invalid, not a bug).
- **Opt-in flags surface unreliability rather than hiding it** (`verify_text`, `check_consistency`, `consensus`,
  `style_context`). A flag meant to catch a known failure must be tested against that exact failure: two shipped
  wrong once because their tests only checked the value's shape.
- **The reasoner** (`reasoner.py`, Ollama, `artistic_judgment`) may interpret but never override a measurement: its
  fabricated measurements are discarded and the deterministic ones re-attached.
- **Domain routing is a pixel test, not CLIP** (SigLIP2 called photos clip art). `adaptive_threshold` is opt-in; its
  known failure is in the archived risk analysis.
- Technical IQA is MUSIQ via ONNX behind the `[iqa]` extra.
- Granite-Docling and `auto_verify_text` are gone; any reference to them is stale.

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

## A load has to fit: `MemoryManager` releases idle models first (2026-10-25)

The 2026-10-24 test-server deaths were **commit exhaustion**: Windows logged a low-virtual-memory condition with one
python.exe at 26.8 GB of commit, and two earlier crashes were access violations in torch's allocator (`c10.dll`).
`memory.MemoryManager` (torch-free, on the eager path) is shared by every `IdleReleased`. Before a factory runs it
reads live free RAM and commit (`GlobalMemoryStatusEx`; free RAM + swap on Linux), compares them with the model's
`constants.MODEL_MEMORY_GB` estimate plus `MEMORY_RESERVE_GB`, and releases least-recently-used idle models
(`try_release`, never mid-call, never blocking on another model's lock) until it fits. Still short on commit: it
raises `ModelMemoryError`, a tool error rather than a dead server. Short on RAM alone: it loads with a warning.
`--memory-budget GB` adds a soft cap. Loads are serialised. This runs in `persistent` mode too, which is what lets the
session-scoped test server pass (523 of 523, 25 releases to make room; `Future_Ideas/archive/memory_manager.md`).

Two things to know. **The cache name is the table key**: a new `_lazy(factory, "Name")` in `app_lifespan` needs a
`MODEL_MEMORY_GB` entry (measure it with VisionBench `model_memory.py`) or it is estimated at 1 GB. And **the table is
peaks, measured** (2026-10-25, one fresh process per model, one call): the private-bytes delta the server logs after
each load misses what a call adds (SAM 3 +3.2 GB loaded, 6.5 GB peak; Grounding DINO 0.7 / 1.9; SAM2 0.2 / 1.3) and
reads ~0 for PP-OCRv6, EasyOCR and ImageQuality, which load on first call, so a logged size only ever raises a figure.
Moondream2 peaks at 4.5 GB on CPU here, not the ~7.5 GB fp32 quoted elsewhere in this file; LFM ONNX 7.9, LFM bf16
6.4. Everything loaded at once would peak near 40 GB.

**Heavy calls are checked too, not only loads.** `MODEL_RESIDENT_GB` is what a model holds between calls; a call needs
`MODEL_MEMORY_GB` minus that on top, on every call. A call adding at least `CALL_CHECK_MIN_GB` reserves its extra
while it runs (`MemoryManager.calling`, entered by `IdleProxy` around each method call): calls run side by side when
they fit, wait for each other when they don't, and with nothing running a call releases idle models or is refused.
Reservations are per thread and a thread never waits on its own, so a model method that calls another proxy cannot
deadlock; keep it that way if a wrapper ever starts calling proxies.

**A watcher releases idle models between calls too.** `app_lifespan` starts `MemoryManager.start_watcher` (every 5 s,
in every memory mode): free commit under 4 GB releases any idle model, free RAM under 4 GB those idle 30 s+, free RAM
under 25% of the total those idle 2 min+ (`MEMORY_WATCH_*` in `constants.py`). So a model can be gone well before its
idle timer when other processes squeeze the machine; that is the point, and the next call reloads it.

**It also watches paging, because free RAM alone misses Windows' answer to a squeeze.** M3's S3 (2026-10-25) pushed
free RAM to 3.05 GB; within 5 s Windows paged other programs out and held ~4.6 GB free, above the 4 GB critical level,
while 10 GB of idle models stayed loaded. `memory.PagingMonitor` reads `\Memory\Pages Output/sec` through pdh.dll
(ctypes, English counter name; 0 at idle here) or Linux's `/proc/pressure/memory`; high on two checks in a row with
RAM under the tight level is a burst, remembered 60 s (`MEMORY_WATCH_PAGING_MEMORY_S`): while it is, one model is
released per check as it passes 30 s idle ("paging: N pages out/s Xs ago"). The S3 re-run showed why: the burst
lasted ~10 s, right after the models were used, so requiring both at once never fired. Not Pages Input/sec:
it counts memory-mapped file reads, which every model load here is.

**`server_status` reports all of it** (`tools/status.py`, `MemoryManager.status()`): models loaded and what they hold,
live free memory, heavy calls running, and the last 20 loads, releases with their reason, and refusals. It loads no
model. Any new release or refusal path should `record()` an event, or the tool will not explain it.

**`--transport streamable-http` shares one server between clients** (`docs/shared_server.md`; opt-in, stdio stays the
default for everyone who downloads this). The SDK's streamable-HTTP manager enters `app_lifespan` **once** for all
sessions (`mcp/server/streamable_http_manager.py`), so every client shares one `AppContext`, one copy of each model
and one `MemoryManager`; `tests/test_http_transport.py` pins that (a second session sees the model the first loaded),
plus the SDK's loopback DNS-rebinding protection and the non-loopback warning. There is no authentication, so it
listens on 127.0.0.1 unless told otherwise. `scripts/run_shared_server.ps1` restarts it on Windows, where Task
Scheduler does not.

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
