# `critique_composition` Frontier Improvement Plan (Opus 5.5 Implementation Spec)

> **Status:** Reviewed 2026-11-02 (see **Review** below, which supersedes the body). Q1–Q4 answered 2026-11-02; next is R3 step 0 (cache-only checks).
> **Target Tool:** [`critique_composition`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L129-L265) (`src/fusion_vision_mcp/tools/aesthetics.py:129`, `src/fusion_vision_mcp/analysis.py:815-912`, `src/fusion_vision_mcp/geometry.py:639-672`)
> **Scoreboard Reference:** [`Future_Ideas/frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md#L103) (Row 103: `🔻 weaker: about −7 points` on prose critique; Section *"Where Things Still Fall Short — Composition prose critique"* at lines 155–160)
> **Model Origin Policy:** Strictly Western / non-Chinese models (`LiquidAI/LFM2.5-VL-3B` — US; `microsoft/Florence-2-large-ft` — US; `facebook/sam3` & `facebook/sam2.1-hiera-large` — US; `openai/clip-vit-base-patch32` — US).

---

## Review (2026-11-02) — read this first

Checked against the code at `469e439` and the VisionBench result files on disk. Where this section and the body
disagree, this section wins. The body is kept as written for the record.

### R1. Facts in the body that are wrong

| Body says | Actually |
|---|---|
| A multi-image batch path (`aesthetics.py:212-255`) diverges from `_critique_one` (Bug C, Step 4B, §6.4.1) | **There is no batch path.** The tool takes one `src` and critiques page 1; lines 212-255 are docstring and `compare_with` mode, which calls `_critique_one` for both images. Bug C and Step 4B are dropped. |
| `_WHOLE_FRAME = 0.85` | `0.95` (`analysis.py:833`) |
| `_pick_primary_subject` scores `(area/img_area) * (0.5 + 0.5*centrality)`, rejects boxes over 85%, returns a dict with `"box"` | Scores `area * centrality`, has no whole-frame rejection, returns `list[int] \| None`. The plan's `picked_det["box"]` would raise. |
| `rule_of_thirds(image_size, box)` returns `[0, 1]` values rounded to 4 places | `rule_of_thirds(box, image_size)`; `centroid` and `nearest_gridpoint` are pixels, the offsets are diagonal-normalised, nothing is rounded. |
| `VQA_MAX_NEW_TOKENS = 64` truncates critiques (§3.1.4); greedy decoding, `VQA_SEED = 42` | `VQA_MAX_NEW_TOKENS = 128`, `VQA_SEED = 0`, and LFM decoding is sampled with the seed, not greedy. The truncation diagnosis doesn't apply. |
| `MAIN_SUBJECT_QUESTION` / `BUSY_BACKGROUND_QUESTION` wordings in §2.1 | Not the shipped wordings (`constants.py:321-332`). |
| LFM2.5-VL ONNX is 2.7 GB peak / 2.5 GB resident | `8.7` / `7.7` (`MODEL_MEMORY_GB`, `MODEL_RESIDENT_GB`). The memory win only shows up in a session that already has LFM loaded for `caption`/`query_image` (true on the shared server). A cold call that only critiques costs *more* than Moondream2 (5.0 / 4.1). |
| `app.vqa_model`, `--vqa-model lfm2.5-vl`, falling back on `AttributeError` | `AppContext` has no `vqa_model`. The choices are `lfm-onnx` / `moondream`, and the server's pattern is `app.answerer is None` → Moondream2, plus `except (OSError, ImportError, MemoryError)` (`analysis.vqa_answers`). |
| Gate K3-0 baselines (`iou_mean 0.619`, `iou_at_0_5 70.8%`, `edges_exact`, `shuffled_f1 16.5%`, `delta_over_shuffled +2.3`) | Not in any result file. The real figures for the shipped `md_florence` box are: IoU≥0.5 at 79.5 / 82.0 / 84.4% (tune / confirm / box2), horizontal agreement 89.9% (confirm) and 88.5% (box2), fill agreement 82.0% / 76.0%, and declined-on-none 0/11 and 0/24. The K2 prose baseline is 18.8 F1. It has no shuffled control: K2's `shuffled` swaps the *facts sentence* and scores 11.8. |
| K1 validated fill at 0.10/0.35/0.75, three horizontal zones at 0.40/0.60, plus vertical zone and touching edges | K1's fill cut-points are **0.10 / 0.40 / 0.75**. Horizontal is **five** bins (`left, left_third, center, right_third, right`). Vertical and edge contact were not scored, and K1's header says box-edge contact did **not** match the user's cut-off calls. |
| Reference critiques run 48-62 words | Worker critiques (the tune/confirm references) have a **median of 24 words**. Phase 7 references: Gemini 33, GPT 36, Claude 44. |
| Labels at `VisionBench\datasets\composition_labels.json` with a `critique` field | Labels are at `C:\AI\Projects\VisionBench-labels\worker_handoff\composition_labels.json`, under `labels.critique`. `benchmarks\datasets\composition_labels.json` is only the index. |
| CLIP is `openai/clip-vit-base-patch32` | ViT-L/14 with the LAION head. |

The plan also never mentions that **the prose critique is only produced when the aesthetic score is below
`low_score_threshold` (5.0)**. Phase 7 forces it with `low_score_threshold=10.0`. Most real calls return no
critique, so Track 1's gain only reaches callers if that changes (Q1).

### R2. What holds, and what changes

The core idea is right. `busy_background`, `name_main_subject` and the critique line are the last Moondream2 calls on
the default path, and moving them to the already-loaded LFM matches every other migration. The prose confirm split
really is unseen: `composition_prose.py` was only ever run on tune.

1. **Track 1 (prose) goes first; it is the gap on the scoreboard.**
   - **Length.** Token F1 depends on length, and the two reference sets differ (24 words on tune/confirm, 33-44 on
     phase 7). Drop the 35-75-word band. Report F1 at matched length as well as raw, as the caption gates do.
   - **Prompt content.** K2 found the references are about *feel, subject and light, not placement*. P2's rubric
     (thirds, centring, clipping) steers toward exactly what the references don't say. P2 should name subject,
     light/mood and one concrete flaw instead. Fix the wording before the tune run. P3 (soft hints) stays as one arm.
     P4 is dropped (it was conditional and undefined).
   - **Anti-boilerplate control**, defined properly: each critique scored against *another photo's* reference (seeded
     derangement). Compute it for Moondream2's cached K2 baseline first; that needs no model run. Set the control
     threshold from that number before any LFM output exists.
   - **Ship rule on confirm (run once).** Paired CI lower bound above 0 against Moondream2, and the control margin
     held. Drop the 60% win rate: it is redundant with the CI and the number was arbitrary. Phase 7 runs once
     afterwards, as a scoreboard reading, not a gate.
2. **Bug A (first box rather than the best one) is plausible but not established.** Florence-2's
   `<CAPTION_TO_PHRASE_GROUNDING>` order isn't documented as raster order, and K1 cached only the chosen box. It can
   be measured cheaply: re-run Florence-2 grounding on K1's cached `md_subject` names (confirm + box2, Florence only),
   count how often there is more than one box, and score `bboxes[0]` against `_pick_primary_subject`. It ships only if
   it is no worse, since this path is gated. Do it before the naming swap, so the swap is measured on the final box rule.
3. **Subject naming swap (Moondream2 → LFM).** Non-inferiority on the seen confirm + box2 sets is acceptable here by
   precedent: the count-estimate swap was gated on 457 seen counts. The margin should be a **paired CI lower bound**
   on the IoU≥0.5 difference, fixed beforehand (−5 points proposed, Q4), not a 1.5-point gap in the point estimate.
   The question and the "none" parse stay as they are. The article stripping in Step 2B is untested and is dropped.
4. **Busy background swap.** With 52 confirm labels the CI is about ±12 points, so a 1.7-point non-inferiority
   margin can't be shown. The reliability string ("Moondream2 … 69% of 52") has to be restated whatever happens. See Q2.
5. **Blur (Gate K3-3): park it.** There are only 7 blurred photos in tune and 12 in confirm, all labels have been
   seen twice, and K1's confirm CI was 44.8-77.2. A 75/72/CI≥60 rule on 12 positives can't be decided. The
   edge-spread ratio would be a new hand-built measure fitted on those same 19 positives.
6. **Abstention (Step 2C, S2): park it.** Falling back to the region pick on "none" is the documented behaviour.
   "Declined on none" isn't a quality measure the user has signed off on, and S2's area band and background-word
   list would be fitted by looking at the data.
7. **Geometry fields: ship them without a gate, but with K1's rules** (fill at 0.40, five horizontal bins), because
   those rules are what the 88-90% agreement measured. Vertical zone: score it first against `labels.subject_vertical`
   on K1's cached boxes (no model run) and add it only if it agrees about as well. **Leave out `touching_edges`**:
   K1 found that edge contact doesn't mean "cut off", and a field named that way invites the wrong reading.
8. **Moondream2 is not retired.** It stays as the fallback in `vqa_answers`, `vqa_count_estimate` and `deep`, so
   `pyvips` and the remote-code pin remain. This plan removes it from the *default* path only. Full removal is a
   separate decision.
9. **Implementation.** The helper should mirror `vqa_answers`: `app.answerer is not None`, catch
   `(OSError, ImportError, MemoryError)`, and pass `max_new_tokens`. Text to update: three `Field` descriptions and
   the docstring in `tools/aesthetics.py`, the `busy_background`/`name_main_subject` docstrings, the `constants.py`
   comments (`:258`, `:319`, `:326`), `cli.py:87`, CLAUDE.md and the scoreboard row. Measure latency; don't claim "2×".
   Parity: the server reproduces the gate's replies on 8 of 8.

### R3. Revised order

0. **Cache-only checks, no model runs** (Opus 5.5): the reference-swap control on the cached K2 baseline, vertical
   agreement from K1's cached boxes, and reference lengths (done above: 24 / 33-44). Then write Gate K3-4's prompts,
   length handling and thresholds into the VisionBench script header before any LFM output.
1. **Gate K3-4 prose**: tune run → pick → confirm once → phase 7 once (scripts stay in the session that wrote them;
   interpreting the results is Opus 5.5).
2. **Subject**: Bug A Florence-only replay, then the LFM naming non-inferiority run on confirm + box2.
3. **Busy background**: as Q2 decides.
4. **Ship**: server edits, unit tests, docs, parity (Sonnet 5.5, new session + handoff file).

Blur and abstention are parked.

### R4. Questions for the user

- **Q1.** Should the critique prose be produced on every call (a `critique` flag, default on, about 5-10 s more),
  or stay below the 5.0 aesthetic threshold? This decides whether Track 1 matters to callers at all.
- **Q2.** Busy background: (a) blind-label box2's 120 photos for busy yourself in the card UI (about 15-20 min),
  giving a real held-out test; or (b) accept the LFM swap on the pooled 77 seen labels if it is not worse in the
  point estimate, with the reliability string restated from those 77.
- **Q3.** OK to park blur and abstention?
- **Q4.** Is −5 points (paired CI lower bound on IoU≥0.5) the right non-inferiority margin for the naming swap?

**Answers (user, 2026-11-02):**
- **Q1: keep the critique for low scorers only.** The 5.0 default stays, and callers who want prose every time pass
  `low_score_threshold=10` (as phase 7 does), so no new flag. The gate still scores every photo, since Phase 7 does
  the same. The tool description should say how to force the prose.
- **Q2: fresh labels, with a screen first.** Run the fixed LFM busy question on the 77 seen labels (about 6 min, no
  tuning). If it is clearly worse than Moondream2 (lower by 10+ points pooled), stop and keep Moondream2 for busy.
  Otherwise the user blind-labels busy on box2's 120 photos in the card UI. That set is the held-out test: paired CI
  lower bound on balanced accuracy ≥ −5 against Moondream2 on the same photos, and the reliability string is restated
  from it.
- **Q3: blur and abstention parked.**
- **Q4: −5 points**, the paired CI lower bound on the IoU≥0.5 difference, on confirm + box2.

**Added 2026-11-02, before step 0:**
- **Placement-contradiction guard (K3-4).** A critique that puts the subject left/right/centre gets checked against
  the measured horizontal class of the shipped box (K1's five bins collapsed to left / centre / right). A candidate
  must not contradict it more often than Moondream2 does (paired, on confirm).
- **Latency limit (K3-4, and both swaps).** Median seconds per call, measured on the same machine in the same session
  as the Moondream2 arm, must be no more than 3 s above Moondream2's.
- **Optional user preference check, after confirm.** 20 confirm photos, LFM vs Moondream2 critique, blind and in
  random order, in the card UI (~5 min). It is reported, not a gate, unless the user says otherwise.
- **`compare_with` mode** runs `_critique_one` twice, so the tests cover it too.

### R5. Step 0 results (2026-11-02, cache only, no model runs)

Script: `scratchpad/step0.py`, which imports VisionBench's `composition_critique` / `composition_prose` scorers.

- **Vertical position (five bins, the same rule as `horiz`, from the tool's `md_florence` box against the worker's
  box).**

  | Set | Photos | Horizontal | Vertical |
  |---|---|---|---|
  | tune | 39 | 84.6% | 82.1% |
  | confirm | 89 | 89.9% | 86.5% |
  | box2 | 96 | 88.5% | 80.2% |

  The horizontal figures reproduce K1's (89.9 and 88.5). Vertical is 3-8 points below horizontal, at the level of
  fill (82.0 / 76.0), which already ships, so **vertical zone ships with the geometry fields**.
- **Reference-swap control.** Moondream2's cached tune critiques score 18.8 F1 against their own photo's reference
  and 9.3 against another photo's (seeded derangement): a gap of +9.5, CI 7.3 to 12.0. **A fixed boilerplate
  sentence scores 14.7.** So the real range of this metric is roughly 15-30, and the body's "+3.5 over shuffled" bar
  is meaningless: Moondream2 already clears it threefold.
- **Lengths:** the references' median is 24 words and Moondream2's 31.5.

### R6. Gate K3-4 rules, fixed 2026-11-02 before any LFM critique exists

**Arms**, all at `max_new_tokens = 128` and `VQA_SEED`, on the full image:
- `md`: Moondream2 with the shipped prompt, re-run in the same session for timing; on tune it should match the K2
  cache.
- `P1`: LFM with the shipped prompt, `"Critique this photo's composition and framing in 2 concise sentences."`
- `P2`: LFM with `"Critique this photo's composition in 2 concise sentences: what the main subject is, how the
  framing, light and mood work, and one specific weakness."`
- `P3`: LFM with `"Measured by a detector (use it only if it matches what you see): <facts>\n"` + P2's prompt, where
  `<facts>` is K2's `facts_sentence` built from the shipped box and Moondream2's busy answer, as cached.

**Measures** (per photo, max over references; paired bootstrap, 5,000 resamples):
- raw token F1 (`rv.token_f1`);
- **matched-length F1**: the candidate cut to `md`'s word count on that photo;
- **image-specific gap**: F1 against the photo's own reference minus F1 against the reference-swap reference
  (seed `SEED + 1`, as in step 0);
- **placement contradiction**: the critique mentions left/right/centre (`centre|center|middle`) and none of those words
  matches the shipped box's horizontal class (`left`+`left_third` → left, `center`, `right_third`+`right` → right);
- median seconds per critique.

**Tune (50) picks one arm:** the highest raw F1 among the LFM arms that pass all four guards against `md` on tune:
1. matched-length F1 ≥ `md`'s;
2. image-specific gap ≥ `md`'s − 2.0;
3. contradiction rate ≤ `md`'s + 5 points;
4. median seconds ≤ `md`'s + 3 s.

If no arm passes, stop: nothing ships and confirm is not run.

**Confirm (100, run once) ships the arm if:**
- raw F1 beats `md` with a paired CI lower bound > 0;
- guards 1-4 still hold.

Then phase 7 runs once and is recorded, not gated. The user's blind preference check (20 photos) follows, reported only.

---

## 1. Executive Summary & Why `critique_composition` Is Next

### 1.1 Why `critique_composition` Was Chosen from `frontier_capability_gaps.md`

The scoreboard in [`Future_Ideas/frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md#L89-L105) lists five capabilities in the **`🔻 weaker`** tier (`5 points or more behind`):

| Scoreboard Row | Capability | Gap vs Frontier | Current Plan / Status |
|---|---|---|---|
| Row 95 | `query_image` (open-ended questions) | `−7.3 pts` (`64.6%` vs `71.9%`) | **Excluded per request** — actively in progress under [`Future_Ideas/Query_Image_Improvement_Plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/Query_Image_Improvement_Plan.md). |
| Row 97 | `count_objects` (overall accuracy) | `−7.0 pts` (`59.1%` vs `66.1%`) | **Closed 2026-10-31** — Phases 1–12 completed in [`Future_Ideas/Counting_Improvement_Plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/Counting_Improvement_Plan.md). |
| Row 99 | `ocr` on natural-scene photos (TextVQA) | `−14.9 pts` (`57.4%` vs `72.3%`) | **Closed 2026-10-25 / Shipped 2026-10-30** — [`Future_Ideas/OCR-TextVQA_Improvement_Plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/OCR-TextVQA_Improvement_Plan.md) closed; `lfm_read` default shipped in [`Future_Ideas/ocr_blur_improvement_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/ocr_blur_improvement_plan.md), lifting TextVQA to **`68.5%`** (`+11.1 pts`, within `3.8 pts` of frontier). |
| Row 101 | `ocr` on blurred text | `−25.2 pts` (`34.8%` vs `60.0%`) | **Shipped 2026-10-30** — B2–B5 + `lfm_read` default in [`Future_Ideas/ocr_blur_improvement_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/ocr_blur_improvement_plan.md) lifted real-camera blur to **`62.2%`** (`+17.2 pts`, beating the `60.0%` target). |
| **Row 103** | **`critique_composition` (written critique + structured composition checks)** | **`🔻 weaker: about −7 points`** (`25%` like-for-like F1 vs `29–34%` frontier; `18.8%` F1 on 50-photo `composition_prose` tune) | **No plan yet — last remaining `🔻 weaker` tool.** |

### 1.2 Architectural Payoff: Retiring `Moondream2` from the Active Server Path

Beyond closing the `−7 pt` quality gap, upgrading [`critique_composition`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L129-L265) unlocks a major architectural milestone documented in [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L312-L315):

- `caption(deep=true)` moved to `LFM2.5-VL-3B` on 2026-10-22.
- `caption` default moved to `LFM2.5-VL-3B` ONNX (`app.captioner`) on 2026-10-23.
- `query_image` moved to `LFM2.5-VL-3B` ONNX (`app.answerer`) on 2026-10-24.
- `count_objects(vqa_estimate=true)` moved to `LFM2.5-VL-3B` ONNX (`app.answerer`) on 2026-10-29.
- `check_consistency` moved to `LFM2.5-VL-3B` ONNX (`app.answerer`) on 2026-10-31.
- **`critique_composition` is now the ONLY tool in FusionVisionMCP whose primary path still calls `app.vqa.query` (`Moondream2`).**

Specifically, [`src/fusion_vision_mcp/analysis.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L912) makes **three** direct `app.vqa.query(...)` calls per single-image `critique_composition` invocation:
1. [`busy_background(app, image)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L829) at line 821: `app.vqa.query([image], BUSY_BACKGROUND_QUESTION)[0]`
2. [`name_main_subject(app, image)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L836-L845) at line 844: `app.vqa.query([image], MAIN_SUBJECT_QUESTION)[0]`
3. [`_critique_one(...)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912) at lines 909–911: `app.vqa.query([image], "Critique this photo's composition and framing in 2 concise sentences.")[0]`

Because `Moondream2` (`vqa`) requires **`5.0 GB` peak / `4.1 GB` resident RAM** ([`MODEL_MEMORY_GB`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L355-L369), [`MODEL_RESIDENT_GB`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L375-L385)) and runs at `~10 s` per question on CPU, calling `critique_composition` forces `MemoryManager` to load a separate `5.0 GB` PyTorch model even when `LFM2.5-VL-3B` ONNX (`captioner`, `2.7 GB` peak / `2.5 GB` resident, shared between `app.captioner` and `app.answerer`) is already warm in memory, and spends **`~23–44 s` per image** on CPU.

Migrating those three calls to `app.answerer` (with per-call-site VisionBench gates per [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L312) and graceful fallback to `app.vqa` on `(OSError, ImportError, MemoryError)` or `--vqa-model moondream`):
1. **Eliminates `Moondream2` (`4.1 GB` resident / `5.0 GB` peak) from normal server operation**, preventing `MemoryManager` eviction churn.
2. **Cuts `critique_composition` single-image CPU latency by roughly `2×`** (from `~23–44 s` down to `~11–15 s`, and down to `~8–10 s` if two questions are combined or when `target_subject` is supplied).
3. **Directly resolves the prose critique bottleneck** identified in [`frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md#L160): *"The prose gap is model judgement; code-only critique work has plateaued... Closing the remaining ~7 points needs a stronger model, not better prompt plumbing on the current one."*

---

## 2. Verified Codebase & Dataset Audit

> [!IMPORTANT]
> Every file path, line number, function signature, and dataset split status below has been verified directly against the live workspace in `c:\AI\MCP\FusionVisionMCP` and `C:\AI\Projects\VisionBench`. Do not rely on unverified assumptions.

### 2.1 Live Code Paths in `c:\AI\MCP\FusionVisionMCP`

| File & Lines | Symbol | Current Behavior |
|---|---|---|
| [`src/fusion_vision_mcp/tools/aesthetics.py:129-265`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L129-L265) | `critique_composition(images, target_subject=None)` | Loads up to `MAX_BATCH_IMAGES` (`20`) images; runs [`_critique_one`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912) on single image or batched path on `len(pil_images) > 1`. Note: batched path (`lines 212–255`) still uses `app.florence2.dense_region_caption` + `_pick_primary_subject` when `target_subject is None` (it never received K1's `name_main_subject` + `detect_objects` upgrade) and never sets `background.busy`! |
| [`src/fusion_vision_mcp/analysis.py:815-829`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L829) | `busy_background(app, image)` | Calls `app.vqa.query([image], BUSY_BACKGROUND_QUESTION)[0]`. Parses leading `"yes"` → `True`, leading `"no"` → `False`, else `None`. |
| [`src/fusion_vision_mcp/analysis.py:833`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L833) | `_WHOLE_FRAME = 0.85` | Box covering `> 85%` of image area is rejected as whole-frame fallback. |
| [`src/fusion_vision_mcp/analysis.py:836-845`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L836-L845) | `name_main_subject(app, image)` | Calls `app.vqa.query([image], MAIN_SUBJECT_QUESTION)[0]`. Strips `"."`, returns `""` if `"none"` or `"nothing"` in lowered string. |
| [`src/fusion_vision_mcp/analysis.py:848-912`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912) | `_critique_one(app, image, target_subject, include_vqa=True)` | 1. Calls `busy_background(app, image)` if `include_vqa` (`line 855`).<br>2. If `target_subject`, calls `app.florence2.detect_objects([image], target_subject)[0]`; takes `detected["bboxes"][0]` (`line 865`).<br>3. Else calls `name_main_subject(app, image)` + `detect_objects`; takes `detected["bboxes"][0]` (`line 865`).<br>4. If `box is None` or `box_area > 0.85 * img_area`, falls back to `app.florence2.dense_region_caption([image])[0]` + `_pick_primary_subject(regions, (w, h))` (`lines 870–873`).<br>5. Computes `rule_of_thirds((w, h), box)` (`line 883`) + heuristic notes (`lines 884–904`).<br>6. Calls `app.vqa.query([image], "Critique this photo's composition and framing in 2 concise sentences.")[0]` (`lines 909–911`). |
| [`src/fusion_vision_mcp/analysis.py:1182-1203`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L1182-L1203) | `_pick_primary_subject(regions, image_size)` | Rejects boxes `> 85%` of image area; scores remaining boxes by `(area / img_area) * (0.5 + 0.5 * centrality)`; falls back to largest box if all `> 85%`. |
| [`src/fusion_vision_mcp/geometry.py:639-672`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L639-L672) | `rule_of_thirds(image_size, box)` | Returns `{"centroid": [cx, cy], "thirds_offset": [dx, dy], "center_offset": [dcx, dcy], "nearest_gridpoint": [gx, gy]}` rounded to 4 decimals. |
| [`src/fusion_vision_mcp/constants.py:321-332`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L321-L332) | `MAIN_SUBJECT_QUESTION`, `BUSY_BACKGROUND_QUESTION` | `MAIN_SUBJECT_QUESTION = "In 1 to 3 words, what is the single main foreground subject of this photo? Reply 'none' if it is a wide scene with no single subject."`<br>`BUSY_BACKGROUND_QUESTION = "Is the background behind the main subject cluttered or busy? Answer yes or no."` |
| [`src/fusion_vision_mcp/lfm_onnx.py:150-158`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/lfm_onnx.py#L150-L158) | `LfmOnnxCaptioner.answer(images, prompts, seed=VQA_SEED, max_new_tokens=VQA_MAX_NEW_TOKENS)` | Runs `LFM2.5-VL-3B` ONNX (`LiquidAI/LFM2.5-VL-3B` via `lfm25_vl_3b_onnx` repo) with greedy decoding (`temperature = 0.0`, `VQA_SEED = 42`, `VQA_MAX_NEW_TOKENS = 64`). Already used by `app.answerer` in `query_image`, `count_objects`, and `check_consistency`. |

### 2.2 VisionBench Datasets & Seen vs. Held-Out Status

> [!CAUTION]
> Respecting the data-hygiene contract in [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L1065-L1086) is mandatory. Below is the exact seen vs. unseen status of every dataset on disk in `C:\AI\Projects\VisionBench`.

| Dataset / File | Count | Origin & Contents | Seen vs. Held-Out Status |
|---|---|---|---|
| `C:\AI\Projects\VisionBench\datasets\composition_labels.json` (`split == "tune"`) | **50 photos** | Stratified COCO photos (`person`, `animal`, `object`, `indoor`, `landscape`) labeled by `Gemini 3.8 Flash` (`subject`, `subject_box`, `horizontal`, `vertical`, `fill`, `touching_edges`, `blurred_bg`, `busy_bg`, `critique`). | **SEEN (`tune`)** for K1 structured checks (`composition_critique.py`) and K2 prose critique (`composition_prose.py`). Use **only** for tuning prompts, max tokens, and candidate selection. |
| `C:\AI\Projects\VisionBench\datasets\composition_labels.json` (`split == "confirm"`) | **100 photos** | Same schema as `tune`, stratified across the same 5 buckets. | - **Structured checks (`subject_box`, `horizontal`, `fill`, `touching_edges`):** Evaluated **once** at the K1 gate (`composition_critique_confirm.json`).<br>- **Prose critique (`row["critique"]`):** **100% UNSEEN HELD-OUT!** As documented in [`frontier_capability_gaps.md:160`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md#L160) (*"Confirm and phase 7 were not run, since no variant could ship"*), `composition_prose.py` was **never** run on `--split confirm` (`composition_prose_confirm.json` does not exist). |
| `C:\AI\Projects\VisionBench\datasets\composition_box2.json` | **120 photos** | Fresh stratified COCO photos (`id: box2_000..box2_119`) labeled by `Gemini 3.8 Flash` (`subject`, `subject_box`, `horizontal`, `vertical`, `fill`, `touching_edges`). (`blurred_bg: null`, `busy_bg: null`, `critique: ""`). | - **Subject box (`subject_box`, `horizontal`, `fill`):** Evaluated once in `benchmarks/results/composition_box2.json` (`19` photos user-spotchecked in `SPOTCHECK_BOX2`).<br>- **Background checks (`busy_bg`, `blurred_bg`):** **100% UNSEEN** (never labeled or evaluated for background checks). |
| `SPOTCHECK` + `SPOTCHECK2` in [`composition_labels.py:67-237`](file:///C:/AI/Projects/VisionBench/benchmarks/composition_labels.py#L67-L237) | **77 user-reviewed photos** (`25` in `tune`, `52` in `confirm`) | Blind user labels `(blurred_bg, busy_bg)` from Round 1 (`38` photos) and Round 2 (`39` valid photos; `000000183648.jpg` was `[bad photo]` and is skipped). | Used in K1 (`composition_spotcheck.json`, `composition_spotcheck2.json`) to gate Moondream2 `busy_background` (`88.1%` tune, `69.2%` confirm) and reject `sharp_ratio` (`77.6%` tune, `61.4%`/`54.2%` confirm). |
| Phase 7 Aesthetics (`C:\AI\Projects\VisionBench\golden_dataset_generation\02_critique_composition\`) | **25 photos** (`coco_104424`..`coco_90108`) | `75` frontier golden critiques (`25 × 3` models: `Claude Opus 4.6 Thinking` in `golden_critique_claude.json`, `Gemini 3.1 Pro High` in `golden_critique_gemini.json`, `GPT 5.6 Luna` in `golden_critique_GPT.json`). | **Scoreboard evaluation set** (`run_visionbench.py --phases 7`). Never tune prompts on these 25 images; run Phase 7 **only once** after the `composition_prose.py --split confirm` gate passes. |

---

## 3. Root-Cause Diagnosis of the Four Capability Gaps

### 3.1 Gap 1 — Prose Critique Quality (`−7 pts` vs Frontier, `18.8%` Tune F1, `25%` Like-for-Like Phase 7 F1)

Inspecting all 50 per-image outputs in [`C:\AI\Projects\VisionBench\benchmarks\results\composition_prose_tune.jsonl`](file:///C:/AI/Projects/VisionBench/benchmarks/results/composition_prose_tune.jsonl) and the 25 Phase 7 outputs in `visionbench_report_phases_7_10_extended.json` reveals four concrete failure modes of `Moondream2` (`1.9B`):

1. **Generic Praise Boilerplate (`>70%` of outputs):**
   - `Moondream2` emits stock filler sentences regardless of the image: *"The composition is well-balanced and visually appealing..."* (`000000088265.jpg`, `000000516143.jpg`, `000000022705.jpg`, `000000279278.jpg`) or *"The photo is well-composed and framed..."* (`000000493349.jpg`, `000000060090.jpg`).
   - By contrast, the 3 frontier references (`Claude Opus 4.6`, `Gemini 3.1 Pro`, `GPT 5.6`) and the `Gemini 3.8 Flash` teacher critiques never use empty praise; they name the **specific subject**, its **frame placement / rule-of-thirds / symmetry**, **depth layers (foreground/midground/background)**, **leading lines / negative space**, and **concrete framing flaws** (e.g., edge clipping, cluttered background, tilted horizon, awkward crop).
2. **Plain Scene Captioning Instead of Compositional Analysis:**
   - On images like `000000017029.jpg` (*"A small yellow bird perches on a thin branch against a vibrant green background..."*), `000000045070.jpg`, `coco_109798`, and `coco_30675`, `Moondream2` writes a literal object description with almost no compositional vocabulary (`thirds`, `foreground`, `background`, `negative space`, `framing`, `cropped`, `leading lines`, `depth`, `balance`).
3. **Why Grounding Collapsed on Moondream2 in Sprint K2 (`md_grounded` `12.9%` vs `md_raw` `18.8%`):**
   - In K2 (`composition_prose.py:114-120`), feeding `facts_sentence` (*`"Measured framing: the main subject ('bear') is centered and large in the frame; the background is busy."`*) into `Moondream2` caused it to **shrink from `31.5` words to `26.0` words** (`composition_prose_tune.json`) and either parrot the dry mechanical sentence verbatim or contradict it (`000000493349.jpg`: given `"left of centre"`, Moondream2 wrote `"centrally positioned"`).
4. **Why `VQA_MAX_NEW_TOKENS = 64` Truncates `LFM2.5-VL-3B` Prose Critiques:**
   - Look closely at [`LfmOnnxCaptioner.answer`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/lfm_onnx.py#L150-L158): its default `max_new_tokens` is [`VQA_MAX_NEW_TOKENS = 64`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L298).
   - Frontier critiques in Phase 7 average **`48–62` words (`~65–85` subword tokens)** across 2–3 dense sentences (`golden_critique_claude.json`, `golden_critique_gemini.json`, `golden_critique_GPT.json`). At `max_new_tokens=64`, a 2–3 sentence critique gets cut off mid-sentence on the second sentence! [`LfmOnnxCaptioner.answer`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/lfm_onnx.py#L150-L158) already accepts `max_new_tokens` as an explicit parameter (`max_new_tokens=CRITIQUE_MAX_NEW_TOKENS`, e.g., `128`).

### 3.2 Gap 2 — Subject Naming, Multi-Instance Box Selection, & Batch/Single Parity

Inspecting [`src/fusion_vision_mcp/analysis.py:836-873`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L836-L873) and [`src/fusion_vision_mcp/tools/aesthetics.py:212-255`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L212-L255) exposes **three concrete code bugs** alongside the `Moondream2` → `LFM2.5-VL-3B` model swap:

1. **Bug A — Raster-Order First-Box Selection on Multi-Instance Detections (`analysis.py:865`):**
   - Look at [`src/fusion_vision_mcp/analysis.py:863-865`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L863-L865):
     ```python
     detected = app.florence2.detect_objects([image], subject)[0]
     if detected.get("bboxes"):
         box = detected["bboxes"][0]
     ```
   - When `subject` is `"person"`, `"zebra"`, `"car"`, `"kite"`, or `"bird"` and the photo contains multiple instances, `Florence-2` (`<OD>` / `<OPEN_VOCABULARY_DETECTION>`) emits bounding boxes in **top-left raster scan order** (`y1, x1` ascending) with no confidence scores.
   - Taking `detected["bboxes"][0]` blindly picks whichever instance happens to be highest/leftmost in the frame — often a tiny background bystander or distant car at the top edge — instead of the actual primary foreground subject!
   - Yet [`_pick_primary_subject(regions, image_size)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L1182-L1203) already exists in the very same file (`analysis.py:1182`) and scores candidate boxes by `(area / img_area) * (0.5 + 0.5 * centrality)` while rejecting whole-frame boxes (`> 85%` area). Calling `_pick_primary_subject(detected, (image.width, image.height))` instead of `detected["bboxes"][0]` is a zero-latency, zero-VRAM fix!
2. **Bug B — Zero Abstention on Subjectless Wide Scenes (`declined_on_none = 0.0%` across all 46 `"none"` photos):**
   - In `composition_critique_tune.json` (`11` `"none"` photos), `composition_critique_confirm.json` (`11` `"none"` photos), and `composition_box2.json` (`24` `"none"` photos), `declined_on_none` is **`0.0%` (`0 / 46`)**.
   - Why? Two reasons:
     1. `Moondream2` almost never replies `"none"` to `MAIN_SUBJECT_QUESTION`. By contrast, `LFM2.5-VL-3B` is proven on false-premise / abstention tasks (`66.4%` vs `32.3%` in Round 23, `71.2%` in Round 29).
     2. Even when `name_main_subject(app, image)` **does** return `""` (because the model explicitly answered `"none"`), look at what [`_critique_one`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L859-L873) does at lines 859–873:
        ```python
        subject = target_subject or (name_main_subject(app, image) if include_vqa else "")
        box = None
        if subject:
            ...
        if box is None or _box_area(box) > _WHOLE_FRAME * img_area:
            regions = app.florence2.dense_region_caption([image])[0]
            picked = _pick_primary_subject(regions, (image.width, image.height))
            box = picked["box"] if picked else None
        ```
        When `name_main_subject` returns `""` (`"none"`), `box` is `None`, so `_critique_one` **unconditionally runs `dense_region_caption` and forces a subject box anyway**!
   - **Important nuance:** On some photos where a real subject exists, VLM might mistakenly say `"none"`, OR `detect_objects` might return no box / a whole-frame box (`> 85%`), where falling back to `dense_region_caption` recovers real subjects. We will test both behaviors in Gate K3-1 (`fallback on all box is None` vs `respect explicit "none"` when `dense_region_caption` also has no prominent central box) so `iou_mean` on real subjects never regresses.
3. **Bug C — Single-Image vs. Multi-Image Batch Divergence (`tools/aesthetics.py:212-255`):**
   - In [`src/fusion_vision_mcp/tools/aesthetics.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L196-L255), passing `len(images) == 1` calls [`_critique_one`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912) (which uses `name_main_subject` + `detect_objects`, sets `background: {"busy": busy}`, and asks `"Critique this photo's composition and framing in 2 concise sentences."`), whereas passing `len(images) > 1` runs a stale inline loop (`lines 212–255`) that:
     - Never calls `name_main_subject` when `target_subject is None` (uses only `dense_region_caption`),
     - Uses `det["bboxes"][0]` instead of `_pick_primary_subject`,
     - **Omits the `"background"` key entirely**, and
     - Uses an older VQA prompt (`"Critique the composition and visual balance of this image in 2-3 concise sentences."`).
   - Unifying the batch path so every image goes through the same `LFM2.5-VL-3B` + subject-detection + `busy_background` logic eliminates this silent contract violation.

### 3.3 Gap 3 — Background Checks (`busy_background` `69.2%` Confirm & `background.blurred` Omitted)

1. **`busy_background` (`analysis.py:815-829`):**
   - `Moondream2` scored `88.1%` balanced accuracy on the 25 user-reviewed tune photos and `69.2%` (`95% CI [56.5%, 81.0%]`) on the 52 user-reviewed confirm photos (`SPOTCHECK` + `SPOTCHECK2`), taking `~10 s` per photo.
   - Swapping `app.vqa.query` → `app.answerer.answer` (`LFM2.5-VL-3B` ONNX) must clear a pre-registered gate on those 77 user-reviewed photos (`25` tune + `52` confirm) and, if any prompt wording or hybrid signal is tuned, on a fresh user-spot-checked slice from `composition_box2.json`.
2. **Why `background.blurred` (Depth-of-Field / Bokeh) Failed in K1 & How to Re-Test It Properly:**
   - In Sprint K1, two signals were tested for `background.blurred`:
     - `Moondream2` on `"Is the background behind the main subject blurred or out of focus? Answer yes or no."`: failed (`53.5%` tune / `54.0%` confirm balanced accuracy) because `Moondream2` answered `"no"` on almost everything.
     - `sharp_ratio = log(e_in / e_out)` (mean 1 px Laplacian energy inside vs outside the rectangular subject box, threshold `0.40`): scored `77.6%` on tune (`n=25` user labels, only `7` positives) but collapsed to `61.4%` (Round 1) / `54.2%` (Round 2) on confirm (`n=52` user labels).
   - **Root causes of `sharp_ratio`'s collapse:**
     1. **Plain / low-texture backgrounds:** A smooth blue sky, painted wall, or calm water has `e_out ≈ 0` even when shot at `f/11` (deep depth of field, completely in focus). Raw Laplacian `e_in / e_out` confuses *"smooth untextured surface"* with *"optical defocus blur"*. That is why Round 2 of the user spot-check (`composition_labels.py:152`) had to clarify: *"Is the background clearly out of focus (not just far away or plain)?"*
     2. **Rectangular bounding-box leakage:** A rectangular bounding box `[x1, y1, x2, y2]` around a person, bird, or diagonal object includes `30–60%` background pixels inside the box (`inside`), contaminating `e_in`.
   - **Two principled candidates to test in Gate K3-3 (shipped ONLY if they pass strict pre-registered gates):**
     1. **`LFM2.5-VL-3B` ONNX (`app.answerer`) with the clarified optical-defocus question:** Asking `LFM2.5-VL-3B` the exact distinction that tripped up `sharp_ratio`: *"Is the background behind the main subject optically blurred (shallow depth of field / bokeh), rather than just plain or empty? Answer yes or no."*
     2. **Edge-Spread Sharpness Ratio (`edge_sharp_ratio`) on eroded/central box vs background edges:** Instead of averaging raw Laplacian over flat sky (`e_out`), measure the **1 px vs 4 px gradient ratio** (`mean(|I(x+1)-I(x)|) / mean(|I(x+4)-I(x)|)`, the same multi-scale step-width principle proven in `ocr_blur_improvement_plan.md`) **restricted to the top-15% gradient pixels** inside the inner `70%` core of the subject box vs outside the subject box (excluding flat regions where `grad_4px < threshold`). Optical defocus widens existing edges (low `grad_1px / grad_4px`), whereas a flat sky has no edges and is excluded instead of masquerading as blur!

### 3.4 Gap 4 — Deterministic Framing Geometry Enrichment (`geometry.rule_of_thirds`)

In [`src/fusion_vision_mcp/geometry.py:639-672`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L639-L672), `rule_of_thirds(image_size, box)` currently returns only raw `[0, 1]` coordinates:
```python
{
    "centroid": [cx, cy],
    "thirds_offset": [dx, dy],
    "center_offset": [dcx, dcy],
    "nearest_gridpoint": [gx, gy],
}
```
Yet in Sprint K1 ([`C:\AI\Projects\VisionBench\benchmarks\composition_critique.py:189-203`](file:///C:/AI/Projects/VisionBench/benchmarks/composition_critique.py#L189-L203)), deterministic bucket derivations from `box` and `(w, h)` were already validated on all 270 labeled photos (`50` tune + `100` confirm + `120` box2), achieving **`88.5–89.9%` horizontal zone agreement** and **`76.0–82.0%` fill agreement** on correctly localized boxes:
- `subject_area_ratio`: `round(box_area / img_area, 4)`
- `fill`: `"small"` (`< 0.10`), `"medium"` (`0.10–0.35`), `"large"` (`0.35–0.75`), `"fills"` (`>= 0.75`)
- `horizontal_zone`: `"left"` (`cx < 0.40`), `"center"` (`0.40 <= cx <= 0.60`), `"right"` (`cx > 0.60`)
- `vertical_zone`: `"top"` (`cy < 0.40`), `"middle"` (`0.40 <= cy <= 0.60`), `"bottom"` (`cy > 0.60`)
- `touching_edges`: subset of `["left", "top", "right", "bottom"]` within `1%` (`0.01`) of image border

Adding these five deterministic fields to `framing` (while keeping `centroid`, `thirds_offset`, `center_offset`, and `nearest_gridpoint` 100% unchanged for backward compatibility) costs **0 ms and 0 MB**, enriches the structured JSON output for MCP callers, and provides exact geometric facts for Track 1 prose critique grounding.

---

## 4. Four-Track Improvement Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  critique_composition(images, target_subject)                │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│  TRACK 2: Subject     │   │  TRACK 3: Background  │   │  TRACK 4: Geometry    │
│  Localization         │   │  Analysis             │   │  Enrichment           │
│                       │   │                       │   │                       │
│ • target_subject OR   │   │ • busy_background:    │   │ • rule_of_thirds():   │
│   LFM2.5-VL-3B        │   │   LFM2.5-VL-3B ONNX   │   │   keeps centroid,     │
│   name_main_subject() │   │   (app.answerer)      │   │   thirds_offset,      │
│ • Florence-2 detect   │   │   w/ app.vqa fallback │   │   center_offset,      │
│   + _pick_primary_    │   │ • blurred (Gate K3-3, │   │   nearest_gridpoint   │
│   subject() on multi- │   │   shipped ONLY if     │   │ • Adds deterministic: │
│   box returns (fixes  │   │   confirm >= 70% &    │   │   subject_area_ratio, │
│   bboxes[0] bug)      │   │   CI_lo >= 55%)       │   │   fill, horizontal_   │
│ • Fallback to dense   │   │                       │   │   zone, vertical_zone,│
│   regions if needed   │   │                       │   │   touching_edges      │
└───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        ▼
            ┌───────────────────────────────────────────────────────┐
            │  TRACK 1: LFM2.5-VL-3B ONNX Prose Critique            │
            │  (app.answerer, max_new_tokens=128)                   │
            │                                                       │
            │  • Evaluates ungrounded LFM2.5-VL-3B vs soft-grounded │
            │    LFM2.5-VL-3B (given subject + placement + edge/bg  │
            │    cues as context hints, not a rigid template)       │
            │  • Tuned ONLY on 50-photo tune split                  │
            │  • Confirmed once on 100-photo unseen confirm split   │
            │  • Evaluated once on Phase 7 (25 photos × 3 models)   │
            └───────────────────────────────────────────────────────┘
```

### 4.1 Helper: Unified VLM Query Helper `_answer_one` / `_answer_batch` (`src/fusion_vision_mcp/analysis.py`)

To follow the exact pattern established in `tools/vqa.py`, `tools/verification.py`, and `analysis.py` (`count_objects`), add a small internal helper in [`src/fusion_vision_mcp/analysis.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py):

```python
def _vlm_answer_batch(
    app: VisionAppContext,
    images: list[ Image.Image ],
    prompts: list[str] | str,
    *,
    max_new_tokens: int = VQA_MAX_NEW_TOKENS,
) -> list[str]:
    """Answer one prompt per image via LFM2.5-VL-3B ONNX (`app.answerer`),
    falling back to `app.vqa.query` (`Moondream2`) if `--vqa-model moondream`
    is selected or if the ONNX runtime/model is unavailable in tests/offline.
    """
    prompt_list = [prompts] * len(images) if isinstance(prompts, str) else list(prompts)
    if getattr(app, "vqa_model", "lfm2.5-vl") != "moondream":
        try:
            return app.answerer.answer(
                images,
                prompt_list,
                max_new_tokens=max_new_tokens,
            )
        except (OSError, ImportError, MemoryError, AttributeError):
            logger.warning(
                "LFM2.5-VL-3B answerer unavailable in composition analysis; "
                "falling back to Moondream2 VQA"
            )
    if isinstance(prompts, str):
        return app.vqa.query(images, prompts)
    return [app.vqa.query([img], p)[0] for img, p in zip(images, prompt_list)]
```

**Why this helper is critical:**
1. Every existing unit/server test in [`tests/test_analysis_helpers.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_analysis_helpers.py) and [`tests/test_server.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_server.py) that mocks `app.vqa` without defining `app.answerer` (or where `app.answerer` raises `OSError`) continues to work cleanly, while tests that provide `app.answerer` exercise the primary `LFM2.5-VL-3B` path.
2. Users who explicitly pass `--vqa-model moondream` on the CLI ([`src/fusion_vision_mcp/cli.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/cli.py)) get `Moondream2` as requested.
3. Passing `max_new_tokens` allows short 1–3 word / yes-no answers (`busy_background`, `name_main_subject`) to use a small cap (`16` tokens, saving latency) while the 2–3 sentence prose critique uses `CRITIQUE_MAX_NEW_TOKENS = 128`.

---

### 4.2 Track 1 — Prose Critique Upgrade (`LFM2.5-VL-3B` ONNX + Prompt & Soft-Grounding Study)

#### Why `LFM2.5-VL-3B` Changes the Game
- In Sprint K2, `Moondream2` (`1.9B`) topped out at `18.8%` F1 on the 50 tune photos (`composition_prose_tune.json`) and collapsed to `12.9%` when given `facts_sentence` because `Moondream2` is not instruction-tuned for structured multi-aspect critique synthesis.
- `LFM2.5-VL-3B` (`LiquidAI/LFM2.5-VL-3B`, `2.7 GB` INT4/Q4 ONNX) is an instruction-tuned 3B vision-language model that already lifted `caption` by `+5.6 F1`, `caption(deep=true)` by `+4.4 F1`, `query_image` by `+11.6 pts`, and `check_consistency` by `+18.9 pts`.

#### Candidates to Compare on the 50-Photo `tune` Split (Gate K3-4a)
We will test **at most 4 pre-registered candidates** on the 50-photo `tune` split in [`C:\AI\Projects\VisionBench\benchmarks\composition_prose.py`](file:///C:/AI/Projects/VisionBench/benchmarks/composition_prose.py) (comparing against `md_raw = 18.8%` F1 and against each candidate's **shuffled-pairing control** where critiques are paired with random images from the same split):

1. **Candidate P1 (`lfm_direct`, baseline swap):**
   - Exact existing prompt (`"Critique this photo's composition and framing in 2 concise sentences."`) on `app.answerer.answer(..., max_new_tokens=128)`.
   - Tests the pure model upgrade (`Moondream2` → `LFM2.5-VL-3B`) with zero prompt changes.
2. **Candidate P2 (`lfm_structured_prompt`, compositional rubric prompt, ungrounded):**
   - Prompt (`CRITIQUE_COMPOSITION_PROMPT` in `constants.py`):
     ```python
     CRITIQUE_COMPOSITION_PROMPT = (
         "Critique this photo's composition and framing in 2 to 3 concise sentences. "
         "Name the main subject and its placement in the frame (such as rule of thirds, "
         "centered, or foreground/background layers), and note one specific strength "
         "and one framing flaw or distraction (such as background clutter, tight cropping, "
         "or edge clipping)."
     )
     ```
   - Why: Aligns the output structure directly with what frontier models (`Claude Opus 4.6`, `Gemini 3.1 Pro`, `GPT 5.6`) and the `Gemini 3.8 Flash` reference critiques write (subject + placement/depth + strength + concrete flaw instead of generic praise).
3. **Candidate P3 (`lfm_soft_grounded`, visual critique with measured geometric hints):**
   - Unlike K2's `md_grounded` (which ordered Moondream2 to *"Write a 2-sentence composition critique that states these facts naturally"*, turning the VLM into a blind text rephraser), P3 gives `LFM2.5-VL-3B` a compact hint block from Track 2–4 while keeping the image primary:
     ```python
     # Example hint when box is found:
     # "Detected framing notes: primary subject 'bear' (large, center of frame, touching bottom edge); background busy=yes."
     ```
     followed by `CRITIQUE_COMPOSITION_PROMPT`.
   - Tests whether `LFM2.5-VL-3B` benefits from the measured box/edge/background cues or whether it critiques better directly from pixels (`P2`).
4. **Candidate P4 (`lfm_hybrid_lead`, concise measured lead + LFM visual critique):**
   - Only if `P3` shows that grounding improves image-specific discrimination (`delta_over_shuffled`) without hurting token F1.

> [!IMPORTANT]
> **Anti-Boilerplate Control (`delta_over_shuffled`):**
> As discovered in Sprint K2 (`composition_prose.py:181-193`), a generic stock sentence full of photography buzzwords (*"The main subject sits near the center of the frame against the background..."*) can score `~18–20%` raw token F1 even when shuffled across random photos (`delta_over_shuffled ≈ +0.4 pts`).
> Therefore, our Gate K3-4 selection rule requires **both**:
> 1. Higher mean token F1 on tune (`>= +2.0 pts` over `md_raw`'s `18.8%`), **AND**
> 2. Strongly positive image-specific discrimination: `delta_over_shuffled = mean_f1 - shuffled_f1 >= +3.5 pts` (vs `md_raw`'s `+2.3 pts`), proving the critique describes **this specific photo** rather than reciting generic photography vocabulary.

---

### 4.3 Track 2 — Subject Naming & Multi-Instance Box Selection (`name_main_subject` + `_critique_one`)

#### Step 2A: Fix the Multi-Instance Raster-Order Box Bug (`analysis.py:865`)
In [`src/fusion_vision_mcp/analysis.py:863-865`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L863-L865), replace:
```python
detected = app.florence2.detect_objects([image], subject)[0]
if detected.get("bboxes"):
    box = detected["bboxes"][0]
```
with [`_pick_primary_subject`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L1182-L1203):
```python
detected = app.florence2.detect_objects([image], subject)[0]
picked_det = _pick_primary_subject(detected, (image.width, image.height))
if picked_det is not None:
    box = picked_det["box"]
```
- **Why this works immediately:** `detected` has the exact `{"bboxes": [...], "labels": [...]}` structure expected by [`_pick_primary_subject`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L1182-L1203). When `detect_objects` returns 1 box (that is `<= 85%` of frame area), `_pick_primary_subject` returns that exact box. When `detect_objects` returns `>= 2` boxes (e.g., 4 people or 3 zebras), `_pick_primary_subject` picks the largest, most central foreground instance instead of the top-left raster-order box (`bboxes[0]`).

#### Step 2B: Migrate `name_main_subject(app, image)` from `Moondream2` to `LFM2.5-VL-3B` ONNX (`analysis.py:836-845`)
Update [`name_main_subject`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L836-L845) to call `_vlm_answer_batch(app, [image], MAIN_SUBJECT_QUESTION, max_new_tokens=16)[0]` and sanitize verbose or negative responses:
```python
_NONE_REPLIES = frozenset({"none", "nothing", "no single subject", "no main subject", "n/a"})

def name_main_subject(app: VisionAppContext, image: Image.Image) -> str:
    """Ask the VLM to name the single main foreground subject in 1–3 words.

    Returns an empty string when the model answers 'none' (wide scene with no
    single focal subject) or gives an unusable reply.
    """
    raw = _vlm_answer_batch(
        app, [image], MAIN_SUBJECT_QUESTION, max_new_tokens=16
    )[0].strip().strip(".\"'").strip()
    low = raw.lower()
    if not low or low in _NONE_REPLIES or "none" in low or "nothing" in low:
        return ""
    # Strip leading articles ("a ", "an ", "the ") for cleaner open-vocab detection
    for prefix in ("a ", "an ", "the "):
        if low.startswith(prefix) and len(raw) > len(prefix):
            raw = raw[len(prefix):].strip()
            break
    return raw
```

#### Step 2C: Evaluate Subjectless Scene Abstention (`declined_on_none`) vs Subject Recall
In Gate K3-1, we will test two variants when `target_subject is None` and `name_main_subject(app, image)` returns `""` (i.e. `LFM2.5-VL-3B` explicitly judged the photo to be a wide scene with `"none"` single subject):
- **Variant S1 (Always fallback to `dense_region_caption`):** Current behavior (`if box is None: dense_region_caption + _pick_primary_subject`). Maximizes box recall on photos where a subject exists, at the cost of `declined_on_none = 0%`.
- **Variant S2 (Selective fallback on `"none"`):** When `name_main_subject` returns `""`, only accept a fallback `dense_region_caption` box if its centrality-weighted area score is prominent (e.g., `0.08 <= box_area / img_area <= 0.65` and not a background label like `"sky"`, `"water"`, `"trees"`, `"grass"`, `"mountain"`, `"wall"`, `"floor"`, `"road"`, `"building"`, `"field"`). Otherwise leave `box = None` (`subject_detected = False`).
- **Pre-registered selection rule:** Pick S2 **only if** it raises `declined_on_none` on `"none"` scenes without reducing `iou_at_0_5` or `horiz_agree` by more than `1.0 pt` on single-subject photos across `tune` (50) and `confirm` (100) + `box2` (120); otherwise keep S1.

---

### 4.4 Track 3 — Background Checks (`busy_background` Migration & `background.blurred` Re-Evaluation)

#### Step 3A: Migrate `busy_background(app, image)` to `LFM2.5-VL-3B` ONNX (`analysis.py:815-829`)
Update [`busy_background`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L829) to call `_vlm_answer_batch(app, [image], BUSY_BACKGROUND_QUESTION, max_new_tokens=8)[0]`.
- In Gate K3-2, we evaluate `LFM2.5-VL-3B` (`app.answerer`) against `Moondream2` (`app.vqa`) on the **25 user-reviewed tune photos** and **52 user-reviewed confirm photos** (`SPOTCHECK` + `SPOTCHECK2` in [`composition_labels.py`](file:///C:/AI/Projects/VisionBench/benchmarks/composition_labels.py#L67-L237)).
- We test:
  1. **Direct non-inferiority swap (`B1`):** Exact `BUSY_BACKGROUND_QUESTION` on `app.answerer`. Since `LFM2.5-VL-3B` is already the default VQA model across the server and no prompt parameter is tuned on the spotcheck data, `B1` ships on a **non-inferiority gate** (balanced accuracy on the 77 user-reviewed photos within `2.0 pts` of or above `Moondream2`'s `75.3%` pooled / `69.2%` confirm, with `~2.5×` faster latency and `0 GB` extra RAM).
  2. **Clarified prompt (`B2`, tested on tune=25 only first):** If `B1` shows a specific bias (e.g. over-calling `"yes"` on textured natural scenes or `"no"` on indoor clutter), test a clarified prompt on the 25 tune photos and validate on the 52 confirm photos.

#### Step 3B: Re-Test `background.blurred` (Optical Depth-of-Field Check, Gate K3-3)
In Gate K3-3, we test whether `LFM2.5-VL-3B` ONNX (alone or combined with a texture-gated multi-scale edge sharpness ratio) can pass the held-out gate that `Moondream2` (`54.0%`) and raw Laplacian `sharp_ratio` (`61.4%` / `54.2%`) failed in Sprint K1:
- **Candidate D1 (`lfm_blur_vqa`):** Ask `app.answerer` with `BLUR_BACKGROUND_QUESTION`:
  ```python
  BLUR_BACKGROUND_QUESTION = (
      "Is the background behind the main subject clearly out of focus or optically blurred "
      "(shallow depth of field), rather than just far away or plain? Answer yes or no."
  )
  ```
- **Candidate D2 (`edge_sharp_ratio` + `lfm_blur_vqa` agreement):** Compute the 1 px vs 4 px step-edge sharpness ratio restricted to textured edges (`grad_4px >= 12.0`) inside the central `70%` of `box` vs outside `box`.
- **Strict Ship / No-Ship Rule for `background.blurred`:**
  - Because `SPOTCHECK` (25 tune) + `SPOTCHECK2` (52 confirm) were already inspected during K1, `background.blurred` will **ONLY** be added to `critique_composition`'s output if a candidate achieves **`>= 75.0%` balanced accuracy on the 25 user tune labels AND `>= 72.0%` balanced accuracy with `95% bootstrap CI lower bound >= 60.0%` on the 52 user confirm labels**.
  - If neither D1 nor D2 clears that bar cleanly, **do NOT ship `background.blurred`** — keep `background: {"busy": busy}` only and record the negative result in `CLAUDE.md` and `frontier_capability_gaps.md`. Never ship a noisy metric just to add a JSON key.

---

### 4.5 Track 4 — Deterministic Framing Geometry Enrichment & Batch/Single Unification

#### Step 4A: Enrich `rule_of_thirds(image_size, box)` in `src/fusion_vision_mcp/geometry.py`
In [`src/fusion_vision_mcp/geometry.py:639-672`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L639-L672), keep all 4 existing keys untouched (`centroid`, `thirds_offset`, `center_offset`, `nearest_gridpoint`) and add the 5 deterministic K1-validated framing keys:

```python
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = box_area / (w * h)
    if area_ratio < 0.10:
        fill = "small"
    elif area_ratio < 0.35:
        fill = "medium"
    elif area_ratio < 0.75:
        fill = "large"
    else:
        fill = "fills"

    horizontal_zone = "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")
    vertical_zone = "top" if cy < 0.40 else ("bottom" if cy > 0.60 else "middle")

    edge_margin = 0.01
    touching_edges: list[str] = []
    if x1 <= w * edge_margin:
        touching_edges.append("left")
    if y1 <= h * edge_margin:
        touching_edges.append("top")
    if x2 >= w * (1.0 - edge_margin):
        touching_edges.append("right")
    if y2 >= h * (1.0 - edge_margin):
        touching_edges.append("bottom")
```
Return dictionary:
```python
    return {
        "centroid": [round(cx, 4), round(cy, 4)],
        "thirds_offset": [round(dx, 4), round(dy, 4)],
        "center_offset": [round(dcx, 4), round(dcy, 4)],
        "nearest_gridpoint": [round(nearest[0], 4), round(nearest[1], 4)],
        "subject_area_ratio": round(area_ratio, 4),
        "fill": fill,
        "horizontal_zone": horizontal_zone,
        "vertical_zone": vertical_zone,
        "touching_edges": touching_edges,
    }
```

#### Step 4B: Unify Single-Image and Multi-Image Batch Paths (`src/fusion_vision_mcp/tools/aesthetics.py`)
In [`src/fusion_vision_mcp/tools/aesthetics.py:196-255`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L196-L255):
- Currently, `len(pil_images) == 1` calls `_critique_one(app, pil_images[0], target_subject)`, while `len(pil_images) > 1` runs a divergent inline loop that skips `name_main_subject`, skips `_pick_primary_subject` on `detect_objects`, omits `background: {"busy": ...}`, and uses a different VQA prompt.
- Replace the divergent `len(pil_images) > 1` loop so that multi-image calls produce the **exact same schema, subject localization, `background` dict, and critique prompt** as single-image calls (calling `_critique_one(app, img, target_subject)` per image, or batching the `app.answerer` / `app.florence2` calls across images while preserving identical logic).

---

## 5. Pre-Registered VisionBench Gates & Ship Rules

> [!IMPORTANT]
> Per [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L312) (*"Each still needs its own VisionBench gate before switching"*) and the data-hygiene rules ([`CLAUDE.md:1065-1086`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L1065-L1086)), every change is gated independently in the order below. All commands are Windows PowerShell 5.1 compatible.

### Gate K3-0: Baseline Verification & Offline Unit Test Check
- **Goal:** Verify existing unit tests pass and record current baseline numbers from disk before touching any benchmark script or model path.
- **Baseline numbers on disk (verified):**
  - `composition_critique_tune.json` (`n=50`): `iou_mean = 0.566`, `iou_at_0_5 = 66.7%`, `horiz_agree = 79.5%` (`89.7%` on good box), `fill_agree = 71.8%` (`82.8%` on good box), `edges_exact = 61.5%`, `declined_on_none = 0.0%` (`0/11`), `busy_vqa_bal_acc = 80.1%` (vs teacher) / `88.1%` (vs 25 user spotcheck labels).
  - `composition_critique_confirm.json` (`n=100`): `iou_mean = 0.619`, `iou_at_0_5 = 70.8%`, `horiz_agree = 83.1%` (`89.9%` on good box), `fill_agree = 68.5%` (`76.8%` on good box), `edges_exact = 64.0%`, `declined_on_none = 0.0%` (`0/11`), `busy_vqa_bal_acc = 69.2%` (`95% CI [56.5%, 81.0%]` vs 52 user spotcheck labels).
  - `composition_box2.json` (`n=120`): `iou_mean = 0.664`, `iou_at_0_5 = 72.9%`, `horiz_agree = 79.2%` (`88.6%` on good box), `fill_agree = 66.7%` (`77.1%` on good box), `declined_on_none = 0.0%` (`0/24`).
  - `composition_prose_tune.json` (`n=50`): `md_raw` mean token F1 = `18.8%` (`median = 18.2%`, `shuffled_f1 = 16.5%`, `delta_over_shuffled = +2.3%`, `mean_words = 31.5`).
  - Phase 7 (`n=25`, `run_visionbench.py --phases 7`): `critique_composition` token F1 = `26.6%` (max over 3 references) / `25.0%` like-for-like (max over 2 references vs `29–34%` frontier-vs-frontier).

### Gate K3-1: Track 2 — Subject Naming (`LFM2.5-VL-3B`) + Multi-Box `_pick_primary_subject` Fix
- **What is tested:**
  1. Sub-step 1A (code-only bug fix): `detected["bboxes"][0]` → `_pick_primary_subject(detected, (w, h))` in `analysis.py:865`.
  2. Sub-step 1B (model swap): `name_main_subject` using `app.answerer` (`LFM2.5-VL-3B` ONNX) vs `app.vqa` (`Moondream2`).
  3. Sub-step 1C (abstention check): Variant S1 vs S2 on `"none"` photos.
- **Evaluation splits:**
  - Tune on `composition_labels.json` `split == "tune"` (`n=50`).
  - Confirm once on `composition_labels.json` `split == "confirm"` (`n=100`) and `composition_box2.json` (`n=120`).
- **Pre-registered pass criteria:**
  - **Sub-step 1A (`_pick_primary_subject` on multi-box detections):** Ships if `iou_mean` and `horiz_agree` on `tune` (`50`) and `confirm` (`100`) + `box2` (`120`) are `>=` baseline (zero regression, strictly better on multi-instance photos).
  - **Sub-step 1B (`LFM2.5-VL-3B` for `name_main_subject`):** Because this replaces the legacy `Moondream2` dependency with the already-resident `LFM2.5-VL-3B` ONNX model (`0 GB` additional RAM, `~2.5×` faster), it ships on **non-inferiority**: on `confirm` (`100`) and `box2` (`120`), `horiz_agree` must be within `1.5 pts` of or above `Moondream2` (`83.1%` confirm / `79.2%` box2) and `iou_mean` within `0.015` of or above `Moondream2` (`0.619` confirm / `0.664` box2).

### Gate K3-2: Track 3A — `busy_background` Migration to `LFM2.5-VL-3B` ONNX
- **What is tested:** `busy_background(app, image)` via `app.answerer` (`LFM2.5-VL-3B` ONNX) vs `app.vqa` (`Moondream2`).
- **Evaluation splits:**
  - Tune: `25` user-reviewed photos in `SPOTCHECK` + `SPOTCHECK2` (`split == "tune"`).
  - Confirm: `52` user-reviewed photos in `SPOTCHECK` + `SPOTCHECK2` (`split == "confirm"`).
- **Pre-registered pass criteria:**
  - **Candidate B1 (exact `BUSY_BACKGROUND_QUESTION` on `LFM2.5-VL-3B`):** Ships on **non-inferiority** if balanced accuracy on the `52` confirm user labels is `>= 67.5%` (within `1.7 pts` of `Moondream2`'s `69.2%`, well inside its `[56.5%, 81.0%]` CI) with `95% bootstrap CI lower bound >= 55.0%`.
  - **Candidate B2 (if prompt wording is modified on tune):** Because modifying the prompt uses the `25` tune labels, a modified prompt must score `>= 72.0%` balanced accuracy on the `52` confirm user labels (`CI_lo >= 58.0%`).

### Gate K3-3: Track 3B — Optional `background.blurred` Re-Evaluation
- **What is tested:** Candidate D1 (`LFM2.5-VL-3B` with `BLUR_BACKGROUND_QUESTION`) and Candidate D2 (`edge_sharp_ratio` + `LFM2.5-VL-3B`) on the `25` tune user labels and `52` confirm user labels.
- **Pre-registered pass criteria (strict superiority required to add a new output key):**
  - Must achieve **`>= 75.0%` balanced accuracy on tune (`n=25`) AND `>= 72.0%` balanced accuracy on confirm (`n=52`) with `95% bootstrap CI lower bound >= 60.0%`**.
  - **Default outcome if criteria are not met:** Keep `background.blurred` omitted (`background = {"busy": busy}`).

### Gate K3-4: Track 1 — `LFM2.5-VL-3B` Prose Critique (Tune → Unseen Confirm → Phase 7 Scoreboard)
- **Step K3-4a (Tune on 50 photos):**
  - Run [`C:\AI\Projects\VisionBench\benchmarks\composition_prose.py`](file:///C:/AI/Projects/VisionBench/benchmarks/composition_prose.py) `--split tune` (`n=50`) comparing `md_raw` (`18.8%`) against `P1` (`lfm_direct`), `P2` (`lfm_structured_prompt`), and `P3` (`lfm_soft_grounded`).
  - Pick the **single best candidate** on `tune` that achieves:
    1. `mean_f1 >= 21.0%` (`>= +2.2 pts` over `md_raw`'s `18.8%`), AND
    2. `delta_over_shuffled = mean_f1 - shuffled_f1 >= +3.5 pts` (vs `md_raw`'s `+2.3 pts`), AND
    3. `mean_words` between `35` and `75` words (matching the 2–3 sentence reference length without truncation).
- **Step K3-4b (One-Shot Confirmation on 100 Unseen Confirm Photos):**
  - Run `composition_prose.py --split confirm` (`n=100`, **never previously run in K2**) comparing `md_raw` vs the **single chosen candidate** from Step K3-4a.
  - **Pre-registered ship rule on `confirm` (`n=100`):**
    1. `mean_f1` of the chosen `LFM2.5-VL-3B` candidate beats `md_raw` by **`>= +2.5 pts`**,
    2. `95%` paired bootstrap CI on `(f1_lfm - f1_md)` (`2,000` resamples, `seed=42`) **excludes `0.0`** (`CI_lo > 0.0`),
    3. `delta_over_shuffled >= +3.5 pts`, and
    4. Win rate (`%` of the 100 photos where `f1_lfm > f1_md`) is **`>= 60.0%`**.
- **Step K3-4c (Phase 7 Scoreboard Evaluation on 25 Photos × 3 Frontier References):**
  - Once Step K3-4b passes, run Phase 7 (`run_visionbench.py --phases 7`) on the 25 golden photos (`golden_critique_claude.json`, `golden_critique_gemini.json`, `golden_critique_GPT.json`).
  - Record both:
    - **Standard Phase 7 score** (max token F1 across all 3 references, baseline `26.6%`), and
    - **Like-for-like pairwise score** (mean of max token F1 against each pair of 2 references, baseline `25.0%` vs `29–34%` frontier-vs-frontier).
  - Target: Lift Phase 7 token F1 from `26.6%` (`25.0%` like-for-like) to **`>= 30.0–33.0%`** (`>= 28.5–31.0%` like-for-like), moving `critique_composition` from `🔻 weaker` (`−7 pts`) to **`≈ close` / `✅ matches` frontier level**!

---

## 6. Exact Implementation Specification for `c:\AI\MCP\FusionVisionMCP`

### 6.1 [`src/fusion_vision_mcp/constants.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L318-L333)
In the `# Composition critique (Sprint K1)` section (`lines 318–332`), add:
```python
# Maximum new tokens for 2-3 sentence composition critiques via LFM2.5-VL-3B ONNX.
# VQA_MAX_NEW_TOKENS (64) truncates 2-3 sentence critiques mid-sentence; 128 tokens
# (~85-95 words max) fits 2-3 complete sentences cleanly.
CRITIQUE_MAX_NEW_TOKENS = 128

# Maximum new tokens for short composition classification/naming questions.
CRITIQUE_SHORT_MAX_NEW_TOKENS = 16

# Prompt for 2-3 sentence composition & framing critique (selected by Gate K3-4).
CRITIQUE_COMPOSITION_PROMPT = (
    "Critique this photo's composition and framing in 2 to 3 concise sentences. "
    "Name the main subject and its placement in the frame (such as rule of thirds, "
    "centered, or foreground/background layers), and note one specific strength "
    "and one framing flaw or distraction (such as background clutter, tight cropping, "
    "or edge clipping)."
)
```

### 6.2 [`src/fusion_vision_mcp/geometry.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L639-L672)
Update [`rule_of_thirds(image_size, box)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L639-L672) to append the 5 deterministic framing descriptors (`subject_area_ratio`, `fill`, `horizontal_zone`, `vertical_zone`, `touching_edges`) while preserving all 4 existing keys and their rounding (`4` decimal places).

### 6.3 [`src/fusion_vision_mcp/analysis.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L912)
1. Add `_vlm_answer_batch(app, images, prompts, *, max_new_tokens=VQA_MAX_NEW_TOKENS)` above `busy_background` (`line 815`), using `app.answerer.answer(...)` when `getattr(app, "vqa_model", "lfm2.5-vl") != "moondream"` and falling back to `app.vqa.query(...)` on `(OSError, ImportError, MemoryError, AttributeError)` or when `vqa_model == "moondream"`.
2. Update [`busy_background(app, image)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L815-L829) (after Gate K3-2 passes) to use `_vlm_answer_batch(app, [image], BUSY_BACKGROUND_QUESTION, max_new_tokens=CRITIQUE_SHORT_MAX_NEW_TOKENS)[0]`.
3. Update [`name_main_subject(app, image)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L836-L845) (after Gate K3-1 passes) to use `_vlm_answer_batch(app, [image], MAIN_SUBJECT_QUESTION, max_new_tokens=CRITIQUE_SHORT_MAX_NEW_TOKENS)[0]` and strip leading articles (`"a "`, `"an "`, `"the "`).
4. Update [`_critique_one(app, image, target_subject, include_vqa=True)`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912):
   - At line 865, replace `box = detected["bboxes"][0]` with:
     ```python
     picked_det = _pick_primary_subject(detected, (image.width, image.height))
     if picked_det is not None:
         box = picked_det["box"]
     ```
   - At lines 909–911, replace `app.vqa.query([image], ...)[0]` with the winning prose critique call from Gate K3-4 (`_vlm_answer_batch(app, [image], prompt, max_new_tokens=CRITIQUE_MAX_NEW_TOKENS)[0]`).

### 6.4 [`src/fusion_vision_mcp/tools/aesthetics.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/aesthetics.py#L129-L265)
1. Unify the multi-image batch branch (`len(pil_images) > 1`, lines 212–255) with [`_critique_one`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L848-L912) so multi-image calls include `background: {"busy": ...}`, use `name_main_subject` + `detect_objects` + `_pick_primary_subject`, and run the same `LFM2.5-VL-3B` critique prompt.
2. Update the tool docstring (`lines 136–177`) to document:
   - The `LFM2.5-VL-3B` VLM + `Florence-2` subject localization pipeline,
   - The enriched `framing` keys (`subject_area_ratio`, `fill`, `horizontal_zone`, `vertical_zone`, `touching_edges`), and
   - The updated VisionBench benchmark note.

### 6.5 [`src/fusion_vision_mcp/cli.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/cli.py) & [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md)
1. Update the `--vqa-model` help string in [`src/fusion_vision_mcp/cli.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/cli.py) so it accurately states that `lfm2.5-vl` is the default for `query_image`, `count_objects(vqa_estimate=true)`, `check_consistency`, and `critique_composition` (with `moondream` available only as an explicit legacy override).
2. Update [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md) (`critique_composition` section, memory/model tables, and lessons learned) and [`Future_Ideas/frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md) (Row 103 and Section *"Composition prose critique"*) with the confirmed post-gate numbers.

### 6.6 Permanent Automated Tests (`tests/test_analysis_helpers.py` & `tests/test_server.py`)
1. [`tests/test_analysis_helpers.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_analysis_helpers.py):
   - `TestBusyBackground` (`lines 423–441`): Add tests verifying `busy_background` prefers `app.answerer.answer` when `app.vqa_model == "lfm2.5-vl"` and falls back cleanly to `app.vqa.query` when `app.vqa_model == "moondream"` or when `app.answerer` raises `OSError`.
   - `TestNameMainSubject` (`lines 443–452`): Add tests verifying article stripping (`"a brown bear"` → `"brown bear"`), `"none"` / `"nothing"` abstention, and `app.answerer` routing with `app.vqa` fallback.
   - `TestCritiqueOneMultiBoxSelection` (new test class): Verify that when `app.florence2.detect_objects` returns a small top-left box `[5, 5, 25, 25]` first and a large central foreground box `[150, 120, 450, 420]` second on a `600×600` image, `_critique_one` selects the large central box `[150, 120, 450, 420]` instead of `bboxes[0]`.
   - `TestRuleOfThirdsEnrichment` (new test class): Verify `rule_of_thirds` returns both the original 4 keys and the 5 new keys (`subject_area_ratio`, `fill`, `horizontal_zone`, `vertical_zone`, `touching_edges`) with exact expected values across edge-touching, small, medium, large, and full-frame boxes.
2. [`tests/test_server.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_server.py):
   - Update `test_critique_composition_*` (`lines 887–947`, `1166–1185`) to assert single-image and multi-image batch parity (both return `background.busy`, `framing.fill`, `framing.horizontal_zone`, `framing.vertical_zone`, `framing.touching_edges`, and `critique`).
   - Run the torch-free eager import check:
     ```powershell
     .\.venv\Scripts\python.exe -c "import sys, fusion_vision_mcp; assert 'torch' not in sys.modules, 'torch leaked!'"
     ```

---

## 7. Step-by-Step Execution Order for Opus 5.5

1. **Step 1 — Run Gate K3-0 (Unit tests + baseline check):**
   - Run `.\.venv\Scripts\pytest.exe tests/test_analysis_helpers.py tests/test_server.py -q` in `c:\AI\MCP\FusionVisionMCP`.
2. **Step 2 — Run Gate K3-1 (Track 2: `_pick_primary_subject` on multi-box detections + `LFM2.5-VL-3B` `name_main_subject`):**
   - Evaluate `_pick_primary_subject` on `detect_objects` and `LFM2.5-VL-3B` (`app.answerer`) in `C:\AI\Projects\VisionBench\benchmarks\composition_critique.py` on `--split tune` (`50`), then confirm once on `--split confirm` (`100`) and `--box2` (`120`).
3. **Step 3 — Run Gate K3-2 & Gate K3-3 (Track 3: `busy_background` on `LFM2.5-VL-3B` + optional `background.blurred` check):**
   - Evaluate `LFM2.5-VL-3B` on the 25 tune and 52 confirm user-reviewed labels (`SPOTCHECK` + `SPOTCHECK2`). Ship `busy_background` on `LFM2.5-VL-3B` if Gate K3-2 passes; ship `background.blurred` **only** if Gate K3-3 clears both tune (`>= 75%`) and confirm (`>= 72%`, `CI_lo >= 60%`).
4. **Step 4 — Run Gate K3-4 (Track 1: `LFM2.5-VL-3B` Prose Critique):**
   - Extend `C:\AI\Projects\VisionBench\benchmarks\composition_prose.py` with `P1` (`lfm_direct`), `P2` (`lfm_structured_prompt`), and `P3` (`lfm_soft_grounded`) at `max_new_tokens=128`.
   - Run `--split tune` (`n=50`) and select the single winning `LFM2.5-VL-3B` variant.
   - Run `--split confirm` (`n=100`, unseen) once to verify the pre-registered ship rule (`>= +2.5 pts` over `md_raw`, paired bootstrap `CI_lo > 0`, `delta_over_shuffled >= +3.5 pts`, win rate `>= 60%`).
5. **Step 5 — Ship Winning Components in `src/fusion_vision_mcp/` & Add Permanent Tests:**
   - Apply edits to `constants.py`, `geometry.py`, `analysis.py`, `tools/aesthetics.py`, `cli.py`, `tests/test_analysis_helpers.py`, and `tests/test_server.py`.
   - Run full test suite (`.\.venv\Scripts\pytest.exe -q`) and torch-free import check.
6. **Step 6 — Run Phase 7 Scoreboard Evaluation (`run_visionbench.py --phases 7`) & Update Docs:**
   - Run Phase 7 (`25` photos × `3` frontier references) once to measure the new scoreboard number and latency.
   - Update [`Future_Ideas/frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md), [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md), and record final results in this plan.
