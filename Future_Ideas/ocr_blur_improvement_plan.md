# OCR Blurred-Image Improvement Plan

**Created:** 2026-10-20 (original draft, Sections 1-7). **Rewritten:** 2026-10-25 (Section 0).
**Status:** Plan only. Section 0 is the plan to iterate through and supersedes Sections 1-7 where they differ.
Nothing has run. Each step waits for the user's go-ahead.

## 0. Current plan (2026-10-25)

### 0.1 Why the original draft needed rewriting

Between 2026-10-21 and 2026-10-25 the OCR-TextVQA plan (`OCR-TextVQA_Improvement_Plan.md`, Sections 0.6 and
0.16-0.18) ran most of this draft's ideas, with `p11_blur` scored as a set:

| Idea in Sections 1-7 | What happened | Consequence here |
| :--- | :--- | :--- |
| Track 3 arbiters: re-read PP-OCR's boxes with another recognizer (3A-3C) | Gate 0 wave 1 (VisionBench `ocr_gate0.py`, `results/ocr_gate0_wave1.json`): `p11_blur` 37.4 -> ppocr_rec 41.8, trocr_small 41.9, parseq_orig 40.6, trocr_str 40.4, doctr_parseq 40.0, easyocr_rec 39.6; winocr combinations ~37.4. Failed the overall gate | A ~+4 lead **on blur only**, never tested gated to blurred images. Carried into B2 as one arm |
| Step 1C.2 grey-zone rescue (0.52-0.75 with corroboration) | Fails the text-free controls at every grid point, even with two readers agreeing ("S" on blank images) | Dropped. The 0.75 floor stays |
| 3D, PP-OCRv5 server / medium models | Track 4 + 4b: every arm reads stray single characters and CJK on text-free images; no floor or Latin filter qualified | Dropped |
| B3, "the detector downscales 1920x1080 images 2-2.7x" | Wrong: `limit_side_len 736` uses `limit_type: min`, so it only **upscales** a short side below 736 | Dropped as a diagnosis. Large side lengths stay an untested arm, not a fix |
| Track 2, rotation | Rotation is at par (-5.7 vs -5 to -10 points) | Out of scope. Rotated sets are kept as controls only |
| Track 1A/1B, a sharpness gate and image restoration before the engine | **Never tested** | The core of this plan |
| Blur-gated detector settings | Never tested; PP-OCR detector settings are allowed since 2026-10-23 (the user's no-Chinese-models exception) | Part of B2 |

Two measurement facts also change the gates:

- **`p11_blur` is synthetic and narrow.** It is phase 11's text images with one Gaussian blur, radius 2
  (`rv._perturbations()["blur"]`). It has no motion blur, defocus, low resolution or JPEG damage, and it is
  one strength. A gain on it may not carry to real photos, and a sharpness gate calibrated on it may not fire on
  real blur.
- **The frontier figure (-20 to -30 points) is an estimate** (E, low confidence), not a measurement.

### 0.2 Data

Decided 2026-10-25: confirm on **real** blurred images, with synthetic blur for tuning.

- **Tune:** `p11_blur` tune half (155 images, baseline 37.4%, `p11_text` 76.8%), plus a **synthetic ladder** built
  from the same `p11_text` tune images: Gaussian 1 / 2 / 3, horizontal motion blur (9 and 15 px), and 0.5x
  downscale then JPEG quality 40, upscaled back. Labels come free (phase 11 key phrases), so it shows how gains
  change with blur strength and type.
- **Real blur, round 25 (new; worker sources, Claude builds, per the benchmark image policy):** about 150 photos
  and screenshots whose text is blurred in the original (hand shake, missed focus, motion, low resolution,
  heavily compressed), spread across signage, documents, screens, packaging and vehicles. The worker writes key
  phrases **and marks each one as legible to a person or not**; only legible phrases are scored, so the set
  measures reading, not guessing. The user blind-checks a sample of the legibility calls. Split **40 tune / 110
  confirm**. The tune 40 are only for checking that the sharpness gate fires on real blur, never for picking
  settings.
- **Confirm (each scored once, at the end):** round 25 confirm (110) and the `p11_blur` confirm half, which no
  blur arm has scored yet.
- **Controls, at every gate:** text-free sets (p11 text-free tune: 4 non-empty at baseline; the 4 fixtures: 0),
  **also blurred** (Gaussian 2 and 3); `p11_text`, `rendered`, `rendered_jpeg`, `lines`, TextVQA tune
  (the sharpness gate should leave all of them alone); the phase 13 rotated images.
- **Frontier measurement (approved 2026-10-25):** Claude reads round 25 confirm with native vision (the user gave
  permission for this task), replacing the -20 to -30 estimate. It does not depend on B2-B4, so it runs as soon as
  the images are built, **before** stage A's labels exist, and the reading is committed before any label is seen.
  Scored once stage A and the blind check are done. Round 25 is labelled by Gemini, so a Gemini reading, if run, is
  scored only on phrases the user's blind check confirmed.
- **2026-10-26 update.** Round 25 kept only 65 of 217 (92 had sharp text at full size), so round 25b (VisionBench
  `ROUND25B_PLAN.md`, ~100 more, candidates checked at full size) tops it up and the two are pooled. Readers, by the
  user's choice: **Opus** reads round 25's 65 kept images; **Sonnet** reads round 25b. Each model's figure is
  reported on its own images, never blended. A single pooled figure would need one model to read both rounds.

### 0.3 Steps

Each step stops and reports. Benchmarks run on CPU in VisionBench, never while the user is stress-testing or
another timed job is running.

**B0. Paperwork, no CPU (Opus 5.5).** This section (done 2026-10-25). Round 25's brief is written (VisionBench
`worker_handoff/ROUND25_PLAN.md`, 2026-10-25: six kinds, ~150 kept, key phrases rated `clear` / `with effort`,
guesses left out). Still to do: B1's rules into its script docstring, and the learned-deblur shortlist.

**B1. Diagnosis and sharpness gate (Sonnet 5.5; light CPU, mostly saved results).**
- **Miss buckets on `p11_blur` tune:** each missed key phrase is (a) never boxed, (b) boxed but every span under
  0.75, or (c) boxed and misread. Use Gate 0's saved boxes and spans where they exist; re-run PP-OCR only for what
  is missing. Save the per-phrase table.
- **Sharpness gate:** an edge-width ratio (1 px vs 4 px gradient steps) and contrast-normalised Laplacian
  energy, measured over the highest-gradient patches only, so a sharp caption on a blurred background reads sharp.
  Pure numpy, a few ms. **Rule:** a threshold exists that flags >= 90% of `p11_blur` tune and of the synthetic
  ladder at Gaussian 2+, and <= 2% of each sharp control set. Report its rate on round 25 tune once that exists.
- **Stop rule:** if no threshold meets the rule, report before B2. A gate that also fires on sharp images makes
  every B2 arm a whole-pipeline change, held to the sharp controls at full size.

**B2. Restore or re-read, blurred images only (Sonnet 5.5 runs, Opus 5.5 reviews).** Every arm runs only on
images the B1 gate flags; others take today's path unchanged.
- Arms, each alone and then the best combination:
  1. CLAHE on lightness.
  2. Unsharp masking, a small grid (sigma 1.5 / 2.5, amount 1.0 / 1.6 / 2.2).
  3. Deconvolution: Richardson-Lucy (5 / 10 iterations) and Wiener, Gaussian PSF sized from the edge-width ratio.
  4. 1.5x upscale before (1)-(3), for small images.
  5. Detector settings: `Det.thresh` 0.22, `box_thresh` 0.30, `unclip_ratio` 1.9, and a larger side length.
  6. Gate 0's best box re-read (ppocr_rec or trocr_small), gated.
  7. **If the user wants it:** a learned deblur model, non-Chinese (to be shortlisted in B0; NAFNet is excluded).
- **Fixed:** the 0.75 floor, PP-OCRv6 English.
- **Pick on tune:** most `p11_blur` key phrases, ties to the faster arm. Report the synthetic ladder for the pick.
- **Rule to go on:** `p11_blur` tune >= +8.0 points (paired bootstrap CI lower bound > 0). No text-free image
  newly non-empty, blurred or not; sharp controls within -0.5 points; median time on blurred images <= 5 s and
  under 5 ms added on sharp ones.
- **Stop rule:** if every arm gains < +3 points, stop and report the B1 buckets. Restoration cannot help bucket
  (c) misreads much, and that is B3's question.

**B3. Only if B1 shows bucket (c) is large and B2 leaves it: LFM2.5-VL-3B reads blurred boxes (Opus 5.5).**
LFM is already loaded for captions and `query_image` and is non-Chinese. It sees only a padded box PP-OCR already
found on a blurred image, and its reading is kept only if it shares at least half its characters with PP-OCR's
raw reading (so it can correct letters but not invent a line). Low prior: other crop re-readers gained ~4 points.
Same rules as B2, plus the text-free sets blurred.

**B4. Confirm, once (Sonnet 5.5 runs, Opus 5.5 verifies).** The frozen winner on round 25 confirm and the
`p11_blur` confirm half. **Ship rule, fixed before round 25 is built:** round 25 confirm legible-phrase recall
CI lower bound > 0 and `p11_blur` confirm >= +5.0 points; every control as in B2. Report the frontier
comparison if Gemini was run.

**B5. Ship (Opus 5.5).** `is_text_blurred` and the restoration step on the `ocr_pages` path (torch-free, constants
in `constants.py`); tests: a sharp page takes 0 extra steps, a blurred `layout_paragraph.png` recovers words, a
blurred `tests/sample.jpg` stays empty. `caption`'s `Visible text` line and `query_image`'s OCR context inherit
it; check both still reproduce their gate parity. Update `CLAUDE.md`, the global routing note and the
scoreboard row.

**Outcome (2026-10-30, shipped).** B1: the sharpness gate passed synthetic blur but fired on ~10% of real blur, so
B2 ran ungated as an additive second pass. B2 (12 arms): only unsharp masks gained on both blur sets; every
finalist failed the text-free check on lone letters. B2b: added spans need 3+ characters (rule written after B2).
B4, once on held-out images: real blur +4.0 (CI 2.1 to 6.2), p11_blur +8.1, sharp controls up, text-free held,
+1.4 s median. B3 was not run. B5 shipped it on `ocr` (and batch / structured OCR), not on `caption`'s text line or
`query_image`'s OCR context, which keep the first reading until their own gate. Details: CLAUDE.md, the `ocr`
section.

**Beyond the plan (2026-10-30, shipped as default by the user's choice).** B3 as written (more preprocessing) was
judged too timid once B2 showed heavy blur at +0 on every arm; PP-OCR has no language context. LFM2.5-VL-3B ONNX
was screened as a whole-page transcriber whose new lines are appended only on pages already read non-empty
(VisionBench `ocr_blur_lfm_screen.py`, then `ocr_blur_lfm_confirm.py`; rules fixed before each). Confirm, once: real
blur 51.9 -> 62.2% (+10.3, CI 6.7 to 14.3), p11_blur +7.1, TextVQA +11.1, sharp controls up, text-free held; median
2.7 -> 7.6 s per photo, past the 5 s limit, so the rules allowed opt-in only; the user made it the default
(`ocr(lfm_read=false)` for speed). A blur-gated variant was considered and not built (needs a fresh round).

### 0.4 Order and waiting points

1. B0 now (paperwork).
2. Round 25 brief to the worker as soon as the user approves it. It runs in parallel with B1-B2 and costs no CPU
   here.
3. B1, then B2 (and B3 if earned), when the CPU is free.
4. B4 once round 25 is built and blind-checked.
5. B5 on the user's say-so.

### 0.5 Open questions

- Learned deblur (B2 arm 7): the user is fine either way. Include it only if a non-Chinese, CPU-friendly model
  shortlists in B0.
- Settled 2026-10-25: Claude native vision is the frontier reader (0.2). Optional second figure: Gemini on
  user-confirmed phrases.

---

*Sections 1-7 below are the original 2026-10-20 draft, kept for its reasoning and formulas. Where they
disagree with Section 0, Section 0 wins.*
**Target Tool:** `ocr` (`src/fusion_vision_mcp/tools/text.py`) and its backing pipeline (`src/fusion_vision_mcp/analysis.py`, `src/fusion_vision_mcp/ppocr_engine.py`, `src/fusion_vision_mcp/layout.py`, `src/fusion_vision_mcp/easyocr_engine.py`)

## 1. Scoreboard Context & Problem Framing

### 1.1 Current Scoreboard Standing (`Future_Ideas/frontier_capability_gaps.md`)

| Scoreboard Row | Backing Stack Today | Current FusionVisionMCP Score | Frontier Model Best Estimate | Status & Gap |
| :--- | :--- | :--- | :--- | :--- |
| **`ocr`, rotated / blurred** | `PP-OCRv6` English (`RapidOCR` on ONNX Runtime) + geometric column splitting (`layout.reading_blocks`) + `_read_upright` 90°/270° page re-read | **Rotation:** `−5.7 pts` (improved from `−26.0 pts` on 2026-10-03)<br>**Blur:** `−52.0 pts` (`37.4%` vs `76.8%` on `p11_blur` tune; Phase 13 key-phrase recall drop) | **Rotation:** `−5 to −10 pts` (**E**)<br>**Blur:** `−20 to −30 pts` (**E**; reads top-down context and infers blurred words) | **Rotation:** ✅ **Par** (`−5.7` vs `−5 to −10`)<br>**Blur:** 🔻 **Weaker by 22 to 32 pts** (`−52` vs `−20 to −30`) |
| **`ocr`, clean screens & documents** | Same as above | **97.3%** verbatim lines (Phase 14, 60 DOM-grounded rendered pages) | **~97–99%** (**E**); DocVQA **91–95%** (**P**) | ✅ **Par** (`0 to −2 pts`) — **Must hold as a control** |
| **`ocr`, real images by line** | Same as above | **76.2%** line recall (100 held-out sourced images; label ceiling `~80%`) | **~80%** (**L** frontier self-agreement) | ✅ **Par** (`~−4 pts`) — **Must hold as a control** |
| **`ocr`, photos (TextVQA)** | Same as above (`Det.box_thresh=0.4`) | **58.5%** answer-text recall (200 photos; `59.8%` on 92-item tune split) | **76.5–80.8%** (**P** VQA metric); **~75–80%** (**E**) | 🔻 **Weaker by 17 to 22 pts** (companion plan in `OCR-TextVQA_Improvement_Plan.md`) |
| **Text-free negative controls** | `OCR_MIN_CONFIDENCE = 0.75` floor | **4 / 4 (100%)** fixture controls empty; **92.2%** `p11_textfree` tune empty (`87.1%` full Phase 11) | **~95–100%** empty | ✅ **Load-bearing control** — **Must never regress** |

### 1.2 Objective

1. **Primary Target — Close the Blur Gap (`−52 pts` → `≤ −25 to −30 pts`, reaching ✅ Par):**
   Lift blurred key-phrase recall on the 155-image `p11_blur` tune split from **37.4% → 52.0–58.0%+** (and cut the Phase 13 blur penalty from **−52 points to ≤ −28 points**), matching the frontier model estimate (`−20 to −30 points`) without slowing down sharp images.
2. **Secondary Target — Close Residual Orientation Misses (`−5.7 pts` → `≤ −3.0 pts`):**
   Keep 90°/270° full-page rotation at par while fixing the four unaddressed orientation failure modes: (a) rotated pages with `< 3` high-confidence initial spans (especially **rotated + blurred** images), (b) **180° upside-down** pages and boxes, (c) **single vertical signs** on otherwise horizontal photos, and (d) **in-plane skewed/tilted text quads** (`8°–45°`).
3. **Hard Invariants & Negative Controls:**
   - **Zero regression on sharp images:** Blur recovery must be strictly gated by a deterministic pixel sharpness check (Laplacian variance / Brenner gradient) so sharp documents (`97.3%` rendered lines) and sharp photos take the existing fast path with **0.0% score change and < 2 ms overhead**.
   - **Text-free images stay empty:** `4/4` synthetic/photo text-free fixtures (`tests/sample.jpg`, `spatial_containment.png`, `spatial_touch_separate.png`, `detect_blank_canvas.png`) must return `[]`, and `p11_textfree` empty rate (`92.2%` tune) must not drop by more than `2.0` points.
   - **CPU Latency Cap:** Median latency on blurred/rotated images must stay **≤ 5.0 s per image** on CPU (18 threads).
   - **Torch-Free Eager Import Invariant:** `import fusion_vision_mcp` must never load `torch` (`assert 'torch' not in sys.modules`).

## 2. What Has Already Been Tried (Do Not Repeat)

Every prior OCR experiment across `CLAUDE.md`, `CHANGELOG.md`, `Future_Ideas/detector_ocr_replacement_plan.md`, and `Future_Ideas/OCR-TextVQA_Improvement_Plan.md` has been audited so rejected paths are not repeated:

| Prior Experiment | Where Measured | Outcome & Why It Failed or Succeeded | Rule for This Plan |
| :--- | :--- | :--- | :--- |
| **Whole-page 90°/270° re-read (`analysis._read_upright`)** | VisionBench Phase 13 (`2026-10-03`), `tests/test_ocr_orientation.py` | **Shipped.** Checks if `≥ 3` spans (`len ≥ 3`) have `h > 1.5w` and make up `≥ 60%` of spans; if so, re-reads at `ROTATE_90` and `ROTATE_270` and keeps the orientation with the most characters. Cut rotation loss from **−26 pts to −5.7 pts**. | Keep `_read_upright` as the foundation, but fix its blind spots on `< 3` spans, 180° rotation, and pre-rotation column splitting. |
| **PP-OCR Detector Sweep on Clean Images (`ppocr_det_sweep.py`)** | 200 TextVQA + 60 rendered pages + 100 sourced images (`2026-10-03`) | **Shipped `Det.box_thresh = 0.4`** (TextVQA `55.0% → 58.5%`). **Rejected as unconditional defaults:** larger `limit_side_len`, `Det.thresh = 0.2`, `Det.unclip_ratio = 2.0`, and unconditional `1.5×` upscale (gained less or lost lines on clean rendered pages). | **Never change detector geometry (`thresh`, `unclip_ratio`, upscale) unconditionally on sharp images.** Apply blur-adapted detector parameters *only* when the image or crop fails the sharpness gate. |
| **PP-OCRv5 Server Detector (`ppocr_det_sweep.py`)** | 200 TextVQA + 60 rendered pages (`2026-10-03`) | Read photos best (`61–62%`), **but** dropped clean rendered pages to `87%` when unconditional, and when routed to all photos lost on held-out sourced lines while doubling photo latency. | Do not run `PP-OCRv5` server detector unconditionally on documents or all photos; only evaluate as a **blur-gated** or **sparse-photo fallback**. |
| **Unconditional Lowering of `OCR_MIN_CONFIDENCE` below `0.75`** | `CLAUDE.md`, `tests/test_ppocr_engine.py` | At `< 0.75`, PP-OCRv6 invents `"S"` (`0.52`) on `tests/sample.jpg`, `"0"` (`0.57`) on a shapes fixture, and garbles the last line of `layout_small_text_paragraph.png` (`"chee o ee s s ns s eg ge."` at `0.70`). | **Never lower `OCR_MIN_CONFIDENCE = 0.75` globally for a single uncorroborated pass.** Grey-zone spans (`0.45 ≤ conf < 0.75`) may only be rescued if they jump above `≥ 0.78` after deblurring/sharpening or are corroborated by a second recognizer. |
| **EasyOCR Magnification (`1.5×`, `2×`) & Beam Search (`ocr_param_sweep.py`)** | 200 TextVQA + synthetic paragraphs (`2026-09-29`) | Magnification did not beat defaults on photos; beam search dropped a third of a paragraph's words. EasyOCR scored `37.5%` vs PP-OCRv6's `53.0%`. | Do not revert to EasyOCR or use EasyOCR beam search for degraded text. |
| **Sprint R Gate 1 OCR Replacement Bake-Off (`replacement_gate1.py`)** | 534-item OCR tune suite incl. `p11_blur` (`2026-10-12`) | Tested `docTR` (`LinkNet`/`FAST`/`DBNet` + `PARSeq`/`ViTSTR`) and `TrOCR-base-str` / `TrOCR-large-str`. Best `docTR` (`FAST+PARSeq`) scored `54.3%` primary vs PP-OCRv6's `67.8%` and failed text-free controls at every single global floor. `TrOCR-large-str` failed Gate 0 speed (`6.7 s` median, `130 s` max). `TrOCR-base-str` had a `<pad>`-token log-prob bug in the harness that broke its confidence scores. | Do not replace `PP-OCRv6` wholesale with `docTR` or `TrOCR-large-str`. Use `PARSeq`, `TrOCR-base-str` (with pad-masking fixed), or `Florence-2` / `Moondream2` strictly as **targeted crop/blur arbiters**, not whole-page replacements. |
| **Tight Region Cropping on `ocr` (Gate R, `region_inspect_gate.py`)** | 341 text region questions (`2026-10-15`) | **Failed on `ocr`:** recall dropped `71.6% → 54.5%` (51 of 341 read nothing) because tight crops cut words at the edge and starve DBNet of background margin. | **Never pass tight unpadded crops into the full DBNet detector.** When re-reading a detected line crop, either pad generously (or canvas-pad with white/border pixels) or pass the crop directly to the **recognition head** (bypassing DBNet). |
| **Dual-OCR + Dictionary/Lexicon Filter (Sprint 20)** | `garbled_text_negative_result.md` (`2026-09-11`) | **Failed negative controls:** real brand names, acronyms, UI tokens, and non-English words fail dictionary lookups just like gibberish. | **Never use a dictionary/spellcheck filter to accept, reject, or "autocorrect" OCR output.** |
| **Fuzzy OCR Rewriting of Caption Prose (Gate B & `auto_verify_text`)** | `caption_prose_fix_gate.py`, `caption_text_recheck.py` (`2026-10-15`) | `auto_verify_text` gained `+0.0 pts` at `2×` latency (retired); Gate B dropped hits (`2,810 → 2,800`) because OCR misreads overwrote valid words. | Keep `ocr` improvements inside `ocr_pages` / `ppocr_engine.py`, which `caption(read_text=True)` automatically inherits via the verbatim `Visible text:` line. |

## 3. Root-Cause Diagnosis: Why Current Scores Are Low on Blurred & Rotated Images

### 3.1 Five Compounding Failure Modes on Blurred Images (`−52 points`)

Tracing a blurred image through `ocr_pages` (`src/fusion_vision_mcp/analysis.py`) → `layout.reading_blocks` (`src/fusion_vision_mcp/layout.py`) → `PPOCREngine.readtext` (`src/fusion_vision_mcp/ppocr_engine.py`) reveals that blur attacks **every stage of the pipeline simultaneously**:

1. **Failure Mode B1 — The `0.75` Confidence Cliff Silently Drops Correct-but-Soft Recognizer Spans:**
   - In `ppocr_engine.py`, `spans()` enforces a hard cliff: `if float(score) < min_confidence: continue` (`OCR_MIN_CONFIDENCE = 0.75`).
   - PP-OCRv6's recognizer computes `score` as the mean softmax probability of the chosen characters across the CTC/SVTR sequence. On sharp text, clean character peaks yield `0.90–0.99`. On Gaussian- or motion-blurred text (such as Phase 13's blurred images), softened character strokes flatten the softmax distribution so that a completely or mostly legible word (`"Settings"`, `"Total Due"`, `"Warning"`) scores **`0.50–0.74`** and is **discarded outright**.
   - Because `0.75` is calibrated to block hallucinations on *sharp text-free* images (`"S"` at `0.52` on `sample.jpg`), applying the same unconditioned `0.75` cliff after blur — without sharpening the crop first — throws away a massive share of detected lines.

2. **Failure Mode B2 — DBNet Kernel Erosion & Box-Edge Clipping Under Blur:**
   - PP-OCR's DBNet detector predicts a probability map trained to shrink text regions (`shrink ratio = 0.4`), binarizes it at `Det.thresh = 0.3`, filters connected components by mean score `Det.box_thresh = 0.4`, and expands the polygon outward by `Det.unclip_ratio = 1.6`.
   - Under blur, high-frequency stroke edges dissipate into the background:
     - **Missed boxes:** Small or low-contrast blurred text kernels fall below `Det.thresh = 0.3` / `Det.box_thresh = 0.4` and never produce a box.
     - **Truncated word ends:** Even when the center of a blurred word clears `0.4`, the shrunken probability kernel is narrower than on sharp text. Expanding by the standard `unclip_ratio = 1.6` leaves the bounding box **1–2 characters short on the left and right edges**, causing the recognizer to miss the first/last letters of key phrases (which then fail `run_visionbench.phrase_found`).

3. **Failure Mode B3 — Pre-Detector Downscaling of High-Resolution Blurred Images:**
   - RapidOCR's detector defaults to `Det.limit_side_len = 736` (or `960`), resizing any 1920×1080 screenshot, chart, or document down by `2×–2.7×` before detection.
   - When an image is already blurred, downscaling it by `2×` collapses the remaining inter-character and inter-line spacing below the Nyquist limit of the DBNet feature map. On clean images, a larger side length was unnecessary; on **blurred** high-res images, preserving native resolution (or upscaling small blurred images) before sharpening is critical.

4. **Failure Mode B4 — Zero Deconvolution / Contrast Restoration Before the 48px Mobile Recognizer:**
   - `PPOCREngine.readtext` passes `np.array(image.convert("RGB"))` straight into RapidOCR with zero pre-filtering.
   - RapidOCR crops each detected box and resizes it to a fixed `48px` height for the mobile SVTR/CRNN recognizer. A blurred crop fed raw into a mobile recognizer trained predominantly on sharp synthetic/scene text has washed-out stroke contrast and smeared vertical stems (`m/n/r/l/i/e/c/o` confusion).
   - Classical, deterministic signal restoration — specifically **multi-scale Unsharp Masking (USM) + CLAHE local contrast normalization** or **lightweight Wiener / Richardson-Lucy deconvolution** gated strictly on blurred images/crops — directly restores stroke edge gradients in `< 15 ms` on CPU using `cv2`/`scipy`/`numpy` already installed in `.venv`.

5. **Failure Mode B5 — Lack of Top-Down Contextual Reading on Severely Blurred Spans:**
   - As `frontier_capability_gaps.md` notes, frontier VLMs lose only `20–30 points` on blur because they **read top-down context and infer partially blurred words** from surrounding words and scene semantics.
   - A line-isolated CTC recognizer cannot use document/scene context, and on heavily blurred crops (`0.35 ≤ conf < 0.78` even after sharpening) it outputs character salads (`"Administra\ufffdo"`, `"proudly resents"`, `"GEORGLA"`).
   - Meanwhile, FusionVisionMCP already holds `Florence-2-large` and `Moondream2` (or a dedicated VLM/transformer line reader) in memory, which have full vision-language decoders capable of reading a padded blurred line/block crop in context when classical sharpening is not enough.

### 3.2 Four Failure Modes in the Remaining Orientation / Rotation Path (`−5.7 points` + Uncovered Angles)

Inspecting `src/fusion_vision_mcp/analysis.py` (`_read_page`, `_looks_sideways`, `_read_upright`, lines 519–568) explains why rotation still loses `5.7 points` on Phase 13 and fails completely on several real-world orientations:

1. **Failure Mode R1 — The `≥ 3` Spans & `≥ 0.75` Confidence Trap on Rotated + Blurred or Sparse Images:**
   - `_looks_sideways(regions)` requires:
     ```python
     counted = [r["box"] for r in regions if len(r["text"].strip()) >= 3]
     tall = sum(int((y2 - y1) > 1.5 * (x2 - x1)) for x1, y1, x2, y2 in counted)
     return tall >= 3 and tall >= 0.6 * len(counted)
     ```
   - **The Catch-22:** `regions` comes from `_read_page(app, image)`, which already drops every span with `confidence < 0.75`! When a page is rotated 90°, RapidOCR often detects vertical boxes (`y2 - y1 > 1.5 * (x2 - x1)`), **but** if the sideways recognition score on several of those boxes falls below `0.75` (or if the image only has 1–2 text regions, such as a rotated sign, chart title, or logo), `len(counted) < 3`, so **`_looks_sideways` returns `False` and the page is never rotated**!
   - Even worse: if an image is **both rotated and blurred**, initial sideways recognition confidence almost always drops below `0.75`, guaranteeing `_looks_sideways` fails to fire.

2. **Failure Mode R2 — `layout.reading_blocks` Runs *Before* Rotation on Sideways Multi-Column Pages:**
   - In `_read_upright`, `_read_page(app, image)` calls `layout.reading_blocks(image)` on the unrotated sideways image first.
   - On a 90°-rotated page, the text lines run vertically (`x`-axis) and inter-line gaps look like vertical gutters! `_find_gutters` can either slice a rotated page into strips between text lines or mis-zone it before `_looks_sideways` even has a chance to evaluate the page.

3. **Failure Mode R3 — 180° Upside-Down Pages & Single Vertical Boxes Are Never Rescued:**
   - **180° Upside-Down:** An upside-down page or sign has wide horizontal boxes (`x2 - x1 > y2 - y1`), so `_looks_sideways` is `0%` tall and never triggers. While RapidOCR has an optional `Cls` (angle classifier) module, in `ppocr_engine.py` `use_cls` is not tuned for low-confidence upside-down scene text, and upside-down multi-line reading order is reversed (bottom-to-top, right-to-left) unless the page/crop is rotated 180°.
   - **Single Vertical Sign on an Upright Photo:** If a street photo has 4 horizontal signs and 1 vertical sign (`h > 2.0w`), `tall = 1 < 60%`, so `_looks_sideways` leaves the page upright (correct for the page) — and the 1 vertical sign is never rotated individually if RapidOCR's internal crop orientation misses it.

4. **Failure Mode R4 — In-Plane Skew (`5°–45°`) & Quad-to-AABB Collapse:**
   - `ppocr_engine.py::spans` collapses RapidOCR's 4-point oriented polygon (`box`) into an axis-aligned rectangle `[min(xs), min(ys), max(xs), max(ys)]`.
   - When a document scan, receipt, or sign is tilted by `10°–35°`, `layout.reading_blocks` projects ink strictly along horizontal and vertical axes (`ink.sum(axis=0)` and `ink.sum(axis=1)`), where tilted lines overlap in `y`-projection and tilted gutters close up in `x`-projection. Deskewing the page when dominant line angles are tilted (`|θ| ∈ [3°, 40°]`) restores both `reading_blocks` and horizontal recognition.

## 4. Multi-Track Improvement Plan

Every track is designed to be **modular, torch-free at import time, and gated by deterministic image/box measurements** so sharp, upright pages pay zero accuracy or latency penalty.

```
Input Image
   │
   ▼
[Stage 1: Fast Orientation & Deskew Pre-Check (Track 2)]
   ├─ Detect dominant stroke/line orientation & raw detector box aspect ratios (before 0.75 filter)
   ├─ If sideways (even 1-2 tall boxes or raw tall boxes) -> test 90°/270° BEFORE column split
   ├─ If upside-down signature (high raw box count, low rec confidence) -> test 180°
   └─ If page skew |θ| in [3°, 35°] -> affine deskew + inverse box mapping
   │
   ▼
[Stage 2: Sharpness & Blur Diagnosis Gate (Track 1A)]
   ├─ Compute text-band Laplacian variance (S_lap) & Tenengrad gradient on ink regions
   ├─ Sharp (S_lap >= T_sharp) ──► Standard Fast Path (reading_blocks + PP-OCRv6 @ 0.4 / 0.75)
   └─ Blurred (S_lap < T_sharp) ─► Blur-Aware Path (Track 1B + Track 1C + Track 3)
                                      ├─ Bypass spurious column splits on blurred halos
                                      ├─ Multi-scale Unsharp Mask + CLAHE + Richardson-Lucy (3-5 iter)
                                      ├─ Blur-adapted DBNet (Det.thresh=0.22, box_thresh=0.30, unclip=1.9, limit_side_len=1536)
                                      ├─ Per-Crop Two-Pass Sharpened Re-Read & Grey-Zone Rescue (0.50-0.75)
                                      └─ Context-Aware VLM / Point-OCR Arbiter on Surviving Low-Conf Crops (Track 3)
```

### Track 1: Blur-Gated Signal Restoration & Detector/Recognizer Adaptation (Engine-Independent, Zero New Weights)

This is the highest-ROI track because it directly attacks Failure Modes **B1, B2, B3, and B4** using `opencv-python-headless`, `scipy`, and `numpy` already in the environment, with **0 MB of new weights** and **0.0% change on sharp images**.

#### Step 1A: Deterministic Text-Region Blur Detector (`image_blur_metrics`)
- **Why whole-image Laplacian variance is not enough:** A sharp text overlay on a blurred bokeh photo background has low whole-image Laplacian variance, while a blurred text block on a noisy background can have moderate whole-image variance.
- **Mechanism (pure NumPy/OpenCV/SciPy, `< 3 ms` per image):**
  1. Convert image to grayscale `L` and identify candidate stroke regions using local morphological gradient (`max_filter(L, 5) - min_filter(L, 5) > 20`).
  2. Compute two scale-normalized sharpness metrics **specifically over the top-20% highest-gradient patches**:
     - **Normalized Laplacian Energy (`S_lap`):** $\text{Var}(\nabla^2 I) / (\text{Var}(I) + \epsilon)$, which normalizes for contrast differences between dark-on-light and light-on-dark text.
     - **Brenner / Tenengrad Edge Width (`W_edge`):** Ratio of 1-pixel step gradient $|I(x+1,y) - I(x-1,y)|$ to 4-pixel step gradient $|I(x+4,y) - I(x-4,y)|$. On sharp text, 1-pixel gradients approach the 4-pixel step (`ratio > 0.55`); on Gaussian/defocus blur ($\sigma \ge 1.5\text{ px}$), the 1-pixel step collapses relative to the 4-pixel step (`ratio < 0.38`).
  3. **Strict Gate (`is_blurred`):** An image enters the Blur-Aware Path **only** when `(S_lap < T_lap and W_edge < T_edge)` **or** when the standard pass finds `≥ 2` raw detector boxes whose median recognizer confidence sits in the blur-characteristic band (`0.30 ≤ median_conf < 0.72`).
  4. **Control Guarantee:** All 60 Phase 14 rendered pages and sharp document fixtures score `W_edge > 0.65` and bypass the blur path 100% of the time.

#### Step 1B: Dual-Stage Image & Crop Restoration (Unsharp Mask + CLAHE + Fast Richardson-Lucy)
When `is_blurred` triggers, a single global sharpening kernel is not optimal for all blur radii ($\sigma \in [1.2, 4.0]\text{ px}$). We apply a fast two-stage restoration pipeline:
1. **Whole-Page Pre-Detection Restoration (for DBNet box recovery, `< 20 ms`):**
   - **Upscale small blurred images:** If `max(width, height) < 1200`, upscale `1.5×` with `Image.Resampling.LANCZOS` so blurred strokes span enough pixels for sharpening kernels without aliasing.
   - **CLAHE Local Contrast Enhancement (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))` on L channel in LAB space):** Restores local ink-to-background separation washed out by blur halos.
   - **Difference-of-Gaussians / Unsharp Mask (`USM`):**
     $$I_{\text{sharp}} = \text{clip}\big(I_{\text{clahe}} + \alpha (I_{\text{clahe}} - G_{\sigma}(I_{\text{clahe}})),\, 0,\, 255\big)$$
     with $\sigma = 1.5, \alpha = 1.6$ for mild/medium blur and $\sigma = 2.5, \alpha = 2.2$ for heavy blur (`W_edge < 0.28`).
2. **Per-Crop Deconvolution for the Recognizer (for `0.40 ≤ conf < 0.80` boxes, `< 8 ms` per crop):**
   - Once boxes are localized, pad each box by **20% of box height horizontally and 15% vertically**, upscale to `height = 64 px` (`1.33×` the recognizer's 48px grid), and run **5 iterations of damped Richardson-Lucy deconvolution** (pure NumPy/SciPy FFT convolution with an isotropic Gaussian PSF estimated from `W_edge`, plus a directional 1D motion-blur kernel check if horizontal vs vertical `W_edge` differs by `> 1.8×`).
   - Feed the restored crop directly to RapidOCR's recognition session (`engine.text_rec`), bypassing a second full-page detector pass.

#### Step 1C: Blur-Adapted Detector Parameters & Corroborated Grey-Zone Rescue (`0.55–0.75`)
1. **Blur-Gated Detector Settings ( addressing Failure Mode B2 & B3):**
   - Instead of creating a second heavy model in RAM, run `RapidOCR` on the restored image (or with a lightweight blur-parameter override on the existing detector):
     - Raise `Det.limit_side_len` from `736` → **`1536`** on blurred images so high-res blurred screenshots/documents are not downscaled before DBNet.
     - Lower `Det.thresh` `0.30 → 0.22` and `Det.box_thresh` `0.40 → 0.30`, and widen `Det.unclip_ratio` `1.6 → 1.9` so eroded blurred text kernels are not clipped at the first and last letters.
2. **Two-Pass Confidence Arbitration (addressing Failure Mode B1 without breaking Text-Free Controls):**
   - **Why text-free controls stay 100% empty:** On a text-free image (`tests/sample.jpg`), noise spans ("S" at `0.52`) are high-frequency texture artifacts, not blurred text; sharpening does *not* turn them into coherent multi-character words.
   - **Acceptance Rule for Blurred Images:**
     - **Rule 1 (Standard):** Any span with `conf_raw >= 0.75` or `conf_restored >= 0.75` is kept (taking the reading with the higher confidence).
     - **Rule 2 (Corroborated Grey-Zone Rescue, `0.52 ≤ conf < 0.75`):** A span scoring in `[0.52, 0.75)` on the restored pass is rescued **only if all three anti-hallucination guards hold**:
       1. `len(alnum_chars) >= 3` (blocks single-character `"S"` / `"0"` noise on `sample.jpg` and shapes fixtures),
       2. `conf_restored >= conf_raw + 0.08` **or** both raw and restored passes agree on the normalized string (`SequenceMatcher.ratio >= 0.75` with both `>= 0.55`), and
       3. The crop's internal stroke contrast after restoration exceeds the ink-to-background floor (`std(crop_L) >= 22.0`).

---

### Track 2: Orientation, 180° Upside-Down, Skew & Pre-Split Geometry Fixes

This track eliminates the remaining `−5.7 pt` rotation loss and fixes the four orientation blind spots (**R1–R4**) — notably the interaction where **rotated + blurred** images fail to rotate at all.

#### Step 2A: Fix `_looks_sideways` to Inspect Raw Detector Boxes & Handle `< 3` Spans (Failure Mode R1 & R2)
- **Root Cause Fixed:** Today, `_looks_sideways` in `src/fusion_vision_mcp/analysis.py` only counts spans that *already survived* `OCR_MIN_CONFIDENCE = 0.75` and `len(text) >= 3`, and requires `tall >= 3`. On a rotated blurred image or a rotated image with 1–2 signs/titles, sideways recognition confidence drops below `0.75`, so `_looks_sideways` sees `0–2` spans and refuses to rotate the image!
- **Fix in `ppocr_engine.py` & `analysis.py`:**
  1. Have `PPOCREngine` expose raw detector box count and aspect ratios (boxes clearing `Det.box_thresh = 0.4` *before* the `0.75` recognizer filter), or evaluate `_looks_sideways` using:
     - **Standard trigger:** `tall >= 3 and tall >= 0.6 * len(counted)` (existing rule), **OR**
     - **Sparse / Low-Conf Sideways trigger:** `(tall >= 1 and len(counted) <= 2 and all_tall)` (a 1- or 2-line rotated sign/logo where every detected box is tall, `h > 1.8w`), **OR**
     - **Raw-Box Sideways trigger (critical for rotated + blurred):** At least `3` raw detector boxes have `h > 1.6w` ( constituting `≥ 60%` of raw boxes) while the upright pass yielded fewer than 3 confident spans (`>= 0.75`).
  2. **Order-of-operations fix:** When `_looks_sideways` triggers, run `layout.reading_blocks` on the `ROTATE_90` and `ROTATE_270` images (where columns and gutters are now properly vertical!), and select the orientation that maximizes **confidence-weighted character count** $\sum \text{len}(\text{text}_i) \cdot \text{conf}_i$ (rather than unweighted character count, so 3 crisp upright words beat 4 low-confidence sideways fragments).

#### Step 2B: 180° Upside-Down Page & Box Rescue (Failure Mode R3a)
- **When it triggers (zero cost on normal upright pages):**
  - **Page-level 180° trigger:** Only when the upright pass detects `≥ 2` raw horizontal text boxes (`w > 1.5h`), **but** `< 35%` of raw boxes survive the `0.75` recognizer floor (or mean recognizer confidence across raw boxes is in the upside-down signature band `[0.35, 0.68)`).
  - When triggered, run `_read_page(app, image.transpose(Transpose.ROTATE_180))` and map boxes back via `[width - 1 - x2, height - 1 - y2, width - 1 - x1, height - 1 - y1]`. Accept the 180° reading if its confidence-weighted character count exceeds the upright pass by `≥ 25%` and at least one span scores `≥ 0.82`.
  - **Box-level 180° trigger (for a single upside-down label in an upright photo, e.g. TextVQA `#72`):** Any individual horizontal raw box with `0.40 ≤ conf < 0.75` is cropped, rotated 180°, and re-read by the recognizer; it replaces the raw reading only if the 180° confidence jumps to `≥ 0.85`.

#### Step 2C: Per-Box 90°/270° Vertical Sign Rescue on Upright Pages (Failure Mode R3b)
- On an upright page (`not _looks_sideways`), if an individual detected box has `h > 1.8w` and `conf < 0.88` (or was dropped in the `[0.35, 0.75)` raw band), crop that box with a 15% pad, rotate it `90° CW` and `90° CCW`, and run the recognizer on both. Keep the best reading if its confidence reaches `≥ 0.80`.

#### Step 2D: Quad-Angle Page Deskew & Oriented Polygon Rectification (Failure Mode R4)
- **Preserve 4-point polygons in `PPOCREngine`:** RapidOCR's `result.boxes` are 4-point oriented quadrilaterals `[tl, tr, br, bl]`. Compute each quad's baseline angle $\theta_i = \text{atan2}(y_{tr} - y_{tl}, x_{tr} - x_{tl})$.
- **Page-level Deskew (`3° ≤ |median(θ)| ≤ 35°`):**
  - If `≥ 3` boxes share a consistent tilt angle (`std(θ) < 5°` and `3° ≤ |median(θ)| ≤ 35°`), rotate the image by `-median(θ)` with `Image.Resampling.BICUBIC` (`expand=True`, fill with border median color) before running `layout.reading_blocks` and recognition, then map box coordinates back to original page coordinates via the inverse 2D rotation matrix.
- **Box-level Perspective Rectification:** For individual tilted quads on photos (`|θ_i| > 8°` where `conf < 0.88`), rectify the 4-point quad using `PIL.Image.transform(..., Image.Transform.QUAD, ...)` (with a 15% horizontal pad along the quad's long axis) before re-scoring.

---

### Track 3: Context-Aware Point-OCR & VLM Arbiter for Severely Blurred Text

Even after Track 1's classical sharpening (USM + CLAHE + Richardson-Lucy), severe blur ($\sigma \ge 3.0\text{ px}$) destroys fine intra-character strokes so that a purely bottom-up CTC recognizer (`PP-OCRv6`) can only read ~60–70% of the characters in a line. As the scoreboard notes, frontier models achieve `−20 to −30 pts` on blur because they **use language context to resolve ambiguous blurred letters**.

We can give `ocr` that exact top-down contextual capability on severely blurred regions **without violating the "never invent text on text-free images" invariant**:

#### How the Strict Detector-Gated Arbiter Preserves Anti-Hallucination Controls
> **Load-Bearing Rule:** A VLM or sequence-to-sequence recognizer (`Florence-2`, `Moondream2`, `TrOCR`, `PARSeq`) must **never** be asked to read a text-free image or propose unverified text on its own. Instead, Track 3 runs **only on bounding boxes already localized by the OCR detector** where the sharpened `PP-OCRv6` reading still sits in the ambiguous confidence band (`0.40 ≤ conf_sharp < 0.82`) or contains obvious CTC corruption (isolated 1-char fragments between words, mixed symbol/letter stutter).

#### Candidate Arbiters (Evaluated Head-to-Head in Gate 1, Non-Chinese Priority First)

| Priority | Arbiter Candidate | Origin & License | Memory & Load Cost | Mechanism on Blurred Line / Block Crops |
| :---: | :--- | :--- | :--- | :--- |
| **3A (Zero New Weights)** | **`Florence-2-large` `<OCR>` on Padded Zone/Line Crops** (already in `AppContext.florence2`) | Microsoft (US), MIT | **0 MB extra** (already resident in server stack) | `Florence-2`'s seq2seq language decoder uses BPE language priors to complete partially blurred English words when fed a **padded multi-word line/block crop** (upscaled to `≥ 384 px` wide). Guarded by requiring token-level overlap (`_ratio >= 0.45`) with the raw/sharpened `PP-OCRv6` characters so it can never hallucinate unrelated prose. |
| **3B (Zero New Weights)** | **`Moondream2` Contextual Crop Transcription** (already in `AppContext.vqa`) | Vikhyat / M87 (US), Apache-2.0 | **0 MB extra** (already resident in server stack) | Prompted on the padded blurred block crop with the partial `PP-OCRv6` high-confidence anchor words: `"Read the exact text in this cropped image line by line. Output only the verbatim text."` Filtered through the same character-overlap corroboration gate against `PP-OCRv6`'s raw reading. |
| **3C (Lightweight Non-Chinese Point OCR)** | **`PARSeq` (`doctr` recognizer) or `TrOCR-base-str` (Pad-Mask Fixed)** | Mindee (France, Apache-2.0) / Microsoft (US, MIT) | ~95 MB (`PARSeq`) / ~330 MB (`TrOCR-base`) | Permutation language modeling (`PARSeq`) iteratively refines character predictions using bidirectional character context (`GEORGLA → GEORGIA`, `BLUE GIRE → BLUE GIRL`). In Sprint R Gate 1, `docTR`'s *detector* lost boxes; running `PARSeq` or pad-fixed `TrOCR-base-str` **on PP-OCR's detected crops** isolates their strong contextual recognition head without docTR's detector loss. |
| **3D (Conditional Chinese-Model Exception, Only If 3A–3C Fall Short)** | **Blur-Gated `PP-OCRv5` Server Detector + Recognizer (`rapidocr`)** | Baidu (Apache-2.0, already supported in installed `rapidocr>=3.9`) | ~180 MB ONNX (loaded lazily only when `is_blurred` is True) | In `ppocr_det_sweep.py`, `PP-OCRv5` server models were rejected for *clean* pages because they slowed down clean photos and dropped rendered pages to 87%. Gated **exclusively** behind `is_blurred == True`, clean rendered pages (`97.3%`) and clean photos (`58.5%`) are 100% untouched while blurred images get the heavier server backbone. |

---

## 5. Pre-Registered Evaluation Protocol & VisionBench Gates

Per `.clinerules/10-visionbench-testing.md`, `feedback_no_runs_without_decision.md`, and `feedback_sprint_pacing.md`:
- All benchmark scripts live in `C:\AI\Projects\VisionBench\benchmarks\` and run on CPU (pinned to 18 cores).
- Every gate's rules and thresholds are committed before the scored run begins.
- **Stop after each gate and report to the user** before advancing to the next gate or touching `FusionVisionMCP/src/`.

### Gate 0 — Diagnostic & Latency Smoke on Blurred + Rotated Tune Sample (`ocr_blur_gate0.py`)
- **Model Assignment:** **Opus 5.5** (to design the diagnostic script, inspect failure buckets on `p11_blur`, and verify sharpness/orientation thresholds) or **Sonnet 5.5** (new session with handoff once script is written).
- **Dataset:**
  - 30 stratified images from `p11_blur` tune (and their paired clean counterparts in `p11_text` tune),
  - 20 rotated images from Phase 13 (90°, 180°, 270°, and skewed),
  - All 4 `TEXT_FREE` fixtures (`sample.jpg`, `spatial_containment.png`, `spatial_touch_separate.png`, `detect_blank_canvas.png`) + their blurred (`σ = 2.0, 3.5`) and rotated (`90°, 180°`) variants.
- **What Gate 0 Measures:**
  1. Decomposes `p11_blur` misses into: (a) boxes detected by DBNet but dropped by the `0.75` floor (`0.40 ≤ conf < 0.75`), (b) boxes missed or edge-clipped by DBNet under blur, and (c) boxes found where the recognizer misread characters.
  2. Verifies that the sharpness gate (`is_blurred`) fires on **≥ 90% of `p11_blur`** and **≤ 2% of clean rendered pages (`Phase 14`)**.
  3. Verifies CPU latency: median **≤ 5.0 s** per blurred/rotated image and **< 5 ms overhead** on sharp upright images.
  4. Verifies `TEXT_FREE` fixtures stay **100% empty** across clean, blurred, and rotated variants.
- **Stop Rule:** If Track 1 (signal restoration + blur-gated detector/confidence rules) gains `< +5.0 points` on the 30-image `p11_blur` sample, stop and inspect before running the full 534-item tune suite.

### Gate 1 — Full 534-Item OCR Tune Suite & Ablation (`ocr_blur_gate1.py`)
- **Model Assignment:** **Sonnet 5.5** (new session + handoff) to execute the run; **Opus 5.5** to review results.
- **Dataset (the exact 534-item tune split from `replacement_gate1.py`, plus Phase 13 rotation tune):**
  - `p11_blur` tune (155 blurred text images; shipped baseline **37.4%** key phrases)
  - `p11_text` tune (155 clean text images; shipped baseline **76.8%** key phrases — measures the clean-minus-blurred gap directly)
  - `phase13_rot` tune (rotated 90°/180°/270° images; shipped baseline **−5.7 pts** on 90°/270°)
  - `textvqa` tune (92 scene photos; shipped baseline **59.8%**)
  - `rendered` tune (26 DOM-grounded pages; shipped baseline **97.5%** verbatim lines)
  - `lines` tune (51 sourced real images; shipped baseline **73.0%** lines)
  - `p11_textfree` tune (51 text-free generalist images; shipped baseline **92.2%** empty)
  - `fixture_textfree` (4 text-free fixtures; shipped baseline **4 / 4** empty)
- **Ablation Ladder (run in order; stop once a stage no longer adds `≥ +1.5 pts` on `p11_blur`):**
  1. **`V1_restore_only`:** Track 1A + 1B + 1C (Sharpness gate + USM/CLAHE/Richardson-Lucy + blur-gated DBNet `thresh=0.22, box_thresh=0.30, unclip=1.9, side_len=1536` + corroborated grey-zone rescue).
  2. **`V2_restore_plus_orient`:** `V1` + Track 2 (raw-box `_looks_sideways` fix for `< 3` spans and rotated+blurred pages, 180° upside-down rescue, per-box vertical rescue, and quad deskew).
  3. **`V3_arbiter_non_chinese`:** `V2` + best Track 3 non-Chinese arbiter (`3A Florence-2 <OCR>`, `3B Moondream2`, or `3C PARSeq/TrOCR-base`) on surviving low-confidence blurred crops (`0.40 ≤ conf < 0.80`).
  4. **`V4_server_blur_fallback` (only if `V1–V3` leave `p11_blur` `< 50.0%`):** `V2` + blur-gated `PP-OCRv5` server models (`3D`).
- **Gate 1 Pass Rule (All Must Hold on Tune):**
  1. **`p11_blur` key-phrase recall:** **≥ +10.0 points** over the `37.4%` baseline (i.e., **≥ 47.4%**, targeting **52–58%** so the clean-to-blur drop narrows from `~40–52 pts` into the frontier `20–30 pt` band).
  2. **Rotation recall (`phase13_rot`):** 90°/270° loss **≤ 4.5 points** (improving or holding the `−5.7 pt` par score), and 180° upside-down loss **≤ 8.0 points**.
  3. **`fixture_textfree` empty:** **4 / 4 (100%)** on clean, blurred, and rotated variants.
  4. **`p11_textfree` empty:** **≥ 90.2%** (within `−2.0 pts` of the `92.2%` baseline).
  5. **Sharp controls (`rendered`, `p11_text`, `lines`, `textvqa`):** Zero regression `> 0.5 points` on `rendered` (`≥ 97.0%`) or `p11_text` (`≥ 76.3%`).

### Gate 2 — Held-Out Confirm Split & Phase 13 Full Confirmation (`ocr_blur_gate2.py`, Run Once)
- **Model Assignment:** **Sonnet 5.5** to run the frozen winning configuration once; **Opus 5.5** to verify.
- **Dataset:** Held-out `confirm` half of the OCR suite (including `p11_blur` confirm, `p11_text` confirm, `rendered` confirm, `p11_textfree` confirm, and Phase 13 full blur/rotation evaluation), scored with a 5,000-sample paired bootstrap (`seed = 1`).
- **Ship Rule (All Must Hold on Confirm):**
  1. **`p11_blur` confirm gain:** **≥ +8.0 points** with the 95% paired bootstrap CI lower bound **> +4.0 points**, bringing the overall Phase 13 blur drop from `−52 points` to **≤ −30 points** (✅ Par with frontier models).
  2. **Rotation confirm:** Holds **✅ Par** (`≤ −5.7 points` lost on 90°/270°, with measurable improvement on 180° and rotated+blurred items).
  3. **All negative controls and sharp controls hold:** `fixture_textfree == 4/4`, `p11_textfree` within `−2.0 pts`, `rendered >= 96.5%`, median CPU latency `≤ 5.0 s`.

## 6. Implementation Touchpoints & Permanent Regression Tests (When Gate 2 Passes)

No production files in `FusionVisionMCP` are modified until Gate 2 passes and the user gives go-ahead to ship. At that point, changes land cleanly in:

| File | Planned Production Change |
| :--- | :--- |
| `src/fusion_vision_mcp/constants.py` | Add documented, gate-pinned constants: `OCR_BLUR_EDGE_RATIO`, `OCR_BLUR_LAP_FLOOR`, `OCR_BLUR_DET_BOX_THRESHOLD`, `OCR_BLUR_UNCLIP_RATIO`, `OCR_BLUR_GREY_FLOOR`, and `OCR_DESKEW_MIN_DEG`. |
| `src/fusion_vision_mcp/ppocr_engine.py` | Preserve 4-point quads (`quad`) alongside axis-aligned `box`; expose raw pre-floor box aspect ratios/counts for orientation detection; support blur-gated crop sharpening + second-pass recognition (`readtext_blurred`) without loading a second model on sharp images. |
| `src/fusion_vision_mcp/analysis.py` | Add deterministic `is_text_blurred(image)` check; upgrade `_looks_sideways` to use raw box aspect ratios and handle `< 3` tall spans; add 180° upside-down rescue, per-box vertical rescue, and quad-angle deskew in `_read_upright`. |
| `tests/test_ocr_orientation.py` | Add permanent unit tests for: (a) 1- and 2-span rotated signs, (b) rotated + blurred pages where initial sideways confidence is `< 0.75`, (c) 180° upside-down pages and single boxes, and (d) tilted/skewed line deskewing. |
| `tests/test_ppocr_engine.py` | Add permanent unit tests for: (a) sharp pages bypassing the blur path (`0` extra OCR calls), (b) synthetic Gaussian-blurred (`σ = 2.5`) fixture recovery on `layout_two_column.png` and `layout_paragraph.png`, and (c) blurred `tests/sample.jpg` still returning `[]` (text-free negative control under blur). |
| `CLAUDE.md` & `Future_Ideas/frontier_capability_gaps.md` | Record the measured Gate 1 / Gate 2 numbers, CIs, and any rejected variants in the project's native "why"-focused voice, and update the `ocr, rotated / blurred` scoreboard row. |

## 7. Recommended Execution Sequence

1. **Step 1 — Gate 0 Diagnostic & Prototype (`Opus 5.5`, this session or next, ~35 min):**
   Build `C:\AI\Projects\VisionBench\benchmarks\ocr_blur_gate0.py`, run the 30-image `p11_blur` + 20-image rotation + 4 text-free control diagnostic, and report the exact bucket breakdown and Track 1 / Track 2 gains to the user. *(Wait for user go-ahead.)*
2. **Step 2 — Gate 1 Full Tune Ablation (`Sonnet 5.5`, new session + handoff, ~60–75 min CPU):**
   Run `V1_restore_only`, `V2_restore_plus_orient`, and `V3_arbiter_non_chinese` on the 534-item OCR tune suite. Stop and report the comparison table to the user. *(Wait for user go-ahead.)*
3. **Step 3 — Gate 2 Held-Out Confirm (`Sonnet 5.5` run / `Opus 5.5` verify, ~45 min CPU):**
   Run the single frozen winner on the confirm split and Phase 13 benchmark once. *(Wait for user go-ahead.)*
4. **Step 4 — Wire Into `FusionVisionMCP` & Add Permanent Unit Tests (`Opus 5.5` / `Sonnet 5.5`, ~30 min):**
   Port the confirmed changes into `ppocr_engine.py`, `analysis.py`, and `constants.py`, run fast unit tests (`pytest tests -q --ignore=tests/test_server.py`), verify the torch-free import invariant, and update the scoreboard.
