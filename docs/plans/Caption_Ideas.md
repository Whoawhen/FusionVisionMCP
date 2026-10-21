# Plan to Improve the Caption Tool / Function

*Compiled: 2026-10-16*
*Context: Follows the scoreboard in `Future_Ideas/frontier_capability_gaps.md#L79-L103`, the diagnostics in `Future_Ideas/caption_sprint_c4_plan.md`, and VisionBench evaluations.*

> **Status (2026-10-17):** reviewed against `Future_Ideas/caption_sprint_c4_plan.md`, which is where the work is
> tracked now. **Track 1** became C4-2b there, cap first: on the cached recheck rows all 307 dropped phrases are the
> 40-word cap (none the ` / ` join), so the word cap is stage 1 and title-first ordering stage 2. **Track 2 is done
> and did not ship** (C4-1, 2026-10-16: encode-once shipped as `cfa0d31`, but the theme line failed COCO CHAIR,
> CHAIR-s +2.4 against a +1.0 limit; `type_q` was dropped at the 20-image screen, 2 of 5). **Tracks 3 and 4** were
> already C4-2a/c and C4-3b. One correction to Track 1's guards: there is no tune subset of `caption_text_recheck`
> (its 315 images are the phase 11 scoreboard set), so phrase recall there is report-only.
>
> **Status (2026-10-19):** Track 1 stage 1 shipped as the 80-word line (C4-2b, 2026-10-18); stage 2, title-first
> ordering, gained nothing (+0.4, CI −1.4 to 2.2). Phase 11 on 2026-10-19: key text phrases 59.2 → 67.3%, keywords
> 61.1 → 62.4%. C4-7 (cleaned 160-word line with a cut marker) shipped 2026-10-20; phase 11 on 2026-10-21: keywords 63.1%,
> key text phrases 75.2%.
>
> **C4-6 (2026-10-20): failed the blind label check (75.9%, need 85%); the image-type idea is closed.**
>
> **C4-7 result (2026-10-20): shipped.** On round 20 (150 fresh text-heavy images, 30 per kind incl. 26 from the
> round 20b top-up; the user's spot-check 144 of 144 phrases right) the cleaned 160-word line with a cut marker took
> key phrases 48.1 → 60.0% (+11.9, CI 9.1 to 14.9) and `deep` 54.3 → 63.0% (+8.8, CI 6.4 to 11.2), no image worse,
> absent objects named unchanged (0 / 1). Report-only: clean at 80 +2.4, raw 160 +10.8, clean uncapped +19.3 (67.4%).
> COCO CHAIR exactly unchanged in both slots; round 11 no new absent object (its key phrases 54.9 → 65.0%). Results
> in VisionBench `benchmarks/results/caption_text_line_r20*.json`; phase 11's scoreboard figure moves at the next
> caption rerun.

---

## 1. Diagnosis of the Current Scoreboard

| Metric / Slot | Current Score | Frontier Ceiling | Gap | Latency (Budget) | Key Bottleneck Identified in Data |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`caption` (default)**<br>`Florence-2-large` + `read_text` | **60.9%** (280 confirm)<br>**63.1%** (400 Phase 11, 2026-10-21, 160-word line; 62.4% at 80, 61.1% at 40) | ~75% (**L**) | **🔻 −12 pts** | 13.1 s (≤ 15 s;<br>**1.9 s left**) | Photos (79%) & artwork (76%) are at par. Weak on **AI images (48%)**, **illustrations (51%)**, **game UI (54%)**, **charts (57%)**, **documents (61%)**, **app/web (64%)**. |
| **`caption(deep=true)`**<br>`Moondream2` long + `read_text` | **66.3%** (280 confirm) | ~75% (**L**) | **🔻 −9 pts** | 35.8 s (≤ 45 s;<br>**~9 s left**) | `C4-0d` loop stopper fixed the ~7% runaway tail (57.3 s → 40.7 s). Remaining misses are image genre/type (27/111), object/HUD detail (34/111), and concept/theme (13/111). |
| **`caption` Key Text Phrases**<br>(`caption_text_recheck.py`) | **75.2%** in `caption` (160 words, phase 11 2026-10-21; 67.3% at 80, 59.2% at 40)<br>vs **74.9%** in `ocr` | ~60–75% (**E**) | **≈ / 🔻 −1 to −16 pts** | Included in above | **48% of all text misses (307 of 640 phrases) were already read verbatim by `ocr_pages` during the `caption` call**, but discarded by `visible_text_line`'s raster order + 40-word cap (`screenshots`: 88.8% in `ocr` → 53.2% in `caption`; `charts`: 87.2% → 68.4%; `documents`: 64.7% → 52.6%). |
| **`caption(region=...)`** | **80.3%** vs 67.8% whole-image | — | **✅ Shipped** | Same | Shipped 2026-10-15 (Gate R, +12.5 pts on 528 region questions). |

### What Is Already Settled (Will Not Be Repeated)
* **Florence-2 Caption LoRA (`C4-4p`)**: Completed 2026-10-16. Tune gained +3.2%, but confirm (280 images) scored **61.5% vs 60.9% (+0.6, CI −1.3 to +2.6) and +0.0% at matched length**. Per the pre-registered rule, the LoRA route (`C4-4s`) is stopped.
* **Prose Rewriting Against OCR (`Gate B`) & `auto_verify_text`**: Both failed to improve accuracy (`auto_verify_text` +0.0 pts at 2× latency, retired 2026-10-15; `Gate B` 2,810 → 2,800 hits).
* **Moondream2 Greedy Decoding (`C4-0a`) & Loop Stopping (`C4-0d`)**: Greedy tied sampled; `C4-0d` online loop truncation shipped (`6538257`).
* **Public Adapters (`C4-3a`)**: Paper check ruled out all candidates (wrong base model or older Moondream2 revision).

---

## 2. Four-Track Improvement Plan (Ordered by ROI per CPU Hour)

### Track 1: Recover the 307 Dropped OCR Phrases in `visible_text_line` (Sprint C4-2b Expanded)
* **Why**: `caption` *already* runs `ocr_pages(app, images)` on every call, which computes `text_regions` with bounding boxes `[x1, y1, x2, y2]`, and then throws the boxes away by passing plain raster-order `page["text"]` into `visible_text_line`. On busy screenshots, charts, and documents, top-of-screen chrome (browser tabs, menu bars, status bars, headers) consumes the 40-word cap before the title, chart axis, or main heading is reached.
* **What We Will Test (on cached tune outputs — ~0 model inference cost)**:
  1. **Variant B1 — Salience/Title-First Ordering (`C4-2b`, 40-word cap unchanged)**:
     * Pass `page["text_regions"]` into `visible_text_line`.
     * Group regions into height tiers (using bounding-box height `y2 - y1` quantized into bands so 1–2 px jitter within a line doesn't break reading order), placing the tallest headings/titles first, followed by body text in reading order, up to 40 words.
     * **Cost**: **0.0 s latency, 0 MB memory, 0 extra words**, and 100% engine-independent (carries over when PP-OCRv6 is replaced). Lifts **both** default and `deep`.
  2. **Variant B2 — Adaptive Word Cap on Text-Dense Screens (40 → 60 words)**:
     * On the 38/120 tune images with >40 OCR words (`C4-0b2`), test whether a 60-word cap (with B1's title-first ordering) recovers the remaining `cap_or_join` misses on screenshots (53.2% → 88.8% ceiling) and charts (68.4% → 87.2% ceiling).
* **Pre-registered Ship Guards**:
  * Evaluated on the 120 tune split (`caption_ab_tune`) and the corresponding tune subset of `caption_text_recheck`:
    * 95% bootstrap CI on keyword recall > 0 (and key-phrase recall on `caption_text_recheck` rises toward the ≥65% target).
    * `absent_object_named` does not increase.
    * For B2 (if tested against B1): must beat B1 on keyword recall by ≥1.0 pt with 0 absent-object increase to justify the extra 20 verbatim words.

---

### Track 2: `deep` Slot Encode-Once + Micro-Probes for Genre & Theme (Sprint C4-1) — DONE, not shipped (see Status)
* **Why**: In `C4-0b`, **Image type/genre (27/111)** and **Concept/theme (13/111)** account for **36% of all remaining misses**. In Sprint C3, appending `Theme:` from Moondream2 gained **+1.3 pts (CI +0.4 to +2.5, 0 images worse)** and lifted AI images (our weakest kind at 48%), but was dropped *only* because `model.query(image, ...)` re-encoded the image from scratch (+22 s).
* `Moondream.caption` can call `encoded = self.model.encode_image(rgb_img)` once and pass `encoded` to both `caption(encoded, length="long")` and `query(encoded, prompt)`, paying only for the ~5–15 generated tokens.
* **Steps**:
  1. **Step 2.1 — 20-Image Timing & KV-Cache Equivalence Gate (~15 min CPU)**:
     * Time `encode_image`, `caption(encoded, "long")`, and `query(encoded, ...)` on 20 tune images (18 threads).
     * **Equivalence check**: Under greedy decoding (`temperature=0`), verify that `query(encoded, prompt)` *after* `caption(encoded)` produces **byte-identical** text to `query(fresh_image, prompt)` (verifying Moondream2's attention mask cleanly hides the caption's trailing KV-cache entries).
     * **Stop rule**: Stop if `query(encoded, ...)` adds > 6 s median or fails equivalence.
  2. **Step 2.2 — 120-Image Tune Evaluation (at most 2 fixed variants)**:
     * **`deep+text+theme`**: Reuses `encoded` to ask `THEME_PROMPT` (`caption_theme.py`) and appends `Theme: ...`.
     * **`deep+text+type_q`**: Reuses `encoded` to ask `"What kind of image is this? Answer in a few words."` and appends `Image type: ...`.
       * *Fast screen*: Check kind agreement on the 20 timing images first; drop `type_q` immediately if <80%. On the 120 tune split, it must clear the **≥90% kind-agreement floor** (where CLIP's 83% failed).
* **Pre-registered Ship Guards**:
  * Total `deep` median ≤ 45 s; keyword recall 95% CI > 0; 0 images worse; ≥90% kind agreement for `type_q`; COCO CHAIR not worse.

---

### Track 3: Default Slot Beam Reduction & Dense-Region Supplement (Sprint C4-2a & C4-2c)
* **Why**: Default `caption` is at 13.1 s against a 15 s limit (1.9 s headroom). Meanwhile, the single largest miss category in `C4-0b` is **Object or scene detail (34/111)**, whereas Florence-2's own `<DENSE_REGION_CAPTION>` head gets **74% box recall** (⭐ beating frontier models).
* **Steps**:
  1. **Step 3.1 — `num_beams=1` for Caption Only (`C4-2a`)**:
     * In `Florence2._run`, test `num_beams=1` on `<MORE_DETAILED_CAPTION>` while keeping `num_beams=3` for all structured tasks (`<OD>`, `<DENSE_REGION_CAPTION>`, `<OCR_WITH_REGION>`).
     * **20-image smoke**: If `num_beams=1` saves < 1.0 s median, stop here.
     * **120-image tune**: Adopt `num_beams=1` if recall is within the tune CI, repeated 8-gram loops ≤ 2/120, `absent_object_named` does not rise, and COCO CHAIR is not worse.
  2. **Step 3.2 — Deduplicated `<DENSE_REGION_CAPTION>` Supplement (`C4-2c`, conditional on ≥4 s latency headroom)**:
     * Append `Also visible: ...` (up to 12 words of `<DENSE_REGION_CAPTION>` labels not already present in the caption prose or `Visible text`).
     * **Mandatory Hallucination Gate**: Because Florence-2 region heads can over-box, this variant *must* pass the 300-image COCO CHAIR check (`caption_hallucination.py`) and `absent_object_named` guard before confirm.

---

### Track 4: Stronger Local Model Swap for `deep` — Gemma 3 4B QAT 4-bit via `llama.cpp` (Sprint C4-3b)
* **Why**: With the Florence-2 LoRA (`C4-4p`) ruled out, **Gemma 3 4B in 4-bit GGUF (`google/gemma-3-4b-it-qat-q4_0-gguf`)** is the single remaining non-Chinese model upgrade that can close both the `caption(deep=true)` (−9 pts) and `query_image` (−9 to −13 pts) semantic gaps on CPU.
* **Gate 0 is already passed**: Official Google QAT GGUF (3.16 GB + 0.85 GB mmproj) and `llama-cpp-python` 0.3.36 Windows CPU wheel (`MTMDChatHandler`) / `llama.cpp` `win-cpu-x64` binaries are verified.
* **Steps**:
  1. **Gate 1 — 20-Image CPU Timing & Memory Gate (~20 min)**:
     * Measure median latency (18 threads) for caption + `read_text` and resident RAM.
     * **Hard Stop**: If median > 45 s, stop C4-3b immediately.
  2. **Gate 2 — 120-Image Tune Split (only if Gate 1 passes)**:
     * Compare `gemma3_4b_q4+text` against `moondream_long+text` at full and matched length, plus `absent_object_named` and COCO CHAIR.

---

## 3. Confirmation & Permanent Verification Protocol

To prevent overfitting the 280-image confirm split (which has now adjudicated C1, C2, C3, Option C, and C4-4p):
1. **Tune Selection**: Individual tracks are filtered strictly on the **120-image tune split** (and 20-image timing gates).
2. **Bundled Confirm + Fresh Holdout**: Winning changes are verified on the **280-image confirm split** and confirmed **once** on the untouched **Round 11 holdout (`benchmarks/holdout_round11/`, 235 images, frozen at `fa06afc`)**.
3. **Permanent Tests & Touchpoints**:
   * `src/fusion_vision_mcp/analysis.py` (`visible_text_line`) + unit tests in `tests/test_analysis_helpers.py`
   * `src/fusion_vision_mcp/moondream.py` (`encode_image` reuse) + unit/equivalence tests in `tests/test_moondream.py`
   * `src/fusion_vision_mcp/florence2.py` (per-task beam config if 3.1 ships)
   * Torch-free import invariant check (`assert 'torch' not in sys.modules` on package import).
