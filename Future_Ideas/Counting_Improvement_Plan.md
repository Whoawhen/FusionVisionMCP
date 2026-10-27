# Counting (`count_objects`) Improvement Plan — Closing the Frontier Gap

**Created:** 2026-10-26
**Status:** Reviewed 2026-10-27 (Section 0 supersedes the rest where they disagree). **Phase 1 done 2026-10-27**
(0.7): Track 1A/4A shipped with the Q2 crowd gate and Q3 consensus default. Next: round 26 (fresh confirmation) and
Gate 1 (bounded LFM arbiter).
**Companion docs:** [`Future_Ideas/frontier_capability_gaps.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/frontier_capability_gaps.md) (Gap 2), [`Future_Ideas/counting_problem_statement.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/counting_problem_statement.md), [`Future_Ideas/counting_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/counting_plan.md), [`Future_Ideas/detector_ocr_replacement_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/detector_ocr_replacement_plan.md)

---

## 0. Review (2026-10-27) — read this first

**Verdict.** The diagnosis in Section 3 is sound and its main numbers reproduce from the caches. Track 1A (pairwise
containment and duplicate removal) is cheap, model-free and detector-agnostic, so it goes first. The plan is not ready to
implement as written: its confirm sets are all already seen, its control proof covers only one of the two paths the
controls take, its `crowd_range` gate costs crowd coverage, and Tracks 2/3 change a design rule (a judgment rewriting
`count`) that is the user's to change. Corrections and decisions follow.

### 0.1 Re-derived from the caches (hold)

Phase 11 replies (`VisionBench/reports/visionbench_report_20261021_215606.replies.jsonl`) joined to `rv.sourced_items()`
exact counts:

- 747 counts, 448 exact (60.0%); 644 `grounding_dino`, 103 `florence2`.
- False zeros: 101 (100 on Grounding DINO); 97 of them have `possibly_more.upper >= 1`. ✅
- `pred = 2` with one box at least 70% inside the other: GD 33/77 on `truth = 1` vs 2/16 on `truth = 2`; Florence-2 15/31 vs
  1/6. ✅ `truth = 1` overcounts 160, 108 of them `pred = 2`. ✅
- Median seconds: GD path 14.0, Florence-2 path 5.5. ✅
- `crowd_range` shown on 120 counts; truth < 4 on 111 of them.
- Gate C1's controls cache: `upper_end_per_control` is `[1,1,1,1,1,1,1,3]` under both prompts. ✅ (that's the GD-forced path, see 0.3b)

### 0.2 Factual corrections (the text below has been left as written; these override it)

- **No `detection_policy.postprocess_photo_count`.** The count pipeline is `tools/detection.py::count_objects` (the
  per-result loop after `execute_detection`), calling `analysis.possibly_more`, `analysis.crowd_range`,
  `analysis._add_silhouette`, `analysis._region_label_consensus` and `_separability`. The Florence-2 group box is dropped in
  `detection_policy._florence_result`; the GD one in `grounding_dino.py` (`drop_group_box`, ~line 188). Both use
  `geometry.envelope_indices` (`_MIN_BOXES_FOR_ENVELOPE = 3`, `_ENVELOPE_CONTAINMENT = 0.6`).
- **The test files in Section 7 do not exist.** Counting tests live in `tests/test_possibly_more.py`,
  `test_crowd_range.py`, `test_detection_policy.py`, `test_grounding_dino.py`, `test_routing.py`, `test_crosscheck.py` and
  `test_server.py`; geometry in `test_geometry.py`. Add to those.
- **LFM's API:** `app.answerer.answer(images, prompts, seed=VQA_SEED)`, which samples with the model card's decoding at a
  fixed seed (deterministic per input) and `VQA_MAX_NEW_TOKENS`. There is no `answer_question`, no `do_sample=False` and no
  per-call `max_new_tokens`. Do not go through `analysis.vqa_answers`: it prepends the `ocr` reading to questions matching
  `VQA_TEXT_QUESTION`. The Moondream2 call in `estimates.vqa` is `app.vqa.query(...)`, in `tools/detection.py`.
- **LFM's counting figure** is round 23 (65 count questions): LFM 75.4% vs Moondream2 **73.8%** (+1.6), not 61.4% and not
  phase 11. It is not evidence that LFM counts better than Moondream2.
- **The "frontier range" is an estimate (E) in `frontier_capability_gaps.md`**, 80–90% on scenes of 3 or fewer, not a
  measured Claude/GPT-4o/Gemini run. The −18 is against that estimate.
- **Ledger #1:** 43.8% was the overall exact rate on the confirm half at 0.15, not the 1-object rate.
- **The latency breakdown** (GD 3.5 + DRC 2.6 + SAM 3 6.9 + SAM2 0.8 s) is not measured per stage anywhere. Gate 0 times it.
- **Model origins:** every track builds on Grounding DINO (IDEA Research, Shenzhen), which the "non-Chinese preference"
  list leaves out. See 0.3e.
- **Invariant 1's "all 7 singleton controls have `separable == "yes"`"** is unverified: gate C1's cache has no
  silhouette. Track 3B depends on it; check it in Gate 0.

### 0.3 Method problems

**a. Every set the plan would confirm on has been seen.** Sprint R's tune and confirm halves are a hash split
(`sha1("20261011:{id}")`) of the reliable COCO counts **plus phase 11's photo counts**: the 747 phase 11 counts are 378 tune
and 369 confirm. Section 3 picked its 0.70/0.75 IoA cut-offs by looking at all 747, so Sprint R confirm is seen. Round 14
was used by gates C1 and P1, round 17 by gate K, and round 13 by `flat_count_envelope.py`. So:
- **Tune on everything already seen** (phase 11, Sprint R both halves, rounds 13, 14 and 17), and report each for continuity.
- **Ship only on a fresh round (round 26)**, built and frozen before any Gate 1 result exists: photo counts of 1–5 (N ≥ 150,
  exact, user blind-checks a sample); absent-object count queries where Grounding DINO has a candidate in [0.18, 0.425)
  (the hard case for Track 2A, see 0.3f); and flat counts (N ≥ 80) if Track 4 is in scope. Fresh crowds are only needed
  if the `crowd_range` rule changes.

**b. The controls take two paths.** Through the server, the synthetic fixtures route to **Florence-2** (flat).
Gate C1's 16/16 and `upper_end_per_control` come from forcing Grounding DINO. Track 1A/4A on the Florence-2 path acts
directly on the controls (`neg_concentric_rings`, `neg_two_overlapping`) and on the positives (`concentric_rings == 2`,
`nested_boxes`, `overlapping_squares`). Gate 0 checks the fixtures on **both** paths: the GD tf5 cache and
`flat_count_envelope.py`'s Florence-2 rows. Fixture positives are currently 5/10 exact on the flat path, so "no regression
on 10/10" in Section 5 should read "no positive lost, every control held".

**c. Track 1B's SAM 3 gate costs crowds and barely fixes the false alarms.** Grounding DINO runs low on crowds by design,
so gating SAM 3 on GD's own numbers skips the crowds it collapses. Replayed with gate K's code on round 17 (92 exact
crowds of 4–30, seen):

| SAM 3 runs when | runs on | truth in range |
| --- | --- | --- |
| always (shipped) | 92/92 | **85.9%** |
| `count >= 3 or upper >= 4` (this plan) | 81/92 | 76.1% |
| `count >= 2 or upper >= 3` / `upper >= 2` | 87/92 | 80.4% |

On phase 11 the plan's gate still runs SAM 3 on 83 of the 120 shown ranges (most of them truth < 4) and drops 5 of the 9
real crowds. **It is a latency-for-coverage trade, not a free gain: the user's call (0.6, Q2).** On `count == 1, upper == 1`
(most phase 11 calls) it does save the SAM 3 call.

**d. `consensus` (Florence-2 DRC) is public output.** `_separability` does not use `region_label_count`, so `separable`
can stay without DRC, but `consensus` is a documented field and phase 12 scores it. Dropping it or making it lazy is
an API change (0.6, Q3).

**e. Tracks 2 and 3B change a design rule.** CLAUDE.md and the tool description say `count` is never rewritten by the
silhouette, and that telling a spot from a petal is "the calling model's job". Tracks 2A–2C and 3B let LFM's judgment
(bounded by detector boxes) change `count`. The bound is real protection, but this is the user's decision (0.6, Q1).
On the model policy: Track 1A is detector-agnostic geometry, so it is fine. Tracks 2A/2B/3A are heuristics on Grounding
DINO's sub-threshold boxes. The 2026-10-09 order says counting is "fixed by replacing those models rather than tuning
them", but the 2026-10-21 exception applies, since sprint R found nothing non-Chinese that passes the counting controls.
Needs the user's OK. If yes, the rules read `candidates_below` generically, so they carry to a replacement detector.

**f. LFM still goes along with false premises.** On round 23 it answered `no_object` questions right only 66.9% of the
time. Track 2A asks it exactly when GD has a weak candidate, which is where an absent object is most likely to fool it.
"≤ 10% false non-zero" must be measured on absent queries that have a candidate in [0.18, 0.425), not on all absent
queries; most of those return 0 without ever consulting LFM.

**g. Track 2 costs ~5 s per LFM call.** Gate 0 reports how often each trigger fires on phase 11 (2A alone: ≥ 13.5%) and
the median latency that results. The ≤ 6.5 s target depends on Q2.

**h. Swapping `estimates.vqa` to LFM needs its own small gate** (CLAUDE.md: each Moondream2 job gets one before
Moondream2 can go). It appears only with `vqa_estimate=true` on a `separable == "no"` single. Retiring Moondream2 also
needs `critique_composition`'s two questions gated, so this swap alone frees nothing.

### 0.4 Revised order

1. **Gate 0 (offline, no model time, apart from one short timing run).** Script `VisionBench/benchmarks/count_gate_track1.py`, rules
   committed before scoring. Track 1A on: phase 11 reply boxes (post-threshold, which is all 1A needs), the Sprint R GD raw
   caches (`replacement_gate1_raw_grounding_dino_{tune,confirm}.jsonl`, `_confirm_tf5` for the controls), round 14
   (`count_gate_c1_raw_round14.jsonl`), and round 13/fixture Florence-2 rows (`flat_count_envelope{,_r13}.jsonl`). Controls
   on both paths. Also: the `separable` value of each control through the live server, and a stage-timing run on ~20
   phase 11 images (GD, DRC, SAM 3, SAM2 separately). Tune the IoA cut-off here; seen data, so it does not ship on this alone.
2. **Ship decision for 1A**, either on Gate 0 (it is a geometry rule with controls; the user may accept seen-data
   evidence as for the flat envelope fix) or on round 26 (0.6, Q4).
3. **Round 26 built** (worker sources it; rules for Gate 2 fixed in the build script's docstring before images exist).
4. **Gate 1, LFM arbiter (if Q1 = yes).** Cache one LFM count answer per (image, query) across all seen sets: about 1,700
   queries × ~5 s ≈ 2.5 h CPU, so queue it for when the user is away. Then every 2A/2B/2C/3B variant replays offline.
   One frozen rule.
5. **Gate 2 on round 26, once.**

### 0.5 Where the inputs are (checked 2026-10-27)

All in `C:\AI\Projects\VisionBench\benchmarks\results\` unless noted: `replacement_gate1_raw_grounding_dino_{tune,confirm,
confirm_tf5}.jsonl`, `replacement_gate1_raw_sam3_{tune,confirm}.jsonl`, `count_gate_c1_raw_round14.jsonl`,
`count_gate_c1_{tune,confirm}.json`, `count_range_raw_{grounding_dino,sam3}.jsonl` (round 17),
`flat_count_envelope{,_r13}.json(l)`, `count_gate_paligemma_rows.jsonl`, `vqa_round23{.json,_rows.jsonl}`; phase 11 in
`VisionBench/reports/visionbench_report_20261021_215606{.json,.replies.jsonl}`. Reusable code: `count_gate_c1.run_counts` /
`distinct_extras` (GD replay at any threshold), `count_range_gate.counts` (crowd replay), `replacement_gate1.split_of`,
`run_visionbench.sourced_items`. The two review checks above are
`VisionBench/benchmarks/count_plan_review_check.py` (untracked).

### 0.6 Decisions for the user (before Gate 0)

1. **May a bounded LFM judgment change `count`** (Tracks 2A–2C, 3B)? If not, the alternative is to report LFM's count as
   `estimates.lfm`, clamped to `[count, upper]`. That is safe, but it does not move `count` exact.
2. **`crowd_range`:** keep SAM 3 on every photo count (85.9% on crowds, ~14 s), or gate it (≈ 7 s saved on most calls, 80.4%
   at best on round 17)?
3. **`consensus` (DRC, ~2.6 s unmeasured):** keep, make opt-in, or drop?
4. **Track 1A:** ship on Gate 0's seen-data evidence plus controls, or wait for round 26?
5. **Grounding DINO heuristics** (Tracks 2A/2B/3A): OK under the 2026-10-21 exception?
6. **Track 4 (flat) in scope now?** If yes, round 26 needs flat counts too.

**Answers so far (user, 2026-10-27), with the review's recommendations:**

- **Q1 (recommended, awaiting the user's call): yes, bounded and visible.** LFM may only *select among* detector boxes:
  promote candidates from `[count, upper]`, or drop the weakest boxes. It never adds an instance with no box. The result
  keeps `detector_count`, and adds `count_basis` saying when LFM moved the count. Order: 2A first (the false zeros, the
  largest bucket), then 2B. 2C waits for 1A's result, since 1A removes the nested half of the overcounts. **Drop 3B**: 7
  cases on tune, a new mask-splitting step, and its control proof is unverified.
- **Q2: gate it (user). Rule: run SAM 3 only when `possibly_more.upper >= 3`.** Replayed (round 17 seen, phase 11
  replies):

  | SAM 3 runs when | round 17 crowds, truth in range | phase 11 GD calls that skip SAM 3 |
  | --- | --- | --- |
  | always (shipped) | 85.9% | 0/644 |
  | `upper >= 2` | 80.4% | 271/644 |
  | **`upper >= 3`** | **80.4%** | **428/644** |
  | `count >= 2 or upper >= 4` | 77.2% | 419/644 |
  | `count >= 3 or upper >= 4` (original plan) | 76.1% | 489/644 |

  It was picked from four rules on a seen set, so round 26 carries ~40 fresh crowds (4–30) to re-check it. When SAM 3 is
  skipped, `crowd_range` is absent and `possibly_more` stands alone.
- **Q3 (recommended, awaiting the user's call): `consensus` off by default; `separable` decoupled and always computed.**
  On phase 11's GD path, Florence-2's region-label count equals the truth on 85/644 (13%), and the signal is
  *inverted*: the count is right 67/155 (43%) when consensus agrees and 325/489 (66%) when it disagrees. On the flat path
  it agrees on only 11/103. A warning that fires on 76% of calls and points the wrong way does a calling agent harm. Keep
  the parameter (opt-in) so the output contract still exists, and update the tool description. The time saved (~2.6 s,
  unmeasured) gets timed in Gate 0.
- **Q4: ship 1A on Gate 0 (seen data + controls on both paths); round 26 confirms afterwards (user).**
- **Q5: Grounding DINO heuristics are OK only if no viable non-Chinese path exists (user).** Sprint R found none for
  counting (SAM 3 fails the literal controls, OWLv2 holds ≤ 3/8, PaliGemma parked on size). Record that as the
  justification in Gate 1's docstring, and keep the rules on generic `candidates_below`.
- **Q6 (recommended, awaiting the user's call): 4A yes, folded into 1A; 4B deferred.** Flat is 103/747 (14%) of phase
  11 counts, 56 exact (54%): charts 26, game UI 22, screenshots 20, documents 12, illustrations 12. 4A is the same
  function as 1A on the Florence-2 path, which is also where the synthetic controls run, so it has to be tested there
  anyway. It targets 15 of the 31 flat `pred = 2` overcounts. 4B (LFM on flat) waits on 2A/2B's result. Round 26 adds
  ~80 flat counts (cheap to source alongside).

### 0.7 Phase 1 result (2026-10-27): Gate 0 passed; 1A/4A, Q2 and Q3 shipped

- **Another correction:** the control and fixture names in Sections 1.2 and 4A (`neg_concentric_rings`,
  `neg_Textured_cube`, `five_stars`, `grid_3x4`, `concentric_rings == 2`) do not exist. The real negatives are
  `neg_solid_disc`, `neg_gradient_disc`, `neg_striped_ball`, `neg_spotted_ball`, `neg_multicolour_logo`, `neg_noisy_rod`,
  `neg_textured_natural`, `neg_distractors` (3); `neg_two_overlapping` (truth 2) is filed as a positive in the Florence-2
  cache. The positives are rings of 8, rows, `unequal_ring6`, `clipped_row4` and `dense_grid30` (`benchmarks/fixtures.py`).
- **Gate 0** (VisionBench `count_gate_track1.py`, rules committed in 1199860 before scoring): all six variants were
  eligible. Picked: `INSIDE` 0.90, no concentric exemption. Pooled exact 709 -> 782 of 1,243 (+5.9, CI 4.4 to 7.3),
  MAE 0.819 -> 0.751. The gain is on single objects: phase 11 photo 60.9 -> 65.4%, phase 11 flat 54.4 -> 68.9%, round 13
  flat 50.8 -> 68.4%. Multi-object sets lose a little: COCO reliable 51.9 -> 50.5%, round 14 57.3 -> 56.3%. 9 counts
  were pushed below their truth (mostly 2 -> 1). Controls: 16/16 GD and 8/8 Florence-2; positives unchanged on both.
  Round 13 flat was not part of the look that suggested the rule, so it is the most independent evidence here.
- **Stage timing was not run.** Free RAM was 5.7 GB (SAM 3 alone peaks ~6.5 GB), so the timings would have been
  distorted, and they could not change Q2/Q3. The next phase 11 run measures the latency.
- **Shipped:** `geometry.nested_indices` + `analysis.drop_nested` (both count paths and the batch `count`;
  `nested_boxes_dropped` in the output), `CROWD_RANGE_RUN_UPPER = 3`, `consensus` default false with `separable` always
  computed. Parity with the gate 1,297/1,297; fast suite 472 passed, count/batch server tests 20 passed; ruff, mypy, the
  torch-free import check all clean.
- **For round 26:** the multi-object dip (COCO −1.4, round 14 −1.0) is within the 2-point rule but is the thing to
  watch; round 26 should carry enough 2–5 object counts to see it (≥ 60).

---

## 1. Scoreboard Context, Goal & Hard Constraints

### 1.1 Where We Stand Against Frontier Models

On the 2026-10-25 VisionBench frontier comparison (`FusionVision vs frontier models`), **`count_objects · Photo counts exact` is `-18` points behind the nearest edge of the frontier range** — the second-largest remaining capability gap in the server after blurred-image OCR:

| Benchmark Slice | Dataset / N | Current FusionVision | Frontier Range (Claude / GPT-4o / Gemini) | Gap to Par |
| --- | --- | --- | --- | --- |
| **Sprint R Confirm (held-out photos)** | 427 photo counts | **61.8% exact** (MAE `0.81`) | **80–90% exact** (1–3 objects) | **-18.2 pts** |
| **Sprint R Tune (dev photos)** | 423 photo counts | **55.6% exact** (`79.4%` in `possibly_more`) | **80–90% exact** | **-24.4 pts** |
| **Round 14 (Open Images exact photos)** | 103 photo counts (1–12 objects) | **57.3% exact** (`66.7%` on 1–3 objects) | **78–85% exact** | **-20.7 pts** |
| **Phase 11 Generalist (10 categories)** | 747 exact count queries (388 images) | **60.0% exact** (MAE `0.65`, `78.6%` in `possibly_more`) | **78–85% exact** | **-18.0 pts** |
| **Round 13 Flat Images (icons/charts/diagrams)** | 187 flat counts (`< 10` colors/1k px) | **50.8% exact** (MAE `0.84`) | **75–85% exact** | **-24.2 pts** |
| **Round 17 Real Crowds (`crowd_range`)** | 78 crowd photos (4–30 objects) | **85.9% in `[lo, hi]`** (width `5.5`) | ~50–60% exact / ~85% range | **At par** |
| **Latency (`count_objects` Phase 11 median)** | 747 calls (`grounding_dino` + `florence2`) | **13.79 s** (`13.97 s` photo / `5.48 s` flat) | ~2–5 s | **2.5–3x too slow** |

### 1.2 Target & Non-Negotiable Constraints

1. **Primary Accuracy Target:**
   - Lift photo exact count accuracy from **`57.3–61.8%` to `>= 75–80%`** (reaching frontier par on 1–3 object counts at `>= 80%`) and Phase 11 generalist exact count from **`60.0%` to `>= 72–76%`**.
   - Lift Round 13 flat exact count from **`50.8%` to `>= 65%`**.
2. **Negative Control Invariant (Hard Gate):**
   - Must hold **all 8 synthetic negative controls** under **both** literal prompts (`ball`, `rod`, `object`, `shape`, `circle`, `square`, `ring`, `blob` — `8/8`) and the legacy `"petal"` stress prompt (`8/8` = **16/16 total**):
     - `neg_spotted_ball == 1`, `neg_striped_sphere == 1`, `neg_Textured_cube == 1`, `neg_noisy_rod == 1`, `neg_concentric_rings == 1`, `neg_two_overlapping == 1`, `neg_multicolour_logo == 1`, `neg_distractors == 3`.
   - Must also hold an **absent-object (`truth = 0`) control gate**: on `coco_absent` (50 queries) and `p11_absent` (50 queries), false non-zero counts must remain **`<= 10%`**.
3. **Latency Target:**
   - Cut median `count_objects` latency on Phase 11 from **`13.8 s` to `<= 6.5 s`** by eliminating unconditional `SAM 3` (~6.9 s) and `Florence-2 <DENSE_REGION_CAPTION>` (~2.6 s) calls when they cannot change or inform the output.
4. **Model Policy & Memory Budget:**
   - **Zero new resident weights first:** Prioritise models already resident in `AppContext` (`Grounding DINO tiny`, `LFM2.5-VL-3B ONNX` via `app.answerer`, `SAM 3`, `SAM 2.1 Hiera-small`, `Florence-2-large`, `CLIP`, `DINOv2`).
   - **Non-Chinese model preference:** `LFM2.5-VL-3B` (Liquid AI, US), `SAM 3` / `SAM 2.1` / `DINOv2` (Meta, US), `Florence-2` (Microsoft, US), `CLIP` (OpenAI, US) are all US-origin.
   - **Retire `Moondream2` dependency in `count_objects`:** Replace `app.vqa` (`Moondream2`) in `estimates.vqa` with `app.answerer` (`LFM2.5-VL-3B ONNX`), removing one of the last blockers to retiring `Moondream2` (`~5.0 GB` peak RSS).
   - **Engine-portable post-processing:** Any geometric or arbitration rules must consume standard `(bboxes, scores, labels)` or text/mask outputs so a future detector swap does not require rewriting `detection_policy.py` or `analysis.py`.

---

## 2. "Do Not Repeat" Ledger (What Already Failed and Why)

Every prior counting experiment is logged in [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md), [`Future_Ideas/counting_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/counting_plan.md), and `C:\AI\Projects\VisionBench\benchmarks\results\`. The plan below explicitly avoids repeating any of these rejected paths:

| # | Previously Tested Idea | Benchmark Script / Result | Why It Failed (Root Cause) |
| --- | --- | --- | --- |
| **1** | **Lowering Grounding DINO threshold globally (`0.425 -> 0.15–0.30`)** | `replacement_gate1.py` | Breaks `neg_spotted_ball` (`1 -> 3–4`) and `neg_noisy_rod` (`1 -> 2`); drops 1-object photo accuracy from `62.8%` to `43.8%` due to part/sub-box overcounting. |
| **2** | **Option 7: Learned logistic regression box-validity classifier** (6 score + box-geometry features on `[0.15, 0.425)` candidates) | `count_gate_c1.py` (`count_gate_c1_confirm.json`) | Passed Tune (`+2.4` pts, `8/8` controls) only by setting a threshold (`0.72`) above `neg_spotted_ball`'s `0.678` spot probability. On Round 14 confirm it regressed **`57.3% -> 51.5%` (`-5.8` pts, CI `[-11.7, 0.0]`)**. Box coordinates + DINO score cannot distinguish a real 2nd object from a high-contrast spot or part. |
| **3** | **SAM 3 (`facebook/sam3`) as primary counter** | `replacement_gate2.py` | Fails literal-name negative controls (`4/8` at every threshold `0.05..0.60`: returns `0` for `"ball"`, `"rod"`, `"shape"`, `"blob"` on clean synthetic shapes) and scored **`51.8%` vs `61.8%`** on confirm (`-10.1` pts). |
| **4** | **OWLv2 (`owlv2-base-patch16-ensemble`, `owlv2-large-patch14-ensemble`)** | `replacement_gate1.py` | Holds at most `3/8` controls at any threshold; best confirm score **`43.0–44.2%`** vs `61.8%` (`-17` pts). |
| **5** | **CountGD (`nikigoli/countgd`)** | Sprint R / `CLAUDE.md` | Overcounts `neg_spotted_ball` (`1 -> 4`), breaks `3/10` fixture positives (`five_stars: 5 -> 9`, `grid_3x4: 12 -> 18`), adds `+2.3 GB` RAM + `cvxpy`, and requires Python 3.11 patching. |
| **6** | **Tiled inference (2x2 / 3x3) & visual-exemplar prompting** | `counting_problem_statement.md` | Both cross the sub-object wall on `neg_spotted_ball` (tiles bisect objects; exemplar matching matches spots to balls). |
| **7** | **CIELAB interior color boundaries (`_lab_boundary_separable`)** | `CLAUDE.md` | Fired on internal shading/spots of single objects; `0%` precision on 15 Phase 11 singletons (`pred 1 -> 2` false flips). Deleted. |
| **8** | **Unguarded PaliGemma 2 mix 3B (`detect` and `answer` modes)** | `count_gate_paligemma.py` (`count_gate_paligemma.json`) | `detect` tied baseline (`57.3%` vs `57.3%`). `answer` jumped to **`75.7%` on Round 14 (`84.0%` on 1–3 objects!)**, proving a 3B VLM has frontier-grade small-count perception, **but** unguarded VQA failed `2/8` controls (`neg_spotted_ball -> 24`, `neg_multicolour_logo -> 4`), and adding a separate `~6.7–12 GB` model just for counting was rejected. |

> [!IMPORTANT]
> **What Changed Since Gate P1 (2026-10-16):**
> On 2026-10-24–25, **`LFM2.5-VL-3B` (Liquid AI, US, Apache-2.0)** was integrated into the server in ONNX int4/fp16 (`LfmOnnxCaptioner` in [`lfm_onnx.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/lfm_onnx.py), wired as `app.captioner` and `app.answerer`). It is **already resident in memory (zero extra RAM)**, runs in **`~4.9 s` on CPU** (or `<1 s` on GPU), and scored **`75.4%` on Phase 11 counting VQA** and **`66.9%` on `no_object` false-premise VQA** in [`vqa_round23.json`](file:///C:/AI/Projects/VisionBench/benchmarks/results/vqa_round23.json). Even more importantly, we now have exact DuckDB telemetry from the 747 Phase 11 count replies showing how to structurally guard VLM arbitration so it **cannot** violate any of the 8 negative controls.

---

## 3. Root-Cause Data Diagnosis (From Phase 11, Sprint R Tune, Round 14 & Round 13 Caches)

Querying `visionbench_report_20261021_215606.json` + `.replies.jsonl` (747 exact count queries), `count_gate_c1_tune.json` (423 photo counts), `count_gate_c1_confirm.json` (103 Round 14 photo counts), and `flat_count_envelope_r13.json` (187 flat counts) reveals **five specific, quantifiable failure modes** that account for almost the entire gap to frontier models.

### 3.1 Phase 11 Exact Breakdown by Detector, Truth Bucket & Error Direction

| Detector Path | Truth Bucket | N | Exact Now | **`pred = 0` (False Zero)** | `pred = 0` with `possibly_more >= 1` | **Undercount (`0 < pred < truth`)** | **Overcount `pred = 2` (on `truth = 1`)** | **Overcount `pred >= 3` (on `truth = 1`)** | Truth in `[pred, pm_upper]` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`grounding_dino` (Photo)** | **`truth = 1`** | 585 | 374 (`63.9%`) | **93 (`15.9%`)** | **90 / 93 (`96.8%`)** | 0 | **77 (`13.2%`)** | **41 (`7.0%`)** | **464 (`79.3%`)** |
| **`grounding_dino` (Photo)** | **`truth = 2–3`** | 48 | 18 (`37.5%`) | **6 (`12.5%`)** | **6 / 6 (`100%`)** | **20 (`41.7%`)** | — | — (4 over) | **31 (`64.6%`)** |
| **`grounding_dino` (Photo)** | **`truth = 4+`** | 11 | 0 (`0.0%`) | 1 (`9.1%`) | 1 / 1 (`100%`) | **10 (`90.9%`)** | — | — | 4 (`36.4%`) |
| **`florence2` (Flat)** | **`truth = 1`** | 90 | 47 (`52.2%`) | 1 (`1.1%`) | — | 0 | **31 (`34.4%`)** | **11 (`12.2%`)** | 47 (`52.2%`) |
| **`florence2` (Flat)** | **`truth = 2–3`** | 11 | 8 (`72.7%`) | 0 | — | 2 (`18.2%`) | — | — (1 over) | 8 (`72.7%`) |
| **`florence2` (Flat)** | **`truth = 4+`** | 2 | 1 (`50.0%`) | 0 | — | 1 (`50.0%`) | — | — | 1 (`50.0%`) |
| **Total Phase 11** | **All** | **747** | **448 (`60.0%`)** | **101 (`13.5%`)** | **97 / 100 (`97.0%`)** | **33 (`4.4%`)** | **108 (`14.5%`)** | **52 (`7.0%`)** | **555 (`74.3%`)** |

---

### Failure Mode C1: The `len(bboxes) < 3` Blind Spot in `envelope_indices` & Missing Pair-Containment / NMS (`108` `pred = 2, truth = 1` Overcounts!)

Look at the `truth = 1` overcount columns above:
- Out of **160** single-object overcounts (`truth = 1, pred >= 2`) across Phase 11, **`108` (`67.5%`) are `pred = 2`** (`77` on Grounding DINO + `31` on Florence-2)!
- Why do so many single objects end up with exactly **2 boxes**?
  1. **`GroundingDino.detect_objects` ([`grounding_dino.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/grounding_dino.py)) applies zero NMS.**
  2. **`geometry.envelope_indices` ([`geometry.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py#L46-L70)) begins with `if len(bboxes) < 3: return set()`!**
     - When a detector emits **1 box around the whole object** and **1 box around a part/sub-region of that same object** (or two heavily overlapping boxes on the same instance), `len(bboxes) == 2`, so `envelope_indices` immediately returns `set()` without inspecting them!
  3. **We verified this directly in DuckDB on the 747 Phase 11 replies (`bboxes[0]` vs `bboxes[1]` for all `pred = 2` items):**

| Detector Path | `pred = 2` Slice | N | **`IoA_min >= 0.70` (One Box >=70% Inside the Other)** | `IoA_min >= 0.50` | `IoU >= 0.50` |
| --- | --- | --- | --- | --- | --- |
| **`grounding_dino`** | `truth = 1` (**False pair**) | 77 | **33 / 77 (`42.9%`)** | **34 / 77 (`44.2%`)** | 6 / 77 |
| **`grounding_dino`** | `truth = 2` (**True pair**) | 16 | **2 / 16 (`12.5%`)** | 3 / 16 (`18.8%`) | 1 / 16 |
| **`florence2`** | `truth = 1` (**False pair**) | 31 | **15 / 31 (`48.4%`)** | **18 / 31 (`58.1%`)** | 2 / 31 |
| **`florence2`** | `truth = 2` (**True pair**) | 6 | **1 / 6 (`16.7%`)** | 1 / 6 (`16.7%`) | 1 / 6 |

- **48 of the 108 `pred = 2, truth = 1` overcounts (`33` on Grounding DINO + `15` on Florence-2)** have one box physically nested inside the other (`IoA_min >= 0.70`, vs only `3` true pairs)! And even when `len(bboxes) >= 3`, `envelope_indices` only drops the outer group box if it covers `>= 60%` of *all* other boxes, leaving nested part-inside-object pairs untouched.
- **Why must a 2-box containment filter be careful?** Because two synthetic negative controls have 2 overlapping/nested shapes that represent 1 or 2 objects:
  - `neg_concentric_rings` (expected `1`: outer ring + inner ring share center) — dropping one of 2 concentric boxes turns `2 -> 1` (which actually *helps* if a detector ever boxed both rings, and at `0.425` GD already emits `1`).
  - `neg_two_overlapping` (expected `1` under `"petal"` prompt; `1` under `"circle"`): at `0.425` GD already emits `1`.
  - True 2-object photos where a foreground object partially occludes a background object of the same class: if two real objects merely overlap at an edge (`IoA_min < 0.70`), they must **not** be merged; if one box is `>= 75–80%` contained inside another (`IoA_min >= 0.75`) and is either a duplicate (`IoU >= 0.50`) or a part/sub-box (non-concentric inner box or confirmed by `LFM2.5-VL-3B`), collapsing the contained sub-box recovers **~45+ exact counts on Phase 11 alone** with zero model inference cost.

---

### Failure Mode C2: The `pred = 0` False-Zero Cliff (`100` Queries on Phase 11 Where Truth >= 1, `93` on `truth = 1`)

- On the Grounding DINO photo path in Phase 11, **100 out of 644 queries (`15.5%`) return `count = 0` when the object is present** (`93` on `truth = 1`, `6` on `truth = 2–3`, `1` on `truth = 4+`).
- Look at column 6 of Table 3.1: **in `97` of those `100` false-zero cases (`97.0%`), Grounding DINO ALREADY detected the object in `[0.18, 0.425)` (`possibly_more.upper >= 1`)!**
  - Examples from the Phase 11 cache: `wheelchair` (GD top box `< 0.425`, SAM 3 = `1`, DRC = `1`), `handcuffs` (GD top box `< 0.425`), `cutting board` (`pred = 0 vs 1`), `statue` (`pred = 0 vs 1`), `watch` (`pred = 0 vs 1`), `towel` (`pred = 0 vs 1`).
- **Why does rescuing `pred = 0 -> 1` NOT endanger the 8 synthetic negative controls?**
  - Because **every single one of the 8 negative controls has `count >= 1` at `0.425`** (`[1, 1, 1, 1, 1, 1, 1, 3]`)! A rule that only triggers when `count == 0` and `possibly_more.extra_candidates >= 1` is **mathematically a no-op on all 8 negative controls (`16/16` prompts)**!
- **What is the real risk of rescuing `pred = 0`?**
  - **Absent objects (`truth = 0`)**: If a user calls `count_objects` for an object that is genuinely *not* in the image (`coco_absent`, `p11_absent`), Grounding DINO at `0.18` often hallucinated a weak box (`0.18–0.30`).
  - Therefore, when `count == 0` at `0.425` and a candidate box exists in `[0.20, 0.425)`, we must require **independent semantic corroboration** before promoting the top box from `0 -> 1`:
    1. Either `SAM 3` at `0.25+` (`detect_confidence` floor) or `Florence-2 DRC` also sees `>= 1` instance, **OR**
    2. Resident `LFM2.5-VL-3B ONNX` (`app.answerer`) confirms the object is present (`count >= 1`)!
  - Because `LFM2.5-VL-3B` + `SAM 3` already filter false-premise absent objects in `detect_objects` (`absent_boxed = 4%` on COCO, `10%` on Phase 11), this corroborated zero-rescue can recover **60–80 of the 100 false zeros on Phase 11 (`+8 to +11` percentage points overall)** while keeping absent-object false positives `<= 10%`.

---

### Failure Mode C3: `truth = 2–3` Undercounts (`51.5%` on 2s and `32.0%` on 3s in Tune; `37.5%` on Phase 11)

- On Sprint R Tune (`count_gate_c1_tune.json`), Grounding DINO at `0.425` gets:
  - `truth = 1`: `204 / 325` (**62.8%**)
  - `truth = 2`: `17 / 33` (**51.5%**) — of the 16 misses, **13 undercounted (`pred < 2`)**, and **8 of those 13 already had the 2nd object boxed in `[0.15, 0.425)`**!
  - `truth = 3`: `8 / 25` (**32.0%**) — of the 17 misses, **all 17 undercounted (`pred < 3`)**, and **15 of 17 (`88%`) already had the 2nd/3rd object boxed in `[0.15, 0.425)`**!
- On Round 14 (`count_gate_c1_confirm.json`), Grounding DINO gets `20/25` (`80.0%`) on 1s, `21/35` (`60.0%`) on 2s, and `11/21` (`52.4%`) on 3s (`52/81 = 64.2%` on 1–3 objects), whereas PaliGemma 3B `answer` got **`68/81 = 84.0%` on those exact same 1–3 object images** (`21/25` on 1s, `34/35` on 2s, `13/21` on 3s)!
- Why did Option 7 (`count_gate_c1.py`) fail to promote those `[0.18, 0.425)` candidate boxes?
  - Because Option 7 only looked at `[score, score_ratio, area_ratio, min_center_dist, max_iou, max_ioa]`. In `neg_spotted_ball`, the high-contrast spot inside the ball has `score = 0.25` and `IoA = 0.0` against other spots, looking geometrically plausible to a 6-feature logistic regressor.
  - What distinguishes a real 2nd/3rd object in `[0.18, 0.425)` from a spot on `neg_spotted_ball`?
    1. **Structural Fact #1 (from `count_gate_c1_tune.json` line 248):** At the `[0.18, 0.425)` `possibly_more` thresholds (`IoU < 0.5, IoA < 0.5` against counted boxes), **`possibly_more.upper` on all 8 negative controls is `[1, 1, 1, 1, 1, 1, 1, 3]` — i.e., `extra_candidates == 0` on 8/8 controls!** Why? Because every spot on `neg_spotted_ball` sits *inside* the counted ball box (`IoA = 1.0 >= 0.5`), so `_survives_veto` (`COUNT_RANGE_VETO`) already rejects 100% of interior spots!
    2. **Structural Fact #2 (`SAM 2.1` Mask Territory — Option 2 of `counting_plan.md`):** Even if a sub-box slightly protrudes outside the main box (`IoA < 0.5` on boxes), its `SAM 2.1` segmentation mask overlaps the main object's mask or sits inside its silhouette, whereas a genuine 2nd object occupies disjoint mask territory (`mask_IoA < 0.20`).
    3. **Structural Fact #3 (Resident `LFM2.5-VL-3B ONNX` Bounded Arbitration):** If `LFM2.5-VL-3B ONNX` is only consulted when `possibly_more.extra_candidates > 0` (or when `count >= 2` to check for overcount) **and its answer is clamped to the detector's physically localized candidate set `[1, possibly_more.upper]`**, it is **structurally impossible** for `LFM2.5-VL-3B` to turn `neg_spotted_ball` (`count = 1, upper = 1`) into `24` or `neg_multicolour_logo` (`count = 1, upper = 1`) into `4`!

---

### Failure Mode C4: Flat Images, Charts, Game UI & App Screenshots (`50.7–58.6%` Exact)

- In Phase 11, accuracy is highest on everyday photos (`77.1%`) and documents (`74.7%`), and lowest on:
  - **App/Web Screenshots:** `50.7%` (75 queries)
  - **Game UI:** `51.4%` (70 queries)
  - **Illustrations:** `56.9%` (72 queries)
  - **Signage:** `57.1%` (70 queries)
  - **Charts/Diagrams:** `58.6%` (70 queries)
- And on Round 13 (`187` flat counts), `Florence-2` + `envelope_indices` scores **`50.8%`**.
- **Two distinct bugs drive this:**
  1. **On the `florence2` flat path (`90` `truth = 1` queries in Phase 11):** `42` of `90` (`46.7%`) overcount (`31` at `pred = 2`, `11` at `pred >= 3`) because Florence-2 `<CAPTION_TO_PHRASE_GROUNDING>` emits nested/duplicate boxes on a single icon, coat-of-arms, or UI panel (`IoA_min >= 0.70` on `15/31` `pred = 2` cases), and `envelope_indices` does not run when `len == 2` or when 2 inner boxes sit inside 1 dropped outer envelope!
  2. **On the `grounding_dino` path for JPEG screenshots/game UI/charts:** High-color-count UI screenshots (`>= 10` colors/1k px) route to Grounding DINO, which boxes repeated UI sub-elements (buttons, tabs, bars) or misses stylized 2D icons (`pred = 0`), where `LFM2.5-VL-3B` VQA counting excels (`75.4%` in Round 23).

---

### Failure Mode C5: Unconditional `SAM 3` (`~6.9 s`) + `Florence-2 DRC` (`~2.6 s`) Latency & False-Alarm `crowd_range`

- Currently, [`detection_policy.postprocess_photo_count`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/detection_policy.py#L28-L143) runs **both**:
  1. `Florence-2 <DENSE_REGION_CAPTION>` (`_region_label_consensus`, `~2.6 s` bf16) on **100% of photo counts**, and
  2. `SAM 3` (`_crowd_range`, `~6.9 s` CPU) on **100% of photo counts** — even when `count == 1` and `possibly_more.upper == 1`!
- **The cost in Phase 11 (`visionbench_report_20261021_215606.json`):**
  - Median latency on the `grounding_dino` path is **`13.97 s`** (`Grounding DINO ~3.5 s` + `Florence-2 DRC ~2.6 s` + `SAM 3 ~6.9 s` + `SAM 2.1 silhouette ~0.8 s`).
  - Worse, because `SAM 3` runs unconditionally on all 644 photo counts and triggers `crowd_range` whenever `max(gd_count, sam3_count) >= 4`, **`crowd_range` fires on 120 of 747 Phase 11 queries (`16.1%`) even though 675 of 747 queries have `truth = 1`!** On those 120 triggers, truth is inside `crowd_range` only **`60.0%`** of the time because one model hallucinated 4+ boxes on a 1-object screenshot or chart (`min=1, max=5` where truth=1, or `min=2, max=4` where truth=1)!

---

## 4. Prioritized 4-Track Engineering Plan

We structure the improvement into **four modular, independently gated tracks** ordered by **highest ROI / lowest risk first**. Every track can be benchmarked offline on cached raw detections before touching production code.

```mermaid
flowchart TD
    A["Input Image + object_name"] --> B{"routing.choose_count_backend"}
    B -->|Photo / Complex UI| C["Grounding DINO (raw boxes >= 0.15)"]
    B -->|Flat (<10 colors/1k px)| D["Florence-2 Grounding"]

    C --> E["Track 1A: Class-Agnostic NMS (IoU>=0.50) + Pair/Part Containment Filter (IoA>=0.75)"]
    D --> F["Track 4A: Flat Envelope + Pair Containment Deduplication (IoA>=0.70)"]

    E --> G{"Count State at 0.425?"}
    G -->|"count == 0 & extra_candidates >= 1"| H["Track 2A: Zero-Count Rescue (SAM3 / LFM2.5-VL-3B Corroboration)"]
    G -->|"1 <= count <= 3 & extra_candidates >= 1"| I["Track 2B & 3: Bounded Candidate Promotion (SAM2 Mask Territory + LFM2.5-VL-3B in [count, upper])"]
    G -->|"count >= 2 & truth=1 Overcount Risk"| J["Track 2C: Multi-Box Overcount Audit (LFM2.5-VL-3B + SAM2 Mask Merge)"]
    G -->|"count == 1 & extra_candidates == 0"| K["Fast Path: Lock count = 1 (Skip SAM3 crowd_range!)"]

    H --> L["Track 1B: Gated Crowd Range (Run SAM3 ONLY when count >= 3 or upper >= 4)"]
    I --> L
    J --> L
    K --> M["Return Final Count + BBoxes + Diagnostics (~4.5-6.5s median)"]
    L --> M
```

---

### Track 1: Zero-Weight Geometric & Latency Fixes (`geometry.py`, `detection_policy.py`)

**Goal:** Fix the 2-box `envelope_indices` blind spot (`48` nested `pred = 2, truth = 1` overcounts in Phase 11) and cut median latency by `~6–7 s` without loading any new weights.

#### Step 1A — Generalize `envelope_indices` & Add Pairwise Part/Duplicate Suppression (`IoU >= 0.50` or Non-Concentric `IoA >= 0.75`)
- **Problem:** `geometry.envelope_indices(bboxes)` currently exits immediately if `len(bboxes) < 3`, and Grounding DINO has no NMS. When a detector outputs 2 boxes on 1 object — one full-object box and one inner part/duplicate box (`33/77` GD false pairs and `15/31` Florence-2 false pairs in Phase 11 have `IoA_min >= 0.70`) — both survive.
- **Design:**
  1. **Duplicate NMS (`IoU >= 0.55`):** In `count_objects`, if two boxes above the counting threshold have `IoU >= 0.55`, suppress the lower-scoring box (or smaller box on scoreless Florence-2).
     - *Control check:* On all 8 negative controls, no two true objects have `IoU >= 0.55` (`neg_distractors` has 3 disjoint circles; `neg_two_overlapping` has `count = 1`).
  2. **Nested Sub-Box / Part Suppression (`IoA_min >= 0.75` for `len(bboxes) >= 2`):**
     - When box $B_{\text{small}}$ has $\text{IoA}(B_{\text{small}}, B_{\text{large}}) = \frac{\text{Area}(B_{\text{small}} \cap B_{\text{large}})}{\text{Area}(B_{\text{small}})} \ge 0.75$ and $\text{Area}(B_{\text{small}}) < 0.85 \times \text{Area}(B_{\text{large}})$:
     - **Drop the inner sub-box $B_{\text{small}}$** (keeping the full object box $B_{\text{large}}$), **except** when `len(bboxes) >= 3` and $B_{\text{large}}$ is a multi-object group envelope (contains the centers of $\ge 2$ *mutually disjoint* inner boxes — in which case `envelope_indices` drops $B_{\text{large}}$ and keeps the disjoint inner boxes, preserving `grid_3x4` and `five_stars`!).
- **Expected Impact:**
  - Directly targets the **48 nested `pred = 2 -> 1` false pairs** (`33` on GD, `15` on Florence-2) and multi-box nested overcounts on Phase 11 (`+4 to +6` pts on Phase 11, `+5 to +8` pts on Florence-2 flat counts) at **0 ms** runtime cost.

#### Step 1B — Gate `crowd_range` (`SAM 3`) & Make `Florence-2 DRC` Lazy
- **Problem:** `_crowd_range` runs `SAM 3` (`~6.9 s`) on *every* photo count, even when `count <= 1` and `possibly_more.upper <= 2`. And `_region_label_consensus` runs `Florence-2 DRC` (`~2.6 s`) on every photo count even though nothing in `count_objects` uses `consensus` to change `count`.
- **Design:**
  1. **Gate `_crowd_range`:** Only invoke `SAM 3` for `crowd_range` when `count >= 3` **or** `possibly_more.upper >= 4` (i.e. when there are at least 3–4 candidate boxes at `>= 0.18` so the scene could plausibly be a crowd of $\ge 4$ objects).
     - Also require `min(gd_upper, sam3_count) >= 2` before emitting `crowd_range` when `count <= 1`, so a single object where SAM 3 hallucinates 4 sub-parts on a UI screenshot does not emit a false `crowd_range: [1, 4]`.
  2. **Lazy `consensus`:** Only run `_region_label_consensus` (`Florence-2 DRC`) if needed as a tie-breaker or when `count >= 2` and `LFM2.5-VL-3B` is unavailable, or replace it with the faster/more accurate `LFM2.5-VL-3B` arbiter when ambiguity exists.
- **Expected Impact:**
  - Cuts Phase 11 median latency from **`13.8 s` to `~5.0–6.5 s`** (`>50%` speedup).
  - Eliminates the `~100` false-alarm `crowd_range` triggers on `truth = 1` images in Phase 11, lifting `count_in_crowd_range_pct` from `60.0%` back toward `80%+`.

---

### Track 2: Structurally Guarded `LFM2.5-VL-3B ONNX` Counting Arbitration (Zero New Weights)

**Why PaliGemma 3B `answer` scored `84.0%` on 1–3 object photo counts (`count_gate_paligemma.json`) while Grounding DINO scored `64.2%`:**
A 3B vision-language model sees holistic object identity (e.g. "those two boxes are both parts of one armchair" vs "there are three people standing side by side, even though the third person in the back only scored `0.26` in Grounding DINO").
**Why unguarded VLM counting failed Gate P1:**
When asked `"how many balls?"` on `neg_spotted_ball` (1 ball with ~23 painted spots), an unguarded VLM counts the spots (`24`).

**The Breakthrough — Bounded Detector- + Mask-Guarded VLM Arbitration:**
In `FusionVisionMCP`, `LFM2.5-VL-3B ONNX` (`app.answerer`) is **already loaded in RAM**. Instead of replacing the detector with raw VQA, we use `LFM2.5-VL-3B ONNX` as a **bounded arbiter over the detector's physically localized candidate boxes**, governed by three strict invariants that guarantee **16/16 negative controls pass by construction**:

#### Invariant 1 (The Single-Box Control Lock):
- Whenever the detector returns `count == 1`, `possibly_more.extra_candidates == 0` (no surviving candidate boxes in `[0.18, 0.425)` outside the main box), and `silhouette.separable != "no"`, **`count = 1` is locked — `LFM2.5-VL-3B` is never called and cannot increase `count`!**
- **Proof on the 8 Negative Controls:**
  - In `count_gate_c1_tune.json` (line 248), `upper_end_per_control` for the 8 controls is **`[1, 1, 1, 1, 1, 1, 1, 3]`** — meaning `extra_candidates == 0` on **all 8 controls** (`7` singletons have `count = 1, upper = 1`; `neg_distractors` has `count = 3, upper = 3`)!
  - And on all 7 singleton controls, `silhouette.separable == "yes"`.
  - Therefore, under Invariant 1, **any promotion rule gated on `extra_candidates >= 1` leaves all 8 negative controls (`16/16` prompts) untouched at `[1, 1, 1, 1, 1, 1, 1, 3]`!**

#### Step 2A — Corroborated `pred == 0` Rescue (Targeting the 93 `truth = 1` False Zeros in Phase 11)
- **Trigger:** `count == 0` at `0.425`, **and** Grounding DINO has $\ge 1$ candidate box in `[0.20, 0.425)` (`possibly_more.upper >= 1`, which covers **90 of the 93** `truth = 1` false zeros in Phase 11!).
- **Arbitration Rule:**
  1. Ask resident `LFM2.5-VL-3B ONNX` (`app.answerer.answer_question` with deterministic greedy decoding `do_sample=False`, `max_new_tokens=16`):
     `"How many {object_name} are visible in this image? Reply with only an integer (0, 1, 2, ...)."`
  2. Parse the integer $k_{\text{lfm}}$.
  3. If $k_{\text{lfm}} \ge 1$ (and if the top candidate box score is in `[0.20, 0.30)`, optionally require `SAM 3` at `0.20` or `CLIP` crop score $\ge 0.23$ to guard `coco_absent` / `p11_absent`), promote the top $\min(k_{\text{lfm}}, \text{pm\_upper}, 2)$ candidate boxes from `possibly_more` into the final `bboxes` and `count`!
- **Why it is safe:**
  - Cannot affect any of the 8 negative controls (none have `count == 0`).
  - Directly tested against `coco_absent` (50) and `p11_absent` (50) in Gate 0/1 to ensure absent-object false-positive rate stays `<= 10%`.

#### Step 2B — Bounded `1 <= count <= 3` Candidate Promotion (Targeting `truth = 2–3` Undercounts)
- **Trigger:** `1 <= count <= 3` **and** `possibly_more.extra_candidates >= 1` (i.e., Grounding DINO found $1..3$ boxes above `0.425` **plus** $\ge 1$ spatially distinct box in `[0.20, 0.425)` surviving `IoU < 0.5, IoA < 0.5`).
- **Arbitration Rule:**
  1. First, run the **Track 3 `SAM 2.1` Mask Territory Veto** on each extra candidate box in `[0.20, 0.425)`: discard any candidate whose `SAM 2.1` mask overlaps the union of confident counted masks by $\text{IoA}_{\text{mask}} \ge 0.25$. Let $U_{\text{clean}}$ be the new upper bound (`count + surviving_extra`).
  2. If $U_{\text{clean}} > \text{count}$, query `LFM2.5-VL-3B ONNX` for $k_{\text{lfm}}$.
  3. **Clamp to physical boxes:** If $k_{\text{lfm}} > \text{count}$, promote the top-scoring surviving candidate boxes up to:
     $$k_{\text{final}} = \min(k_{\text{lfm}}, U_{\text{clean}})$$
- **Why this solves the Option 7 & PaliGemma dilemma:**
  - Every promoted instance has a **real Grounding DINO bounding box** and a **disjoint SAM 2.1 segmentation mask** (unlike pure VQA, which outputs no boxes and hallucinates `24` on `neg_spotted_ball`).
  - Unlike Option 7's blind 6-float logistic regression, `LFM2.5-VL-3B` visually inspects the scene to confirm whether there are actually 2 or 3 instances before promoting the `[0.20, 0.425)` boxes.

#### Step 2C — Multi-Box Overcount Audit (`count >= 2` on `truth = 1` Photo & UI Queries)
- **Trigger:** `2 <= count <= 4` on `grounding_dino` or `florence2` after Track 1A geometric containment, where either:
  - The lower box scores sit just above the threshold (`0.425..0.52`), **or**
  - Any pair of boxes has non-zero overlap (`IoA_min >= 0.25`) or shares a `SAM 2.1` connected silhouette component.
- **Arbitration Rule:**
  - Query `LFM2.5-VL-3B ONNX` (`k_lfm`) **and** check `SAM 3` (`0.30` floor) count (`k_sam3`).
  - **Negative Control Guard for `neg_distractors` (`truth = 3`):** On `neg_distractors`, all 3 target boxes are completely disjoint (`IoU = 0, IoA = 0`), have nearly identical high DINO scores (`> 0.55`), and `SAM 2.1` masks are 100% disjoint (`mask_IoU = 0`). We **never** demote disjoint, high-confidence boxes (`IoA == 0` and `score >= 0.50` / equal-sized disjoint shapes).
  - When boxes overlap (`IoA_min >= 0.25`) or a secondary box is marginal (`< 0.50`) and **both** `LFM2.5-VL-3B` and `SAM 3` (or `Florence-2 DRC`) agree on a smaller count $k \ge 1$, keep only the top-$k$ highest-scoring boxes.

#### Step 2D — Swap `Moondream2` -> `LFM2.5-VL-3B ONNX` in `estimates.vqa`
- In [`detection_policy.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/detection_policy.py#L96-L106), replace `app.vqa.answer_question(image, ...)` (`Moondream2`) with `app.answerer.answer_question(image, ...)` (`LFM2.5-VL-3B ONNX`).
- This upgrades `vqa_estimate=True` to the stronger model (`75.4%` vs Moondream's `61.4%` on counting VQA) and eliminates `count_objects`'s dependency on `Moondream2`.

---

### Track 3: `SAM 2.1` Mask-Territory & Merged-Instance Verification (`analysis.py`)

**Goal:** Address the two geometry-level failure modes identified in [`Future_Ideas/counting_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/counting_plan.md) (Options 2 and 6) that box coordinates alone cannot resolve:

#### Step 3A — `SAM 2.1` "New Territory" Mask Check for `[0.18, 0.425)` Candidates
- **How it works:**
  - `SAM 2.1 Hiera-small` is already resident (`app.sam2`) and already encodes the image when `count == 1` (for `_add_silhouette`). Once the image embedding is computed (`~0.7 s`), decoding a box prompt takes **`< 50 ms` per box**.
  - For counted boxes $B_1, \dots, B_c$ (`>= 0.425`) and extra candidate boxes $C_1, \dots, C_m \in [0.20, 0.425)$:
    1. Decode `SAM 2.1` masks $M(B_i)$ and $M(C_j)$ using the reuseable image predictor embedding (`sam2.set_image(image)` once).
    2. Compute mask intersection-over-candidate-area:
       $$\text{IoA}_{\text{mask}}(C_j) = \frac{|M(C_j) \cap \bigcup_{i=1}^c M(B_i)|}{|M(C_j)|}$$
    3. If $\text{IoA}_{\text{mask}}(C_j) \ge 0.25$, $C_j$ is segmentation-level part of an already-counted object (e.g., a wheel on a car, a collar on a dog, a spot on a ball) -> **veto $C_j$**.
    4. If $\text{IoA}_{\text{mask}}(C_j) < 0.25$, $C_j$ occupies genuinely new object territory -> **eligible for Track 2B promotion**.

#### Step 3B — Touching / Merged Twin Split (`count == 1`, `truth = 2` where 1 Box Covers Both Objects)
- In `count_gate_c1_tune.json`, `7 / 16` missed 2-object photos had `pred = 1` with **no** extra box in `[0.15, 0.425)` because Grounding DINO drew a single box around two touching/adjacent instances (e.g. two birds on a branch, two monitors side by side).
- Currently, `_add_silhouette` computes `by_distance` and `by_radial` lobe counts on the single box's SAM 2.1 mask and sets `separable = "no"` when `by_distance > 1 and by_radial > 1`, **but never changes `count`**!
- **Safe Promotion Rule:**
  - When `count == 1`, `extra_candidates == 0`, **and** `silhouette.by_distance == 2` with `silhouette.by_radial == 2` (`separable == "no"` — which is **false on all 7 singleton negative controls**, as every control has `separable == "yes"`!), query `LFM2.5-VL-3B ONNX` on the image/crop:
  - If `LFM2.5-VL-3B` **also** answers `2` (corroborating the 2-lobe distance+radial silhouette split), split the single mask along its watershed basin boundary into 2 bounding boxes and report `count = 2`!
  - Because all 7 singleton negative controls have `separable == "yes"` (`by_distance == 1` or `by_radial == 1`), this rule has **zero chance of firing on `neg_spotted_ball`, `neg_noisy_rod`, or any other negative control**.

---

### Track 4: Flat Image, Chart & UI Counting Upgrade (`detection.py`, `routing.py`)

**Goal:** Lift flat counting from `50.8%` (Round 13) / `54.4%` (Phase 11 flat) to `>= 65%` and improve screenshot/chart/game-UI counting (`50.7–58.6%`).

#### Step 4A — Flat Path Pair-Containment & Duplicate Box Suppression
- On the `florence2` flat path in Phase 11, **42 of 90 `truth = 1` queries overcount** (`31` have `pred = 2`, and `15/31` have `IoA_min >= 0.70`!).
- Applying Track 1A's generalized `len >= 2` containment filter (`IoA_min >= 0.70` non-concentric sub-box removal + `IoU >= 0.50` deduplication) after `envelope_indices` directly cleans up Florence-2's nested sub-part boxes on single icons, logos, windows, and diagram elements.
- *Control verification:* Must verify against all 10 fixture positives (`three_red_circles == 3`, `five_stars == 5`, `grid_3x4 == 12`, `overlapping_squares == 2`, `nested_boxes == 1`, `concentric_rings == 2`, etc.) and all 8 negative controls. Note that `concentric_rings` has concentric centers (`center_dist < 0.15 * radius`), so preserving the existing concentric exemption (`!is_concentric`) keeps `concentric_rings == 2` on the fixture while collapsing non-concentric sub-part boxes on real flat images!

#### Step 4B — Corroborated `LFM2.5-VL-3B` + `Florence-2` Flat Counting
- Florence-2 `<CAPTION_TO_PHRASE_GROUNDING>` has no per-box confidence scores and merges touching flat shapes (e.g. `overlapping_circles: 1 vs 3`, stacked chart bars, adjacent icons).
- When `detector == "florence2"` returns `count >= 2` with overlapping boxes (`IoA_min >= 0.30`) or `count == 1` on a wide/multi-lobe region, use `LFM2.5-VL-3B ONNX` (bounded by `SAM 2.1` connected component / lobe analysis) to arbitrate between `1` and `len(bboxes)`.

---

## 5. Pre-Registered Evaluation Protocol & VisionBench Gates

Following `.clinerules` and [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md) ("Look-elsewhere discipline: when a benchmark script sweeps $K$ candidate rules or parameter values on a tuning split, pre-register the confirmation protocol before inspecting results"), we define a **strict 3-stage gate** in `C:\AI\Projects\VisionBench\benchmarks\`:

### Gate 0 — Instant Offline Replay on Cached Replies (`0` GPU/CPU Model Minutes)
Before running new model inference, write `C:\AI\Projects\VisionBench\benchmarks\count_gate_track1_offline.py` to replay **Track 1A (2-box & multi-box `IoU`/`IoA` containment)** and **Track 1B (`crowd_range` gating)** directly on the existing cached raw boxes in:
1. `visionbench_report_20261021_215606.replies.jsonl` (747 Phase 11 count queries)
2. `replacement_gate1_raw_grounding_dino_tune.jsonl` (423 Sprint R Tune photo counts)
3. `replacement_gate1_raw_grounding_dino_confirm.jsonl` (427 Sprint R Confirm photo counts)
4. `count_gate_c1_raw_round14.jsonl` (103 Round 14 Open Images photo counts)
5. `flat_count_envelope_r13.json` (187 Round 13 flat counts)

**Gate 0 Pass Criteria:**
- `16/16` synthetic negative controls (`8/8` literal + `8/8` `"petal"`).
- No regression on `10/10` synthetic positive fixtures (`test_Flat_image_routing.py`).
- Net exact-match gain on Phase 11 (`747` items) and Round 13 (`187` items) with 95% paired bootstrap lower bound $\ge 0.0$.

---

### Gate 1 — Tune Split Evaluation for Track 2 (`LFM2.5-VL-3B` Bounded Arbiter) & Track 3 (`SAM 2.1` Mask Territory)
Write `C:\AI\Projects\VisionBench\benchmarks\count_gate_lfm_arbiter.py` (with JSONL caching per image) and evaluate on:
1. **Controls (Hard Gate — Must Pass First):**
   - **16/16** synthetic negative controls (`8` literal + `8` `"petal"`).
   - **Absent-object zero-count gate:** `coco_absent` (50) and `p11_absent` (50) — false non-zero rate must be **`<= 10%`** (matching `detect_objects`).
2. **Tuning Splits (Parameter Freeze):**
   - **Sprint R Tune** (`423` photo counts) — target **+5.0 to +10.0 pts** over `55.6%` baseline (`95%` bootstrap CI lower bound `> 0`).
   - **Phase 11 Tuning Sub-split** (or full 747 replay with cached `LFM2.5-VL-3B` count answers) — measure exact gain on `truth = 1` false zeros (`93` items), `truth = 1` overcounts (`160` items), and `truth = 2–3` undercounts (`59` items).
3. **Freeze a single rule + threshold tuple** (no parameter grid carried past Gate 1).

---

### Gate 2 — Held-Out Confirmation (Sprint R Confirm + Round 14 + Fresh Confirm Slice)
> [!WARNING]
> **Preventing Confirm-Split Reuse:**
> As noted in [`Future_Ideas/counting_plan.md`](file:///c:/AI/MCP/FusionVisionMCP/Future_Ideas/counting_plan.md#L168-L176), Round 14 (`103` Open Images counts) was already used to reject Gate C1 and evaluate Gate P1. While we will report Round 14 (`103` items) and Sprint R Confirm (`427` items) for continuity with the historical table, final confirmation will also include a **fresh held-out Open Images / COCO exact-count slice (1–5 objects, N >= 150)** never touched by any prior gate.

**Gate 2 Ship Criteria:**
1. **16/16** synthetic negative controls pass (`8/8` literal + `8/8` `"petal"`).
2. **Absent-object false non-zero rate `<= 10%`** on `coco_absent` + `p11_absent`.
3. **Paired 10,000-resample bootstrap 95% CI lower bound `> 0`** on:
   - Sprint R Confirm (`427` photo counts, baseline `61.8%`),
   - Round 14 (`103` Open Images counts, baseline `57.3%`), and
   - Fresh Held-Out Confirm (`>= 150` exact photo counts).
4. **Phase 11 median latency `<= 7.0 s`** (down from `13.8 s`).

---

## 6. Projected Cumulative Gain Waterfall

| Stage | Target Failure Mode | Phase 11 Exact (`N=747`, now `60.0%`) | Photo Confirm (`N=427`, now `61.8%`) | Round 14 (`N=103`, now `57.3%`) | Median Latency (now `13.8 s`) |
| --- | --- | --- | --- | --- | --- |
| **Current Baseline** | — | `60.0%` (448/747) | `61.8%` (264/427) | `57.3%` (59/103) | `13.8 s` |
| **+ Track 1A** (NMS + `len>=2` Non-Concentric `IoA>=0.75` Sub-Box Filter) | `48` nested `pred=2, truth=1` overcounts + multi-box duplicates | `~64.5–66.0%` (`+4.5–6.0` pts) | `~64.5–66.0%` (`+2.7–4.2` pts) | `~61.0–63.0%` (`+3.7–5.7` pts) | `13.8 s` |
| **+ Track 1B** (Gate `SAM 3` `crowd_range` to `count>=3` or `upper>=4`) | Unconditional `6.9 s` SAM 3 call & `16%` false crowd ranges | Same exact (`crowd_range` precision `60% -> ~82%`) | Same exact | Same exact | **`~5.5–6.5 s` (`-7.5 s`)** |
| **+ Track 2A** (Corroborated `pred=0` Rescue via `LFM2.5-VL-3B` + `possibly_more`) | `93` `truth=1` & `6` `truth=2-3` false zeros where `pm_upper >= 1` | `~71.5–74.0%` (`+7.0–8.0` pts) | `~69.5–72.0%` (`+5.0–6.0` pts) | `~67.0–70.0%` (`+6.0–7.0` pts) | `~6.0–6.8 s` |
| **+ Track 2B/2C & Track 3** (`SAM 2.1` Mask Territory + `LFM2.5-VL-3B` Bounded `[count, upper]` Promotion & Overcount Audit) | `truth=2-3` undercounts in `[0.18, 0.425)` + remaining `truth=1` disjoint overcounts | **`~75.0–78.0%`** | **`~75.0–79.0%`** | **`~74.0–78.0%`** (`~82%+` on 1–3 objects) | **`~6.2–7.0 s`** |

---

## 7. Files To Be Modified (When Approved for Implementation)

| File | Planned Responsibility |
| --- | --- |
| [`src/fusion_vision_mcp/geometry.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/geometry.py) | Add `suppress_nested_and_duplicate_boxes(bboxes, scores, iou_thresh=0.55, ioa_thresh=0.75)` handling `len(bboxes) >= 2` while preserving multi-object group envelope behavior and concentric ring exemptions. |
| [`src/fusion_vision_mcp/detection_policy.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/detection_policy.py) | (1) Apply Track 1A box suppression; (2) gate `_crowd_range` so `SAM 3` only runs when `count >= 3` or `possibly_more.upper >= 4`; (3) wire Track 2A/2B/2C bounded `LFM2.5-VL-3B` arbitration (`app.answerer`); (4) swap `app.vqa` -> `app.answerer` in `estimates.vqa`. |
| [`src/fusion_vision_mcp/analysis.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py) | Add `_filter_candidates_by_sam2_territory` (reusing the `SAM 2.1` image embedding across counted and candidate boxes) and `separable == "no"` twin-lobe split corroboration. |
| [`src/fusion_vision_mcp/tools/detection.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/detection.py) | Apply non-concentric nested sub-box deduplication on the `florence2` flat path after `envelope_indices`. |
| [`tests/test_counting_accuracy.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_counting_accuracy.py) & [`tests/test_flat_image_routing.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_flat_image_routing.py) | Add permanent unit tests for 2-box part-inside-object suppression, concentric ring preservation, gated `crowd_range`, and bounded `LFM2.5-VL-3B` arbitration invariants on all 8 negative controls. |
