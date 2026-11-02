# `query_image` Frontier-Grade Improvement Plan

**Status:** Drafted 2026-10-30. **Reviewed 2026-10-30: the Review section below supersedes the rest where they
disagree.** R.4 steps 1-4 done (see "Progress" below the review). Round 29 briefed 2026-11-01 (VisionBench
`worker_handoff/ROUND29_PLAN.md`, `ROUND29B_PLAN.md`; Gate V5 rules in `benchmarks/vqa_gate_v5.py`). **Gate V5 ran
2026-11-02: the routing and the Track 5 rule both pass** (Progress, step 6). Next: the user's go to ship them.
**Target tool:** `query_image` (`src/fusion_vision_mcp/tools/vqa.py:29–191`) and its supporting pipeline in `src/fusion_vision_mcp/analysis.py`, `src/fusion_vision_mcp/constants.py`, `src/fusion_vision_mcp/question.py`, and `src/fusion_vision_mcp/inspection.py`.
**Current scoreboard standing (`Future_Ideas/frontier_capability_gaps.md` row 100):**
- **VisionBench Phase 11 screen (`benchmarks/results/vqa_premise.json`, 1,200 questions across 400 generalist images):** **82.4%** (`ocr_text` arm, `989/1200`) vs. Moondream2 **78.8%** (`+3.6 pts`), median `4.62 s` (excluding `~1.8 s` OCR on routed questions).
- **VisionBench Round 23 confirm (`benchmarks/results/vqa_round23.json`, 892 fresh questions across 223 Round 21 images):** **78.5%** (`700/892`) vs. Moondream2 **66.5%** (`+12.0 pts`), median `4.91 s` vs. `9.77 s`.
- **VisionBench Round 22 TextVQA confirm (`benchmarks/results/vqa_round22_textvqa.json`, 154 unseen TextVQA val questions):** **52.4%** VQA exact match / **76.6%** answer-contained on `shipped` (`5.11 s`), while adding `"\nAnswer with a single word or phrase."` (`short` arm) reached **76.6%** VQA exact match (`+24.2 pts`, 95% CI `[+17.5, +31.4]`, `3.99 s`).
- **Frontier target (`frontier_capability_gaps.md`):** **~88–92%** on generalist VQA (`-6 to -10 points` current gap on Phase 11 / Round 23) and **76–81%** on TextVQA exact match (`-24 points` current gap on `shipped` due to sentence verbosity).

---

## Review (2026-10-30) — read this first

**Verdict.** The failure diagnosis holds, and most of its numbers reproduce from the caches. The biggest finding is
real: 32 of round 23's 105 "no text here" questions miss `VQA_TEXT_QUESTION`, and LFM gets only 4 of those 32 right.
Track 1's bugs are real too. The plan is not ready to build as written, for five reasons:

- Several code facts are wrong: Steps 1B, 1C and 1D, and `question.py`'s symbols.
- One ledger row (Gate R) is wrong.
- The `check_consistency` "simulation" was run against a stand-in for the control question, not the real one.
- It leaves out round 22's main result: the `ocr` reading adds nothing measurable on TextVQA.
- Its trajectory counts scorer fixes as model gains.

Corrections, a revised order and the open decisions follow. The body is left as written. Checks:
`qi_review_check.py` (session scratchpad, read-only, run with FusionVisionMCP's `.venv` from VisionBench).

### R.1 Re-derived from the caches

- **Round 23, by type and routing:** every cell matches. Two exceptions are the overall routed and unrouted counts:
  218/**274** and 482/**618**, not 268 and 624. ✅
- **Phase 11 (`ocr_text` arm):** every cell matches. ✅
- **The proposed two-branch regex's routing table:** matches exactly on both sets. ✅ But the regex was written
  by reading these sets' misses ("leaderboard", "p-value", "suggestion", "affixed"). Its coverage there is fitted,
  not evidence. Round 29 is the only test of it.
- **LFM's default-token answers on round 23:** 124 answers, 122 right (109 `yes_no`, 15 `no_text`). ✅
- **`_vqa_consistency` "44.4% flagged, 60.6% quiet on right":** this only reproduces when the cached `premise` arm
  (question + `PREMISE_SUFFIX`) stands in for the control answer. Paired with `plain` instead, it gives 27.6% / 72.9%.
  **No cache holds a real control answer** (`"Looking carefully at this image, answer precisely: …"`). There is also
  no phase 12 report since LFM shipped. So the "calibrated" figures in Step 5A (15.6%, 90.9%, 88.7%) were fitted to
  a stand-in, on the same data. Treat them as a sketch, not a measurement.
- **Step 3A's "+2 net, zero regression" on phase 11:** applying the `premise` arm to unrouted What/Where/Which/Who
  questions gives colour +1, object +1, position +2, count 0, and **text −5** (147 questions). The plan left text
  out. With the wider regex most of those 147 would be routed instead, so the loss would shrink. It still has to be
  counted. The cache also holds `PREMISE_SUFFIX`'s wording, not the proposed `VQA_WH_PREMISE_SUFFIX`, so the
  proposed wording needs a live run.

### R.2 Factual corrections (these override the body)

- **Line numbers are stale.** In `analysis.py`: `_vqa_consistency` is at :188, `_vqa_cross_check` :247,
  `_size_measurement` :294, `_TEXT_QUESTION_RE` :965, `lfm_vqa_prompt` :968, `vqa_answers` :980, and its
  `except (OSError, ImportError)` :992. The `VQA_*` constants are at `constants.py:259-266`. `ModelMemoryError` lives
  in `memory.py:74`; there is no `memory_manager.py`.
- **The Gate R ledger row is wrong.** Gate R (`region_inspect_gate.py`, 2026-10-15) used boxes the Gemini worker
  drew, not Grounding DINO crops, and it had no upscale. On 1,055 questions `query_image` went 79.0 → 80.3% (CI −1.0
  to +3.7), so it was not shipped: text +15, counts +13, colour −8, yes/no −5. `ocr` fell 71.6 → 54.5%. The "−7.8 pts,
  116 questions" figure is not in the gate. What the gate actually says, that a pointed crop helps text and hurts
  colour, bears directly on Track 4.
- **A ledger row is missing:** sprint Q1 (2026-10-10, `vqa_ocr.py`, Moondream2). Its third variant tagged each OCR
  line with its 3x3 grid cell, which is Track 4A's layout idea. It gained +1.1 (CI −1.7 to 3.9) and failed. Its
  lesson: "the limit is the model reasoning over text, not reading it." It was run on Moondream2, so it does not rule
  the idea out for LFM. The plan's Q1 row (+5.0 / +5.9) also does not match `frontier_capability_gaps.md`; re-check
  it against `vqa_ocr_tune.json`.
- **Round 22 was left out, and it matters.** `vqa_round22_textvqa.py`'s own result says contained is flat across
  arms: shipped 76.6, plain 77.9, `ocr_all` 74.7 (−1.9, CI −7.1 to 3.2). "**The `ocr` reading adds nothing measurable
  here.**" Shipped's median reply is **1 word**, not 12–40. The exact-match gap is answer format, and `ocr_all`'s +20.1
  comes from its prompt drawing short replies, not from reading. So:
  - Track 2's "~72.5% on round 22" is the wrong mechanism.
  - Round 22's 154 questions are now seen, so Gate V2B is a screen.
  - `frontier_capability_gaps.md` already scores TextVQA as ✅ par (76.6% short-answer). It is not a −24 gap.
- **The frontier 88–92% is an estimate (E),** made on phase 11. Round 23 has no frontier run.
- **`question.py` has none of the symbols named.** There is no `_COUNT_PATTERNS`, `_clean_span` or `_STOP_WORDS`.
  The real path is `_COUNT_PATTERN`, then `_trim_count_noun`, which cuts at the first `_COUNT_FILLER` substring
  (`" are"`, `" is"`, ...), then a `_STOPWORDS` check on the first word. The bugs themselves are real (checked live):
  - "…vases stand on…" returns `['matching pink vases stand']`.
  - "…figures appear inside…" returns `['heraldic figures appear']`.
  - "How many image panels…" and "How many photo thumbnails…" return `None`.

  Fix: add the finite verbs to `_COUNT_FILLER` and match them as whole words. A plain substring `find(" stand")` would
  also cut "standing", in the same way `" in"` already cuts "inside". Let a first-word stopword through when a noun
  follows it ("image panels").
- **Step 1D is backwards.** `head` and `finger` are used: `_PART_ANATOMY` holds arm, leg, head and finger, and turns
  them into observations. `hand` is the part `_PART_ANATOMY` does not tally; it only feeds the coverage line.
  Dropping head and finger would change the observations. The saving that is bit-identical is to reuse
  `_check_anatomy`'s `hand` call in `_check_defect_probe`. The arguments are the same (`adaptive_threshold=True`), and
  the probe's parts are `["hand", "face"]`. That takes 8 Grounding DINO passes to **7**, saving one call (~2-3.5 s),
  not 18 s.
- **Step 1B is only half right.** The answer and the control both call
  `ocr_pages(sharpened=False, lfm_read=False)`, so the two readings are identical and sharing one is a pure saving
  (~1.8 s). The cross-check's OCR is a different reading: `ocr_pages(app, [image])` with the defaults, which is the
  full `ocr` tool (sharpened pass plus LFM transcription), as documented ("`ocr` for a text-reading question"). Keep
  it. Feeding it the VQA reading would silently change `cross_check.text`.
- **Step 1C(1) describes the count pipeline wrongly.** Here is what `count_objects` actually does
  (`tools/detection.py:391-425`):
  1. `execute_detection(backend "auto", threshold, candidates_floor=COUNT_RANGE_FLOOR, drop_group_box=True)`
  2. `drop_nested`
  3. `possibly_more`
  4. `crowd_range`, only when `upper >= 3`
  5. `lfm_count_check` (a weak count of 1 becomes 0, or 0 becomes 1) on photos, or `lfm_flat_check` on Florence-2

  Flat counts run on **Florence-2, not SAM 3** (SAM 3 is `detect_objects`' flat path), and there is no "count 2–9" LFM
  step. Taking the per-result loop out of `count_objects` into a helper touches a shipped path. Re-run
  `count_lfm_parity.py` and `count_flat_lfm_parity.py` after it. It also adds SAM 3 and up to two LFM calls to a
  low-confidence cross-check (~10-15 s). See R.6 Q3.
- **Step 1C(2)'s call is incomplete.** `execute_detection` needs `backend_preference`. Mirroring `detect_objects`
  means `backend_preference="auto", threshold=DETECT_BOX_THRESHOLD, adaptive_threshold=False, locate=True`. That
  changes the output, though: picking the largest or smallest box among SAM 3's candidates, when only its first box
  is trusted (71-73%), may be noisier than Grounding DINO's. It is a contract change, not a pure bug fix, so it gets a
  fixture test (`spatial_containment.png`-style sizes) before it ships. Leave `_spatial_measurement` as it is: it
  mirrors `spatial_relations`, which stays on Grounding DINO + SAM2 by measurement.
- **Step 1A is fine.** If memory is short, the fallback loads Moondream2 (4.5 GB peak), and that can be refused in
  turn, which becomes a tool error. Acceptable; log which model answered.
- **The seed (invariant 3).** `answer()` defaults to `VQA_SEED`, but the VQA gates seeded with the question index
  (`vqa_premise`, `vqa_round23`). Gates V1-V5 should call the server's own functions at `VQA_SEED`, as
  `vqa_round22_textvqa.py` did. Then the configuration measured is the one that ships, and parity is exact.
- **The no-Chinese-models policy.** Routing more questions to `ocr_pages` and reading `text_regions` both go through
  the engine-agnostic interface. That is not tuning PP-OCRv6, so it is allowed.

### R.3 Method problems

a. **Separate the two mechanisms in Track 2.** On phase 11, OCR context added only +0.7 overall (`plain` 81.7 vs
   `ocr_text` 82.4), and on round 22 nothing. Round 23's big routed gain is on `no_text`, where the suffix ("answer
   that there is none") is the likelier cause. Add arm **`route_suffix_only`**: newly routed questions get
   `VQA_TEXT_SUFFIX` with no OCR context. If it matches `route_only`, the cheaper fix wins: no ~1.8 s OCR on another
   ~20% of questions.

b. **Fold `route_240` into another arm, or drop it.** It targets 2 + 2 misses on round 23 and 5 + 2 on phase 11.
   The screen cannot tell that from noise.

c. **Count the Wh-premise loss on every unrouted Wh question**, text included (R.1). Gate V2A as written only checks
   colour + object + position.

d. **Track 3B (concise phrasing) mostly optimises a metric.** On round 22, contained was 76.6 → 76.0 under `short`.
   The gain is TextVQA's exact match only. A calling agent can already ask for a short answer in its own question.
   See R.6 Q1.

e. **Gate V4 is not a test.** Phase 12's `check_consistency` questions are TextVQA (phase 2) plus phase 11, and the
   Step 5A rule was shaped on phase 11. Fix the rule using **real** control answers on round 23 (seen; ~892 LFM calls,
   ~75 min CPU), then score it once on round 29. Run the control question there in the same pass as Gate V5 (+~75 min).

f. **The trajectory (Section 3) adds Track 6's scorer fixes as gains.** Remove that row from the trajectory and
   report scorer-audit flips separately. The "not" fix also needs care. A refusal regex is fine, but adding a bare
   "not" to `_NO_TEXT` would credit "the sign is not red, it is blue" on a false-premise colour question.

g. **Round 29's questions come from the same worker and brief as round 23's.** The regex was fitted to that worker's
   phrasing, so the confirmation may run optimistic. Two mitigations:
   - Don't put the regex's terms in the brief.
   - Draw a slice (~15%) of round 29's questions some other way: written by the user, or by Claude from the
     image's labels without seeing the regex.

   Use freshly sourced images, or round 24's. Not round 25: it is a blur set, not generalist.

h. **Track 4's tiling has no server path.** `processor_for("tiles4")` belongs to VisionBench's `OnnxLfm`. The server's
   `LfmOnnxCaptioner` builds one processor with `do_image_splitting=False`. Tiling needs a wrapper change, a
   `MODEL_MEMORY_GB` check, and a measured trigger rate (the "< 12%" is unmeasured). Round 23 has only 3 T2 layout
   misses to aim the anchor idea at.

### R.4 Revised order

1. **Track 1 (no gate; unit tests + parity).** In this order:
   - 1A.
   - 1B: share the reading between answer and control only.
   - 1C: the `question.py` fixes, plus aligning the cross-check count per Q3, with the count parity re-run.
   - 1C(2), with its fixture test.
   - 1D: reuse the hand detections, 8 → 7 passes.

   Update the tool description and CLAUDE.md where these change what callers see. *(Sonnet 5.5, new session +
   handoff; it is well specified. The pre-commit hook loads real models, so commit only while no benchmark is
   timing.)*
2. **Gate V1 screen on round 23 + phase 11, seen.** Four arms:
   - `route_only`
   - `route_suffix_only`
   - `route_item_suffix` (the wider "text or an item" suffix)
   - `route_item_suffix_verbatim`

   Rules go in the docstring before any run. Live calls only where a prompt changes (~2 h CPU across the arms). Queue
   it for when the user is away. *(Opus writes the rules; the run stays in that session.)*
3. **Gate V2A screen (Wh-premise) on round 23**, on top of V1's pick, with loss counted per R.3c. Drop or park V2B per Q1.
4. **Track 5 rule design** on real round 23 control answers. These can be cached in step 2's run if they are queued
   together.
5. **Round 29 built** (brief per R.3g). Gate V5's rules and the Track 5 rule are fixed before the questions exist,
   then the user's 40-question blind check (port 8785 is free).
6. **Gate V5 once:** the V1/V2 pick against shipped, plus Track 5's scoring from the same run. Ship, with a parity
   check of 8 of 8.
7. **Track 4** only if V5 leaves the text kinds (screenshots and charts, ~71%) clearly behind. Needs a new round.
8. **Track 6, separately, at any time.** It changes no model.

### Progress (2026-10-31)

- **Step 1, Track 1: done.** Commits `6b46119..63bbf7f`, pushed. The noun-like verbs (cross, form, border, rest, lead,
  touch, hold, make, lie) that cut compounds ("red cross symbols" -> "red") were fixed 2026-11-01: `_COUNT_VERBS` end
  the noun only before a preposition, article or quantifier (`_AFTER_VERB`) or the question's end.
- **Control answers cached** for step 4: VisionBench `benchmarks/vqa_control_cache.py`, 892 rows in
  `results/vqa_round23_control_rows.jsonl`.
- **Step 2, Gate V1: done** (VisionBench `benchmarks/vqa_gate_v1.py`, `results/vqa_gate_v1.json`; seen sets). All
  four arms were eligible. Pick: **`route_item_suffix_verbatim`**, pooled +3.82 (CI 2.77 to 4.92; round 23 78.5 ->
  84.4%, phase 11 82.4 -> 84.7%). It beat `route_only` (+3.25) by 0.57, just past the 0.5 tie rule.
- **Step 3, Gate V2A: no arm eligible, so no Wh suffix** (VisionBench `benchmarks/vqa_gate_v2a.py`,
  `results/vqa_gate_v2a.json`; on top of V1's pick; 898 live answers, 0 errors). Both arms raised round 23's
  `no_object` 84.7 -> 95.8% (+11.1). Both lost on the other unrouted Wh questions (402; limit -1.5): `wh_premise`
  343 -> 325 (-4.5), `wh_premise_any` 343 -> 334 (-2.2). Both had a type with 30+ questions losing more than 3:
  `object` -6.3 on round 23, and -5.5 on phase 11. `wh_premise` also failed phase 11 overall (-0.8; limit -0.5).
  Pooled: -0.24 (CI -0.91 to 0.38) and +0.19 (CI -0.38 to 0.81). The suffix catches false premises, but it makes
  the model doubt objects that are there. **V5 tests `route_item_suffix_verbatim` alone.**
- **Step 4, Track 5 rule: designed and fixed** (VisionBench `benchmarks/vqa_consistency_rule.py`, `300dfc2`;
  `results/vqa_consistency_rule.json`; round 23 seen, no model run). The shipped `_vqa_consistency` carries almost no
  signal on LFM: it flags 42.2%, stays quiet on 58.9% of right answers, error ratio 1.16x. The control answers in a
  word or two, so exact match and containment fail, and "agreed defaults" are LFM's most accurate answers. The new
  rule compares by question kind: polarity, number, absent vs present, else half of the shorter answer's content
  words in the longer. Round 23: flagged 11.8%, quiet on right 93.9%, catches wrong 32.3%, unflagged accuracy +5.0,
  error ratio 3.57x (image halves 3.20x / 4.05x). Ceiling: when the primary is wrong, the control is right only 28 of
  192 times. **Gate V5's Track 5 bars** (round 29, once): quiet on right >= 90%, catches wrong >= 25%, error ratio
  >= 2.5x, unflagged accuracy >= overall + 3.0. The server change waits on that.
- **Step 5, round 29: built and checked** (2026-11-01). Slice a: Gemini 3.8 Flash, 191 images; slice b: Gemini 4
  Argon, 39 images (round 24's kept images; VisionBench `benchmarks/mixed_round24/round29_slices.json`). The user's
  check (`round29_check_serve.py`, 40 cards, 10 from slice b): premise 19 of 20 false, ordinary 18 of 20 right; the
  3 questions marked b dropped, 917 left.
- **Step 6, Gate V5: both pass** (VisionBench `benchmarks/vqa_gate_v5.py`, `results/vqa_gate_v5.json`, 2,537 live
  answers, 0 errors). **Routing:** `route_item_suffix_verbatim` 80.2 vs shipped 76.8% (+3.4, CI 1.6 to 5.1; 50 gained,
  19 lost); premise 59.4 -> 71.2% (`no_object` 46.1 -> 60.9, `no_text` 72.8 -> 81.6); text +2.5; worst type
  position -2.2 (n 46); slice a +3.7, slice b +1.9; routed 30.4 -> 43.2%; median 6.1 -> 6.4 s per answer, OCR
  included. **Track 5** (candidate's pairs): flagged 12.1%, quiet on right 92.5%, catches wrong 30.8%, unflagged
  accuracy 84.4% (+4.2), error ratio 3.23x; the shipped rule on the same pairs 36.0 / 66.0 / 44.0 / +2.5 / 1.40x. Weak
  spot: on `no_object` the rule's flags carry no signal (ratio 0.84x). Next: the user's go, then the server change
  (both), parity 8 of 8, unit tests, tool description and CLAUDE.md.
- **Shipped 2026-11-02** (364f933): both, with parity 1,834 of 1,834 prompts and agreement decisions and 8 of 8 live
  answers (VisionBench `vqa_v5_parity.py`). Pre-V5 code frozen in VisionBench `vqa_shipped_20261024.py`.
- **Track 6 done 2026-11-02** (VisionBench 448afc8): of 3,982 cached answers the three grader fixes flipped 32; the
  user's blind review applied single-letter (2 of 2) and adjective forms (3 of 3); "not"/"cannot" refusals failed
  (22 of 26: "cannot identify the car" accepts the car).
- **Track 8 (new, 2026-11-02): presence check before an unrouted Wh question.** The largest gap left: round 29's
  unrouted What/Which/Who/Where `no_object` questions, 30 of 68 right. V2A's suffix fixed those and broke real
  objects, so here it (or a fixed "none" reply) applies only when LFM first says the thing asked about is not in the
  image. Gate V6 screen on the three seen sets (VisionBench `benchmarks/vqa_gate_v6.py`, rules in its docstring:
  changed `no_object` +8.0, other changed questions -1.0 at most, each set -0.3, no type -3.0); a pass goes to a
  fresh round 30 (Gate V7). ~1,500 short check calls + up to 300 answers, ~1.5 h CPU.
- **Gate V6 ran 2026-11-02: `A_suffix` picked** (VisionBench f3ad8ff, `results/vqa_gate_v6.json`; 0 errors). Check A
  ("Is the thing this question asks about in the image? ... Answer yes or no.") said no to 94.8% of the 116 changed
  premise questions and 4.3% of the 633 others. Changed `no_object` 63 -> 105 of 115 (+36.5), other changed questions
  542 -> 540 (-0.3), pooled 83.3 -> 84.65% (+1.36, CI 0.93 to 1.83): round 23 +1.4, phase 11 0.0, round 29 +3.1;
  worst type colour -1.7 (round 23). `B_suffix` tied (+1.40) and lost the tie to A by order. Both fixed-reply arms
  failed (other changed -2.7 / -1.4, colour -5.9 / -5.1): when the check is wrong the suffix still lets the model
  answer. Adds ~3.9 s to every unrouted Wh question. Seen sets only; next is round 30 (fresh images) and Gate V7.

### R.5 Kept from the plan

The seen/unseen rule, the pre-registration rule, the Track 1 exception, never adding a suffix to `yes_no` or `count`
questions, never replacing the whole image with an unverified crop, Gate V5's four criteria (with the latency
including live OCR), and the scorer non-circularity rule.

### R.6 Decisions for the user (before Gate V1)

1. **Track 3B, concise answers:** (a) drop it, since callers can ask for a short answer themselves (recommended);
   (b) an opt-in parameter; or (c) screen it as the plan says.
2. **Track 4:** after Gate V5 and only if needed (recommended), or screen it alongside V1?
3. **Cross-check count (Step 1C):** the full `count_objects` pipeline (matches the tool, ~10-15 s more on a
   low-confidence count; recommended), or only `execute_detection` at `COUNT_BOX_THRESHOLD` with `drop_nested` (cheaper,
   but its counts can differ from `count_objects`')?
4. **Round 29 images:** freshly sourced by the worker (recommended) or round 24's pool? And should a ~15% question
   slice come from outside the worker (R.3g)?
5. **Track 5:** OK to spend ~75 min CPU caching real round 23 control answers before round 29 exists?

**User's answers (2026-10-30): yes to every recommendation.**
- Q1: Track 3B dropped.
- Q2: Track 4 after Gate V5, and only if needed.
- Q3: the cross-check count runs the full `count_objects` pipeline (shared helper, count parity re-run).
- Q4: round 29 on freshly sourced images, with a ~15% question slice written outside the worker.
- Q5: yes, but only once the CPU is free.

The cache script is VisionBench `benchmarks/vqa_control_cache.py`. Its rows go to
`results/vqa_round23_control_rows.jsonl`, with seed = question index to match the cached primary answers. It is
scheduled as Windows task "VisionBench vqa control cache" for 2026-10-31 03:00 local, via
`scratch/night_queue/run_vqa_control.ps1`. That script waits until no benchmark, harness or pytest process is
running (gives up after 6 h) and logs to `logs/vqa_control_cache.log` and `summary.txt`.

---

## 0. Methodological Rules, Dataset Hygiene & Prior-Experiment Ledger

Every worker executing this plan must follow five methodological rules established across `Counting_Improvement_Plan.md` (Section 0), `Caption_Improvement_Plan.md`, and `OCR-TextVQA_Improvement_Plan.md`:

### 0.1 Seen vs. Unseen Datasets (Strict Separation)

1. **Seen datasets (Screen & Tune ONLY — never cite as confirmation evidence):**
   - **Phase 11 (`C:\AI\Projects\VisionBench\benchmarks\datasets\sourced_annotations.json`):** 400 generalist images, 1,200 questions (`375 yes_no`, `341 text`, `164 count`, `155 color`, `76 position`, `53 no_text`, `36 object`). Cached LFM replies live in `benchmarks/results/vqa_premise_rows.jsonl`; cached `ocr`, `count_objects`, and `detect_objects` replies live in `reports/visionbench_report_20261020_184831.replies.jsonl`.
   - **Round 23 (`C:\AI\Projects\VisionBench-labels\benchmarks\mixed_round21\questions_round23.json`):** 892 questions on 223 Round 21 generalist images (`233 text`, `171 yes_no`, `118 color`, `118 no_object`, `105 no_text`, `80 object`, `65 count`, `2 position`). Cached LFM and Moondream2 replies live in `benchmarks/results/vqa_round23_rows.jsonl`; cached OCR readings live in `benchmarks/results/caption_ocr_r21.jsonl`.
   - **Round 22 TextVQA (`C:\AI\Projects\VisionBench-labels\benchmarks\textvqa_round22\`):** 154 TextVQA validation questions (`scratch/vqa_round22/rows.jsonl`, summary in `benchmarks/results/vqa_round22_textvqa.json`).
   - **Phase 2 (`benchmarks/datasets/textvqa_annotations_extended.json`) & Phase 12 (`run_visionbench.py:evaluate_flags`):** Seen benchmark suites.
2. **Unseen confirmation dataset (Required before shipping any prompt, routing, or arbitration change to `src/`):**
   - **Round 29 (`C:\AI\Projects\VisionBench-labels\benchmarks\vqa_round29\`):** A fresh blind VQA confirmation round (~225 images × 4 questions = ~900 questions: 3 ordinary questions + 1 false-premise `no_text` or `no_object` question per image).
   - **Pre-registration rule:** The confirmation gate script (`C:\AI\Projects\VisionBench\benchmarks\vqa_round29.py`) and its pass/fail numbers must be committed in its module docstring **before** the Round 29 questions are generated or scored, and a 40-question blind user spot-check (`benchmarks/round29_check_serve.py` on port `8785`) must pass before `vqa_round29.py run` is executed.
   - **Exception for pure bug fixes / contract alignment (Track 1):** Deterministic bug fixes that do not alter default `query_image(src, question)` prompts or answers on healthy runs (e.g., catching `MemoryError` on fallback, deduplicating redundant `ocr_pages` calls in `check_consistency=True`, fixing `_vqa_cross_check` detector threshold drift) may ship immediately with unit tests in `tests/`.

### 0.2 Prior-Experiment Ledger ("Do Not Repeat")

| Date | Script / Gate | What Was Tested | Outcome & Lesson |
| :--- | :--- | :--- | :--- |
| 2026-10-16 | `benchmarks/vqa_ocr.py` (Gate Q1) | Prepending `ocr` text to **Moondream2** on Phase 2 TextVQA (`n=100`) and Phase 11 `text` (`n=341`) | `+5.0 pts` contained on TextVQA (`60.0% -> 65.0%`), `+5.9 pts` on Phase 11 `text` (`58.1% -> 64.0%`). Superseded by LFM2.5-VL-3B on 2026-10-24 (`vqa_premise.py`, `vqa_round23.py`). |
| 2026-10-18 | `benchmarks/region_inspect_gate.py` (Gate R) | Cropping a target region via Grounding DINO + 2× LANCZOS upscale and passing **only the crop** to `query_image` on 116 Phase 11 target-phrase questions | **Failed (`-7.8 pts`, `74.1% -> 66.4%`; 9 helped, 18 hurt).** Grounding DINO's first-box precision (`~68%`) meant wrong crops blinded the VLM to the rest of the scene. **Never replace the full image with an unverified detector crop.** |
| 2026-10-23 | `benchmarks/vqa_premise.py` (`premise` arm) | Appending `PREMISE_SUFFIX = "\nIf the question assumes something that is not in the image, say so instead of guessing."` to **every** question on Phase 11 (`n=1,200`) | Lifted `no_text` from `75.5% -> 100.0%` (`+13 hits`), but **lost 25 hits (`-2.1 pts`) on ordinary questions** (`count` `79.9% -> 76.2%`, `text` `62.5% -> 59.5%`, `yes_no` `97.6% -> 94.7%`), dropping overall from `81.7%` (`plain`) to `80.7%`. **Never append an unconditional skepticism suffix to all ordinary questions** (especially `yes_no` and `count`). |
| 2026-10-23 | `benchmarks/vqa_premise.py` (`ocr_text_t4` arm) | Running `LfmOnnxCaptioner` with `do_image_splitting=True, max_tiles=4` on all 325 `VQA_TEXT_QUESTION`-routed questions on Phase 11 | Gained only `+0.4 pts` overall (`82.4% -> 82.8%`; `text` `61.6% -> 62.8%` with 17 wins and 12 losses) while **nearly tripling routed latency** (`4.50 s -> 12.50 s` median, `5.69 s -> 15.70 s` p90 on routed questions). Failed the `+1.0 pt` replacement gate. **Do not enable unconditional 4-tile splitting across all routed text questions.** |
| 2026-10-24 | `benchmarks/vqa_round22_textvqa.py` (`ocr_all`, `short`) | Tested `shipped`, `plain`, `ocr_all`, and `short` (`+ "\nAnswer with a single word or phrase."`) on 154 unseen TextVQA questions | `shipped` scored `52.4%` VQA exact match (`76.6%` contained) because only `34/154` (22%) questions matched `VQA_TEXT_QUESTION` and unrouted answers were multi-word sentences. `ocr_all` jumped exact match to `72.5%` (`+20.1 pts`); `short` jumped exact match to **`76.6%` (`+24.2 pts`, `3.99 s`)**. Not yet tested against Phase 11 / Round 23 containment grading. |
| 2026-10-05 | `benchmarks/decomposed_vqa_eval.py` & `vqa_generated_clean_control.py` | Cropped face/hand VQA defect probing with Moondream2 (`structured_analysis` research) | Found that VLM "deformed" calls on generated faces were confounded by generation style rather than true structural defects; led to the DINOv2 linear probe (`defect_probe.py`) shipped in `inspection.py`. |

### 0.3 Architectural & Policy Invariants

1. **Torch-free eager import boundary (`tests/test_question.py`, `CLAUDE.md`):** `src/fusion_vision_mcp/constants.py`, `src/fusion_vision_mcp/analysis.py`, `src/fusion_vision_mcp/question.py`, and `src/fusion_vision_mcp/tools/vqa.py` must **never** import `torch`, `onnxruntime`, `transformers`, or any model wrapper at module top level. All shared regexes, thresholds, and prompt constants belong in `constants.py`.
2. **Non-Chinese model preference policy (`CLAUDE.md`):** `LFM2.5-VL-3B` (Liquid AI, US), `SAM 3` / `SAM 2.1` (Meta, US), `DINOv2` (Meta, US), `CLIP` (OpenAI, US), and `Florence-2` (Microsoft, US) are US-origin and permanent. `PP-OCRv6` (`ocr_pages`) and `Grounding DINO` (`app.counter`) are Chinese-origin; any `query_image` integration with them must go through the existing engine-portable interfaces (`ocr_pages(app, images)` returning `{text, text_regions}` and `execute_detection(app, images, ...)` returning `{bboxes, scores, count}`) and must never hardcode engine-specific internals.
3. **Determinism (`VQA_SEED = 0`):** Every `app.answerer.answer(images, prompts, seed=VQA_SEED)` call in `analysis.py` must remain deterministic with `seed=VQA_SEED` (`0`).

---

## 1. Current Architecture & Quantitative Failure Diagnosis (Step 0 Cache Audit)

### 1.1 Current `query_image` Call Graph

1. **Default VQA path (`tools/vqa.py:118–119, 169–170` → `analysis.py:911–929`):**
   - `query_image(ctx, src, question, check_consistency=False, structured_analysis=False)` loads `images` via `get_images(src)` and calls `vqa_answers(app, images, question)`.
   - `vqa_answers` checks `app.answerer` (`LfmOnnxCaptioner` on CPU, `LiquidAI/LFM2.5-VL-3B-ONNX` rev `020f931`, `do_image_splitting=False` -> single image of `<= 256` visual tokens, `VQA_MAX_NEW_TOKENS = 128`, `VQA_SEED = 0`, `DECODING = {"temperature": 0.1, "min_p": 0.15, "repeat_penalty": 1.05}`).
   - If `question` matches `_TEXT_QUESTION_RE` (`re.compile(VQA_TEXT_QUESTION, re.IGNORECASE)` in `analysis.py:898`), `vqa_answers` runs `ocr_pages(app, images)` (`PP-OCRv6` via RapidOCR) and wraps each image's prompt via `lfm_vqa_prompt(question, page["text"])` (`analysis.py:901–908`):
     - When `ocr_text` is empty: `"An OCR pass found no readable text in this image.\n" + question + VQA_TEXT_SUFFIX`
     - When `ocr_text` is non-empty (up to `VQA_OCR_WORDS = 160` words joined with `" / "`): `f'Text read from the image by OCR (it may be incomplete or misread): "{quoted}"\n' + question + VQA_TEXT_SUFFIX`
     - Where `VQA_TEXT_SUFFIX = "\nIf the question asks about text that is not in the image, answer that there is none."`
   - If `question` does **not** match `_TEXT_QUESTION_RE`, `vqa_answers` sends `[question] * len(images)` verbatim with no OCR call and no suffix.
2. **Consistency & Cross-Check path (`tools/vqa.py:172–190` → `analysis.py:180–279`):**
   - When `check_consistency=True`, `query_image` constructs `control_question = f"Looking carefully at this image, answer precisely: {question}"`, calls `vqa_answers(app, images, control_question)`, and passes `(answer, control_answer)` to `_vqa_consistency(answer, control_answer)`.
   - If `entry["confidence"] == "low"`, `query_image` calls `_vqa_cross_check(app, image, question)`, which classifies the question via `question.classify(question)` (`SPATIAL`, `COUNT`, `OCR`, `SIZE`), extracts object names via `question.names_for(category, question)`, and runs a secondary tool measurement.
3. **Structured Analysis path (`tools/vqa.py:121–167` → `inspection.py:345–417`):**
   - When `structured_analysis=True`, `query_image` runs `vqa_answers`, then `analyze_inspection(app, image, [answer])` (`_check_anatomy` + `_check_defect_probe` with `DINOv2-large`), `analyze_metadata_anomalies(image)`, and `app.ocr_specialist.readtext(image)` (`EasyOCR` confidence `< 0.20` check).

### 1.2 Offline Cache Audit: Round 23 (`n = 892`) & Phase 11 (`n = 1,200`)

Direct inspection of `benchmarks/results/vqa_round23_rows.jsonl` (`n=892`) and `benchmarks/results/vqa_premise_rows.jsonl` (`n=1,200`, `ocr_text` arm) reveals the exact accuracy by question type and `_TEXT_QUESTION_RE` routing status:

| Question Type | Round 23 Total (`n=892`) | Round 23 Routed (`_TEXT_QUESTION_RE` Match) | Round 23 Unrouted (`Plain`) | Phase 11 Total (`n=1,200`) | Phase 11 Routed (`_TEXT_QUESTION_RE` Match) | Phase 11 Unrouted (`Plain`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`yes_no`** | **95.3%** (`163/171`) | `100.0%` (`16/16`) | `94.8%` (`147/155`) | **97.6%** (`366/375`) | `94.1%` (`32/34`) | `97.9%` (`334/341`) |
| **`color`** | **95.8%** (`113/118`) | `87.5%` (`7/8`) | `96.4%` (`106/110`) | **87.7%** (`136/155`) | `88.9%` (`16/18`) | `87.6%` (`120/137`) |
| **`object`** | **81.2%** (`65/80`) | `50.0%` (`1/2`) | `82.1%` (`64/78`) | **86.1%** (`31/36`) | `n/a` (`0/0`) | `86.1%` (`31/36`) |
| **`position`** | **100.0%** (`2/2`) | `n/a` (`0/0`) | `100.0%` (`2/2`) | **81.6%** (`62/76`) | `92.3%` (`12/13`) | `79.4%` (`50/63`) |
| **`count`** | **75.4%** (`49/65`) | `87.5%` (`7/8`) | `73.7%` (`42/57`) | **79.9%** (`131/164`) | `85.7%` (`12/14`) | `79.3%` (`119/150`) |
| **`text`** | **68.7%** (`160/233`) | **`73.0%`** (`119/163`) | **`58.6%`** (`41/70`) | **61.6%** (`210/341`) | **`63.2%`** (`122/193`) | **`59.5%`** (`88/148`) |
| **`no_text`** (premise) | **65.7%** (`69/105`) | **`89.0%`** (`65/73`) | **`12.5%`** (`4/32`) | **100.0%** (`53/53`) | **`100.0%`** (`53/53`) | `n/a` (`0/0`) |
| **`no_object`** (premise) | **66.9%** (`79/118`) | `75.0%` (`3/4`) | `66.7%` (`76/114`) | `n/a` (`0/0`) | `n/a` (`0/0`) | `n/a` (`0/0`) |
| **Overall** | **78.5%** (`700/892`) | **81.3%** (`218/268`) | **77.2%** (`482/624`) | **82.4%** (`989/1200`) | **76.0%** (`247/325`) | **84.8%** (`742/875`) |

And by image domain (`kind`) on Round 23 (`n=892`):
- **Artwork and paintings:** `91.3%` (`84/92`)
- **AI-generated images:** `91.2%` (`62/68`)
- **Documents:** `82.3%` (`79/96`)
- **Illustrations, cartoons and comics:** `81.5%` (`75/92`)
- **Signage, logos and packaging:** `81.2%` (`78/96`)
- **Game UI and HUD:** `80.3%` (`61/76`)
- **Charts, diagrams and infographics:** `70.8%` (`68/96`)
- **App and web screenshots:** `70.7%` (`65/92`)
- **Everyday photos:** `69.6%` (`128/184` — dragged down by unrouted `no_text` and `no_object` false-premise questions)

### 1.3 The Five Root Failure Modes

#### Failure Mode 1: Narrow `_TEXT_QUESTION_RE` (`constants.VQA_TEXT_QUESTION`) Fails to Route 30.5% of `no_text`, 30.0%–43.4% of `text`, and 77.9% of TextVQA Questions
- **The smoking gun in Round 23 `no_text` (`n=105`):**
  - When `_TEXT_QUESTION_RE` matched (`73` questions), LFM scored **65 / 73 (89.0%)**.
  - When `_TEXT_QUESTION_RE` did **not** match (`32` questions), LFM scored **4 / 32 (12.5%)** — **28 misses out of 32 questions** (`14.6%` of all 192 misses in Round 23!).
  - Why did those 32 `no_text` questions miss `VQA_TEXT_QUESTION`? Because they asked about text/numbers/labels using words absent from the 23-word regex:
    - `"What Roman numerals are visible on the fireplace mantel clock?"` -> LFM hallucinated `I, II, III...`
    - `"What room number is painted above the doorway?"` -> LFM hallucinated `3`
    - `"What discount percentage is displayed on the yellow shelf tags?"` -> LFM hallucinated `15%`
    - `"What ship name is painted across the hull of the model ships?"` -> LFM hallucinated `"Bounty"`
    - `"What speed limit number is shown on the road sign?"` -> LFM hallucinated `25`
    - `"What hourly parking rate is listed on the green EXIT sign?"` -> LFM hallucinated `$2.00`
    - `"What manufacturer name is stamped on the black fire pit bowl?"` -> LFM hallucinated `"Kamado Joe"`
    - `"What serial number is engraved on the crest of the sculpted head?"` -> LFM hallucinated `1901`
    - `"What player username is displayed on the screen?"` -> LFM hallucinated `"Münzen"`
    - `"What social media handle is listed on the membership form?"` -> LFM hallucinated `@parm_mexico`
    - `"What email address is listed on the certificate?"` -> LFM hallucinated `p.kgf@poland`
- **The smoking gun in `text` questions (`n=233` on Round 23; `n=341` on Phase 11):**
  - Cross-tabulating all `73` Round 23 `text` misses and all `131` Phase 11 `text` misses against the cached `ocr` output (`caption_ocr_r21.jsonl` and `visionbench_report_20261020_184831.replies.jsonl`):
    - **Bucket `T1a` (Unrouted `text` question, ground-truth answer IS in the first 160 words of `ocr`):** **15 misses** on Round 23 and **36 misses** on Phase 11! Examples from Round 23 where `ocr` already had the exact answer, but `query_image` never called `ocr_pages` because `_TEXT_QUESTION_RE` didn't match:
      - `r21_03d133c2af`: *"What telephone number is painted on the side of the wooden honesty stall?"* (`ocr` had `'01872 273482'`; unrouted LFM guessed `'01234 258844'`).
      - `r21_646e5bb1d8`: *"What number is underlined in red in the welcome section?"* (`ocr` had `'350.029'`; unrouted LFM guessed `'2003'`).
      - `r21_1165250825`: *"What is the top spelling suggestion in the context menu?"* (`ocr` had `'Bha'`; unrouted LFM guessed `'Bhha'`).
      - `r21_eb7947675f`: *"Which top menu is currently opened in LibreOffice?"* (`ocr` had `'Inserir'`; unrouted LFM guessed `'File'`).
      - `r21_ce1322b797`: *"What layer name is currently selected in the Layers panel on the right?"* (`ocr` had `'Layer 6'`; unrouted LFM guessed `'paint'`).
      - `r21_e91c7f48f3`: *"What is the name of the file open in the top editor pane?"* (`ocr` had `'TrackModel.scala'`; unrouted LFM guessed `'trackmodel.vs'`).
      - `r21_9c93da58be`: *"What C source file name is displayed in the active editor tab?"* (`ocr` had `'sample.c'`; unrouted LFM guessed `'main.c'`).
    - **Bucket `T1b` (Unrouted `text` question, ground-truth answer IS in `ocr` after word 160):** **2 misses** on Round 23 and **5 misses** on Phase 11.
    - **Bucket `T1c` (Unrouted `text` question, answer not in `ocr`):** **12 misses** on Round 23 and **19 misses** on Phase 11.
  - Even on `count` questions in Round 23 (`16` misses total), **5 of the 16 `count` misses** are actually unrouted OCR number-reading questions (`"What number labels the specimen shown in the top left corner?"` — note `"labels"` plural/verb missed `\b(label|labell?ed)\b`! — `"What page number is specified after 'To face page'?"`, `"What is the cost of the upgrade in solar currency?"`).

#### Failure Mode 2: Routed `text` Misses (`T2`, `T3`, `T4`) & Phase 2 TextVQA Verbosity Penalty (`-24.2 pts`)
- **Bucket `T2` (Routed `text` question, ground-truth answer IS in the first 160 words of `ocr`, yet LFM still missed — 14 misses on Round 23, 29 misses on Phase 11):**
  Inspecting all 14 `T2` misses on Round 23 shows three distinct sub-causes:
  1. **LFM overrides clean OCR spelling with a blurry 256-token thumbnail misread (5/14 on R23):** Because `lfm_vqa_prompt` prefaces the OCR reading with `Text read from the image by OCR (it may be incomplete or misread): "..."`, LFM distrusts the exact OCR string and substitutes its own low-resolution visual misread:
     - `r21_84624ff44a`: OCR read `'Strengthened trap'`, LFM answered `'Strengthened Irag'`.
     - `r21_4f44857cb7`: OCR read `'2002 / 2022 / Strive'`, LFM answered `'Strike'`.
     - `r21_d4c9090081`: OCR read `'24V / 10A SMPS POWER SUPPLY CIRCUIT'`, LFM answered `'24V / 16A SMPS POWER SUPPLY CIRCUIT'`.
     - `r21_b929ad2133`: OCR read `'ARCH.IR.S.RAVESTEYN'`, LFM answered `'Ravestijn'`.
     - `r21_c9370cf9a9`: OCR read `'GAES'`, LFM answered `'There is none.'`.
  2. **Partial / abbreviated label extraction (4/14 on R23):** LFM drops part of a compound proper name or title even though the full phrase is in the OCR reading:
     - `r21_6a5903ca90`: want `'Baby Boem'`, got `'Boem'` (OCR: `'Lex / Baby / Boem / Groente'`).
     - `r21_83a186ffec`: want `'BrOffice Impress'`, got `'BrOffice'` (OCR: `'Sem título 1 - BrOffice Impress'`).
     - `r21_66403bfbfc`: want `'REPUBLIC OF ARMENIA'`, got `'Armenia'` (OCR: `'SPECIMEN / REPUBLIC OF ARMENIA / ... / PASSPORT'`).
     - `r21_9f480e8618`: want `'Balt./Wash. Pkwy'`, got `'Balt.'` (OCR: `'-Balt./Wash. Pkwy / imore'`).
  3. **Loss of spatial / line-grouping structure when flattening `ocr_pages` (3/14 on R23):** `ocr_pages(app, images)` returns `text_regions: [{text, box}, ...]`, but `vqa_answers` discards `text_regions` and joins all lines with `" / "`. When a question asks `"What word appears directly under 'MEDICAL' on the sign?"` (`r21_d5a3add058`) or `"What number is printed on the left side of the blue banner?"` (`r21_0ad3ff48c4`), the flat `" / "` string has no vertical/horizontal layout cues.
- **Bucket `T3` (Routed `text` question, ground-truth answer truncated past `VQA_OCR_WORDS = 160` — 2 misses on Round 23, 2 misses on Phase 11):** Increasing `VQA_OCR_WORDS` from `160` to `240` recovers dense-page text at negligible CPU prefill cost (`~60` extra text tokens).
- **Bucket `T4` (Routed `text` question, ground-truth answer absent from `ocr` — 28 misses on Round 23, 40 misses on Phase 11):** Caused by tiny UI/chart text, stylized/curved lettering, or non-Latin characters (`予定.txt`, `1 км`, `Générique de fin`, `título`) where `PP-OCRv6` missed a character and single-tile LFM (`<= 256` visual tokens) lacked resolution.
- **Phase 2 / Round 22 TextVQA Verbosity Penalty (`52.4%` exact match vs. `76.6%` contained):** On Phase 2 (`run_visionbench.py:evaluate_query_image`), `vqa_accuracy` grades normalized exact match (`normalize_answer(pred) == normalize_answer(gt)`). Because LFM answers unrouted questions with full sentences (median `12–40` words on `color`, `count`, `object`, `no_object`), `shipped` loses **24.2 percentage points** purely from sentence framing (`"The brand on the laptop is Dell."` scores `0.0` exact match against `"dell"`).

#### Failure Mode 3: `no_object` False-Premise Questions (`66.9%` on Round 23 — 39 Misses out of 118)
- On Round 23, `114` of `118` `no_object` questions did not match `_TEXT_QUESTION_RE`, so they received no OCR context and no false-premise escape instruction.
- Inspecting all 39 `no_object` misses on Round 23 reveals:
  - **25 of the 39 misses (64.1%)** ask about a **textual/brand/model/sign attribute of an absent object** (`"What brand of digital scale is sitting on the fruit counter?"`, `"What brand of microwave oven is on the counter?"`, `"What license plate number is on the car?"`, `"What make of sports car is shown in the thumbnail photos?"`, `"What brand of mouse is sitting on the drawing canvas?"`). Under our expanded two-branch `VQA_TEXT_QUESTION` regex (Track 2), **all 25 of these are routed**! However, current `VQA_TEXT_SUFFIX` only says `"If the question asks about text that is not in the image, answer that there is none."` — which fails to tell LFM to answer `"none"` when the *object/brand/item itself* is not in the image!
  - **4 of the 39 misses** are actually **correct refusals** by LFM (`"The cartoon doesn't show any automobiles competing in the race."`, `"The plate does not depict a laboratory microscope."`, `"The flowchart does not display any corporate company logo."`, `"The man in the image is not wearing any eyeglasses."`) that failed `grade_answer` only because `_NO_TEXT = {"no", "none", "nothing", "illegible", "unreadable"}` lacks `"not"` — and LFM phrased its refusal with `"does not"` / `"doesn't"` / `"not"` because it was not given `VQA_TEXT_SUFFIX` (`"answer that there is none"`)!
  - **The remaining 10 of 39 misses** are pure visual questions about an absent object (`"What color is the umbrella held by the customer?"`, `"What color is the bus arriving at the stop?"`, `"What animal is accompanying the men in the gallery?"`).

#### Failure Mode 4: Contract Drift in `_vqa_cross_check`, `_size_measurement`, and `question.py`
1. **`_vqa_cross_check` count drift (`src/fusion_vision_mcp/analysis.py:266–273`):**
   When `check_consistency=True` flags a `COUNT` question as `confidence="low"`, `_vqa_cross_check` executes:
   ```python
   result = app.counter.detect_objects([image], names[0])[0]
   ```
   This calls Grounding DINO directly at `DEFAULT_BOX_THRESHOLD = 0.15`! It completely bypasses `count_objects`'s shipped pipeline:
   - It misses `COUNT_BOX_THRESHOLD = 0.425` (so `0.15` floods the count with false-positive boxes).
   - It misses `execute_detection` (so flat images — charts, diagrams, screenshots, comics, artwork — are not routed to `SAM 3` and nested boxes are not filtered via `drop_nested`).
   - It misses `lfm_count_check` (natural photos count `2–9`) and `lfm_flat_check` (flat images count `>= 1`).
2. **`_size_measurement` detection drift (`src/fusion_vision_mcp/analysis.py:293`):**
   Calls `app.counter.detect_objects([image], object_name)[0]` directly instead of `execute_detection(app, [image], object_name)[0]`.
3. **`question.names_for(COUNT, question)` lexical bugs (`src/fusion_vision_mcp/question.py:55–61, 130–145`):**
   - `_COUNT_PATTERNS` only stops at auxiliaries/participles (`are|is|do|does|...|standing|sitting...`), not finite lexical verbs (`stand`, `lead`, `appear`, `flank`, `represent`, `make`, `surround`, `remain`, `rest`, `sit`, `lie`, `hang`). On Round 23, `names_for(COUNT, "How many matching pink vases stand on the mantelpiece?")` returns `['matching pink vases stand']` and `names_for(COUNT, "How many heraldic figures appear inside the top shield?")` returns `['heraldic figures appear']`.
   - `_STOP_WORDS` (`question.py:31–34`) strips `"image"` and `"photo"` unconditionally, so UI/screenshot count questions like `"How many image panels are shown side by side?"` or `"How many photo thumbnails are displayed along the right edge?"` get truncated to `""` (`None`).

#### Failure Mode 5: Uncalibrated `_vqa_consistency`, Triple-OCR Latency Bug, `MemoryError` Gap & `structured_analysis` Redundancy
1. **`_vqa_consistency` (`src/fusion_vision_mcp/analysis.py:180–225`) is uncalibrated for LFM2.5-VL-3B:**
   - Across all 892 Round 23 questions, LFM emitted an answer in `_DEFAULT_ANSWERS` (`{"none", "nothing", "yes", "no", ...}`) **124 times** (`109` on `yes_no`, `15` on `no_text`), and those 124 answers were **122 / 124 = 98.4% accurate** (`98.2%` on `yes_no`, `100.0%` on `no_text`)! Yet `_vqa_consistency` marks every agreed answer in `_DEFAULT_ANSWERS` as `both_default = True -> confidence = "low"`.
   - Meanwhile, on non-polar questions where LFM answers in sentences (`median 12–57 words`), `_vqa_consistency` requires `a == c or a in c or c in a` (one full sentence must be a verbatim substring of the other).
   - Simulating `_vqa_consistency` on the 1,200 Phase 11 LFM answer pairs shows it flags **44.4%** of all answers (`quiet_on_right_pct = 60.6%`), falsely flagging **382 right answers** (`78` right `yes_no` answers via `both_default` and `303` right prose answers via failed full-sentence substring matching).
2. **Triple-OCR call bug in `check_consistency=True` (`src/fusion_vision_mcp/tools/vqa.py:119, 175` & `analysis.py:275`):**
   - For any text question with `check_consistency=True`, `vqa_answers(app, images, question)` runs `ocr_pages(app, images)` once; `vqa_answers(app, images, control_question)` runs `ocr_pages(app, images)` a **second time** on the exact same `images`; and if `confidence == "low"` on a `question.OCR` query, `_vqa_cross_check` runs `ocr_pages(app, [image])` a **third time**!
3. **`MemoryError` fallback bug in `vqa_answers` (`src/fusion_vision_mcp/analysis.py:925`):**
   - `vqa_answers` catches `(OSError, ImportError)` when falling back from `app.answerer` to `app.vqa`, omitting `MemoryError` (`ModelMemoryError` inherits from `MemoryError` in `memory_manager.py`). Every other LFM caller in `analysis.py` (`lfm_count_check` at line 635, `lfm_flat_check` at line 656, `vqa_count_estimate` at line 676) catches `(OSError, ImportError, MemoryError)`.
4. **Redundant Grounding DINO passes in `structured_analysis=True` (`src/fusion_vision_mcp/inspection.py:165–177, 360–369`):**
   - `_check_anatomy` runs 6 Grounding DINO passes (`person`, `arm`, `leg`, `hand`, `head`, `finger`), even though `head` and `finger` counts are never used to emit observations or anomalies (retired 2026-10-06), and `_check_defect_probe` then runs Grounding DINO a **second time** for `hand` (plus `face`), totaling **8 Grounding DINO passes per image** instead of 5 (`person`, `arm`, `leg`, `hand`, `face`).

---

## 2. Track-by-Track Improvement Plan & Pre-Registered Gates

```
Track 1 (Bug Fixes & Contract Alignment in src/ + tests/)  ──► Ship immediately with pytest coverage
       │
       ▼
Track 2 (Surgical VQA_TEXT_QUESTION + OCR Prompt Grounding) ──► Gate V1 (Phase 11 + R23 + R22 screen)
       │
       ▼
Track 3 (Scoped False-Premise Guard + Concise Phrasing)     ──► Gate V2 (Phase 11 + R23 + R22 screen)
       │
       ▼
Track 4 (OCR-Anchor Region Crop / Selective Tile Escalation)──► Gate V3 (Dense UI/Chart/Text subset screen)
       │
       ▼
Track 5 (LFM-Calibrated check_consistency + Phase 12 Eval)  ──► Gate V4 (Phase 12 flag_quality + cross_check)
       │
       ▼
Track 6 & 7 (Unseen Round 29 Confirmation + Scorer Audit)   ──► Gate V5 (Blind Round 29 confirm before shipping Tracks 2–4)
```

---

### Track 1: Zero-Model-Cost Bug Fixes, Deduplication & Contract Alignment

Track 1 fixes four verified code bugs and contract drifts without altering default `query_image(src, question)` prompts or outputs on healthy runs.

#### Step 1A: `MemoryError` Fallback Parity in `vqa_answers`
- **File & Symbol:** [`vqa_answers`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L911-L929) in `src/fusion_vision_mcp/analysis.py` (line 925).
- **Change:** Update `except (OSError, ImportError):` to `except (OSError, ImportError, MemoryError):` so `ModelMemoryError(MemoryError)` raised by `MemoryManager` under commit pressure falls back cleanly to `app.vqa.query(images, question)` just like `lfm_count_check` (`analysis.py:635`), `lfm_flat_check` (`analysis.py:656`), and `vqa_count_estimate` (`analysis.py:676`).
- **Test:** Add `test_vqa_answers_falls_back_on_memory_error` in [`test_vqa_answers.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_vqa_answers.py).

#### Step 1B: Deduplicate Redundant `ocr_pages` Calls in `check_consistency=True`
- **Files & Symbols:** [`vqa_answers`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L911-L929) and [`_vqa_cross_check`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L239-L279) in `src/fusion_vision_mcp/analysis.py`; [`query_image`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/vqa.py#L118-L190) in `src/fusion_vision_mcp/tools/vqa.py`.
- **Change:**
  1. Add an optional keyword parameter `ocr_readings: Sequence[dict[str, Any]] | None = None` to `vqa_answers(app, images, question, *, ocr_readings=None)` and `ocr_reading: dict[str, Any] | None = None` to `_vqa_cross_check(app, image, question_text, *, ocr_reading=None)`.
  2. In `tools/vqa.py:query_image`, when `_TEXT_QUESTION_RE.search(question)` is true (or when `check_consistency=True` needs OCR), compute `ocr_readings = ocr_pages(app, images)` **once** and pass `ocr_readings=ocr_readings` into `vqa_answers(app, images, question)`, `vqa_answers(app, images, control_question)`, and `ocr_reading=ocr_readings[i]` into `_vqa_cross_check`.
- **Impact:** Cuts `ocr_pages` execution on `query_image(..., check_consistency=True)` from 2–3 passes per image down to **1 pass per image** (saving `~3.5–5.3 s` on CPU per call).
- **Test:** Add `test_query_image_check_consistency_runs_ocr_once` in [`test_vqa_answers.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_vqa_answers.py) asserting `ocr_pages` is called once across `answer`, `control_answer`, and `cross_check`.

#### Step 1C: Fix Contract Drift in `_vqa_cross_check`, `_size_measurement`, and `question.py`
- **Files & Symbols:**
  - [`_vqa_cross_check`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L266-L273) and [`_size_measurement`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L286-L310) in `src/fusion_vision_mcp/analysis.py`.
  - [`_COUNT_PATTERNS`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/question.py#L55-L61) and [`_clean_span`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/question.py#L130-L152) in `src/fusion_vision_mcp/question.py`.
- **Changes:**
  1. **Align `_vqa_cross_check` (`question.COUNT`) with `count_objects`:** Instead of calling `app.counter.detect_objects([image], names[0])[0]` at `0.15` box threshold, run the shipped `count_objects` core measurement (`execute_detection(app, [image], names[0], box_threshold=COUNT_BOX_THRESHOLD)[0]`, plus `lfm_flat_check` on flat images and `lfm_count_check` on natural photos in `COUNT_LFM_CHECK_MIN..COUNT_LFM_CHECK_MAX` when `app.answerer` is available — extracted into a shared helper `measure_object_count(app, image, object_name)` in `analysis.py` used by both `count_objects` and `_vqa_cross_check`).
  2. **Align `_size_measurement` (`question.SIZE`) with `detect_objects`:** Replace `app.counter.detect_objects([image], object_name)[0]` at `analysis.py:293` with `execute_detection(app, [image], object_name)[0]` so flat images route to `SAM 3` and nested duplicate boxes are dropped before computing extremum area.
  3. **Fix verb-swallowing and UI-noun stripping in `question.py`:**
     - Add finite verbs (`stand|stands|lead|leads|appear|appears|flank|flanks|represent|represents|make|makes|surround|surrounds|remain|remains|rest|rests|sit|sits|lie|lies|hang|hangs|wear|wears|hold|holds|carry|carries|form|forms|touch|touches|cross|crosses|overlap|overlaps|border|borders`) to the right-hand boundary of `_COUNT_PATTERNS` (`question.py:57`).
     - In `_clean_span` (`question.py:137`), only treat `"image"` / `"photo"` / `"picture"` as a stop-word boundary when it is the final word or preceded by `"in the"` / `"of the"`, not when it modifies a UI head noun (`"image panels"`, `"image files"`, `"photo thumbnails"`, `"picture frames"`).
- **Tests:** Update [`test_question.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_question.py) and [`test_crosscheck.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_crosscheck.py) to cover finite verbs in `"How many ..."` questions, UI compound nouns, and `COUNT_BOX_THRESHOLD` / `execute_detection` routing in `_vqa_cross_check` and `_size_measurement`.

#### Step 1D: Eliminate Redundant Grounding DINO Passes in `structured_analysis=True`
- **File & Symbols:** [`_check_anatomy`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/inspection.py#L156-L236), [`_check_defect_probe`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/inspection.py#L316-L342), and [`analyze_inspection`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/inspection.py#L345-L417) in `src/fusion_vision_mcp/inspection.py`.
- **Change:**
  1. Remove `"head"` and `"finger"` from `_ANATOMY_PARTS` (`inspection.py:165`), since `_check_anatomy` only checks `_PART_THRESHOLDS = {"arm": 2, "leg": 2, "hand": 2}` and discards `"head"` and `"finger"` boxes completely.
  2. Pass the already-detected `hand` result dict (`part_results["hand"]`) from `_check_anatomy` into `_check_defect_probe(app, image, observations, anomalies, hand_detections=...)` so `_check_defect_probe` only needs to run `detector.detect_objects([image], "face", adaptive_threshold=True)` instead of re-detecting `"hand"`.
- **Impact:** Reduces Grounding DINO passes in `query_image(..., structured_analysis=True)` from **8 passes per image down to 5 passes per image** (`~37.5%` faster detector stage, saving `~18 s` on CPU per image) with **bit-identical** `observations` and `anomalies`.
- **Test:** Add unit test in `tests/test_inspection.py` verifying `detect_objects` is called for `("person", "arm", "leg", "hand", "face")` (5 calls, not 8) and returns identical `observations` and `anomalies`.

---

### Track 2: Surgical Expansion of `VQA_TEXT_QUESTION` & OCR Context Formatting (Gate V1)

Track 2 targets **Failure Mode 1** (`28` unrouted `no_text` misses + `17` unrouted `text` misses with answer in OCR + `5` unrouted OCR-number `count` misses on Round 23; `41` unrouted `text` misses with answer in OCR on Phase 11) and **Failure Mode 2 (`T2` & `T3` routed text misses)**.

#### Step 2A: Two-Branch Linguistic Expansion of `VQA_TEXT_QUESTION`
- **File & Symbol:** [`VQA_TEXT_QUESTION`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L223-L226) in `src/fusion_vision_mcp/constants.py`.
- **Design:** Replace the 23-word list with a two-branch regular expression that matches:
  1. **Branch 1 (`_EXPLICIT_TEXT_TERMS`, matched anywhere in the question):** Explicit lexical terms for writing, typography, inscriptions, and identifiers, including plural/verb `"labels"`:
     ```python
     _EXPLICIT_TEXT_TERMS = (
         r"\b(text|written|writing|say|says|said|read|reads|title|titled|headline|subheading|subtitle|"
         r"label|labels|labell?ed|word|words|printed|print|inscription|inscribed|caption|slogan|"
         r"lettering|letters|spelled|spelt|brand\s+name|surname|username|filename|"
         r"numerals?|digits?|initials?|signature|signed|publisher|author|"
         r"stamped|engraved|stencill?ed|embroidered|handwritten|typed|underlined|"
         r"email|url|website|telephone|p-value|phrase|sentence|quote|quoted|leaderboard)\b"
     )
     ```
  2. **Branch 2 (`_WH_TEXT_TERMS`, matched only when the question starts with `What / Which / Who / Where` and is NOT a `What/Which color...` question):** Queries asking for a specific named, numbered, priced, or UI-listed value:
     ```python
     _WH_TEXT_TERMS = (
         r"^\s*(?:what|which|who|where)\b(?!\s+colou?rs?\b).*\b("
         r"brand|name|named|names|number|numbers|numbered|credit|"
         r"date|dated|year|time|price|priced|cost|rate|salary|amount|percentage|percent|value|score|version|"
         r"code|address|handle|phone|currency|symbol|temperature|"
         r"sign|banner|plaque|sticker|tags?|badge|logo|make|model|"
         r"menu|option|tab|button|heading|header|footer|legend|"
         r"affixed|listed|displayed|indicated|specified|mentioned|recorded|highlighted|"
         r"message|notice|warning|definition|suggestion)\b"
     )
     ```
- **Verified Routing Coverage Across Cached Benchmarks (Zero Over-Routing on Visual Types):**

  | Dataset & Question Type | Current `VQA_TEXT_QUESTION` | Proposed Two-Branch `VQA_TEXT_QUESTION` | Why It Matters |
  | :--- | :--- | :--- | :--- |
  | **Round 23 `no_text` (`n=105`)** | `73 / 105` (`69.5%`) | **`105 / 105` (`100.0%`)** | Eliminates the `12.5%`-accuracy unrouted `no_text` blind spot (`28` misses) |
  | **Round 23 `text` (`n=233`)** | `163 / 233` (`70.0%`) | **`222 / 233` (`95.3%`)** | Routes `+59` text questions (`17` of which already have the exact answer in `ocr`) |
  | **Round 23 `no_object` (`n=118`)** | `4 / 118` (`3.4%`) | **`71 / 118` (`60.2%`)** | Routes `25` of the `39` `no_object` misses (`What brand/make/model/logo/sign...`) |
  | **Round 23 `count` (`n=65`)** | `8 / 65` (`12.3%`) | **`19 / 65` (`29.2%`)** | Routes the `"What number labels..."` / `"What is the cost..."` OCR number questions |
  | **Round 23 `color` / `object` / `yes_no`** | `6.8%` / `2.5%` / `9.4%` | `7.6%` / `7.5%` / `12.3%` | Keeps non-text visual questions unrouted (`88%–93%` unrouted) |
  | **Phase 11 `text` (`n=341`)** | `193 / 341` (`56.6%`) | **`315 / 341` (`92.4%`)** | Routes `+122` text questions (`41` of which already have the exact answer in `ocr`) |
  | **Phase 11 `no_text` (`n=53`)** | `53 / 53` (`100.0%`) | **`53 / 53` (`100.0%`)** | Preserves `100.0%` routing |
  | **Phase 11 `object` / `color` / `yes_no`** | `0.0%` / `11.6%` / `9.1%` | `0.0%` / `14.2%` / `10.9%` | Preserves fast plain path on visual questions |

#### Step 2B: OCR Prompt Grounding, Verbatim Spelling Trust & `VQA_OCR_WORDS = 240`
- **Files & Symbols:** [`VQA_OCR_WORDS`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L228) and [`VQA_TEXT_SUFFIX`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L222) in `src/fusion_vision_mcp/constants.py`; [`lfm_vqa_prompt`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L901-L908) in `src/fusion_vision_mcp/analysis.py`.
- **Hypotheses & Candidate Arms to Screen in Gate V1 (`benchmarks/vqa_gate_v1_screen.py`):**
  1. **`route_only`:** Shipped `lfm_vqa_prompt` + `VQA_OCR_WORDS = 160`, changing **only** `VQA_TEXT_QUESTION` to the two-branch regex from Step 2A. Isolates the pure routing gain.
  2. **`route_240`:** `route_only` + raising `VQA_OCR_WORDS` from `160` to `240` (recovers the `T1b + T3` dense-document truncation misses where the answer lay between word 161 and word 240).
  3. **`route_240_premise`:** `route_240` + updating `VQA_TEXT_SUFFIX` from:
     `"\nIf the question asks about text that is not in the image, answer that there is none."`
     to:
     `"\nIf the question asks about text or an item that is not in the image, answer that there is none."`
     so that the `71/118` routed `no_object` false-premise questions (`"What brand of microwave oven is on the counter?"`, `"What make of sports car is shown in the thumbnail photos?"`) are explicitly covered by the `"answer that there is none"` instruction.
  4. **`route_240_verbatim`:** `route_240_premise` + replacing the distrust-inducing prefix `Text read from the image by OCR (it may be incomplete or misread):` with `Text read from the image by OCR:` and instructing LFM to quote exact names/numbers from the OCR text when they answer the question (`"\nUse the exact spelling and full name from the OCR text when it answers the question. If the question asks about text or an item that is not in the image, answer that there is none."`). Targets the `T2` misses where LFM overrode `'Strengthened trap'` -> `'Strengthened Irag'`, `'Strive'` -> `'Strike'`, `'10A'` -> `'16A'`, and truncated `'REPUBLIC OF ARMENIA'` -> `'Armenia'`.
- **Efficiency Note for Gate V1 Screen:** On both Phase 11 (`1,200` questions) and Round 23 (`892` questions), `ocr` outputs are already cached (`visionbench_report_20261020_184831.replies.jsonl` and `caption_ocr_r21.jsonl`), and for `route_only` any question that was already routed or remains unrouted can reuse existing cached rows where prompts are identical — only the newly routed questions (`~158` on Phase 11, `~179` on Round 23) need live LFM evaluation for `route_only` (`~25 minutes` total on CPU)!
- **Pre-Registered Gate V1 Screen Criteria (Seen Phase 11 + Seen Round 23):**
  - Overall accuracy on Round 23 (`n=892`) must improve by **`>= +3.0 pts`** over `78.5%` (to `>= 81.5%`).
  - Overall accuracy on Phase 11 (`n=1,200`) must improve by **`>= +1.0 pt`** over `82.4%` (to `>= 83.4%`).
  - No question type with `n >= 30` on either dataset may regress by more than `1.5 pts`.
  - Pick the highest-scoring passing arm (`route_only`, `route_240`, `route_240_premise`, or `route_240_verbatim`) to carry forward into Track 3 and the unseen Round 29 confirmation gate.

---

### Track 3: Scoped False-Premise Protection (`no_object`) & Concise Answer Formatting (Gate V2)

Track 3 addresses the two largest remaining gaps after Track 2:
1. The **unrouted `no_object` false-premise questions** (`47/118` on Round 23, accounting for `14` of the `39` `no_object` misses such as `"What color is the umbrella held by the customer?"`, `"What color is the bus arriving at the stop?"`, `"What animal is accompanying the men in the gallery?"`).
2. The **24.2-point verbosity penalty on Phase 2 / Round 22 TextVQA exact match** (`52.4%` exact match vs. `76.6%` contained).

#### Step 3A: Scoped False-Premise Escape Clause for Unrouted Wh-Questions (Never Touching `yes_no` or `count`)
- **Why global `PREMISE_SUFFIX` failed in `vqa_premise.py`:** In `vqa_premise.py`, appending `PREMISE_SUFFIX` to **all 1,200 Phase 11 questions** cost `25` hits on ordinary questions: `11` losses on `yes_no` (`366 -> 355`), `6` losses on `count` (`131 -> 125`), and `10` losses on `text` (`213 -> 203`). Notice that **`object` (+1 hit, `31 -> 32`), `position` (+1 hit, `62 -> 63`), and `color` (0 net change, `137 -> 137`) had ZERO net loss from `PREMISE_SUFFIX` on Phase 11!**
- **Why that insight is decisive:**
  - `yes_no` questions (`"Is there a dog?"`, `"Are the lights on?"`) do **not** have a false presupposition — they already ask `yes` or `no`! Appending a false-premise warning to a `yes_no` question confuses the model into hedging instead of answering `Yes` or `No`.
  - `How many ...` questions (`count`) already allow `"0"` as a direct numeric answer.
  -Routed text/brand/sign questions (`_TEXT_QUESTION_RE`) already get `VQA_TEXT_SUFFIX`.
  - Therefore, the **only** unrouted questions that carry an unguarded existential presupposition are **unrouted `What / Where / Which / Who` questions** (`color`, `object`, `position`, and unrouted `no_object` questions like `"What color is the bus arriving at the stop?"` or `"What animal is shown on the slide?"`)!
- **Verification from existing cached `vqa_premise_rows.jsonl` (Zero new model runs needed for Phase 11):**
  On Phase 11 (`1,200` questions), applying `PREMISE_SUFFIX` **only** to unrouted `What / Where / Which / Who` questions (leaving `yes_no`, `How many` `count`, and `_TEXT_QUESTION_RE`-routed questions untouched) changes Phase 11 `color + object + position` by **`+2` net hits (`+0` on `color`, `+1` on `object`, `+1` on `position`)** — **zero regression on Phase 11**!
- **Candidate Arms to Screen on Round 23 (`benchmarks/vqa_gate_v2_premise.py`):**
  - **Arm `wh_premise_suffix`:** For unrouted questions matching `_WH_PRESUPPOSITION_RE = re.compile(r"^\s*(?:what|where|which|who)\b", re.IGNORECASE)`, append `VQA_WH_PREMISE_SUFFIX = "\nIf the object asked about is not in the image, answer that there is none."` (using the exact `"answer that there is none"` phrasing that `grade_answer` scores cleanly and that succeeded on `ocr_text`).
  - **Screen Cost:** Only unrouted `What/Where/Which/Who` questions on Round 23 (`~215` questions) need to be run (`~16 minutes` on CPU).
  - **Pre-Registered Gate V2A Criteria:** Must lift Round 23 `no_object` accuracy by **`>= +8.0 pts`** over Track 2 alone while losing **`<= 1.5 pts`** across unrouted `color + object + position` combined.

#### Step 3B: Evaluating Concise Phrasing Across Both Exact-Match (TextVQA) and Containment (Phase 11 / Round 23)
- **The trade-off to test:** On Round 22 TextVQA (`n=154`), appending `"\nAnswer with a single word or phrase."` (`short` arm) lifted `vqa_accuracy` exact match from **`52.4% -> 76.6%` (`+24.2 pts`)** and cut median latency from `5.11 s -> 3.99 s`. However, on `object` and `position` questions in Phase 11 / Round 23, a 1–2 word reply gives `grade_answer` (`contains_phrase`) only one lexical surface form (e.g. `"mug"` vs. `"A white ceramic coffee mug or cup"`), which could reduce containment hits if the model picks a synonym of the label.
- **Candidate Arms to Screen in `benchmarks/vqa_gate_v2_concise.py`:**
  1. **`concise_routed`:** Append `"Answer with a short phrase (1 to 4 words), or say there is none if it is not in the image."` **only to `_TEXT_QUESTION_RE`-routed questions** (which covers **92%–95%** of `text` and TextVQA questions, where answers are verbatim strings/numbers/names, while leaving visual `object`, `position`, and `color` questions free to use descriptive sentences!).
  2. **`concise_all`:** Append `"Answer concisely with a short phrase."` to all questions.
- **Pre-Registered Gate V2B Criteria:**
  - On Round 22 TextVQA (`n=154`), `vqa_accuracy` exact match must be **`>= 70.0%`** (`>= +17.6 pts` over `shipped` `52.4%`) and `answer_contained_pct >= 75.5%`.
  - On Round 23 (`n=892`) and Phase 11 (`n=1,200`), overall containment accuracy must not drop by more than **`0.5 pts`** compared to the Track 2 winner, and no question type with `n >= 30` may drop by more than **`2.0 pts`**.

---

### Track 4: OCR-Anchor Guided Region Crop & Selective High-Resolution Routing (Gate V3)

Track 4 targets the hardest remaining visual/text errors: **spatial/UI/chart text lookups (`T2` layout misses + `T4` tiny-text misses)** and **fine-grained diagram/UI counts** (`app and web screenshots` `70.7%`, `charts, diagrams and infographics` `70.8%`).

#### Why Gate R (`region_inspect_gate.py`) and `ocr_text_t4` (`vqa_premise.py`) Failed — And What Is Different Here
1. **Gate R failed (`-7.8 pts`)** because it used Grounding DINO (`~68%` first-box precision) to guess a crop and **discarded the full image**.
2. **`ocr_text_t4` failed its efficiency gate (`+0.4 pts`, `4.50 s -> 12.50 s`)** because it ran 4-tile splitting (`1,280` visual tokens) on **every** text question, including simple large-sign questions that `ocr` had already solved, and tile boundaries split large banners/titles in half (causing 12 losses).

#### Step 4A: Spatial Line-Order & Anchor-Region Highlighting from `ocr_pages` (`0 ms` Extra Model Cost)
- **Observation:** `ocr_pages(app, images)` (`analysis.py:725–814`) **already computes and returns `text_regions: list[{"text": str, "box": [x1, y1, x2, y2]}]`, sorted in reading order!** Currently, `vqa_answers` (`analysis.py:921`) throws away `page["text_regions"]` and only passes `page["text"]`.
- **Two zero-latency improvements using `page["text_regions"]`:**
  1. **Quoted-Anchor Neighbourhood Extraction:** Many `T1`, `T2`, and `T4` questions in Round 23 and Phase 11 explicitly quote or name an anchor string on the sign, UI, or chart:
     - *"What word appears directly under 'MEDICAL' on the sign?"* (`r21_d5a3add058`)
     - *"What software suite name appears in the title bar after 'Sem título 1 - '?"* (`r21_83a186ffec`)
     - *"What engine name is displayed after 'Fill_block' in parentheses?"* (`r21_a0095315ba`)
     - *"What camera model is listed under 'Informace o snímku'?"* (`r21_...`)
     - *"What word is printed beneath the year 2022?"* (`r21_4f44857cb7`)
     - *"What author's name is printed after 'TRADUIT de l'anglais de'?"* (`sdf8c4e982b`)
     When the question contains a quoted anchor `'...'` or `"..."` (or `after/under/beneath/above/next to <anchor>`) that matches a region in `page["text_regions"]`:
     - Even if the anchor sits past word 240 on a dense page, we **always include the matched anchor region and its spatially adjacent `text_regions`** (above, below, left, right, with relative position labels) in `lfm_vqa_prompt`!
  2. **Selective 4-Tile Escalation (`max_tiles=4` on `LfmOnnxCaptioner`) ONLY When Needed:**
     Instead of running 4 tiles on all 325 routed questions (`ocr_text_t4`), test enabling `max_tiles=4` (`processor_for("tiles4")`) **only** when:
     - The image is high-resolution (`max(image.size) >= 1000`), **AND**
     - Either (a) the question is a routed text question where `ocr_pages` found `0` words or the question asks about a quoted anchor not found in `ocr_pages`, OR (b) the question is a UI/chart/document `count` question (`"How many rows/columns/swimlanes/items/files/thumbnails/windows..."`).
     Because this triggers on `< 12%` of questions, overall median latency stays at `~4.8 s` (`p90 <= 8.5 s`) instead of `12.5 s`!
- **Pre-Registered Gate V3 Criteria (`benchmarks/vqa_gate_v3_selective_hires.py`):**
  - Must improve accuracy on the triggered subset by **`>= +5.0 pts`** and overall Phase 11 / Round 23 accuracy by **`>= +0.8 pts`**, with `0` regressions on untriggered questions and overall median latency `<= 5.5 s`.

---

### Track 5: Re-Calibrating `check_consistency=True` for LFM2.5-VL-3B (Gate V4 / Phase 12)

Track 5 resolves the explicit open item in [`CLAUDE.md`](file:///c:/AI/MCP/FusionVisionMCP/CLAUDE.md#L283) and [`tools/vqa.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/vqa.py#L44):
> *"It has not been re-measured with LFM2.5-VL, today's answerer."*

#### Step 5A: Question-Aware `_vqa_consistency(answer, control_answer, question="")`
- **File & Symbol:** [`_vqa_consistency`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L180-L225) in `src/fusion_vision_mcp/analysis.py` and its call site in [`query_image`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/vqa.py#L179).
- **Backward Compatibility:** Keep `question: str = ""` as an optional parameter with default `""` so existing callers and unit tests calling `_vqa_consistency(answer, control_answer)` continue to work without signature breakage.
- **Calibration Rules for LFM2.5-VL-3B:**
  1. **Open-ended defect / anomaly judgment questions (`"describe anything wrong"`, `"what is wrong"`, `"are there any defects/errors/anomalies/artifacts"`, or when `question == ""` and both answers are in `{"none", "nothing", "n/a", "na", "i don't know", "unknown", "not sure", ""}`):**
     - Preserve the original flat-default trap: if both answers reduce to a flat default/refusal token on an open-ended judgment query, return `consistent=True, confidence="low"`.
  2. **Polar (`yes/no`) questions (`^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b`):**
     - Extract unambiguous polarity (`"yes"` vs. `"no"`) from the leading token / token set of `answer` and `control_answer`.
     - If both answers have the same unambiguous polarity (`"yes"` & `"yes"`, or `"no"` & `"no"`), return `consistent=True, confidence="normal"`. (Remember: on Round 23, LFM's 1-word `"Yes"`/`"No"` answers on `yes_no` questions are **98.2% accurate**!)
     - If they have opposite polarities or either hedges with both `"yes"` and `"no"`, return `consistent=False, confidence="low"`.
  3. **False-premise refusal agreement on factual Wh-questions:**
     - If `question` is a factual Wh-question (not an open defect prompt) and both `answer` and `control_answer` explicitly state the queried text/object is absent (`none`, `nothing`, `there is no`, `not visible/present/mentioned/shown`, `does not appear/show/depict`), return `consistent=True, confidence="normal"`.
     - If one answer refuses the premise and the other invents a factual answer, return `consistent=False, confidence="low"`.
  4. **Quantity (`How many`) questions:**
     - Normalize number words (`zero`..`twelve`) to digits and extract integers. If both `answer` and `control_answer` contain a single integer (or start with an integer), `consistent = (int_a == int_c)` and `confidence = "normal" if consistent else "low"`.
  5. **Substantive prose answers:**
     - Compare article/punctuation-normalized strings `na` and `nc`: `consistent = (na == nc or na in nc or nc in na or token_f1(na, nc) >= 0.45)`.
- **Verified Simulation on 1,200 Phase 11 Questions (from Step 0 Cache Audit):**
  - **Old `_vqa_consistency`:** `flagged_pct = 44.4%`, `quiet_on_right_pct = 60.6%` (falsely flags 39.4% of right answers!), leaving only `667/1200` answers unflagged.
  - **Calibrated `_vqa_consistency`:** `flagged_pct = 15.6%`, **`quiet_on_right_pct = 90.9%` (`+30.3 pts` specificity!)**, `catches_wrong_pct = 46.0%`, `accuracy_if_unflagged_pct = 88.7%` (`+6.3 pts` above baseline `82.4%`, with `1,013/1,200` answers unflagged and a **4.6× higher error rate on flagged vs. unflagged answers**!).

#### Step 5B: Pre-Registered Gate V4 (Live Phase 12 `evaluate_flags` + `cross_check` Value Audit)
- **Script:** `C:\AI\Projects\VisionBench\benchmarks\vqa_gate_v4_consistency.py` + `run_visionbench.py --phases 12 --limit 100`.
- **Gate V4 Pass Criteria:**
  1. `quiet_on_right_pct >= 85.0%` (specificity on right answers).
  2. `catches_wrong_pct >= 35.0%` (sensitivity on wrong answers) and error rate among flagged answers `>= 2.5×` error rate among unflagged answers.
  3. `accuracy_if_unflagged_pct >= accuracy_pct + 3.5 pts`.
  4. **Cross-check rescue audit:** Report how often `entry["cross_check"]` (after Step 1C's contract alignment) provides the correct count/text/spatial relation on the subset of questions where `confidence == "low"` and a cross-check tool ran.
  5. Update the docstring of `query_image` (`tools/vqa.py:44`) and `CLAUDE.md` (`line 283`) with the measured LFM2.5-VL `check_consistency` numbers.

---

### Track 6: Benchmark Scorer Audit (Step S-VQA in VisionBench — Separate from Model Gates)

During our Step 0 cache audit of Round 23 (`n=892`) and Phase 11 (`n=1,200`), we identified three concrete scorer defects in `C:\AI\Projects\VisionBench\run_visionbench.py`:

1. **Single-letter `"A"` erasure bug in `normalize_answer` (`run_visionbench.py:320–324`):**
   - `normalize_answer` drops `_ARTICLES = {"a", "an", "the"}` from every string. When the ground-truth expected answer is the single letter `"A"` (e.g., Round 23 `r21_...` `[signage, logos and packaging]`: *"What letter is the man holding in his hand?"*, `expected = "A"`, LFM `answer = "A"`), `normalize_answer("A")` returns `""`, and `contains_phrase` (`run_visionbench.py:420–421`) immediately returns `False` (`if not needle: return False`). Thus **any question with expected answer `"A"` is mathematically impossible to score right**.
   - **Fix in `normalize_answer`:** If stripping `_ARTICLES` leaves `tokens` empty while the pre-filtered token list was non-empty (`["a"]`, `["an"]`, `["the"]`), keep the pre-filtered tokens!
2. **Negation contraction gap in `_NO_TEXT` (`run_visionbench.py:1474, 1488–1489`):**
   - `_NO_TEXT = {"no", "none", "nothing", "illegible", "unreadable"}` omits `"not"` and `"neither"`. On Round 23, 7 valid false-premise refusals (`"The man in the image is not wearing any eyeglasses."`, `"The specific building room number is not mentioned in the image."`, `"The dollar currency symbol ($) does not appear in this chart."`, `"The cartoon doesn't show any automobiles competing in the race."`, `"The plate does not depict a laboratory microscope."`, `"The flowchart does not display any corporate company logo."`, `"I don't see any brand logos..."`) were scored as misses because `normalize_answer` turns `"doesn't"` into `"doesn t"` and `"not"` is absent from `_NO_TEXT`.
3. **Derivational word-form gap on `object` / `shape` answers (`run_visionbench.py:1490`):**
   - On Round 23 `object` questions, `"The glowing window in the center of the machine is rectangular"` fails `contains_phrase(answer, "rectangle")` and `"The handheld sign held by the figure is octagonal in shape"` fails `contains_phrase(answer, "octagon")`, even though Phase 11's caption scorer (`word_forms` / `words_match` at `run_visionbench.py:1404–1445`) already handles suffix stripping.

> [!IMPORTANT]
> **Strict Non-Circularity Rule (Following `Caption_Improvement_Plan.md` Step S):**
> 1. **No model/prompt/routing gate (Gates V1, V2, V3, V5) may rely on a scorer change to pass.** Every gate in Tracks 2, 3, 4, and 7 must pass under the **existing, unchanged `run_visionbench.grade_answer`** scorer first.
> 2. Any proposed scorer update to `run_visionbench.py` (`normalize_answer` keeping single-token `"a"` and `_NO_TEXT` adding `"not"` when paired with a negative verb/predication) must be audited via a standalone script (`benchmarks/scorer_vqa_audit.py`) and a blind user review page (`benchmarks/scorer_vqa_check_serve.py`) showing every flipped question before `run_visionbench.py` is modified.

---

### Track 7: Unseen Confirmation Gate — Round 29 (Gate V5)

Before any prompt, routing, or selective-tiling change from Tracks 2–4 is committed to `src/fusion_vision_mcp/`, the winning configuration from the seen screens (Gates V1–V3) must pass **Gate V5** on a fresh, unseen VQA round (**Round 29**).

#### Step 7A: Sourcing & Spot-Checking Round 29 (`VisionBench-labels/benchmarks/vqa_round29/`)
- **Image Pool:** ~225 unseen images balanced across the 9 generalist domains (`everyday photos`, `app and web screenshots`, `charts, diagrams and infographics`, `documents`, `game UI and HUD`, `signage, logos and packaging`, `illustrations, cartoons and comics`, `artwork and paintings`, `AI-generated images`) — either drawn from the held-out Round 24/25 image pools that have never had VQA questions written for them, or freshly sourced from Wikimedia Commons following `worker_handoff/ROUND23_PLAN.md`.
- **Question Specification:** 4 questions per image (`~900` questions total):
  - 3 ordinary questions covering `text`, `yes_no`, `color`, `object`, `count`, and `position`.
  - 1 false-premise question (`no_text` or `no_object`, split ~50/50) asking about text or an object plausible for the scene's theme that is **not** present in the image.
- **Blind User Spot-Check (`benchmarks/round29_check_serve.py`, port `8785`):**
  - Serves a 40-question random sample (20 ordinary + 20 false-premise) for user verification before any model answers are generated.
  - Gate to proceed: `>= 18/20` ordinary answers verified right and `>= 18/20` false premises verified truly absent (any rejected item is dropped before running models).

#### Step 7B: Pre-Registered Gate V5 Rules (`benchmarks/vqa_round29.py`)
- **Arms compared on CPU (`seed = question index`):**
  1. `shipped_baseline`: Current `query_image` (`constants.VQA_TEXT_QUESTION` as of 2026-10-24, `VQA_OCR_WORDS = 160`, `lfm_vqa_prompt` as of 2026-10-24, single image `<= 256` tokens).
  2. `candidate`: The winning combination from Tracks 2–4.
- **Pass Criteria (all four must hold under the unchanged `run_visionbench.grade_answer` scorer):**
  1. **Overall gain:** `candidate` overall accuracy `>= shipped_baseline + 2.5 pts` with the 95% paired bootstrap CI over images excluding `0.0`.
  2. **False-premise gain:** Combined false-premise accuracy (`no_text + no_object`) `>= shipped_baseline + 8.0 pts`.
  3. **Per-type non-regression:** No question type with `n >= 30` loses more than `2.0 pts` vs. `shipped_baseline`.
  4. **Latency ceiling:** Overall median seconds per question (including live `ocr_pages` time on newly routed questions) `<= 6.5 s` on CPU (well below Moondream2's `9.77 s`).
- **Parity Gate Before Shipping (`benchmarks/lfm_onnx_vqa_parity.py`):**
  - After updating `src/fusion_vision_mcp/constants.py` and `src/fusion_vision_mcp/analysis.py`, run an 8-question server parity check confirming `analysis.vqa_answers` produces 8/8 identical prompts and 8/8 identical answers to the `candidate` arm in `vqa_round29.py`.

---

## 3. Expected Trajectory Toward Frontier Grade (`~88–92%`)

Based on the exact counts in our Step 0 offline cache audit of Round 23 (`n=892`) and Phase 11 (`n=1,200`), the expected cumulative impact of each track is:

| Stage / Track | Round 23 (`n=892`) | Phase 11 (`n=1,200`) | Round 22 TextVQA Exact (`n=154`) | Primary Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Current Shipped (2026-10-30)** | **78.5%** (`700/892`) | **82.4%** (`989/1200`) | **52.4%** (`76.6%` cont.) | LFM2.5-VL-3B single-tile + 23-word `VQA_TEXT_QUESTION` |
| **+ Track 2 (Gate V1: Surgical `VQA_TEXT_QUESTION` + `VQA_OCR_WORDS=240` + Verbatim/Premise OCR Prompt)** | **~84.0–85.5%** (`+5.5 to +7.0 pts`) | **~85.0–86.2%** (`+2.6 to +3.8 pts`) | **~72.5%** (`+20.1 pts`) | Fixes `28` unrouted `no_text` misses (`12.5% -> ~89%`), `17` unrouted `T1a/b` `text` misses, `25` routed `no_object` misses, and `5–9` `T2` OCR-override misses |
| **+ Track 3 (Gate V2: Scoped Wh-Premise Guard + Concise Routed Phrasing)** | **~85.5–87.0%** (`+1.5 pts`) | **~85.5–86.5%** (`+0.3 pts`) | **~76.6%** (`+4.1 pts`) | Covers remaining unrouted `no_object` Wh-questions without hurting `yes_no`/`count`; reaches frontier TextVQA exact match (`76.6%`) |
| **+ Track 4 (Gate V3: OCR-Anchor Neighbourhood & Selective 4-Tile Escalation)** | **~86.8–88.2%** (`+1.2 to +1.5 pts`) | **~86.8–88.0%** (`+1.2 to +1.5 pts`) | **~77.5–78.5%** | Solves spatial sign/UI lookups (`T2`) and small-text/diagram questions (`T4`) on high-res screenshots & charts |
| **+ Track 6 (Step S-VQA: Non-Circular Scorer Fixes for `"A"` & `"not"`)** | **~88.0–89.5%** (reported alongside unchanged scorer) | **~87.5–88.8%** | **~77.5–78.5%** | Fixes impossible `expected="A"` grading bug and valid `"does not / not"` refusal grading |

---

## 4. File-by-File Implementation Reference for Worker Handoff

| File Path | Key Symbols | Planned Changes | Track |
| :--- | :--- | :--- | :--- |
| [`src/fusion_vision_mcp/constants.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/constants.py#L220-L229) | `VQA_TEXT_QUESTION`, `VQA_TEXT_SUFFIX`, `VQA_OCR_WORDS`, `VQA_WH_PREMISE_SUFFIX` | Expand `VQA_TEXT_QUESTION` to two-branch regex (`_EXPLICIT_TEXT_TERMS \| _WH_TEXT_TERMS`); update `VQA_TEXT_SUFFIX` to cover absent items; raise `VQA_OCR_WORDS` from `160` to `240`; add `VQA_WH_PREMISE_SUFFIX` if Gate V2A passes. | Tracks 2, 3 |
| [`src/fusion_vision_mcp/analysis.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/analysis.py#L180-L310) | `_vqa_consistency`, `_vqa_cross_check`, `_size_measurement`, `lfm_vqa_prompt`, `vqa_answers` | Catch `MemoryError` in `vqa_answers` (Step 1A); accept `ocr_readings` in `vqa_answers` and `_vqa_cross_check` (Step 1B); route `_vqa_cross_check` (`COUNT`) and `_size_measurement` (`SIZE`) through `execute_detection` + `COUNT_BOX_THRESHOLD` (Step 1C); update `lfm_vqa_prompt` with anchor neighbourhood & verbatim trust (Tracks 2–4); calibrate `_vqa_consistency` for LFM2.5-VL-3B (Track 5). | Tracks 1, 2, 3, 4, 5 |
| [`src/fusion_vision_mcp/question.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/question.py#L55-L155) | `_COUNT_PATTERNS`, `_clean_span`, `names_for` | Add finite lexical verbs to `_COUNT_PATTERNS` stop boundary; preserve `"image"` / `"photo"` when modifying UI head nouns (`panel`, `thumbnail`, `file`, `window`). | Track 1 (Step 1C) |
| [`src/fusion_vision_mcp/tools/vqa.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/tools/vqa.py#L29-L191) | `query_image` | Share single `ocr_pages` result across `vqa_answers` (primary + control) and `_vqa_cross_check`; pass `question` into `_vqa_consistency(answer, control_answer, question)`; update docstring with Round 29 and Phase 12 numbers. | Tracks 1, 5, 7 |
| [`src/fusion_vision_mcp/inspection.py`](file:///c:/AI/MCP/FusionVisionMCP/src/fusion_vision_mcp/inspection.py#L165-L370) | `_ANATOMY_PARTS`, `_check_anatomy`, `_check_defect_probe`, `analyze_inspection` | Drop unused `"head"` and `"finger"` passes from `_ANATOMY_PARTS`; reuse `"hand"` detections in `_check_defect_probe` (8 GD passes -> 5). | Track 1 (Step 1D) |
| [`tests/test_vqa_answers.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_vqa_answers.py) | `TestVqaAnswers`, `TestLfmVqaPrompt`, `TestTextQuestionRouting` | Add tests for `MemoryError` fallback, single `ocr_pages` call under `check_consistency=True`, expanded `VQA_TEXT_QUESTION` positive/negative cases, and updated `lfm_vqa_prompt`. | Tracks 1, 2, 3 |
| [`tests/test_question.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_question.py) & [`tests/test_crosscheck.py`](file:///c:/AI/MCP/FusionVisionMCP/tests/test_crosscheck.py) | `test_names_for_count`, `test_vqa_consistency`, `test_vqa_cross_check` | Add tests for finite verbs & UI compound nouns in `names_for(COUNT, ...)`, `COUNT_BOX_THRESHOLD` & `execute_detection` in `_vqa_cross_check`, and polar/numeric/prose agreement in `_vqa_consistency`. | Tracks 1, 5 |
| `C:\AI\Projects\VisionBench\benchmarks\` | `vqa_gate_v1_screen.py`, `vqa_gate_v2_premise.py`, `vqa_gate_v2_concise.py`, `vqa_gate_v3_selective_hires.py`, `vqa_gate_v4_consistency.py`, `vqa_round29.py`, `round29_check_serve.py` | Screen gates V1–V4 on seen caches/subsets; confirm winning configuration on unseen Round 29 (Gate V5) after 40-question user spot-check. | Tracks 2, 3, 4, 5, 6, 7 |
