# Overnight workplan, 2026-10-22 → 23 (Opus 5.5)

Written 2026-10-22 ~21:45 before the user went to bed. Everything below is **measurement only**: nothing is shipped to
`src/`, no FusionVisionMCP commit (its hook runs the full real-model test suite, which would skew timings), and every
step's variants and rules are committed in VisionBench **before** it runs. Shipping decisions wait for the user.
Steps run one at a time (CPU timings stay clean); a step that fails its rule stops its own line of work, not the queue.

## Where things stand

- `deep` = LFM2.5-VL-3B bf16 (shipped `46b0c3d`, CI green): confirm 74.8% vs Moondream2 69.0%.
- Default caption candidate = LFM2.5-VL-3B Q8 GGUF on llama-server, prompt `cautious` ("Describe only what is
  clearly visible in this image."). Screen passed: tune 72.9% vs Florence-2 65.3% (+7.6 raw, +8.1 matched), COCO
  CHAIR 11.0 / 16.3 vs Florence-2's 12.1 / 22.3, 2.4 s. **Confirm is running** (`caption_default_lfm.py confirm`:
  280 confirm images + the fresh COCO set `coco_chair2`).
- Known issue: llama-server stalled twice in ~1,300 calls (connection timeout); the benchmark restarts it. A shipped
  version must do the same.

## Queue

### 1. Finish the default-slot confirm (running, ~22:30)
Gate (fixed in `caption_default_lfm.py`): vs Florence-2, raw and matched recall ≥ +1.5, CHAIR-i/-s on `coco_chair2`
≤ Florence-2 + 1.0, median caption + OCR ≤ 15 s. Then the `deep` rule fixed before results: if bf16 `deep` gains
< 2.0 points raw over the new default on confirm, `deep` retires as a separate model (parameter kept as an alias).
**Output for the morning:** pass/fail, numbers, the `deep` verdict.

### 2. Default caption improvement round ("prompt round"; only if step 1 passes)
Script `caption_default_prompt.py` (new; rules committed before running). Same model, server and decoding as `cautious`
unless stated. Variants, from tonight's analysis (cautious loses ~5-6 points against `base` on every kind of image,
because it says less: 94 vs 138 words):
- `detailed_visible`: "Describe this image in detail, mentioning only what is clearly visible."
- `thorough_visible`: "Describe everything that is clearly visible in this image."
- `detailed_visible_greedy`: `detailed_visible` at temperature 0.

**Screen** on tune (120) + the old COCO set (`coco_chair`, the screen set): advance if it passes the default gate vs
Florence-2 (raw/matched ≥ +1.5, CHAIR ≤ +1.0, median ≤ 12 s) **and** beats `cautious` on tune by ≥ +1.0 raw. Pick the
highest tune recall.
**Confirm** on 280 confirm + `coco_chair2`: the same default gate, and ≥ `cautious`'s confirm recall + 1.0 raw.
Caveat to record: `coco_chair2` will then have judged two rounds (never used to pick). No third fresh COCO set exists
under the accepted licences (29 photos left), so this is the honest best; a new CHAIR source (Open Images' verified
negative labels) is the fix if a third round is ever needed.

*Not run tonight:* the text-aware prompt (`base` on pages with ≥ 10 OCR words, `cautious` elsewhere; +2.9 on tune).
CHAIR only sees COCO photos, which rarely carry text, so it would pass by construction while `base`'s extra inventions
on screenshots go unmeasured. It needs a precision check by the worker and the user (like 2B's), so it waits.

### 3. Can LFM take over `query_image` from Moondream2? (VQA screen)
Script `vqa_lfm.py` (new; rules committed before running). LFM2.5-VL-3B Q8 on llama-server answers phase 11's VQA
questions verbatim (no prompt engineering, card decoding, at most 128 tokens), scored with phase 11's own
answer-contained scorer, against Moondream2's cached phase 11 answers on the same questions (78.8%).
No tuning happens, so phase 11 can serve as the test. **Rule:** a pass is ≥ Moondream2 + 1.5 overall with no question
category losing more than 5 points; a pass only qualifies LFM for the next checks (`check_consistency`'s control
question, `critique_composition`'s busy-background and subject naming, `count_objects`' estimate), each needing its own
gate before Moondream2 could retire (~7.5 GB fp32, the remote-code pin and `pyvips` would go).

### 4. Phase 11 estimate for the new default caption
Run the step-1 winner (or step 2's, if it passes) on phase 11's 400 images and score keyword recall with phase 11's
scorer and the cached `ocr` text, to see where the scoreboard's default row would land (green = within ~5 of the
77.1% ceiling). Florence-2 is 64.5% there. Report only; the official figure comes from a phase 11 express run after
shipping.

### 5. ONNX build and test (last, so its timing runs alone)
Script `lfm_onnx.py` (new). `LiquidAI/LFM2.5-VL-3B-ONNX` at a pinned revision on `onnxruntime` 1.30 (already a
dependency), in-process: transformers' own LFM processor at `max_tiles` 4 (as bf16 was measured), the fp32 vision
encoder and embeddings, the q8 merged decoder, and a hand-written generation loop (image embeddings spliced into the
prompt, decoder with its past state, the card's sampling). Built and debugged first on a handful of images.
- **Probe** (20 tune images, the default prompt that step 1/2 produced): viable if median caption ≤ 6 s and it runs
  without errors.
- **If viable:** tune 120 with that prompt vs the llama.cpp version on the same images; reported, not a ship gate.
  If it holds (raw within -1.0), ONNX becomes the packaging candidate and needs its own confirm + `coco_chair2` run
  (the runtime changes the captions, so the llama.cpp gate does not carry over). That confirm is for the user's go.
- If ONNX is too slow or the loop cannot be made right, packaging falls back to a pinned llama-server release.

### 6. Housekeeping
- Check GitHub CI on anything pushed; push VisionBench results at each milestone.
- Morning report: one table per step, decisions needed, nothing shipped.

## Decisions waiting for the user in the morning
1. Ship LFM Q8 as the default caption (and which prompt), and through which runtime (ONNX in-process vs pinned
   llama-server). Packaging work starts only after this.
2. `deep`: retire or keep, per the step-1 rule's result.
3. The text-aware prompt's precision check (needs the worker and the user).
4. If step 3 passes: whether to run the remaining Moondream2-replacement gates.
5. Still open from before: GGUF `deep` re-test (needs a fresh set), Gemma 4 E2B quantized (only if LFM fails),
   the Cline rules refresh, the OCR-TextVQA plan.

## Results (morning report, 2026-10-23 ~01:00, Opus 5.5)

Everything ran; nothing shipped. VisionBench commits `a6c8e82`, `f1a33c0`, `224bea8`, `64d1791`, `7cfe2ca`, `64b6079`
(all pushed).

**Read this first: every llama-server timing was on the GPU.** The winget llama.cpp build ships Vulkan and defaults to
`--n-gpu-layers auto`; its verbose log shows every layer on the RTX 5060. Found during step 5 (ONNX on CPU decoded at
62 ms/token against llama-server's apparent ~18). Quality figures are unaffected (same weights; CPU vs GPU recall on
tune +0.0), but the default slot's time gates were never really met on CPU. ~4,000 Vulkan calls ran without a crash.
CPU-only llama-server needs `LLAMA_ARG_N_GPU_LAYERS=0`, `LLAMA_ARG_DEVICE=none`, `LLAMA_ARG_MMPROJ_OFFLOAD=0`.

### 1. Default-slot confirm (LFM Q8, `cautious`): quality passes, CPU time fails on llama-server
| | Florence-2 bf16 | LFM cautious | gate |
|---|---|---|---|
| confirm recall (280) | 64.8% | **71.3%** (+6.5, CI 4.3-8.8) | ≥ +1.5 ✅ |
| matched length | | +8.4 (CI 5.9-10.8) | ≥ +1.5 ✅ |
| CHAIR-i / -s, fresh COCO (300) | 13.3 / 27.7% | **12.4 / 24.0%** | ≤ +1.0 ✅ |
| median caption + OCR | 6.6 s caption (CPU) | 4.7 s on **GPU**; CPU llama-server 25.6 s caption (tune) → ~28 s | ≤ 15 s ❌ on CPU |

Ahead on all nine kinds (documents +12.4, illustrations +12.3). No server restarts this run.
**`deep` rule:** bf16 `deep` 74.8% vs the new default 71.3% = +3.5 raw (CI 1.6-5.4) ≥ 2.0, so **`deep` stays**; but
matched length it is -1.3 (CI -3.5 to 0.8) and behind on documents (80.5 vs 83.2): its gain is length. (The rule was in
the docstring, not the code; computed afterwards from the cached captions, same scorer.)

### 2. Prompt round: fails
Screen (tune 120 + coco_chair): `detailed_visible` 75.6% (+2.7 over cautious), CHAIR 12.2 / 21.7 → picked;
`thorough_visible` 77.9% but CHAIR 16.6 / 29.7 ✗; greedy 73.2% (+0.3) ✗. **Confirm: `detailed_visible` 71.3% = cautious
71.3%** (needed +1.0). The tune gain did not replicate. `cautious` stays.

### 3. LFM as `query_image`: fails
Phase 11, 1,200 questions: **84.8% vs Moondream2 78.8% (+6.0)**; text +15.5, count +12.8, position +9.2, colour +3.2,
yes/no -0.5, but **no_text -18.9** (n 53) and object -5.5 (n 36). The no_text loss is false-premise questions ("What text
is written on the green sofa?"): LFM invents an answer ("MADAME", a whole pie-chart title) where Moondream2 says none.
Moondream2 stays. A premise check (e.g. ask "Is there any text on X?" first, or route text questions through `ocr`)
might fix it; that would need its own gate.

### 4. Phase 11 caption estimate (report only)
`cautious` + the 160-word line: **70.9% vs Florence-2 64.5%** (old scorer 69.3 vs 63.1). Ceiling 77.1, so 6.2 below:
just outside green, up from 12.6 below. Ahead on every kind.

### 5. ONNX: the loop works, CPU speed is the question
Build: coherent captions, 0 errors. Probe (20 tune images, CPU): **single image 10.6 s median** (vision 1.4, prefill 2.3,
decode ~62 ms/token), 4 tiles 20.1 s. Both failed the probe's 6.0 s bar, so tune did not run. That bar assumed
llama-server's 2.4 s was CPU; it was not. CPU-only llama-server took **24.3 s** on the same 20 images (25.6 s on 120),
so ONNX single image is the fastest CPU runtime measured, and 10.6 s + ~2.4 s OCR ≈ 13 s is inside the default slot's
12 s / 15 s gates. Its quality is unmeasured (no tiling, which llama.cpp may do on large images).

## Decisions for the user
1. **Default caption runtime.** On CPU the only candidate inside the time gates is ONNX single-image (~10.6 s vs
   Florence-2's 6.6 s). Proposed next: `lfm_onnx.py` tune (120) vs the cautious captions, then a full confirm +
   `coco_chair2` on ONNX (the runtime changes captions). Needs your go; the probe's rule said stop.
   Alternatives: allow GPU (fast, but the laptop's CUDA crash history), or keep Florence-2.
2. **`deep`:** stays by the rule; worth rewording its description (its gain is length: more detail, not better per word).
3. **Moondream2 replacement:** VQA failed on false-premise text questions; a premise-check variant could be gated.
4. Still open: text-aware prompt precision check (worker + you), GGUF `deep` re-test (its timing was GPU too), Gemma 4
   E2B (only if LFM fails), Cline rules refresh, OCR-TextVQA plan.

## User decisions, 2026-10-23 morning
1. **ONNX single image: go.** `lfm_onnx.py tune1` (120 tune, raw within -1.0 of llama-server, median <= 12 s), then
   `confirm` (280 + `coco_chair2`, the default-slot gate) if it holds. Rules committed before running.
2. **`deep` wording: held back, corrected.** The "its gain is length" finding is against the *LFM* default only. Against
   the shipped default (Florence-2) `deep` is +10.0 raw and **+8.1 at matched length** (CI 5.8-10.5) on confirm, so the
   current description ("covers clearly more of what is there") stays true. The rewording ships in the same commit as
   the default switch, if that happens, e.g.: "against the default caption it adds detail rather than accuracy per
   word: +3.5 key facts (74.8 vs 71.3%), -1.3 at equal length; longer (142 vs 94 words) and ~3x slower".
3. **Premise-check VQA: go.** `vqa_premise.py screen` on phase 11 (arms plain / premise / ocr_text, ONNX single
   image, CPU). Phase 11 is a screen here (the variants answer its failures); a pass needs a fresh VQA set with
   false-premise questions (a worker round) before anything ships.
