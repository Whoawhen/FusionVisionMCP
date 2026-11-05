# Open ideas, by tool

The one list of ideas that are still worth doing, gathered 2026-11-02 from every plan that is now in
[`archive/`](archive/). Each line names where its detail lives. An idea that gets picked up becomes its own plan
at the top of `Future_Ideas/`; when that plan is done it moves to `archive/` and its line here goes.

One tool has a live plan, and its open steps live there, not here:

- `critique_composition`: [Critique_Composition_Improvement_Plan.md](Critique_Composition_Improvement_Plan.md)
  (R3 step 0 next). Whether LFM can take over Moondream2's two questions here, and so retire Moondream2, rides on
  this plan.

Standing rules that shape every line below: no Chinese-developed models in new work, and no heavy tuning on
Grounding DINO or PP-OCRv6 while they wait for replacement ([frontier_capability_gaps.md](frontier_capability_gaps.md),
"Replacing Chinese-developed models"); fine-tunes and LoRAs wait until the suite meets the agreed "good" bar (same
file, "What good means").

## `caption`

- **`Image type:` line on `deep`.** Shipped on the default caption only (2026-10-24); `deep`, `region` and the
  Florence-2 fallback were never measured. The user asked to be reminded. Source: [docs/history.md](../docs/history.md), "the default caption
  starts with `Image type:`".
- **Escalate some images to `deep`** (old step 2A). Worth about +2 points when the default was Florence-2 and `deep`
  Moondream2; both slots are LFM2.5-VL now, so the premise has to be re-derived from cached captions before
  anything runs. Source: [archive/Caption_Improvement_Plan.md](archive/Caption_Improvement_Plan.md) 0.3 and 0.12.

## `ocr`

- **A stray filter for PP-OCRv6's medium models.** They read more (+8 net on round 22 tune) but emit stray single
  characters and CJK glyphs on text-free images, and the strays score like real text, so a confidence floor can't
  remove them. Needs a new idea for telling strays from text. Round 22's 154 confirm questions are still unseen and
  held in reserve. Source: [archive/OCR-TextVQA_Improvement_Plan.md](archive/OCR-TextVQA_Improvement_Plan.md) 0.18.
- **Replace PP-OCRv6** (no-Chinese rule). docTR, the preferred candidate, trailed by 13 points in sprint R gate 1;
  Windows' built-in OCR (`Windows.Media.Ocr`, local) is untried and would need its own gate 0. A docTR fine-tune on
  scene text waits for the "good" bar. Sources: [archive/detector_ocr_replacement_plan.md](archive/detector_ocr_replacement_plan.md),
  [archive/sam3_expansion_plan.md](archive/sam3_expansion_plan.md) "Not in scope".
- **Learned deblur.** Allowed if a non-Chinese, CPU-friendly model can be shortlisted (NAFNet is excluded); none
  was. Heavy blur is still +0 on every preprocessing arm. Source: [archive/ocr_blur_improvement_plan.md](archive/ocr_blur_improvement_plan.md) 0.3, B2 arm 7.

## `count_objects` (plan closed by the user 2026-10-31)

- **Hard negatives.** Grounding DINO still counts a verified-absent object about 7 times in 10. The photo veto `V2`
  failed Gate 4 and the guarded `V2g` was not taken to a fresh round (the user's choice). Source:
  [archive/Counting_Improvement_Plan.md](archive/Counting_Improvement_Plan.md) 0.12, 0.15, 0.16.
- **Flat-image counts of present objects** (34.8% exact on round 28). Closed with the current models: LFM's own
  count gained +19 points but fails the spotted-ball and multicolour-logo controls. Needs a different model, not a
  setting. Source: same, 0.18.
- **`crowd_range`'s 0.3 SAM 3 floor** was picked by looking; any change needs a fresh crowd set. Source: [docs/history.md](../docs/history.md),
  "`crowd_range` brackets crowds with SAM 3".
- **Adaptive threshold prevalence** (opt-in `adaptive_threshold`, default off). It provably drops a real but weaker
  second instance for some score pairs; how often real photos produce that pair was never measured. Low priority
  while it stays opt-in. Source: [archive/ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md](archive/ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md).

## `spatial_relations`

- **A 5 px contact tolerance for SAM 3 masks**, which stop a few pixels short of a hand. Gate T's guard was
  written wrongly and waits on a fresh held-object set (round 12b). Source:
  [archive/sam3_expansion_plan.md](archive/sam3_expansion_plan.md).

## `query_image`

Nothing open. Its plan is done ([archive/Query_Image_Improvement_Plan.md](archive/Query_Image_Improvement_Plan.md)).
Routing multiple-choice questions to `classify_image` was closed 2026-11-05 without a gate: only 62 of the 3,789
seen questions offer a choice (mostly "left or right"), the shipped tool already gets 55 of them right, and most of the
7 misses have "yes" as their label, which a pick among the options cannot give. At most 2-3 were winnable.

## `query_image(structured_analysis)` and the defect probe

- **DINOv2-base instead of large** (a quarter of the size; 0.709 vs 0.686 on round 10, within noise). Waits on a
  fresh reviewed AI-defect set, with the rule fixed before it runs. Source: [frontier_capability_gaps.md](frontier_capability_gaps.md),
  candidate 11.
- **Hand and face crops without Grounding DINO.** Swapping the locator means retraining the probe and re-scoring
  round 10. Source: [archive/sam3_expansion_plan.md](archive/sam3_expansion_plan.md) "Not in scope".
- **Specificity** is the open gap (66.7% of clean images left alone end to end). Fine-tuning DINOv2 itself is
  lower priority: more labels already made the linear probe worse on unseen images. Source: [docs/history.md](../docs/history.md), the defect
  probe sections; [archive/model_variants_and_loras.md](archive/model_variants_and_loras.md) section 5.

## Grounding DINO's remaining callers

`count_objects`, `spatial_relations`, the defect probe's crops and `query_image`'s cross-checks still run on it.
OWLv2 and SAM 3 both fail the literal-name counting controls (they return 0 boxes for "ball" or "rod"), and SAM 3
lost on held objects in `spatial_relations`. No candidate is in hand. Source:
[archive/detector_ocr_replacement_plan.md](archive/detector_ocr_replacement_plan.md) gates 1-2.

## Server and memory

- **M5, a cross-process memory lock**, only if the memory manager proves not enough. Probably moot: since
  2026-10-25 one shared HTTP server serves every client, so one manager sees every call. Source:
  [archive/memory_manager.md](archive/memory_manager.md).

## New capabilities (none built)

From the scoreboard's "Candidate additions" ([frontier_capability_gaps.md](frontier_capability_gaps.md); numbers
are its rows): 1 tables and document layout (RapidOCR's table and layout packages, Chinese-origin, so this needs
another route), 3 image diff (no new model), 4 chart to data table (DePlot), 5 UI element detection (OmniParser's
detector is AGPL), 7 handwriting OCR (needs handwriting test images), 8 multilingual OCR (PP-OCR's multilingual
recognisers are barred; waiting on round 6's language tags), 9 depth (Depth Anything V2 is ByteDance/HKU, so
barred; needs a non-Chinese model).

## Parked

- **Gemini as an optional backend** for `caption` and `query_image`. Built on branch `feat/gemini-optional-backend`
  (`9757a88`), never merged. It was written to replace Moondream2 and Florence-2, which LFM2.5-VL has since
  replaced in both tools, so it needs re-justifying before it is revived. Source:
  [archive/gemini-optional-backend.md](archive/gemini-optional-backend.md).
- **PaliGemma 2 mix 3B**, tested for counting 2026-10-15 and parked by the user (3B too big for the gain).

## Dropped on this review

These appear as open in archived plans but no longer apply:

- **Florence-2 caption LoRA (C4-4).** Florence-2 is no longer the default captioner.
- **Gemma 3 4B for `deep` (C4-3b), Moondream2 tiling (1C-b), Florence-2 beam changes (C4-2a, 1B).** `deep` and the
  default are LFM2.5-VL now.
- **`model_variants_and_loras.md` matrix items 1-4.** All measured or ruled out (its own status table, 2026-10-11).
- **OWLv2 and docTR as drop-in replacements.** Both failed sprint R gate 1; kept above only as the open
  replacement problems they belong to.
