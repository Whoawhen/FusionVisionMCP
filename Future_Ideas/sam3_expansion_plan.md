# SAM 3 beyond `detect_objects` on photos: `spatial_relations` and flat images

Drafted 2026-10-13, after SAM 3 shipped for `detect_objects` on photos (262b9b0). Each gate waits for the user's
go-ahead, and its rules are committed in its VisionBench script before any scored run.

**Status 2026-10-13: gate S passed and shipped (7bb783e).** `sam3_spatial_gate.py` (rules b364c03, results
ed64fa3): agree of 36, gd_sam2 25, sam3_sam2 27, sam3_mask 25; all located 35 and passed both synthetic
fixtures; the rule picked `sam3_mask` (faster by 0.7 s, ~15 s vs 8 s shipped). SAM 3 won separate 12-7 and lost
contact 6-9, every contact loss a held object where the best match was another instance of the name. Round 12
(VisionBench 87c1063, `worker_handoff/ROUND12_PLAN.md`) builds 171 Open Images test pairs to measure that and to
test an instance-pairing fix against unheld-instance controls. Gate F has not run.

## Why these two

- **`spatial_relations`** locates each named object with Grounding DINO (best-scoring box per name), then SAM2
  masks inside that box, then `geometry.relation` measures. SAM 3 does both steps from the name. If it holds up,
  this tool stops using Grounding DINO, and possibly SAM2.
- **Flat images** (screenshots, documents, charts, clip art; the domain router's flat branch) still go to
  Florence-2's grounding head for `detect_objects`. That head cannot say "not here": it boxed the absent object
  on 100% of the 20 flat phase 11 images (`detect_flat_ab.json`), with 51.1% recall. Grounding DINO at 0.35 was no
  better (95% absent boxed, 40.0% recall). SAM 3 was never measured on flat images.

## What exists to measure with, and its limits

| Set | Size | Labels |
|---|---|---|
| Open Images VRD relation fixtures (`open_images_relation_fixtures.py`) | 36 photos, 12 per relation (contain, touch, separate) | official VRD relationship plus both objects' boxes |
| Synthetic spatial fixtures (`tests/spatial_touch_separate.png`, `spatial_containment.png`) | 2 | exact by construction (colour discrimination, containment) |
| Phase 11 flat images (router's flat branch) | 20 | phase 11 object boxes and one labelled absent object each |

Both real sets are too small for a tune/confirm split, so **neither gate tunes anything**. Each candidate runs at a
setting fixed below, and the whole set is scored once. A promising but inconclusive result means more data, not a
re-run with new settings.

## Gate S: `spatial_relations` (Opus 5.5 to write the rules and script, Sonnet 5.5 to run and report)

**Variants, fixed now:**
- `gd_sam2`, as shipped: Grounding DINO best box at its shipped floor, SAM2 mask, `geometry.relation`.
- `sam3_sam2`: SAM 3's best-scoring box per name (floor `LOCATE_BOX_THRESHOLD`, 0.125), SAM2 mask in it. Swaps
  the locator only.
- `sam3_mask`: SAM 3's own instance mask for that best-scoring detection, with no SAM2. Its mask threshold is 0.5,
  as the gates' post-processing used.

**Measured on all 38 images:** both objects located; measured state agrees with the official relationship (the
existing `check_agreement` rule in `evaluate_spatial_relations.py`, unchanged); the two synthetic fixtures'
assertions as `test_server.py` pins them; seconds per call.

**Decision**, against `gd_sam2`:
- A variant **passes** if it agrees on at least as many of the 36 as `gd_sam2` (25 of 36 in the 2026 run, to be
  re-measured in the same session), locates both objects on at least as many, and passes both synthetic
  fixtures.
- `sam3_mask` is preferred over `sam3_sam2` only if it passes and is faster.
- If neither passes, `spatial_relations` stays as shipped.
- A tie passes on purpose: the goal is removing Grounding DINO without losing anything, not beating it on 36
  images.

Cost: about 38 x 3 runs of 7-10 s, under 20 minutes.

## Gate F: flat images (same labelling)

**Candidates, fixed now:** Florence-2 as shipped; SAM 3 at 0.125 (its photo floor, not re-chosen).

**Measured on the 20 flat images,** with `detect_flat_ab.py`'s grader unchanged: recall, precision, absent boxed.
Also, as controls: the synthetic clip-art shapes in `fixtures.py` (`ring`, `distractors`) asked for by literal name,
counting boxes against the known truth.

**Decision:**
- SAM 3 **advances** if absent boxed falls by at least 30 points (from 100%) and recall is no more than 10 points
  below Florence-2's.
- With 20 images that is a screen, not a ship decision. If SAM 3 advances, the next step is sourcing about 60
  more flat images (worker proposes, Claude builds, per the image policy), then a confirm run under the same
  rule.
- If it does not advance, the flat path stays on Florence-2.

Cost: about 20 x 2 runs, under 10 minutes.

## Not in scope here

- **The defect probe's hand and face boxes.** The probe was trained on Grounding DINO's crops, so swapping the
  locator means retraining it and re-scoring round 10. That's its own plan.
- **`count_objects`.** SAM 3 failed the counting controls in gate 2. It stays on Grounding DINO.
- **OCR.** PP-OCRv6 stays until a candidate comes within reach (gate 1: best trailed by 13 points). Windows'
  built-in OCR (`Windows.Media.Ocr`, Microsoft, local) is an untried candidate for a separate gate 0.

## Order

Gate S first, since it can remove Grounding DINO from a second tool. Then gate F. Each waits for the user's
go-ahead.
