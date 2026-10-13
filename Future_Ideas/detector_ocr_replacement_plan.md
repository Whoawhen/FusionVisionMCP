# Replacing Grounding DINO and PP-OCRv6 (sprint R): candidates and bake-off plan

Drafted 2026-10-11. Nothing has been downloaded or run for this yet. Both models are to be replaced under
the no-Chinese-models rule, and four 🔻 scoreboard rows sit on them: `count_objects`, `detect_objects` on
absent objects, `ocr` on photos and `ocr` on blur.

## What each model does today

| Model | Used by | Current numbers to hold or beat |
|---|---|---|
| Grounding DINO tiny (IDEA, Shenzhen) | `detect_objects` on photos (0.35 floor), `count_objects` (0.15 floor, envelope filter), `spatial_relations` (best box per name), `structured_analysis` hand/face crops for the defect probe | 143 COCO scenes: recall 66.4% (≤3 objects) / 55.1% (crowded), precision 62.7%, absent category boxed 2.8%; plausible neighbour boxed 62% of 130; counting fixtures 9/10 positives, 8/8 controls; crowded COCO counts exact 13%; ~2 s per call |
| PP-OCRv6 English via RapidOCR (Baidu) | `ocr`, `caption` `read_text` and `auto_verify_text` | TextVQA 200 answer recall 58.5%; rendered pages 97.5% of lines; 100 held-out sourced images 76.2% of lines; text-free fixtures empty; blur −52 points |

## Candidates

Facts from the two worker research rounds (checked in `VisionBench-labels/worker_handoff/MODEL_RESEARCH_VERIFIED.md`),
plus a 2026-10-11 check of the Hugging Face API and docs.

**Detector**

| Candidate | Maker | Size | Licence | Why it is on the list |
|---|---|---|---|---|
| OWLv2 base / large (`google/owlv2-*-ensemble`) | Google | 155M / 438M | Apache-2.0 | Open-vocabulary boxes with scores, plus an objectness head; native in transformers |
| SAM 3 (`facebook/sam3`) | Meta | 860M (fp32) | SAM Licence, gated (manual approval) | Text prompt → every matching instance with a score and a mask. Its **presence head** is built to say "not here", the exact gap. Masks too, so it could also take over SAM2's part in `spatial_relations`. Native in transformers (`Sam3Model`). CPU speed unknown: a 1008 px ViT backbone, likely the slowest |
| Moondream2 detect (already loaded) | M87 Labs | — | Apache-2.0 | Free baseline: no scores, and it collapsed classes when counting before |

Ruled out as Chinese-developed: YOLO-World (Tencent), YOLOE (Tsinghua), MM-Grounding-DINO (OpenMMLab), OmDet-Turbo,
LLMDet, DINO-X / Rex-Omni (IDEA), and CountGD (Grounding DINO weights underneath, and already worse).

**OCR**

| Candidate | Maker | Licence | Why |
|---|---|---|---|
| docTR (detector + recognizer, PyTorch) | Mindee (France) | Apache-2.0 | Light and CPU-friendly; boxes and confidences per word; recognizer fine-tuning is supported. Detectors: LinkNet, DBNet, FAST, LW-DETR. Recognizers: CRNN, SAR, MASTER, ViTSTR, PARSeq, VIPTR |
| TrOCR scene-text (`microsoft/trocr-base-str`, `-large-str`) on docTR's detector boxes | Microsoft | None on the Hugging Face card (the `microsoft/unilm` code is MIT); weights are `pytorch_model.bin` only (pickle, no safetensors) | Transformer recognizer trained for scene text; may help the photo and blur rows. Line-level only, so it needs a detector. The licence gap needs settling before it could ship |
| Tesseract | Google / open source | Apache-2.0 | Cheap floor; known weak on photos |
| EasyOCR (installed) | Jaided AI (Thailand) | Apache-2.0 | The previous engine (37.5% on TextVQA); the floor any replacement must beat |
| PaliGemma 2 3B mix-448 `ocr` prompt (already downloaded) | Google | Gemma Terms | Text only, no boxes, slow on CPU; only as a fallback reader for photos, if at all |
| Surya | Datalab (US) | Weights free under $5M revenue or funding | Strong on documents, heavy on CPU; kept off the first shortlist |

**Architectures are fine (user, 2026-10-11).** The rule is about who developed and trained the model, so every
docTR architecture qualifies, including those from Chinese papers (DBNet, FAST, CRNN, SAR, MASTER, VIPTR, LW-DETR).

## Bake-off, in gates

Each gate's rule is committed in its script before the run, as with the caption sprints.

**Gate 0: smoke (Sonnet 5.5).** Load each candidate on CPU, run 10 images, record median latency and peak memory.
Out if a call takes more than 10 s median for the detector (Grounding DINO ~2 s) or 5 s per photo for OCR.

**Gate 1: tune half (Opus 5.5 designs, Sonnet 5.5 runs).** Each surviving candidate runs on the tune half of
the existing sets, and its score threshold is chosen on that half only.
- Detector: COCO 143 split in half (recall by density, precision, absent category boxed), the 130-image
  plausible-neighbour probe, the counting fixtures (positives and all 8 negative controls, `neg_spotted_ball`
  first).
- OCR: TextVQA 200 split in half, rendered pages, the 100 held-out sourced images, text-free fixtures, the
  phase 13 blur set.

**Gate 2: confirm half, run once, at the gate 1 threshold.** Because the old models have to go regardless, the
rule is *no worse*, not *better*: a candidate ships if, against the current model on the confirm half,
- its primary metric's paired 95% CI lower bound is above −3 points (detector: recall at IoU 0.5 on COCO;
  OCR: TextVQA answer recall), and
- every control holds: absent category boxed ≤ 5% and counting controls 8/8 (detector); text-free fixtures
  empty and rendered-page lines ≥ 96% (OCR).
Among candidates that pass, the one with the best primary metric wins. A gain on the 🔻 rows (absent
objects, crowded counts, photo recall, blur) is reported beside it.

**Gate 3: wire it in, then re-measure what depends on it.** Swap the engine behind the existing
`InstanceDetector` / OCR protocols, re-run phase 11 on CPU, and re-check the two downstream uses whose numbers
were fitted to the old outputs:
- the defect probe was trained on Grounding DINO's hand and face crops, so a new detector means re-cropping
  round 7, retraining, and re-scoring with `ai_defect_rescore.py` before the probe can move;
- `caption(read_text)` and `auto_verify_text` move with the OCR engine, so their tune numbers are re-run.

Grounding DINO and PP-OCRv6 stay installed until gate 3 passes for their replacement, so rollback is a
config change.

## Status

- 2026-10-11: user approved gate 0 and requested SAM 3 access (pending Meta's manual review). Gate 0 is
  VisionBench `replacement_gate0.py` (rules committed b63745d). The candidates run in a separate venv
  (`VisionBench/.venv-replace`: torch 2.14.1 CPU, transformers 5.19, docTR 1.1.0) because SAM 3 needs
  transformers 5 and docTR needs `opencv-python`. Shipping either one into the server would therefore mean a
  transformers 5 upgrade (Moondream2's pinned remote code must be re-checked) or an OpenCV dependency fix.
- 2026-10-12: **gate 0 done** (VisionBench `results/replacement_gate0.json`; 10 photos, warm, 18 threads, CPU idle).
  Detectors, median / max seconds per call and peak memory: Grounding DINO tiny 4.2 / 5.0 s, 1.4 GB (baseline);
  OWLv2 base 2.2 / 2.4 s, 1.2 GB; OWLv2 large 8.5 / 9.7 s, 2.2 GB; SAM 3 9.5 / 15.4 s, 4.3 GB (155 s to load). OCR:
  PP-OCRv6 1.3 / 4.2 s (baseline); docTR LinkNet+PARSeq 0.5 / 1.4 s; FAST+PARSeq 0.9 / 5.1 s; DBNet+PARSeq
  1.0 / 3.3 s; DBNet+ViTSTR 1.4 / 6.4 s; TrOCR-base-str 3.7 / 55 s; **TrOCR-large-str out** (6.7 s median, 130 s
  on a dense page). OWLv2 returns many overlapping boxes at 0.1 with no NMS, so gate 1 adds NMS before
  choosing its threshold.
- 2026-10-12: **gate 1 started** (VisionBench `replacement_gate1.py`, rules committed e7ff830 before any scored
  run). Tune half: 689 detector queries (63 COCO scenes plus their absent categories, 323 phase 11 objects, 171
  plausible-neighbour probes, 100 reliable counts, 18 synthetic fixtures) and 534 OCR items (92 TextVQA
  questions, 26 rendered pages, 51 sourced-line images, 155 phase 11 text images clean and blurred, 51 text-free
  images, 4 text-free fixtures). Primary metrics: pooled detect recall, pooled exact counts, and mean per-item OCR
  recall. Each candidate's thresholds are chosen on tune under the controls; at most two per role advance to gate 2.
- 2026-10-12: **gate 1 done, nothing advances under the committed rules** (VisionBench
  `results/replacement_gate1_tune.json`, 6259e6e).
  - **Detectors.** None gets a count threshold: no threshold holds all 8 negative controls (OWLv2 base at most
    3, large 2, SAM 3 1). They don't fragment the controls; they return **0 boxes** for "petal" on a plain disc,
    ball or rod, where Grounding DINO boxes anything as a petal. SAM 3 counts the distractor fixture's 3 pink
    circles exactly at every threshold. On real photos SAM 3 leads clearly. At its detect threshold (0.125):
    COCO recall 88.1 vs 55.8, phase 11 recall 65.0 vs 58.5, plausible-neighbour absent boxed 10.5% vs 62.6%,
    COCO absent 1.6% (tie). Its best count threshold (0.6) gets 52.0% exact vs 40.2%. OWLv2 large peaks at
    44.2% exact; OWLv2 base at 43.0%. Median seconds per query: SAM 3 10.2, OWLv2 large 9.7, OWLv2 base 2.1.
  - **OCR.** No docTR pair or TrOCR comes close to PP-OCRv6 (primary 67.8; TextVQA 59.8, rendered lines 97.5,
    blur 37.4). Best is FAST+PARSeq at 54.3, then DBNet+PARSeq 52.5 and DBNet+ViTSTR 50.8. None meets the
    text-free rule at any floor. This is a clear loss, not a threshold artefact.
  - Waiting on the user: either stop here, or amend the counting controls for literal detectors before gate 2.
- 2026-10-12: **tuned Grounding DINO vs SAM 3** (user asked what the comparison looks like without the
  no-Chinese-models rule; VisionBench `replacement_gd_tuned.py`, rules 6db6c30, result 0638b50, tune half,
  replayed from cache). The gate 1 selection rule picks detect 0.4 and count 0.425 (adaptive), holding 8/8
  controls (fixture positives 9/10 -> 7/10).
  - **Detection: SAM 3 still leads clearly.** Detect recall 75.7 vs 52.3 (+23.4, CI 16.5 to 29.8); boxes a
    plausible neighbour that isn't there on 10.5% vs 49.7% (-39.2, CI -47.4 to -31.0). No Grounding DINO
    threshold gets near both: at 0.55 it boxes 19.9% but its recall is 34.
  - **Counting: the shipped 0.15 count threshold is the problem.** At 0.425, Grounding DINO's exact counts go
    40.2 -> 55.6% on real photos. SAM 3 at its best threshold, ignoring controls, reaches 52.0% (-3.6, CI -8.9 to
    1.9). 0.15 was chosen on synthetic fixtures only. Not shipped: it is a Grounding DINO tune, which the
    no-Chinese-models rule bars, and it has not been confirmed on held-out images.
- 2026-10-12: **gate 2 done** (confirm half, run once; VisionBench `replacement_gate2.py`, rules a497117 before
  any run; results committed after it). Counting fixtures were asked for by literal name ("pink circle", "ball",
  "rod"). All differences are against shipped Grounding DINO, paired bootstrap 95% CIs.
  - **SAM 3 for `detect_objects`: passes.** Detect recall 76.4 vs 60.2 (+16.2, CI 10.5 to 21.9). Plausible
    neighbours boxed 9.0% vs 65.7% (-56.7, CI -64.5 to -48.8). COCO absent boxed 0.0% vs 3.8%. Its boxes are less
    precise (COCO 33.1 vs 83.1%, phase 11 22.3 vs 38.6%): it finds more, with more strays.
  - **SAM 3 for `count_objects`: fails.** No threshold held the 8 literal-name controls (at most 4). It never
    split one object into several: it returned 0 for "ball", "rod", "shape" and "blob". It counted all 10 positive
    fixtures exactly. For context only: at its best tune threshold (0.6) it gets 51.8% exact on confirm, vs 61.8%
    for Grounding DINO at 0.425 (CI -16.3 to -3.5).
  - **Grounding DINO count threshold 0.425: confirmed.** Exact counts 61.8% vs 43.8% (+18.0, CI 14.1 to 22.2),
    holding 8/8 controls under both prompts (fixture positives 8/10 -> 7/10). Its detect setting of 0.4 costs
    6.5 points of recall, so only the count threshold is worth taking. Shipping it is a Grounding DINO tune, which
    the no-tuning rule bars, so it needs the user's call.
  - Median seconds per query on confirm: Grounding DINO 3.4, SAM 3 7.3.
- Downloads: about 0.6 / 1.7 GB for OWLv2, about 3.4 GB for SAM 3, a few hundred MB for docTR, and the TrOCR
  checkpoints.
