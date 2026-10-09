# FusionVisionMCP vs a frontier vision model

A working list of what FusionVisionMCP can and can't do compared with a frontier
multimodal model (GPT, Gemini or Claude class), and what might close each gap locally.
Started 2026-09-30 from a static review of the tools; update it as VisionBench produces
numbers.

## The framing

FusionVisionMCP is a toolkit for a calling model that does the reasoning. It doesn't have
to match a frontier model everywhere. A gap is worth closing locally when a small model or
plain pixel maths can give the caller a **fact** it can't get otherwise: a position, a
count, exact text, a measurement.

## Capability table

| Capability | FusionVision today | Gap | Evidence |
| --- | --- | --- | --- | --- |
| Locate, count, segment, measure contact/containment | Grounding DINO + SAM2, `spatial_relations` | **Ahead.** Frontier models don't return masks or contact measurements. | Phase 4 (verified labels): contact 91.9%, containment 67.6% (the coffee-cup ring failure) |
| Image quality / aesthetics | LAION, MUSIQ, composition critique | On par for comparing similar images | Phase 7 (2026-10-01): Spearman rho 0.49, critique token F1 27%, dense-caption box recall 74% |
| Printed text (OCR) | PP-OCRv6 English, column splitting | Good on English print. **No handwriting, formulas or strong multilingual models.** | TextVQA answer recall 53%. Phase 14 (2026-10-01), 60 rendered pages with exact line boxes: 98.7% of lines located, 95% of those verbatim, 98.8% character accuracy; the weak template is slides (47% verbatim: a false column split cuts words in half). Real images by line (phase 14 sourced): 75% recall against a ~80% label ceiling; documents 64%, signage 63%, non-English 71%. Phase 13: blur costs 52 points of key-phrase recall. Phase 11 full (400 images, 2026-10-04): key phrases 74.9% (screenshots 89%, charts 87%, artwork 15%, comics 46%); empty on 87.1% of text-free images |
| Document structure (tables, forms, key-value fields) | Plain lines of text | **Missing.** No table output, no form fields. | — |
| Charts and diagrams (read values, trends) | Caption + OCR only | **Missing** | — |
| Screenshots / UI (elements, "where is the Save button") | OCR boxes can find text labels | **Partial.** No detector for buttons or icons. | Phase 11 full: OCR key phrases 89% on app/web, 76% on game UI; `detect_objects` recall 34% on app/web, 56% on game UI; an absent object is boxed 70% of the time across kinds |
| Open visual Q&A, reasoning | Moondream2 | **Largest gap.** A small model paraphrases and misses details. | TextVQA: 48% exact, 77% answer contained. Phase 11 full: 78.8% contained (yes/no 98%, colour 87%, count 70%, position 70%, reading text 58%; says 'no text' correctly 89%). Phase 12: `check_consistency` catches 25% of wrong answers, quiet on 94% of right ones |
| Naming a species, landmark, style or scene type | CLIP powers only `style_context` | **Missing as a tool**, though CLIP is already loaded | — |
| Comparing two images (what changed) | `compare_with` gives scores only | **Missing** general diff | — |
| Depth, camera angle, foreground/background | None | Missing | — |
| AI-generation defects | `structured_analysis` (anatomy) + EXIF check | **Weak** | Phase 10 (2026-10-01): 6/13 sensitivity, 11/13 specificity. Phase 11 full: 14/24 sensitivity, 9/16 specificity |
| Video | None | Out of scope for now | — |

## Candidate additions, in rough order of value per cost

| # | Idea | Approach | Cost | Open questions | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Tables and document layout | RapidOCR's table-recognition and layout-detection packages (ONNX), which fit the current stack | Medium | Output format for tables (Markdown? cells with boxes?) | Idea |
| 2 | Zero-shot classification tool ("which of these labels fits?") | The CLIP model already loaded for aesthetics | Very low | Label sets: caller-supplied only, or built-in sets too? | Idea |
| 3 | Image diff ("what changed between A and B?") | Pixel and structural comparison, then captions on the changed regions; no new model | Low | Alignment when the images are different sizes or crops | Idea |
| 4 | Chart reading (chart → data table) | A small chart-to-table model such as DePlot | Medium | Licence check; memory | Idea |
| 5 | UI element detection | OmniParser-style icon/button detector | Medium | **Licence: OmniParser's icon detector is AGPL** | Idea |
| 6 | Stronger local VQA | Replace Moondream2 with a larger local VLM | High (memory) | Overlaps with the parked Gemini-backend branch; plan together | Parked |
| 7 | Handwriting OCR | A handwriting recogniser such as TrOCR | Medium | Needs handwriting test images (none in VisionBench yet) | Idea |
| 8 | Multilingual OCR | PP-OCR multilingual recognisers | Low-medium | Worker round 6 is tagging languages, which will show how much non-English text there is | Waiting on data |
| 9 | Depth / scene layers | A small monocular depth model (e.g. Depth Anything V2 Small) | Medium | Licence of the chosen size; what output a caller needs | Idea |
| 10 | Closer look at a region ("inspect") | Optional `region=[x1,y1,x2,y2]` on `caption`, `query_image` and `ocr`: the caller (which can reason about where to look) points, the tool crops so the region gets the model's full input resolution (Florence-2 resizes every image to 768 px, so a 1080p HUD's minimap is a few pixels) | Low | Test first on phase 11: VQA and caption keywords on the whole image vs a crop around the labeled object's box; game UI (worst kind) is the likeliest gain | **Tested for caption, no gain** (2026-10-04, VisionBench `region_crop_pilot.py`): on 60 labeled objects the caption named the object 33% on the whole image vs 28% on a crop (control 3%); small objects 21% vs 9%, as the crop loses the context that identifies them. Untested for `ocr` and `query_image` |

## How we'll know

VisionBench phase 11 scores every tool on 400 generalist images (game UI, screenshots,
documents, charts, AI-generated images, artwork, photos). Its labels come from a
frontier-class model, so it doubles as a rough frontier comparison: how close
FusionVision gets to the labels' own repeatability ceiling (about 90% for key phrases,
75% for caption keywords) shows where it is weakest, kind by kind. Add the per-kind
numbers to the Evidence column as runs come in. The first full 400-image run finished
2026-10-04 (8.5 h on the CPU, 0 tool errors).

## What "good" means, before any fine-tuning

Model variants, LoRAs and fine-tunes wait until the whole suite is "good". A tool is
good when it meets its target **or** its failures are caught by a flag or stated plainly
in its description. Targets (agreed 2026-10-02 as reasonable starting points, not fixed):

| Tool | Model(s) | Now | Good | Frontier model (best estimate) |
| --- | --- | --- | --- | --- |
| `ocr`, clean screens and documents | PP-OCRv6 English (RapidOCR, ONNX) + geometric column splitting; EasyOCR for non-English | 97.3% of lines verbatim (phase 14) | ≥ 95% — met | **E** ~97–99%; **P** DocVQA 91–95% (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro). About parity |
| `ocr`, real images by line | same as above | 76.2% (label ceiling ~80%) | ≥ 75% — met | **L** ~80%: the labels are a frontier model's reading, and they agree with themselves about that often |
| `ocr`, photos (TextVQA) | same as above | 58.5% (was 53%; detector box_thresh 0.4, 2026-10-03) | ≥ 60% | **P** 76.5–80.8% on TextVQA's own VQA metric (Claude 3.5 Sonnet 76.5, GPT-4o 77.4, Gemini 1.5 Pro 78.8, Claude 3 Opus 80.8); not the same metric as ours. **E** ~75–80% here |
| `ocr`, rotated / blurred | same as above | −5.7 / −52 points (rotation was −26) | rotation ≤ −5 — essentially met; blur documented as a limit | **E** rotation −5 to −10; blur −20 to −30 (reads context, guesses words). Low confidence |
| `caption` | Florence-2-large (default); Moondream2 (2025-06-21) long caption (`deep=true`); Florence-2-large OCR head (`verify_text`) | 54.1% of label keywords on 400 images (ceiling ~75%); photos 78%, artwork 76%, game UI 42%, AI images 48%. `deep=true` (Moondream2 long): 61.5% vs 53.3% on 280 held-out images, ~3x slower | ≥ 60% — met with `deep=true`; the fast default stays at ~54% | **L** ~75%: a frontier labeller's keyword agreement with itself |
| `caption(auto_verify_text)` | Florence-2-large + PP-OCRv6 on the cropped text regions | lowered key phrases 25.5% → 20.8% | never makes a caption worse | n/a: a frontier model reads the text while captioning |
| `detect_objects` | Grounding DINO tiny (photos, 0.35 score floor); Florence-2-large (clip art) | 59.2% recall, 42.7% precision on 400 images (app/web 34%); absent object boxed 70% | ≥ 60% recall — essentially met; limit stated in description — stated | **E** recall 60–70% with looser boxes, precision ~70%; boxes an absent object 10–20% of the time (it can just say "not there"). Frontier is clearly ahead on absent objects |
| `dense_region_caption` | Florence-2-large | 74% box recall (phase 7) | — | **E** ~60–70%: frontier boxes are less tight |
| `count_objects` | Grounding DINO tiny + Florence-2-large dense regions (consensus) + silhouette maths | 57.3% exact on 75 held-out COCO scenes of 3 or fewer (1: 85%, 2: 50%, 3: 24%); 41.6% on 400 generalist images (MAE 3.4; screenshots and game UI 24%). `separable` "no" was right 2/12 on predicted singles, now 2/5 | ≥ 50% exact on scenes of 3 or fewer — met | **E** 80–90% on scenes of 3 or fewer; **P** compositional counting of simple shapes: GPT-4o 35%, Gemini 2.5 Flash 47% (VLMCountBench). Frontier is ahead on small counts, weak on large or mixed ones |
| `query_image` | Moondream2 (2025-06-21) | 78.8% answer contained on 400 images; position 69.7% (n=76; the 0% was a 25-image sample) | ≥ 75% — met | **E** ~88–92%. **P** TextVQA as above (~77–81% vs Moondream2's 48% exact here). The largest gap |
| `spatial_relations` | Grounding DINO tiny + SAM2.1 Hiera-small + geometry maths | contact 92%, containment 83.8% (was 67.6%; `enclosed_by`, 2026-10-03) | containment ≥ 80% — met | **E** 75–85% as a yes/no judgement, and no masks or distances at all. FusionVision ahead |
| `score_aesthetics` | CLIP ViT-L/14 + LAION improved aesthetic predictor; MUSIQ (ONNX, optional `[iqa]`); optional Ollama reasoner | ρ 0.49 | ≥ 0.5 | **E** ρ 0.5–0.65 zero-shot; trained scorers (Q-Align) do better than either. About parity |
| `critique_composition` | CLIP ViT-L/14 + LAION; Florence-2-large (subject box); Moondream2 (2025-06-21) (explanation on low scores) | token F1 27% against label critiques (phase 7) | — | **E** ~40%; the labels are a frontier critique |
| AI-defect checks | CLIP ViT-L/14 + logistic defect probe on Grounding DINO tiny hand/face crops; EXIF metadata; EasyOCR (gibberish text); Moondream2 (2025-06-21) answer | sensitivity 46%, specificity 85% (26 images); phase 11 full: 58%, 56% (40 images). **Reviewed rounds 7/8 (2026-10-05): 38.8% / 70.0% on 149 images, flagging defective 38.8% vs clean 30.0%, near chance; only 8 of 19 hits on the defect's own part, 47 of 69 anomalies are leg counts. Held-out round 8: 23.1% sensitivity**. **With the defect probe and the part count demoted (2026-10-06), on 95 images it never trained on (round 8, phase 10, phase 11): 59.7% / 66.7% (was 41.9% / 72.7%); probe alone 51.6% / 69.7%; round 7 CV, probe alone: 63.3% / 85.0%**. **2026-10-08, phase 11 graded by the user's anatomy verdicts (94 images, 47/47): 63.8% / 63.8%; retrained with round 7 crop reviews 72.3% / 63.8% (not shipped yet)** | ≥ 60% / ≥ 85% — rules can't reach it: 5 of 7 misses are omissions, which a part count can't tell from occlusion. First candidate for a model variant; worker round 7 is building a 150-image set | **M** against the user's labels, 2026-10-08: Claude Opus 5.5 first-pass drafts 42% / 98% (round 9, 246 images) and 38% / 81% (phase 11); Gemini 100% / 52% (phase 11, 39 images, judging any glitch, not only anatomy). Two frontier models, opposite trade-offs, neither balanced. **P** zero-shot VLMs need scaffolding for this; ArtifactLens reports +45% from it |

**Model sizes** (parameters, approximate): Florence-2-large 0.77B, Moondream2 1.9B, CLIP ViT-L/14 0.43B,
Grounding DINO tiny 0.17B, SAM2.1 Hiera-small 46M; PP-OCRv6 and MUSIQ are small ONNX models. All run on CPU
by default. For scale, the frontier models in the last column are estimated in the hundreds of billions.

**Frontier column key.** **M** measured here, on the same images and labels. **L** the label ceiling: the
labels were written by a frontier-class model, so its agreement with itself stands in for its score. **P**
a published benchmark, usually a different metric from ours. **E** an informed estimate, not measured.
Only **M** is a head-to-head. A scripted GeminiVisionMCP run on the AI-defect set and a phase 11 sample would
turn the main **E** rows into **M** (it sends those images to Google's API).

Sources for **P**: [Qwen2.5-VL report](https://arxiv.org/pdf/2502.13923) (TextVQA, DocVQA, ChartQA for
GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro), [InternVL3.5 report](https://arxiv.org/pdf/2508.18265) (Claude 3
Opus), [VLMCountBench](https://arxiv.org/pdf/2510.04401) (counting),
[ArtifactLens](https://arxiv.org/pdf/2602.09475) (artifact detection), [OCRBench
v2](https://arxiv.org/pdf/2501.00321) (GPT-4o 47.6%, Gemini 1.5 Pro 51.6% English: hard text-localisation
tasks also trouble frontier models).

**Stop rule.** If hard work on a target stops making tangible progress, or the changes
start fitting the test rather than improving the tool, stop and try a different approach
(other data, model or method) instead of grinding. The line-box finder is the example:
pixel heuristics plateaued at ~57% agreement, and rendering pages in a browser gave
exact boxes outright.

## Notes and decisions

- 2026-09-30: list created. Nothing scheduled yet; the Gemini backend branch stays parked.
- 2026-10-01: Evidence column filled from VisionBench phases 7, 10, 11 (25-image sample), 12, 13 and 14.
- 2026-10-02: "good" targets and the stop rule added; fine-tuning waits for them.
- 2026-10-03: OCR photos 53 → 58.5%, rotation −26 → −5.7, containment 67.6 → 83.8% (met). GPU benchmark runs crash the laptop on phase 11 (hypervisor I/O MMU fault); benchmarks run on CPU, GPU only for phases 7 and 10.
- 2026-10-03 (evening): counting target met on held-out COCO (57.3%); separable's weak-split "no" removed; AI-defect detection judged a model problem, not a rules problem (worker round 7 sourcing 150 images).
- 2026-10-04: first full phase 11 run (400 images, CPU). `query_image` met (78.8%, after a grader fix for "no text" questions); `detect_objects` recall essentially met (59.2%); caption 54.1% is the main open target besides AI defects. Phase 11's AI-defect numbers disagree with phase 10's specificity (56% vs 85%), more reason to wait for round 7's 150 images.
- 2026-10-05: caption decided on 280 held-out images (VisionBench `caption_confirm_analysis.py`). PromptGen v2.0 +2.9 points, all of it caption length (+0.3 when cut to the default's length), so it was not shipped: not worth a second 1.5 GB Florence-2. Moondream2 long +8.2 (CI 5.3–11.2) shipped as opt-in `caption(deep=true)`, at no extra memory; its gain is also mostly length (+1.9 length-matched), which is coverage a caller asking for detail wants. Region crops tested earlier: no gain.
- 2026-10-05: AI-defect answer key finished: round 7 (149 images from Commons and DiffusionDB, 49 defective, user-reviewed; the worker's draft had 14 false defects and 20 repeated images) is the tune set, round 8 (31 of the user's own DreamShaperXL images) the holdout. The shipping part count is near chance on it (VisionBench `ai_defect_baseline.py`). Next: the CLIP crop probe (AUC 0.88 on SynArtifact) against both rounds.
- 2026-10-05 (later): the SynArtifact-trained probe did not transfer (round 7 image AUC 0.65; it scored real photos as more defective than defects). Retrained on round 7's reviewed hand/face crops it did (round 8 image AUC 0.885) and shipped in `structured_analysis` as `defect_probe` (weights in `data/defect_probe.json`, ~15 KB; reuses the aesthetics CLIP). End to end on 95 unseen images: defective flagged 41.9% → 71.0%, clean left alone 72.7% → 54.5%. Specificity is now the gap; the part count costs ~12 points of it for ~11 of sensitivity.
- 2026-10-06: part count demoted to an observation (user decision). `structured_analysis` on the 95 unseen images: 59.7% of defective flagged, 66.7% of clean left alone (was 41.9% / 72.7% before the probe). Next for AI defects: specificity (85% target), and phase 11's Gemini-made defect labels, which nobody has reviewed.
- 2026-10-08: frontier column added to the "good" table (measured where possible, otherwise label ceilings, published benchmarks or estimates, each marked). Phase 11's AI-defect key replaced by the user's anatomy verdicts (8 defective of 39, where Gemini's had 24); round 9 (246 DiffusionDB images, 141 defective) reviewed for probe training.
