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
| 11 | Faster defect probe: DINOv2-base instead of DINOv2-large | Same probe and rule, a quarter of the parameters (86M vs 300M) and faster on CPU | Very low (no new code: VisionBench `defect_backbones.py` already trains it) | Needs a **fresh** reviewed AI-defect test set: round 10 was used to pick large, so it can't also pick base. Fix the rule before running: switch to base if its AUC is no more than ~0.02 below large's | **Waiting on images** (user, 2026-10-09). Round 10: base 0.709 vs large 0.686, within noise; on the old 94 images large led (0.76-0.81 vs 0.68-0.75). Run it the next time AI-defect images are collected (SANA and Stable Diffusion give the most defects) |

## Replacing Chinese-developed models

**Decision (user, 2026-10-08):** no Chinese-developed models in new work, including models built on a
Chinese base model (e.g. Molmo-7B-D on Qwen2), and the two already shipping get replaced in time. Until
then, **don't tune anything heavily that relies on them**: no threshold sweeps, fine-tunes or
detector-specific heuristics on Grounding DINO or PP-OCRv6. Keep work that touches them light and
portable (labels and measurements that a replacement can reuse).

| Shipping model | Developer | What depends on it | Alternatives to evaluate | Notes |
| --- | --- | --- | --- | --- |
| Grounding DINO tiny | IDEA Research (Shenzhen) | `detect_objects` (photos), `count_objects`, `spatial_relations` (boxes for SAM2), the defect probe's hand/face crops, `_check_anatomy` part count, `query_image` cross-checks | **OWLv2** (Google, Apache-2.0): open-vocabulary boxes with scores, the closest drop-in. **SAM 3** (Meta): text-prompted segmentation, could replace detector + SAM2 in one model (check licence and CPU speed). **MediaPipe** hand and face detectors (Google, already in `.venv`) for the defect probe's crops only. Florence-2 grounding (Microsoft, already loaded) has no scores, so it can't say "not here" | Exclude YOLO-World (Tencent). Re-run VisionBench phases 4, 11 and the COCO counting set before switching; the 0.35 score floor and counting thresholds were tuned for Grounding DINO and will need re-measuring, not copying |
| PP-OCRv6 English (via RapidOCR) | Baidu (PaddleOCR); RapidOCR packaging also Chinese-origin | `ocr`, `caption(auto_verify_text)` | **docTR** (Mindee, France, Apache-2.0): detection + recognition, ONNX export, the strongest candidate. **EasyOCR** (Jaided AI, Thailand): already installed as the non-English fallback, but 37.5% vs 53% on TextVQA photos. Tesseract (Apache-2.0): weak on photos. TrOCR (Microsoft): recognizer only, would pair with a detector | The layout code (`reading_blocks`, column zoning, sideways re-read) is engine-independent and carries over. Gates: phase 14 rendered pages (97.3% verbatim), TextVQA photos (58.5%), and empty on text-free images |

Not affected: Florence-2 (Microsoft), Moondream2 (SigLIP + Phi, US), CLIP and the LAION predictor
(OpenAI / LAION), SAM2.1 (Meta), MUSIQ (Google), EasyOCR (Jaided AI; its CRAFT detector is Naver, Korea).

## How we'll know

VisionBench phase 11 scores every tool on 400 generalist images (game UI, screenshots,
documents, charts, AI-generated images, artwork, photos). Its labels come from a
frontier-class model, so it doubles as a rough frontier comparison: how close
FusionVision gets to the labels' own repeatability ceiling (about 90% for key phrases,
75% for caption keywords) shows where it is weakest, kind by kind. Add the per-kind
numbers to the Evidence column as runs come in. The first full 400-image run finished
2026-10-04 (8.5 h on the CPU, 0 tool errors).

## What "good" means, before any fine-tuning

Model variants, LoRAs and fine-tunes wait until the whole suite is "good". Since 2026-10-09 (user decision)
each tool is compared with the best frontier model's figure rather than a fixed target; the fixed targets
agreed on 2026-10-02 are in git history (before this change). A tool whose failures are caught by a flag or
stated plainly in its description still counts in its favour. The plan (2026-10-09): work the 🔻 rows one at
a time; once they are near par and progress stalls, the user decides when to move on to other models and
fine-tunes. There is no fixed bar; the ~5-point par band below is the reference.

| Tool | Model(s) | Now | Frontier model (best estimate) | Difference |
| --- | --- | --- | --- | --- |
| `ocr`, clean screens and documents | PP-OCRv6 English (RapidOCR, ONNX) + geometric column splitting; EasyOCR for non-English | 97.3% of lines verbatim (phase 14) | **E** ~97–99%; **P** DocVQA 91–95% (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro). About parity | ✅ par: 0 to −2 points |
| `ocr`, real images by line | same as above | 76.2% (label ceiling ~80%) | **L** ~80%: the labels are a frontier model's reading, and they agree with themselves about that often | ✅ par: about −4 against the label ceiling |
| `ocr`, photos (TextVQA) | same as above | 58.5% (was 53%; detector box_thresh 0.4, 2026-10-03) | **P** 76.5–80.8% on TextVQA's own VQA metric (Claude 3.5 Sonnet 76.5, GPT-4o 77.4, Gemini 1.5 Pro 78.8, Claude 3 Opus 80.8); not the same metric as ours. **E** ~75–80% here | 🔻 weaker: −17 to −22 points |
| `ocr`, rotated / blurred | same as above | −5.7 / −52 points (rotation was −26) | **E** rotation −5 to −10; blur −20 to −30 (reads context, guesses words). Low confidence | ✅ rotation par (−5.7 vs −5 to −10 points lost); 🔻 blur weaker (−52 vs −20 to −30) |
| `caption` | Florence-2-large (default); Moondream2 (2025-06-21) long caption (`deep=true`); PP-OCRv6 text line (`read_text`, default on); Florence-2-large OCR head (`verify_text`) | 60.9% of label keywords on 280 held-out images with the text line (53.3% without; ceiling ~75%); `deep=true` 66.3% (61.5% without), ~3x slower. Before the text line, on all 400: photos 78%, artwork 76%, game UI 42%, AI images 48% | **L** ~75%: a frontier labeller's keyword agreement with itself | 🔻 weaker: about −14 points (default), −9 with `deep=true` |
| `caption(auto_verify_text)` | Florence-2-large + PP-OCRv6 on the cropped text regions | Share of the image's key text phrases that appear correctly in the caption: 25.5% before the correction, 24.5% after (20 text images, phase 12 after fixes; it was 20.8% before them). Triggers on 70% of text images and 20% of text-free ones | **E** ~60–75%: a frontier model reads the text while it captions, so key text lands in its caption about as often as it reads text at all (TextVQA 76–81%, discounted because a caption doesn't quote everything) | 🔻 weaker: about −35 to −50 points, and the correction itself costs ~1 point instead of helping |
| `detect_objects` | Grounding DINO tiny (photos, 0.35 score floor); Florence-2-large (clip art) | 59.2% recall, 42.7% precision on 400 images (app/web 34%); absent object boxed 70% | **E** recall 60–70% with looser boxes, precision ~70%; boxes an absent object 10–20% of the time (it can just say "not there"). Frontier is clearly ahead on absent objects | ✅ recall par (−1 to −11); 🔻 absent objects weaker (boxed 70% of the time vs 10–20%) |
| `dense_region_caption` | Florence-2-large | 74% box recall (phase 7) | **E** ~60–70%: frontier boxes are less tight | ⭐ beats: +4 to +14 points of box recall |
| `count_objects` | Grounding DINO tiny + Florence-2-large dense regions (consensus) + silhouette maths | 57.3% exact on 75 held-out COCO scenes of 3 or fewer (1: 85%, 2: 50%, 3: 24%); 41.6% on 400 generalist images (MAE 3.4; screenshots and game UI 24%). `separable` "no" was right 2/12 on predicted singles, now 2/5 | **E** 80–90% on scenes of 3 or fewer; **P** compositional counting of simple shapes: GPT-4o 35%, Gemini 2.5 Flash 47% (VLMCountBench). Frontier is ahead on small counts, weak on large or mixed ones | 🔻 weaker: −23 to −33 points on scenes of 3 or fewer |
| `query_image` | Moondream2 (2025-06-21) | 78.8% answer contained on 400 images; position 69.7% (n=76; the 0% was a 25-image sample) | **E** ~88–92%. **P** TextVQA as above (~77–81% vs Moondream2's 48% exact here). The largest gap | 🔻 weaker: −9 to −13 points |
| `spatial_relations` | Grounding DINO tiny + SAM2.1 Hiera-small + geometry maths | contact 92%, containment 83.8% (was 67.6%; `enclosed_by`, 2026-10-03) | **E** 75–85% as a yes/no judgement, and no masks or distances at all. FusionVision ahead | ⭐ beats: contact +7 to +17 points, containment par; frontier models give no masks or distances |
| `score_aesthetics` | CLIP ViT-L/14 + LAION improved aesthetic predictor; MUSIQ (ONNX, optional `[iqa]`); optional Ollama reasoner | ρ 0.49 | **E** ρ 0.5–0.65 zero-shot; trained scorers (Q-Align) do better than either. About parity | ✅ par: ρ 0.49 vs 0.5–0.65 |
| `critique_composition` | CLIP ViT-L/14 + LAION; Florence-2-large (subject box); Moondream2 (2025-06-21) (explanation on low scores) | token F1 27% against label critiques (phase 7, 25 images); 25% scored like for like (against two of the three references) | **M** 29–34%: each frontier critique (Claude, Gemini, GPT) scored against the other two | 🔻 weaker: about −7 points (measured 2026-10-10; the old ~40% estimate was too high) |
| AI-defect checks | DINOv2-large + logistic defect probe on Grounding DINO tiny hand/face crops (CLIP ViT-L/14 until 2026-10-09); EXIF metadata; EasyOCR (gibberish text); Moondream2 (2025-06-21) answer | **Probe alone, round 10 (299 user-graded images, six generators it never trained on, 154 defective): 44.8% of defective caught, 84.1% of clean left alone, AUC 0.686** (the CLIP probe it replaced: 47.4% / 68.3%, AUC 0.612). Full tool on the older 94-image set, through the real server: 68.1% / 76.6% with DINOv2 (72.3% / 63.8% with CLIP). History: part count alone near chance (38.8% / 70.0% on round 7, 2026-10-05); first CLIP probe 59.7% / 66.7% (2026-10-06); retrained with the user's crop calls 72.3% / 63.8% (2026-10-08); round 9 as extra training data made it worse | **M** against the user's labels: Gemini 3.8 Flash worker drafts 17% / 98% (round 10, 300 images, 2026-10-09); Claude Opus 5.5 drafts 42% / 98% (round 9, 246 images) and 38% / 81% (phase 11); Gemini 100% / 52% (phase 11, 39 images, judging any glitch). Frontier models either miss most defects or flag most clean images; none is balanced. **P** zero-shot VLMs need scaffolding for this; ArtifactLens reports +45% from it | ✅ par, mixed: ahead of Gemini 3.8 Flash on the same 300 images (balanced accuracy 64.5% vs 57.5%); behind Claude Opus 5.5's drafts on round 9 (70%, a different set) |

**Model sizes** (parameters, approximate): Florence-2-large 0.77B, Moondream2 1.9B, CLIP ViT-L/14 0.43B, DINOv2-large 0.30B (defect probe),
Grounding DINO tiny 0.17B, SAM2.1 Hiera-small 46M; PP-OCRv6 and MUSIQ are small ONNX models. All run on CPU
by default. For scale, the frontier models in the last column are estimated in the hundreds of billions.

**Frontier column key.** **M** measured here, on the same images and labels. **L** the label ceiling: the
labels were written by a frontier-class model, so its agreement with itself stands in for its score. **P**
a published benchmark, usually a different metric from ours. **E** an informed estimate, not measured.
Only **M** is a head-to-head. A scripted GeminiVisionMCP run on the AI-defect set and a phase 11 sample would
turn the main **E** rows into **M** (it sends those images to Google's API).

**Difference column key.** ⭐ **beats** the frontier figure by more than ~5 points; ✅ **par**, within ~5 points
or inside the frontier's estimated range; 🔻 **weaker** by more than ~5 points. Against an **E** or **L** figure
the difference is only as good as the estimate. Where a tool has two measures that disagree, both are marked.

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
- 2026-10-08 (night): defect probe retrained on round 7 plus the user's 216 crop calls and shipped: 72.3% / 63.8% on the 94 unseen images (was 63.8% / 63.8%). Round 9 as extra training data lowered it to 51.1% / 76.6%, so the CLIP crop probe has plateaued; next is a different backbone (DINOv2) or landmarks (MediaPipe). Confirmed through the real server (VisionBench ai_defect_e2e.py): identical, probe flags 94/94 image for image.
- 2026-10-09: backbone experiment, research only (VisionBench defect_backbones.py, same crops, labels and rules). DINOv2-base ties CLIP (unseen-image AUC 0.68-0.75 vs 0.741). DINOv2-large is above CLIP in all 8 variants (0.76-0.81), best +0.069 with a 95% interval of -0.02 to +0.16 on 94 images: promising, not proven, and phase 11 stays weak (~0.6) for every backbone. MediaPipe landmarks: no signal on hands (landmarked on 17% of crops either way), weak on faces (AUC 0.62). Next for AI defects: more independent test images before any backbone switch.
- 2026-10-08 (later): no Chinese-developed models in new work, and Grounding DINO (IDEA Research) and PP-OCRv6 (Baidu) to be replaced in time; no heavy tuning on either meanwhile (user decision). Alternatives listed in "Replacing Chinese-developed models". Caption candidates narrowed to Gemma 3 4B, PaliGemma 2, Phi-4-multimodal, SmolVLM2, newer Moondream and MolmoE-1B.
- 2026-10-09: round 10 (300 images the user graded, six generators no probe trained on) decided the defect probe's backbone by a pre-registered rule: DINOv2-large replaces CLIP (AUC 0.686 vs 0.612). DINOv2-base did as well (0.709) and is idea 11, waiting on a fresh test set. Caption sprint C1 built: CHAIR-style hallucination on 300 unused COCO images, and blind claim grading of 60 tune images by the Gemini worker.
- 2026-10-09 (later): order for the 🔻 rows, agreed with the user. Caption first, as planned (C1 hallucination check, then the C2 model bake-off: non-model fixes gained only length). Then `query_image` and `critique_composition` (prompting and routing, no new model). `detect_objects` absent objects, `count_objects` and `ocr` on photos sit on Grounding DINO and PP-OCRv6, which are slated for replacement, so they are fixed by replacing those models rather than tuning them.
- 2026-10-09 (night): DINOv2-large probe confirmed through the real server (VisionBench `ai_defect_e2e.py`, fresh server, CPU): probe flags match the offline scoring on all 94 images. Full tool on those 94: 68.1% of defective images flagged, 76.6% of clean left alone (CLIP probe 72.3% / 63.8%); probe alone 57.4% / 80.9%, image AUC 0.81. Round 10's 44.8% / 84.1% stays the headline, since those generators are unseen by every probe.
- 2026-10-09 (night): caption sprint C1 done, both halves; these are C2's baselines. **COCO, 300 unused photos** (VisionBench `caption_hallucination.py`, CHAIR-style against COCO's object labels): Florence-2 detailed (the default) names an object that isn't there 12.2% of the time (CHAIR-i; 23.3% of captions have at least one), recall 71.8%, median 7.5 s. Moondream2 long (`deep=true`) 17.1% / 30.3%, recall 77.2%, median 25.7 s; cut to Florence-2's length it is 13.5% / 20.3% at recall 63.0%, so its extra length buys recall at the cost of more invented objects. Most invented by both: dining table, person, car. **Claims on 60 generalist images** (`caption_claims.py`, graded by Gemini, corrected by the user's blind 80-card spot-check, where Gemini's "absent" held 69% of the time and "present" 19/19): about 27% of Florence-2's claims are wrong and about 18% of Moondream2's. So on photos Moondream2's long caption invents slightly more, and on screenshots, documents and artwork it is the more accurate of the two. Next: C2 gate 0 (model facts, from the worker's research), then the bake-off.
- 2026-10-10: caption sprint C2 ended with nothing to ship. Gate 0 (worker research, checked): no Chinese origins; Moondream2 has no revision newer than our pin; Moondream 3.1 has no CPU runtime (Photon needs an NVIDIA or Apple GPU). Gate 1 (20 images, CPU): Gemma 3 4B too slow (79 s bf16, 66 s int8), PaliGemma 2 at fp32 too big (15 GB; bf16 6.7 GB passes). Gate 2, tune split of 120 (card prompts fixed before running): Moondream2 long 67.8%, SmolVLM2 2.2B 58.6% (104 words, 34 s), Florence-2 55.9%, PaliGemma 2 DOCCI 52.1%, PaliGemma 2 mix 29.8% (12 words). Nothing beats Moondream2 for `deep`, and the only candidate fast enough for the default slot (PaliGemma mix) recalls about half of what Florence-2 does. COCO hallucination runs stopped once they could no longer change the decision. Caption stays Florence-2 + Moondream2; untried: Gemma 3 through a 4-bit GGUF runtime, Moondream 3.1 on GPU. PaliGemma 2 mix's `detect` boxes remain a possible Grounding DINO replacement, not yet measured.
- 2026-10-10: caption sprint C3 shipped `caption(read_text=true)`, on by default: the `ocr` engine's verbatim text appended as `Visible text: "..."` (40 words at most). No new model. Of the 447 tune keywords both captioners missed 122, about a third of them names written in the image. Variants fixed before running (VisionBench `caption_text_grounding.py`); confirmed once on the 280 held-out images: default 53.3% -> 60.9% (+7.7, CI 6.0-9.5; 68 images better, 0 worse), `deep` 61.5% -> 66.3% (+4.7, CI 3.3-6.3; 41 better, 0 worse). No absent object came from the text. Lost on tune and not shipped: asking Moondream2 for the image type first (-9.2, -6.0 with the OCR text in the prompt; its answers are a third the length), and a language tag (no recall, ~1 in 4 wrong). Next for caption: the image-type misses (CLIP type label, option C) and the "concept" misses on AI images, which need a smarter model.
- 2026-10-10: option C, an `Image type:` label from zero-shot CLIP (already loaded), not shipped. Recall held on the 280 held-out images (+3.2 default, +3.5 `deep`, none worse), but the guard fixed in advance failed: labels agreed with the image's kind 83.0% of the time against a 90% floor (91.4% on tune, after one revision that dropped "3D render"). The misses are real false claims: everyday photos called a "table of data" or "bar chart", game screens called a "computer desktop". VisionBench `caption_type_label.py`.
- 2026-10-10: concept misses (AI images' "work-life balance concept" and the like): asking Moondream2 "what idea, theme or message does this image convey?" and adding `Theme:` to `deep` gained +1.3 on tune (CI 0.4-2.5; AI images 50.0 -> 51.9) but costs ~22 s, putting `deep` over its 45 s limit, so the confirm split was not run (VisionBench `caption_theme.py`). Code-only caption work has plateaued here: what is left (concepts, specialist knowledge, image types without false labels) needs a stronger model than CPU allows today.
- 2026-10-10: `query_image` sprint Q1 ended with nothing to ship (VisionBench `vqa_ocr.py`, 360 tune questions, variants fixed before running, ship rule: overall 95% CI above zero). Baseline 78.9%. Giving Moondream2 the `ocr` engine's text always: 78.9% (text questions 55.2 -> 57.3, but no-text questions 85.7 -> 71.4). Routed by the server's question classifier: 77.8%. Third and last variant, each OCR line tagged with its 3x3 grid cell, only for questions worded about text (tune showed text questions ask *where*: "top right corner"): 80.0%, +1.1 (CI -1.7 to 3.9), text 61.5%, no-text 76.2%, 14 fixed / 10 broken. Fails the rule, so the confirm split was not run. 59 of 96 text answers are in the OCR text, yet Moondream2 picks the right one only about half the time: the limit is the model reasoning over text, not reading it. Like caption, code-only `query_image` work has plateaued; the next gain needs a stronger VQA model.
