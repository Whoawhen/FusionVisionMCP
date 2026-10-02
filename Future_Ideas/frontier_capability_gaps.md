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
| --- | --- | --- | --- |
| Locate, count, segment, measure contact/containment | Grounding DINO + SAM2, `spatial_relations` | **Ahead.** Frontier models don't return masks or contact measurements. | Phase 4 (verified labels): contact 91.9%, containment 67.6% (the coffee-cup ring failure) |
| Image quality / aesthetics | LAION, MUSIQ, composition critique | On par for comparing similar images | Phase 7 (2026-10-01): Spearman rho 0.49, critique token F1 27%, dense-caption box recall 74% |
| Printed text (OCR) | PP-OCRv6 English, column splitting | Good on English print. **No handwriting, formulas or strong multilingual models.** | TextVQA answer recall 53%. Phase 14 (2026-10-01), 60 rendered pages with exact line boxes: 98.7% of lines located, 95% of those verbatim, 98.8% character accuracy; the weak template is slides (47% verbatim: a false column split cuts words in half). Real images by line (phase 14 sourced): 75% recall against a ~80% label ceiling; documents 64%, signage 63%, non-English 71%. Phase 13: blur costs 52 points of key-phrase recall, a 90° turn 26 (no orientation handling). Phase 11 sample: empty on only 40% of text-free images |
| Document structure (tables, forms, key-value fields) | Plain lines of text | **Missing.** No table output, no form fields. | — |
| Charts and diagrams (read values, trends) | Caption + OCR only | **Missing** | — |
| Screenshots / UI (elements, "where is the Save button") | OCR boxes can find text labels | **Partial.** No detector for buttons or icons. | Phase 11 sample: OCR key phrases 87% on app/web, 50% on game UI; `detect_objects` recall 40% on app/web, and an absent object is still boxed 64% of the time across kinds (flat images take Florence-2's path) |
| Open visual Q&A, reasoning | Moondream2 | **Largest gap.** A small model paraphrases and misses details. | TextVQA: 48% exact, 77% answer contained. Phase 11 sample: 68% contained, position questions 0%. Phase 12: `check_consistency` catches 25% of wrong answers, quiet on 94% of right ones |
| Naming a species, landmark, style or scene type | CLIP powers only `style_context` | **Missing as a tool**, though CLIP is already loaded | — |
| Comparing two images (what changed) | `compare_with` gives scores only | **Missing** general diff | — |
| Depth, camera angle, foreground/background | None | Missing | — |
| AI-generation defects | `structured_analysis` (anatomy) + EXIF check | **Weak** | Phase 10 (2026-10-01): 6/13 sensitivity, 11/13 specificity |
| Video | None | Out of scope for now | — |

## Candidate additions, in rough order of value per cost

| # | Idea | Approach | Cost | Open questions | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Tables and document layout | RapidOCR's table-recognition and layout-detection packages (ONNX), which fit the current stack | Medium | Output format for tables (Markdown? cells with boxes?) | Idea |
| 2 | Zero-shot classification tool ("which of these labels fits?") | The CLIP model already loaded for aesthetics | Very low | Label sets: caller-supplied only, or built-in sets too? | Idea |
| 3 | Image diff ("what changed between A and B?") | Pixel and structural comparison, then captions on the changed regions; no new model | Low | Alignment when the images are different sizes or crops | Idea |
| 4 | Chart reading (chart → data table) | A small chart-to-table model such as DePlot | Medium | Licence check; memory | Idea |
| 5 | UI element detection | OmniParser-style icon/button detector | Medium | **Licence: OmniParser's icon detector is AGPL** | Idea |
| 6 | Stronger local VQA | Replace Moondream2 with a larger local VLM | High (memory) | Overlaps with the parked Gemini-backend branch; plan together | Parked |
| 7 | Handwriting OCR | A handwriting recogniser such as TrOCR | Medium | Needs handwriting test images (none in VisionBench yet) | Idea |
| 8 | Multilingual OCR | PP-OCR multilingual recognisers | Low-medium | Worker round 6 is tagging languages, which will show how much non-English text there is | Waiting on data |
| 9 | Depth / scene layers | A small monocular depth model (e.g. Depth Anything V2 Small) | Medium | Licence of the chosen size; what output a caller needs | Idea |

## How we'll know

VisionBench phase 11 scores every tool on 400 generalist images (game UI, screenshots,
documents, charts, AI-generated images, artwork, photos). Its labels come from a
frontier-class model, so it doubles as a rough frontier comparison: how close
FusionVision gets to the labels' own repeatability ceiling (about 90% for key phrases,
75% for caption keywords) shows where it is weakest, kind by kind. Add the per-kind
numbers to the Evidence column once the first phase-11 run is in. A 25-image
sample is in (2026-10-01); a full 400-image run is still to do.

## Notes and decisions

- 2026-09-30: list created. Nothing scheduled yet; the Gemini backend branch stays parked.
- 2026-10-01: Evidence column filled from VisionBench phases 7, 10, 11 (25-image sample), 12, 13 and 14.
