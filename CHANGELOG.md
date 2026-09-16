# Changelog

## v0.8.2

A correctness and cost release. The test suite could not be collected at all, which is how
three runtime-breaking defects reached a tagged release unnoticed; fixing collection came
first, and everything below was verified against a suite that actually runs.

### Fixed: tools that could not run

- **`detect_objects` raised `AttributeError` on every call.** `detection_policy` called
  `app.processor`, a field renamed to `app.florence2` some time earlier. `count_objects`
  hit the same branch whenever routing chose Florence-2, and `IdleProxy` loaded the model
  before raising, so the failure cost a full model load.
- **`query_image` could not load Moondream.** `attn_implementation="sdpa"` is rejected by
  `HfMoondream` across the whole supported transformers range; it now passes `"eager"`, as
  the error itself prescribes. Every VQA consumer was affected.
- **`ocr` emitted literal `
` characters** instead of newlines, corrupting every
  multi-line transcription in both the plain and `detail=true` payloads.
- **`pytest` aborted during collection** (3 errors, 0 tests run). Added
  `[tool.pytest.ini_options]` with `testpaths`, deleted two tests importing the removed
  `granite_docling` module, and renamed the root/benchmark probe scripts off the `test_*`
  prefix so collection no longer spawns a server or downloads weights.

### Fixed: five OCR tests asserted a hallucination

`test_ocr`, `test_ocr_url` and both PDF tests ran against a text-free fixture and asserted
`len(text) > 0` -- which passed under Florence-2's OCR head only because it emitted
something. EasyOCR correctly returns nothing. The text assertions now run against a
text-bearing fixture, the PDF tests pin the decode path, and a new
`test_ocr_textless_image_returns_no_text` locks in the anti-hallucination behaviour.

### Connect time: the torch-free import invariant, restored

Importing the package pulled in torch and all six model wrappers, reintroducing the
4.5s-warm/14.1s-cold connect cost that the invariant exists to prevent. Wrapper imports
moved back under `TYPE_CHECKING` with per-factory local imports; `detection_policy` now
takes `DEFAULT_BOX_THRESHOLD` from `constants` rather than from torch-importing
`grounding_dino`; `EasyOCREngine` takes a device *string* like every other wrapper.
Package import is now ~0.86s with torch absent from `sys.modules`, pinned by a
regression test.

### Removed: `Florence2SP` subprocess mode

Every call reconstructed the full model inside a fresh process, and it was selected only
by `--memory-mode persistent` -- so the mode documented as fastest reloaded Florence-2 on
every request. Removed along with `subprocess.py`, the `dill` dependency and the
`--cache-model` flag (**breaking**: use `--memory-mode persistent`). `IdleReleased` with
timeout 0 already provides a persistent in-process model.

### Idle release no longer fires mid-inference

The idle timer measured time since the last *attribute lookup* and was rebuilt -- a fresh
OS thread -- on every one. A call longer than the timeout was released while still
running, dropping the model and running `gc.collect()` plus a Windows working-set trim
during inference. One self-rearming timer now measures time since the last *completed*
call, an in-flight guard defers release, and `IdleProxy.release()` no longer loads the
model in order to release it.

### Other correctness

- `adaptive_threshold` filtered competing gaps by value, so tied maximal gaps both
  vanished and any evenly-spaced score list read as a confidence cliff. Now filtered by
  index.
- `geometry`: a mask too small to measure returned `nan` from `elongation`, and
  `nan > threshold` is False, so it *passed* the rosette gate instead of being rejected.
  `relation` now rejects mismatched mask shapes instead of raising a raw broadcast error.
- The reasoner treated any well-formed JSON as well-shaped; a list, a bare string, or a
  string confidence raised out of the tool. All fields are now shape-checked, and the
  endpoint and timeout are configurable.
- `score_aesthetics`/`critique_composition` encoded the same image twice when
  `style_context=true`, and re-encoded all 16 style prompts on every call. Both heads now
  share one encode (scores verified bit-identical) and the prompt embeddings are cached.
- Technical IQA and the reasoner failed silently behind `except Exception: pass`; failures
  are now logged and reported as explicit error fields.
- `_aesthetic_comparison` shared one mutable reference dict across every page.
- Florence-2 ran float32 on CUDA, unlike the four other wrappers -- 2x memory for no gain.
- Outbound image/PDF fetches had no timeout.

### Performance

- `geometry.count_lobes` ran 76 connected-component passes over the whole frame; they are
  now restricted to the mask's bounding box, which is exactly equivalent since the field
  is zeroed outside it.
- `layout.py`'s three hand-rolled per-row/per-column Python scanners are vectorised. All
  fixtures split identically, negative controls included.
- EasyOCR loaded four language models by default; now one, with `--ocr-languages` to widen.

### Measured and rejected

- **Batching the six anatomy detector passes into one multi-phrase prompt.** ~5x faster
  (3.2s vs 16.4s) but not the same detections: Grounding DINO returns boxes whose text
  span maps to no part, reported with an empty label (22 of 31 boxes on one COCO frame),
  and competing phrases suppress weak parts entirely (another frame went from 35
  detections across six parts to 4, all `person`, losing every limb). Per-part tallies
  cannot be built from that, and the containment/dedup constants were swept against the
  per-part behaviour.
- **MUSIQ preprocessing was suspected of mismatching the exported graph.** It does not:
  the ONNX signature declares a fixed `['batch', 3, 224, 224]` input, and the score falls
  monotonically under increasing blur (41.1 -> 39.4 -> 21.8 -> 20.6). Recorded so it is
  not re-investigated. The loader now prefers an available accelerator and fails with an
  actionable message when the weights cannot be fetched.
- **`ocr` column-offset drift.** Reported as a defect; it is not. `split_columns` crops at
  gutter midpoints over `[0, *splits, width]`, so the crops tile the full width and the
  running offset is correct.

### Housekeeping

- Consolidated two copies of `_iou` into `geometry.box_iou`.
- `corroborated` documented: it is carried by control flow, not hardcoded. The comment
  claiming Sprint 19 was still pending was stale.
- `forensics` documents that a negative result is no evidence either way.
- Granite-Docling prose removed from the package description, manifest and `ocr_fusion`.
- Version reconciled to 0.8.2 across `pyproject.toml`, `manifest.json` and `server.json`;
  the v0.8.2 release commit had landed without it.
- Dockerfile and `run_battery.py` pre-cached and benchmarked `Florence-2-base` while the
  shipped default is `-large`; both now use the default, centralised as
  `DEFAULT_FLORENCE2_MODEL`.

## v0.8.1

This is a correctness release replacing fabricated anomalies with real defect measurements.

### Strict Anatomical Association & Deduplication (Sprints 17-18)
The initial `structured_analysis` implementation counted anatomy globally across the frame and tallied overlapping detector boxes as separate parts, creating false positives on real, unaltered photographs.
- **Real-photograph baseline (Sprint 17):** Added a measured negative control against 35 person-bearing COCO photographs.
- **Bounding Box Deduplication:** Overlapping box detections for the same part (e.g., Grounding DINO detecting the same arm multiple times) are now merged via an IoU-based envelope dedup.
- **Strict Spatial Association:** Anatomy parts are now strictly associated with the smallest containing person box via `_containment()` bounding-box math, rather than pooled globally. 
- These changes dropped the false anomaly rate on the COCO negative controls from 7/24 down to 1/24 (a single documented, accepted residual involving a sheep). 

### Computed Corroboration and Payload Cleanup (Sprint 19)
- **Breaking Change**: The unmeasured `severity_estimate` field was removed from the `Anomaly` object returned by `structured_analysis` in the `query_image` tool.
- The `corroborated` flag on `Observation` is now computed. It evaluates to `True` only when a part count survives deduplication, is associated with a specific person, and violates that person's anatomical ratio. Bare global tallies (e.g., detection coverage) now report `corroborated: false`.

### Garbled-text Artifacts: Documented Negative Result (Sprint 20)
We evaluated detecting garbled text artifacts using a dual-OCR (Florence-2 + Granite-Docling) lexicon check. We found that real, unaltered stylized signage and foreign words consistently trigger the same "disagreement + dictionary failure" heuristic as synthetic gibberish. The negative controls could not be held; to protect against false positives, no heuristic was shipped.

## v0.8.0

This release graduates the "Agentic Implementation" components to production. It implements deep structural improvements across visual reasoning, technical image quality assessment, and structured output parsing.

### Aesthetic Refactor (Sprint 13)

The aesthetic tools (`score_aesthetics` and `critique_composition`) have been refactored to cleanly split the concept of aesthetics (Spec 21).

- The legacy LAION aesthetic head's score is now mirrored explicitly as `photographic_aesthetic`.
- Using domain routing context (Sprint 4/5), non-photographic media (paintings, digital art) now surface `photographic_aesthetic_applicable: false` to prevent misinterpretation of the raw photographic bias model scores.
- Wired in the `image_quality.py` (Sprint 12) IQA model to supply a direct `technical_quality` float.
- Wired in the `reasoner.py` (Sprint 11) backend to dynamically produce an LLM-derived `artistic_judgment` payload synthesizing these numeric measurements (e.g., using `--reasoner-provider ollama`).
- Backward compatibility for the raw `score` field is strictly maintained.

### Technical image quality assessment (Sprint 12)

`image_quality.py` introduces a fast, lightweight technical IQA pipeline using a Multi-scale Image Quality Transformer (MUSIQ) running via ONNX.

- **Tradeoff decision**: Evaluated ONNX models for MUSIQ and CLIP-IQA on memory, latency, and accuracy. Both solved the task in ~15ms with low CPU overhead (126-150MB), but MUSIQ was selected as it natively produces a highly-standardized 0-100 technical quality score and uses ~20% less peak memory.
- **Strict output matching (Spec 22)**: Returns only the `technical_quality` score and model source, explicitly avoiding inventing unrelated submetrics that the underlying MUSIQ model does not calculate.
- **Optional Dependencies**: Since ONNX execution introduces external binaries, it is completely siloed behind a new `[iqa]` optional dependency group (`onnxruntime`, `huggingface-hub`).

### Optional reasoning backend (Sprint 11)

`reasoner.py` adds a `VisionReasoner` to support optional external interpretation of structured visual evidence (Spec 19–20).

- **Architectural Scope**: Supports only `none` (default) and `ollama` provider modes, as Ollama is natively available on the host system.
- **Strict Invariant**: Enforces the contract that direct measurements (e.g., from Grounding DINO or SAM2) must *never* be silently overridden by the LLM. If the LLM generates a payload attempting to fabricate its own measurements, the reasoner automatically discards them and re-attaches the original raw measurements.
- **Error Handling**: Graceful fallback to `none`-like behavior if the LLM output fails to parse as strict JSON.

### Structured visual inspection for generative AI artifacts (Sprint 10)

`query_image` now supports a `structured_analysis` option (Spec 17) to perform rigorous visual inspection using physical measurements to override unreliable VLM judgment.

#### Observation and Anomaly structures (Spec 17)

- **`inspection.py` (`Observation`, `Anomaly`)**: Implements `analyze_inspection` to evaluate images for structural anomalies (generative AI artifacts like extra limbs).
- **Proactive anatomical measurement**: Since Moondream2 is documented to be completely blind to generated human artifacts, `structured_analysis` automatically runs Grounding DINO to count standard anatomy (`person`, `head`, `arm`, `leg`, `hand`, `finger`).
- **Defect Corroboration Strategy**: Overrides the VLM's blindness by evaluating anatomical proportions against physiological bounds (e.g. `arm > persons * 2 + 1`). When the rule is violated, an `Anomaly` is raised and corroborated by the physical measurement (`Observation`), protecting against uncorroborated VLM hallucinations.
- **Fixture Verification**: The first version of this structural check measured nothing real, and that was caught in the v0.8.1 sweep. The original Sprint 10 code claimed to successfully flag "10 arms generated for 2 people" on the HADM demo image, but it was just detector noise clearing a flawed global threshold on an image whose real defect (garbled text) the pipeline couldn't yet see. Sprints 17-19 established a real-photograph ground-truth baseline (35 COCO scenes), replaced the global frame-tally with strict part-to-person spatial association (`_containment`) and deduplication (`_merge_duplicate_boxes`), and proved that the 10-arm measurement drops to exactly 0 anatomical anomalies on that fixture when measured correctly. Sprint 20 evaluated whether the real defect (garbled text) could be robustly flagged using a dual-OCR lexicon check, but demonstrated it creates unavoidable false positives on real stylized signage; no garbled text heuristic was shipped.

### Semantic ambiguity reporting in `count_objects` (Sprint 9)

`count_objects` now evaluates and reports diagnostic semantic ambiguity (Spec 15) when independent detectors collapse overlapping instances into a single connected region:

#### Semantic ambiguity and count semantics (Spec 15)

- **`ambiguity.py` (`AmbiguityResult`, `check_semantic_ambiguity`)**: Analyzes detection counts, silhouette geometry (e.g. `lobes`, `by_radial`), and multi-head consensus to determine whether a detection represents a discrete tally or a collapsed composite structure.
- **Reporting fields**:
  - `semantic_ambiguity`: Boolean flag (`True` on collapsed structures like flower petals or overlapping tool blades).
  - `count_semantics`: Clarifies tally meaning (`"minimum_visible_instances"`, `"measured_tally"`, or `"unverified_tally"`).
  - `ambiguity`: Structured dictionary (`{"ambiguous": true, "reason": "...", "evidence": [...]}`) when ambiguity is flagged.
- **Strict "measure, don't judge" adherence**: `count` is never overwritten or inflated to a fabricated higher number; when the detector collapses on overlapping instances, `count` remains the measured tally (e.g. `1`), the radial lobe count is preserved as an actionable estimate in `estimates.outline`, and `count_semantics: "minimum_visible_instances"` alerts downstream consumers to the ambiguity.
- **Regression test**: Live verification in `test_server.py` on the flower/petal fixture (`tests/sample.jpg`) asserts `count: 1`, `separable: "no"`, `semantic_ambiguity: true`, and `count_semantics: "minimum_visible_instances"`.

### OCR fusion and automatic text verification in `caption` (Sprint 8)

`caption` now supports an `auto_verify_text` option (default `False`) that automatically detects when a caption quotes or describes embedded text, signs, logos, or proper nouns, and cross-checks it against both Florence-2 OCR and Granite-Docling specialist OCR.

#### Fused consensus and small-text upscale crops (Spec 13 & 14)

- **Text detection gating**: `contains_likely_text()` in `textmatch.py` scans generated captions for signage words (`reading`, `says`, `banner`, `logo`, etc.), quotes (`"..."`), CamelCase tokens, and acronyms. Ordinary photos with no embedded text bypass OCR entirely, preserving speed and avoiding unnecessary model loads.
- **Small-text crop & upscale**: Bounding boxes detected by Florence-2's region head are cropped from the full image and upscaled using bicubic interpolation (3x for boxes <35px tall, 2x for boxes <100px tall) before being transcribed by Granite-Docling.
- **Consensus reporting**: When `auto_verify_text=True`, the response returns:
  - `caption`: The original Florence caption prose (never overwritten).
  - `caption_text_warning`: `True` if any quoted token is disputed by OCR.
  - `text_consensus`: `{caption, florence_ocr, specialist_ocr, agreeing_sources}` recording source agreement.
  - `caption_corrected`: Clean copy of the caption with consensus text substituted.
  - `text_regions`: Verbatim bounding boxes and labels.
  - `corrections`: Full audit list of all substituted tokens.

### Standalone `granite_docling.py` specialist OCR (Sprint 7)

Integrated `ibm-granite/granite-docling-258M` as a standalone OCR specialist:
- Model architecture: `AutoProcessor` + `AutoModelForImageTextToText` (`idefics3` architecture).
- Lazy loading: Wrapped with `IdleReleased`/`IdleProxy` so the ~515MB model only loads when needed and frees memory after inactivity.
- Added `[project.optional-dependencies]` group `ocr-specialist` in `pyproject.toml`.

### Automatic clip-art routing in `count_objects` (Sprint 5)

`count_objects` now automatically routes each request to the correct detection back-end
without the caller needing to set `clip_art=True`.  When `clip_art` is omitted (the default),
a new `routing.py` module inspects the image's pixel statistics and selects Florence-2 for
flat synthetic/vector art or Grounding DINO for real photographs.  Explicit `clip_art=True`
remains fully authoritative and bypasses routing entirely; the routing path only fires on the
existing default.

#### Routing architecture: three pixel-statistics in cascade, no neural network

Sprint 4 showed SigLIP2 alone cannot be trusted for this split: the only real photograph
tested was confidently classified as `clip_art` (0.60 softmax, margin 0.36 — a correct
margin threshold would have required it to be lower than the actual top-1 confidence, which
is impossible).  Sprint 15 established a model-free alternative: unique RGB colors per 1,000
pixels cleanly separates real photographs (65–686, mean 260, across 35 COCO val2017 images)
from flat synthetic art (under 3 in every fixture tested).

`choose_count_backend` in `routing.py` uses a three-step cascade:

1. **Color density** (unique RGB colors / 1k px).  At or above 10.0 → photograph →
   Grounding DINO.  Sprint 15's empirical gap: lowest photo score 65.0, highest synthetic
   score < 3.0; threshold at 10 keeps a ~6× margin on each side.

2. **Mean HSV saturation**.  When color density is below the photo threshold, the image
   could be flat synthetic art OR a genuine greyscale photograph (Sprint 15 found one:
   `coco_bird_3`, a real photo scoring 0.94 colors/1k px because the camera image was
   greyscale).  Saturated solid fills score high (clip-art scene: 87.2/255); a truly
   greyscale image scores ~0.  At or above the greyscale gate threshold (15.0/255) →
   saturated fills → Florence-2.  Below → potentially greyscale → continue.

3. **Luminance standard deviation** (V channel in HSV).  Applied only to low-color-count,
   low-saturation images.  A real greyscale photograph has high tonal variance from camera
   noise and lighting; a flat near-uniform grey synthetic image has near-zero variance.
   At or above 20.0/255 → greyscale photograph → Grounding DINO.  Below → Florence-2.

SigLIP2 is not consulted for the routing decision.  Its domain label is available as
descriptive metadata (see `_routing` block below) but does not influence the back-end choice.
This is the same "measure, don't trust a single model's judgment" posture `count_objects` and
`spatial_relations` use elsewhere.

#### `_routing` diagnostic block in `count_objects` results

Every `count_objects` result where routing fired (i.e., where `clip_art` was not explicitly
`True`) now includes a `_routing` block:

```json
{
  "_routing": {
    "backend": "florence2",
    "unique_colors_per_1k_px": 0.05,
    "mean_saturation": 87.2,
    "luminance_stddev": 0.0,
    "reason": "color density 0.05 < threshold 10.0 and mean saturation 87.2 >= threshold 15.0 -> saturated flat fills -> synthetic / clip-art"
  }
}
```

The prefix `_` signals the block is informational; the primary payload (`count`, `bboxes`,
etc.) is unchanged.  `mean_saturation` and `luminance_stddev` are 0.0 when the cascade
short-circuits before computing them.

#### Regression test fixture: `tests/clipart_scene.png`

New synthetic flat-art fixture committed: a 512×512 scene with 2 trees (triangle canopy +
rectangle trunk) and 3 houses (rectangle walls + triangle roof + rectangle door), drawn in
13 distinct solid fills.  Measured: 0.050 unique colors/1k px, mean saturation 87.2/255.
This is the "tree/house clip-art fixture" referenced in the Sprint Plan (Sprint 5 line 348)
and CHANGELOG Sprint 4 ("6 instead of 2 trees on a clip-art scene, plus a spurious 4th
house past the true 3").

Sprint 5 regression test (live integration, `tests/test_server.py`):
- `count_objects(tree)` without `clip_art` → `_routing.backend == "florence2"`, count == 2
- `count_objects(house)` without `clip_art` → `_routing.backend == "florence2"`, count == 3
- `count_objects(tree, clip_art=True)` → no `_routing` key, count == 2 (routing bypassed)
- `count_objects(petal)` on `sample.jpg` → `_routing.backend == "grounding_dino"`,
  `unique_colors_per_1k_px >= 65`, `scores` list present (Grounding DINO provides scores)

#### New module: `routing.py`

`src/fusion_vision_mcp/routing.py` exposes:
- `count_unique_colors_per_1k_px(image)` — unique (R,G,B) tuple count, per 1k pixels
- `mean_saturation(image)` — mean HSV S-channel, 0–255 scale
- `luminance_stddev(image)` — std dev of HSV V-channel, 0–255 scale
- `RoutingDecision` — frozen dataclass: `backend`, `unique_colors_per_1k_px`,
  `mean_saturation`, `luminance_stddev`, `reason`
- `choose_count_backend(image)` — the public routing API

Thresholds (`COLOR_DENSITY_PHOTO_THRESHOLD`, `MEAN_SATURATION_GRAYSCALE_THRESHOLD`,
`LUMINANCE_STDDEV_PHOTO_THRESHOLD`) are module-level constants imported and exercised by
20 pure unit tests in `tests/test_routing.py`.

### `detect_objects` no longer returns a false full-frame match on a maximally vague noun


v0.7.1 closed this for `count_objects(clip_art=true)` (`Florence2.detect_objects`'s
`exclude_full_frame` option) but left the `detect_objects` tool itself unfixed — the exact tool
the original "Second Vision Pass" finding (F7) named. `detect_objects("object")` on a blank
canvas still returned a box spanning nearly the whole frame (`[0, 0, 599, 399]` on the fixture),
confirmed live against the *unmodified* v0.7.1 server as part of proving this increment doesn't
touch it.

New `query_policy.py` (`is_generic_query`) recognizes only an exact, maximally vague noun
("object", "thing", "item", "stuff", "something", "shape", "entity", and their plurals) — not
any phrase containing one of those words. `detect_objects` now passes `exclude_full_frame=True`
only when the query matches that exact vocabulary; a specific noun that genuinely fills the frame
(the "wood" case from v0.7.1's regression testing) is untouched, since it was never at risk in
the first place — the guard only ever applies to the vocabulary itself, not by area or content.

Verified: `detect_objects("object")` on `tests/detect_blank_canvas.png` now returns `bboxes: []`;
`detect_objects("the small object on the left")` on the same image is unfiltered, confirming the
guard is exact-match rather than substring-triggered.

### The same guard now covers `count_objects` and `spatial_relations`, not just `detect_objects`

The Sprint 0 fix above closed the gap for `detect_objects` alone; spec doc §8 explicitly names
`count_objects` and `spatial_relations`'s per-object detection as the two other places the guard
belongs. Measured live against the *unmodified* path first: `GroundingDino.detect_objects("object")`
on `tests/detect_blank_canvas.png` returns one box spanning nearly the whole frame — `count: 1`,
box `[1.72, 0.26, 598.70, 398.88]`, score `0.46` — the same failure mode as Florence-2's grounding
head, just from the other detector `count_objects`'s plain (non-`clip_art`) path and
`spatial_relations` both call.

New `query_policy.suppress_generic_full_frame` (built on the existing `is_generic_query` plus a new
`is_full_frame_box`, mirroring `Florence2.detect_objects`'s own `>=98%`-of-frame cutoff) filters a
Grounding-DINO-shaped result the same way: only when `object_name` is the exact generic vocabulary,
and only the near-full-frame entries within it — a real small detection returned alongside a
spurious full-frame one for the same generic query keeps the real detection. Wired into
`count_objects`'s plain path (right after `app.counter.detect_objects`, before the silhouette/
consensus logic runs) and into `spatial_relations`'s per-object `app.counter.detect_objects` call
(before the best-scoring-box selection, so a filtered-to-nothing result is skipped exactly like a
name that was never found).

Verified: `count_objects("object")` on the blank-canvas fixture now returns `bboxes: [], count: 0`
instead of counting the false full-frame box as one instance; `count_objects("the small object on
the left")` on the same image is unfiltered. `spatial_relations(objects=["object"])` on the same
fixture now returns `objects: [], relations: []` with the existing "none of the requested objects
were found" note, instead of reporting the full-frame box as a located match. All three tools' own
specific-noun regression tests (the "wood" case) continue to pass unfiltered, confirming the guard
is still driven by the query's exact wording, not by how much of the frame a box covers.

### `adaptive_threshold.py` — adaptive detection threshold selection (pure logic, Sprint 2)

The fixed `threshold=0.15` default was chosen by sweeping the counting benchmark fixtures: it is
the *lowest* box-confidence floor at which every negative control still holds. On clean scenes
it works well; on cluttered scenes with a strong target (~0.9) and many distractors (~0.2–0.3),
the fixed floor lets distractors through (spec doc F9).

New `src/fusion_vision_mcp/adaptive_threshold.py` implements `choose_threshold(scores)`:
- Sorts scores descending, computes consecutive gaps.
- If a gap ≥ 0.20 is found (a "confidence cliff") AND the top score ≥ 0.50 AND the gap
  dominates the distribution (not a shallow gradient), places a candidate threshold midway
  between the two scores spanning the cliff.
- Clamps conservatively: never below `base_threshold` (0.15), never above `top_score - 0.01`.
- Returns `ThresholdResult(threshold, used, confidence, reason)` — caller can inspect
  `used` to know if adaptation fired, and `reason` for diagnostics.
- Pure Python, no model dependencies, unit-testable against synthetic score distributions.

13 pure-logic unit tests cover: empty scores, single detection, weak top score suppression,
clean cliff (star-among-18-distractors case: target 0.90, distractors 0.19–0.32 → adapts),
shallow gradient rejection, clamping, two-score cliff, three-score middle cliff, confidence
scaling with cliff strength, custom parameter overrides.

Wiring into `count_objects`/`grounding_dino.py` is Sprint 3 — this sprint delivers only the
policy logic, no behavior change yet.

### Adaptive thresholding wired into `count_objects` and `grounding_dino.py` (Sprint 3)

`GroundingDino.detect_objects` gained an `adaptive_threshold` parameter (default `True`):
- Runs post-processing once with threshold=0.0 to collect all raw scores.
- Feeds raw scores to `choose_threshold` (from Sprint 2's `adaptive_threshold.py`) which
  selects a conservative floor based on the score distribution's "confidence cliff".
- Returns an `adaptive_threshold` diagnostic block: `{threshold, used, confidence, reason,
  base_threshold, raw_detection_count}` — so callers can see what happened.
- `count_objects` tool exposes `adaptive_threshold` (default `True`) passing it through.

Verified against the existing counting benchmark fixtures:
- All 9 positive fixtures (ring/row/grid/overlap) and 8 negative controls (single objects
  with distractors/textures) still pass — no regressions from the fixed 0.15 floor.
- The star-among-18-distractors fixture (target ~0.90 vs distractors 0.19–0.32) now
  automatically adapts: `adaptive_threshold.used=true`, threshold raised from 0.15 to
  ~0.55, suppressing the distractors. The fixed threshold would have required manual
  tuning; now it happens automatically when the score distribution warrants it.
- On clean scenes without a clear cliff, `used=false` and threshold stays at 0.15.
- `adaptive_threshold=false` preserves the original fixed-threshold behavior exactly.

**Independently re-verified**, with three additions the original build didn't cover:

- The `threshold=0.0`-then-manually-filter approach was checked against a direct call at
  the target threshold on two real images -- bit-identical boxes and scores, confirming
  `adaptive_threshold=false` really does reproduce the original behavior rather than just
  being untested. `post_process_grounded_object_detection` returns 900 raw scores per call
  regardless of image content (Grounding DINO's fixed query-slot count) -- `raw_detection_count`
  in the diagnostic block is that number, not a measure of scene clutter; worth knowing before
  reading anything into it.
- The star-among-distractors claim was re-run against the actual fixture, not a
  reconstruction: fixed threshold gives 6 (the documented F9 bug), adaptive gives 1
  (correct) at threshold≈0.61. Confirmed no bad interaction with the v0.7.1 occlusion/
  envelope fix either -- if anything the occlusion case improves incidentally (2 boxes
  including a loose duplicate → 1 clean box), since there aren't enough surviving boxes
  left for the group-envelope logic to engage.
- **Gap found and closed**: zero automated tests existed for any of this — the entire
  verification was manual scripts, despite the change becoming the *default* for every
  tool that calls `GroundingDino.detect_objects` (`count_objects`, `spatial_relations`,
  and the VQA cross-check helpers all inherit it silently; only `count_objects` exposes an
  explicit opt-out). Added `tests/count_cluttered_target.png` (the real F9 fixture) plus
  two `test_server.py` regressions: the adaptive default correctly isolates the star, and
  `adaptive_threshold=false` genuinely reproduces the old count on the same image.
- **Residual risk noted here, characterized and closed in Sprint 3b below**: the algorithm
  can't distinguish "a big score gap because the second thing is a false positive" from "a
  big score gap because a second *real* instance is just less confidently detected." Two ad
  hoc reproduction attempts at the time didn't trigger it — see Sprint 3b for why (the wrong
  axis was searched) and for the actual measured boundary.

### Sprint 3b — Adaptive threshold residual risk: characterized, not a general risk

Closes out Sprint 3's open item above with an empirical sweep rather than leaving it as a
standing "nothing rules it out" caveat. Full method, including a self-correction of an earlier
flawed proxy check, is in `ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md`.

- **Finding**: across 51 valid two-instance synthetic scenes spanning five perturbation axes
  (occlusion to 80%, Gaussian blur to 10px on a 70px object, size ratio down to 1:7,
  desaturation to 10%, and combinations of those), the danger zone — where `choose_threshold`
  drops a real second instance — triggered exactly once (2.0% prevalence), and only on the
  edge-crop axis: an instance cropped to ~14% visible at the frame edge. Bisected the boundary
  directly: `offset=115` (18% visible) stays safe, `offset=120` (14% visible) triggers
  (`threshold=0.526 > s1=0.358`). Occlusion, blur, size, and contrast — the scenarios that
  actually motivated the original concern (partially occluded/blurry/smaller, not absent) —
  produced zero hits even at their most aggressive settings tested.
- **A second, separate bug was caught along the way**: the sweep's first draft used a
  hand-written 3-condition proxy (`gap>=0.20 AND ratio>2 AND top>=0.50`) instead of calling
  `choose_threshold` directly. Re-checking borderline rows against the real function found the
  proxy both over-counted (flagged `offset=140` as a hit — that disc is drawn entirely outside
  the 512px canvas, 0% visible, not a real instance to begin with) and under-counted (missed
  `offset=120`, ratio 1.94 just under the proxy's hard `>2` cutoff, which the real
  tail-density-dependent test does trip). The shipped sweep script calls `choose_threshold`
  directly and reports `visible_fraction` per row so a degenerate (0%-visible) case can't be
  silently folded into the prevalence count again.
- **F9, the case this feature exists for, verified unaffected**: three target discs
  (0.78–0.80) vs. four square distractors (0.03–0.10) — a 0.68 gap — adapts to
  `threshold=0.439`, keeping every target and dropping every distractor.
- **Benchmark suite re-run, unaffected**: 9/10 positives exact, 8/8 negatives held — same as
  Sprint 3's own numbers; nothing in this sprint touches `grounding_dino.py` or
  `adaptive_threshold.py` themselves, only the sweep tooling that measures them.
- **Conclusion**: no mitigation implemented. A real second instance has to be cropped to
  roughly 15–18% visibility or less before this reproduces — a scenario closer to "the object
  is barely in the frame at all" than "partially occluded or less confident." Same posture as
  the flower/petal limitation elsewhere in this project: measured, named, and accepted rather
  than hidden or chased with a fix that risks breaking the F9 case above.
- Evidence: `benchmarks/adaptive_threshold_sweep.py`,
  `benchmarks/results/adaptive_threshold_danger_zone_sweep.csv` (51 valid rows + 1 degenerate,
  reproducible by re-running the script).

### Sprint 4 — Domain router: SigLIP2 zero-shot classification (honest numbers, not the first draft's)

Built `domain_router.py`: lazy-loaded `google/siglip2-base-patch16-224` (native
`SiglipModel`/`SiglipProcessor`, no `trust_remote_code` needed — checked directly, not
assumed) via the existing `IdleReleased`/`device.py` conventions. Returns a `DomainResult`
with the top label, a softmax confidence/margin (not sigmoid — see the docstring for why:
raw sigmoid rounds to ~0.000 for every label here, useless for a margin check, though the
top-1 ranking is identical either way since sigmoid is a monotonic transform of the same
logits), and an `ambiguous` flag gated by a tunable margin threshold.

**The first version of this sweep measured nothing real, and that was caught before
shipping.** It tested SigLIP2 against all 23 counting-benchmark fixtures from
`fixtures.py`, every one of which is a flat, synthetic PIL-drawn shape except one real
photograph (`flower()`, the actual `tests/sample.jpg`) — and that one photo's "expected"
ground truth had been hardcoded to `clip_art` in the sweep script, not because that's true,
but because that's what SigLIP2 already predicted for it. "20/23 correct" measured
agreement with the model's own output, not accuracy against reality, and covered zero
painting/document/screenshot cases despite the sprint explicitly requiring new fixtures for
them.

**Rebuilt with `benchmarks/domain_fixtures.py`**: honest ground truth (the real photo's
truth is `photograph`, full stop), plus new minimal fixtures for the three domains that had
zero coverage. Result — 3/6 correct, and the 3 misses are documented, not hidden:

- **`photo_flower` → predicted `clip_art` (0.60 confidence), wrong.** The one real photo
  available is confidently misrouted. This is the single most operationally important
  finding here, since photograph-vs-clip_art is exactly the distinction Sprint 5's routing
  will depend on — this fixture (a flat-lit paper flower on white) may be an atypical,
  graphic-looking photo rather than representative of photos generally, and only one real
  photo was available to test; that's a real limitation of this validation pass, stated
  plainly rather than glossed over.
- **`painting_stylized` → predicted `clip_art` (0.37, effectively tied with
  `a photorealistic image` at 0.36), wrong.** Six materially different synthetic
  constructions were tried (posterize, saturation+blur+grain, colour quantization+blur,
  mode-filter smear, heavy blur alone, a wholly synthetic landscape, hundreds of angled
  brush-dab strokes) — not one ever placed `oil painting`/`watercolor painting` even in the
  top 3. A convergent negative result, same posture as this project's flower/petal counting
  finding: recorded because it held up across genuinely different attempts, not because one
  fixture happened to fail.
- **`screenshot_ui` → predicted `document or scanned page` (0.56), wrong.** Two mockups
  tried (native-app settings panel, then a browser-chrome mockup with tab/address bar and
  page content, expected to be a much stronger signal) — both read as a document, never as
  `computer screenshot`.
- **`clipart_ring8`, `clipart_distractors`, `document_page` all correct**, the latter
  confidently (0.86).

**Margin threshold: `0.15` kept, but not because it improves accuracy — it can't.** The
margin only gates the `ambiguous` flag; it has zero effect on which label wins (softmax is a
monotonic transform of the logits at every threshold). Swept `{0.05..0.50}` against the
fixed classifications: `0.15`–`0.20` is the only band that flags both genuinely-uncertain
cases (`clipart_ring8`, correct but a 0.125 margin; `painting_stylized`, wrong with a 0.013
margin) without also false-flagging the confidently-correct cases (`clipart_distractors`, at
a 0.212 margin, becomes a false positive above `0.20`). **No threshold value ever
flags the two confidently-wrong cases** (`photo_flower` margin 0.359, `screenshot_ui` margin
0.327) — a margin-based signal structurally cannot catch a confident misclassification, only
an uncertain one. Sprint 5's routing design needs to account for this: `ambiguous=False`
does not mean "trust this."

**Also fixed during verification**: nothing had been committed (third sprint in a row with
this pattern); a stray copy of `domain_router.py` had been placed directly in main's `src/`
(the live install's actual source tree — confirmed inert, nothing imports it, but a step
past prior sprints' stray-file mistakes) and a `CHANGELOG.md` edit again landed in main
instead of here, both reverted; `ruff check` had 9 errors and `ruff format` wanted 3 files
reformatted; `DomainResult`'s docstring claimed sigmoid while the code computed softmax;
`classify_domain`'s docstring was Google-style, flagged once already in Sprint 2;
`trust_remote_code=True` was carried over from the Moondream2 pattern without checking —
removed, confirmed unnecessary. `benchmarks/test_domain_router.py` (a print-only script with
no assertions, despite the `test_` name) was deleted; real pure-logic unit tests added at
`tests/test_domain_router.py` instead.

**Regression check**: counting benchmark suite unaffected (9/10 positives, 8/8 negatives —
this sprint touches no shared code). 208/208 tests pass, ruff check + format clean, mypy
clean.

- Evidence: `src/fusion_vision_mcp/domain_router.py`, `benchmarks/domain_fixtures.py`,
  `benchmarks/tune_domain_threshold.py`,
  `benchmarks/results/domain_router_threshold_sweep.csv`, `tests/test_domain_router.py`.

### Sprint 15 — Real-photograph ground truth from COCO val2017 & unique-color measurement

Prior to this sprint, the repository relied almost entirely on synthetic geometric fixtures,
with only a single real photograph (`tests/sample.jpg`, a flower). Sprint 4 showed that SigLIP2
zero-shot classification confidently misclassified that lone photograph as `clip_art` (0.60
confidence, no ambiguity flag), exposing the lack of photographic ground truth as a critical
blind spot.

- **Curated COCO val2017 subset**: pulled 35 real photographic scenes from Microsoft COCO's
  official `val2017` validation split into `benchmarks/coco_images/` (7.58 MB total).
- **Strict license audit**: each candidate image was mechanically cross-referenced against COCO's
  official `instances_val2017.json` `licenses` metadata. All 35 selected images carry License ID 4
  (`Attribution License`, CC-BY 2.0), permitting open-source redistribution and derivative works
  (cropping/resizing). All No-Derivatives licenses (License 3 `CC BY-NC-ND` and License 6
  `CC BY-ND`) and non-commercial/ambiguous licenses were strictly excluded.
- **Spread of density and categories**: fixtures span 29 distinct object categories across three
  pre-specified instance-density tiers:
  - Low density (10 fixtures, 1–2 instances): clean single- and dual-object grounding baselines.
  - Medium density (13 fixtures, 3–5 instances): standard multi-object scenes.
  - High density / cluttered (12 fixtures, 6–22 instances): stress tests for distractor suppression
    and counting.
- **Fixture module**: `benchmarks/coco_fixtures.py` provides `CocoFixture`, `coco_fixtures()`,
  and `coco_counting_fixtures()` (converting directly to `benchmarks.fixtures.Fixture` for use with
  `benchmarks/harness.py`). Normalized bounding boxes and ground-truth counts are preserved in
  `benchmarks/coco_annotations.json`.
- **Model-free color statistic measurement (`measure_coco_colors.py`)**:
  Sprint 5 reframed automatic clip-art routing around whether a cheap, model-free statistic --
  unique RGB colors per 1,000 pixels -- could distinguish photographs from flat-art/synthetic
  illustrations. Computed the statistic across all 35 COCO photographs into
  `benchmarks/results/coco_unique_colors.csv`:
  - 34 of 35 photographs scored between **65.02 and 686.33** (mean: 268.0, median: 254.8), separated
    from synthetic/flat-art fixtures (< 3.0) by roughly two orders of magnitude.
  - **Empirical confirmation of the Sprint Plan's predicted edge case**: one fixture
    (`coco_bird_3_000000456496`) scored **0.94**, landing right in "synthetic" territory.
    Inspection confirmed it is an authentic **grayscale** photograph ($R=G=B$ across all pixels).
    Because 8-bit grayscale is bounded at 256 unique RGB values, 256 colors across 272,640 pixels
    strictly caps the metric at 0.94. This directly proves the warning in the Sprint Plan
    (lines 364–367): grayscale photographs cannot be distinguished from synthetic illustrations by
    unique color count alone, and any auto-router must account for channel saturation/variance
    before leaning on color count.
- **Tests**: `tests/test_coco_fixtures.py` adds 7 unit tests covering fixture loading, disk
  presence, strict license ID invariance (`{4, 7, 8}`), tier distribution, bounding-box coordinate
  validity, `Fixture` conversion, and color calculation.
- Evidence: `benchmarks/coco_fixtures.py`, `benchmarks/coco_annotations.json`,
  `benchmarks/coco_images/`, `benchmarks/measure_coco_colors.py`,
  `benchmarks/results/coco_unique_colors.csv`, `tests/test_coco_fixtures.py`.

### Sprint 16 — Real spatial-relationship ground truth from Open Images & agreement evaluation

Prior to this sprint, `spatial_relations` and the underlying geometric measurements in
`geometry.py` were validated against exactly two synthetic fixtures
(`tests/spatial_touch_separate.png` and `tests/spatial_containment.png`).

- **Curated Open Images VRD subset**: pulled 36 real photographic scenes from the official
  Open Images V6 Visual Relationship Detection validation split into `benchmarks/open_images/`
  (8.51 MB total).
- **Strict license audit**: cross-referenced against Open Images official image metadata
  (`validation-images-with-rotation.csv`). 100% of the 36 selected images carry verified
  Creative Commons Attribution 2.0 (`https://creativecommons.org/licenses/by/2.0/`), permitting
  open-source redistribution.
- **Coverage of core `geometry.py` relationship analogues**:
  - Containment (12 fixtures): official relationships `contain` and `inside_of` (e.g. coffee cup
    containing coffee, person inside car, flowerpot containing plant).
  - Contact / Touching (12 fixtures): official relationships `on` and `holds` (e.g. musician
    holding instrument, athlete holding racket, person on bicycle/surfboard).
  - Separation (12 fixtures): official relationship `at` with disjoint bounding boxes and positive
    spatial clearance.
- **Fixture module**: `benchmarks/open_images_relation_fixtures.py` provides `SpatialRelationFixture`
  and `relation_fixtures()`. Ground-truth bounding boxes, relationship triples, and analogue states
  are preserved in `benchmarks/open_images_annotations.json`.
- **Live agreement evaluation (`evaluate_spatial_relations.py`)**:
  Executed the live multi-model pipeline (Grounding DINO detector + SAM2 segmenter +
  `geometry.relation`) across all 36 fixtures, recording metrics into
  `benchmarks/results/open_images_spatial_agreement.csv`:
  - **Detector success**: located both subject and object in **35/36 fixtures (97.2%)**.
  - **Overall relationship agreement**: **25/36 (69.4%)**.
    - Containment: **8/12 (66.7%)** agreement.
    - Contact: **9/12 (75.0%)** agreement.
    - Separation: **8/12 (66.7%)** agreement.
  - **Empirical observations recorded (without tuning)**:
    - Contact held cleanly across held instruments/sports equipment (`holds` → `overlapping`/`touching`).
    - Disagreements in containment were predominantly boundary segmentations (e.g. coffee liquid
      segmented abutting the mug inner rim with a 1.4px gap, yielding `touching` rather than
      `overlapping`).
    - Disagreements in separation were caused by 2D camera projection (e.g. a chair physically at a
      table whose 2D silhouette overlaps the table edge from the camera perspective).
- **Tests**: `tests/test_open_images_fixtures.py` adds 7 unit tests covering fixture loading,
  disk presence, CC-BY license invariance, tier distribution, bounding-box coordinate validity,
  and agreement logic.
- Evidence: `benchmarks/open_images_relation_fixtures.py`, `benchmarks/open_images_annotations.json`,
  `benchmarks/open_images/`, `benchmarks/evaluate_spatial_relations.py`,
  `benchmarks/results/open_images_spatial_agreement.csv`, `tests/test_open_images_fixtures.py`.

## v0.7.1 · 2026-08-28

Fixes for six of the seven gaps documented in the "Second Vision Pass" comparison
against Claude's native vision. Each was root-caused rather than patched at the
symptom, verified against the exact fixture that exposed it, and checked against
this project's existing benchmark/test fixtures for regressions before shipping.

### Occluded objects recovered at the *default* threshold — no manual tuning needed

A ~60%-occluded rectangle went from 0 detections to the correct one, at Grounding
DINO's existing default `threshold=0.15`. The actual bug was not threshold at all:
`grounding_dino.py`'s group-envelope filter (built to drop a box drawn around a
*group* of separate instances, e.g. all eight petals of a flower) was misreading the
correct high-confidence box as an envelope around several looser, lower-confidence
*duplicate* detections of that same partially-hidden object, and dropping the one
real detection. Fixed by requiring a candidate envelope's "members" to have low
mutual overlap with each other before it's dropped — real group members barely
overlap; duplicate re-detections of one object overlap heavily (measured at IoU
~0.8 here). Re-verified against every ring/row/grid case in `benchmarks/fixtures.py`:
identical envelope decisions on all nine, zero regressions.

### Small-text OCR scrambling — the real cause was column-detection, not legibility

The small-text fixture that motivated this fix was previously misdiagnosed (in the
"Second Vision Pass" writeup) as a Florence-2 glyph-legibility problem. It is not:
Florence-2 reads the original, un-upscaled text correctly on its own. The actual bug
is in `layout.py`'s automatic column-gutter detector, which misfired six spurious
column splits on a single short paragraph — with only two real text lines landing in
the analyzed page body, word-gaps coincidentally lined up closely enough to look like
column boundaries. `find_column_splits` now counts independent text lines first and
skips splitting when there are too few (2) for that alignment to mean anything,
while every existing multi-line column fixture (5+ lines) is unaffected. This fixes
the scrambling, not small-text transcription accuracy in general: re-verified through
the real MCP server on the same fixture, the output now comes back in the correct
order with nothing interleaved, but Florence-2's OCR head still drops two words,
duplicates one, and merges two across punctuation at this font size, against the
fixture's known-exact source string — an ordinary, separate small-text limit this
fix doesn't touch.

### Containment recovers from an occlusion hole, without swallowing a real one

`geometry.relation` now fills a hole in either mask before measuring containment —
but only when the hole is a minority (<30%) of the mask's own area. SAM2 commonly
segments a container object with a hole exactly where something in front of it
occludes it (a circle sitting on a square gets segmented as a square with a
circle-shaped bite missing), which previously measured as barely overlapping
instead of contained. Unlike a blind fill, a hole large enough to plausibly be real
structure — a ring, a washer, a picture frame — is left alone; both the target case
and a ring-shaped negative control are now regression tests in `test_geometry.py`.

### Clip-art counting: Florence-2 gets both measured cases exactly right

`count_objects(clip_art=true)` routes counting through Florence-2's grounding head
instead of Grounding DINO, whose training distribution is real photographs and can
over-detect on flat vector art (measured: 6 instead of 2 trees on a clip-art scene,
plus a spurious 4th house past the true 3). Florence-2 gets both exactly right on
the same scene. `count_objects(threshold=...)` also now exposes Grounding DINO's
box-confidence floor directly, for a visually cluttered scene with distractor
shapes near the target — a manual lever (raising it can just as easily drop real
instances on a normal scene), not an automatic fix.

### Blank-canvas false detection, without breaking a genuine full-frame match

Florence-2's grounding head confidently draws a box around the entire canvas when
nothing matches the query. `Florence2.detect_objects` gained an opt-in
`exclude_full_frame` parameter that drops any box covering >=98% of the frame —
opt-in specifically because the same shape is also the *correct* answer when the
object genuinely fills the frame (a close-up of wood, asked for "wood"); shipping
this as the unconditional default would have silently broken that case. Wired into
`count_objects(clip_art=true)`, where a false full-frame catch-all would otherwise
inflate a count.

### `query_image`'s low-confidence routing now covers "largest/smallest" questions

`question.py` gained a `SIZE` category (alongside the existing spatial/count/OCR
ones) for "which is the largest/smallest <name>" wording. When `check_consistency`
flags a low-confidence judgment answer in that shape, the cross-check now detects
every instance of the named object and resolves the extremum by bounding-box area,
returning its coordinates — a measurement, not a repeated guess. Verified on a
four-circle fixture with two size pairs: correctly resolved both the largest and
the smallest instance by measured area.

### Not fixed: occlusion thresholding as originally proposed did not work

An externally-authored first attempt at these fixes proposed lowering Grounding
DINO's box-confidence threshold from 0.15 to 0.05 for `spatial_relations`. Directly
tested against the occlusion fixture: both the old and new threshold returned zero
detections, identical. Lowering the threshold never reached the actual bug, because
the drop happens in the group-envelope filter *after* threshold filtering — see
above for the fix that actually worked, at the unchanged default threshold.

### Verification: re-checked through the real MCP protocol, not just direct-Python

Everything above was first verified by importing the underlying modules directly —
faster to iterate on, but it skips the request-parsing layer a real client actually
goes through. All six fixes were re-run against the real fixtures a second time
through the actual `fusion-vision-mcp` server (a fresh subprocess, spawned exactly
the way `tests/test_server.py` does, over the real MCP stdio protocol), and every
one reproduced its direct-Python result exactly. The `size` cross-check needed a
harder question phrasing to reproduce end-to-end — the first phrasing tried got a
confident, correct answer from Moondream2 on its own, so the low-confidence gate
correctly never routed to it; a phrasing that tripped a self-contradiction did.

## v0.7.0 · 2026-08-27

v0.6.0 surfaced four deficiencies by *flagging* them; v0.7.0 turns each flag into an
*actionable result* by combining tools already in the project. No new models — each
new path reuses the lazily-loaded stack and adds the second signal v0.6.0 only
exposed. Every feature keeps its negative control and an honest "limit" sentence.

### `caption` — corrects the close misses, not just surfaces them

`verify_text=true` already ran the OCR-with-region head and returned the verbatim
spans; v0.7.0 also corrects the caption: each token the caption quoted that is
close to (but not identical to) a verbatim OCR span is substituted with the verbatim
text in a `caption_corrected` copy, and every change is listed in `corrections`
(quoted-in-caption, verbatim-from-ocr, box, similarity). Pure-Python
`textmatch.py` over the two Florence-2 heads' outputs — no new model. Re-verified on
the banner: the caption still says "FusionVisionMP", but `caption_corrected` now
carries "FusionVisionMCP" and `corrections` records the substitution. Limit: the
substitution is best-effort and only fires for high-similarity same-word matches;
the raw `corrections` list is always present so a caller can audit every change.

### `count_objects` — adds an actionable outline estimate on collapse

When `separable` is `"no"` (the detector collapsed overlapping instances) but the
outline still carries the lobe pattern (`silhouette.by_radial > 1`), an `estimates`
block now reports that outline count as a number — a *measurement* (angular
notches), not a judgment, so it stays within "measure, don't judge". `count` is
never overwritten. New opt-in `vqa_estimate=true` also asks Moondream2 "how many
<name>?" and attaches that judgment as `estimates.vqa` (clearly marked, not a
tally), off by default so a session that never asks keeps Moondream unloaded.
Re-verified on the flower: `separable: "no"`, `estimates.outline: 8` with `count`
still 1. Limit: it's an estimate, not a tally — the detector could not separate the
instances, so treat any number here accordingly.

### `query_image` — routes a low-confidence answer to the measurement that answers it

`check_consistency=true` already flagged flat default answers; v0.7.0 then routes
the unreliable VQA judgment to the measurement that actually answers the question
when one applies: `spatial_relations` for a contact/containment question,
`count_objects` for "how many", `ocr` for a text-reading question, attached as
`cross_check`. New pure-Python `question.py` classifies the wording and best-effort
parses the object names; the cross-check is **omitted** (not guessed) when no
measurement applies or the names can't be parsed to the required arity. Re-verified:
"does the hand touch the shield" routes to `spatial_relations`; "describe the mood"
produces no cross-check. Limit: routing is best-effort; a low-confidence answer with
no measurable fallback still has no `cross_check`.

### `score_aesthetics` / `critique_composition` — calibrated relative comparison

The predictor's documented valid use is like-with-like comparison, so v0.7.0 routes
callers there instead of a single bias-affected absolute number. New `compare_with`
parameter: both images are scored/critiqued and the result carries the per-image
scores, the `delta`, and `preferred` (`"image"`/`"reference"`/`"tie"`, tie when
|delta| < 0.05). With `style_context=true`, both media are classified and a
`cross_medium_warning` is added when they differ (cross-medium comparison is out of
calibrated scope). Re-verified: identical image vs itself → `delta: 0`,
`preferred: "tie"`. Limit: the absolute score is not recalibrated — an oil painting
still lands around 5.8; the relative delta is the actionable output, and only
within a shared medium.

## v0.6.0 · 2026-08-26

Four opt-in parameters, added to four existing tools, closing measured gaps against Claude's native
multimodal vision. No new tools, no new models. Each mitigation cross-checks the unreliable output against a
second signal or surfaces the disagreement instead of masking it — none of them make the underlying local model
more capable, since none of these four deficiencies are fixable locally.

### `caption` — misreads text embedded in the image

Florence-2's caption head produces a fluent scene description, but any text it mentions is being *described*,
not transcribed. It's a captioning model, not an OCR model, so it reproduces what a piece of text *plausibly
looks like* rather than reading it character-by-character.

**Concrete evidence:** run live against this project's own banner (`FusionVisionMCP-Dark.jpg`), it rendered the
logo as "FusionVisionMP" mid-sentence — a dropped "C". `ocr` and `query_image`, asked about the identical image,
both read "FusionVisionMCP" correctly.

**Why it happens:** captioning is a semantic/holistic task; the model has no incentive during training to get
every character right, only to produce a plausible-sounding sentence. It will confidently substitute a
wrong-but-similar string rather than fail visibly.

**Blast radius:** any name, brand, or label a caption quotes back is unverified. Treating a caption's quoted
text as fact is the actual risk — the description of the scene is usually fine, it's specifically embedded text
that's unreliable.

**v0.6.0 mitigation, and its limit:** `verify_text=true` runs the OCR-with-region head in the same call and
returns verbatim `text_regions` alongside the caption. Re-verified after shipping: the *caption* still says
"FusionVisionMP" — the head itself is unfixed — but `text_regions` now correctly returns "FusionVisionMCP" in
the same response, so a caller can catch the discrepancy without a second round trip. Claude's native vision
doesn't have this failure mode at all; it reads the logo correctly on the first pass.

---

### `count_objects` / `detect_objects` — collapses on ambiguous, overlapping instances

This is the deepest, most-tested deficiency in the project — a paper flower with overlapping petals
(`tests/sample.jpg`) returns `count: 1` from every method tried, not just the shipped one:

- Florence-2's own grounding head
- Moondream2's detect head
- Grounding DINO at every box threshold down to 0.10
- Silhouette/outline lobe-counting (`count_lobes`)
- SAM2 in segment-everything mode (one mask for the whole flower)
- CIELAB interior-colour boundary analysis

Two measurements explain why, and close the case rather than leaving it open: the flower's silhouette has
**solidity 0.984** (its outline is essentially a smooth disc — no notches for an outline method to find), and
its interior colour-boundary strength is **0.87**, barely above a plain textured blob's **0.52** (the petals
are too pastel/low-contrast for a colour method to find edges either). There is no outline evidence and no
colour evidence — nothing measurable distinguishes one petal from the next.

The project's own conclusion is a direct admission of the ceiling here: *"what separates a spot from a petal is
knowing what the object is, which is the calling model's job... not a measurement this server can make."*
That's exactly the kind of semantic call native reasoning (Claude) can plausibly make and a local detector
structurally cannot.

A related, less total failure: heavy-but-not-total overlap undercounts rather than fails outright — eight
identical shapes overlapping by roughly two-thirds of their width count as 6, not 8.

A cautionary side-finding: every approach tried to raise recall on this class of problem (lower detection
thresholds, tiled 2×2/3×3 inference, visual-exemplar prompting, interior-colour analysis) broke the project's
negative controls by turning a single object covered in texture (a spotted ball) into many objects — 6, 15, and
31 respectively. Anything that "zooms in" or "matches appearance" can't tell a spot from a petal from a stripe;
that's the same wall from a different angle.

**v0.6.0 mitigation, and its limit:** `consensus=true` (default) adds a second-opinion count and a `separable`
flag reading the silhouette's `by_distance` / `by_radial` / `agreement` fields together — `by_distance` sweeps
the distance transform, `by_radial` is a rosette-specific angular-harmonic estimator, and `agreement` is true
only when the two concur. On the flower it correctly reports `separable: "no"`: `by_distance=1` (no saddle
found — the overlapping petals read as one blob), `by_radial=8` (the angular notch pattern is still there),
`agreement=false` — flagging the collapse instead of returning a clean-looking `count: 1`. (A first-cut version
of this logic read `by_distance` alone and said `"yes"` on this exact case, sitting next to a contradicting
`consensus.agree: false` in the same payload — fixed before release.) It still doesn't count the flower; it
tells the caller not to trust the number.

---

### `query_image` — unreliable for open-ended judgment

Moondream2 is a small VLM, and it's specifically weak on open-ended judgment calls rather than closed factual
questions.

**Documented failure:** asked to "describe anything wrong in this image" across six different images that all
had a real, human-visible defect, it answered a flat "None" on all six. It's also documented to give the same
yes/no answer across genuinely different images — a default response pattern, not a real observation.

**Why this matters structurally:** this is the exact reason `spatial_relations` was designed to only *measure*
(contact, gap, containment) and never render a verdict — the project decided early that a small VLM can't be
trusted for judgment the way the calling model's own reasoning generally can.

**v0.6.0 mitigation, and its limit:** `check_consistency=true` asks a rephrased control question and returns
`{answer, control_answer, consistent, confidence}`. `confidence` is `"low"` either when both answers reduce to
the same default token ("None"/"None") or when the two substantively disagree. Verified live on the flower with
"describe anything wrong": one answer invented a missing centerpiece, the control answer said nothing was wrong
— a direct contradiction, correctly flagged `confidence: "low"`. (A first-cut version only caught the
agreed-default case and scored this exact contradiction `"normal"` — fixed before release.) This surfaces
unreliability; it doesn't make the underlying judgment correct.

---

### `score_aesthetics` / `critique_composition` — photography bias

The scoring head is CLIP plus the LAION "improved aesthetic predictor," trained specifically on human ratings
of photographs (the LAION/SAC/AVA datasets).

**Concrete evidence:** Hokusai's *The Great Wave off Kanagawa* — a world-famous masterpiece — scores only
5.83/10 ("average"). The predictor has no calibrated sense of quality for paintings, illustrations, or other
non-photographic media; the score isn't *wrong* exactly, it's answering a question ("does this look like a good
photograph") that doesn't apply.

**Scope, not just medium:** it's also not meant for absolute cross-content comparison generally — designed for
comparing like with like (edits of one image, or several shots of one subject), not as a verdict on quality
across genuinely different images.

**A finding that looked like a second bug but wasn't:** a flat vector-style graphic scored nearly identically
crisp vs. heavily blurred (4.19 vs 4.16), which initially read as blur-insensitivity. Retested on an actual
photograph at four blur levels, the score dropped monotonically (5.23 → 4.40 → 4.08 → 3.95) — the tool works
correctly within its documented scope; the vector-graphic test was invalid methodology, not a defect. Worth
knowing so it isn't re-investigated as if still open.

**v0.6.0 mitigation, and its limit:** `style_context=true` classifies the image's medium via zero-shot CLIP (16
style prompts, reusing the already-loaded backbone) and returns `style` / `style_distribution` alongside the
score. Verified: a vector-graphic fixture correctly classifies as "vector graphic" (59%) rather than
photograph. This tells the caller the score is for a non-photographic medium so it's read with the right
caveat — it does not recalibrate the score itself; an oil painting will still land around 5.8 regardless of how
good it actually is.
