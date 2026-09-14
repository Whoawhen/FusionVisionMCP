# FusionVisionMCP v0.8.0 — Adaptive Vision Routing & Reasoning

**Repository:** `https://github.com/Whoawhen/FusionVisionMCP`  
**Starting version:** `0.7.1`  
**Target version:** `0.8.0`

## Mission

Make FusionVisionMCP automatically choose the appropriate existing or specialist vision capability instead of requiring the caller to know which backend to select.

The core rule is:

> Use the smallest specialist that can answer the question, fuse independent evidence when available, and never convert uncertainty into false precision.

## 1. Version checkpoint — FIRST

Before implementing functionality:

1. Confirm the repository is at `0.7.1`.
2. Update the project version to `0.8.0`.
3. Update every duplicated version location.
4. Add the `0.8.0` changelog section.
5. Run the existing test suite.
6. Commit/checkpoint the version bump.
7. Only then implement features.

Suggested changelog:

```markdown
## 0.8.0

### Added
- Automatic visual-domain routing.
- Adaptive detection/counting thresholds.
- Generic-noun and blank-canvas safeguards.
- Automatic specialist OCR verification.
- Centralized detection policy.
- Semantic ambiguity reporting for inseparable instances.
- Structured open-ended visual inspection.
- Optional evidence-based reasoning backend.
- Separate technical quality and photographic aesthetic context.

### Changed
- Clip-art counting can automatically route to Florence-2.
- Caption text can be automatically verified when embedded text is detected.
- Detection/counting can adapt to clutter.
- Open-ended visual questions can use structured evidence.

### Compatibility
- Existing tool arguments remain backward compatible.
- Explicit caller options override automatic routing.
```

## 2. Established weaknesses

### F1 — Clip-art counting

Default Grounding DINO performs poorly on flat vector/icon scenes. Florence-2 grounding performs better, but only through `clip_art=true`.

**Fix:** automatic image-domain routing.

### F7 — Blank canvas / generic noun

`detect_objects("object")` can produce a near-full-frame false detection.

**Fix:** generic-query guard plus blank/presence validation. Wire the policy into `detect_objects`, not only `count_objects`.

### F9 — Distractors

A target score around `0.897` versus distractors around `0.19–0.32` is a strong confidence separation, but the fixed `0.15` threshold accepts distractors.

**Fix:** adaptive thresholding based on score distribution, absolute score, geometry, and known negative patterns.

### Small-text OCR

Florence OCR still loses/duplicates/merges small text.

**Fix:** independent OCR specialist, primarily Granite-Docling 258M.

Official model:
`ibm-granite/granite-docling-258M`

https://huggingface.co/ibm-granite/granite-docling-258M

### Caption text

Florence captioning can render `FusionVisionMCP` as `FusionVisionMP`.

**Fix:** automatically detect likely embedded text and invoke specialist OCR. Preserve the original caption and expose the correction/evidence.

### Open-ended judgment

Moondream2 can answer broad questions such as “what is wrong with this image?” with `None`.

**Fix:** structured observations plus existing measurable evidence; optionally pass that evidence to a configurable reasoning backend.

### Heavily overlapping instances

When petals/objects are visually inseparable, DINO, Florence, CountGD and SAM all return one.

**Fix:** do not fake a count. Report a measured minimum plus `semantic_ambiguity=true`. A reasoning backend may provide a separate contextual estimate.

### Aesthetic scoring

The current LAION/CLIP predictor is photograph-biased.

**Fix:** distinguish:
- technical quality
- photographic aesthetic
- artistic judgment

Do not pretend an IQA model provides artistic judgment.

## 3. Architecture

```text
IMAGE
  |
  v
DOMAIN ROUTER
  |
  +-- PHOTO ---------> Grounding DINO
  |
  +-- CLIP-ART ------> Florence grounding
  |
  +-- DOCUMENT ------> OCR specialist
  |
  v
CENTRAL DETECTION POLICY
  |
  +--> adaptive threshold
  +--> generic-query guard
  +--> full-frame guard
  +--> existing geometry/SAM evidence
  |
  v
EVIDENCE FUSION
  |
  +--> COUNT
  +--> DETECT
  +--> OCR
  +--> ANOMALIES
  +--> QUALITY
  |
  v
OPTIONAL REASONER
  |
  v
MCP RESULT:
result + evidence + confidence + disagreement + limitations
```

## 4. Domain router

Create:

`src/fusion_vision_mcp/domain_router.py`

First candidate: SigLIP 2.

Project/model information:
https://huggingface.co/blog/siglip2

Potential model:
`google/siglip2-base-patch16-224`

Use it as a lightweight zero-shot router, not as another general-purpose VLM.

Initial labels:

```python
DOMAIN_LABELS = [
    "a photograph",
    "a photorealistic image",
    "flat vector clip art",
    "vector artwork",
    "cartoon illustration",
    "digital illustration",
    "oil painting",
    "watercolor painting",
    "technical diagram",
    "document or scanned page",
    "computer screenshot",
]
```

Use top score plus margin over the second candidate.

```python
@dataclass
class DomainResult:
    domain: str
    confidence: float
    scores: dict[str, float]
    ambiguous: bool
```

Do not hard-code thresholds without benchmarking.

## 5. Routing

Create:

`src/fusion_vision_mcp/routing.py`

```python
@dataclass
class VisionRoutingPolicy:
    domain: str = "auto"
    adaptive_threshold: bool = True
    generic_query_guard: bool = True
    specialist_ocr: bool = True
    semantic_ambiguity: bool = True
```

Suggested count routing:

```python
def choose_count_backend(domain_result):
    if domain_result.domain in {"clip_art", "vector_art", "illustration"}:
        return "florence"
    if domain_result.ambiguous or domain_result.domain == "unknown":
        return "consensus"
    return "grounding_dino"
```

Explicit caller choices always override automatic routing.

## 6. Automatic clip-art counting

Modify `count_objects`:

```text
clip_art=true       -> force Florence
clip_art=false      -> preserve existing behavior
clip_art omitted    -> domain="auto"
```

If the router is ambiguous, use existing consensus rather than guessing.

Expose routing only under a diagnostic option where possible.

## 7. Generic query policy

Create:

`src/fusion_vision_mcp/query_policy.py`

```python
GENERIC_OBJECT_QUERIES = {
    "object", "objects",
    "thing", "things",
    "item", "items",
    "stuff", "something",
    "shape", "shapes",
    "entity", "entities",
}

def is_generic_query(query: str) -> bool:
    return query.strip().lower() in GENERIC_OBJECT_QUERIES
```

Do not treat every phrase containing “object” as generic.

## 8. Blank-canvas guard

Wire the guard into `detect_objects`, `count_objects`, and relevant `spatial_relations` paths.

For generic nouns:

- reject unsupported full-frame boxes
- use a lightweight presence/scene check
- retain legitimate full-frame detections when independently supported

Suggested diagnostic result:

```json
{
  "detections": [],
  "suppressed_detections": [
    {
      "reason": "generic_full_frame_without_independent_evidence"
    }
  ]
}
```

Never apply this blanket rule to specific nouns.

## 9. Adaptive threshold

Create:

`src/fusion_vision_mcp/adaptive_threshold.py`

```python
@dataclass
class ThresholdResult:
    threshold: float
    used: bool
    confidence: float
    reason: str

def choose_threshold(
    scores: list[float],
    base_threshold: float = 0.15,
) -> ThresholdResult:
    ...
```

For sorted scores:

```python
scores = sorted(scores, reverse=True)
gaps = [
    scores[i] - scores[i + 1]
    for i in range(len(scores) - 1)
]
```

A strong confidence cliff can define a candidate threshold:

```python
max_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
max_gap = gaps[max_gap_index]

if max_gap >= 0.20:
    candidate = (
        scores[max_gap_index] +
        scores[max_gap_index + 1]
    ) / 2
```

Clamp conservatively and benchmark.

Never adapt from the largest gap alone. Require a reasonable top score and valid geometry.

## 10. Preserve raw detection evidence

Do not silently replace the original result.

When diagnostics are enabled, expose:

```json
{
  "base_threshold": 0.15,
  "adaptive_threshold": 0.61,
  "adaptive_used": true,
  "raw_detection_count": 6,
  "filtered_count": 1
}
```

## 11. Central detection policy

All of these must share one policy:

- `detect_objects`
- `count_objects`
- `spatial_relations`

Create a shared helper such as:

```python
def prepare_detection_policy(
    query,
    domain=None,
    threshold="auto",
    exclude_full_frame="auto",
):
    ...
```

Do not duplicate threshold logic across tools.

## 12. Granite-Docling OCR

Create:

`src/fusion_vision_mcp/granite_docling.py`

Model:

`ibm-granite/granite-docling-258M`

https://huggingface.co/ibm-granite/granite-docling-258M

Implement lazy loading.

Do NOT guess the current Transformers API. The coding agent must inspect the current model card, installed Transformers version, and actual output structure, then add a real smoke test.

Suggested interface:

```python
class GraniteDoclingOCR:
    def __init__(self, model_id=DEFAULT_MODEL, device=None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        ...

    def ocr(self, image):
        ...
```

Also investigate current llama.cpp multimodal support for Granite-Docling. If stable, hide it behind:

```text
ocr_backend = auto | transformers | llama_cpp
```

Do not make llama.cpp mandatory.

## 13. Automatic caption OCR

Add:

```text
auto_verify_text=true
```

Detect likely embedded text from:

- existing Florence text regions
- quotation/signage language in the caption
- OCR-visible regions
- suspicious short uppercase tokens
- other conservative signals

Then:

```text
caption
  -> detect likely text
  -> specialist OCR
  -> text fusion
```

Example:

```json
{
  "caption": "A banner reading FusionVisionMP...",
  "caption_text_warning": true,
  "text_consensus": {
    "caption": "FusionVisionMP",
    "florence_ocr": "FusionVisionMCP",
    "specialist_ocr": "FusionVisionMCP",
    "agreeing_sources": 2
  },
  "caption_corrected": "A banner reading FusionVisionMCP..."
}
```

Never erase the original caption.

## 14. Small-text OCR crops

If Florence provides text boxes:

```text
full image
 -> text region
 -> 2x/3x upscale
 -> optional mild preprocessing
 -> specialist OCR
```

Benchmark:

- character accuracy
- word accuracy
- line ordering

Do not claim improvement without measurements.

## 15. Semantic ambiguity

Create:

`src/fusion_vision_mcp/ambiguity.py`

```python
@dataclass
class AmbiguityResult:
    ambiguous: bool
    reason: str | None
    evidence: list[str]
```

When independent methods all report one connected/indivisible region but contextual structure suggests multiple parts, return:

```json
{
  "count": 1,
  "count_semantics": "minimum_visible_instances",
  "semantic_ambiguity": true
}
```

Do NOT return a contextual estimate as the measured count.

Optional reasoning may return:

```json
{
  "contextual_estimate": 8,
  "contextual_estimate_confidence": "medium"
}
```

## 16. Structured visual inspection

Create:

`src/fusion_vision_mcp/inspection.py`

Replace one-shot broad judgment with structured observations:

```python
OBSERVATION_PROMPTS = [
    "List the distinct major objects visible in the image.",
    "Are any objects visibly duplicated or repeated?",
    "Are any objects visibly intersecting or passing through other objects?",
    "Are there obvious anatomical inconsistencies?",
    "Are there obvious perspective or scale inconsistencies?",
    "Is any visible text malformed, duplicated, or inconsistent?",
    "Is anything visually out of place relative to the surrounding scene?",
    "Are there visible rendering or image-generation artifacts?",
]
```

Represent:

```python
@dataclass
class Observation:
    category: str
    description: str
    source: str
    confidence: float
```

Then corroborate with:

- Grounding DINO
- SAM2
- OCR
- spatial relations
- count results
- geometry

## 17. Anomaly representation

```python
@dataclass
class Anomaly:
    type: str
    description: str
    confidence: float
    evidence: list[str]
```

Types:

```text
duplicate_object
object_intersection
anatomical_anomaly
text_mismatch
count_mismatch
spatial_anomaly
scale_anomaly
perspective_anomaly
rendering_artifact
semantic_ambiguity
```

A VLM-only observation should remain low-confidence unless independently corroborated.

## 18. query_image

Add:

```text
structured_analysis=true
```

When enabled, gather structured observations and measurable evidence.

When disabled, preserve existing behavior.

## 19. Optional reasoning backend

Create:

`src/fusion_vision_mcp/reasoner.py`

```python
class VisionReasoner:
    def analyze(
        self,
        image=None,
        observations=None,
        evidence=None,
        question=None,
    ):
        ...
```

Possible provider modes:

```text
none
local
ollama
llama_cpp
external
```

Only implement providers supported by the current project architecture.

The reasoner should receive structured evidence, not just “what is wrong?”

Example:

```json
{
  "question": "Describe anything wrong with this image.",
  "observations": [
    {
      "source": "moondream",
      "description": "Sword appears to pass through hand."
    }
  ],
  "measurements": {
    "sam_overlap": 0.31
  },
  "detections": [
    {
      "label": "sword",
      "box": [...]
    }
  ]
}
```

## 20. Reasoner output

Require structured output:

```json
{
  "judgment": "possible rendering error",
  "confidence": 0.82,
  "claims": [
    {
      "claim": "sword intersects hand",
      "confidence": 0.91,
      "evidence": [
        "moondream",
        "sam2"
      ]
    }
  ]
}
```

Direct measurements must not be silently overridden by reasoning.

## 21. Aesthetic refactor

Conceptually expose:

```text
technical_quality
photographic_aesthetic
artistic_judgment
```

Keep the existing `aesthetic_score` for compatibility.

For paintings/illustrations:

```json
{
  "photographic_aesthetic_applicable": false
}
```

Do not call a technical IQA score an artistic score.

## 22. Technical IQA

Evaluate one lightweight model only.

Candidates:

- MUSIQ
- HyperIQA
- CLIP-IQA
- lightweight ONNX alternatives

Potential model collection:

https://huggingface.co/86Cao/IQA-ONNX-Models

Select based on:

- CPU RAM
- CPU latency
- accuracy
- licensing
- dependency complexity

Create:

`src/fusion_vision_mcp/image_quality.py`

```python
class ImageQuality:
    def score(self, image):
        return {
            "technical_quality": ...,
            "model": self.model_id,
        }
```

Do not invent submetrics that the selected model does not provide.

## 23. Artistic judgment

Do not solve this with an IQA model.

Without a reasoning backend:

```json
{
  "artistic_judgment": null,
  "reason": "No artistic reasoning backend configured."
}
```

With a reasoning backend, artistic evaluation can be returned with explicit criteria and confidence.

## 24. Memory rules

Every new model must:

- load lazily
- respect existing `device.py`
- use existing idle-release mechanisms
- never initialize at MCP startup
- avoid permanent GPU residency
- avoid downloading weights during installation

Prefer sequential specialist execution:

```text
router
 -> unload
specialist
 -> unload
next specialist
```

## 25. Optional dependencies

Use optional extras in `pyproject.toml`.

Conceptually:

```toml
[project.optional-dependencies]

routing = [
    "..."
]

ocr-specialist = [
    "..."
]

iqa = [
    "onnxruntime>=..."
]

reasoning = [
    "..."
]
```

Use exact versions only after compatibility testing.

## 26. Tests

### Domain

- photograph
- clip-art
- vector illustration
- painting
- document
- screenshot
- diagram

### Detection

- blank canvas
- generic noun
- full-frame legitimate object
- specific noun

### Counting

- clean target
- cluttered target
- clip-art scene
- overlapping instances
- touching instances

### OCR

- normal text
- 11pt small text
- logo
- multi-column text
- punctuation-heavy text

### Judgment

Reuse the six known defect fixtures.

### Aesthetic

- photograph
- oil painting
- digital illustration
- CGI
- screenshot

## 27. Required regression outcomes

### Clip-art

Calling:

```text
count_objects(target)
```

without `clip_art=true` should automatically select Florence when the domain is confidently clip-art.

### Blank canvas

Calling:

```text
detect_objects("object")
```

on the blank fixture should produce no confident false full-frame detection.

### Distractors

The star fixture should use adaptive evidence to suppress the known 0.19–0.32 distractors while retaining the approximately 0.897 target.

### OCR

Measure Florence vs specialist vs fused result.

### Caption

The banner fixture must detect the `FusionVisionMP` vs `FusionVisionMCP` disagreement and preserve the correct OCR evidence.

### Overlap

The flower fixture should report semantic ambiguity rather than pretending the measured count is the true count.

### Aesthetic

Hokusai/other artwork should not be described as poor merely because a photographic aesthetic predictor is middling.

## 28. Backward compatibility

Existing arguments remain supported:

```text
verify_text
consensus
verify_silhouette
check_consistency
style_context
clip_art
threshold
exclude_full_frame
```

Priority must be:

```text
explicit user choice
    >
automatic policy
    >
default
```

## 29. Diagnostics

Where appropriate, add:

```text
diagnostics=false
```

When true, expose:

- domain
- routing
- model(s) used
- raw counts
- thresholds
- specialist results
- disagreements
- ambiguity
- timing

Keep normal responses compact.

## 30. Logging

Useful events:

```text
domain_detected
backend_selected
adaptive_threshold_selected
specialist_invoked
specialist_failed
evidence_disagreement
semantic_ambiguity
reasoning_fallback
```

Avoid unnecessary logging of OCR/image contents.

## 31. Failure handling

Optional specialists must never break the primary tool.

```python
try:
    specialist_result = specialist.run(image)
except Exception:
    logger.debug("Optional specialist failed", exc_info=True)
    specialist_result = {
        "available": False,
        "error": "specialist_unavailable",
    }
```

Do not expose tracebacks in normal MCP responses.

## 32. Implementation order

### Phase 1 — Version checkpoint
- 0.7.1 → 0.8.0
- changelog
- tests
- checkpoint

### Phase 2 — Policy infrastructure
- `domain_router.py`
- `routing.py`
- `query_policy.py`
- `adaptive_threshold.py`

### Phase 3
Automatic clip-art routing.

### Phase 4
Adaptive thresholding.

### Phase 5
Generic-noun/blank-canvas protection.

### Phase 6
Centralize detection policy across detection/counting/spatial tools.

### Phase 7
Granite-Docling OCR.

### Phase 8
Automatic caption OCR verification.

### Phase 9
Semantic ambiguity.

### Phase 10
Structured inspection.

### Phase 11
Optional reasoning backend.

### Phase 12
Technical IQA.

### Phase 13
Documentation.

### Phase 14
Full regression and benchmark.

## 33. Files likely to be added

```text
src/fusion_vision_mcp/domain_router.py
src/fusion_vision_mcp/routing.py
src/fusion_vision_mcp/query_policy.py
src/fusion_vision_mcp/adaptive_threshold.py
src/fusion_vision_mcp/granite_docling.py
src/fusion_vision_mcp/ambiguity.py
src/fusion_vision_mcp/inspection.py
src/fusion_vision_mcp/reasoner.py
src/fusion_vision_mcp/image_quality.py
```

Do not create a new module when an existing module already provides the appropriate abstraction.

## 34. Agentic coding guardrails

The coding agent MUST:

- inspect the current 0.7.1 code and tests before editing
- reuse existing helpers
- reuse image loading
- reuse device selection
- reuse idle-release logic
- reuse Florence/DINO/SAM/OCR implementations
- lazy-load every specialist
- inspect real model output shapes
- run real inference smoke tests
- benchmark improvements
- preserve explicit user options
- expose disagreements
- keep optional dependencies optional

The coding agent MUST NOT:

- replace Florence-2
- replace Moondream2
- remove Grounding DINO
- remove SAM2
- remove consensus
- remove textmatch
- silently alter historical aesthetic scores
- make network verification mandatory for existing tools
- add Qwen, DeepSeek, Kimi, GLM, or other Chinese-origin general-purpose models
- make SmolVLM2 mandatory
- claim a model is better without benchmark evidence
- turn contextual estimates into measured counts

## 35. Decision rule for new models

Before adding a model ask:

```text
Does it provide an independent capability?
```

If the answer is only:

```text
"It is another small VLM that might describe images better."
```

do not add it automatically.

Prefer:

```text
routing + specialist + evidence fusion
```

over:

```text
another general VLM
```

## 36. Final design principle

FusionVisionMCP should not attempt to become a tiny Claude.

It should become a **better evidence engine for a reasoning model**.

For straightforward facts:

```text
measure locally
```

For specialized tasks:

```text
use a specialist
```

For ambiguous evidence:

```text
report ambiguity
```

For open-ended interpretation:

```text
collect structured evidence
then optionally reason over it
```

The target behavior is:

> Less confidence when evidence is weak, higher accuracy when a specialist exists, and substantially more useful evidence when interpretation is required.
