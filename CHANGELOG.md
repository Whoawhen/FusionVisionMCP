# Changelog

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
