# FusionVisionMCP

Fork of [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2), renamed and rebranded as
**FusionVisionMCP** (Python package `fusion-vision-mcp`, module `fusion_vision_mcp`, binary
`fusion-vision-mcp.exe`), adding Moondream2 VQA and counting (`query_image`, `count_objects`), object grounding
(`detect_objects`, `dense_region_caption`), CLIP/LAION aesthetic scoring, batch analysis, and a `--memory-mode`
that controls how long models stay resident after use. See [README.md](README.md) for the tool and option
reference.

## This is an editable install — checking out a branch changes the running server

`fusion-vision-mcp` is installed as an editable `uv` tool pointing at this checkout (`uv tool install --editable . --force ...`). The MCP server both Cline and Claude Code run **is whatever branch is checked out here**, live, with no reinstall needed to pick up source changes.

This bit us once already: checking out `main` to catch up with upstream silently removed the idle-release option and the Moondream tools from the running server, and it came back as `✘ Failed to connect` in both clients because `main` doesn't accept that flag. The same trap now applies to `--memory-mode`, which is newer still. Always confirm you're on `feature/moondream-vqa-and-idle-release` (or a later feature branch) before assuming the server has this fork's tools, and re-run `uv tool install --editable . --force ...` after any change to `pyproject.toml` — source edits are live immediately, but dependency changes are not until reinstalled.

## `pyvips` is required transitively, not by this package directly

`moondream2`'s `trust_remote_code` module imports `pyvips`, which needs native libvips libraries that plain `pip install pyvips` does not provide. Without them the server fails at startup with `ModuleNotFoundError: No module named '_libvips'`, or `OSError: cannot load library 'libvips-42.dll'` if libvips isn't on `PATH`.

The fix already applied here: `pyproject.toml` declares `pyvips[binary]`, which ships the libvips shared libraries inside the wheel. No system libvips install and no `PATH` entry are needed. If you ever see either error, check that the dependency still reads `pyvips[binary]` rather than plain `pyvips` — this is exactly the failure that broke the upstream project's own test suite for anyone without a system-wide libvips.

## Commands

```powershell
# Reinstall after a dependency change (not needed for source-only edits)
uv tool install --editable . --force --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# Lint / format / type-check
uvx ruff@0.16.1 check src tests
uvx ruff@0.16.1 format src tests
uv run --with mypy --with types-requests --with scipy-stubs mypy src

# Tests (integration tests spawn the real server and download Florence-2-base on first run)
uv run --with pytest --with anyio pytest tests -q
```

`uv run` and `uv tool install` can fail here in ways specific to the environment, not the code — see the OneDrive note below if this checkout is ever moved back under a synced folder.

## Remotes

`origin` is this fork (`Whoawhen/FusionVisionMCP`, public, renamed from `warrens951/mcp-florence2` on 2026-08-20 —
both the repo name and the GitHub account username changed that day); `upstream` is `jkawamoto/mcp-florence2`,
kept only for reference. `main` no longer tracks `upstream/main` as a clean mirror — that convention existed to
keep a PR back to upstream possible, but by 2026-08-27 the two projects had diverged too far for it to mean
anything (11 tools across five local models here vs. upstream's original three tools on Florence-2 alone; the
`mcp_florence2` package was deleted outright early on, replaced by this repo's own `fusion_vision_mcp` — see
below). Commit directly to `main` as needed; there's no longer a reason to keep its history free of
fork-specific commits. Attribution to the original author is retained regardless, via the MIT copyright header in
`florence2.py` and the `authors` list in `pyproject.toml`, since this fork's `Florence2` wrapper class still
descends from his original implementation even though it's since grown well past it.

## Two OCR paths — pick by text type, don't default to `ocr`

`ocr` (Florence2's `<OCR>` head) and `query_image` (Moondream2 VQA, asked to transcribe) both read text, but they fail differently, so route by what the text looks like rather than always reaching for `ocr`. This is now stated directly in both tools' MCP descriptions (see `src/fusion_vision_mcp/__init__.py`), since a calling agent reads those at tool-selection time, not this file:

- **`ocr` (Florence2)** for dense, printed, document-style text — receipts, scanned pages, paragraphs. It's built for verbatim character-level transcription over a lot of text.
- **`query_image` (Moondream2)**, e.g. `question="What does the text/watermark say, exactly?"`, for stylized/logo/cursive/low-contrast text — photo watermarks, signage, logotypes. Florence2's OCR head misreads these; it read a real watermark reading "Ride the Sky / Equine Photography / ridetheskyequine.com" as "SQUINT PHOTOGRAPHY / squentphotography.com" (2026-08-20 test on `testpette.jpg`). Moondream2 read the same image correctly.

Don't hard-route `ocr` to always call Moondream instead — Moondream is a VQA model, not a transcription specialist, and is more prone to paraphrasing rather than verbatim-transcribing long or dense text blocks. Keep both tools and choose per call.

There's a third path that is not a text tool at all and must not be used as one: **`caption` describes text, it does not transcribe it.** Tested live on this repo's own banner (2026-08-25): `caption` rendered the logo "FusionVisionMCP" as "FusionVisionMP" mid-sentence, while `ocr` and `query_image` both read the same image exactly right. A caption quoting a name, brand or label is not evidence of what it says — confirm it with `ocr` or `query_image` per the routing above. This is now stated in `caption`'s MCP description, for the same reason the OCR routing is.

## Multi-column layouts are handled by geometry, not by asking the model harder

`ocr` reads strictly in raster order, so a document laid out in side-by-side columns (a two-column form, meeting
notes, a resume) gets its fields interleaved: a synthetic two-column fixture (`tests/layout_two_column.png`) came
back as `Attendee: Alice, Location: Room 4B, Attende: Ben, Duration: 45 min, ...` — alternating between two
unrelated columns line by line, with two names mangled in the process (2026-08-25).

**Re-prompting `query_image` to read the columns separately is not a reliable fix.** Four phrasings were tried —
asking for both columns in one call, asking for each column separately, insisting on an exact line count, asking
for a table — and `Deadline: Sept 10` never appeared in *any* of them, matching Moondream2's documented tendency
to paraphrase rather than exhaustively transcribe (see the OCR-routing note above). One phrasing even bled left-
column fields into the right-column answer.

**What ships instead: `layout.find_column_splits`/`split_columns` (`src/fusion_vision_mcp/layout.py`), pure
numpy/PIL, no model call.** It sums ink pixels per x-column, finds a vertical strip with near-zero ink density
that's wide enough and far enough from the edges to be a real gutter rather than word-spacing or a margin, and
crops there. `ocr` calls it automatically on every page: each column is OCR'd independently and the results
joined in reading order, so nothing has to reason about layout *and* transcribe exhaustively in the same call.
On the fixture above this recovers all ten fields in the correct order, including the one `query_image` never
produced.

Two details were load-bearing enough to test explicitly, both in `tests/test_layout.py`:

- **A ruled divider line down the middle of a real gutter must not defeat detection.** A thin (≤3px) ink run
  flanked by gutter on both sides is bridged and folded into the gutter, rather than being read as a second,
  narrower column boundary. `tests/layout_two_column_ruled.png` produces the identical split to the unruled
  version.
- **A page heading spans the full width above the columns** and would otherwise mask a real gutter that only
  starts below it — the top 15% of the image is excluded from gutter detection for exactly this reason.

**Negative controls, held before this shipped:** a table (`tests/layout_table.png`) has its own column gaps, but
every row carries ink in most columns, so there's no vertical strip blank across the whole body height — it is
correctly *not* split, and Test 1 in the original comparison (a clean invoice table) already showed both `ocr`
and `query_image` handle a proper table fine without this. A single wrapped paragraph
(`tests/layout_paragraph.png`) and a stray content sliver near the edge are also correctly left unsplit. The
approach generalizes past two columns without new code — `tests/layout_three_column.png` splits twice, at both
gutters.

**v0.7.1 found a second false-positive mode, and it was misdiagnosed at first.** A dense, small (11pt) single
paragraph came back scrambled into out-of-order word fragments, and a "Second Vision Pass" comparison against
Claude's native vision (2026-08-26) filed this as a Florence-2 *legibility* problem at small text scale.
Direct testing disproved that: calling Florence-2's OCR head on the un-upscaled original text, bypassing this
module entirely, reads it correctly. The real cause is the gutter detector above, misfiring on this specific
input: with only two of the paragraph's three lines landing inside the analyzed page body (the heading-exclusion
band at the top absorbs part of the first line), word-gaps in those two lines coincidentally lined up closely
enough to pass the gutter test, producing six spurious splits and slicing the paragraph into fragments as narrow
as 26px. Fixed by counting independent text lines first (`layout._count_line_bands`) and refusing to split when
there are more than one but fewer than three — too few for that kind of coincidental alignment to mean anything
real. Every existing multi-column fixture (five-plus lines) clears this floor easily; `layout_small_text_paragraph.png`
(committed from the exact fixture that exposed the bug) is the regression test.

**That fix eliminates the scrambling, but does not make Florence-2's OCR head perfect at small text.** Verified
end-to-end through the real MCP server (fresh subprocess, real protocol call, not a direct-Python import) on the
same fixture: the transcription now comes back in the correct order with no interleaving, but it still drops two
words ("font", "pixel"), duplicates one ("wide"), and merges two words across punctuation where a space should be
("scale,independent", "directlycheckable") against the fixture's known-exact source string. That is ordinary
small-text transcription error — a pre-existing, separate limitation of Florence-2's OCR head that this fix was
never intended to solve and does not claim to. Don't read "no longer scrambled" as "verbatim at small scale";
for text this small, a caller that needs the exact wording should still treat the output as approximate rather
than confirmed.

## Counting is a separate tool from detection, and it is still not solved

`detect_objects`' region count was never a tally (see the `wing` and `sword blade` cases in
[README_DETAILED.md](README_DETAILED.md#detect_objects-)), so `count_objects` routes "how many" elsewhere.

**Grounding DINO backs it, chosen by measurement rather than taste.** Moondream2's detect head held the job
first; both were built and benchmarked against the same cases before either shipped. They tie on eight shapes
separated (8) and touching (8), but Moondream collapses to **1** when those same *separated* shapes are asked
for as `pink circle` instead of `petal`, where Grounding DINO still returns 8 — the class-name sensitivity is
gone. It is also ~4x faster (2.1s vs 8.6s per call). Both pass the negative controls (one blob → 1, a rod → 1).

Grounding DINO returns one box around the whole arrangement *in addition to* the instances — 68% of the frame
when they are separated, and carrying the **highest** score, so a confidence cut removes the real instances
first. `grounding_dino.py` drops boxes that swallow most of the others' centres and reports
`group_boxes_dropped`. Without that filter every count is one too many.

**Tune counting against `benchmarks/`, never against one image.** The fixture suite draws its synthetic cases
programmatically, so their counts are exact by construction, and it includes eight negative controls that are
each exactly one object. Two settings were chosen from it and should not be nudged without re-running it:
the box threshold (0.25 → **0.15**) and the envelope filter's floor (4 → **3** boxes, since two instances plus
their envelope is only three). Together those took exact positives from 6/10 to 9/10 and mean absolute count
error from 1.07 to 0.10, holding all eight controls.

The sweep is also the clearest illustration of why this repo insists on negative controls. Thresholds of 0.125
and 0.10 score a **perfect 10/10 on positives** — and fragment a spotted ball into 3–4 objects and a rough-edged
rod into 2. Optimising on positives alone would have shipped exactly the failure mode the project exists to
avoid. 0.15 is the *lowest* threshold at which every control still holds, and that is the reason it is the
default.

**The honest negative result: the flower is not counted by anything here.** `tests/sample.jpg` — a paper flower
with overlapping petals — returns 1 from *every* approach tried: Florence-2 grounding, Moondream's detect head,
Grounding DINO (at every box threshold down to 0.10), the `count_lobes` outline measurement, SAM2 in
segment-everything mode (one mask for the whole flower), and a CIELAB interior-colour experiment.

Two measurements explain it, and together they close the case. Its silhouette has **solidity 0.984**, so the
outline is essentially a smooth disc — nothing there. And its interior colour boundary strength is **0.87**,
barely above a plain textured blob's 0.52, because the petals are pastel and low-contrast; the interior
experiment returns 1 even with its validity gate disabled entirely, so that is an absence of signal rather than
a threshold rejecting one. See `benchmarks/interior_structure.py`, which is kept *only* to make this
reproducible. Don't spend another pass on silhouettes, detectors, or colour for this image.

That experiment also failed the controls it had to pass: a striped ball came back as **7** objects and a
four-colour logo as **4**. Interior colour cannot distinguish a pattern from a group of parts, and fixing that
would still not deliver the flower, because the contrast is not there to begin with.

**CountGD was ported and measured, and is worse than what ships.** It was the counting plan's recommended
candidate, and the port turned out easy: its checkpoint is Grounding DINO Swin-B plus a single 1x1 conv for
visual exemplars, so HF's own conversion mapping loads it onto stock `GroundingDinoForObjectDetection` with zero
substantive missing keys, CPU-only, no compiled ops — none of the GCC/CUDA/Python-3.9 apparatus its repository
demands. On identical fixtures it scores 8/10 positives against 9/10, holds **4 of 8** negative controls against
8/8, and triples the mean count error. It misses a rough-edged rod entirely (0 objects), and the flower comes
back as 11, 4 and 5 at three resolutions of the *same image* — a spread that is noise, not a count. Rejected;
see `benchmarks/countgd_spike.py`, which is kept only to make that reproducible.

**Tiling and exemplar prompting were also tried, and fail the same way.** Tiled inference (2x2 and 3x3
overlapping crops, merged by IoU) buys one point on the heavy-overlap case and costs 5-12x the latency while
dropping to 7/8 and then 5/8 negative controls. CountGD's visual-exemplar path, reimplemented onto the HF graph,
breaks five of six controls. See `benchmarks/tiling.py` and `benchmarks/countgd_exemplar.py`.

**All four rejected approaches failed in exactly the same place, and it is worth naming.** Interior colour,
tiled crops, and exemplar matching each raise recall by attending to sub-object detail — and each one turns the
**spotted ball**, a single object covered in high-contrast spots, into many objects: 6 at 2x2 tiling, 15 at 3x3,
and 31 with an exemplar. Any future method that zooms in, matches appearance, or reads interior contrast will
meet the same wall. A spot, a stripe, a logo segment and a petal are the same thing to a detector; what
separates them is knowing what the object *is*, which is the calling model's job and not a measurement this
server can make. Test any new idea against `neg_spotted_ball` first — it is the cheapest possible disproof.

One trap the text-only spike recorded is worth knowing generally: **safetensors deduplicates shared tensors.** CountGD's
six `bbox_embed` heads are stored once with the aliases in `__metadata__`, so a naive `load_file` leaves 66
decoder box-head tensors missing and randomly initialised. HF ties those weights, so the outputs looked fine
anyway — restore aliases from the metadata before converting any checkpoint, and check *substantive* missing
keys rather than trusting that inference produced plausible numbers.

Two rules follow, both stated in the tool's own MCP description because a calling agent reads that and not this
file. A low count on something expected to be many means *"could not separate them"*, not a tally. And `count`
is never rewritten by the silhouette check — `count: 1` beside `silhouette.lobes: 8` reports two methods
disagreeing, which is information, rather than hiding one behind the other.

The Moondream pin still moved `2025-01-09` → `2025-06-21` (`src/fusion_vision_mcp/moondream.py`) and stays there
for `query_image`: it fixed a measured *self-consistency* failure, where asked to count the petals in
`tests/sample.jpg` the old pin answered 12 then listed 6 colors for them, and the new pin answers 10 and lists
exactly 10.

## `count_lobes` splits a silhouette, and the pixel floor is what makes it safe

`geometry.count_lobes` estimates how many repeated parts compose one mask, for when a detector collapses a group
into a single region. Two estimators, reported side by side and never reconciled: a distance-transform level
sweep that requires a count to hold across a *contiguous run* of levels (persistence without a merge tree), and
a hull-residual angular harmonic that only speaks for rosettes and returns `0` for *not measured* otherwise.

The hull step is not decoration. A raw radius profile reports a **square as four lobes** and a rod as two,
because those shapes are genuinely non-circular; dividing by the convex hull's radius measures *concavity*
instead, so every convex shape scores zero by construction and the negative controls pass structurally rather
than by tuning a threshold.

The constant that actually matters was found by sweep, not judgement: `_SMOOTH_FLOOR_PX`. A rod with ±2px edge
jitter — standing in for a photographic silhouette — **splits into 8 spurious lobes at a floor of 1.0px** and
holds at 1 from 2.0px upward, at every sigma tried. The scale-relative `_EDT_SMOOTH` term cannot defend against
this, because a tenth of a thin object's inradius is under a pixel. Raising sigma instead of the floor breaks
the positives: 8 discs at 40% overlap collapse to 1 at sigma 0.25. If you touch either constant, re-run the
whole case table, negative controls included.

## `spatial_relations` measures; it does not judge

`spatial_relations` (Grounding DINO boxes → SAM2 masks → `geometry.py`) reports numbers and deliberately stops
there. That split came out of testing, not taste: Moondream answered a plain "describe anything wrong in this
image" with a flat `"None"` on six different images that all contained a real, human-visible defect, and closed
yes/no questions gave the *same* answer across genuinely different images often enough that the answer was
clearly a default rather than an observation. A small VLM does not reliably supply that judgement. The calling
model does — and what it cannot do is measure, so that is what the tool provides.

**Locating the named objects doesn't reliably use color to discriminate, and this was found the hard way.** A
usage-quality pass (2026-08-26) built a synthetic scene with one red, one blue and one green circle and asked
`spatial_relations` for `['red circle', 'blue circle', 'green circle']`. Every query returned the *same three
boxes* — the detector (both Florence-2's grounding head and Grounding DINO were tested; both do this) found
"circle" and largely ignored the color word. The relations computed from that were nonsense: "red circle" vs
"blue circle" came back `overlapping` with `a_inside_b: 1.0`, because both labels pointed at the same physical
region compared against itself.

What made this fixable: the correctly-matching box scored highest for its own query, every time, in every test
run (5/5 on the color case, plus the pre-existing size-disparity case in `spatial_containment.png`). Switching
the detector from Florence-2's `detect_objects` (no per-box confidence available) to Grounding DINO (which
already backs `count_objects` and does carry scores) and keeping only the single best-scoring match per
requested name turned that ranking signal into a real filter. Verified end-to-end through a freshly spawned
server on both fixtures: three distinct, correctly-colored boxes with clean `touching`/`separate` relations, and
`b_inside_a: 1.0` on the containment case with no more self-comparison noise. Regression tests:
`test_spatial_relations_discriminates_same_shaped_objects_by_color` and `test_spatial_relations_measures_containment`
in `tests/test_server.py`, against `tests/spatial_touch_separate.png` and `tests/spatial_containment.png`.

The real cost of this fix: `spatial_relations` now assumes **one instance per requested name**. A scene with two
swords and you ask for `'sword'` twice gets you the same single best match twice, not two different swords —
give them distinguishing names, or use `count_objects` for an actual tally. This wasn't a real regression so
much as making an already-fuzzy assumption explicit: the tool never had a principled way to pair multiple
same-label instances (it just returned everything and computed every pairwise relation), so this trades that
loosely-defined behavior for one clear, documented rule.

**A companion finding from the same pass turned out not to be a bug.** `score_aesthetics` scored a crisp flat
vector-style graphic and a heavily blurred, noised version of it almost identically (4.19 vs 4.16). Re-run on an
actual photograph (`tests/sample.jpg`) at four blur levels, the score dropped monotonically (5.23 → 4.40 → 4.08
→ 3.95) — blur sensitivity works fine within the tool's already-documented scope ("rates photography, not fine
art"). The flat-graphic test was invalid, not the tool; see the `score_aesthetics` section of
`README_DETAILED.md` for the numbers. Recorded here so it isn't re-investigated as if it were still open.

Two things follow for anyone extending this:

- **Prefer aggregate statistics over a mask to fine topological derivatives of one.** Overlap fractions,
  distance transforms, principal axes and per-band centroids all average over many pixels and were stable on
  real photographs. Skeleton-based measures were tried first for curvature and for counting branch tips: both
  produced *inverted* results on real images, because `skeletonize` on a rough silhouette turns bark-level
  texture into spurious branches. `geometry.straightness` gets its centreline from band centroids for exactly
  this reason.
- **Validate any new metric against a negative control, not just a positive one.** Every check in `geometry.py`
  was measured on a known-good image as well as a known-bad one; two candidate checks were dropped precisely
  because the "good" case scored worse than the "bad" one, which a positive-only test would have hidden.

`geometry.py` is pure numpy/scipy and is unit-tested against synthetic masks in `tests/test_geometry.py`, so
its behaviour can be checked without downloading a model.

## v0.6.0: four tools got an opt-in flag that surfaces unreliability instead of hiding it

Four gaps against Claude's native vision were tracked as open: `caption` misreading embedded text (this
project's own banner logo came back "FusionVisionMP"), `query_image` on open-ended judgment (documented flat
"None" on six images with real defects), `count_objects`/`detect_objects` on overlapping cases (the flower — see
above), and `score_aesthetics`'s photography bias. None of these are fixable locally — a caption head that
paraphrases can't be made to transcribe, a small VQA model can't be made to reason reliably, a detector with no
notion of object identity can't separate a spot from a petal, and a CLIP head trained on photographs can't be
retrained into a fine-art critic without new weights. What shipped instead, on all four existing tools with no
new tools and no new models: an opt-in parameter that either cross-checks the unreliable output against a second
signal, or surfaces the disagreement instead of masking it. Same philosophy as the "`spatial_relations` measures;
it does not judge" section above — measure and flag, don't paper over.

- **`caption(verify_text=true)`** also runs Florence-2's `<OCR_WITH_REGION>` head and returns verbatim
  `text_regions` alongside the caption. Re-verified live against the actual `FusionVisionMCP-Dark.jpg` banner:
  the caption still reads "FusionVisionMP" (the head itself is unchanged and still wrong), but `text_regions`
  correctly returns `[{"text": "FusionVisionMCP", "box": [562, 211, 1493, 300]}]` in the same call. This closes
  the gap of having to guess or make a second blind call, not the gap of the caption head misreading text.
- **`query_image(check_consistency=true)`** asks a rephrased control question and returns
  `{answer, control_answer, consistent, confidence}`.
- **`count_objects(consensus=true)`** (default on) adds a `dense_region_caption`-based second opinion and a
  `separable` flag reading `count` against the silhouette's `by_distance`/`by_radial`/`agreement` fields.
- **`score_aesthetics`/`critique_composition`(`style_context=true`)** classifies the image's medium via
  zero-shot CLIP (16 style prompts, reusing the already-loaded aesthetic backbone) and returns it alongside the
  score, so a non-photographic result is read with the documented caveat instead of as an absolute verdict.

**Both of the first two shipped once already wrong, and both were caught by testing against the specific
documented case rather than a generic unit test.** `separable` originally read only `by_distance` (the
silhouette's `lobes` field), so on the flower/petal case — the canonical example the flag exists to catch — it
returned `separable: "yes"` sitting directly next to `consensus.agree: false` in the same response: two fields
in one payload contradicting each other. The fix makes `by_radial` (the rosette-specific angular-harmonic
estimator) and `agreement` load-bearing: `by_distance=1, by_radial=8, agreement=false` — a detector collapse the
outline's angular structure still catches — is now the one case documented to return `"no"`. Separately,
`_vqa_consistency` originally scored `confidence: "low"` only when both answers reduced to the *same* short
default token (e.g. "None"/"None"); asked to "describe anything wrong" on `tests/sample.jpg`, one answer invented
a missing centerpiece and the control answer said nothing was wrong — a direct, substantive self-contradiction —
and the original logic scored that `"normal"`. Fixed so `confidence` is `"low"` whenever the two answers
disagree, not only when they agree on a default. The lesson both cases share: a flag meant to catch a known
failure mode has to be tested against that exact failure mode, not just checked for producing *a* valid-shaped
value — the original `count_objects` test only asserted `separable in ("yes", "no", "unknown")`, which the buggy
version also satisfied.

**What this does and does not buy.** All four are opt-in (default `false` except `count_objects`'s `consensus`,
which is cheap and on by default) and none of them make the underlying model more capable. They convert a
silent wrong answer into a flagged one a calling agent can act on — decline to trust it, fall back to
`spatial_relations` or its own reasoning, or ask the user. That is the ceiling for what a local model stack can
do about a gap that is genuinely about semantic understanding rather than measurement.

## v0.8.0: The "Agentic Implementation" Refactor (Sprints 6–16)

Sprints 6 through 16 shifted the project from raw model wrappers to "agentic" capabilities—making the server smart enough to route, measure, and cross-check its own answers rather than just passing VLM hallucinations through to the user.

**The Routing Cascade (Sprint 5) & Adaptive Thresholding (Sprint 6)**
We needed `count_objects` to handle both real photographs and flat vector art without the caller explicitly passing `clip_art=True`. We originally tried zero-shot CLIP (SigLIP2, Sprint 4) for domain routing, but found it confidently misclassified real photographs as clip art. The fix was model-free (Sprint 15): a simple pixel-statistics check (unique RGB colors per 1,000 pixels) cleanly separates the two domains.
Similarly, `count_objects` using Grounding DINO on photographs originally used a static threshold (`0.30`). On dense scenes (like a flock of birds), it failed entirely. Sprint 6 introduced adaptive thresholding: it sweeps the threshold down to `0.10` if no boxes are found at `0.30`. This introduces a known risk (the "danger zone" of hallucinated boxes), so it is strictly gated: the lower thresholds are only accepted if `score_aesthetics` (or rather, its internal confidence) confirms the image is actually densely populated. We measure, we don't blindly lower the bar.

**Granite-Docling OCR & Text Fusion (Sprints 7 & 8)**
Florence-2's caption head paraphrases and misspells text in the image. We integrated `ibm-granite/granite-docling-258M` as a standalone OCR specialist (loaded lazily to save ~515MB). We did not replace Florence-2; instead, `caption` now uses `auto_verify_text=true` to scan the generated prose for signage words. If found, it crops and upscales the text regions, cross-checks against Docling, and substitutes the verbatim text back into the caption. This avoids loading the heavy OCR model for non-text images.

**Ground-Truth Suites (Sprints 15 & 16)**
We historically relied on synthetic, geometric fixtures. We realized this was a massive blind spot. We pulled 35 real photographs from COCO val2017 and 24 from Open Images v7 into `benchmarks/`. Every structural change is now regression-tested against real-world complexity, not just white backgrounds.

**Technical IQA & Reasoner (Sprints 11 & 12)**
We needed to split aesthetics into `photographic_aesthetic`, `technical_quality`, and `artistic_judgment`. 
- `technical_quality` uses MUSIQ via ONNX. We chose MUSIQ over CLIP-IQA because it produces a native 0-100 score and uses 20% less memory, and we siloed it behind an `[iqa]` optional dependency so we don't force ONNX binaries on everyone.
- `artistic_judgment` uses `VisionReasoner` (`reasoner.py`), an optional Ollama-backed LLM backend. A strict invariant here: the LLM is allowed to interpret, but it is explicitly forbidden from silently overriding direct physical measurements (like object counts). The code discards LLM-fabricated measurements and re-attaches the raw deterministic ones.

**Structured Visual Inspection (Sprint 10)**
VLMs are notoriously blind to generative AI artifacts (e.g., humans with 3 arms). We built `structured_analysis` into `query_image` to physically measure anatomy using Grounding DINO (`person`, `arm`, `leg`, etc.) and evaluate against physiological ratios (e.g., `arm > persons * 2 + 1`). If the ratio fails, it raises an `Anomaly` backed by a physical `Observation`, overriding the VLM's hallucination. (Sprints 17-19 later refined this by enforcing strict part-to-person association and bounding-box deduplication to ensure the anomaly flags aren't just detector noise).

## Package import must stay torch-free, or MCP clients time out on connect

`__init__.py` used to import `Aesthetic`/`Florence2`/`Florence2SP`/`GroundingDino`/`Moondream`/`Sam2` at module
level, purely to get their names for use inside lazy `IdleProxy(IdleReleased(lambda: ...))` wrappers in
`app_lifespan`. Every one of those five wrapper modules does `import torch` (and `transformers`) at its own top
level, so the package import paid that cost regardless of whether any model was ever actually used — 4.5s warm,
and a measured 14.1s cold under memory pressure (5.2GB free of 31.5GB, 62 other `python.exe` processes). Real
successful connects in this project's own history ranged from 4.2s to 21.2s; several genuinely timed out against
Claude Code's 30s `CONNECT_TIMEOUT`, logged as zero `Server stderr:` output in 30 seconds — not a hang, just an
import that hadn't finished.

Fixed by moving every constant/enum `__init__.py` needs as a function-signature default (`DEFAULT_*`,
`MASK_DECODE_RESOLUTION`, `CaptionLevel`) into a new `constants.py` with no torch dependency, and deferring the
six heavy imports entirely. Each wrapper module still imports its own constant(s) back from `constants.py` and
re-exports them, so `from fusion_vision_mcp.grounding_dino import DEFAULT_BOX_THRESHOLD` (used by
`tests/test_grounding_dino.py`) keeps working unchanged.

**The first attempt at deferring the six class imports was wrong, and it is worth recording why.** Hoisting them
from module level to the *top of `app_lifespan`* measurably made no difference — `initialize` still took 15s.
`app_lifespan` runs as part of server startup, before the MCP handshake is answered, so anything unconditional
there is exactly as blocking as a module-level import; only the log line's timing moved. The fix that actually
worked: move each import *inside its own lazy factory function* (`_florence2`, `_moondream`, etc.), so it only
runs when `IdleReleased.get()` calls the factory on first real tool use (see `idle.py`) — matching the laziness
the `IdleProxy` wrapper already claimed to provide. Verified end-to-end against a real subprocess driving the
actual MCP wire protocol (not a direct Python import): `initialize` now returns in ~1.05s warm (down from 4.5s;
14.1s cold), with no `torch`/`transformers` appearing anywhere in `python -X importtime -c "import
fusion_vision_mcp"`'s trace. A subsequent real tool call (`caption`) still logs `Loading Florence-2` and pays the
full model-load cost, just deferred to when it's actually needed rather than at connect time — nothing about
model behavior changed, only when the cost lands.

Since return-type annotations on those per-model factory functions (e.g. `def _florence2() -> Florence2:`) are
evaluated at `def` time (this file has no `from __future__ import annotations`), and the six class names only
exist under a `TYPE_CHECKING` guard now, the annotations have to be quoted forward references (`-> "Florence2"`)
— ruff's `TC004` catches the unquoted form as "used at runtime" and it would otherwise be a `NameError` the
first time `app_lifespan` runs.

This also fixed something that was arguably already broken independent of the timeout: with the default config
(no `--cache-model`, so `subprocess=True`), every model call already ran in a spawned child process
(`multiprocessing`, start method `spawn` on Windows) that reimports `fusion_vision_mcp` from scratch and pays its
own torch import regardless of what the parent already imported. The parent's eager import bought nothing for
inference in that mode — it was pure waste layered on top of the timeout risk.

## A note on where this lives

This checkout used to live under OneDrive. `uv tool install` failed there with a hardlink error, and `uv run`/`uv sync` separately failed removing a `.dist-info/licenses` directory that OneDrive had turned into a cloud placeholder — neither error message mentions OneDrive. Moving the checkout to `C:\AI\MCP\FusionVisionMCP` (a plain local path, renamed from `C:\AI\MCP\mcp-florence2` on 2026-08-20) resolved both. Keep it out of any synced folder.
