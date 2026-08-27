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
both the repo name and the GitHub account username changed that day); `upstream` is `jkawamoto/mcp-florence2`.
Feature work happens on branches off `main`, which tracks `upstream/main` — keep `main` itself a clean mirror of
upstream so a PR can be opened from a branch without carrying unrelated history.

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

## A note on where this lives

This checkout used to live under OneDrive. `uv tool install` failed there with a hardlink error, and `uv run`/`uv sync` separately failed removing a `.dist-info/licenses` directory that OneDrive had turned into a cloud placeholder — neither error message mentions OneDrive. Moving the checkout to `C:\AI\MCP\FusionVisionMCP` (a plain local path, renamed from `C:\AI\MCP\mcp-florence2` on 2026-08-20) resolved both. Keep it out of any synced folder.
