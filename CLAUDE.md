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

`spatial_relations` (Florence-2 boxes → SAM2 masks → `geometry.py`) reports numbers and deliberately stops
there. That split came out of testing, not taste: Moondream answered a plain "describe anything wrong in this
image" with a flat `"None"` on six different images that all contained a real, human-visible defect, and closed
yes/no questions gave the *same* answer across genuinely different images often enough that the answer was
clearly a default rather than an observation. A small VLM does not reliably supply that judgement. The calling
model does — and what it cannot do is measure, so that is what the tool provides.

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

## A note on where this lives

This checkout used to live under OneDrive. `uv tool install` failed there with a hardlink error, and `uv run`/`uv sync` separately failed removing a `.dist-info/licenses` directory that OneDrive had turned into a cloud placeholder — neither error message mentions OneDrive. Moving the checkout to `C:\AI\MCP\FusionVisionMCP` (a plain local path, renamed from `C:\AI\MCP\mcp-florence2` on 2026-08-20) resolved both. Keep it out of any synced folder.
