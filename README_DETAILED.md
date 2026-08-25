# FusionVisionMCP

> **🚧 Work in progress — not ready for use.** This project is still being built out and is not
> published as a release. Nothing here is stable yet; expect breaking changes without notice.

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Application](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml/badge.svg)](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![GitHub License](https://img.shields.io/github/license/Whoawhen/FusionVisionMCP)](https://github.com/Whoawhen/FusionVisionMCP/blob/main/LICENSE)

An MCP server fusing [Florence-2](https://huggingface.co/microsoft/Florence-2-large),
[Moondream2](https://huggingface.co/vikhyatk/moondream2), [SAM2](https://huggingface.co/facebook/sam2.1-hiera-small),
[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) and a
[CLIP](https://huggingface.co/openai/clip-vit-large-patch14)-backed
[LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) into one
computer-vision toolset. Fork of [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2), which
provides exactly three tools — `ocr`, `caption`, `process` — all against Florence-2. This fork adds everything
else: Florence-2's other task heads exposed as their own tools (`detect_objects`,
`dense_region_caption`), Moondream2 for open-ended visual question answering (`query_image`, since Florence-2 has
no VQA head), Grounding DINO for instance counting (`count_objects`, since Florence-2's grounding head's region
count is not a tally), a CLIP/LAION aesthetic scorer (`score_aesthetics`), a batch dispatch convenience
(`batch_analyze_images`), and configurable memory release. Two tools aren't just a new model wired in: no model
here answers "does this actually touch that" on its own, so `spatial_relations` is built from Florence-2 boxes,
SAM2 masks, and a from-scratch geometry module; and no model combines localization, framing, and a quality
judgment into one answer, so `critique_composition` combines Florence-2, a rule-of-thirds geometry function, the
aesthetic predictor, and (for low-scoring images) Moondream2. See the tags on each tool below.

**Legend:** 🔼 upstream (unchanged from `mcp-florence2`) · ➕ added in this fork (wraps a model already in the
stack) · ✦ novel (new capability — see [spatial_relations](#spatial_relations-) and
[critique_composition](#critique_composition-)).

You can process images or PDF files stored on a local or web server to extract text using OCR (Optical Character
Recognition), generate descriptive captions summarizing the content of the images, locate named objects and
return their bounding boxes or centre points, caption every salient region, ask free-form questions about an
image, and rate how aesthetically pleasing an image looks.

Florence-2 handles captioning, OCR, detection and grounding. Moondream2 backs the `query_image` tool, because
Florence-2 has no open-ended visual question answering task. Grounding DINO backs `count_objects`, because a
sequence-emitting grounding head cannot produce a reliable tally and is sensitive to how the object is named.
SAM2 backs `spatial_relations`, because bounding boxes cannot answer questions about contact or containment.
A CLIP ViT-L/14 backbone plus a small trained head backs `score_aesthetics`, because none of the other models has
any notion of how good an image looks.

Each model loads on first use and is released independently, so a request only pays for what it needs: OCR never
loads Moondream2, SAM2, Grounding DINO, or the aesthetic predictor. Weights are not bundled in this repository — the CLIP backbone
downloads from the Hugging Face Hub on first use and is cached locally by `transformers`, the same as any other
Hugging Face model; the small aesthetic head downloads separately from a pinned commit of its original GitHub
repository and is cached locally after the first request, with its checksum verified on every download.

> **OCR vs. query_image**: Florence-2's OCR head is built for dense, printed, document-style text and can
> confidently misread stylized, cursive, or low-contrast text (watermarks, logos, signage) rather than failing
> visibly. For that kind of text, prefer `query_image` with a question like *"What does the text/watermark say,
> exactly?"* — see the routing note in each tool's description below.

## Installation

### [Claude](https://claude.com/download)
Download the latest MCP bundle `fusion-vision-mcp.mcpb` from
the [Releases](https://github.com/Whoawhen/FusionVisionMCP/releases) page,
then open the downloaded `.mcpb `file or drag it into the Claude Desktop's Settings window.

The bundle offers one setting at install time, **Memory mode**, which controls how long the vision models stay
in memory after use — see [Memory modes](#memory-modes) for the choices. It defaults to `standard` and can be
changed later from the server's settings.

<details>
<summary>Manually configuration</summary>

You can also manually configure this server for Claude Desktop.
Edit the `claude_desktop_config.json` file by adding the following entry under `mcpServers`:

```json
{
  "mcpServers": {
    "fusionvision": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Whoawhen/FusionVisionMCP",
        "fusion-vision-mcp"
      ]
    }
  }
}
```

After editing, restart the application.

</details>

For more information,
see: [Connect to local MCP servers - Model Context Protocol](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

### [goose](https://block.github.io/goose/)

You can directly edit the config file (`~/.config/goose/config.yaml`) to include the following entry:

```yaml
extensions:
  fusionvision:
    name: FusionVisionMCP
    cmd: uvx
    args: [ --from, git+https://github.com/Whoawhen/FusionVisionMCP, fusion-vision-mcp ]
    enabled: true
    type: stdio
```

For more details on configuring MCP servers in Goose, refer to the documentation:
[Using Extensions | goose](https://block.github.io/goose/docs/getting-started/using-extensions#mcp-servers).

### [LM Studio](https://lmstudio.ai/)

Add an MCP server entry pointing at this package, using the same `command`/`args` shown in the manual
configuration above.

## Tools

### ocr 🔼

Process an image file or URL using OCR to extract text.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.

### caption 🔼

Describe what an image shows, as one detailed prose caption of the whole scene. No coordinates. Use
`dense_region_caption` instead for a separate caption and box per region, `query_image` to ask something
specific, and `ocr` to transcribe text rather than describe it. Returns one caption per page for a PDF.

> **Don't trust text a caption quotes back.** A caption that mentions a name, brand or label is *describing*
> it, not transcribing it, and Florence-2 misspells text here that it reads correctly elsewhere. Tested live
> on this repository's own banner image: `caption` rendered the logo "FusionVisionMCP" as **"FusionVisionMP"**
> mid-sentence, while `ocr` and `query_image` both read the identical image exactly right. When a specific
> piece of text matters, confirm it with `ocr` (printed, document-style) or `query_image` (stylized, cursive,
> low-contrast) rather than quoting the caption. Same routing principle as the OCR note above, one level up.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.

### detect_objects ➕

Locate a named object in an image, returning `bboxes` (`[x1, y1, x2, y2]` each), `points` (the centre of each
box) and `labels`, all index-aligned.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **object_name**: Name of the object to locate, e.g. `person`, `car`, `face`.

> **Box count ≠ object count** on an ambiguous class name. Tested live on `wing`: a griffin with
> one wing missing still returned 3 overlapping boxes (a whole-body box plus two sub-part boxes,
> all labelled `wing`). Tested on `sword blade` against an image with two fused blades: one box
> spanning both, not two. Prefer a more specific `object_name`, and treat results as candidates to
> inspect, not a reliable count. When the question is genuinely "how many", use
> [`count_objects`](#count_objects-) instead — but read its limits first, since neither tool can
> separate heavily overlapping instances.

> Boxes cannot tell you whether two objects actually touch, or whether one is inside another — they overlap the
> moment one object is merely in front of another. Use [`spatial_relations`](#spatial_relations-) for that.

> **Merged in this fork.** Centre points used to be a separate `point_objects` tool. It ran the identical
> Florence-2 grounding call and only averaged the boxes afterwards, so a caller who wanted points paid for a
> second, redundant model pass. The centres now ride along with the boxes at no extra cost.

### count_objects ✦

Count how many instances of a named object an image contains. Backed by **Grounding DINO**, which scores a fixed
set of parallel object queries and suppresses duplicates, so overlapping instances stay separate detections —
unlike `detect_objects`, whose region count is explicitly not a tally. When only one instance is found, the
region is segmented with SAM2 and its outline measured by `geometry.count_lobes`, adding a second, independent
estimate of how many parts it contains.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **object_name**: Name of the object to count, e.g. `person`, `petal`, `car`.
- **verify_silhouette**: Run the outline measurement when only one region is found. Defaults to `true`; set it
  false to keep a run off the segmenter entirely.

#### Returns:

`count`, plus `bboxes` / `points` / `labels` / `scores` for the instances found, index-aligned and in the same
pixel-space convention `detect_objects` uses. `scores` are per-detection confidences, so a count resting on
marginal detections is visible rather than implied. `group_boxes_dropped` counts detections that enclosed the
whole arrangement rather than one instance (see below). When only one region is found, `silhouette` carries the
box that was segmented and the full `count_lobes` measurement of it.

**`count` is never rewritten by the silhouette check.** The two numbers come from different methods and neither
overrides the other: `count: 1` beside `silhouette.lobes: 8` means the detector could not separate the instances
while the outline shows eight cores.

#### Why this backend

Measured head-to-head against Moondream2's detection head, which previously backed this tool, on eight identical
shapes in a ring:

| Case | Moondream2 | Grounding DINO |
| --- | --- | --- |
| Eight separated | 8 ✅ | 8 ✅ |
| Eight touching | 8 ✅ | 8 ✅ |
| Eight overlapping (~⅔ of their width) | 1 ❌ | 2 ❌ |
| Eight separated, asked as `pink circle` | **1** ❌ | **8** ✅ |
| Negative control: one blob | 1 ✅ | 1 ✅ |
| Negative control: a rod | 1 ✅ | 1 ✅ |
| Warm inference, per call | 8.6s | **2.1s** |

Grounding DINO matches on the clean cases, is **not class-name sensitive** where Moondream was badly so, and is
four times faster. It costs a ~690MB checkpoint, loaded on first use and released on the same idle timer as every
other model — a session that never counts never pays for it.

One artifact is corrected rather than passed through: asked for a repeated part, the model returns the instances
*and* one box drawn around the whole arrangement (68% of the frame when the shapes are separated, 38% when
touching, carrying the **highest** score both times, so a confidence cut would remove the real instances first).
Boxes that swallow most of the others' centres are dropped and counted in `group_boxes_dropped`; this is what
turns a 9 into the correct 8.

#### Measured accuracy

Scored by the fixture suite in [`benchmarks/`](benchmarks/), which draws its synthetic cases programmatically so
their instance count is exact by construction. Tuning the box threshold from the upstream default of 0.25 down
to 0.15, and letting the group-box filter apply at three detections rather than four:

| | default (0.25, floor 4) | shipped (0.15, floor 3) |
| --- | --- | --- |
| Positives exact | 6/10 | **9/10** |
| Negative controls held | 8/8 | **8/8** |
| Mean absolute count error | 1.07 | **0.10** |

Two cases drove that: a 30-object grid was undercounted at 15 and is now exact, and every two-instance count was
one too high, because the detector returns both instances plus one box spanning the pair and the filter's floor
excluded that three-box case.

0.15 is a floor, not an optimum. Thresholds of 0.125 and 0.10 score a *perfect* 10/10 on positives, and both
break the negative controls -- fragmenting a spotted ball into 3-4 objects and a rough-edged rod into 2. That is
the failure this project refuses to ship, so the shipped value is the lowest threshold at which every control
still holds. Pass `threshold` explicitly if your images have no such texture.

#### Limits

**Heavy overlap still defeats it.** Eight shapes overlapping by roughly two-thirds of their width count as 6, not
8 -- improved from 3 at the old threshold, but still short. A count below what you expect means *"could not
separate them"*, not a real tally.

**Some evidence is simply not in the picture — the honest negative result.** On `tests/sample.jpg`, a paper
flower whose petals overlap, *seven* independent approaches all return 1: Florence-2 grounding, Moondream2's
detect head, Grounding DINO, the `count_lobes` outline measurement, SAM2 in segment-everything mode (one mask for
the whole flower), tiled inference, and a CIELAB interior-colour analysis. Threshold tuning does not touch it
either — it stays at 1 at every box threshold down to 0.10, including values low enough to fragment the textured
controls.

Two measurements explain why, and together they close the case: the flower's silhouette has **solidity 0.984**,
so its outline is very nearly a smooth disc, and its **interior contrast is 0.87**, barely above a plain textured
blob's 0.52. There is no outline evidence and no colour evidence. Ask `query_image` for a count in that situation
and treat the answer as an estimate.

**Everything that raises recall breaks the texture controls.** Four approaches were built and rejected —
interior colour, tiled crops, visual exemplars, and lower thresholds — and each one turns a *single* ball covered
in high-contrast spots into many objects: 6 with 2×2 tiling, 15 with 3×3, 31 with an exemplar, and 3–4 at a box
threshold of 0.10. A spot, a stripe, a logo segment and a petal are indistinguishable to a detector; separating
them requires knowing what the object *is*, which is the calling model's job rather than a measurement. The
evidence for each rejection is reproducible under [`benchmarks/`](benchmarks/).

So: a real improvement for separated, touching, dense, two-instance and awkwardly-named cases, a partial one for
heavy overlap, and none where the evidence is neither in the outline nor the colour. Counting is not solved.

### dense_region_caption ➕

Caption every salient region of an image at once, with bounding boxes. Unlike `detect_objects`, it discovers the
objects itself rather than needing one named, which makes it the tool for inventorying an image you know nothing
about.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.

### query_image ➕

Ask a free-form question about an image (visual question answering). Backed by Moondream2.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **question**: A free-form question to ask about the image.

### batch_analyze_images ➕

Run one operation across many images in a single call — the batch form of `caption`, `ocr`, `detect_objects`,
`count_objects`, `dense_region_caption` and `query_image`. Costs one round trip instead of one per image, and loads each model
once for the whole run. Each image reports its own success or failure, so one bad file does not abort the batch;
results come back in the order given.

For a single image, call the named tool directly: its arguments are checked up front rather than depending on
`operation`.

#### Arguments:

- **srcs**: File paths or URLs of the images to process.
- **operation**: One of `caption`, `ocr`, `detect`, `count`, `dense_caption`, `query`.
- **question**: Required when the operation is `query`.
- **object_name**: Required when the operation is `detect` or `count`.

### spatial_relations ✦

Measures how named objects sit relative to one another: whether they touch, how far apart they are, how much of
one lies inside the other and how deeply, plus each object's elongation, straightness and end-to-end width
profile. Florence-2 locates the objects, SAM2 segments them, and the geometry is computed from the masks.

This answers questions a bounding box cannot. Two boxes overlap as soon as one object is merely *in front of*
another, so boxes cannot tell contact from occlusion; silhouettes can. It reports measurements rather than
verdicts — the caller decides what the numbers mean for the scene at hand.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **objects**: Names of the objects to locate and compare, e.g. `["hand", "sword", "shield"]`.

#### Returns:

Per object — `area`, `elongation`, `straightness` (`max_deviation` from a straight line, `kink` from a smooth
curve) and `width_profile` (`end_a_width`, `mid_width`, `end_b_width`, `end_symmetry`).

Per pair — `state` (`separate` / `touching` / `overlapping`), `gap` in pixels, `a_inside_b` and `b_inside_a`
area fractions, and `embed_depth`, how far the overlap reaches from the other object's boundary.

Worked examples, measured on real images:

| Situation | Signal |
| --- | --- |
| A sword that should be held, but floats free of the hand | `state: separate`, `gap: 54px` |
| A hand gripping a shield's rim | `a_inside_b: 15%`, `embed_depth: 3.3px` |
| A hand pushed through a shield | `a_inside_b: 52%`, `embed_depth: 4.2px` |
| A hand fused into the middle of a shield face | `a_inside_b: 97%`, `embed_depth: 12.6px` |
| A blade with a point at *both* ends | `end_symmetry: 0.94`, versus 0.75–0.83 for blades with one tip and a hilt |

The containment figures separate cleanly and in order. `end_symmetry` separates too, but by a narrower margin —
0.94 against 0.83 — so it is better read as evidence alongside the width numbers themselves than as a threshold
to trust on its own.

Note that `spatial_relations` is the only tool that loads SAM2, and it loads it on first use — a server that is
only ever asked for captions or OCR never pays for it.

#### Limits

Masks are decoded on a fixed 256×256 grid before being upscaled to the image, so detail finer than roughly
`max(image side) / 256` pixels is not resolved. The measurements are also only as good as the detection they
start from: `detect_objects` returns nothing useful for vague classes, and can label the same region two
different ways in an ambiguous pose, which the geometry then faithfully measures.

`straightness` reliably separates a straight rod from a curved one, but it does not distinguish a naturally
curved object from an unnaturally bent one — on two branch-like staffs it scored 0.057 and 0.065, too close to
threshold on. Treat it as a shape description, not a defect detector.

### score_aesthetics ➕

Rate how aesthetically pleasing an image looks, independent of its content. Backed by a CLIP ViT-L/14
backbone plus the LAION "improved aesthetic predictor" head, trained on human aesthetic ratings.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.

#### Returns:

One `{"score": float, "rating": str}` object per page/image. `score` is roughly on a 1-10 scale;
`rating` buckets it into `poor` / `average` / `excellent` for quick triage. Reflects visual qualities
like lighting, composition and clarity — not whether the subject matter is correct or matches a
prompt. A technically accurate but flatly-lit, cluttered photo can score low; a blurry but
beautifully lit one can score comparatively higher.

Note that `score_aesthetics` is the only tool besides `critique_composition` that loads the CLIP
backbone (~1.7GB), and it loads it on first use — a server that is only ever asked for captions or
OCR never pays for it.

> **Rates photography, not fine art.** Tested live on Hokusai's *The Great Wave off Kanagawa*: a
> world-famous masterpiece still scored only 5.83 ("average"). The predictor is trained on human
> ratings of photographs (LAION/SAC/AVA), so it has no calibrated sense of quality for paintings,
> illustrations, or other non-photographic content — treat scores on that kind of image as noise,
> not signal.

### critique_composition ✦

Critiques an image's composition: locates the main subject, checks its framing against the rule of
thirds, and — only for low-scoring images — asks Moondream2 to explain what specifically looks
unbalanced. Florence-2 locates the subject, a from-scratch geometry function measures its framing,
`score_aesthetics`'s CLIP/LAION predictor scores the shot, and Moondream2 is consulted only when the
score is low enough to need an explanation.

This is the second tool, alongside `spatial_relations`, that isn't just a model wired in: no single
model in the stack combines localization, framing, and a quality judgment into one answer.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **target_subject**: Name of the main subject, e.g. `"the dog"`. Optional — omit it to auto-detect
  the most prominent region via `dense_region_caption` (scored by area weighted toward the image
  center).
- **low_score_threshold**: Aesthetic score below which Moondream2 is asked to explain why. Defaults
  to `5.0`.

#### Returns:

`image_size`, `subject_box`, `aesthetics` (the same shape `score_aesthetics` returns), and `framing`:
`centroid`, `thirds_offset` (distance from the nearest rule-of-thirds gridline intersection, near 0
for a well-composed shot), `center_offset` (distance from dead-centre, for comparison), and
`nearest_gridpoint` — both offsets normalized by the frame's diagonal, so they are comparable across
image sizes. When the aesthetic score is below `low_score_threshold`, a `critique` field carries
Moondream2's plain-language explanation.

If no subject can be located (an empty or genuinely featureless image), returns a soft-failure shape
— `image_size`, `aesthetics`, and a `note` explaining why — rather than raising, the same convention
`spatial_relations` uses when none of its requested objects are found.

#### Limits

Inherits `score_aesthetics`'s CLIP backbone cost, its photography-only calibration (see above), and
`spatial_relations`'s detection caveats: a vague or wrong `target_subject` grounds onto whatever
Florence-2 finds closest, the same way an ambiguous class name misleads `detect_objects` — tested
live with `target_subject="face"` against a photo with no face in it, which still returned a box
spanning most of the frame instead of reporting nothing found.

Auto-detection (omitting `target_subject`) is only informative when the photo has one clear,
smaller foreground subject. Tested live: on a night street scene with a clear light source, it
picked a real, off-center subject and reported meaningfully off-center framing. On a busy
whole-frame composition (a woodblock print, a hand holding a passport), it picked a box spanning
nearly the entire image, which trivially reports as "centered" — not because the composition is
centered, but because a near-full-frame box always is. The rule-of-thirds check only considers the
primary subject's box; it does not account for secondary subjects, leading lines, or other
compositional techniques.

### process 🔼

Processes an image file with a custom prompt using the Florence-2 model. Useful for Florence-2
task tokens this server does not expose as their own tool.

#### Arguments:

- **src**: A file path or URL to the image file that needs to be processed.
- **prompt**: A custom prompt for the Florence-2 model.

## Options

- **--model**: The Florence-2 model used for captioning, OCR, detection and grounding.
- **--cache-model**: Keep the Florence-2 model loaded between requests instead of running each one in a fresh
  subprocess.
- **--moondream-model** / **--moondream-revision**: The Moondream2 model and revision backing `query_image`.
- **--sam2-model**: The SAM2 model backing `spatial_relations`. Defaults to `sam2.1-hiera-small`; measured on
  CPU, `tiny` is ~0.06s faster per call for a slightly worse mask, and `base-plus` roughly doubles inference
  time for a marginal gain.
- **--grounding-dino-model**: The Grounding DINO model backing `count_objects`. Defaults to
  `IDEA-Research/grounding-dino-tiny` (~690MB); `grounding-dino-base` roughly triples the download for a
  marginal gain on the short noun phrases this server sends.
- **--aesthetic-model**: The CLIP model backing `score_aesthetics` and `critique_composition`. Defaults to
  `openai/clip-vit-large-patch14`, the checkpoint the LAION aesthetic head was trained against — swapping this
  invalidates the head's weights, so change it only alongside a matching head.
- **--memory-mode**: How long each model stays in memory after its last use. See
  [Memory modes](#memory-modes) below.
- **--idle-timeout**: Deprecated alias for `--memory-mode` expressed in minutes; overrides it when given.
- **--device**: Torch device all models load onto, e.g. `cpu`, `cuda`, `cuda:1`, `mps`. Auto-detected (MPS,
  then CUDA, then CPU) when unset. Set this to pin the server to a specific accelerator, force CPU on a shared
  GPU box, or target a non-default GPU index — including a GPU-equipped cloud VM, since this is a plain local
  process with no separate cloud deployment path of its own.

## Memory modes

The models are large, so a server left running can hold several gigabytes. `--memory-mode` sets where you sit on
the trade between memory and speed. When installing the `.mcpb` bundle, the same setting is offered as the
**Memory mode** configuration field, so it can be chosen without touching a command line.

| Mode | Behaviour | Choose it when |
| --- | --- | --- |
| `aggressive` | Releases after **5 minutes** of inactivity. | Memory is tight, or you work in short scattered bursts. Lowest memory of the presets. |
| `standard` | Releases after **10 minutes** of inactivity. **Default.** | General use — repeat calls stay fast during a burst of work, and the memory comes back once the work stops. |
| `persistent` | Never releases; models stay loaded for the server's lifetime. | Throughput matters more than memory, e.g. a dedicated machine or a long batch job. Highest memory, fastest. |
| *a number* | Releases after that many minutes, e.g. `--memory-mode 30`. | You know your own working rhythm and want to match it. `0` is the same as `persistent`. |

Every mode reloads automatically on the next request, so none of them can lose work — only time. Releasing is
also per model: a session that only ever captions never loads SAM2 or the CLIP backbone in the first place, so
these settings govern what is held *after* use, not what gets loaded.

There is deliberately no "release after every call" mode. The idle timer is restarted by attribute lookup, so a
timeout short enough to approximate one would also fire mid-inference, while the caller still holds a live
reference and nothing is actually freed. Set a small number of minutes instead if you want memory back quickly.


## License

This application is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
