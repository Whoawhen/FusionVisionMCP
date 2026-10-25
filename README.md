# FusionVisionMCP: Multi-model Vision Server

[![GitHub License](https://img.shields.io/github/license/Whoawhen/FusionVisionMCP)](https://github.com/Whoawhen/FusionVisionMCP/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Application](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml/badge.svg)](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

🚧 **Work in progress** — see [README_DETAILED.md](README_DETAILED.md) for current limits and measured results before relying on this for anything important.

An MCP server that fuses local, CPU-capable vision models — Florence-2, Moondream2, LFM2.5-VL, SAM2, Grounding DINO, PP-OCRv6, EasyOCR, IQA, and CLIP/LAION — behind twelve tools: OCR, captioning, object detection/grounding, instance counting, visual question answering, multiple-choice classification, spatial measurement (touch/gap/containment), and aesthetic scoring (including technical quality and artistic judgment via Ollama), plus a `server_status` report on what it holds in memory.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="FusionVisionMCP-Dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="FusionVisionMCP-Light.jpg">
  <img alt="FusionVisionMCP" src="FusionVisionMCP-Light.jpg">
</picture>

---

> **Quick Navigation**
>
> [Why FusionVisionMCP?](#why-fusionvisionmcp) | [Tools](#tools) | [Installation](#installation) | [Memory Modes](#memory-modes) | [Architecture](#architecture)

---

## Why FusionVisionMCP?

I kept running computer-vision tasks through Claude's native (frontier-model) vision and burning through a 5-hour
usage allocation in about an hour — CV work is token-hungry in a way that's easy to underestimate until you watch
the budget disappear. Moving that work to my GPU wasn't an option either: the GPU was already committed to image
generation, and running vision inference alongside those testing loops would have contended for the same VRAM.

What was sitting idle was CPU and system RAM. FusionVisionMCP runs Florence-2, Moondream2, SAM2, Grounding DINO
and CLIP entirely on CPU, so routine vision work — reading text, describing an image, counting objects, checking
whether two things touch — no longer has to spend a frontier model's own multimodal tokens. Building it out also
meant finding the gaps against Claude's native vision one by one and closing them; it's now close to that
capability for most everyday CV tasks, and a few tools here (`spatial_relations`' touch/gap/containment
measurement, `count_objects`' parallel-query instance counting) aren't things Claude's native vision does at all.

The remaining gaps are structural, not tunable — a caption head that paraphrases text, a small VQA model that
can't be trusted for open-ended judgment, a detector with no notion of what separates a spot from a petal, an
aesthetic score trained on photographs alone. Neither v0.6.0 nor v0.7.0 closes those (they're not closable
locally). v0.6.0 added opt-in parameters to four tools that surface the failure instead of hiding it — a
cross-checkable text span next to a caption's guess, a consistency flag on a VQA answer, a second opinion plus a
separability flag on an ambiguous count, a medium classification alongside an aesthetic score. v0.7.0 turns each
of those flags into an actionable result by combining tools already in the project, no new models: the caption
gets an auto-corrected copy, a collapsed count gets an outline-based estimate, a low-confidence VQA answer routes
to the measurement that actually answers the question, and the aesthetic score gains a calibrated like-with-like
comparison mode. v0.7.1 fixed six further gaps that a second head-to-head pass against Claude's native vision
turned up (a partially-occluded object recovered at the detector's *existing* default threshold, a segmentation
occlusion hole no longer defeats containment, a false column split on sparse small text, clip-art counting, a
blank-canvas false detection, and a "which is largest/smallest" VQA question routed to a real measurement) — each
root-caused against the exact fixture that exposed it, not patched at the symptom. See
[README_DETAILED.md](README_DETAILED.md) for what's verified against each case.

| Capability | How it's provided |
|---|---|
| OCR & document text | PaddleOCR PP-OCRv6 English via RapidOCR (ONNX, CPU) — printed, stylized and low-contrast text through one tool, with a per-span confidence score. EasyOCR covers non-English languages and the generated-gibberish check |
| Captioning | LFM2.5-VL-3B (whole scene), Florence-2 (per region) |
| Object detection / grounding | SAM 3 on photos and flat images (it can say "not here") — scored boxes and center points for a named object |
| Instance counting | Grounding DINO — parallel object queries, not a sequential emission, so overlapping-but-separate instances don't collapse into one |
| Visual question answering | LFM2.5-VL-3B (ONNX, the default caption's model) — free-form questions about image content |
| Spatial measurement | SAM2 masks plus a from-scratch geometry module — touch, gap, containment depth, shape |
| Aesthetic scoring | CLIP + a LAION-trained aesthetic head |
| Memory footprint | Configurable idle-release timers, per model, chosen at install time |
| Hardware | CPU by default; an NVIDIA GPU is opt-in at install time via the `cu130` extra, and is then auto-detected at runtime |

## Tools

| Tool | Model(s) | Description |
|---|---|---|
| `ocr` | PP-OCRv6 English | Transcribe any text in an image or PDF — printed, stylized or low-contrast alike — with a per-span confidence score. Multi-column pages are split by geometry and read in reading order. Text-free images return nothing. |
| `caption` | LFM2.5-VL-3B (ONNX; `deep` in bf16) | Describe what an image shows as one detailed prose caption of the whole scene: key facts 71.0% vs 64.8% for Florence-2, its captioner before 2026-10-23, on 280 held-out images, at ~11 s vs ~7 s (`--caption-model florence2` keeps Florence-2). `verify_text=true` also corrects close text misses against a verbatim OCR pass; `deep=true` gives a longer, fuller description (~35 s; 74.8%, by saying more rather than more sharply). By default (`read_text`) the caption ends with `Visible text: "..."`, the `ocr` engine's verbatim reading of any text in the image (up to 400 words), and (`image_type`) starts with `Image type: ...`, CLIP's best guess at what kind of image it is (73.2% of key facts with it on the 280 images). `region=[x1, y1, x2, y2]` captions just that part at full resolution, for small text or HUD detail (named the answer 80% of the time vs 68% for the whole image). |
| `detect_objects` | SAM 3 (Florence-2 / Grounding DINO if it can't load) | Locate a named object, returning scored bounding boxes, center points and labels. On photos it rarely boxes an object that isn't there. |
| `dense_region_caption` | Florence-2 | Caption every salient region of an image at once, without naming objects first. |
| `query_image` | LFM2.5-VL-3B (ONNX) | Ask a free-form question about an image (visual question answering): 78.5% right vs 66.5% for Moondream2, its answerer before 2026-10-24, on 892 fresh questions, at ~5 s vs ~10 s, and far less prone to answering a false premise ("what does the sign say" with no sign). A question about text gets the `ocr` reading as context. `--vqa-model moondream` keeps Moondream2. `check_consistency=true` routes a low-confidence answer to the measurement that actually answers it, when one applies. |
| `classify_image` | CLIP | Pick which of your labels an image shows (species, landmark, art style, kind of image), with a probability for each. 98.5% vs `query_image`'s 77.5% on 200 fresh images, in 0.25 s. It only chooses among the labels given. |
| `count_objects` | Grounding DINO (photos), Florence-2 (flat images) | Count how many instances of a named object an image contains. Use this, not `detect_objects`, for "how many" questions. Flat images (clip art, vector charts, synthetic shapes) count less reliably: touching shapes can merge into one. On a collapse, adds an actionable outline estimate. On photos, `possibly_more: {upper, extra_candidates, reliability}` flags a count that may be too low, and on crowds `crowd_range: {low, high}` (Grounding DINO and SAM 3 together) holds the true count 86% of the time. |
| `spatial_relations` | Grounding DINO + SAM2 | Measure contact, gaps, containment depth and shape between two named objects. |
| `score_aesthetics` | CLIP + LAION | Rate how aesthetically pleasing an image looks on a 1-10 scale. `compare_with` switches to a calibrated like-with-like comparison against a reference image. |
| `critique_composition` | Florence-2 + CLIP/LAION + Moondream2 | Check framing against the rule of thirds; for low-scoring shots, explain what looks off. Also supports `compare_with`. |
| `batch_analyze_images` | (routes to any tool above) | Run one operation across many images in a single call, isolating failures per image. |
| `process` | Florence-2 | Run a raw Florence-2 task token for tasks the named tools don't cover. |
| `server_status` | (no model) | Report what the server holds in memory: models loaded, live free RAM, heavy calls running, and recent loads, releases and refusals with their reasons. For when a call was slow or refused. |

Full argument reference, measured accuracy, and known limits for each tool: [README_DETAILED.md](README_DETAILED.md).

### Example

Once connected, an assistant calls these tools on its own when a request needs them — there's nothing to invoke
by hand. For example, asking *"How many bolts are in this photo?"* with an image attached routes to
`count_objects(src=..., object_name="bolt")`, which returns:

```json
{
  "count": 6,
  "bboxes": [[102, 44, 138, 79], "... 5 more"],
  "scores": [0.91, "... 5 more"],
  "group_boxes_dropped": 0
}
```

### What's genuinely new here, not just a wrapper

Most tools above expose one underlying model's own capability directly. Two do not — no single model in the
stack answers these on its own:

- **`spatial_relations`** combines Florence-2 boxes, SAM2 masks, and a from-scratch geometry module to measure
  whether two objects actually touch, contain, or overlap — a bounding box alone can't answer that, since boxes
  overlap the instant one object is merely in front of another.
- **`critique_composition`** combines Florence-2 localization, a from-scratch rule-of-thirds check, the CLIP/LAION
  aesthetic score, and — only for low-scoring images — a Moondream2 explanation, into one composition critique.
  Since 2026-10-11 it also reports `background: {busy, reliability}` by default (`check_background`), Moondream2's
  yes/no on whether the background competes with the subject: a hint, agreeing with a person's blind calls on 69%
  of 52 held-out photos. Without a `target_subject`, Moondream2 now names the subject and Florence-2 boxes it
  (84.4% of boxes overlapping a labeller's on 96 fresh photos, against 75.0% for the old region pick).

---

## Installation

### Claude Desktop
1. Download the latest MCP bundle `fusion-vision-mcp.mcpb` from [Releases](https://github.com/Whoawhen/FusionVisionMCP/releases)
2. Open the downloaded `.mcpb` file, or drag it into Claude Desktop's Settings window
3. Pick a **Memory mode** (or leave it on *Standard* and change it later)
4. Leave **Compute device** on *cpu* unless you have an NVIDIA GPU — see [CPU or GPU](#cpu-or-gpu)

Also connectable from Cursor, Windsurf, VS Code, or any other MCP-compatible client via manual configuration below.

### Manual Installation

#### Prerequisites
- Python 3.12+
- Git
- 8GB+ RAM recommended

#### Setup
```bash
git clone https://github.com/Whoawhen/FusionVisionMCP.git
cd FusionVisionMCP
uv sync --extra cpu --extra ocr-specialist --extra iqa
```

Use `--extra cu130` in place of `--extra cpu` for an NVIDIA GPU. Exactly one of the two is
required — see [CPU or GPU](#cpu-or-gpu) for why, and why this needs `uv` rather than `pip`.

#### CPU or GPU

`torch` is not a plain dependency here. It sits behind two mutually exclusive extras, so the
build is chosen at install time:

| Extra | Installs | Use it when |
|---|---|---|
| `cpu` | `torch+cpu` from PyTorch's CPU index | The default. Works everywhere, and downloads ~2 GB less. |
| `cu130` | `torch+cu130` (CUDA 13) plus the CUDA runtime | You have an NVIDIA GPU and a driver supporting CUDA 13. |

**Naming neither installs no torch, and the server will not load a model.** That is the cost of
making the choice explicit; `uv` will not guess for you.

Why it is worth the friction: `torch` from the default index pulls 15 NVIDIA runtime packages
totalling **2.09 GB** on Linux — cuDNN alone is 527 MB — into what is otherwise a CPU-only
server. Windows and macOS never saw this, because torch gates those packages on
`sys_platform == 'linux'`, which is why the waste went unnoticed.

**This requires `uv`, not `pip`.** The index routing lives in `[tool.uv.sources]` and
`[tool.uv.index]`, which `pip` ignores entirely — a `pip install -e ".[cpu]"` resolves `torch`
from PyPI and silently defeats the whole arrangement.

**macOS** takes its usual CPU/MPS build from PyPI either way: PyTorch's CPU index publishes only
`manylinux_2_28_x86_64` and `win_amd64` wheels, so the `cpu` extra deliberately does not apply
there. Use `--extra cpu`; you will get the right thing.

`cu130` rather than `cu128`, for anyone wondering: the `cu128` index stops at torch 2.9.1 and
this project requires `torch>=2.13`. Verified end to end on an RTX 5060 Laptop (Blackwell,
`sm_120`) — the wheel ships real `sm_120` kernels, and fp16 matmul, cuDNN convolution and
scaled-dot-product attention all run correctly on it, which matters because every model wrapper
selects fp16 on a non-CPU device.

#### Configuration
```json
{
  "mcpServers": {
    "fusionvision": {
      "command": "uv",
      "args": ["run", "fusion-vision-mcp", "--memory-mode", "standard"]
    }
  }
}
```

Swap `standard` for `aggressive`, `persistent`, or any number of minutes.

Running several MCP clients at once (Claude Code with a few chats open, Cline, Claude Desktop)? Each starts its own
server and loads its own copy of the models. `--transport streamable-http` runs one server they all share instead:
see [docs/shared_server.md](docs/shared_server.md).

### System Requirements

- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 16GB free space for model weights (downloaded automatically on first use, then cached locally)
- **OS**: Windows 10+, macOS 12+, or Linux

---

## Memory Modes

Vision models are large. FusionVisionMCP lets you decide how long each one stays resident in memory after its
last use, picked at install time — no config file required:

| Mode | Models released | Best for |
|------|----------------|----------|
| **Aggressive** | After 5 minutes idle | Tight memory budgets, short bursts of work |
| **Standard** *(default)* | After 10 minutes idle | Everyday use — fast during work, tidy afterwards |
| **Persistent** | Never | Maximum speed on a dedicated machine |
| **Custom** | After *N* minutes you set | Matching your own working rhythm |

Models reload automatically on the next request, so no setting can lose work — only time. Release is per model:
a session that only captions never loads the segmentation or aesthetic models at all.

---

## Architecture

The main models, each loaded on demand and released on its own idle timer:

- **Florence-2** (Microsoft) — OCR with regions, object detection/grounding, dense region captioning (and `caption` with `--caption-model florence2`)
- **Moondream2** (Vikhyat) — `critique_composition`'s busy-background and subject hints, `count_objects`' estimate (and `query_image` with `--vqa-model moondream`)
- **LFM2.5-VL-3B** (Liquid AI) — `caption`: ONNX Runtime for the default caption, bf16 for `deep=true` (weights under the LFM Open License v1.0)
- **SAM2** (Meta) — segmentation masks, the basis for `spatial_relations`
- **Grounding DINO** (IDEA-Research) — open-vocabulary detection backing `count_objects`
- **CLIP + LAION aesthetic head** — aesthetic quality scoring

Runs on CPU by default. An NVIDIA GPU is opt-in at install time (the `cu130` extra) and is then detected and used automatically; `--device` pins it either way. Because inference happens locally, no image
data leaves the machine, and the CPU/RAM budget it uses is generally idle capacity rather than resources
competing with a GPU-bound workload.

Originally derived from [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2) (MIT), which
provides three tools — `ocr`, `caption`, `process` — against Florence-2 alone. This is no longer a fork: the
history was severed and the original `mcp_florence2` package replaced outright. Parts of `florence2.py` are
still that project's code, so its copyright notice travels with them — see [LICENSE](LICENSE).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
