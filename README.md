# FusionVisionMCP: Multi-model Vision Server

[![GitHub License](https://img.shields.io/github/license/Whoawhen/FusionVisionMCP)](https://github.com/Whoawhen/FusionVisionMCP/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Application](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml/badge.svg)](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

🚧 **Work in progress** — see [README_DETAILED.md](README_DETAILED.md) for current limits and measured results before relying on this for anything important.

An MCP server that fuses five local, CPU-capable vision models — Florence-2, Moondream2, SAM2, Grounding DINO and
CLIP/LAION — behind eleven tools: OCR, captioning, object detection/grounding, instance counting, visual question
answering, spatial measurement (touch/gap/containment), and aesthetic scoring.

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
comparison mode. See [README_DETAILED.md](README_DETAILED.md) for what's verified against each case.

| Capability | How it's provided |
|---|---|
| OCR & document text | Florence-2's OCR head, or Moondream2 for stylized/logo text (see [routing notes](README_DETAILED.md)) |
| Captioning | Florence-2, whole-scene or per-region |
| Object detection / grounding | Florence-2's grounding head — boxes and center points for a named object |
| Instance counting | Grounding DINO — parallel object queries, not a sequential emission, so overlapping-but-separate instances don't collapse into one |
| Visual question answering | Moondream2 — free-form questions about image content |
| Spatial measurement | SAM2 masks plus a from-scratch geometry module — touch, gap, containment depth, shape |
| Aesthetic scoring | CLIP + a LAION-trained aesthetic head |
| Memory footprint | Configurable idle-release timers, per model, chosen at install time |
| Hardware | Runs on CPU; uses a GPU automatically if one is available |

## Tools

| Tool | Model(s) | Description |
|---|---|---|
| `ocr` | Florence-2 | Transcribe dense, printed, document-style text from an image or PDF. |
| `caption` | Florence-2 | Describe what an image shows as one detailed prose caption of the whole scene. `verify_text=true` also corrects close text misses against a verbatim OCR pass. |
| `detect_objects` | Florence-2 | Locate a named object, returning bounding boxes, center points and labels. |
| `dense_region_caption` | Florence-2 | Caption every salient region of an image at once, without naming objects first. |
| `query_image` | Moondream2 | Ask a free-form question about an image (visual question answering). `check_consistency=true` routes a low-confidence answer to the measurement that actually answers it, when one applies. |
| `count_objects` | Grounding DINO | Count how many instances of a named object an image contains. Use this, not `detect_objects`, for "how many" questions. On a collapse, adds an actionable outline estimate. |
| `spatial_relations` | Florence-2 + SAM2 | Measure contact, gaps, containment depth and shape between two named objects. |
| `score_aesthetics` | CLIP + LAION | Rate how aesthetically pleasing an image looks on a 1-10 scale. `compare_with` switches to a calibrated like-with-like comparison against a reference image. |
| `critique_composition` | Florence-2 + CLIP/LAION + Moondream2 | Check framing against the rule of thirds; for low-scoring shots, explain what looks off. Also supports `compare_with`. |
| `batch_analyze_images` | (routes to any tool above) | Run one operation across many images in a single call, isolating failures per image. |
| `process` | Florence-2 | Run a raw Florence-2 task token for tasks the named tools don't cover. |

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

---

## Installation

### Claude Desktop
1. Download the latest MCP bundle `fusion-vision-mcp.mcpb` from [Releases](https://github.com/Whoawhen/FusionVisionMCP/releases)
2. Open the downloaded `.mcpb` file, or drag it into Claude Desktop's Settings window
3. Pick a **Memory mode** (or leave it on *Standard* and change it later)

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
pip install -e .
```

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

Five models, each loaded on-demand and released on its own idle timer:

- **Florence-2** (Microsoft) — captioning, OCR, object detection/grounding, dense region captioning
- **Moondream2** (Vikhyat) — visual question answering
- **SAM2** (Meta) — segmentation masks, the basis for `spatial_relations`
- **Grounding DINO** (IDEA-Research) — open-vocabulary detection backing `count_objects`
- **CLIP + LAION aesthetic head** — aesthetic quality scoring

Runs on CPU by default; uses a GPU automatically if one is available. Because inference happens locally, no image
data leaves the machine, and the CPU/RAM budget it uses is generally idle capacity rather than resources
competing with a GPU-bound workload.

Fork of [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2), which provides three tools —
`ocr`, `caption`, `process` — against Florence-2 alone.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
