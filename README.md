# FusionVisionMCP: Multi-core Vision Server

[![GitHub License](https://img.shields.io/github/license/Whoawhen/FusionVisionMCP)](https://github.com/Whoawhen/FusionVisionMCP/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Application](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml/badge.svg)](https://github.com/Whoawhen/FusionVisionMCP/actions/workflows/python-app.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**One comprehensive package - 10 cutting-edge computer vision tools**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="FusionVisionMCP-Dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="FusionVisionMCP-Light.png">
  <img alt="FusionVisionMCP" src="FusionVisionMCP-Light.png">
</picture>

While other vision tools offer basic OCR or simple captioning, FusionVisionMCP delivers comprehensive visual intelligence built from industry leading solutions — all in a single, easy-to-use package.

---

> **Quick Navigation**
>
> [Get Started](#get-started-in-seconds) | [Powerful Tools](#powerful-computer-vision-tools) | [Memory Modes](#memory-modes-you-choose-the-trade-off) | [Real-World Examples](#real-world-examples) | [Installation](#installation)

---

## Get Started in Seconds

**Step 1: Download the latest release**

Download the latest MCP bundle `fusion-vision-mcp.mcpb` from our [Releases](https://github.com/Whoawhen/FusionVisionMCP/releases) page.

**Step 2: Install with one click**

Open the downloaded `.mcpb` file or drag it into Claude Desktop's Settings window.

That's it! FusionVisionMCP is now available in your AI assistant.

### Works With Your Favorite AI Tools

- **Claude Desktop** - Native integration with one-click setup
- **Cursor / Windsurf / VS Code** - Connect via MCP configuration
- **Any MCP-compatible client** - Universal compatibility

---

## Powerful Computer Vision Tools

Ten specialized tools, each with one clear job — no overlapping duties for your AI to guess between:

### 🔍 Text Extraction & Understanding
Extract text from any image, document, or screenshot with industry-leading accuracy.

### 📝 Intelligent Image Captioning
Generate rich, contextual descriptions that capture the essence of complex visuals.

### 🎯 Object Detection & Pointing
Locate specific objects with precise bounding boxes *and* center coordinates in one call.

### ❓ Visual Question Answering
Ask open-ended questions about images and get detailed, accurate answers.

### 🧭 Spatial Relationship Analysis
Understand how objects relate physically—whether they touch, contain, or overlap each other.

### 📊 Dense Region Captioning
Automatically caption every important region in complex images.

### ⚡ Batch Processing
Process multiple images simultaneously for large-scale analysis.

### 🔧 Custom Prompt Processing
Run specialized Florence-2 prompts for unique use cases.

### ⭐ Aesthetic Quality Scoring
Rate how visually pleasing an image is, independent of what it depicts.

### 🖼️ Composition Critique
Check framing against the rule of thirds and get a plain-language explanation for low-scoring shots.

---

## Memory Modes: You Choose the Trade-Off

Vision models are large. FusionVisionMCP lets you decide how long they stay in memory after use — picked right
at install time, no config file required:

| Mode | Models released | Best for |
|------|----------------|----------|
| **Instant** | Immediately after every call | Tight memory budgets, occasional vision work |
| **Aggressive** | After 5 minutes idle | Short bursts of work, memory back quickly |
| **Standard** *(default)* | After 10 minutes idle | Everyday use — fast during work, tidy afterwards |
| **Persistent** | Never | Maximum speed on a dedicated machine |
| **Custom** | After *N* minutes you set | Matching your own working rhythm |

Models reload automatically on the next request, so no setting can ever lose work — only time. And releasing is
per model: a session that only captions never loads the segmentation or aesthetic models at all.

## Installation

### Claude Desktop (Recommended)
1. Download the latest MCP bundle `fusion-vision-mcp.mcpb` from [Releases](https://github.com/Whoawhen/FusionVisionMCP/releases)
2. Open the downloaded `.mcpb` file or drag it into Claude Desktop's Settings window
3. Pick a **Memory mode** — or leave it on *Standard* and change it later

### Manual Installation
For advanced users or custom setups:

#### Prerequisites
- Python 3.10+
- Git
- 8GB+ RAM recommended

#### Setup Steps
```bash
git clone https://github.com/Whoawhen/FusionVisionMCP.git
cd FusionVisionMCP
pip install -e .
```

#### Configuration
Add to your MCP client configuration:
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

Swap `standard` for `instant`, `aggressive`, `persistent`, or any number of minutes.

---

## System Requirements

- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 15GB free space for model weights
- **OS**: Windows 10+, macOS 12+, or Linux
- **Internet**: Required for initial model download

Models are automatically downloaded on first use and cached locally for offline operation.

---

## Real-World Examples

### Document Processing
Turn receipts, contracts, and forms into searchable text instantly.

### Product Analysis
Examine product photos to extract specifications, compare features, and answer questions.

### Technical Diagrams
Understand flowcharts, schematics, and architectural diagrams by asking specific questions.

### Quality Control
Verify that components are properly assembled by analyzing spatial relationships.

### Educational Content
Extract equations, diagrams, and illustrations from textbooks and academic papers.

### Social Media Analysis
Process screenshots and memes to understand context and sentiment.

---

## Why FusionVisionMCP?



| Feature | Basic Vision Tools | FusionVisionMCP |
|---------|-------------------|------------------|
| Image Understanding | ✅ Basic OCR & Captioning | ✅ Advanced Spatial Analysis |
| Object Detection | ❌ | ✅ Precise Bounding Boxes |
| Visual Question Answering | ❌ | ✅ Open-Ended Insights |
| Spatial Reasoning | ❌ | ✅ Touch, Containment, Distance |
| Memory Efficiency | ❌ | ✅ Five Selectable Memory Modes |
| Multi-Model Integration | ❌ | ✅ Florence-2, Moondream2, SAM2, CLIP/LAION |
| Hardware Flexibility | Limited | ✅ CPU/GPU Adaptive Processing |
| Resource Optimization | ❌ | ✅ GPU Conservation for Primary AI Tasks |
| Aesthetic Quality Scoring | ❌ | ✅ CLIP-Based LAION Predictor |

## Tool Composition

| Tool Name | Provider | Core Functions | Unique Advantages |
|-----------|----------|----------------|-------------------|
| **Florence-2** | Microsoft (Original) | OCR, captioning, custom prompting, object detection & grounding, dense region captioning | Fast, efficient multi-task vision model |
| **Moondream2** | Vikhyat | Visual question answering | Specialized for open-ended VQA |
| **SAM2** | Meta (Original) | Segmentation masks | Precise pixel-level object segmentation |
| **CLIP + LAION aesthetic head** | OpenAI / LAION | Aesthetic quality scoring | Trained specifically on human aesthetic ratings |
| **FusionVisionMCP** | Whoawhen | `spatial_relations`, `critique_composition` | Combines the four models above into measurements none of them reports alone |

### FusionVisionMCP's Novel Functions

Most of FusionVisionMCP's tools wrap a capability one of its four underlying models already has: `detect_objects`
and `dense_region_caption` are Florence-2 task heads, `query_image` is Moondream2's own VQA, and
`score_aesthetics` is the CLIP/LAION predictor's own output. The two genuinely new capabilities, not provided by
any single model in the stack, are:

- **`spatial_relations`** - Measures how objects relate spatially (touch, containment, distance, shape) by combining
  Florence-2 boxes, SAM2 masks, and a from-scratch geometry module. No model here answers "does this actually touch
  that" on its own — see [README_DETAILED.md](README_DETAILED.md#spatial_relations-) for how it's built and its
  measured limits.
- **`critique_composition`** - Locates the main subject (Florence-2), checks its framing against the rule of thirds
  (a from-scratch geometry function), scores the shot's aesthetic quality (the CLIP/LAION predictor), and — only for
  low-scoring images — asks Moondream2 to explain what looks off. No single model in the stack combines localization,
  framing, and a quality judgment into one answer.

### Need More Technical Details?

See our [complete technical documentation](README_DETAILED.md) for full API specifications, tool arguments, and advanced configuration options.

---

## Technical Architecture

FusionVisionMCP integrates four state-of-the-art computer vision models into one MCP server:

- **Microsoft Florence-2**: Foundation model for captioning, OCR, and object detection
- **Moondream2**: Specialized for open-ended visual question answering
- **SAM2 (Segment Anything Model 2)**: Advanced segmentation for spatial reasoning
- **CLIP + LAION aesthetic predictor**: Rates how aesthetically pleasing an image looks

Each model loads on-demand and unloads automatically to conserve memory, ensuring optimal performance.

Fork of [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2), which provides exactly three tools —
`ocr`, `caption`, `process` — all against Florence-2.

---

## Hardware Flexibility & Resource Optimization

FusionVisionMCP is designed to run efficiently across multiple hardware configurations:

### Multi-Hardware Support
- **CPU-Only Systems** - Optimized to run on capable CPUs with sufficient system memory
- **GPU-Accelerated Systems** - Leverages GPUs for faster processing when available
- **Hybrid Configurations** - Intelligently distributes workload based on system capabilities

### Resource Optimization Benefits
- **GPU Conservation** - Offloads token-intensive computer vision tasks to local processing, freeing up valuable GPU resources for primary AI workloads
- **Scalable Performance** - Adapts to available hardware without requiring dedicated high-end GPUs
- **Memory Management** - Automatic model loading/unloading conserves system resources during inactive periods
- **Cost-Effective Deployment** - Reduces dependency on expensive cloud GPU instances for routine vision tasks

FusionVisionMCP enhances your AI workflow without competing for critical computational resources.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
