# Model Variants, Community LoRAs, and Fine-Tunes for FusionVisionMCP

*Compiled: 2026-10-02*
*Context: Research and evaluation of community-built LoRAs, fine-tunes, and upgraded checkpoints to improve the stated roles of the models across the FusionVisionMCP pipeline.*

---

## 1. System Pipeline Architecture Baseline

| Model | Current Checkpoint | Stated Role in FusionVisionMCP |
| :--- | :--- | :--- |
| **Florence-2 large** | `florence-community/Florence-2-large` | `caption`, `dense_region_caption`, `process`, and `detect_objects` on flat or clip-art images. Its OCR head is used only by `caption(verify_text)`. |
| **Moondream2** | `vikhyatk/moondream2` (rev `2025-06-21`) | `query_image` (Visual question answering, visual reasoning, detail querying). |
| **Grounding DINO tiny** | `IDEA-Research/grounding-dino-tiny` | `count_objects`, `detect_objects` on photos, and the detection step in `spatial_relations` and `query_image(structured_analysis)`. |
| **SAM 2.1 small** | `facebook/sam2.1-hiera-small` | `spatial_relations` masks, and the silhouette check in `count_objects`. |
| **CLIP ViT-L/14** | `openai/clip-vit-large-patch14` | Backbone for `score_aesthetics`, plus the zero-shot medium label from `style_context`. |
| **LAION aesthetic predictor** | `sac+logos+ava1-l14-linearMSE.pth` | The 1–10 score on top of CLIP. |
| **MUSIQ (ONNX)** | `86Cao/IQA-ONNX-Models` | `technical_quality` in the aesthetics output; needs the `[iqa]` extra. |
| **PP-OCRv6 English** | via RapidOCR on ONNX Runtime | `ocr`, plus the crop re-reads in `caption(auto_verify_text)`. |
| **EasyOCR** | Default English reader | The gibberish check in `query_image(structured_analysis)`, and `ocr` when the server is started with non-English `--ocr-languages`. |

---

## 2. Research: Florence-2 Large LoRAs & Community Fine-Tunes

### Understanding "Florence-2 LoRA" in the Community
In the AI community, discussions around "Florence-2 LoRA" fall into two categories:
1. **LoRAs / Fine-Tuning Adapters *for* Florence-2-Large:** Vision-language models trained using PEFT/LoRA (or merged LoRAs) on top of Microsoft's `microsoft/Florence-2-large` to specialize it for generative prompt crafting, table parsing, document layout, or domain-specific captioning.
2. **Florence-2 *as a tool* to train Diffusion LoRAs (Civitai/ComfyUI):** In the Stable Diffusion / Flux community, Florence-2 is widely used inside ComfyUI and Kohya scripts as an automated natural-language captioner to build datasets for training image generation LoRAs.

### Top Community Models & LoRAs for Florence-2-Large

| Model / Adapter | Primary Focus | Type | Likes / Rating | Monthly Downloads |
| :--- | :--- | :--- | :---: | :---: |
| [**MiaoshouAI/Florence-2-large-PromptGen-v2.0**](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0) | Generative Image Prompting & Tagging | Merged FT / LoRA | **122** | **~9,000** |
| [**MiaoshouAI/Florence-2-large-PromptGen-v1.5**](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5) | Generative Image Prompting & Tagging | Merged FT / LoRA | **70** | **~700** |
| [**gokaygokay/Florence-2-Flux-Large**](https://huggingface.co/gokaygokay/Florence-2-Flux-Large) | FLUX.1 Dataset Captioning | Merged FT | **62** | **~1,800** |
| [**ucsahin/Florence-2-large-TableDetection**](https://huggingface.co/ucsahin/Florence-2-large-TableDetection) | Table Detection & Layout Extraction | Fine-Tuned | **19** | **~110** |
| [**yayayaaa/florence-2-large-ft-moredetailed**](https://huggingface.co/yayayaaa/florence-2-large-ft-moredetailed) | Ultra-Detailed Image Descriptions | Fine-Tuned | **16** | **~45** |
| [**ljnlonoljpiljm/florence-2-large-nsfw-pretrain**](https://huggingface.co/ljnlonoljpiljm/florence-2-large-nsfw-pretrain) | Uncensored & NSFW Content Captioning | Fine-Tuned | **13** | **~230** |
| [**F16/florence2-large-ft-gufeng_v3**](https://huggingface.co/F16/florence2-large-ft-gufeng_v3) | Classical Chinese / GuFeng Art Captioning | Fine-Tuned | **11** | **~70** |
| [**anushreeberlia/fashion-florence**](https://huggingface.co/anushreeberlia/fashion-florence) | Structured Fashion Tagging (JSON) | LoRA Adapter | **3** | **~250** |
| [**Mit1208/Florence-2-large-DocLayNet**](https://huggingface.co/Mit1208/Florence-2-large-DocLayNet) | Document Layout Analysis | Fine-Tuned | **3** | **~15** |
| [**Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate**](https://huggingface.co/Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate) | Aerial / Remote Sensing Rotated BBoxes | Fine-Tuned | **3** | **~35** |
| [**Mitchins/Florence-Chan-2.1**](https://huggingface.co/Mitchins/Florence-Chan-2.1) | Anime & Stylized Illustration Captioning | LoRA-SFT Merged | **1** | **~20** |

#### Detailed Model Profiles

1. **[MiaoshouAI/Florence-2-large-PromptGen-v2.0](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0)** (and **v1.5**)
   * **Description:** Community fine-tune explicitly trained to bridge vision descriptions and diffusion prompts (Midjourney/Flux/SDXL).
   * **Task Additions:**
     * `<GENERATE_TAGS>`: Comma-separated keyword tags.
     * `<MIXED_CAPTION>`: Blends natural language with descriptive tags.
     * `<ANALYZE>`: Deconstructs image composition, lighting, camera angles, and color palette.
   * **Impact:** 122 Likes, ~9,000 downloads/mo.
2. **[gokaygokay/Florence-2-Flux-Large](https://huggingface.co/gokaygokay/Florence-2-Flux-Large)**
   * **Description:** Fine-tuned on the `fluxdev_controlnet_16k` dataset to generate flow-of-thought descriptions in the exact syntax and density that FLUX.1 models respond best to.
   * **Impact:** 62 Likes, ~1,760 downloads/mo.
3. **[Mitchins/Florence-Chan-2.1](https://huggingface.co/Mitchins/Florence-Chan-2.1)**
   * **Description:** Derived from Florence-2-Large using LoRA + SFT and baked into standalone BF16 safetensors weights. Specialized for anime, illustrations, and stylized characters where base Florence-2 misidentifies flat boundaries or stylized anatomy.
4. **[anushreeberlia/fashion-florence](https://huggingface.co/anushreeberlia/fashion-florence)** (Standalone PEFT LoRA)
   * **Description:** Extracts structured fashion metadata and catalog attributes (garment type, fabric, neckline, pattern, sleeve length) formatted directly as JSON.
   * **Impact:** 3 Likes, ~250 downloads/mo.
5. **[ucsahin/Florence-2-large-TableDetection](https://huggingface.co/ucsahin/Florence-2-large-TableDetection)**
   * **Description:** Trained on PubTables-1M (1,500 curated dense samples) to detect, isolate, and predict bounding coordinates for tabular structures.
6. **[Mit1208/Florence-2-large-DocLayNet](https://huggingface.co/Mit1208/Florence-2-large-DocLayNet)**
   * **Description:** Fine-tuned on DocLayNet to identify multi-column layouts, figures, captions, section headers, and paragraph boundaries in documents.
7. **[NikshepShetty/Florence-2-pixelprose](https://huggingface.co/NikshepShetty/Florence-2-pixelprose)** & **[NikshepShetty/Florence-2-DOCCI-FT](https://huggingface.co/NikshepShetty/Florence-2-DOCCI-FT)**
   * **Description:** Official PEFT LoRA adapters trained on UMD PixelProse and Google DOCCI datasets for dense, hyper-accurate descriptive captions without conversational hallucination.

#### Loading Code References

##### Standalone / Merged Checkpoint (e.g. PromptGen):
```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_id = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
).to("cuda").eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

##### PEFT LoRA Adapter on Base Florence-2:
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch

base_model_id = "microsoft/Florence-2-large"
adapter_id = "anushreeberlia/fashion-florence"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
).to("cuda")

# Attach the LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_id).eval()
processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
```

---

## 3. Comprehensive Evaluation Across All FusionVisionMCP Pipeline Models

### 3.1 Florence-2 Large
* **Current Checkpoint:** `florence-community/Florence-2-large`
* **Stated Role:** `caption`, `dense_region_caption`, `process`, and `detect_objects` on flat or clip-art images; OCR head used only by `caption(verify_text)`.
* **Identified Bottleneck:** `florence-community/Florence-2-large` is the base pre-trained foundation model. It has not undergone Microsoft's downstream task fine-tuning on dense region captioning or OCR region grounding.
* **Recommended Upgrades:**
  1. **`microsoft/Florence-2-large-ft`** (Official downstream fine-tune; 393 likes, ~41k downloads/mo):
     * *Why & How:* Direct drop-in replacement. Specifically trained on `<DENSE_REGION_CAPTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`, and `<OCR_WITH_REGION>`. Immediately sharpens bounding coordinates and increases character recall on `caption(verify_text)`.
  2. **`MiaoshouAI/Florence-2-large-PromptGen-v2.0`** (Merged fine-tune; 122 likes, ~9k downloads/mo):
     * *Why & How:* Provides richer spatial composition and granular descriptions; useful if `process` or `caption` outputs need prompt-grade visual fidelity.
  3. **`Mitchins/Florence-Chan-2.1`** (Fine-tuned for 2D/Anime/Clip-art):
     * *Why & How:* Directly addresses `detect_objects` on flat or clip-art images. Standard Florence-2 struggles on 2D line art and vector graphics; this model resolves flat object boundaries accurately.

### 3.2 Moondream2
* **Current Checkpoint:** `vikhyatk/moondream2`, revision `2025-06-21`
* **Stated Role:** `query_image` (Visual question answering, visual reasoning, detail querying).
* **Identified Bottleneck:** As a compact ~1.86B parameter VLM, Moondream2 occasionally exhibits hallucination on fine-grained counting or spatial position questions.
* **Recommended Upgrades:**
  1. **`fal/moondream2-docci-instruct`** (Instruction fine-tune; 9 likes, ~30 downloads):
     * *Why & How:* Instruction-tuned on Google's DOCCI dataset. Focuses on strict grounding and visual question-answering, noticeably reducing hallucination on granular object positions and small details.
  2. **`gokaygokay/moondream-prompt`** (Visual fine-tune; 11 likes, ~110 downloads):
     * *Why & How:* Specialized in deconstructing visual attributes, improving depth of diagnostic answers.
  3. **`moondream/moondream-2b-2025-04-14` / Architectural evolutions**:
     * *Why & How:* Later official revisions provide improved coordinate pointing and visual reasoning calibration.

### 3.3 Grounding DINO Tiny
* **Current Checkpoint:** `IDEA-Research/grounding-dino-tiny`
* **Stated Role:** `count_objects`, `detect_objects` on photos, detection step in `spatial_relations` and `query_image(structured_analysis)`.
* **Identified Bottleneck:** Swin-T backbone has lower recall on small/clustered items and generates overlapping/envelope detections on touching instances.
* **Recommended Upgrades:**
  1. **`IDEA-Research/grounding-dino-base`** (Drop-in upgrade; 212 likes, ~1.53M downloads):
     * *Why & How:* Uses Swin-B backbone. Increases zero-shot detection mAP from 48.4 to 52.5 (+4.1 mAP). Tightens bounding boxes, resolves occluded items, and dramatically reduces false-positive group envelopes in `count_objects` and `spatial_relations`.
  2. **`openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det`** (19.2k downloads):
     * *Why & How:* Trained on Objects365 + GoldG + V3Det (13,000+ classes vocabulary). Expands open-vocabulary recognition for uncommon objects in `detect_objects`.
  3. **`onnx-community/grounding-dino-tiny-ONNX`** (7 likes, ~9.2k downloads):
     * *Why & How:* Drop-in ONNX Runtime export providing ~2x faster inference latency with identical detection precision.

### 3.4 SAM 2.1 Small
* **Current Checkpoint:** `facebook/sam2.1-hiera-small`
* **Stated Role:** `spatial_relations` masks, and silhouette check in `count_objects`.
* **Identified Bottleneck:** Fixed 256x256 internal mask decode resolution can yield blurry or broken contours on thin structures (stems, cables, limbs) and touching boundaries.
* **Recommended Upgrades:**
  1. **`facebook/sam2.1-hiera-base-plus`** (Scaling step; 145 likes, ~143k downloads):
     * *Why & How:* 80M parameters (vs 46M in small). Optimal Pareto balance of speed and mask sharpness; resolves thin contours, hollow cavities, and touching silhouettes with cleaner boundaries for silhouette overlap and IoU checks.
  2. **`diffusionstudio/sam2.1-tiny-video-onnx-fp16`** (ONNX acceleration):
     * *Why & How:* Cuts mask decoding latency from ~50ms to sub-10ms for fast prompt-to-mask queries.
  3. **HQ-SAM / Boundary-Refined Adapters:**
     * *Why & How:* Segment Anything in High Quality (HQ-SAM) architectures add boundary-preserving feature fusion, preventing pixelation when masks are upscaled to original image resolution.

### 3.5 CLIP ViT-L/14 & LAION Aesthetic Predictor
* **Current Checkpoints:** `openai/clip-vit-large-patch14` + `sac+logos+ava1-l14-linearMSE.pth`
* **Stated Role:** Backbone for `score_aesthetics`, plus zero-shot medium label from `style_context`.
* **Identified Bottlenecks:**
  1. Direct `.pth` download requires manual SHA256 caching and unpickling.
  2. Linear head is trained on AVA (traditional photography) and underrates digital art, anime, and illustrations.
  3. 224x224 input lacks resolution to detect brushwork, paper texture, or film grain in `style_context`.
* **Recommended Upgrades:**
  1. **`shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`** (10 likes, ~16.8k downloads):
     * *Why & How:* Modern Hugging Face Hub packaging of the LAION predictor v2 with `safetensors`. Eliminates raw `.pth` network fetching and integrates cleanly.
  2. **`openai/clip-vit-large-patch14-336`** (310 likes, ~2.2M downloads):
     * *Why & How:* 336x336 input resolution. For `style_context`, high-resolution features improve discernment of artistic media (e.g. watercolor vs gouache, pencil hatching vs digital vector, film grain vs sensor noise).
  3. **`Blackroot/Anime-Aesthetic-Predictor-Medium`**:
     * *Why & How:* Provides calibrated aesthetic scoring when `style_context` classifies an image as an illustration, anime, or 2D render.

### 3.6 MUSIQ (ONNX)
* **Current Checkpoint:** `86Cao/IQA-ONNX-Models` (file: `musiq_model.onnx`)
* **Stated Role:** `technical_quality` in the aesthetics output; needs the `[iqa]` extra.
* **Identified Bottleneck:** `musiq_model.onnx` in this repo is exported with a fixed 224x224 input graph. Downscaling high-resolution images compresses micro-blur and sensor noise, causing the model to evaluate downscaled composition rather than true native sharpness.
* **Recommended Alternatives (Available directly in `86Cao/IQA-ONNX-Models`):**
  1. **MANIQA (`maniqa_model.onnx` + `.data`)**:
     * *Why & How:* Winner of the CVPR NTIRE Image Quality Assessment challenge. Captures micro-blur, sensor noise, and fine compression artifacts significantly better than fixed-input MUSIQ. Drop-in file swap within the existing repo.
  2. **HyperIQA (`hyperiqa_model.onnx` + `.data`)**:
     * *Why & How:* Uses a self-adaptive hyper-network to decouple image semantics from technical quality, avoiding false low scores on dark or abstract compositions.
  3. **CLIPIQA (`clipiqa_model.onnx` + `.data`)**:
     * *Why & How:* Maps technical quality assessment directly to semantic CLIP prompts ("sharp photo", "blurry image").

### 3.7 PP-OCRv6 English (RapidOCR) & EasyOCR
* **Current Checkpoints:** RapidOCR with PP-OCRv6 English on ONNX Runtime; EasyOCR default English reader.
* **Stated Roles:**
  * PP-OCR: Primary `ocr` and crop verification in `caption(auto_verify_text)`.
  * EasyOCR: Gibberish detector in `query_image(structured_analysis)` and fallback for non-English languages.
* **Recommended Upgrades:**
  1. **PP-OCRv4 / v6 Server Recognition Models (`_server_rec`)**:
     * *Why & How:* The server weights offer a +2.5% accuracy gain over mobile weights on degraded, low-resolution, or angled crops during `caption(verify_text)`.
  2. **DBNet++ with Deformable Convolutions**:
     * *Why & How:* Improves text detection bounding polygons on curved, skewed, or perspective-distorted text in scene photos.
  3. **TrOCR (`microsoft/trocr-small-printed`)**:
     * *Why & How:* Vision Transformer OCR. Used as an optional secondary arbiter on low-confidence crops (<0.75), resolving digit and punctuation confusion (e.g., `$12.50` vs `812.50`).

---

## 4. Priority Action Matrix

| Priority | Component | Action | Expected Improvement |
| :---: | :--- | :--- | :--- |
| **1** | **Florence-2** | Switch `DEFAULT_FLORENCE2_MODEL` from `florence-community/Florence-2-large` to `microsoft/Florence-2-large-ft` | Zero-code upgrade improving dense region captions, bounding coordinates, and OCR verification. |
| **2** | **Grounding DINO** | Switch `DEFAULT_GROUNDING_DINO_MODEL` to `IDEA-Research/grounding-dino-base` | +4.1 mAP zero-shot detection, fewer envelope false-positives, higher counting accuracy. |
| **3** | **Technical IQA** | Swap `musiq_model.onnx` for `maniqa_model.onnx` in `image_quality.py` | SOTA sensitivity to fine blur, noise, and compression degradation using the same ONNX repository. |
| **4** | **Aesthetic Predictor** | Migrate `aesthetic.py` to `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE` | SafeTensors format with Hugging Face Hub caching, removing raw `.pth` pickle download risks. |

### Status of the matrix (2026-10-11)

| Priority | Status |
| :---: | :--- |
| **1** | Measured, not shipped: `Florence-2-large-ft` scored 36.9% caption keyword recall vs 55.9% on the tune split, with 38-word captions (VisionBench `caption_ab.py`). Community Florence-2 fine-tunes (PromptGen v2, moredetailed, Flux, Chan) were also tested in caption sprint C2; none shipped. |
| **2** | Ruled out: Grounding DINO is Chinese-developed (IDEA Research); no tuning on it while it awaits replacement (user rule, 2026-10-08). |
| **3** | Untested. Check MANIQA's origin first (believed to come from a Chinese university; not verified). |
| **4** | Untested. A file-format and safety change for the same model, not a quality change. |

---

## 5. Further ideas (Claude, 2026-10-11)

Added at the user's request. Ranked by which 🔻 row of the scoreboard (`frontier_capability_gaps.md`) each could
close. No Chinese-developed models. Origins and licences below are from memory and need the usual gate 0 paper
check before any run.

| Rank | Idea | Target gap | Why | Cost and risks |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **PaliGemma 2 mix 3B (Google) for `query_image`** | `query_image` −9 to −13 | Its ~12-word answers lost it the caption bake-off (C2) but suit VQA. Trained on VQA mixtures that include TextVQA and DocVQA, and sprint Q1 found Moondream2's limit is reasoning over the text it can see, not reading it | bf16 fits memory (6.7 GB, C2 gate 1). One tune run, text questions first. Gemma Terms of Use (already accepted for C2). Would sit alongside Moondream2, not replace it, unless it wins on all question types |
| **2** | **Gemma 3 4B (Google) through llama.cpp, 4-bit** for `caption(deep)` and `query_image` | caption −9 (`deep`); `query_image` | One stronger model for both rows that the scoreboard says need one. C2 ruled it out on speed in transformers (66–79 s), not quality | Google's own QAT 4-bit GGUF, ~3 GB. New runtime dependency (`llama-cpp-python`). Already in the caption C4 plan as C4-3b, gate 1 = 20-image timing |
| **3** | **docTR (Mindee, France, Apache-2.0) as the PP-OCRv6 replacement, then fine-tune it** on scene-text photos | `ocr` on photos −17 to −22 | Already the preferred replacement. Ships training scripts, so a fine-tune is possible without tuning a Chinese-developed model | Gates as listed under "Replacing Chinese-developed models": phase 14 rendered pages, TextVQA photos, empty on text-free images. Fine-tune data licences to check |
| **4** | **OWLv2 (Google) to replace Grounding DINO**; PaliGemma 2 `detect` as a second candidate | `detect_objects` absent objects (boxed 70% vs 10–20%); `count_objects` | On the replacement list. If idea 1 ships, PaliGemma 2 is already loaded | Envelope filter, adaptive threshold and `spatial_relations` best-match rule all need re-measuring on the new detector |
| **5** | **Florence-2-large caption LoRA** (trained) | caption −14 (default slot) | Targets the weakest kinds (game UI, AI images, illustrations, charts) with long, dense captions | Caption C4-4: highest build cost, the only idea needing the GPU (guarded local training, user decision 2026-10-11). Adapter kept unmerged so the other Florence-2 tools don't change |

Lower priority:
- **Fine-tuning DINOv2 for the AI-defect probe** (instead of a linear probe on frozen features): more labels already
  made the probe worse on unseen images, and ~1,000 crops are likely to be memorised. DINOv2-base (idea 11 on the
  scoreboard) comes first.
- **`critique_composition`:** about −7 points; not worth a model change yet.
- **Retire `caption(auto_verify_text)`** (not a model idea): `read_text` now puts the verbatim text in every
  caption, and the correction step costs about a point instead of helping.

Suggested order: idea 1 after caption sprint C4-1 (one tune run, already passes memory), then idea 2's timing gate
(a pass helps two tools).
