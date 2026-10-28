"""Model-choice constants and the `CaptionLevel` enum, kept free of torch/transformers.

Split out of the model wrapper modules (`aesthetic.py`, `florence2.py`,
`grounding_dino.py`, `moondream.py`, `sam2.py`) so that `fusion_vision_mcp/__init__.py`
can use these as function-signature defaults -- evaluated at package-import time --
without importing torch and transformers just to answer an MCP `initialize` request.
Each wrapper module still imports its constant(s) from here and re-exports them, so
`from fusion_vision_mcp.sam2 import DEFAULT_SAM2_MODEL` (etc.) keeps working unchanged.
"""

from enum import StrEnum
from typing import Final


class CaptionLevel(StrEnum):
    NORMAL = "<CAPTION>"
    DETAILED = "<DETAILED_CAPTION>"
    MORE_DETAILED = "<MORE_DETAILED_CAPTION>"


#: The aesthetic head below was trained on embeddings from this exact CLIP checkpoint;
#: swapping the backbone silently invalidates the head's weights.
DEFAULT_AESTHETIC_MODEL: Final[str] = "openai/clip-vit-large-patch14"

#: `tiny` (~690MB) is the default for the same reason `sam2.1-hiera-small` is: it is
#: the smallest checkpoint that does the job, and the larger `base` roughly triples the
#: download for a marginal gain on the kind of query this server sends.
DEFAULT_GROUNDING_DINO_MODEL: Final[str] = "IDEA-Research/grounding-dino-tiny"

#: Box confidence below which a detection is dropped. Lives here rather than in
#: `grounding_dino.py` (which re-exports it) because `__init__.py` also needs it as a
#: `Protocol` method default, evaluated at package-import time.
#:
#: 0.15 rather than the upstream default of 0.25, chosen by sweeping the whole counting
#: fixture set (`benchmarks/`): it takes exact positives from 8/10 to 9/10 and mean
#: absolute count error from 1.15 to 0.10 while still holding all eight negative
#: controls. Dropping further scores better on positives alone -- 0.125 and 0.10 both
#: reach 10/10 -- but each breaks a negative control, fragmenting a spotted ball into
#: 3-4 objects and a rough-edged rod into 2. That is the failure this project refuses to
#: ship, so 0.15 is the floor: the lowest threshold at which every control still holds.
DEFAULT_BOX_THRESHOLD: Final[float] = 0.15

#: `count_objects`' Grounding DINO score floor on photos (and the batch `count` operation).
#: 0.15 above was chosen on the synthetic fixtures alone, and on real photos it overcounts.
#: VisionBench sprint R (`replacement_gd_tuned.py`, `replacement_gate2.py`, 2026-10-12)
#: chose 0.425 on the tune half, by the most exact counts that still hold all eight negative
#: controls. On the held-out half it took exact counts on real photos from 43.8% to 61.8%
#: (+18.0, 95% CI 14.1 to 22.2). Every control held under both the "petal" and the
#: literal-name prompts; fixture positives went 8/10 -> 7/10. The adaptive threshold
#: still runs on top. Other Grounding DINO callers (spatial relations, structured
#: analysis's part crops) keep 0.15.
COUNT_BOX_THRESHOLD: Final[float] = 0.425

#: `count_objects`' `possibly_more`: Grounding DINO boxes from this floor up to the count threshold
#: that sit neither on (IoU) nor inside (share of their area) a counted box or an already-accepted
#: one, at `COUNT_RANGE_VETO` or more. VisionBench gate C1 (`count_gate_c1.py` option 1, 2026-10-14):
#: the truth lay in [count, count + extras] on 78.6-79.4% of three photo-count sets against 55.6-61.8%
#: exact, and no negative control got an extra. Both values came with the proposal, not from a fit.
COUNT_RANGE_FLOOR: Final[float] = 0.18
COUNT_RANGE_VETO: Final[float] = 0.5

#: `count_objects`' `crowd_range` on photos: Grounding DINO's count runs low on crowds and SAM 3's, at this floor (with
#: `detect_objects`' NMS and envelope filter), runs high, so `[min, max]` of the two brackets the truth; shown when the
#: higher is at least `CROWD_RANGE_SHOW`. VisionBench gate K (`count_range_gate.py`, 2026-10-16, round 17: fresh Open
#: Images photos of 4-30 objects): truth inside 85.9% vs `possibly_more`'s 65.2% (CI +12.0 to +29.3), median width 0.61
#: of the count, shown on 17.5% of counts of 1-3. The 0.3 came from sprint R's tune half; change it, re-run the gate.
CROWD_RANGE_SAM3_FLOOR: Final[float] = 0.3
CROWD_RANGE_SHOW: Final[int] = 4
#: SAM 3 runs for `crowd_range` only when `possibly_more.upper` is at least this (user's choice, 2026-10-27), saving
#: ~7 s on most counts. Replayed (`count_plan_review_check.py`): round 17's crowds held the truth 85.9% -> 80.4%, and
#: 428 of phase 11's 644 photo counts skip SAM 3; picked from four rules on seen sets, so round 26 re-checks it.
CROWD_RANGE_RUN_UPPER: Final[int] = 3
#: `count_objects`' presence checks on photos (`analysis.lfm_count_check`, counting plan Gates 1b/1c/2, 2026-10-28), asked
#: of the LFM2.5-VL-3B ONNX answerer with VQA_SEED. A count of 1 whose box scores under COUNT_CHECK_WEAK becomes 0 when
#: the presence question is answered "no" (V1w); a count of 0 with a distinct candidate box under the threshold becomes
#: 1 when the count question gives 1 or more *and* the presence question "yes" (P1). VisionBench `count_gate_lfm_veto.py`
#: and `count_gate_lfm_presence.py` on seen sets (counts 61.2 -> 70.3%, absent false non-zero 34.6 -> 17.5%, controls
#: 16/16), then `count_gate2_round26.py` on fresh round 26: counts of 1-5 exact 63.9 -> 68.1% (CI 0.8 to 8.4), hard
#: negatives counted non-zero 74.5 -> 69.1%. The prompts, 0.6 and the parse rules were fixed before those runs.
COUNT_CHECK_PRESENCE: Final[str] = "Is there a {name} in this image? Answer yes or no."
COUNT_CHECK_COUNT: Final[str] = "How many {name} are in this image? Answer with a single number."
COUNT_CHECK_WEAK: Final[float] = 0.6

#: `caption(region=...)`: the caller's box is widened by this share of its width and height on each side, to at
#: least `REGION_MIN_SIDE` px each way, then clipped to the image. Exactly the crop VisionBench gate R measured
#: (`region_inspect_gate.py`, 2026-10-15): on 528 questions about a worker-chosen region, the default caption of the
#: crop named the answer 80.3% of the time against 67.8% for the whole image (+12.5, 95% CI 8.3 to 16.9).
REGION_PAD: Final[float] = 0.15
REGION_MIN_SIDE: Final[int] = 32

#: `classify_image`: each caller label is put through this template and CLIP ranks them by softmax of
#: `logit_scale` x cosine similarity. The template is gate Z's tune choice (`classify_gate.py`, 2026-10-15), and gate
#: Z2 (`classify_gate2.py`, 2026-10-16) measured exactly it: top-1 98.5% vs `query_image`'s 77.5% on 200 fresh
#: species, landmark and style images (CI +15.5 to +27.0). Changing it means re-running the gate on fresh images.
CLASSIFY_TEMPLATE: Final[str] = "a photo of {}"
#: More labels than this is a vocabulary search rather than a choice, and the text batch grows with it.
CLASSIFY_MAX_LABELS: Final[int] = 200

#: `detect_objects`' Grounding DINO score floor, deliberately higher than counting's
#: 0.15: locating a named object has to be able to say "not here". Swept on 143 COCO
#: scenes (VisionBench detect_backend_ab.py), querying each image once for its real
#: category and once for a category COCO says is absent. At 0.15 boxes came back for the
#: absent category on 31% of images; at 0.35 on 2.8%, while still matching Florence-2's
#: recall on scenes of 3 or fewer (66.4% vs 65.7%) and beating it on crowded ones (55.1%
#: vs 35.6%). Florence-2's grounding head boxed the absent category on every image.
#: Precision there is 62.7% against Florence-2's 83.7%: more stray boxes, but scored.
DETECT_BOX_THRESHOLD: Final[float] = 0.35

#: SAM 3 backs `detect_objects` on photographs (`sam3.py`); Grounding DINO at the 0.35 above
#: remains the fallback when SAM 3 cannot be loaded (it is gated on Hugging Face). 0.125 is the
#: threshold VisionBench sprint R chose on the tune half (`replacement_gate1.py`): the best mean of
#: COCO recall, phase 11 recall and 100 minus the plausible-neighbour boxing rate, with absent COCO
#: categories boxed on <= 5% of scenes. Confirmed on the held-out half (`replacement_gate2.py`).
LOCATE_BOX_THRESHOLD: Final[float] = 0.125
DEFAULT_SAM3_MODEL: Final[str] = "facebook/sam3"

DEFAULT_MOONDREAM_MODEL: Final[str] = "vikhyatk/moondream2"
#: Pinned rather than tracking `main`, since this repository ships its model code via
#: `trust_remote_code` and reshapes it between revisions. `2025-06-21` is chosen over the
#: earlier `2025-01-09` pin because it fixed a measured self-consistency failure: asked to
#: count the petals in `tests/sample.jpg` the old pin answered 12 and then listed 6 colours
#: for them, while this one answers 10 and lists exactly 10.
DEFAULT_MOONDREAM_REVISION: Final[str] = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"

#: The `-large` checkpoint is the shipped default: it backs caption, OCR-with-region,
#: grounding and dense-region-caption. `-base` is used only by the CI server test, where
#: load time matters more than quality. Keep Dockerfile and benchmarks on this value, or
#: they measure/cache a model that never actually ships.
DEFAULT_FLORENCE2_MODEL: Final[str] = "florence-community/Florence-2-large"

#: Measured on CPU over tiny/small/base-plus: `small` costs ~0.06s more per call
#: and ~34MB more than `tiny` for a better mask, while `base-plus` roughly doubles
#: inference time for a marginal gain. `small` is the middle that earns its keep.
DEFAULT_SAM2_MODEL: Final[str] = "facebook/sam2.1-hiera-small"

#: Spans PP-OCRv6 reads below this confidence are dropped. Unfiltered, it read a lone
#: "S" (0.52) on a text-free photo and "0" (0.57) on a shapes fixture, and turned the last
#: line of a small-text paragraph into "chee o ee s s ns s eg ge." at 0.70. On 200 TextVQA
#: photos, recall is unchanged from no floor up to 0.70 and drops only one item at 0.80, so
#: 0.75 clears all three with margin rather than sitting exactly on the garbage line.
#: Calibrated for PP-OCRv6 only: EasyOCR's scores run lower and mean something different.
OCR_MIN_CONFIDENCE: Final[float] = 0.75

#: `caption(read_text=true)` appends at most this many words of the `ocr` engine's text.
#: Fixed before measuring (VisionBench `caption_text_grounding.py`, 2026-10-10); on 280
#: held-out images it lifted the default caption's keyword recall 53.3% -> 60.9%.
#: Raised 40 -> 80 on 2026-10-18 (C4-2b stage 1b, `caption_text_cap_r18.py`, rule fixed
#: before the images existed): on 136 fresh text-heavy images key phrases 46.8% -> 57.5%
#: (default) and 53.9% -> 63.6% (`deep`), no image worse, no absent object named; COCO
#: CHAIR unchanged and round 11's line +6.6 points with no absent object added.
#: Raised 80 -> 160 on 2026-10-20 (C4-7, `caption_text_line_r20.py`, rule fixed before the
#: images existed), together with the clean-up and cut marker in `visible_text_line`: on 150
#: fresh text-heavy images key phrases 48.1% -> 60.0% (default, CI +9.1 to +14.9) and
#: 54.3% -> 63.0% (`deep`), no image worse, no absent object added; COCO CHAIR unchanged.
#: Raised 160 -> 400 on 2026-10-24 (`caption_text_cap_lfm.py`, rule fixed before the run, the user chose 400): with the
#: LFM ONNX default's prose, round 20's key phrases 62.5% -> 65.7 (240), 67.2 (320, CI +3.1 to +6.7), 67.8 (400), 68.5%
#: (uncapped); absent objects and COCO CHAIR unchanged. The median line stays ~117 words on text-heavy pages (10 on a
#: general mix); 400 bounds the dense ones (uncapped reached 1,985 words) and cuts 7% of text-heavy pages.
CAPTION_TEXT_WORDS: Final[int] = 400

#: `caption(deep=true)` stops Moondream2's long caption once it loops: when LOOP_RUN consecutive
#: LOOP_NGRAM-word spans all repeat earlier text, the caption is cut where that run began. The
#: sampled long caption loops on ~7% of images (hundreds of "effects, effects, ..." up to the
#: 768-token cap, 97-170 s). Replayed on cached tune captions (VisionBench `caption_c4_0.py`,
#: 2026-10-12) it cut 14 of 120 and cost 0.2 points of keyword recall (one group, hit only
#: inside a loop).
LOOP_NGRAM: Final[int] = 8
LOOP_RUN: Final[int] = 16

#: `caption(deep=true)` is LiquidAI's LFM2.5-VL-3B since 2026-10-22 (caption plan Track 4, VisionBench
#: `caption_track4.py`, rules fixed before each run). Against Moondream2's long caption on 280 confirm images:
#: keyword recall 69.0 -> 74.8% (+5.7, CI 3.4 to 8.0; matched length +8.0, CI 5.8 to 10.3), median 34.7 s, COCO
#: CHAIR-i 17.1 -> 16.5%, CHAIR-s 30.3 -> 31.0%. Measured exactly as below: this revision, bf16, the image processor's
#: max_tiles 4 (10 by default: 46.8 s on tune, over the 45 s cap, for no more recall), this prompt, the model card's
#: decoding. Changing any of them means re-running the gate.
DEFAULT_DEEP_CAPTION_MODEL: Final[str] = "LiquidAI/LFM2.5-VL-3B"
DEFAULT_DEEP_CAPTION_REVISION: Final[str] = "35a118d938ce6d123ac2d371649f24a8efb69058"
DEEP_CAPTION_MAX_TILES: Final[int] = 4
DEEP_CAPTION_PROMPT: Final[str] = "Describe this image in detail."
DEEP_CAPTION_MAX_NEW_TOKENS: Final[int] = 768

#: The default `caption` since 2026-10-23: LFM2.5-VL-3B on ONNX Runtime (Liquid AI's transformers.js export at this
#: revision: fp32 vision encoder and token embeddings, q8 decoder), one image of at most 256 tokens (no tiles),
#: CAPTION_PROMPT, the card's decoding with CAPTION_SEED. VisionBench `lfm_onnx.py` (rules fixed before each run): on
#: 280 held-out images 64.8% -> 71.0% of key facts against Florence-2-large bf16 (+6.2, CI 4.0 to 8.5; matched length
#: +9.1), fresh COCO CHAIR-i 13.3 -> 12.4%, CHAIR-s 27.7 -> 23.0%, median caption 11.4 s and caption + `ocr` 14.1 s on
#: CPU (Florence-2: 6.6 s), peak ~8.3 GB. Four tiles took 20 s; the prompt round's fuller prompts did not beat this one
#: on confirm. Changing any setting means re-running the gate. Florence-2 stays loaded for every other consumer.
DEFAULT_CAPTION_ONNX_MODEL: Final[str] = "LiquidAI/LFM2.5-VL-3B-ONNX"
DEFAULT_CAPTION_ONNX_REVISION: Final[str] = "020f9311343b8f3c6d9789b4dd300e749ec52cd1"
CAPTION_ONNX_FILES: Final[dict[str, str]] = {
    "vision": "onnx/vision_encoder.onnx",
    "embed": "onnx/embed_tokens.onnx",
    "decoder": "onnx/decoder_model_merged_q8.onnx",
}
CAPTION_PROMPT: Final[str] = "Describe only what is clearly visible in this image."
CAPTION_MAX_NEW_TOKENS: Final[int] = 768
#: Sampled, as measured; seeded so a caption repeats on the same image.
CAPTION_SEED: Final[int] = 0
#: As the gate ran (`-t 18` on a 24-thread CPU); capped at the machine's count.
CAPTION_ONNX_THREADS: Final[int] = 18
#: `--caption-model` choices.
CAPTION_MODELS: Final[tuple[str, ...]] = ("lfm-onnx", "florence2")

#: Since 2026-10-24 the default caption (LFM ONNX, not `deep` or `region`) starts with `Image type: <x>.`, x being CLIP's
#: top pick among these (prompts "a/an <type>", `Aesthetic.zero_shot`, the `score_aesthetics` model). VisionBench
#: `caption_type_line.py`, rules fixed before each run: tune 72.0 -> 74.5% (an LFM question instead: 74.0%, 4.4 s, its
#: answers vaguer), round 21 64.6 -> 67.0% (+2.4, CI 1.4 to 3.6; matched +1.1; game UI +10.4, app/web +6.2, no kind
#: worse), the 280 confirm images 71.3 -> 73.2%; 0.35 s. No label names a COCO object, so CHAIR is unchanged by
#: construction. The line agreed with the labelled kind on 69% of round 21. The list was written before looking at
#: which types the labels name; changing it means re-running the gate.
CAPTION_IMAGE_TYPES: Final[tuple[str, ...]] = (
    "photograph", "screenshot of a website", "screenshot of a mobile app", "screenshot of a desktop application",
    "video game screenshot", "chart or graph", "diagram", "infographic", "map", "satellite image", "scanned document",
    "text document", "poster", "logo", "product packaging", "street sign", "oil painting", "watercolor painting",
    "drawing or sketch", "cartoon or comic", "anime illustration", "digital illustration", "3D render", "pixel art",
)  # fmt: skip

#: `query_image` since 2026-10-24: the same LFM2.5-VL-3B ONNX model as the default caption (one loaded copy serves
#: both), answering the question verbatim, except that a question matching VQA_TEXT_QUESTION gets the `ocr` engine's
#: reading of the image in front (at most VQA_OCR_WORDS words) and VQA_TEXT_SUFFIX after; the card's decoding with
#: VQA_SEED, at most VQA_MAX_NEW_TOKENS tokens. VisionBench `vqa_premise.py` arm `ocr_text` (screened on phase 11, 82.4
#: vs Moondream2's 78.8%) and `vqa_round23.py` (rule fixed before round 23's questions existed): on 892 fresh questions
#: 78.5% vs 66.5% (+12.0), false-premise questions 66.4% vs 32.3% (text that is not there 65.7 vs 13.3%), no question
#: type worse, median 4.9 s vs 9.8 s per answer. The regex was fixed before phase 11's labels were looked at; changing
#: it or the prompts means re-running the gate. Moondream2 keeps critique_composition's hints and count_objects' estimate.
VQA_TEXT_QUESTION: Final[str] = (
    r"\b(text|written|writing|say|says|said|read|reads|title|headline|label|labell?ed|word|words|printed|print|"
    r"inscription|inscribed|caption|slogan|lettering|letters|spelled|brand name)\b"
)
VQA_TEXT_SUFFIX: Final[str] = "\nIf the question asks about text that is not in the image, answer that there is none."
VQA_OCR_WORDS: Final[int] = 160
VQA_MAX_NEW_TOKENS: Final[int] = 128
VQA_SEED: Final[int] = 0
#: `--vqa-model` choices.
VQA_MODELS: Final[tuple[str, ...]] = ("lfm-onnx", "moondream")

#: `caption(deep=true)` on Moondream2 (the fallback when no deep captioner is configured) tiles the image into at most DEEP_CROPS_TEXT local crops when the page's `ocr` text
#: has DEEP_TEXT_WORDS words or more, and DEEP_CROPS below that (Moondream2's own setting is 12; plan step
#: 1C-b, VisionBench `caption_1c_b.py`, 2026-10-21, both arms greedy). 6 crops: tune keyword recall 71.8 ->
#: 72.3%, COCO CHAIR-i 18.5 -> 18.2%, about 5 s faster. On 140 confirm images with 30+ OCR words it tied
#: (70.4 -> 70.6%, CI -1.8 to +2.3), but documents dipped on both sets (tune 72.5 -> 67.5%, confirm 79.3 ->
#: 77.2%), so text-heavy pages keep 12. The rule fixed before that run picked 6 for every call; the user
#: chose to keep 12 on text-heavy pages.
DEEP_CROPS: Final[int] = 6
DEEP_CROPS_TEXT: Final[int] = 12
DEEP_TEXT_WORDS: Final[int] = 30

#: `caption(deep=true)` also asks Moondream2 this on the image encoding its caption already made, and
#: adds `Theme: <answer>.` to the prose. Fixed before measuring (VisionBench `caption_theme.py`,
#: 2026-10-10): +1.3 points of keyword recall on tune and on 280 held-out images (CI 0.6 to 2.0),
#: 0 images worse. On a fresh encode it cost ~13 s; on the reused one ~2 s (`caption_c4_1.py`).
THEME_PROMPT: Final[str] = "In one sentence, what idea, theme or message does this image convey?"

#: Without a `target_subject`, `critique_composition` asks Moondream2 this and boxes the answer with
#: Florence-2. Fixed before measuring (VisionBench `composition_critique.py`, 2026-10-11).
MAIN_SUBJECT_QUESTION: Final[str] = (
    "What is the main subject of this photo? Answer with a short noun phrase, "
    "or 'none' if there is no single main subject."
)

#: `critique_composition(check_background=true)` asks Moondream2 this, word for word. Fixed before
#: measuring (VisionBench `composition_critique.py`, 2026-10-11): against a person's blind calls it
#: agreed 88.1% (balanced) on 25 tune photos and 69.2% on 52 held-out ones.
BUSY_BACKGROUND_QUESTION: Final[str] = (
    "Is the background busy or cluttered, with things that compete with the main subject for "
    "attention? Answer yes or no."
)

#: PP-OCR's detector keeps a candidate text box when its mean probability reaches this.
#: RapidOCR's default, 0.5, is tuned for documents; on 200 TextVQA photos, answer recall
#: climbs 55.0% (0.5) -> 57.0 (0.45) -> 58.5 (0.4) and plateaus (58.0 at 0.35 and 0.3).
#: Held constant: rendered pages 97.5% of lines, the held-out sourced images' lines 75.6% ->
#: 76.2%, text-free fixtures still empty -- the 0.75 recognition floor above still decides
#: what is returned. VisionBench `benchmarks/ppocr_det_sweep.py`, 2026-10-03.
OCR_DET_BOX_THRESHOLD: Final[float] = 0.4

#: SAM2 decodes masks on a fixed grid regardless of input size. Masks are upscaled
#: to the image's own pixels for reporting, but detail finer than
#: `max(image side) / MASK_DECODE_RESOLUTION` pixels is not actually resolved.
MASK_DECODE_RESOLUTION: Final[int] = 256

#: Memory each model needs to load and run once, in GB, keyed by the name its `IdleReleased`
#: cache is given in `app_lifespan`. `memory.MemoryManager` checks this against live free memory
#: before a load. Measured 2026-10-25 on CPU (VisionBench `model_memory.py`: one fresh process per
#: model, one representative call on a 2,000 px image), by a rule fixed before running: max(peak
#: commit, peak working set) above the post-import baseline, x 1.1, rounded up to 0.1 GB. Peaks,
#: not after-load sizes: SAM 3 holds 3.2 GB once loaded but peaks at 6.5 GB inside a call.
#: VisionReasoner is not measured (Ollama runs out of process). A size measured on load in the
#: running process can raise a figure there, never lower it.
MODEL_MEMORY_GB: Final[dict[str, float]] = {
    "Florence-2": 2.9,
    "Moondream": 5.0,
    "LFM2.5-VL": 7.0,
    "LFM2.5-VL ONNX": 8.7,
    "SAM2": 1.5,
    "SAM 3": 7.1,
    "Aesthetic": 2.0,
    "Grounding DINO": 2.1,
    "DINOv2": 1.4,
    "EasyOCR": 1.7,
    "PP-OCRv6": 0.5,
    "ImageQuality": 0.2,
    "VisionReasoner": 0.1,
}

#: What each model holds once loaded, between calls, in GB: the same 2026-10-25 run's after-load
#: figures by the same rule. A call needs `MODEL_MEMORY_GB` minus this on top (SAM 3: 3.6 held,
#: 3.5 more inside a call). PP-OCRv6, EasyOCR and ImageQuality build their sessions on first call,
#: so they are left out: taken to hold their peak once used. So is anything unmeasured.
MODEL_RESIDENT_GB: Final[dict[str, float]] = {
    "Florence-2": 2.3,
    "Moondream": 4.1,
    "LFM2.5-VL": 6.6,
    "LFM2.5-VL ONNX": 7.7,
    "SAM2": 0.3,
    "SAM 3": 3.6,
    "Aesthetic": 1.9,
    "Grounding DINO": 0.8,
    "DINOv2": 1.3,
}

#: A call adding less than this over the model's resident size is neither checked nor reserved.
#: CLIP and DINOv2 calls add about 0.1 GB; every other measured model adds 0.4 GB or more.
CALL_CHECK_MIN_GB: Final[float] = 0.25

#: The memory watcher (`MemoryManager.relieve_pressure`) reads free memory this often, in seconds.
MEMORY_WATCH_INTERVAL_S: Final[float] = 5.0

#: Free commit or RAM under this, in GB, is critical: the watcher releases idle models until
#: `MEMORY_WATCH_TARGET_GB` is free again. Twice the load reserve, so a load would already be
#: releasing to fit here.
MEMORY_WATCH_CRITICAL_GB: Final[float] = 4.0
MEMORY_WATCH_TARGET_GB: Final[float] = 6.0

#: Under critical RAM (not commit), only models idle at least this long go: a tool calls several
#: models in turn, and the one between steps is idle for a moment without being done with.
MEMORY_WATCH_MIN_IDLE_S: Final[float] = 30.0

#: Free RAM under this share of the total is tight: models idle `MEMORY_WATCH_TIGHT_IDLE_S` or
#: longer are released, as if the idle timer ran faster. 25% of 31.5 GB is about 7.9 GB.
MEMORY_WATCH_TIGHT_FRACTION: Final[float] = 0.25
MEMORY_WATCH_TIGHT_IDLE_S: Final[float] = 120.0

#: Paging pressure, which free RAM alone misses on Windows: under a squeeze the OS keeps a few GB free by
#: paging other programs out (VisionBench memory_stress S3, 2026-10-25: squeezed to 3.05 GB free, held at ~4.6 GB,
#: above `MEMORY_WATCH_CRITICAL_GB`, while 10 GB of idle models stayed loaded). Windows: `\Memory\Pages Output/sec`
#: at or above this, which read 0 at idle on the development laptop; Pages Input/sec is not used, as it also counts
#: memory-mapped file reads such as this server's own model loads. Set before the re-run, not tuned.
MEMORY_WATCH_PAGES_OUT_PER_S: Final[float] = 500.0
#: Linux: `/proc/pressure/memory` "some avg10" (percent of the last 10 s some task stalled on memory) at or above this.
MEMORY_WATCH_PSI_SOME_AVG10: Final[float] = 10.0
#: High paging on this many consecutive watcher checks is a burst.
MEMORY_WATCH_PAGING_CHECKS: Final[int] = 2

#: A burst is remembered this long, in seconds: while it is, with free RAM under the tight level, the watcher releases
#: one model per check as it passes `MEMORY_WATCH_MIN_IDLE_S` idle. On Windows a squeeze is a short burst (the S3
#: re-run, 2026-10-25: 70,622 then 27,562 pages out/s, then 0 within 10 s), usually while the models were just used,
#: so requiring the burst and a 30 s-idle model at the same moment missed. Set before its run, not tuned.
MEMORY_WATCH_PAGING_MEMORY_S: Final[float] = 60.0

#: How many recent loads, releases and refusals `server_status` reports.
EVENT_LOG_SIZE: Final[int] = 20

#: Used for a cache name missing from `MODEL_MEMORY_GB`.
DEFAULT_MODEL_MEMORY_GB: Final[float] = 1.0

#: Free memory a load must leave behind, in GB, for the OS, the client and inference buffers.
MEMORY_RESERVE_GB: Final[float] = 2.0
