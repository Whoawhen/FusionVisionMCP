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

#: `caption(region=...)`: the caller's box is widened by this share of its width and height on each side, to at
#: least `REGION_MIN_SIDE` px each way, then clipped to the image. Exactly the crop VisionBench gate R measured
#: (`region_inspect_gate.py`, 2026-10-15): on 528 questions about a worker-chosen region, the default caption of the
#: crop named the answer 80.3% of the time against 67.8% for the whole image (+12.5, 95% CI 8.3 to 16.9).
REGION_PAD: Final[float] = 0.15
REGION_MIN_SIDE: Final[int] = 32

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
CAPTION_TEXT_WORDS: Final[int] = 40

#: `caption(deep=true)` stops Moondream2's long caption once it loops: when LOOP_RUN consecutive
#: LOOP_NGRAM-word spans all repeat earlier text, the caption is cut where that run began. The
#: sampled long caption loops on ~7% of images (hundreds of "effects, effects, ..." up to the
#: 768-token cap, 97-170 s). Replayed on cached tune captions (VisionBench `caption_c4_0.py`,
#: 2026-10-12) it cut 14 of 120 and cost 0.2 points of keyword recall (one group, hit only
#: inside a loop).
LOOP_NGRAM: Final[int] = 8
LOOP_RUN: Final[int] = 16

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
