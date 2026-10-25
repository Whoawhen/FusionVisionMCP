"""Structural types the tools are written against, and the per-session context.

These are `Protocol`s rather than imports of the concrete wrappers on purpose: the
package entry point must stay torch-free (see CLAUDE.md), and every wrapper imports
torch at its own module level. Writing the tools against a protocol lets them be
type-checked without the real class ever being imported at module scope -- and lets
`IdleProxy` stand in for one at runtime.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image

from fusion_vision_mcp.constants import DEFAULT_BOX_THRESHOLD, CaptionLevel
from fusion_vision_mcp.ocr_fusion import SpecialistOCR

__all__ = [
    "AestheticScorer",
    "AppContext",
    "InstanceDetector",
    "Processor",
    "Segmenter",
    "VqaProcessor",
]


class Processor(Protocol):
    """Represents a protocol for processing image data.

    This class provides an interface for implementing image processing
    operations, including optical character recognition (OCR) and generating
    captions based on the content of the images. It is meant to be used as a
    guideline for defining specific processors that conform to this protocol.
    """

    def ocr(self, images: list[Image]) -> list[str]:
        """Performs optical character recognition (OCR) on a list of images.

        This function takes a list of images and processes each image using OCR
        to retrieve the text content present within the images. The function
        returns a list of strings, where each string corresponds to the text
        extracted from the respective image in the input list.
        """
        ...

    def caption(self, images: list[Image], level: CaptionLevel = CaptionLevel.NORMAL) -> list[str]:
        """Generates a list of captions for the given images based on the specified captioning level.

        It processes an input list of images and returns the corresponding captions
        in a text format. The caption level influences the verbosity or granularity
        of the generated captions.
        """
        ...

    def detect_objects(
        self, images: list[Image], object_name: str, exclude_full_frame: bool = False
    ) -> list[dict[str, Any]]:
        """Locates instances of the named object, returning bounding boxes, center points and labels."""
        ...

    def dense_region_caption(self, images: list[Image]) -> list[dict[str, Any]]:
        """Generates a caption for every salient region in the image, with bounding boxes."""
        ...

    def ocr_with_regions(self, images: list[Image]) -> list[dict[str, Any]]:
        """OCR returning verbatim text plus the box each span occupies.

        Unlike `ocr` (which returns flat strings), this preserves *where* each text
        span is, so a caller can locate a phrase or cross-check text a caption quoted.
        Returns `{quad_boxes, bboxes, labels}` per image; `bboxes` are axis-aligned.
        """
        ...

    def generate(self, prompt: str, images: list[Image]) -> list[str]:
        """Generates text responses for the given images based on a custom prompt.

        This function processes a list of images using the Florence-2 model with
        a custom prompt string. It allows for flexible image analysis by accepting
        task-specific prompts that define what information to extract or generate
        from the images.

        Args:
            prompt: A task prompt string that specifies the operation to perform
                on the images (e.g., "<OCR>", "<CAPTION>", or custom task prompts
                supported by the Florence-2 model).
            images: A list of PIL Image objects to be processed.

        Returns:
            A list of strings containing the generated text for each image, where
            each string corresponds to the model's response for the respective
            input image.
        """
        ...


class VqaProcessor(Protocol):
    """Represents a protocol for free-form visual question answering (VQA)."""

    def query(self, images: list[Image], question: str) -> list[str]:
        """Answers a free-form question about each image."""
        ...

    def caption(
        self, images: list[Image], length: str = "long", theme: bool = False, max_crops: list[int] | None = None
    ) -> list[str]:
        """Describes each image in prose ("short", "normal" or "long"); `theme` adds what it conveys.

        `max_crops`, one per image, caps the local crops the image is tiled into (the model's default if None)."""
        ...


class Captioner(Protocol):
    """Represents a protocol for a prose captioner (`caption`'s default, and `caption(deep=true)`)."""

    def caption(self, images: list[Image]) -> list[str]:
        """One detailed prose caption per image."""
        ...


class Answerer(Protocol):
    """Represents a protocol for `query_image`'s answerer (LFM2.5-VL-3B ONNX), given a finished prompt per image."""

    def answer(self, images: list[Image], prompts: list[str]) -> list[str]:
        """One reply per image to its prompt."""
        ...


class InstanceDetector(Protocol):
    """Represents a protocol for open-vocabulary detection that tallies instances."""

    def detect_objects(
        self,
        images: list[Image],
        object_name: str,
        threshold: float = DEFAULT_BOX_THRESHOLD,
        adaptive_threshold: bool = True,
        candidates_floor: float | None = None,
    ) -> list[dict[str, Any]]:
        """Locates every instance of the named object, with a count and per-box scores.

        With `candidates_floor`, also `candidates_below`: unreturned boxes scoring from the floor up to
        `threshold`, for `count_objects`' `possibly_more`. A detector that does not count ignores it.
        """
        ...


class Segmenter(Protocol):
    """Represents a protocol for turning bounding boxes into segmentation masks."""

    def segment(self, image: Image, boxes: list[list[int]]) -> list[NDArray[np.bool_]]:
        """Returns one boolean mask per box, on the image's own pixel grid."""
        ...


class AestheticScorer(Protocol):
    """Represents a protocol for rating how aesthetically pleasing an image is."""

    def score(self, images: list[Image]) -> list[dict[str, Any]]:
        """Returns a {"score": float, "rating": str} dict per image."""
        ...

    def score_and_classify(self, images: list[Image]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Both of the above off a single image encode."""
        ...

    def image_embeddings(self, images: list[Image]) -> list[list[float]]:
        """Unnormalised CLIP image embeddings, one 768-float list per image."""
        ...

    def classify_style(self, images: list[Image]) -> list[dict[str, Any]]:
        """Zero-shot medium/genre classification reusing the CLIP backbone.

        Returns a `{style, distribution}` dict per image; `style` is the top-ranked
        medium (e.g. "photograph", "oil painting") and `distribution` is a sorted
        list of `{style, score}` probabilities.
        """
        ...

    def zero_shot(self, images: list[Image], prompts: list[str]) -> list[list[float]]:
        """CLIP zero-shot probabilities of each caller-supplied prompt, one softmax row per image."""
        ...


class CropEmbedder(Protocol):
    """Represents a protocol for embedding image crops for the learned defect probe."""

    def crop_features(self, images: list[Image]) -> list[list[float]]:
        """One feature list per image, in the layout the shipped probe was trained on."""
        ...


@dataclass
class AppContext:
    """Context for the FastMCP app."""

    florence2: Processor
    vqa: VqaProcessor
    segmenter: Segmenter
    aesthetic: AestheticScorer
    counter: InstanceDetector
    # Both always provided by `app_lifespan` as lazy IdleProxies, never None.
    # `ocr_specialist` is EasyOCR, kept only for structured_analysis's gibberish check:
    # its 0.20 confidence threshold is calibrated to EasyOCR's scores. `ocr_engine` backs
    # the `ocr` tool and caption verification: PP-OCRv6 English, or EasyOCR again when
    # the server is started with non-English --ocr-languages.
    ocr_specialist: SpecialistOCR
    ocr_engine: SpecialistOCR
    iqa: Any | None = None
    reasoner: Any | None = None
    # The defect probe's backbone (DINOv2), lazy like the rest; None only in tests and callers
    # that build an AppContext by hand, which then can only run a CLIP-backed probe.
    defect_embedder: CropEmbedder | None = None
    # SAM 3, `detect_objects`' photo detector. None in tests and hand-built contexts, where
    # `detect_objects` falls back to Grounding DINO (`counter`) as it does when SAM 3 can't load.
    locator: InstanceDetector | None = None
    # LFM2.5-VL-3B, `caption(deep=true)`'s captioner. None in tests and hand-built contexts, where `deep` falls back
    # to Moondream2's long caption (`vqa`), the captioner before 2026-10-22.
    deep_captioner: Captioner | None = None
    # LFM2.5-VL-3B on ONNX Runtime, `caption`'s default captioner since 2026-10-23. None with
    # `--caption-model florence2`, in tests and in hand-built contexts, where the default caption is Florence-2's.
    captioner: Captioner | None = None
    # The same LFM2.5-VL-3B ONNX model (one proxy, one loaded copy), `query_image`'s answerer since 2026-10-24. None
    # with `--vqa-model moondream`, in tests and in hand-built contexts, where `query_image` answers with Moondream2.
    answerer: Answerer | None = None
    # The memory manager every model shares (`memory.MemoryManager`), for `server_status`. None in tests and
    # hand-built contexts, where `server_status` reports that memory is not managed.
    memory: Any | None = None
