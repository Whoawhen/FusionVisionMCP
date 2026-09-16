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


class InstanceDetector(Protocol):
    """Represents a protocol for open-vocabulary detection that tallies instances."""

    def detect_objects(
        self,
        images: list[Image],
        object_name: str,
        threshold: float = DEFAULT_BOX_THRESHOLD,
        adaptive_threshold: bool = True,
    ) -> list[dict[str, Any]]:
        """Locates every instance of the named object, with a count and per-box scores."""
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

    def classify_style(self, images: list[Image]) -> list[dict[str, Any]]:
        """Zero-shot medium/genre classification reusing the CLIP backbone.

        Returns a `{style, distribution}` dict per image; `style` is the top-ranked
        medium (e.g. "photograph", "oil painting") and `distribution` is a sorted
        list of `{style, score}` probabilities.
        """
        ...


@dataclass
class AppContext:
    """Context for the FastMCP app."""

    florence2: Processor
    vqa: VqaProcessor
    segmenter: Segmenter
    aesthetic: AestheticScorer
    counter: InstanceDetector
    ocr_specialist: SpecialistOCR | None = None
    iqa: Any | None = None
    reasoner: Any | None = None
