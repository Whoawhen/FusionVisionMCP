#  __init__.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import importlib.metadata
import math
import os
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager, closing, contextmanager
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Final, Protocol, cast

import numpy as np
import requests
from mcp.server.mcpserver import Context, MCPServer
from numpy.typing import NDArray
from PIL.Image import Image
from PIL.Image import open as open_image
from pydantic import Field
from pypdfium2 import PdfDocument

from fusion_vision_mcp import geometry, layout, question, textmatch
from fusion_vision_mcp.aesthetic import DEFAULT_AESTHETIC_MODEL, Aesthetic
from fusion_vision_mcp.florence2 import CaptionLevel, Florence2, Florence2SP
from fusion_vision_mcp.grounding_dino import DEFAULT_GROUNDING_DINO_MODEL, GroundingDino
from fusion_vision_mcp.idle import IdleProxy, IdleReleased
from fusion_vision_mcp.moondream import DEFAULT_MOONDREAM_MODEL, DEFAULT_MOONDREAM_REVISION, Moondream
from fusion_vision_mcp.sam2 import DEFAULT_SAM2_MODEL, MASK_DECODE_RESOLUTION, Sam2

SERVER_NAME: Final[str] = "FusionVisionMCP"

#: Ceiling on objects compared in one `spatial_relations` call. Relations grow as
#: n(n-1)/2, so an over-broad detection could otherwise return a huge payload.
_MAX_RELATED_OBJECTS: Final[int] = 12

#: `requests`' default User-Agent identifies the library, not a specific client, and
#: some hosts reject it outright as an anti-scraping measure — confirmed live against
#: Wikimedia, which 403s the default UA but accepts a descriptive one.
_USER_AGENT: Final[str] = (
    f"{SERVER_NAME}/{importlib.metadata.version('fusion-vision-mcp')} (+https://github.com/Whoawhen/FusionVisionMCP)"
)


@contextmanager
def get_images(src: os.PathLike[str] | str) -> Iterator[list[Image]]:
    """Opens and returns a list of images from a file path or URL."""
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        res = requests.get(src, headers={"User-Agent": _USER_AGENT})
        res.raise_for_status()

        if res.headers["Content-Type"] == "application/pdf":
            with ExitStack() as stack:
                images = []
                with closing(PdfDocument(res.content)) as doc:
                    for page in doc:
                        images.append(stack.enter_context(page.render().to_pil()))
                yield images

        else:
            with open_image(BytesIO(res.content)) as image:
                yield [image]

    else:
        path = Path(src)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            with ExitStack() as stack:
                images = []
                with closing(PdfDocument(path)) as doc:
                    for page in doc:
                        images.append(stack.enter_context(page.render().to_pil()))
                yield images
        else:
            with open_image(path) as image:
                yield [image]


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

    def detect_objects(self, images: list[Image], object_name: str) -> list[dict[str, Any]]:
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

    def detect_objects(self, images: list[Image], object_name: str) -> list[dict[str, Any]]:
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

    processor: Processor
    vqa: VqaProcessor
    segmenter: Segmenter
    aesthetic: AestheticScorer
    counter: InstanceDetector


@asynccontextmanager
async def app_lifespan(
    _server: MCPServer,
    model_id: str,
    subprocess: bool,
    moondream_model_id: str,
    moondream_revision: str,
    sam2_model_id: str = DEFAULT_SAM2_MODEL,
    aesthetic_model_id: str = DEFAULT_AESTHETIC_MODEL,
    grounding_dino_model_id: str = DEFAULT_GROUNDING_DINO_MODEL,
    idle_timeout: float = 0,
    device: str | None = None,
) -> AsyncIterator[AppContext]:
    """Context manager for the FastMCP app lifespan.

    Each model is wrapped separately, so a request only ever loads the model it
    actually needs: captioning never pulls in Moondream, and nothing but
    `spatial_relations` pulls in SAM2. Each is released on its own idle timer.

    `idle_timeout` is what the CLI's `--memory-mode` resolves to: a positive value
    releases each model that many seconds after its last use, while 0 leaves them
    resident for the process's lifetime -- the fastest, most memory-hungry setting.
    """
    processor: Processor
    vqa: VqaProcessor
    segmenter: Segmenter
    aesthetic: AestheticScorer
    if idle_timeout > 0:
        # Keep each model in this process so repeat calls stay fast, and let the
        # idle timer hand its memory back once the work stops.
        processor = cast(
            Processor,
            IdleProxy(IdleReleased(lambda: Florence2(model_id, device), idle_timeout, "Florence-2")),
        )
        vqa = cast(
            VqaProcessor,
            IdleProxy(
                IdleReleased(
                    lambda: Moondream(moondream_model_id, moondream_revision, device),
                    idle_timeout,
                    "Moondream",
                )
            ),
        )
    else:
        if subprocess:
            processor = Florence2SP(model_id, device)
        else:
            processor = Florence2(model_id, device)
        vqa = Moondream(moondream_model_id, moondream_revision, device)

    # Always lazy, on both paths. `IdleReleased` builds on first use and, with a
    # timeout of 0, simply never schedules a release — so a session that never
    # calls `spatial_relations` never pays for SAM2 at all.
    segmenter = cast(
        Segmenter,
        IdleProxy(IdleReleased(lambda: Sam2(sam2_model_id, device), idle_timeout, "SAM2")),
    )
    # Same rationale as SAM2: always idle-wrapped regardless of idle_timeout, since only
    # score_aesthetics/critique_composition pay for the CLIP backbone, and most sessions
    # never call either.
    aesthetic = cast(
        AestheticScorer,
        IdleProxy(IdleReleased(lambda: Aesthetic(aesthetic_model_id, device), idle_timeout, "Aesthetic")),
    )
    # Same rationale again: only count_objects loads Grounding DINO, so a session that
    # never counts never pays the ~690MB.
    counter = cast(
        InstanceDetector,
        IdleProxy(IdleReleased(lambda: GroundingDino(grounding_dino_model_id, device), idle_timeout, "Grounding DINO")),
    )
    yield AppContext(processor, vqa, segmenter, aesthetic, counter)


def server(
    name: str,
    model_id: str,
    subprocess: bool = True,
    moondream_model_id: str = DEFAULT_MOONDREAM_MODEL,
    moondream_revision: str = DEFAULT_MOONDREAM_REVISION,
    sam2_model_id: str = DEFAULT_SAM2_MODEL,
    aesthetic_model_id: str = DEFAULT_AESTHETIC_MODEL,
    grounding_dino_model_id: str = DEFAULT_GROUNDING_DINO_MODEL,
    idle_timeout: float = 0,
    device: str | None = None,
) -> MCPServer:
    """Creates a new FastMCP server instance with the specified name and model ID."""
    mcp = MCPServer(
        name,
        lifespan=partial(
            app_lifespan,
            model_id=model_id,
            subprocess=subprocess,
            moondream_model_id=moondream_model_id,
            moondream_revision=moondream_revision,
            sam2_model_id=sam2_model_id,
            aesthetic_model_id=aesthetic_model_id,
            grounding_dino_model_id=grounding_dino_model_id,
            idle_timeout=idle_timeout,
            device=device,
        ),
    )

    ImagePath = Annotated[
        os.PathLike[str] | str,
        Field(
            description=(
                "Local file path or http(s) URL of the image to process. PDFs are also accepted "
                "and are rendered one image per page, so tools that return a list return one entry "
                "per page."
            )
        ),
    ]
    ObjectName = Annotated[
        str,
        Field(
            description=(
                "Name of the object to locate, e.g. 'person', 'car', 'face'. A short noun phrase "
                "works too ('the red mug'). More specific names ground more reliably than broad ones."
            )
        ),
    ]
    CustomPrompt = Annotated[
        str,
        Field(
            description=(
                "A Florence-2 task token, e.g. '<OD>', '<CAPTION>', '<REGION_PROPOSAL>'. Not a "
                "natural-language instruction -- plain English here produces garbage, not an answer."
            )
        ),
    ]
    Operation = Annotated[
        str,
        Field(
            description=(
                "One of: 'caption', 'ocr', 'detect', 'count', 'dense_caption', 'query'. Use 'query' "
                "(with `question`) rather than 'ocr' for watermarks, logos, signage, or "
                "stylized/cursive text -- 'ocr' misreads that kind of text confidently. Use 'count' "
                "(with `object_name`) rather than 'detect' for 'how many' -- 'detect' returns "
                "regions, which are not a tally."
            )
        ),
    ]

    @mcp.tool()
    def ocr(
        ctx: Context[AppContext],
        src: ImagePath,
        with_regions: Annotated[
            bool,
            Field(
                description=(
                    "When true, return verbatim text *and* the box each text span occupies "
                    "(via Florence-2's OCR-with-region head), instead of flat strings. Each "
                    "page becomes a dict with `text_regions` ({text, box}[], box=[x1,y1,x2,y2] "
                    "in page coordinates, already offset back out of any column crop) and the "
                    "joined `text`. Use this when you need to know *where* a phrase is, or to "
                    "cross-check text a `caption` quoted."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Process an image file or URL using OCR to extract text.

        Best for dense, printed, document-style text (receipts, scanned pages,
        paragraphs). Do NOT use this for photo watermarks, logos, signage, or
        any cursive/stylized/low-contrast text — it confidently misreads that
        kind of text (fabricates plausible-looking wrong words) rather than
        failing visibly. Use `query_image` instead for that case, e.g. with
        the question "What does the text/watermark say, exactly?".

        A page laid out in side-by-side columns (a form, meeting notes, a
        resume) is detected automatically: each column is OCR'd separately and
        joined in reading order, so fields from different columns don't get
        interleaved the way naive raster-order OCR would interleave them.

        Set `with_regions=true` to also get the location of every text span.
        The output then becomes one dict per page with `text` (the joined
        transcription, as returned when `with_regions=false`) and `text_regions`
        (`[{text, box}, ...]` in page coordinates — boxes are offset back out of
        any column crop, so they index directly into the original image). This is
        the right choice when you need to point at a phrase, or to confirm a
        name/brand a `caption` quoted against what the OCR head actually read.
        """
        processor = ctx.request_context.lifespan_context.processor
        with get_images(src) as images:
            per_page_columns = [layout.split_columns(image) for image in images]

            if not with_regions:
                flat_texts = processor.ocr([crop for columns in per_page_columns for crop in columns])
                results = []
                i = 0
                for columns in per_page_columns:
                    results.append("\n".join(flat_texts[i : i + len(columns)]))
                    i += len(columns)
                return results

            # with_regions: each column crop is OCR'd with its boxes, then the boxes are
            # offset back into page coordinates so a caller can index into the original image.
            # PIL's cropped Image exposes no crop-origin attribute, so the column x-offsets
            # are reconstructed from the same split points `split_columns` crops at.
            page_results: list[dict[str, Any]] = []
            for image, columns in zip(images, per_page_columns, strict=True):
                splits = layout.find_column_splits(image)
                bounds = [0, *splits, image.width]
                # `split_columns` crops at (bounds[i], 0, bounds[i+1], image.height).
                column_offsets = [(bounds[i], 0) for i in range(len(bounds) - 1)]
                regioned = processor.ocr_with_regions(columns)
                text_regions: list[dict[str, Any]] = []
                page_texts: list[str] = []
                for crop, crop_result, (offset_x, offset_y) in zip(columns, regioned, column_offsets, strict=True):
                    for text, box in zip(crop_result.get("labels", []), crop_result.get("bboxes", [])):
                        x1, y1, x2, y2 = box
                        text_regions.append(
                            {
                                "text": text,
                                "box": [
                                    int(x1 + offset_x),
                                    int(y1 + offset_y),
                                    int(x2 + offset_x),
                                    int(y2 + offset_y),
                                ],
                            }
                        )
                        page_texts.append(text)
                page_results.append({"text": "\n".join(page_texts), "text_regions": text_regions})
            return page_results

    @mcp.tool()
    def caption(
        ctx: Context[AppContext],
        src: ImagePath,
        verify_text: Annotated[
            bool,
            Field(
                description=(
                    "When true, also run Florence-2's OCR-with-region head and return, alongside "
                    "the caption, the verbatim text spans it read (`text_regions`: {text, box}[]), "
                    "a `corrections` list of the close misses it fixed, and a `caption_corrected` "
                    "copy with each verbatim OCR span substituted in. The caption head paraphrases "
                    "text and misspells names/brands (it rendered this project's own 'FusionVisionMCP' "
                    "logo as 'FusionVisionMP'); the OCR head transcribes verbatim, so any text the "
                    "caption quotes can be confirmed against `text_regions` and the corrected caption "
                    "used directly. Default false keeps the original list[str] return shape; true "
                    "returns one dict per page ({caption, text_regions, corrections, caption_corrected})."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Describe what an image shows, as one detailed prose caption.

        The default choice for "what is this a picture of". Returns a single
        paragraph covering the scene as a whole, with no coordinates.

        Reach for a different tool when the question is narrower: `query_image`
        to ask something specific about the image, `dense_region_caption` to get
        a separate caption and box for each thing in it, and `ocr` to transcribe
        text rather than describe it. Returns one caption per page for a PDF.

        Do not trust any text this quotes back. A caption that mentions a name,
        brand or label is describing it, not transcribing it, and Florence-2
        misspells text here that it reads correctly under `ocr` -- it rendered a
        logo reading "FusionVisionMCP" as "FusionVisionMP" mid-caption while both
        `ocr` and `query_image` read the same image exactly. When a specific piece
        of text matters, confirm it with `ocr` (printed, document-style) or
        `query_image` (stylized, cursive, low-contrast) rather than quoting this.

        Set `verify_text=true` to have this confirmation done for you: the tool
        also runs the OCR-with-region head and returns each verbatim text span
        alongside its box, so a name the caption quoted can be checked against
        what was actually read without a second call. It then goes further and
        corrects the close misses: any caption token that is similar to (but
        not identical to) a verbatim OCR span is substituted with the verbatim
        text in a `caption_corrected` copy, and every such change is listed in
        `corrections` (with the quoted-in-caption token, the verbatim OCR text,
        the box and the similarity) so the substitution is auditable. The
        return shape becomes one dict per page
        (`{caption, text_regions, corrections, caption_corrected}`) when this
        is set. The correction is best-effort and only fires for high-similarity
        same-word matches; the raw `corrections` list is always present so a
        caller can audit or reject any change.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            captions = app.processor.caption(images, CaptionLevel.MORE_DETAILED)
            if not verify_text:
                return captions

            # Cross-check: the caption head paraphrases/mispells text; the OCR-with-region
            # head transcribes it verbatim. Run both and surface the verbatim spans so a
            # caller can confirm any name/brand the caption quoted before repeating it.
            # Go one step further than v0.6.0's raw spans: correct the close misses by
            # substituting each verbatim OCR span into a corrected copy of the caption,
            # and report each substitution so every change is auditable.
            regioned = app.processor.ocr_with_regions(images)
            results = []
            for caption_text, page_regions in zip(captions, regioned, strict=True):
                text_regions = [
                    {"text": text, "box": [int(v) for v in box]}
                    for text, box in zip(page_regions.get("labels", []), page_regions.get("bboxes", []))
                ]
                corrections, caption_corrected = textmatch.correct_text(caption_text, text_regions)
                results.append(
                    {
                        "caption": caption_text,
                        "text_regions": text_regions,
                        "corrections": [c.as_dict() for c in corrections],
                        "caption_corrected": caption_corrected,
                    }
                )
            return results

    @mcp.tool()
    def detect_objects(ctx: Context[AppContext], src: ImagePath, object_name: ObjectName) -> list[dict[str, Any]]:
        """Locate a named object in an image, as bounding boxes and center points.

        Returns `bboxes` ([x1, y1, x2, y2] each), `points` (the center of each box)
        and `labels`, all index-aligned -- so use this whether you want regions or
        coordinates; the centers come free with the boxes.

        The count of results is NOT a reliable count of objects on an ambiguous
        class name: Florence-2 can return several overlapping results for one
        physical object (a whole-animal box plus sub-part boxes all labelled
        'wing'), or a single result spanning two touching instances (two fused
        blades labelled once as 'sword blade'). Prefer a more specific
        `object_name`, and treat results as candidates to inspect, not a tally.
        Use `count_objects` when you actually need "how many" -- it runs a
        detection head that emits one region per instance, which this head does
        not. Neither can separate heavily overlapping instances, so a count of 1
        from either means "could not separate", not "there is one".

        Boxes cannot answer whether two objects actually touch or whether one is
        inside another -- they overlap the moment one object is merely in front
        of another. Use `spatial_relations` for that.
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.processor.detect_objects(images, object_name)

    @mcp.tool()
    def dense_region_caption(ctx: Context[AppContext], src: ImagePath) -> list[dict[str, Any]]:
        """Caption every salient region of an image at once, with bounding boxes.

        Use this to inventory an image without knowing in advance what is in it --
        it returns `bboxes` and `labels` for each region it finds, discovering the
        objects itself. That is the difference from `detect_objects`, which needs
        you to name the object you are looking for, and from `caption`, which
        describes the whole scene in prose with no coordinates.
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.processor.dense_region_caption(images)

    @mcp.tool()
    def query_image(
        ctx: Context[AppContext],
        src: ImagePath,
        question: Annotated[str, Field(description="A free-form question to ask about the image.")],
        check_consistency: Annotated[
            bool,
            Field(
                description=(
                    "When true, also ask a rephrased control question and report whether the "
                    "two answers agree, flagging short default-looking answers ('None', 'Yes', "
                    "'No', 'Nothing', ...) as low confidence. Moondream2 is a small VLM that "
                    "answers open-ended judgment questions ('describe anything wrong') with a "
                    "flat 'None' on images that all had real visible defects -- this layer makes "
                    "that default-answer behavior visible instead of presenting it as reliable. "
                    "On a low-confidence answer it also routes to the measurement that actually "
                    "answers the question when one applies (spatial_relations for a "
                    "contact/containment question, count_objects for 'how many', ocr for a "
                    "text-reading question) and attaches it as `cross_check`. Default false keeps "
                    "the original list[str] return; true returns one "
                    "{answer, control_answer, consistent, confidence, cross_check?} dict per image."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Ask a free-form question about an image (visual question answering).

        This is the right tool for reading photo watermarks, logos, signage,
        or any cursive/stylized/low-contrast text — ask e.g. "What does the
        text/watermark say, exactly?". The `ocr` tool misreads that kind of
        text confidently; prefer this one for it instead.

        Moondream2 is a small model and is documented to answer open-ended judgment
        questions ("describe anything wrong in this image") with a flat "None" on
        images that all had real visible defects, and to give the same yes/no answer
        across genuinely different images -- a default response, not a real
        observation. Set `check_consistency=true` to make that visible: the tool also
        asks a rephrased control question and returns, per image,
        `{answer, control_answer, consistent, confidence}`. `confidence` is `"low"`
        when both answers are short default-looking strings that agree -- the
        signature of a flat default rather than a genuine observation -- and
        `"normal"` otherwise. A `low` result on a judgment question means you should
        not trust the answer without independent confirmation. When the answer is
        low-confidence, the tool also tries to route to the measurement that
        actually answers the question: it classifies the question's wording and,
        if a measurable category applies and the object names parse from the
        wording, runs that tool's measurement (`spatial_relations` for a
        contact/containment question, `count_objects` for "how many", `ocr` for a
        text-reading question) and attaches it as `cross_check`. The cross-check is
        omitted when no measurement applies or the names can't be parsed -- it
        never guesses.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            if not check_consistency:
                return app.vqa.query(images, question)

            # A rephrasing a genuinely-looking model answers with the same substance,
            # but a model defaulting to a flat answer returns the same short default to.
            control_question = f"Looking carefully at this image, answer precisely: {question}"
            answers = app.vqa.query(images, question)
            control_answers = app.vqa.query(images, control_question)

            results = []
            for image, answer, control_answer in zip(images, answers, control_answers, strict=True):
                entry = _vqa_consistency(answer, control_answer)
                # When the VQA judgment is unreliable, route to the measurement that
                # actually answers the question (measure, don't judge): classify the
                # question's wording and, if a measurable category applies and the
                # object names parse, run that tool's measurement and attach it. The
                # cross-check is omitted (not guessed) when no measurement applies.
                if entry["confidence"] == "low":
                    cross = _vqa_cross_check(app, image, question)
                    if cross is not None:
                        entry["cross_check"] = cross
                results.append(entry)
            return results

    @mcp.tool()
    def count_objects(
        ctx: Context[AppContext],
        src: ImagePath,
        object_name: ObjectName,
        verify_silhouette: Annotated[
            bool,
            Field(
                description=(
                    "When the detector finds only one region, segment it and measure how many "
                    "repeated lobes its outline has. Costs one SAM2 load on the first such call."
                )
            ),
        ] = True,
        consensus: Annotated[
            bool,
            Field(
                description=(
                    "When true (default), also tally how many `dense_region_caption` labels match "
                    "the object name as an independent second opinion, and report a `separable` "
                    "flag. The dense region captioner is a different Florence-2 head than Grounding "
                    "DINO, so agreement between the two is real evidence and disagreement is a "
                    "visible warning. Cheap -- uses the already-loaded Florence-2, no new model."
                )
            ),
        ] = True,
        vqa_estimate: Annotated[
            bool,
            Field(
                description=(
                    "When true and the detector has collapsed overlapping instances (separable is "
                    "'no'), also ask Moondream2 'how many <name>?' and attach its answer as "
                    "estimates.vqa -- a judgment estimate, clearly marked, not a tally. Off by "
                    "default so a session that never asks keeps Moondream unloaded."
                )
            ),
        ] = False,
    ) -> list[dict[str, Any]]:
        """Count how many instances of a named object an image contains.

        Use this, not `detect_objects`, whenever the question is "how many".
        `detect_objects` returns however many regions Florence-2's grounding head
        emits, which is not a tally: it collapses several repeated, undifferentiated
        parts into one box (every petal of a flower, both halves of a fused blade)
        and conversely splits one object into overlapping sub-part boxes.

        Returns `count`, plus `bboxes`/`points`/`labels`/`scores` for the instances
        found, index-aligned and in the same pixel-space convention `detect_objects`
        uses. `scores` are per-detection confidences, so a count resting on marginal
        detections is visible rather than implied. `group_boxes_dropped` counts
        detections that enclosed the whole arrangement rather than one instance.

        Accuracy is measured, not assumed: against a fixture suite whose synthetic
        counts are exact by construction, this is right on 9 of 10 positive cases with
        a mean error of 0.1, while holding all 8 single-object controls. Separated,
        touching, dense, two-instance and awkwardly-named cases are reliable.

        Two measured limits, worth checking before trusting a count:

        Heavy overlap undercounts. Eight identical shapes in a ring count as 8 when
        separated or touching, but 6 once they overlap by roughly two-thirds of their
        width. A count below what you expect means "could not separate them", not a
        real tally -- and it will undercount rather than overcount.

        Some evidence is simply not in the picture. On a photo of a paper flower whose
        petals overlap, this returns 1 at every resolution from 128px to 768px, as does
        every other approach tried -- other detectors, outline geometry, tiled crops,
        and interior-colour analysis alike. That flower's outline is 98% convex and its
        interior contrast is near zero, so nothing measurable distinguishes the petals.
        Ask `query_image` for a count in that situation and treat it as an estimate.

        When only one region is found, a `silhouette` block is added measuring how
        many repeated lobes that region's outline contains. `count` and
        `silhouette.lobes` come from different methods and **neither overrides the
        other**: `count: 1` beside `silhouette.lobes: 8` means the detector could not
        separate the instances while the outline shows eight cores. `agreement: true`
        means a second, independent estimator matched it; `shattered` or `clipped`
        mean the number should not be used at all.

        Set `consensus=true` (the default) for two extra fields that surface
        structural ambiguity instead of hiding it:

        - `consensus`: `{grounding_dino_count, region_label_count, agree}` -- a second
          count from Florence-2's dense region captioner (a different head), which
          tally how many region labels contain the object name. Agreement is real
          evidence; disagreement is a warning.
        - `separable`: `"yes"` if the count is a real tally (the detector separated
          instances, or a single region the silhouette confirms is one lobe), `"no"`
          if the detector collapsed while the outline shows several lobes (the
          overlapping-petal case: nothing local can count it honestly), `"unknown"`
          when there's no silhouette check to confirm either way. Read `count` with
          this in mind: a `count: 1` with `separable: "no"` is a collapse, not a tally.

        When `separable` is `"no"` but the outline still carries the lobe pattern
        (`silhouette.by_radial > 1`), an `estimates` block is added reporting that
        outline count as an actionable number -- it's a *measurement* (angular
        notches in the silhouette), not a judgment, so it stays within "measure,
        don't judge". `count` is never overwritten; `estimates.outline` is the
        number the outline supports, with a `basis` string saying so. Set
        `vqa_estimate=true` to also ask Moondream2 "how many <name>?" and attach
        that judgment as `estimates.vqa` (clearly marked as an estimate, not a
        tally). Both are estimates: the detector could not separate the instances,
        so treat any number here accordingly.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            results = app.counter.detect_objects(images, object_name)
            for image, result in zip(images, results, strict=True):
                if verify_silhouette:
                    _add_silhouette(app, image, result)
                if consensus:
                    region_label_count = _region_label_consensus(app.processor, image, object_name)
                    result["consensus"] = {
                        "grounding_dino_count": result.get("count"),
                        "region_label_count": region_label_count,
                        "agree": result.get("count") == region_label_count,
                    }
                    result["separable"] = _separability(result)

                # When the detector collapsed overlapping instances but the outline
                # still carries the lobe pattern (by_radial > 1), surface that pattern
                # as an actionable estimate instead of only flagging the collapse. The
                # outline count is a *measurement* (angular notches), not a judgment, so
                # it stays within "measure, don't judge". `count` is never overwritten.
                if result.get("separable") == "no":
                    silhouette = result.get("silhouette") or {}
                    by_radial = silhouette.get("by_radial", 0)
                    if isinstance(by_radial, (int, float)) and by_radial > 1:
                        estimates: dict[str, Any] = {
                            "outline": int(by_radial),
                            "basis": (
                                "outline rosette (angular notches); the detector collapsed "
                                "overlapping instances, but the outline still carries the lobe "
                                "pattern"
                            ),
                        }
                        if vqa_estimate:
                            vqa_answer = app.vqa.query([image], f"How many {object_name}? Answer with one number.")[0]
                            estimates["vqa"] = {
                                "value": _first_int(vqa_answer),
                                "raw": vqa_answer,
                                "note": "a Moondream2 judgment estimate, not a tally",
                            }
                        result["estimates"] = estimates
            return results

    @mcp.tool()
    def batch_analyze_images(
        ctx: Context[AppContext],
        srcs: Annotated[
            list[os.PathLike[str] | str], Field(description="File paths or URLs of the images to process.")
        ],
        operation: Operation,
        question: Annotated[str, Field(description="Required when operation is 'query'.")] = "",
        object_name: Annotated[str, Field(description="Required when operation is 'detect' or 'count'.")] = "",
    ) -> list[dict[str, Any]]:
        """Run one operation across many images in a single call.

        The batch form of `caption`, `ocr`, `detect_objects`, `count_objects`,
        `dense_region_caption` and `query_image` -- pick which with `operation`. Use
        it when the same question applies to a whole set of images, since it costs
        one round trip instead of one per image and loads each model once for the
        whole run.

        Failures are isolated per image: a missing file or an unreachable URL is
        reported as its own {"src", "success": false, "error"} entry and the rest
        of the batch still runs. Results come back in the order given.

        For a single image, call the named tool directly -- its arguments are
        checked up front rather than depending on `operation`.
        """
        results = []
        for src in srcs:
            try:
                with get_images(src) as images:
                    result = _dispatch(
                        ctx.request_context.lifespan_context,
                        operation,
                        images,
                        question=question,
                        object_name=object_name,
                    )
                results.append({"src": str(src), "success": True, "result": result})
            except Exception as e:  # noqa: BLE001 - reported per-image, batch must continue
                results.append({"src": str(src), "success": False, "error": str(e)})
        return results

    @mcp.tool()
    def spatial_relations(
        ctx: Context[AppContext],
        src: ImagePath,
        objects: Annotated[
            list[str],
            Field(description="Names of the objects to locate and compare, e.g. ['hand', 'sword', 'shield']."),
        ],
    ) -> dict[str, Any]:
        """Measure how named objects in an image sit relative to one another.

        Locates each object, segments it, and reports measurements that are hard
        to judge by eye: whether two things actually touch, how many pixels apart
        they are, how much of one lies inside the other and how deeply, plus each
        object's own elongation, straightness and end-to-end width profile.

        This reports geometry, not verdicts — it does not decide what is wrong.
        Interpret the numbers against what the scene ought to look like: a hand
        and the grip it holds that come back `separate` with a large `gap` are not
        in contact; a hand `overlapping` a shield with `a_inside_b` near 1.0 and a
        large `embed_depth` is buried in the shield face rather than gripping its
        rim; an elongated object whose `end_symmetry` is near 1.0 is equally wide
        at both ends, unlike a blade that tapers to a point at one end only.

        Useful for checking whether a generated or edited image holds together
        physically, for verifying that an object is where it should be relative to
        another, and for any question of contact, containment or clearance that a
        bounding box cannot answer — boxes overlap whenever one object is simply
        in front of another.

        Takes the single best-scoring match per name, so this assumes one instance of
        each named object. Asked for 'red circle'/'blue circle'/'green circle' on a
        scene with one of each, the detector returned the same three boxes for every
        query — color alone doesn't reliably discriminate same-shaped objects — but the
        correctly-matching box scored highest every time, which is what this relies on.
        For several instances of one kind of thing, give them distinguishing names, or
        use `count_objects` for a tally instead.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]

            located: list[dict[str, Any]] = []
            for call_index, object_name in enumerate(objects):
                if len(located) >= _MAX_RELATED_OBJECTS:
                    break
                detected = app.counter.detect_objects([image], object_name)[0]
                if not detected["bboxes"]:
                    continue
                best = max(range(len(detected["bboxes"])), key=lambda i: detected["scores"][i])
                box = detected["bboxes"][best]
                located.append(
                    {"id": f"{object_name}#{call_index}", "label": object_name, "box": [int(v) for v in box]}
                )

            if not located:
                return {
                    "image_size": [image.width, image.height],
                    "objects": [],
                    "relations": [],
                    "note": "None of the requested objects were found.",
                }

            masks = app.segmenter.segment(image, [cast(list[int], obj["box"]) for obj in located])
            for obj, mask in zip(located, masks, strict=True):
                obj.update(geometry.describe(mask))

            relations = [
                {"a": located[i]["id"], "b": located[j]["id"], **geometry.relation(masks[i], masks[j])}
                for i in range(len(masks))
                for j in range(i + 1, len(masks))
            ]

            return {
                "image_size": [image.width, image.height],
                "mask_resolution": MASK_DECODE_RESOLUTION,
                "objects": located,
                "relations": relations,
            }

    @mcp.tool()
    def score_aesthetics(
        ctx: Context[AppContext],
        src: ImagePath,
        style_context: Annotated[
            bool,
            Field(
                description=(
                    "When true, also classify the image's medium/genre (photograph, oil "
                    "painting, digital illustration, ...) using the already-loaded CLIP "
                    "backbone, and return it alongside the score. The aesthetic head was "
                    "trained on photographs, so a non-photographic medium is the context the "
                    "score must be read in -- an oil painting scoring ~5.8 is not 'wrong'. "
                    "Default false keeps the original {score, rating} shape; true adds {style, "
                    "style_distribution} to each result."
                )
            ),
        ] = False,
        compare_with: Annotated[
            os.PathLike[str] | str | None,
            Field(
                description=(
                    "Path or URL of a reference image to compare against. The predictor's "
                    "documented valid use is like-with-like comparison (edits of one image, or "
                    "several shots of one subject), so this routes you there instead of a single "
                    "bias-affected absolute number: both images are scored and the result carries "
                    "the per-image scores, the `delta`, and `preferred` ('image'/'reference'/'tie', "
                    "tie when |delta| < 0.05). With style_context=true, both media are classified "
                    "and a `cross_medium_warning` is added when they differ (cross-medium "
                    "comparison is out of calibrated scope). Each image page is compared to the "
                    "first page of the reference. Omit (default) for the original single-image "
                    "scoring shape."
                )
            ),
        ] = None,
    ) -> list[dict[str, Any]]:
        """Rate how aesthetically pleasing an image looks, independent of its content.

        Uses a CLIP-based predictor trained on human aesthetic ratings (the LAION
        "improved aesthetic predictor"). Reflects visual qualities like lighting,
        composition and clarity — not whether the subject matter is correct or
        matches a prompt. A technically accurate but flatly-lit, cluttered photo can
        score low; a blurry but beautifully lit one can score comparatively higher.

        Returns one {"score": float, "rating": str} object per page/image. `score`
        is roughly on a 1-10 scale; `rating` buckets it coarsely for quick triage —
        read `score` for anything comparative.

        Its training set was photographic, so it rates photographs, not fine art:
        celebrated paintings and illustrations score middling (Hokusai's "The Great
        Wave" comes back around 5.8) without that meaning anything is wrong with
        them. Use it to compare like with like — several shots of the same subject,
        or successive edits of one image — and do not read a single absolute score
        as a verdict on quality.

        Set `style_context=true` to also get the medium the score is being read in.
        The CLIP backbone (already loaded for scoring) classifies the image as a
        photograph, oil painting, digital illustration, etc., and that `style` plus
        its `style_distribution` are added to each result. This is the local-model
        answer to the photography bias: it doesn't make the head understand fine art,
        but it tells you *that* the score is for a non-photographic medium, so you
        read it with the documented caveat instead of as an absolute verdict.

        Set `compare_with` to a reference image to switch to relative mode: both
        images are scored and the result becomes one entry per image page carrying
        `{image, reference, delta, preferred}`, with `preferred` being `"image"` /
        `"reference"` / `"tie"` (tie when |delta| < 0.05). This is the predictor's
        calibrated use -- like-with-like comparison -- so it sidesteps the absolute
        photography bias that makes a single score misleading across media. With
        `style_context=true`, both media are classified and a `cross_medium_warning`
        is added when they differ (cross-medium comparison is out of calibrated
        scope). The absolute score is not recalibrated; the relative delta is the
        actionable output.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            if compare_with is None:
                results = app.aesthetic.score(images)
                if not style_context:
                    return results
                styles = app.aesthetic.classify_style(images)
                for result, style in zip(results, styles, strict=True):
                    result["style"] = style["style"]
                    result["style_distribution"] = style["distribution"]
                return results

            # Relative mode: the predictor's documented valid use is like-with-like
            # comparison, so route callers there instead of a single bias-affected
            # absolute number. Score both, report the delta and which is preferred;
            # with style_context, warn when the two media differ.
            with get_images(compare_with) as ref_images:
                return _aesthetic_comparison(app, images, ref_images, style_context)

    @mcp.tool()
    def critique_composition(
        ctx: Context[AppContext],
        src: ImagePath,
        target_subject: Annotated[
            str,
            Field(description="Name of the main subject, e.g. 'the dog'. Omit to auto-detect it."),
        ] = "",
        low_score_threshold: Annotated[
            float, Field(description="Below this aesthetic score, ask Moondream2 to explain why.")
        ] = 5.0,
        style_context: Annotated[
            bool,
            Field(
                description=(
                    "When true, also classify the image's medium/genre (photograph, oil "
                    "painting, ...) using the already-loaded CLIP backbone and add it to the "
                    "result as `style` and `style_distribution`. The aesthetic head was trained "
                    "on photographs, so a non-photographic medium is the context the score is "
                    "read in. Default false omits the classification."
                )
            ),
        ] = False,
        compare_with: Annotated[
            os.PathLike[str] | str | None,
            Field(
                description=(
                    "Path or URL of a reference image to compare against. When set, both the "
                    "image and the reference are critiqued and the result becomes "
                    "{image, reference, delta, preferred} (plus cross_medium_warning when "
                    "style_context=true and the media differ). Same like-with-like framing as "
                    "score_aesthetics' compare_with. Omit (default) for the single-image critique."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Critique an image's composition: framing, and, for low-scoring images, why it looks off.

        Combines `score_aesthetics` (numeric quality), a rule-of-thirds/centeredness
        check on the main subject's bounding box, and — only when the aesthetic
        score is below `low_score_threshold` — a Moondream2 VQA explanation of what
        specifically looks unbalanced. Use it over `score_aesthetics` alone when you
        need to know *why* a shot is weak and where its subject sits in the frame,
        rather than just how it scores.

        Returns `image_size`, `subject_box`, `aesthetics`, and `framing` (with
        `thirds_offset` near 0 meaning the subject sits on a rule-of-thirds power
        point, and `center_offset` near 0 meaning it is dead-center instead).

        Pass `target_subject` whenever you know what the subject is — from your own
        context or a prior `caption` call. Auto-detection picks the largest,
        most central region and degrades on busy scenes that fill the frame, where
        no single region is the subject. If nothing can be located, returns a
        soft-failure shape (image size, aesthetic score, and a "note") rather than
        raising. Only the first page of a PDF is assessed.

        Set `style_context=true` to also get the image's medium (`style` plus
        `style_distribution`), so the aesthetic score is read in the context of its
        medium — the documented photography bias means a non-photographic medium
        should not be judged by the raw score.

        Set `compare_with` to a reference image to switch to relative mode: both
        images are critiqued and the result becomes
        `{image, reference, delta, preferred}` (tie when |delta| < 0.05), plus a
        `cross_medium_warning` when `style_context=true` and the two media differ.
        Same like-with-like framing as `score_aesthetics`' `compare_with`; the
        absolute score is not recalibrated, the relative delta is.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]
            if compare_with is None:
                return _critique_one(app, image, target_subject, low_score_threshold, style_context)

            # Relative mode: critique both the image and the reference, then report
            # the per-image critiques, the aesthetic delta, and which is preferred --
            # the same like-with-like framing as score_aesthetics' compare_with. The
            # reference's images must stay open while they are critiqued, so the
            # comparison runs inside the reference's context manager.
            with get_images(compare_with) as ref_images:
                ref = ref_images[0]
                img_result = _critique_one(app, image, target_subject, low_score_threshold, style_context)
                ref_result = _critique_one(app, ref, "", low_score_threshold, style_context)
                img_score = img_result.get("aesthetics", {}).get("score")
                ref_score = ref_result.get("aesthetics", {}).get("score")
                delta: float | None = None
                preferred = "tie"
                if img_score is not None and ref_score is not None:
                    delta = round(img_score - ref_score, 4)
                    if delta > _COMPARE_TIE:
                        preferred = "image"
                    elif delta < -_COMPARE_TIE:
                        preferred = "reference"
                comparison: dict[str, Any] = {
                    "image": img_result,
                    "reference": ref_result,
                    "delta": delta,
                    "preferred": preferred,
                }
                if (
                    style_context
                    and img_result.get("style") is not None
                    and ref_result.get("style") is not None
                    and img_result.get("style") != ref_result.get("style")
                ):
                    comparison["cross_medium_warning"] = (
                        "the image and reference are different media; the score is calibrated "
                        "for like-with-like comparison, so this delta is out of scope"
                    )
                return comparison

    @mcp.tool()
    def process(ctx: Context[AppContext], src: ImagePath, prompt: CustomPrompt) -> list[str]:
        """Run a raw Florence-2 task token against an image (escape hatch).

        `prompt` must be a Florence-2 task token, not an instruction: '<OD>',
        '<REGION_PROPOSAL>', '<OCR_WITH_REGION>' and the like. Passing plain
        English ("describe this image") does not fail — it returns confident
        nonsense, because the model has no such task and decodes the words as
        one anyway.

        Only for task tokens the named tools do not already cover. Prefer
        `caption`, `ocr`, `detect_objects` and `dense_region_caption`: they wrap
        the common tokens, parse the structured output into usable fields, and
        document where each one misleads. This returns raw text either way.
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.processor.generate(prompt, images)

    return mcp


#: Only a single detection triggers the silhouette check -- that is precisely the
#: answer `count_objects` documents as "could not separate them", so the segmentation
#: buys information where there currently is none. Any healthy count skips it, which
#: is what keeps SAM2 unloaded for sessions that never hit the collapse case.
_COLLAPSE_SUSPECT_COUNT: Final[int] = 1

#: A box smaller than this share of the frame is one small object, not a merged group.
_MIN_VERIFY_BOX_AREA: Final[float] = 0.01


def _add_silhouette(app: AppContext, image: Image, result: dict[str, Any]) -> None:
    """Attach a geometric second opinion when the detector collapsed to one region.

    Segments that single region and measures its outline, adding the result under
    `silhouette` without touching `count`. The two numbers are produced by different
    methods and the tool deliberately reports both rather than reconciling them.
    """
    if result.get("count") != _COLLAPSE_SUSPECT_COUNT or not result.get("bboxes"):
        return

    box = [int(v) for v in result["bboxes"][0]]
    width, height = max(box[2] - box[0], 0), max(box[3] - box[1], 0)
    if not image.width or not image.height:
        return
    if width * height < _MIN_VERIFY_BOX_AREA * image.width * image.height:
        return

    masks = app.segmenter.segment(image, [box])
    if not masks:
        return
    result["silhouette"] = {"box": box, **geometry.count_lobes(masks[0])}


def _region_label_consensus(processor: Processor, image: Image, object_name: str) -> int:
    """Second-opinion count: how many `dense_region_caption` labels match the object.

    Florence-2's dense region captioner emits a label per salient region ("a red car",
    "a person", "a yellow flower"). Tallying labels that contain the object name is an
    independent estimate of the instance count -- it comes from a different head than
    Grounding DINO, so agreement between the two is real evidence and disagreement is
    a visible warning. Returns 0 when the captioner finds nothing matching.
    """
    regions = processor.dense_region_caption([image])[0]
    needle = object_name.strip().lower()
    count = 0
    for label in regions.get("labels", []):
        if needle and needle in str(label).lower():
            count += 1
    return count


def _separability(result: dict[str, Any]) -> str:
    """Surfaces whether the reported `count` is a real tally or a structural collapse.

    ``"yes"`` -- the detector separated the instances (count > 1), or a single region
    whose silhouette both estimators (by_distance and the rosette-specific by_radial)
    agree is one lobe.
    ``"no"`` -- the detector collapsed while the silhouette's evidence indicates
    multiple lobes, or both estimators agree the count is > 1.
    ``"unknown"`` -- no silhouette check available, bad data (shattered/clipped), or
    the estimators disagree on a genuinely ambiguous shape with no strong signal either
    way (neither found multiple lobes).
    """
    count = result.get("count")
    if count is None:
        return "unknown"
    if count > 1:
        return "yes"
    # count == 1: only a silhouette block can tell a real singleton from a collapse.
    silhouette = result.get("silhouette")
    if not silhouette:
        return "unknown"
    if silhouette.get("shattered") or silhouette.get("clipped"):
        return "unknown"

    by_distance: int | None = silhouette.get("lobes")  # alias set by count_lobes
    by_radial: int = silhouette.get("by_radial") or 0  # rosette estimator, 0 = not measured
    agreement: bool = bool(silhouette.get("agreement"))

    if by_distance is None:
        return "unknown"

    # by_radial == 0: the rosette estimator was not applicable (not a rosette shape),
    # so by_distance is the only signal the geometry module can provide.
    if by_radial <= 0:
        return "yes" if by_distance <= 1 else "no"

    # by_radial > 0: the rosette estimator was applicable. Agreement is the key signal.
    if agreement:
        # Both estimators agree — the agreed count is trustworthy.
        return "yes" if by_distance <= 1 else "no"

    # Disagreement between the two estimators.
    # by_radial > 1 with by_distance == 1 is the canonical rosette collapse:
    # the distance estimator found no saddle (overlapping petals form one blob),
    # while the radial estimator caught the angular notch pattern (lobes < convex hull).
    # The documented flower case: by_distance=1, by_radial=8 → "no".
    if by_radial > 1:
        return "no"
    return "unknown"


# Short answers Moondream2 emits as a flat default regardless of the image -- the
# documented failure mode where it answers "None" to "describe anything wrong" on
# images that all had real visible defects. Seeing one of these agreed upon by two
# differently-phrased questions is the signature of a default, not an observation.
_DEFAULT_ANSWERS: Final[frozenset[str]] = frozenset(
    {"none", "nothing", "yes", "no", "n/a", "na", "i don't know", "unknown", "not sure", ""}
)


def _vqa_consistency(answer: str, control_answer: str) -> dict[str, Any]:
    """Compares an answer to its control-question answer and flags unreliable responses.

    `consistent` is true when the two answers agree -- exact (case/punctuation-stripped)
    match for short answers, or one containing the other for longer ones.

    `confidence` is `"low"` in two failure modes:

    1. *Agreed flat default* -- both answers reduce to a known default token ("none",
       "yes", "nothing", ...) and they agree: the model produced the same short answer
       to two differently-phrased questions without looking at the image. This is the
       documented signature from the six-image probe where Moondream answered "None"
       to "describe anything wrong" on images with real visible defects.

    2. *Self-contradiction* -- the two answers substantively disagree. One says
       something's wrong, the other says nothing's wrong, or they give opposite
       factual claims about the same image. A model contradicting itself across two
       rephrased questions is weaker evidence than either answer taken alone.
       (The documented "pette" probe: one answer said "missing a centerpiece" and the
       control said "Nothing is wrong" -- both substantive, both contradicting.)

    ``"normal"`` only when the answers are substantive AND not flat defaults AND
    they agree: real observations the consistency layer can confirm.
    """
    a = _normalize_answer(answer)
    c = _normalize_answer(control_answer)

    if a and c:
        consistent = a == c or a in c or c in a
    else:
        consistent = a == c

    both_default = a in _DEFAULT_ANSWERS and c in _DEFAULT_ANSWERS

    # "normal" only when the answers are substantive AND they agree.
    # Everything else is signal for a caller to distrust:
    #   - agreed flat defaults (both_default and consistent)  →  model is not looking
    #   - disagreement  (not consistent)                      →  model is contradicting itself
    confidence = "normal" if (consistent and not both_default) else "low"

    return {
        "answer": answer,
        "control_answer": control_answer,
        "consistent": consistent,
        "confidence": confidence,
    }


def _normalize_answer(text: str) -> str:
    """Lowercases, strips punctuation/whitespace, for default-token comparison."""
    return text.strip().lower().rstrip(".?!,;:")


def _first_int(text: str) -> int | None:
    """First integer appearing in a string, or None (Moondream may answer in prose)."""
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _vqa_cross_check(app: AppContext, image: Image, question_text: str) -> dict[str, Any] | None:
    """Route an unreliable VQA answer to the measurement that answers it.

    Classifies the question's wording and, when a measurable category applies and
    the object names can be parsed from that wording, runs the matching tool's
    underlying measurement and returns its result:

    - ``spatial`` -> ``spatial_relations`` (detect + segment both objects, measure
      their relation -- contact, gap, containment).
    - ``count`` -> ``count_objects`` (detect + count, with a separability flag).
    - ``ocr`` -> ``ocr`` (verbatim transcription, a second opinion from a different
      Florence-2 head).

    Returns None -- meaning "omit the cross-check" -- when no measurable category
    applies or the names can't be parsed to the required arity. It never guesses an
    object name, so a low-confidence "describe the mood" (no measurement) produces
    no cross-check rather than a fabricated one.
    """
    category = question.classify(question_text)
    if category is None:
        return None
    names = question.names_for(category, question_text)
    if names is None:
        return None

    if category == question.SPATIAL:
        return _spatial_measurement(app, image, cast(list[str], names))
    if category == question.COUNT:
        result = app.counter.detect_objects([image], names[0])[0]
        return {
            "tool": "count_objects",
            "object": names[0],
            "count": result.get("count"),
            "separable": _separability(result),
        }
    if category == question.OCR:
        return {"tool": "ocr", "text": app.processor.ocr([image])[0]}
    return None


def _spatial_measurement(app: AppContext, image: Image, names: list[str]) -> dict[str, Any] | None:
    """Detect and segment two named objects and measure their relation.

    Mirrors ``spatial_relations`` for the two-object case: detect each name, keep
    the best-scoring box, segment both, and return ``geometry.relation`` between
    the two masks. Returns None when fewer than two objects could be located, so the
    cross-check is omitted rather than reporting a half-measurement.
    """
    located: list[dict[str, Any]] = []
    boxes: list[list[int]] = []
    for call_index, object_name in enumerate(names):
        if len(located) >= _MAX_RELATED_OBJECTS:
            break
        detected = app.counter.detect_objects([image], object_name)[0]
        if not detected["bboxes"]:
            continue
        best = max(range(len(detected["bboxes"])), key=lambda i: detected["scores"][i])
        box = [int(v) for v in detected["bboxes"][best]]
        located.append({"label": object_name, "box": box})
        boxes.append(box)
    if len(located) < 2:
        return None

    masks = app.segmenter.segment(image, boxes)
    return {
        "tool": "spatial_relations",
        "objects": [obj["label"] for obj in located],
        "relation": geometry.relation(masks[0], masks[1]),
    }


#: Below this absolute delta, two scores are a tie (noise, not a preference).
_COMPARE_TIE: Final[float] = 0.05


def _aesthetic_comparison(
    app: AppContext, images: list[Image], ref_images: list[Image], style_context: bool
) -> list[dict[str, Any]]:
    """Score an image against a reference and report the relative preference.

    The predictor's documented valid use is like-with-like comparison, so this
    routes the caller to a calibrated relative answer instead of a single
    bias-affected absolute number. Each image page is compared to the first page of
    the reference. With ``style_context``, both media are classified and a
    ``cross_medium_warning`` is added when they differ (cross-medium comparison is
    out of calibrated scope). The absolute scores are not recalibrated.
    """
    img_scores = app.aesthetic.score(images)
    ref_score = app.aesthetic.score(ref_images)[0]

    img_styles = app.aesthetic.classify_style(images) if style_context else [None] * len(images)
    ref_style = app.aesthetic.classify_style(ref_images)[0] if style_context else None

    out: list[dict[str, Any]] = []
    for img, ist in zip(img_scores, img_styles, strict=True):
        delta = round(img["score"] - ref_score["score"], 4)
        if delta > _COMPARE_TIE:
            preferred = "image"
        elif delta < -_COMPARE_TIE:
            preferred = "reference"
        else:
            preferred = "tie"

        entry: dict[str, Any] = {
            "image": img,
            "reference": ref_score,
            "delta": delta,
            "preferred": preferred,
        }
        if style_context and ist is not None and ref_style is not None:
            img["style"] = ist["style"]
            img["style_distribution"] = ist["distribution"]
            ref_score["style"] = ref_style["style"]
            ref_score["style_distribution"] = ref_style["distribution"]
            if ist["style"] != ref_style["style"]:
                entry["cross_medium_warning"] = (
                    "the image and reference are different media; the score is calibrated "
                    "for like-with-like comparison, so this delta is out of scope"
                )
        out.append(entry)
    return out


def _critique_one(
    app: AppContext,
    image: Image,
    target_subject: str,
    low_score_threshold: float,
    style_context: bool,
) -> dict[str, Any]:
    """Single-image composition critique -- the body of `critique_composition`.

    Extracted so `critique_composition`'s relative mode can run it for both the
    image and the reference. See that tool's docstring for the return shape.
    """
    box: list[int] | None
    if target_subject:
        detected = app.processor.detect_objects([image], target_subject)[0]
        box = [int(v) for v in detected["bboxes"][0]] if detected["bboxes"] else None
    else:
        box = _pick_primary_subject(app.processor.dense_region_caption([image])[0], (image.width, image.height))

    aesthetics = app.aesthetic.score([image])[0]
    style: dict[str, Any] | None = None
    if style_context:
        style = app.aesthetic.classify_style([image])[0]

    if box is None:
        result: dict[str, Any] = {
            "image_size": [image.width, image.height],
            "aesthetics": aesthetics,
            "note": "Could not locate a subject to assess framing for.",
        }
        if style is not None:
            result["style"] = style["style"]
            result["style_distribution"] = style["distribution"]
        return result

    result = {
        "image_size": [image.width, image.height],
        "subject_box": box,
        "aesthetics": aesthetics,
        "framing": geometry.rule_of_thirds(box, (image.width, image.height)),
    }
    if style is not None:
        result["style"] = style["style"]
        result["style_distribution"] = style["distribution"]
    if aesthetics["score"] < low_score_threshold:
        result["critique"] = app.vqa.query(
            [image], "Critique this photo's composition and framing in 2 concise sentences."
        )[0]
    return result


def _dispatch(app: AppContext, operation: str, images: list[Image], *, question: str, object_name: str) -> Any:
    """Routes a `batch_analyze_images` operation to the right processor call."""
    if operation == "caption":
        return app.processor.caption(images, CaptionLevel.MORE_DETAILED)
    if operation == "ocr":
        return app.processor.ocr(images)
    if operation == "detect":
        if not object_name:
            raise ValueError("object_name is required for the 'detect' operation")
        return app.processor.detect_objects(images, object_name)
    if operation == "dense_caption":
        return app.processor.dense_region_caption(images)
    if operation == "query":
        if not question:
            raise ValueError("question is required for the 'query' operation")
        return app.vqa.query(images, question)
    if operation == "count":
        if not object_name:
            raise ValueError("object_name is required for the 'count' operation")
        # No silhouette check here: batching is the throughput path, and the check
        # would pull SAM2 in behind the caller's back once per collapsed image.
        return app.counter.detect_objects(images, object_name)
    raise ValueError(f"Unknown operation: {operation!r}")


def _pick_primary_subject(regions: dict[str, Any], image_size: tuple[int, int]) -> list[int] | None:
    """Picks the most prominent region from `dense_region_caption`'s output.

    Scores each box by area weighted toward the image center, since the most
    prominent subject in a photo is usually both large and roughly centered.
    Returns None if no regions were found.
    """
    boxes = regions.get("bboxes") or []
    if not boxes:
        return None

    width, height = image_size
    diagonal = math.hypot(width, height)

    def score(box: list[float]) -> float:
        x0, y0, x1, y1 = box
        area = max(x1 - x0, 0) * max(y1 - y0, 0)
        distance = math.hypot((x0 + x1) / 2 - width / 2, (y0 + y1) / 2 - height / 2)
        centrality = 1 - min(distance / diagonal, 1.0) if diagonal else 1.0
        return area * centrality

    return [int(v) for v in max(boxes, key=score)]


__all__: Final = [
    "DEFAULT_AESTHETIC_MODEL",
    "DEFAULT_GROUNDING_DINO_MODEL",
    "DEFAULT_MOONDREAM_MODEL",
    "DEFAULT_MOONDREAM_REVISION",
    "DEFAULT_SAM2_MODEL",
    "SERVER_NAME",
    "server",
]
