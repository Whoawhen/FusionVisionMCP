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

from fusion_vision_mcp import geometry
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
    def ocr(ctx: Context[AppContext], src: ImagePath) -> list[str]:
        """Process an image file or URL using OCR to extract text.

        Best for dense, printed, document-style text (receipts, scanned pages,
        paragraphs). Do NOT use this for photo watermarks, logos, signage, or
        any cursive/stylized/low-contrast text — it confidently misreads that
        kind of text (fabricates plausible-looking wrong words) rather than
        failing visibly. Use `query_image` instead for that case, e.g. with
        the question "What does the text/watermark say, exactly?".
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.processor.ocr(images)

    @mcp.tool()
    def caption(ctx: Context[AppContext], src: ImagePath) -> list[str]:
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
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.processor.caption(images, CaptionLevel.MORE_DETAILED)

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
    ) -> list[str]:
        """Ask a free-form question about an image (visual question answering).

        This is the right tool for reading photo watermarks, logos, signage,
        or any cursive/stylized/low-contrast text — ask e.g. "What does the
        text/watermark say, exactly?". The `ocr` tool misreads that kind of
        text confidently; prefer this one for it instead.
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.vqa.query(images, question)

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

        Measured limits, worth checking before trusting a count:

        Heavy overlap still collapses the count. On eight identical shapes in a ring
        this counts 8 separated and 8 touching, but only 2 once they overlap by
        roughly two-thirds of their width. A count far below what you expect means
        "could not separate them", not a real tally.

        Silhouettes carry no interior structure. On a photo of a paper flower whose
        petals overlap, this returns 1 at every resolution from 128px to 768px -- as
        does every other detector tried. The flower's outline is 98% convex, so the
        petal boundaries exist only as interior colour edges that no detector or
        outline measurement in this server recovers. Ask `query_image` for a count in
        that situation and treat it as an estimate.

        When only one region is found, a `silhouette` block is added measuring how
        many repeated lobes that region's outline contains. `count` and
        `silhouette.lobes` come from different methods and **neither overrides the
        other**: `count: 1` beside `silhouette.lobes: 8` means the detector could not
        separate the instances while the outline shows eight cores. `agreement: true`
        means a second, independent estimator matched it; `shattered` or `clipped`
        mean the number should not be used at all.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            results = app.counter.detect_objects(images, object_name)
            if verify_silhouette:
                for image, result in zip(images, results, strict=True):
                    _add_silhouette(app, image, result)
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
        `dense_region_caption` and `query_image` -- pick which with `operation`.
        Use it when the same
        question applies to a whole set of images, since it costs one round trip
        instead of one per image and loads each model once for the whole run.

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
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]

            located: list[dict[str, Any]] = []
            for object_name in objects:
                detected = app.processor.detect_objects([image], object_name)[0]
                for index, box in enumerate(detected["bboxes"]):
                    if len(located) >= _MAX_RELATED_OBJECTS:
                        break
                    located.append({"id": f"{object_name}#{index}", "label": object_name, "box": [int(v) for v in box]})

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
    def score_aesthetics(ctx: Context[AppContext], src: ImagePath) -> list[dict[str, Any]]:
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
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.aesthetic.score(images)

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
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]

            box: list[int] | None
            if target_subject:
                detected = app.processor.detect_objects([image], target_subject)[0]
                box = [int(v) for v in detected["bboxes"][0]] if detected["bboxes"] else None
            else:
                box = _pick_primary_subject(app.processor.dense_region_caption([image])[0], (image.width, image.height))

            aesthetics = app.aesthetic.score([image])[0]

            if box is None:
                return {
                    "image_size": [image.width, image.height],
                    "aesthetics": aesthetics,
                    "note": "Could not locate a subject to assess framing for.",
                }

            result: dict[str, Any] = {
                "image_size": [image.width, image.height],
                "subject_box": box,
                "aesthetics": aesthetics,
                "framing": geometry.rule_of_thirds(box, (image.width, image.height)),
            }
            if aesthetics["score"] < low_score_threshold:
                result["critique"] = app.vqa.query(
                    [image], "Critique this photo's composition and framing in 2 concise sentences."
                )[0]
            return result

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
