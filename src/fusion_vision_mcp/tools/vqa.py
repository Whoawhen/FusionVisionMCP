"""VQA tools: `query_image` and the `batch_analyze_images` fan-out."""

import logging
import os
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp.analysis import _dispatch, _vqa_consistency, _vqa_cross_check
from fusion_vision_mcp.forensics import analyze_metadata_anomalies
from fusion_vision_mcp.images import get_images
from fusion_vision_mcp.inspection import Anomaly, Observation, analyze_inspection
from fusion_vision_mcp.params import ImagePath
from fusion_vision_mcp.protocols import AppContext

logger = logging.getLogger(__name__)

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


def register(mcp: MCPServer) -> None:
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
        structured_analysis: Annotated[
            bool,
            Field(
                description=(
                    "When true, bypasses the simple VQA return and returns structured Observation and "
                    "Anomaly payloads per image. Corroborates VLM claims against physical Grounding DINO measurements "
                    "to proactively detect anatomical and structural AI generation artifacts (e.g. extra/missing body parts)."
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
        text-reading question, `detect_objects` for "which is largest/smallest")
        and attaches it as `cross_check`. The "largest/smallest" case detects every
        instance of the named object and picks the extremum by bounding-box area,
        returning its box -- so "which circle is biggest" resolves to coordinates,
        not a repeated guess. The cross-check is omitted when no measurement
        applies or the names can't be parsed -- it never guesses.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            answers = app.vqa.query(images, question)

            if structured_analysis:
                results = []
                for image, answer in zip(images, answers, strict=True):
                    observations, anomalies = analyze_inspection(app, image, [answer])

                    # Metadata/EXIF forensics. A hit is strong evidence; an empty
                    # result means nothing either way -- see the function's docstring.
                    meta_anomalies = analyze_metadata_anomalies(image)
                    for ma in meta_anomalies:
                        obs = Observation(
                            claim=ma["claim"], source=ma["source"], corroborated=False, evidence=[ma["evidence"]]
                        )
                        anomalies.append(
                            Anomaly(description="Generative metadata signature detected.", observations=[obs])
                        )

                    # Spec 20: Auto-check for generative text hallucinations
                    try:
                        easyocr = app.ocr_specialist
                        crop_results = easyocr.readtext(image)
                        for r in crop_results:
                            if r["confidence"] < 0.20:
                                obs = Observation(
                                    claim=f"Text span resembles valid language: '{r['text']}'",
                                    source="EasyOCR",
                                    corroborated=False,
                                    evidence=[
                                        f"Confidence score {r['confidence']:.2f} is abnormally low indicating structural hallucination.",
                                        f"Box: {r['box']}",
                                    ],
                                )
                                anom = Anomaly(
                                    description=f"Generative text hallucination detected (gibberish/malformed): '{r['text']}'",
                                    observations=[obs],
                                )
                                anomalies.append(anom)
                    except Exception as e:  # noqa: BLE001 - third-party OCR; logged, never fatal
                        logger.warning(f"Failed to run EasyOCR text hallucination check: {e}")

                    results.append(
                        {
                            "answer": answer,
                            "observations": [o.as_dict() for o in observations],
                            "anomalies": [a.as_dict() for a in anomalies],
                        }
                    )
                return results

            if not check_consistency:
                return answers

            # A rephrasing a genuinely-looking model answers with the same substance,
            # but a model defaulting to a flat answer returns the same short default to.
            control_question = f"Looking carefully at this image, answer precisely: {question}"
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
