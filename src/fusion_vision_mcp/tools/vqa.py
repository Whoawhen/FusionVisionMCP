"""VQA tools: `query_image`, `classify_image` (multiple choice) and the `batch_analyze_images` fan-out."""

import logging
import os
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp.analysis import (
    _dispatch,
    _vqa_consistency,
    _vqa_cross_check,
    classify_pages,
    clean_labels,
    vqa_answers,
)
from fusion_vision_mcp.forensics import analyze_metadata_anomalies
from fusion_vision_mcp.images import get_images
from fusion_vision_mcp.inspection import Anomaly, Observation, analyze_inspection
from fusion_vision_mcp.params import ImagePath, Operation
from fusion_vision_mcp.protocols import AppContext

logger = logging.getLogger(__name__)


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
                    "two answers agree. `confidence` is 'low' in two cases: the answers agree "
                    "on a short default-looking token ('None', 'Yes', 'No', 'Nothing', ...), "
                    "or they substantively contradict each other. Small VLMs answer open-ended "
                    "judgment questions ('describe anything wrong') with a flat default (Moondream2, "
                    "the answerer before 2026-10-24, said 'None' on images that all had real visible "
                    "defects) -- this layer makes that visible instead of presenting it as reliable. "
                    "It has not been re-measured with LFM2.5-VL, today's answerer. "
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
                    "When true, return structured `observations` and `anomalies` per image instead of "
                    "a plain answer, as a check for AI-generation defects. Three checks raise anomalies: "
                    "a learned defect probe that scores every detected hand and face crop with DINOv2 "
                    "('Likely generation defect: ...', the main signal: it can see a fused or misshapen "
                    "hand), generator signatures in metadata, and low-confidence text as likely "
                    "generated gibberish. On 299 AI images from six generators it never trained on, the "
                    "probe alone caught 45% of images with a defect and flagged 16% of clean ones: an "
                    "anomaly is a lead to check, not a verdict, and no anomaly does not mean the image is "
                    "clean. It only looks at hands and faces, so an extra arm or a fused body is usually "
                    "missed. Every probe crop's score is listed in `observations` (source "
                    "'defect_probe'). A per-person body-part count also reports any person with too many "
                    "limbs as an observation (source 'detector'), not an anomaly: on clean images it is "
                    "usually a detection error. Loads DINOv2-large (~1.2GB) when a hand or face is found."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Ask a free-form question about an image (visual question answering).

        Answers come back as free text, often a sentence rather than a single
        word, and the model paraphrases. To read text in the image -- signage,
        logos, watermarks, stylized or cursive lettering included -- use `ocr`,
        which transcribes verbatim and returns an empty result when there is
        no text, rather than asking this tool what the text says.

        Answered by LFM2.5-VL-3B (Liquid AI) on ONNX Runtime since 2026-10-24, the model behind
        `caption`'s default, replacing Moondream2: on 892 fresh questions it answered 78.5% right
        against Moondream2's 66.5%, ahead or level on every kind (text, colour, object, count, yes/no),
        in a median 4.9 s against 9.8 s. A question that mentions text (written, says, label, brand
        name, ...) gets the `ocr` engine's reading of the image as context. It is also much less
        prone to going along with a false premise: asked about text or an object that is not there,
        it said so 66% of the time against Moondream2's 32% -- which still means it invents an answer
        about a third of the time. An answer is not a measurement: for "how many" use
        `count_objects`, for touching/inside `spatial_relations`.

        Small VLMs also answer open-ended judgment questions ("describe anything wrong in this
        image") with a flat default; Moondream2 said "None" on images that all had real visible
        defects. Set `check_consistency=true` to make that visible: the tool also
        asks a rephrased control question and returns, per image,
        `{answer, control_answer, consistent, confidence}`. `confidence` is `"low"`
        in either of two failure modes: the two answers agree on a short
        default-looking token ("None", "Yes", "Nothing", ...), which is the
        signature of a flat default rather than a genuine observation; or they
        substantively contradict each other, which makes either answer weaker
        evidence than it looks alone. `"normal"` requires substantive answers that
        agree. A `low` result on a judgment question means you should not trust the
        answer without independent confirmation. When the answer is
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
            answers = vqa_answers(app, images, question)

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
            control_answers = vqa_answers(app, images, control_question)

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
    def classify_image(
        ctx: Context[AppContext],
        src: ImagePath,
        labels: Annotated[
            list[str],
            Field(
                description=(
                    "The candidate answers to choose between, as short names: species ('red fox', 'coyote'), "
                    "landmarks, art styles, kinds of image, product types. 2 to 200; each is put in "
                    "'a photo of {label}'. The pick is always one of them, so include a catch-all such as "
                    "'something else' if none may fit."
                )
            ),
        ],
    ) -> list[dict[str, Any]]:
        """Pick which of your labels an image shows, with a probability for each (CLIP zero-shot).

        Use it whenever you can list the possible answers: which species, which landmark, which art
        style, photo or screenshot or chart. On 200 fresh species, landmark and art-style images it
        picked the right label 98.5% of the time against `query_image`'s 77.5% asked the same
        multiple-choice question, at 0.25 s per image against 10 s (VisionBench gate Z2,
        2026-10-16); on everyday COCO objects and image kinds the margin was smaller, 79.9% vs 74.1%
        (gate Z). Those images were well-known Commons files that CLIP may have seen in training, so
        expect lower on your own photos. Its probability is usable: right 197 of 198 times at 0.5 or
        more in gate Z2, and 190 of 230 against 29 of 44 below 0.5 in gate Z.

        It cannot answer an open question ("what is this?"): it only ranks the labels it is given,
        and always returns one of them, even when none is in the image. Use `caption` or
        `query_image` for that. Art styles were its weakest set (95%).

        Returns one dict per page/image: `label` (the top pick), its `probability`, and `ranking`,
        every label with its probability, highest first (they sum to 1).
        """
        app = ctx.request_context.lifespan_context
        names = clean_labels(labels)
        with get_images(src) as images:
            return classify_pages(app, images, names)

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

        Batch results are the raw model output, not the full named-tool payload:
        'count' returns the detector's tally without the `silhouette`, `consensus`
        or `separable` fields `count_objects` adds, and none of the opt-in flags
        (`verify_text`, `check_consistency`, ...) are available here.

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
