"""Aesthetic tools: `score_aesthetics` and `critique_composition`."""

import os
from collections.abc import Sequence
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp.analysis import (
    _COMPARE_TIE,
    _aesthetic_comparison,
    _critique_one,
    _enrich_aesthetics,
)
from fusion_vision_mcp.images import get_images
from fusion_vision_mcp.params import ImagePath
from fusion_vision_mcp.protocols import AppContext


def register(mcp: MCPServer) -> None:
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

        In v0.8.0, the returned payload is enriched. The legacy `score` is duplicated
        as `photographic_aesthetic`. If a Reasoner is configured at the server level,
        an `artistic_judgment` is generated. If the server is installed with the `[iqa]`
        extra, a `technical_quality` score (0-100) is included via the MUSIQ ONNX model.

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
                styles: Sequence[dict[str, Any] | None]
                if style_context:
                    results, styles = app.aesthetic.score_and_classify(images)
                else:
                    results = app.aesthetic.score(images)
                    styles = [None] * len(images)
                for image, result, style in zip(images, results, styles, strict=True):
                    if style is not None:
                        result["style"] = style["style"]
                        result["style_distribution"] = style["distribution"]
                    _enrich_aesthetics(app, image, result, style)
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
