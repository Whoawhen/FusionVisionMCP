"""Text tools: `ocr`, `caption` and the raw `process` escape hatch.

Split out of `server()`, which held all eleven tool definitions inline. Each module
here exposes `register(mcp)`; `tools.register_all` calls them in the order the tools
should appear to a client.
"""

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp.analysis import deep_crops, ocr_pages, visible_text_line
from fusion_vision_mcp.constants import CaptionLevel
from fusion_vision_mcp.images import crop_region, get_images
from fusion_vision_mcp.ocr_fusion import fuse_caption_ocr
from fusion_vision_mcp.params import CustomPrompt, ImagePath
from fusion_vision_mcp.protocols import AppContext
from fusion_vision_mcp.textmatch import repair_replacement_chars

logger = logging.getLogger(__name__)


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def ocr(
        ctx: Context[AppContext],
        src: ImagePath,
        detail: Annotated[
            bool,
            Field(
                description=(
                    "When true, return verbatim text *and* the bounding box and confidence score "
                    "for each text span. Each page becomes a dict with `text_regions` "
                    "({text, confidence, box}[], box=[x1,y1,x2,y2] in page coordinates) and the "
                    "joined `text`. Use this when you need to know *where* a phrase is. Spans below "
                    "0.75 confidence are already dropped, so this is not a gibberish detector -- use "
                    "query_image(structured_analysis=true) for that."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Process an image file or URL using OCR to extract text.

        Uses PaddleOCR's PP-OCRv6 English models. Reads documents, tables and
        forms closely (prices, quantities and punctuation included) and handles
        scene text in photos -- on 200 TextVQA photos, the answer's text appeared
        in its output for 58.5% of images. Small, angled or distant text in a busy
        photo is still the weak spot. A page sent sideways is read turned upright. Servers started with non-English
        `--ocr-languages` use EasyOCR here instead.

        This is the right tool for *any* text in an image: dense printed text
        and stylized, logo or low-contrast text alike. `query_image` paraphrases
        rather than transcribing, so it is not a substitute.

        A page laid out in side-by-side columns (a form, meeting notes, a
        resume) is detected automatically: each column is OCR'd separately and
        joined in reading order, so fields from different columns don't get
        interleaved.

        An image with no text returns an empty result, and that is the correct
        answer rather than a failure -- do not re-ask a captioning or VQA model
        to produce text for it, which is how invented text gets into a payload.
        On a busy photo, though, an empty or partial result can also mean small
        text was missed, so treat it as weaker evidence there than on a document.

        Text read with less than 0.75 confidence is dropped rather than returned,
        so a line the model cannot read comes back missing, not garbled: on a
        small-text test paragraph the last line is omitted while the two above it
        are verbatim. Set `detail=true` for per-span boxes and confidence scores.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            pages = ocr_pages(app, images)
            return pages if detail else [page["text"] for page in pages]

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
                    "returns one dict per page ({caption, text_regions, corrections, caption_corrected, "
                    "caption_text_warning, text_consensus})."
                )
            ),
        ] = False,
        deep: Annotated[
            bool,
            Field(
                description=(
                    "When true, ask the same model (LFM2.5-VL-3B) for a longer, fuller description, in "
                    "bf16 with up to 4 tiles: about 3x slower (~35s vs ~11s per image for the caption itself "
                    "on CPU) and about 50% longer (142 vs 94 words). It adds detail rather than accuracy per "
                    "word: key facts 74.8% vs 71.0% for the default on 280 held-out images, but no better "
                    "when both are cut to the same length (-1.3), and it names absent objects more often "
                    "(COCO: 16.5% of the objects it names, against about 11-12% for the default). Use it when the default caption is too thin for what you need; "
                    "for one detail, `region` or `query_image` is cheaper. The return shape is unchanged, "
                    "and `verify_text` still applies."
                )
            ),
        ] = False,
        read_text: Annotated[
            bool,
            Field(
                description=(
                    "When true (the default), also read the image with the `ocr` engine and end the "
                    'caption with `Visible text: "..."`: the verbatim text in reading order, lines '
                    "separated by ' / ', repeats and single characters dropped, at most 400 words; a "
                    "longer text says how many words were left out (`ocr` returns them all). The "
                    "captioners cannot read app and game "
                    "titles, document headings or labels; this gives them to the caption. An image "
                    "with no text gets no line, so its caption is unchanged. Adds about 1-2s per image. "
                    "Set false for the bare caption."
                )
            ),
        ] = True,
        region: Annotated[
            list[float] | None,
            Field(
                description=(
                    "Optional [x1, y1, x2, y2] in the image's pixels: caption only that part, so it gets the "
                    "captioner's whole input resolution (the default caption sees about 512 px, so small text "
                    "and HUD detail on a large screenshot are a few pixels). The box is widened by 15% on each "
                    "side, to at least 32 px, and clipped to the image. Pointed at the part a question is about, "
                    "the caption named the answer 80% of the time against 68% for the whole image (528 "
                    "questions, measured with Florence-2 as the captioner). The caption then describes the "
                    "region, not the image; applies to every page of a PDF."
                )
            ),
        ] = None,
    ) -> list[Any]:
        """Describe what an image shows, as one detailed prose caption.

        The default choice for "what is this a picture of". Returns a single
        paragraph covering the scene as a whole, with no coordinates.

        Since 2026-10-23 the caption comes from LFM2.5-VL-3B (Liquid AI) on
        ONNX Runtime, asked to describe only what is clearly visible. On 280
        held-out mixed images (photos, art, screenshots, documents, charts,
        game UI, signs, illustrations, AI images), with the text line, it named
        71.0% of the labelled key facts against 64.8% for Florence-2-large, the
        captioner before it (+9.1 at equal length), ahead or level on every
        kind, and it names absent objects less often (COCO: 12.4% of the
        objects it names vs 13.3%). It is slower: about 11 s for the caption on
        CPU against Florence-2's ~7 s, ~14 s with the text line. A server
        started with `--caption-model florence2` uses Florence-2 instead, as
        does any server that cannot load the ONNX model.

        Set `deep=true` when the default caption is too thin for what you need:
        the same model, longer and fuller (142 vs 94 words), at ~35 s. It names
        more key facts (74.8% vs 71.0%) because it says more, not because it is
        sharper per word (-1.3 at equal length), and it names absent objects
        more often (COCO: 16.5% of the objects it names, against about
        11-12% for the default).

        By default the caption ends with `Visible text: "..."`, the `ocr`
        engine's verbatim reading of any text in the image (`read_text`). On
        the same 280 images that took the default caption from 53.3% to 60.9%
        of key facts and the deep caption from 61.5% to 66.3%, mostly app and
        game titles, document headings and labels the captioners cannot read;
        no image scored worse, and a text-free image gets no line at all. The
        quoted line is the reliable text; the prose before it is not. It holds
        up to 400 words, skips repeated and single-character lines, and when it
        is cut says how many words it left out (call `ocr` for all of them): on
        150 text-heavy screenshots, documents, charts, game UI and signs that
        took key facts named from 48.1% to 60.0% at 160 words against the earlier
        80-word line (deep 54.3% to 63.0%), with no image worse, and 400 words
        took the default caption from 62.5% to 67.8% against 160 (2026-10-24).
        Most lines are far shorter: about 117 words on a text-heavy page.

        Reach for a different tool when the question is narrower: `query_image`
        to ask something specific about the image, `dense_region_caption` to get
        a separate caption and box for each thing in it, and `ocr` to transcribe
        text rather than describe it. Returns one caption per page for a PDF.

        Do not trust any text the prose quotes back. A caption that mentions a name,
        brand or label is describing it, not transcribing it, and captioners
        misspell text that `ocr` reads correctly -- Florence-2 rendered a logo
        reading "FusionVisionMCP" as "FusionVisionMP" mid-caption while `ocr`
        read the same image exactly. When a specific piece of text matters,
        confirm it with `ocr`, which handles stylized and cursive text as well as
        printed text, rather than quoting this.

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
        (`{caption, text_regions, corrections, caption_corrected, ...}`) when this
        is set. The correction is best-effort and only fires for high-similarity
        same-word matches; the raw `corrections` list is always present so a
        caller can audit or reject any change.

        Pass `region=[x1, y1, x2, y2]` to caption one part of the image at full
        resolution, for small text, a HUD element, or the object a question is
        about. Measured on 528 questions about a region (VisionBench gate R):
        the caption of the region named the answer 80.3% of the time against
        67.8% for the whole image, and colours 86% against 69%. `region` is not
        offered on `ocr` or `query_image`: cropping made `ocr` worse (tight
        crops cut words at the edge) and left `query_image` unchanged.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as pages:
            images = [crop_region(page, region) for page in pages] if region is not None else pages
            # Moondream2's `deep` (the fallback) reads the text first: a text-heavy page keeps 12 crops (DEEP_CROPS).
            moondream_deep = deep and app.deep_captioner is None
            page_texts = (
                [page["text"] for page in ocr_pages(app, images)] if read_text or moondream_deep else [""] * len(images)
            )
            if deep and app.deep_captioner is not None:
                captions = app.deep_captioner.caption(images)
            elif deep:
                crops = [deep_crops(t) for t in page_texts]
                captions = app.vqa.caption(images, length="long", max_crops=crops)
            elif app.captioner is not None:
                try:
                    captions = app.captioner.caption(images)
                except (OSError, ImportError) as exc:
                    # No network for the first download, or no onnxruntime: the caption before 2026-10-23.
                    logger.warning(f"LFM2.5-VL ONNX unavailable ({exc}); caption falls back to Florence-2")
                    captions = app.florence2.caption(images, CaptionLevel.MORE_DETAILED)
            else:
                captions = app.florence2.caption(images, CaptionLevel.MORE_DETAILED)
            texts = page_texts if read_text else [""] * len(captions)
            # A captioner can emit U+FFFD for a split multi-byte character; mend it from the page's text.
            captions = [repair_replacement_chars(c, t) for c, t in zip(captions, texts, strict=True)]
            # Appended after any verify step, so `verify_text` checks only the prose's own
            # quotes, never the verbatim line.
            lines = [visible_text_line(t) for t in texts] if read_text else [""] * len(captions)

            def with_text(caption_text: str, line: str) -> str:
                return f"{caption_text.rstrip()} {line}" if line else caption_text

            if not verify_text:
                return [with_text(c, line) for c, line in zip(captions, lines, strict=True)]

            regioned = app.florence2.ocr_with_regions(images)
            results = []
            for caption_text, page_regions, line in zip(captions, regioned, lines, strict=True):
                text_regions = [
                    {"text": text, "box": [int(v) for v in box]}
                    for text, box in zip(page_regions.get("labels", []), page_regions.get("bboxes", []))
                ]
                fused = fuse_caption_ocr(caption_text, text_regions)
                for key in ("caption", "caption_corrected"):
                    if isinstance(fused.get(key), str):
                        fused[key] = with_text(fused[key], line)
                results.append(fused)
            return results

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
            return ctx.request_context.lifespan_context.florence2.generate(prompt, images)
