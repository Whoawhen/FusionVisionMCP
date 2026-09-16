"""Text tools: `ocr`, `caption` and the raw `process` escape hatch.

Split out of `server()`, which held all eleven tool definitions inline. Each module
here exposes `register(mcp)`; `tools.register_all` calls them in the order the tools
should appear to a client.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp import layout, textmatch
from fusion_vision_mcp.constants import CaptionLevel
from fusion_vision_mcp.images import get_images
from fusion_vision_mcp.ocr_fusion import fuse_caption_ocr
from fusion_vision_mcp.params import CustomPrompt, ImagePath
from fusion_vision_mcp.protocols import AppContext


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
                    "for each text span (via EasyOCR). Each page becomes a dict with `text_regions` "
                    "({text, confidence, box}[], box=[x1,y1,x2,y2] in page coordinates) and the "
                    "joined `text`. Use this when you need to know *where* a phrase is, or to "
                    "evaluate OCR confidence scores for anomaly detection."
                )
            ),
        ] = False,
    ) -> list[Any]:
        """Process an image file or URL using OCR to extract text.

        Uses EasyOCR for robust scene-text extraction. Excels at photos, signage,
        watermarks, logos, and printed text.

        A page laid out in side-by-side columns (a form, meeting notes, a
        resume) is detected automatically: each column is OCR'd separately and
        joined in reading order, so fields from different columns don't get
        interleaved.

        Set `detail=true` to get confidence scores and bounding boxes. This is
        highly recommended for checking generative image artifacts: if an image
        contains gibberish text, the confidence scores will drop significantly.
        """
        easyocr = ctx.request_context.lifespan_context.ocr_specialist
        with get_images(src) as images:
            per_page_columns = [layout.split_columns(image) for image in images]
            page_results: list[dict[str, Any]] = []
            flat_results: list[str] = []

            for image, columns in zip(images, per_page_columns, strict=True):
                # Reconstruct column offsets for coordinate mapping
                # Assuming horizontal splits, so y is always 0.
                x_offset = 0

                page_text_regions = []
                page_texts = []

                for crop in columns:
                    crop_w = crop.width
                    # run EasyOCR on the crop
                    crop_results = easyocr.readtext(crop)

                    for r in crop_results:
                        text = r["text"]
                        conf = r["confidence"]
                        box = r["box"]
                        # offset box [x1, y1, x2, y2]
                        box[0] += x_offset
                        box[2] += x_offset

                        page_texts.append(text)
                        page_text_regions.append({"text": text, "confidence": conf, "box": box})

                    x_offset += crop_w

                joined_text = "\n".join(page_texts)
                flat_results.append(joined_text)
                page_results.append({"text": joined_text, "text_regions": page_text_regions})

            return page_results if detail else flat_results

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
        auto_verify_text: Annotated[
            bool,
            Field(
                description=(
                    "When true, automatically detect embedded text, quotation, or signage tokens "
                    "in the caption or image, running specialist OCR (EasyOCR) with "
                    "small-text upscale crops to cross-check text verbatim (Spec 13 & 14). "
                    "Returns `caption_text_warning`, `text_consensus`, and `caption_corrected` "
                    "with consensus text fused across models."
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

        Set `auto_verify_text=true` to automatically detect when a caption quotes
        embedded text or signage and run specialist OCR (EasyOCR) on
        upscaled crops of the text regions, providing consensus metrics (`text_consensus`)
        and warnings (`caption_text_warning`).
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            captions = app.florence2.caption(images, CaptionLevel.MORE_DETAILED)
            if not verify_text and not auto_verify_text:
                return captions

            should_verify = verify_text
            if auto_verify_text and any(textmatch.contains_likely_text(c) for c in captions):
                should_verify = True

            if not should_verify:
                return captions

            regioned = app.florence2.ocr_with_regions(images)
            results = []
            for image, caption_text, page_regions in zip(images, captions, regioned, strict=True):
                text_regions = [
                    {"text": text, "box": [int(v) for v in box]}
                    for text, box in zip(page_regions.get("labels", []), page_regions.get("bboxes", []))
                ]
                fused = fuse_caption_ocr(
                    caption_text,
                    text_regions,
                    image,
                    specialist=app.ocr_specialist if auto_verify_text else None,
                )
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
