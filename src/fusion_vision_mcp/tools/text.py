"""Text tools: `ocr`, `caption` and the raw `process` escape hatch.

Split out of `server()`, which held all eleven tool definitions inline. Each module
here exposes `register(mcp)`; `tools.register_all` calls them in the order the tools
should appear to a client.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp import textmatch
from fusion_vision_mcp.analysis import ocr_pages
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
        auto_verify_text: Annotated[
            bool,
            Field(
                description=(
                    "When true, run the `verify_text` check only if the caption's wording suggests "
                    "it quotes text (signage, labels, quotation marks), and additionally cross-check "
                    "each text region with the `ocr` engine on an upscaled crop, which is what fills in "
                    "`text_consensus` and `caption_text_warning`. When the caption "
                    "does not look like it quotes text, nothing extra runs and the plain list[str] "
                    "is returned -- so check the return shape rather than assuming dicts."
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
        logo reading "FusionVisionMCP" as "FusionVisionMP" mid-caption while `ocr`
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

        Set `auto_verify_text=true` to pay for that check only when it is needed:
        it fires when the caption's wording suggests quoted text, and adds a
        cross-check by the `ocr` engine on upscaled crops (`text_consensus`,
        `caption_text_warning`). The trigger reads the caption, not the image, so
        text the caption never mentions is not checked; when it does not fire the
        return stays the plain list[str].
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
                    specialist=app.ocr_engine if auto_verify_text else None,
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
