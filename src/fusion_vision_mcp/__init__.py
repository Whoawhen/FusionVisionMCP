"""FusionVisionMCP: local vision models behind eleven MCP tools.

This module is the entry point: it builds the server, owns the per-session model
lifetimes in `app_lifespan`, and delegates the tool definitions to
`fusion_vision_mcp.tools`. Loading images lives in `images`, the structural types in
`protocols`, and the shared cross-checks in `analysis`.

**This module must import without pulling in torch.** See CLAUDE.md -- every model
wrapper imports torch at its own module level, and paying that at import time cost
4.5s warm / 14.1s cold and produced real MCP connect timeouts. The wrappers are
imported inside their factories in `app_lifespan` instead, which run on first tool use.
"""

import importlib.util
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast

from mcp.server.mcpserver import MCPServer

from fusion_vision_mcp.constants import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_FLORENCE2_MODEL,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_MOONDREAM_MODEL,
    DEFAULT_MOONDREAM_REVISION,
    DEFAULT_SAM2_MODEL,
)
from fusion_vision_mcp.idle import IdleProxy, IdleReleased
from fusion_vision_mcp.images import SERVER_NAME, get_images
from fusion_vision_mcp.ocr_fusion import SpecialistOCR
from fusion_vision_mcp.tools import register_all

logger = logging.getLogger(__name__)

# These names are needed only for the quoted return annotations on the factories in
# `app_lifespan`; each is imported for real inside its own factory. The annotations must
# stay quoted -- this module has no `from __future__ import annotations`, so an unquoted
# name would be a NameError at `def` time (and ruff TC004 flags it).
if TYPE_CHECKING:
    from fusion_vision_mcp.aesthetic import Aesthetic
    from fusion_vision_mcp.easyocr_engine import EasyOCREngine
    from fusion_vision_mcp.florence2 import Florence2
    from fusion_vision_mcp.grounding_dino import GroundingDino
    from fusion_vision_mcp.image_quality import ImageQuality
    from fusion_vision_mcp.moondream import Moondream
    from fusion_vision_mcp.reasoner import VisionReasoner
    from fusion_vision_mcp.sam2 import Sam2


from fusion_vision_mcp.protocols import (
    AestheticScorer,
    AppContext,
    InstanceDetector,
    Processor,
    Segmenter,
    VqaProcessor,
)


@asynccontextmanager
async def app_lifespan(
    _server: MCPServer,
    model_id: str,
    moondream_model_id: str,
    moondream_revision: str,
    sam2_model_id: str = DEFAULT_SAM2_MODEL,
    aesthetic_model_id: str = DEFAULT_AESTHETIC_MODEL,
    grounding_dino_model_id: str = DEFAULT_GROUNDING_DINO_MODEL,
    reasoner_provider: Literal["none", "ollama"] = "none",
    reasoner_model: str = "llama3",
    ocr_languages: tuple[str, ...] = ("en",),
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
    vqa: VqaProcessor
    segmenter: Segmenter
    aesthetic: AestheticScorer

    # Each factory imports its own wrapper. `IdleReleased` calls these on first real
    # use, so the torch/transformers cost lands then rather than during startup -- the
    # lifespan runs before the MCP handshake is answered, so an import here would block
    # the connect exactly as a module-level one did.
    def _make_florence2() -> "Florence2":
        from fusion_vision_mcp.florence2 import Florence2

        return Florence2(model_id, device)

    def _make_moondream() -> "Moondream":
        from fusion_vision_mcp.moondream import Moondream

        return Moondream(moondream_model_id, moondream_revision, device)

    def _make_sam2() -> "Sam2":
        from fusion_vision_mcp.sam2 import Sam2

        return Sam2(sam2_model_id, device)

    def _make_aesthetic() -> "Aesthetic":
        from fusion_vision_mcp.aesthetic import Aesthetic

        return Aesthetic(aesthetic_model_id, device)

    def _make_grounding_dino() -> "GroundingDino":
        from fusion_vision_mcp.grounding_dino import GroundingDino

        return GroundingDino(grounding_dino_model_id, device)

    def _make_easyocr() -> "EasyOCREngine":
        from fusion_vision_mcp.easyocr_engine import EasyOCREngine

        return EasyOCREngine(device, list(ocr_languages))

    def _make_image_quality() -> "ImageQuality":
        from fusion_vision_mcp.image_quality import ImageQuality

        return ImageQuality()

    def _make_reasoner() -> "VisionReasoner":
        from fusion_vision_mcp.reasoner import VisionReasoner

        return VisionReasoner(provider=reasoner_provider, model=reasoner_model)

    if idle_timeout > 0:
        # Keep each model in this process so repeat calls stay fast, and let the
        # idle timer hand its memory back once the work stops.
        florence2 = cast(
            Processor,
            IdleProxy(IdleReleased(_make_florence2, idle_timeout, "Florence-2")),
        )
        vqa = cast(
            VqaProcessor,
            IdleProxy(
                IdleReleased(_make_moondream, idle_timeout, "Moondream")
            ),
        )
    else:
        # timeout 0 never schedules a release, so this is a persistent in-process model
        # that is still built lazily on first use.
        florence2 = cast(Processor, IdleProxy(IdleReleased(_make_florence2, 0, "Florence-2")))
        vqa = cast(VqaProcessor, IdleProxy(IdleReleased(_make_moondream, 0, "Moondream")))

    # Always lazy, on both paths. `IdleReleased` builds on first use and, with a
    # timeout of 0, simply never schedules a release — so a session that never
    # calls `spatial_relations` never pays for SAM2 at all.
    segmenter = cast(
        Segmenter,
        IdleProxy(IdleReleased(_make_sam2, idle_timeout, "SAM2")),
    )
    # Same rationale as SAM2: always idle-wrapped regardless of idle_timeout, since only
    # score_aesthetics/critique_composition pay for the CLIP backbone, and most sessions
    # never call either.
    aesthetic = cast(
        AestheticScorer,
        IdleProxy(IdleReleased(_make_aesthetic, idle_timeout, "Aesthetic")),
    )
    # Same rationale again: only count_objects loads Grounding DINO, so a session that
    # never counts never pays the ~690MB.
    counter = cast(
        InstanceDetector,
        IdleProxy(IdleReleased(_make_grounding_dino, idle_timeout, "Grounding DINO")),
    )
    # Specialist OCR is also lazy: only auto_verify_text loads EasyOCR.
    ocr_specialist = cast(
        SpecialistOCR,
        IdleProxy(
            IdleReleased(_make_easyocr, idle_timeout, "EasyOCR")
        ),
    )

    # `find_spec` answers "is onnxruntime installed?" without importing it, so the
    # optional [iqa] extra stays genuinely optional and genuinely lazy.
    iqa = None
    if importlib.util.find_spec("onnxruntime") is not None:
        iqa = IdleProxy(IdleReleased(_make_image_quality, idle_timeout, "ImageQuality"))

    reasoner = None
    if reasoner_provider != "none":
        reasoner = IdleProxy(IdleReleased(_make_reasoner, idle_timeout, "VisionReasoner"))

    yield AppContext(florence2, vqa, segmenter, aesthetic, counter, ocr_specialist, iqa, reasoner)


def server(
    name: str,
    model_id: str = DEFAULT_FLORENCE2_MODEL,
    moondream_model_id: str = DEFAULT_MOONDREAM_MODEL,
    moondream_revision: str = DEFAULT_MOONDREAM_REVISION,
    sam2_model_id: str = DEFAULT_SAM2_MODEL,
    aesthetic_model_id: str = DEFAULT_AESTHETIC_MODEL,
    grounding_dino_model_id: str = DEFAULT_GROUNDING_DINO_MODEL,
    reasoner_provider: Literal["none", "ollama"] = "none",
    reasoner_model: str = "llama3",
    ocr_languages: tuple[str, ...] = ("en",),
    idle_timeout: float = 0,
    device: str | None = None,
) -> MCPServer:
    """Creates a new FastMCP server instance with the specified name and model ID."""
    mcp = MCPServer(
        name,
        lifespan=partial(
            app_lifespan,
            model_id=model_id,
            moondream_model_id=moondream_model_id,
            moondream_revision=moondream_revision,
            sam2_model_id=sam2_model_id,
            aesthetic_model_id=aesthetic_model_id,
            grounding_dino_model_id=grounding_dino_model_id,
            reasoner_provider=reasoner_provider,
            reasoner_model=reasoner_model,
            ocr_languages=ocr_languages,
            idle_timeout=idle_timeout,
            device=device,
        ),
    )

    # Tool definitions live in `fusion_vision_mcp.tools`, grouped by domain. They
    # used to be declared inline here, which is most of why this module reached
    # 1,900 lines.
    register_all(mcp)

    return mcp


__all__: Final = [
    "DEFAULT_AESTHETIC_MODEL",
    "DEFAULT_GROUNDING_DINO_MODEL",
    "DEFAULT_MOONDREAM_MODEL",
    "DEFAULT_MOONDREAM_REVISION",
    "DEFAULT_SAM2_MODEL",
    "SERVER_NAME",
    "get_images",
    "server",
]
