import logging
from typing import Final, Literal

import rich_click as click

from . import (
    CAPTION_MODELS,
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_FLORENCE2_MODEL,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_MOONDREAM_MODEL,
    DEFAULT_MOONDREAM_REVISION,
    DEFAULT_SAM2_MODEL,
    SERVER_NAME,
    VQA_MODELS,
    server,
)

#: Named presets for `--memory-mode`, in minutes of inactivity before a model is
#: released. "persistent" is 0 because a timeout of 0 never schedules a release,
#: which is exactly what it means.
MEMORY_MODE_MINUTES: Final[dict[str, float]] = {
    "aggressive": 5.0,
    "standard": 10.0,
    "persistent": 0.0,
}

DEFAULT_MEMORY_MODE: Final[str] = "standard"

#: Hosts the streamable-http server treats as local. The first is the default. The MCP SDK turns on
#: DNS-rebinding protection (Host and Origin checks) by itself only for these.
LOOPBACK_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")

#: An uncommon port, so the shared server doesn't collide with the 8000/8080 every dev server picks.
DEFAULT_HTTP_PORT: Final[int] = 8765


def resolve_memory_mode(memory_mode: str) -> float:
    """Turns a `--memory-mode` value into minutes of idle time before models are released.

    Accepts a preset name or a bare number of minutes, so the single option covers
    both the presets and the user-defined case. 0 means never release.
    """
    mode = memory_mode.strip().lower()
    if mode in MEMORY_MODE_MINUTES:
        return MEMORY_MODE_MINUTES[mode]

    try:
        minutes = float(mode)
    except ValueError:
        raise click.BadParameter(
            f"{memory_mode!r} is not a preset ({', '.join(MEMORY_MODE_MINUTES)}) or a number of minutes.",
            param_hint="--memory-mode",
        ) from None
    if minutes < 0:
        raise click.BadParameter("minutes cannot be negative.", param_hint="--memory-mode")
    # A user-defined 0 means the same thing "persistent" does: hold the models forever.
    return minutes


@click.command()
@click.option(
    "--model",
    default=DEFAULT_FLORENCE2_MODEL,
    show_default=True,
    help="Specifies the Florence-2 model to be used for caption/OCR/detection/grounding.",
)
@click.option(
    "--caption-model",
    type=click.Choice(CAPTION_MODELS),
    default=CAPTION_MODELS[0],
    show_default=True,
    help=(
        "Captioner behind `caption`'s default: 'lfm-onnx' (LFM2.5-VL-3B on ONNX Runtime, ~11 s, ~6.3 GB download, "
        "names more of an image) or 'florence2' (the --model Florence-2, ~7 s)."
    ),
)
@click.option(
    "--vqa-model",
    type=click.Choice(VQA_MODELS),
    default=VQA_MODELS[0],
    show_default=True,
    help=(
        "Answerer behind `query_image`: 'lfm-onnx' (the same LFM2.5-VL-3B ONNX model as the default caption, one "
        "loaded copy; ~5 s, more answers right) or 'moondream' (Moondream2, ~10 s). Moondream2 still backs "
        "critique_composition's hints and count_objects' estimate either way."
    ),
)
@click.option(
    "--moondream-model",
    default=DEFAULT_MOONDREAM_MODEL,
    show_default=True,
    help="Specifies the Moondream2 model used for the query_image (VQA) tool.",
)
@click.option(
    "--moondream-revision",
    default=DEFAULT_MOONDREAM_REVISION,
    show_default=True,
    help="Specifies the Moondream2 model revision used for the query_image (VQA) tool.",
)
@click.option(
    "--sam2-model",
    default=DEFAULT_SAM2_MODEL,
    show_default=True,
    help="Specifies the SAM2 model used for the spatial_relations tool.",
)
@click.option(
    "--aesthetic-model",
    default=DEFAULT_AESTHETIC_MODEL,
    show_default=True,
    help="Specifies the CLIP model used for the score_aesthetics and critique_composition tools.",
)
@click.option(
    "--grounding-dino-model",
    default=DEFAULT_GROUNDING_DINO_MODEL,
    show_default=True,
    help="Specifies the Grounding DINO model used for the count_objects tool.",
)
@click.option(
    "--reasoner-provider",
    default="none",
    show_default=True,
    help="Provider mode for the VisionReasoner backend ('none' or 'ollama').",
)
@click.option(
    "--reasoner-model",
    default="llama3",
    show_default=True,
    help="Model ID to request from the VisionReasoner backend (e.g. 'llama3' for ollama).",
)
@click.option(
    "--ocr-languages",
    multiple=True,
    default=("en",),
    show_default=True,
    help="OCR languages, repeatable (e.g. --ocr-languages en --ocr-languages de). English alone "
    "uses PP-OCRv6; any other set switches `ocr` to EasyOCR, which loads one recognition model "
    "per language.",
)
@click.option(
    "--memory-mode",
    default=DEFAULT_MEMORY_MODE,
    show_default=True,
    metavar="MODE",
    help=(
        "How long models stay in memory after their last use, trading memory against speed. "
        "'aggressive' releases after 5 minutes idle (lowest memory); 'standard' after 10 minutes; "
        "'persistent' never releases on a timer (fastest, highest memory; idle models still make way "
        "when memory runs short). Any number of minutes also works, "
        "e.g. '30' or '2.5'. Models always reload automatically on the next request, so no mode can "
        "lose work -- only time."
    ),
)
@click.option(
    "--idle-timeout",
    type=float,
    default=None,
    metavar="MINUTES",
    help=(
        "Deprecated alias for --memory-mode expressed in minutes; overrides it when given. "
        "0 keeps the models loaded for the lifetime of the server."
    ),
)
@click.option(
    "--memory-budget",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    metavar="GB",
    help=(
        "Cap on the total size of loaded models, in GB: before a load, the least recently used idle "
        "models are released to stay under it. Unset by default. Independently of this, idle models are "
        "always released when a load would not fit in free memory, and a load that cannot fit at all "
        "returns a tool error instead of crashing the server."
    ),
)
@click.option(
    "--device",
    default=None,
    help=(
        "Torch device all models load onto, e.g. 'cpu', 'cuda', 'cuda:1', 'mps'. "
        "Auto-detected (MPS, then CUDA, then CPU) when unset -- set this to pin the server to a "
        "specific accelerator, force CPU on a shared GPU box, or target a non-default GPU index."
    ),
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    show_default=True,
    help=(
        "'stdio' (the default) is one server per client: the client starts it and talks over a pipe. "
        "'streamable-http' runs one long-lived server that several MCP clients connect to by URL "
        "(http://HOST:PORT/mcp), so they share one copy of each model and one memory manager instead of "
        "each loading their own. Start it yourself (e.g. at logon); clients do not."
    ),
)
@click.option(
    "--host",
    default=LOOPBACK_HOSTS[0],
    show_default=True,
    help=(
        "Address the streamable-http server listens on. Keep it on loopback: there is no authentication, "
        "and anyone who can reach the port can make the server read any image file this user can read."
    ),
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=DEFAULT_HTTP_PORT,
    show_default=True,
    help="Port the streamable-http server listens on.",
)
@click.version_option()
def main(
    model: str,
    caption_model: str,
    vqa_model: str,
    moondream_model: str,
    moondream_revision: str,
    sam2_model: str,
    aesthetic_model: str,
    grounding_dino_model: str,
    reasoner_provider: Literal["none", "ollama"],
    reasoner_model: str,
    ocr_languages: tuple[str, ...],
    memory_mode: str,
    idle_timeout: float | None,
    memory_budget: float | None,
    device: str | None,
    transport: Literal["stdio", "streamable-http"],
    host: str,
    port: int,
) -> None:
    """
    An MCP server for processing images using Florence-2, Moondream2, SAM2, CLIP and Grounding DINO.
    """
    logger = logging.getLogger(__name__)

    idle_minutes = float(idle_timeout) if idle_timeout is not None else resolve_memory_mode(memory_mode)

    s = server(
        SERVER_NAME,
        model,
        moondream_model_id=moondream_model,
        moondream_revision=moondream_revision,
        sam2_model_id=sam2_model,
        aesthetic_model_id=aesthetic_model,
        grounding_dino_model_id=grounding_dino_model,
        reasoner_provider=reasoner_provider,
        reasoner_model=reasoner_model,
        ocr_languages=ocr_languages,
        idle_timeout=idle_minutes * 60,
        device=device,
        caption_model=caption_model,
        vqa_model=vqa_model,
        memory_budget_gb=memory_budget,
    )

    logger.info(f"Starting server with {model} + {moondream_model}@{moondream_revision} (Press CTRL+D to quit)")
    if idle_minutes > 0:
        logger.info(f"Models will be released after {idle_minutes:g} minutes of inactivity")
    else:
        logger.info("Models will stay loaded for the lifetime of the server, unless memory runs short")
    if memory_budget is not None:
        logger.info(f"Loaded models are kept under a {memory_budget:g} GB budget")
    if transport == "stdio":
        s.run()
    else:
        if host not in LOOPBACK_HOSTS:
            # The SDK also only turns DNS-rebinding protection on by itself for loopback hosts.
            logger.warning(
                f"Listening on {host}, not loopback, with no authentication: anyone who can reach port {port} "
                "can make this server read any image file this user can read."
            )
        logger.info(f"Serving MCP over streamable HTTP at http://{host}:{port}/mcp")
        s.run("streamable-http", host=host, port=port)
    logger.info("Server stopped")
