#  cli.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import logging
from typing import Final

import rich_click as click

from . import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_MOONDREAM_MODEL,
    DEFAULT_MOONDREAM_REVISION,
    DEFAULT_SAM2_MODEL,
    SERVER_NAME,
    server,
)

#: Named presets for `--memory-mode`, in minutes of inactivity before a model is
#: released. "instant" is not a duration -- it releases after every call -- and
#: "persistent" never releases, so both are handled separately in `resolve_memory_mode`.
MEMORY_MODE_MINUTES: Final[dict[str, float]] = {
    "aggressive": 5.0,
    "standard": 10.0,
}

DEFAULT_MEMORY_MODE: Final[str] = "standard"


def resolve_memory_mode(memory_mode: str) -> tuple[float, bool]:
    """Turns a `--memory-mode` value into (minutes of idle time, release after every call).

    Accepts a preset name or a bare number of minutes, so the single option covers
    both the four presets and the user-defined case.
    """
    mode = memory_mode.strip().lower()
    if mode == "instant":
        return 0.0, True
    if mode == "persistent":
        return 0.0, False
    if mode in MEMORY_MODE_MINUTES:
        return MEMORY_MODE_MINUTES[mode], False

    try:
        minutes = float(mode)
    except ValueError:
        raise click.BadParameter(
            f"{memory_mode!r} is not a preset (instant, aggressive, standard, persistent) or a number of minutes.",
            param_hint="--memory-mode",
        ) from None
    if minutes < 0:
        raise click.BadParameter("minutes cannot be negative.", param_hint="--memory-mode")
    # A user-defined 0 means the same thing the preset does: hold the models forever.
    return minutes, False


@click.command()
@click.option(
    "--model",
    default="florence-community/Florence-2-large",
    show_default=True,
    help="Specifies the Florence-2 model to be used for caption/OCR/detection/grounding.",
)
@click.option("--cache-model", is_flag=True, help="Keeps the model in VRAM for faster subsequent operations if set.")
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
    "--memory-mode",
    default=DEFAULT_MEMORY_MODE,
    show_default=True,
    metavar="MODE",
    help=(
        "How long models stay in memory after their last use, trading memory against speed. "
        "'instant' releases immediately after every call (lowest memory, slowest repeat calls); "
        "'aggressive' holds for 5 minutes; 'standard' holds for 10 minutes; 'persistent' never "
        "releases (fastest, highest memory). Any number of minutes also works, e.g. '30' or '2.5'. "
        "Models always reload automatically on the next request, so no mode can lose work -- only "
        "time. Implies --cache-model unless set to 'persistent'."
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
    "--device",
    default=None,
    help=(
        "Torch device all models load onto, e.g. 'cpu', 'cuda', 'cuda:1', 'mps'. "
        "Auto-detected (MPS, then CUDA, then CPU) when unset -- set this to pin the server to a "
        "specific accelerator, force CPU on a shared GPU box, or target a non-default GPU index."
    ),
)
@click.version_option()
def main(
    model: str,
    cache_model: bool,
    moondream_model: str,
    moondream_revision: str,
    sam2_model: str,
    aesthetic_model: str,
    memory_mode: str,
    idle_timeout: float | None,
    device: str | None,
) -> None:
    """
    An MCP server for processing images using Florence-2, Moondream2, SAM2 and CLIP.
    """
    logger = logging.getLogger(__name__)

    if idle_timeout is not None:
        idle_minutes, release_after_call = float(idle_timeout), False
    else:
        idle_minutes, release_after_call = resolve_memory_mode(memory_mode)

    s = server(
        SERVER_NAME,
        model,
        subprocess=not cache_model,
        moondream_model_id=moondream_model,
        moondream_revision=moondream_revision,
        sam2_model_id=sam2_model,
        aesthetic_model_id=aesthetic_model,
        idle_timeout=idle_minutes * 60,
        release_after_call=release_after_call,
        device=device,
    )

    logger.info(f"Starting server with {model} + {moondream_model}@{moondream_revision} (Press CTRL+D to quit)")
    if release_after_call:
        logger.info("Models will be released immediately after every call")
    elif idle_minutes > 0:
        logger.info(f"Models will be released after {idle_minutes:g} minutes of inactivity")
    else:
        logger.info("Models will stay loaded for the lifetime of the server")
    s.run()
    logger.info("Server stopped")
