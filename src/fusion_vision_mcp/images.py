"""Loading images from a path or URL, including multi-page PDFs.

Every tool takes its input through `get_images`, so this is the one place that knows a
PDF becomes one image per page and that a URL needs a descriptive User-Agent.
"""

import importlib.metadata
import os
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing, contextmanager
from io import BytesIO
from pathlib import Path
from typing import Final

import requests
from PIL.Image import Image
from PIL.Image import open as open_image
from pypdfium2 import PdfDocument

from fusion_vision_mcp.constants import REGION_MIN_SIDE, REGION_PAD

SERVER_NAME: Final[str] = "FusionVisionMCP"

#: (connect, read) seconds for outbound image/PDF fetches. Without a timeout a hung
#: host blocks the serving thread indefinitely and the MCP client sees a dead server.
_HTTP_TIMEOUT: Final[tuple[float, float]] = (5.0, 30.0)

#: `requests`' default User-Agent identifies the library, not a specific client, and
#: some hosts reject it outright as an anti-scraping measure -- confirmed live against
#: Wikimedia, which 403s the default UA but accepts a descriptive one.
_USER_AGENT: Final[str] = (
    f"{SERVER_NAME}/{importlib.metadata.version('fusion-vision-mcp')} (+https://github.com/Whoawhen/FusionVisionMCP)"
)


def crop_region(image: Image, region: Sequence[float]) -> Image:
    """The part of `image` inside `region` ([x1, y1, x2, y2] pixels), widened as VisionBench gate R measured it.

    Each side moves out by `REGION_PAD` of the box's size, a side shorter than `REGION_MIN_SIDE` grows to it, and the
    result is clipped to the image.
    """
    if len(region) != 4:
        raise ValueError(f"region must be [x1, y1, x2, y2] in pixels, got {list(region)}")
    x1, y1, x2, y2 = (float(v) for v in region)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"region needs x2 > x1 and y2 > y1, got {list(region)}")
    pw, ph = (x2 - x1) * REGION_PAD, (y2 - y1) * REGION_PAD
    x1, x2, y1, y2 = x1 - pw, x2 + pw, y1 - ph, y2 + ph
    if x2 - x1 < REGION_MIN_SIDE:
        cx = (x1 + x2) / 2
        x1, x2 = cx - REGION_MIN_SIDE / 2, cx + REGION_MIN_SIDE / 2
    if y2 - y1 < REGION_MIN_SIDE:
        cy = (y1 + y2) / 2
        y1, y2 = cy - REGION_MIN_SIDE / 2, cy + REGION_MIN_SIDE / 2
    box = (max(0, round(x1)), max(0, round(y1)), min(image.width, round(x2)), min(image.height, round(y2)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"region {list(region)} lies outside the {image.width}x{image.height} image")
    return image.convert("RGB").crop(box)


@contextmanager
def get_images(src: os.PathLike[str] | str) -> Iterator[list[Image]]:
    """Opens and returns a list of images from a file path or URL."""
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        res = requests.get(src, headers={"User-Agent": _USER_AGENT}, timeout=_HTTP_TIMEOUT)
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
