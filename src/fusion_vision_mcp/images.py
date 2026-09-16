"""Loading images from a path or URL, including multi-page PDFs.

Every tool takes its input through `get_images`, so this is the one place that knows a
PDF becomes one image per page and that a URL needs a descriptive User-Agent.
"""

import importlib.metadata
import os
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from io import BytesIO
from pathlib import Path
from typing import Final

import requests
from PIL.Image import Image
from PIL.Image import open as open_image
from pypdfium2 import PdfDocument

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
