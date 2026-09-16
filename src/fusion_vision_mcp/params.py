"""Annotated parameter types shared by the MCP tools.

These carry the `Field(description=...)` text a calling agent reads at tool-selection
time, so they are part of the server's public contract, not decoration. They lived as
locals inside `server()` until the tools were split into their own modules.
"""

import os
from typing import Annotated

from pydantic import Field

ImagePath = Annotated[
    os.PathLike[str] | str,
    Field(
        description=(
            "Local file path or http(s) URL of the image to process. PDFs are also accepted "
            "and are rendered one image per page, so tools that return a list return one entry "
            "per page."
        )
    ),
]
ObjectName = Annotated[
    str,
    Field(
        description=(
            "Name of the object to locate, e.g. 'person', 'car', 'face'. A short noun phrase "
            "works too ('the red mug'). More specific names ground more reliably than broad ones."
        )
    ),
]
CustomPrompt = Annotated[
    str,
    Field(
        description=(
            "A Florence-2 task token, e.g. '<OD>', '<CAPTION>', '<REGION_PROPOSAL>'. Not a "
            "natural-language instruction -- plain English here produces garbage, not an answer."
        )
    ),
]
Operation = Annotated[
    str,
    Field(
        description=(
            "One of: 'caption', 'ocr', 'detect', 'count', 'dense_caption', 'query'. Use 'query' "
            "(with `question`) rather than 'ocr' for watermarks, logos, signage, or "
            "stylized/cursive text -- 'ocr' misreads that kind of text confidently. Use 'count' "
            "(with `object_name`) rather than 'detect' for 'how many' -- 'detect' returns "
            "regions, which are not a tally."
        )
    ),
]
