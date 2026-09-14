#  constants.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Model-choice constants and the `CaptionLevel` enum, kept free of torch/transformers.

Split out of the model wrapper modules (`aesthetic.py`, `florence2.py`,
`grounding_dino.py`, `moondream.py`, `sam2.py`) so that `fusion_vision_mcp/__init__.py`
can use these as function-signature defaults -- evaluated at package-import time --
without importing torch and transformers just to answer an MCP `initialize` request.
Each wrapper module still imports its constant(s) from here and re-exports them, so
`from fusion_vision_mcp.sam2 import DEFAULT_SAM2_MODEL` (etc.) keeps working unchanged.
"""

from enum import StrEnum
from typing import Final


class CaptionLevel(StrEnum):
    NORMAL = "<CAPTION>"
    DETAILED = "<DETAILED_CAPTION>"
    MORE_DETAILED = "<MORE_DETAILED_CAPTION>"


#: The aesthetic head below was trained on embeddings from this exact CLIP checkpoint;
#: swapping the backbone silently invalidates the head's weights.
DEFAULT_AESTHETIC_MODEL: Final[str] = "openai/clip-vit-large-patch14"

#: `tiny` (~690MB) is the default for the same reason `sam2.1-hiera-small` is: it is
#: the smallest checkpoint that does the job, and the larger `base` roughly triples the
#: download for a marginal gain on the kind of query this server sends.
DEFAULT_GROUNDING_DINO_MODEL: Final[str] = "IDEA-Research/grounding-dino-tiny"

#: Box confidence below which a detection is dropped. Lives here rather than in
#: `grounding_dino.py` (which re-exports it) because `__init__.py` also needs it as a
#: `Protocol` method default, evaluated at package-import time.
#:
#: 0.15 rather than the upstream default of 0.25, chosen by sweeping the whole counting
#: fixture set (`benchmarks/`): it takes exact positives from 8/10 to 9/10 and mean
#: absolute count error from 1.15 to 0.10 while still holding all eight negative
#: controls. Dropping further scores better on positives alone -- 0.125 and 0.10 both
#: reach 10/10 -- but each breaks a negative control, fragmenting a spotted ball into
#: 3-4 objects and a rough-edged rod into 2. That is the failure this project refuses to
#: ship, so 0.15 is the floor: the lowest threshold at which every control still holds.
DEFAULT_BOX_THRESHOLD: Final[float] = 0.15

DEFAULT_MOONDREAM_MODEL: Final[str] = "vikhyatk/moondream2"
#: Pinned rather than tracking `main`, since this repository ships its model code via
#: `trust_remote_code` and reshapes it between revisions. `2025-06-21` is chosen over the
#: earlier `2025-01-09` pin because it fixed a measured self-consistency failure: asked to
#: count the petals in `tests/sample.jpg` the old pin answered 12 and then listed 6 colours
#: for them, while this one answers 10 and lists exactly 10.
DEFAULT_MOONDREAM_REVISION: Final[str] = "2025-06-21"

#: Measured on CPU over tiny/small/base-plus: `small` costs ~0.06s more per call
#: and ~34MB more than `tiny` for a better mask, while `base-plus` roughly doubles
#: inference time for a marginal gain. `small` is the middle that earns its keep.
DEFAULT_SAM2_MODEL: Final[str] = "facebook/sam2.1-hiera-small"

#: SAM2 decodes masks on a fixed grid regardless of input size. Masks are upscaled
#: to the image's own pixels for reporting, but detail finer than
#: `max(image side) / MASK_DECODE_RESOLUTION` pixels is not actually resolved.
MASK_DECODE_RESOLUTION: Final[int] = 256
