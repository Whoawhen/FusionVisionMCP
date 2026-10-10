"""DINOv2 features for the defect probe: what a hand or face crop looks like, without captions.

CLIP was trained to match images to captions, so its embedding keeps what a caption would say
("a hand") and can drop what one wouldn't ("with six fingers"). DINOv2 (Meta, Apache-2.0) is
self-supervised on images alone. On VisionBench's round 10 (299 images from six generators the
probe never trained on) the DINOv2-large probe ranked defective images above clean ones at AUC
0.686 against CLIP's 0.612, and left 84% of clean images unflagged against 68%.

The features must match what the probe was trained on exactly (VisionBench
`defect_backbones.embed_dino`): the slow `AutoImageProcessor` (resize 256, centre crop 224) and
the CLS token concatenated with the mean of the patch tokens of `last_hidden_state`, in float32.
"""

from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoImageProcessor, AutoModel

from .device import resolve_device

#: Crops per forward pass. A crowd image can carry dozens of faces.
_BATCH = 16


class Dinov2:
    """Wraps a DINOv2 backbone to embed image crops for the learned defect probe (~1.2GB for large)."""

    device: str
    model: Any
    processor: Any

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        # use_fast=False: the probe was trained through the slow processor, and the fast one
        # resizes differently enough to move the features.
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        # float32 on every device: the probe's weights were fitted to float32 features.
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(self.device)
        self.model.eval()

    def crop_features(self, images: list[Image]) -> list[list[float]]:
        """CLS token followed by the mean patch token, one list per image (2 x hidden size floats)."""
        out: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(images), _BATCH):
                inputs = self.processor(images=images[start : start + _BATCH], return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                features = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
                out.extend(features.float().cpu().tolist())
        return out
