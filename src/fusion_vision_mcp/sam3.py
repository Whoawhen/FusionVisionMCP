"""SAM 3's open-vocabulary detector, backing `detect_objects` on photographs.

It replaced Grounding DINO there after VisionBench sprint R (2026-10-12; plan in
`Future_Ideas/detector_ocr_replacement_plan.md`). On the held-out half, against Grounding DINO at 0.35, it
found 16 points more of the labelled objects and boxed a plausible-but-absent neighbour on 9% of probes
instead of 66%. It did not take over counting: asked for the synthetic fixtures by name it returns no box
for "ball", "rod" or "blob", so `count_objects` stays on Grounding DINO.

Since VisionBench gate S (2026-10-13) it also backs `spatial_relations`, through `locate_masks`: its own
instance mask replaces Grounding DINO's box plus a SAM2 mask (25 of 36 Open Images relations either way).

The post-processing here is exactly what the gates replayed (`replacement_gate1.py`): the processor's
instance post-processing at 0.05, the top 300 boxes, the score floor, class-agnostic NMS at IoU 0.5, then
Grounding DINO's envelope filter. Changing any step means re-running `replacement_gate2.py`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL.Image import Image
from transformers import Sam3Model, Sam3Processor

from . import geometry
from .constants import DEFAULT_SAM3_MODEL, LOCATE_BOX_THRESHOLD
from .device import resolve_device
from .grounding_dino import GroundingDino

logger = logging.getLogger(__name__)

#: Raw candidates kept before the score floor, and the processor's own floor -- both as cached by the gates.
_RAW_TOP_K = 300
_RAW_FLOOR = 0.05
_NMS_IOU = 0.5


class Sam3Detector:
    """SAM 3 (`facebook/sam3`, gated on Hugging Face) as an `InstanceDetector`."""

    def __init__(self, model_id: str = DEFAULT_SAM3_MODEL, device: str | None = None) -> None:
        self.device = resolve_device(device)
        logger.info(f"Loading SAM 3 ({model_id})")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()  # type: ignore[arg-type]

    def _instances(self, image: Image, object_name: str) -> tuple[list[list[float]], list[float], Any]:
        """Boxes and scores sorted by score, highest first, with the instance masks in the same order."""
        inputs = self.processor(images=image, text=object_name, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        r = self.processor.post_process_instance_segmentation(
            out, threshold=_RAW_FLOOR, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        boxes, scores = r["boxes"].tolist(), r["scores"].tolist()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:_RAW_TOP_K]
        return [boxes[i] for i in order], [scores[i] for i in order], r["masks"][order]

    def _raw(self, image: Image, object_name: str) -> tuple[list[list[float]], list[float]]:
        boxes, scores, _ = self._instances(image, object_name)
        return boxes, scores

    @staticmethod
    def postprocess(
        boxes: list[list[float]], scores: list[float], object_name: str, threshold: float
    ) -> dict[str, Any]:
        """Score floor, NMS and the envelope filter over boxes already sorted by score, highest first."""
        kept: list[int] = []
        for i, s in enumerate(scores):
            if s >= threshold and all(geometry.box_iou(boxes[i], boxes[j]) < _NMS_IOU for j in kept):
                kept.append(i)
        b = [boxes[i] for i in kept]
        env = GroundingDino._envelope_indices(b)
        keep = [i for i in range(len(b)) if i not in env]
        b = [b[i] for i in keep]
        return {
            "count": len(b),
            "bboxes": b,
            "points": [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in b],
            "labels": [object_name] * len(b),
            "scores": [scores[kept[i]] for i in keep],
            "group_boxes_dropped": len(env),
        }

    def detect_objects(
        self,
        images: list[Image],
        object_name: str,
        threshold: float = LOCATE_BOX_THRESHOLD,
        adaptive_threshold: bool = False,
    ) -> list[dict[str, Any]]:
        """Scored boxes for every instance of `object_name`, one entry per image.

        `adaptive_threshold` is accepted for the `InstanceDetector` protocol and ignored: the gates measured
        SAM 3 at a fixed floor.
        """
        res = []
        for img in images:
            with img.convert("RGB") as rgb:
                boxes, scores = self._raw(rgb, object_name)
            res.append(self.postprocess(boxes, scores, object_name, threshold))
        return res

    def locate_masks(self, image: Image, object_name: str, threshold: float = LOCATE_BOX_THRESHOLD) -> dict[str, Any]:
        """`detect_objects` for one image, plus `masks`: SAM 3's own instance mask for each kept box.

        Backs `spatial_relations` since VisionBench gate S (2026-10-13), in place of Grounding DINO's box
        followed by a SAM2 mask. Each mask is a boolean array on the image's own pixel grid, thresholded at
        0.5 as the gates' post-processing did.
        """
        with image.convert("RGB") as rgb:
            boxes, scores, masks = self._instances(rgb, object_name)
        result = self.postprocess(boxes, scores, object_name, threshold)
        # postprocess keeps a subset of the sorted boxes in order, so each kept box is found by identity.
        result["masks"] = [masks[boxes.index(b)].cpu().numpy().astype(bool) for b in result["bboxes"]]
        return result
