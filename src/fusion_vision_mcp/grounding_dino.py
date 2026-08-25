#  grounding_dino.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Grounding DINO: open-vocabulary detection that returns one box per instance.

Both detection heads already in this server emit regions as a *sequence*: Florence-2
decodes grounding results token by token, and Moondream's detect head generates points
until it stops. That is why both collapse instances that visually merge -- one run-on
emission covering the group. Grounding DINO instead scores a fixed set of parallel
object queries and suppresses duplicates, so overlapping instances stay separate
detections. It also carries a per-detection confidence, which a sequence head has no
equivalent of.

Ships inside the pinned `transformers`, so this costs a weights download on first use
rather than a new dependency.
"""

from typing import Any, Final

import torch
from PIL.Image import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .device import resolve_device

#: `tiny` (~690MB) is the default for the same reason `sam2.1-hiera-small` is: it is
#: the smallest checkpoint that does the job, and the larger `base` roughly triples the
#: download for a marginal gain on the kind of query this server sends.
DEFAULT_GROUNDING_DINO_MODEL: Final[str] = "IDEA-Research/grounding-dino-tiny"

#: Box confidence below which a detection is dropped, and the separate threshold on how
#: well it matches the prompt text. Both exposed per call, because the right trade
#: between recall and precision depends on what is being counted.
#:
#: 0.15 rather than the upstream default of 0.25, chosen by sweeping the whole counting
#: fixture set (`benchmarks/`): it takes exact positives from 8/10 to 9/10 and mean
#: absolute count error from 1.15 to 0.10 while still holding all eight negative
#: controls. Dropping further scores better on positives alone -- 0.125 and 0.10 both
#: reach 10/10 -- but each breaks a negative control, fragmenting a spotted ball into
#: 3-4 objects and a rough-edged rod into 2. That is the failure this project refuses to
#: ship, so 0.15 is the floor: the lowest threshold at which every control still holds.
DEFAULT_BOX_THRESHOLD: Final[float] = 0.15
DEFAULT_TEXT_THRESHOLD: Final[float] = 0.25

#: Share of the other detections whose centres a box must swallow to count as an
#: envelope around the group rather than a member of it. Measured: asked for `petal`
#: against eight shapes, the model returns the eight plus one box spanning the whole
#: arrangement -- at 68% of the frame when they are separated, 38% when touching, and
#: carrying the *highest* score both times, so score cannot be used to spot it.
#: Dropping it turns a 9 into the correct 8 in both arrangements.
_ENVELOPE_CONTAINMENT: Final[float] = 0.6

#: Below this many detections there is no "group" for a box to enclose, and a genuine
#: pair of nested objects would be discarded instead. Three is the floor worth applying:
#: at three boxes the containment rule below requires one box to swallow *both* others,
#: which is a strong signal, while at two it would only require swallowing one and would
#: discard a genuinely nested object. Measured: two touching instances come back as
#: 2 instances plus 1 envelope, so a floor of 4 left every two-instance count one too high.
_MIN_BOXES_FOR_ENVELOPE: Final[int] = 3


class GroundingDino:
    """Wraps Grounding DINO for open-vocabulary object detection."""

    device: str
    torch_dtype: torch.dtype
    model: Any
    processor: Any

    def __init__(self, model_id: str = DEFAULT_GROUNDING_DINO_MODEL, device: str | None = None) -> None:
        self.device = resolve_device(device)
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, dtype=self.torch_dtype).to(
            self.device
        )
        self.model.eval()

    @staticmethod
    def _envelope_indices(bboxes: list[list[float]]) -> set[int]:
        """Indices of boxes that enclose the group rather than belong to it.

        Asked to find a repeated part, the model reliably returns the instances *and*
        one box drawn around the whole arrangement. That box is not a duplicate of any
        single detection, so non-maximum suppression keeps it, and it tends to score
        highest, so a confidence cut removes the real instances first. What identifies
        it is that it swallows the others' centres.
        """
        if len(bboxes) < _MIN_BOXES_FOR_ENVELOPE:
            return set()

        centres = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]
        envelopes = set()
        for index, (x1, y1, x2, y2) in enumerate(bboxes):
            contained = sum(
                1 for other, (cx, cy) in enumerate(centres) if other != index and x1 <= cx <= x2 and y1 <= cy <= y2
            )
            if contained >= _ENVELOPE_CONTAINMENT * (len(bboxes) - 1):
                envelopes.add(index)
        return envelopes

    @staticmethod
    def _as_prompt(object_name: str) -> str:
        """Grounding DINO expects a lowercase, period-terminated phrase."""
        text = object_name.strip().lower()
        return text if text.endswith(".") else f"{text}."

    def detect_objects(
        self,
        images: list[Image],
        object_name: str,
        threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        drop_group_box: bool = True,
    ) -> list[dict[str, Any]]:
        """Locate every instance of a named object, one entry per image.

        Returns `bboxes` ([x1, y1, x2, y2] in image pixels), `points` (box centres),
        `labels`, `scores` and `count`, all index-aligned. Unlike the sequence heads,
        `count` here is a genuine tally of separate detections rather than however many
        regions a decoder happened to emit before stopping.

        `drop_group_box` discards detections drawn around the whole arrangement rather
        than around one instance, reported as `group_boxes_dropped`. Set it false to
        see the raw detections, including any envelope.
        """
        prompt = self._as_prompt(object_name)
        results = []
        for img in images:
            with img.convert("RGB") as rgb:
                inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(rgb.height, rgb.width)],
                )[0]

                bboxes = [[float(v) for v in box] for box in processed["boxes"].tolist()]
                scores = [float(s) for s in processed["scores"].tolist()]
                # `text_labels` is the phrase each box matched; older builds only carry
                # `labels`. Fall back rather than failing on a key name.
                labels = processed.get("text_labels") or processed.get("labels") or [object_name] * len(bboxes)
                labels = [str(label) for label in labels]

                envelopes = self._envelope_indices(bboxes) if drop_group_box else set()
                keep = [i for i in range(len(bboxes)) if i not in envelopes]
                bboxes = [bboxes[i] for i in keep]
                results.append(
                    {
                        "count": len(bboxes),
                        "bboxes": bboxes,
                        "points": [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes],
                        "labels": [labels[i] for i in keep],
                        "scores": [scores[i] for i in keep],
                        "group_boxes_dropped": len(envelopes),
                    }
                )
        return results
