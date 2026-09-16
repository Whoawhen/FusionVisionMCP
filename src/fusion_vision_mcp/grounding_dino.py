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

from fusion_vision_mcp import geometry

from .adaptive_threshold import ThresholdResult, choose_threshold
from .constants import DEFAULT_BOX_THRESHOLD, DEFAULT_GROUNDING_DINO_MODEL
from .device import resolve_device

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

#: Above this mean pairwise IoU, the boxes a candidate "contains" are not separate
#: instances -- they are noisy, overlapping re-detections of the *same* object at
#: different scales, and the candidate must not be dropped as a group envelope.
#:
#: Found investigating a partially-occluded object (a rectangle ~60% hidden behind
#: another shape) that a confidence cut of 0.15 dropped to zero detections even though
#: the correct box scored 0.66, well above threshold. The drop was not a threshold
#: problem: at the default threshold, the correct box "contained" two looser duplicate
#: boxes of the same rectangle and was misclassified as an envelope around a group.
#: Real group members (separate petals, discs) barely overlap each other -- mean
#: pairwise IoU near 0 on every ring/row/grid fixture in `benchmarks/fixtures.py`.
#: The occlusion case's duplicate members overlapped each other at IoU 0.805. 0.5 sits
#: clear of both: verified to reproduce every existing benchmark fixture's envelope
#: decision unchanged (all 9 positive/negative group-box cases), while recovering the
#: occluded rectangle from 0 detections to 2, one of which is the correct 0.66 box.
_MAX_MEMBER_MUTUAL_IOU: Final[float] = 0.5


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

    #: One shared implementation in `geometry`; `inspection` used to carry a second copy.
    _iou = staticmethod(geometry.box_iou)

    @classmethod
    def _envelope_indices(cls, bboxes: list[list[float]]) -> set[int]:
        """Indices of boxes that enclose a group of separate instances rather than belong to one.

        Asked to find a repeated part, the model reliably returns the instances *and*
        one box drawn around the whole arrangement. That box is not a duplicate of any
        single detection, so non-maximum suppression keeps it, and it tends to score
        highest, so a confidence cut removes the real instances first. What identifies
        it is that it swallows the others' centres.

        That containment test alone also fires on a single partially-occluded object:
        the correct tight box can "contain" several looser, lower-confidence duplicate
        detections of that *same* object at different scales, which look identical to a
        group by centre-containment even though there is no group. What distinguishes
        the two is whether the contained boxes overlap each other: real group members
        (separate petals, discs) barely overlap; duplicate re-detections of one object
        overlap heavily. Only boxes whose "members" have low mutual overlap are dropped.
        """
        if len(bboxes) < _MIN_BOXES_FOR_ENVELOPE:
            return set()

        centres = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]
        envelopes = set()
        for index, (x1, y1, x2, y2) in enumerate(bboxes):
            contained = [
                other for other, (cx, cy) in enumerate(centres) if other != index and x1 <= cx <= x2 and y1 <= cy <= y2
            ]
            if len(contained) < _ENVELOPE_CONTAINMENT * (len(bboxes) - 1):
                continue

            member_boxes = [bboxes[i] for i in contained]
            pairs = [
                cls._iou(member_boxes[a], member_boxes[b])
                for a in range(len(member_boxes))
                for b in range(a + 1, len(member_boxes))
            ]
            mean_member_iou = sum(pairs) / len(pairs) if pairs else 0.0
            if mean_member_iou > _MAX_MEMBER_MUTUAL_IOU:
                continue

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
        adaptive_threshold: bool = True,
    ) -> list[dict[str, Any]]:
        """Locate every instance of a named object, one entry per image.

        Returns `bboxes` ([x1, y1, x2, y2] in image pixels), `points` (box centres),
        `labels`, `scores` and `count`, all index-aligned. Unlike the sequence heads,
        `count` here is a genuine tally of separate detections rather than however many
        regions a decoder happened to emit before stopping.

        `drop_group_box` discards detections drawn around the whole arrangement rather
        than around one instance, reported as `group_boxes_dropped`. Set it false to
        see the raw detections, including any envelope.

        `adaptive_threshold` (default True) enables automatic threshold selection.
        When enabled, the model runs once with a minimal threshold to collect all
        raw scores, then `choose_threshold` selects a conservative floor based on
        the score distribution's "confidence cliff". The result includes an
        `adaptive_threshold` field with the selected threshold, whether it was used,
        confidence, and reason.
        """
        prompt = self._as_prompt(object_name)
        results = []
        for img in images:
            with img.convert("RGB") as rgb:
                inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Run post-processing with a minimal threshold to get ALL raw scores
                # (0.0 is the absolute floor; the processor will still filter at some internal level)
                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=0.0,
                    text_threshold=text_threshold,
                    target_sizes=[(rgb.height, rgb.width)],
                )[0]

                raw_bboxes = [[float(v) for v in box] for box in processed["boxes"].tolist()]
                raw_scores = [float(s) for s in processed["scores"].tolist()]
                # `text_labels` is the phrase each box matched; older builds only carry
                # `labels`. Fall back rather than failing on a key name.
                labels = processed.get("text_labels") or processed.get("labels") or [object_name] * len(raw_bboxes)
                labels = [str(label) for label in labels]

                # Compute adaptive threshold from raw scores
                threshold_result: ThresholdResult | None = None
                if adaptive_threshold and raw_scores:
                    threshold_result = choose_threshold(raw_scores, base_threshold=threshold)
                    effective_threshold = threshold_result.threshold
                else:
                    effective_threshold = threshold

                # Filter by the effective threshold
                keep_scores = [i for i, s in enumerate(raw_scores) if s >= effective_threshold]
                bboxes = [raw_bboxes[i] for i in keep_scores]
                scores = [raw_scores[i] for i in keep_scores]
                labels = [labels[i] for i in keep_scores]

                envelopes = self._envelope_indices(bboxes) if drop_group_box else set()
                keep = [i for i in range(len(bboxes)) if i not in envelopes]
                bboxes = [bboxes[i] for i in keep]
                scores = [scores[i] for i in keep]
                labels = [labels[i] for i in keep]

                result = {
                    "count": len(bboxes),
                    "bboxes": bboxes,
                    "points": [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes],
                    "labels": labels,
                    "scores": scores,
                    "group_boxes_dropped": len(envelopes),
                }
                if threshold_result is not None:
                    result["adaptive_threshold"] = {
                        "threshold": threshold_result.threshold,
                        "used": threshold_result.used,
                        "confidence": threshold_result.confidence,
                        "reason": threshold_result.reason,
                        "base_threshold": threshold,
                        "raw_detection_count": len(raw_bboxes),
                    }
                results.append(result)
        return results
