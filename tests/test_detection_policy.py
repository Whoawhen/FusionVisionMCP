"""Unit tests for detection_policy.execute_detection (no models).

This is where detect_objects routes photos to Grounding DINO at a 0.35 floor and clip art
to Florence-2. It was only exercised end to end; these pin the routing and the guards
against recording stubs.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fusion_vision_mcp import constants
from fusion_vision_mcp.detection_policy import execute_detection
from fusion_vision_mcp.tools import detection as detection_tools


class _RecordingFlorence2:
    def __init__(self, bboxes: list | None = None) -> None:
        self.bboxes = bboxes if bboxes is not None else [[1, 1, 5, 5], [6, 6, 9, 9]]
        self.calls: list[dict] = []

    def detect_objects(self, images: list, object_name: str, **kwargs) -> list[dict]:
        self.calls.append({"object_name": object_name, **kwargs})
        return [
            {
                "bboxes": self.bboxes,
                "points": [[3, 3]] * len(self.bboxes),
                "labels": [object_name] * len(self.bboxes),
            }
            for _ in images
        ]


class _RecordingCounter:
    def __init__(self, bboxes: list | None = None) -> None:
        self.bboxes = bboxes if bboxes is not None else [[1, 1, 5, 5]]
        self.calls: list[dict] = []

    def detect_objects(self, images: list, object_name: str, **kwargs) -> list[dict]:
        self.calls.append({"object_name": object_name, **kwargs})
        n = len(self.bboxes)
        return [
            {
                "count": n,
                "bboxes": [list(b) for b in self.bboxes],
                "points": [[3, 3]] * n,
                "labels": [object_name] * n,
                "scores": [0.9] * n,
            }
            for _ in images
        ]


class _StubApp:
    def __init__(self, florence2=None, counter=None) -> None:
        self.florence2 = florence2 or _RecordingFlorence2()
        self.counter = counter or _RecordingCounter()


def _photo() -> Image.Image:
    """Noise: thousands of unique colours, so routing calls it a photograph."""
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _clip_art() -> Image.Image:
    """A few flat, saturated fills: routing calls it clip art."""
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 30, 30], fill=(255, 0, 0))
    draw.ellipse([34, 34, 58, 58], fill=(0, 0, 255))
    return img


class TestExplicitBackends:
    def test_florence2_calls_only_florence2(self) -> None:
        app = _StubApp()
        result = execute_detection(app, [_photo()], "mug", backend_preference="florence2")[0]
        assert len(app.florence2.calls) == 1
        assert app.counter.calls == []
        assert result["detector"] == "florence2"
        assert result["scores"] is None
        assert result["count"] == len(result["bboxes"]) == 2

    def test_grounding_dino_calls_only_the_counter_with_settings_unchanged(self) -> None:
        app = _StubApp()
        result = execute_detection(
            app, [_clip_art()], "mug", backend_preference="grounding_dino", threshold=0.42, adaptive_threshold=False
        )[0]
        assert app.florence2.calls == []
        assert app.counter.calls == [{"object_name": "mug", "threshold": 0.42, "adaptive_threshold": False}]
        assert result["detector"] == "grounding_dino"

    def test_explicit_backends_carry_no_routing_metadata(self) -> None:
        for backend in ("florence2", "grounding_dino"):
            result = execute_detection(_StubApp(), [_photo()], "mug", backend_preference=backend)[0]
            assert "_routing" not in result


class TestAutoRouting:
    def test_photo_reaches_the_counter(self) -> None:
        app = _StubApp()
        result = execute_detection(app, [_photo()], "mug", backend_preference="auto")[0]
        assert len(app.counter.calls) == 1
        assert app.florence2.calls == []
        assert result["detector"] == "grounding_dino"
        assert result["_routing"]["backend"] == "grounding_dino"
        assert result["_routing"]["reason"]

    def test_clip_art_reaches_florence2(self) -> None:
        app = _StubApp()
        result = execute_detection(app, [_clip_art()], "mug", backend_preference="auto")[0]
        assert len(app.florence2.calls) == 1
        assert app.counter.calls == []
        assert result["detector"] == "florence2"
        assert result["_routing"]["backend"] == "florence2"
        assert result["_routing"]["reason"]


class TestGenericQueryGuard:
    def test_generic_query_excludes_full_frame_on_florence2(self) -> None:
        app = _StubApp()
        execute_detection(app, [_clip_art()], "object", backend_preference="florence2")
        assert app.florence2.calls[0]["exclude_full_frame"] is True

    def test_specific_query_keeps_full_frame_on_florence2(self) -> None:
        app = _StubApp()
        execute_detection(app, [_clip_art()], "red mug", backend_preference="florence2")
        assert app.florence2.calls[0]["exclude_full_frame"] is False

    def test_generic_query_drops_full_frame_grounding_dino_box(self) -> None:
        counter = _RecordingCounter(bboxes=[[0, 0, 64, 64], [4, 4, 20, 20]])
        result = execute_detection(
            _StubApp(counter=counter), [_photo()], "object", backend_preference="grounding_dino"
        )[0]
        assert result["bboxes"] == [[4, 4, 20, 20]]
        assert result["count"] == 1
        assert result["generic_full_frame_dropped"] == 1

    def test_specific_query_keeps_full_frame_grounding_dino_box(self) -> None:
        """A close-up of wood asked for "wood" is a legitimate full-frame match."""
        counter = _RecordingCounter(bboxes=[[0, 0, 64, 64]])
        result = execute_detection(_StubApp(counter=counter), [_photo()], "wood", backend_preference="grounding_dino")[
            0
        ]
        assert result["bboxes"] == [[0, 0, 64, 64]]
        assert result["count"] == 1
        assert "generic_full_frame_dropped" not in result


class _FakeContext:
    """Just enough of Context for the tool body: request_context.lifespan_context."""

    def __init__(self, app) -> None:
        self.request_context = type("_RC", (), {"lifespan_context": app})()


class _CapturingServer:
    """A bare stand-in for MCPServer that hands back the undecorated tool functions."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self):
        def decorate(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorate


def test_detect_objects_tool_calls_the_policy_the_right_way(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_objects must ask for auto routing at the 0.35 floor with no adaptive sweep."""
    seen: dict = {}

    def recording_policy(app, images, object_name, **kwargs):
        seen.update(kwargs, object_name=object_name)
        return [{"count": 0, "bboxes": [], "points": [], "labels": [], "scores": [], "detector": "grounding_dino"}]

    monkeypatch.setattr(detection_tools, "execute_detection", recording_policy)
    server = _CapturingServer()
    detection_tools.register(server)  # type: ignore[arg-type]

    # A TemporaryDirectory, not tmp_path: pytest's own tmp-dir teardown fails on this machine.
    with tempfile.TemporaryDirectory() as tmp:
        photo_path = Path(tmp) / "photo.png"
        _photo().save(photo_path)
        server.tools["detect_objects"](_FakeContext(_StubApp()), str(photo_path), "dog")

    assert seen["backend_preference"] == "auto"
    assert seen["threshold"] == constants.DETECT_BOX_THRESHOLD == 0.35
    assert seen["adaptive_threshold"] is False
    assert seen["object_name"] == "dog"


class _FailingLocator:
    """SAM 3 without Hugging Face access: loading raises an OSError subclass."""

    def detect_objects(self, images: list, object_name: str, **kwargs) -> list[dict]:
        raise OSError("gated repo")


class TestLocate:
    """`detect_objects` (locate=True) uses SAM 3 on photos; counting never does."""

    def _app(self, locator=None) -> _StubApp:
        app = _StubApp()
        app.locator = locator if locator is not None else _RecordingCounter()
        return app

    def test_photo_goes_to_sam3_at_its_threshold(self) -> None:
        app = self._app()
        result = execute_detection(
            app, [_photo()], "mug", backend_preference="auto", threshold=0.35, adaptive_threshold=False, locate=True
        )[0]
        assert app.locator.calls == [{"object_name": "mug", "threshold": constants.LOCATE_BOX_THRESHOLD}]
        assert app.counter.calls == []
        assert result["detector"] == "sam3"

    def test_clip_art_still_goes_to_florence2(self) -> None:
        app = self._app()
        result = execute_detection(app, [_clip_art()], "mug", backend_preference="auto", locate=True)[0]
        assert app.locator.calls == []
        assert result["detector"] == "florence2"

    def test_counting_never_uses_sam3(self) -> None:
        app = self._app()
        result = execute_detection(app, [_photo()], "mug", backend_preference="auto")[0]
        assert app.locator.calls == []
        assert result["detector"] == "grounding_dino"

    def test_unavailable_sam3_falls_back_to_grounding_dino_at_its_own_threshold(self) -> None:
        app = self._app(_FailingLocator())
        result = execute_detection(
            app, [_photo()], "mug", backend_preference="auto", threshold=0.9, adaptive_threshold=True, locate=True
        )[0]
        assert app.counter.calls == [
            {"object_name": "mug", "threshold": constants.DETECT_BOX_THRESHOLD, "adaptive_threshold": False}
        ]
        assert result["detector"] == "grounding_dino"

    def test_a_context_without_a_locator_uses_grounding_dino(self) -> None:
        app = _StubApp()
        result = execute_detection(app, [_photo()], "mug", backend_preference="auto", locate=True)[0]
        assert result["detector"] == "grounding_dino"
