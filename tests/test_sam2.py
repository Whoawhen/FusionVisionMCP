"""Unit tests for the SAM2 wrapper.

The model itself is exercised end-to-end by `spatial_relations` in
`tests/test_server.py`. What is covered here is the wrapper logic around it --
device/dtype selection and the empty-box short circuit -- with the HuggingFace
loaders stubbed, so nothing is downloaded and these stay in the fast suite.
"""

from typing import Any

import torch
from PIL import Image
from pytest import MonkeyPatch

from fusion_vision_mcp import sam2 as sam2_module
from fusion_vision_mcp.sam2 import Sam2


class _StubModel:
    def __init__(self) -> None:
        self.dtype_seen: Any = None
        self.device_seen: Any = None
        self.eval_called = False

    def to(self, device: str) -> "_StubModel":
        self.device_seen = device
        return self

    def eval(self) -> "_StubModel":
        self.eval_called = True
        return self


def _stub_loaders(monkeypatch: MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    model = _StubModel()

    def fake_model(model_id: str, dtype: Any = None) -> _StubModel:
        captured["model_id"] = model_id
        captured["dtype"] = dtype
        return model

    def fake_processor(model_id: str) -> object:
        captured["processor_id"] = model_id
        return object()

    monkeypatch.setattr(sam2_module.Sam2Model, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(sam2_module.Sam2Processor, "from_pretrained", staticmethod(fake_processor))
    captured["model"] = model
    return captured


def test_cpu_loads_in_float32(monkeypatch: MonkeyPatch) -> None:
    """Half precision on CPU is slow or unsupported depending on the op."""
    captured = _stub_loaders(monkeypatch)

    wrapper = Sam2(device="cpu")

    assert wrapper.torch_dtype is torch.float32
    assert captured["dtype"] is torch.float32


def test_accelerator_loads_in_float16(monkeypatch: MonkeyPatch) -> None:
    """Matches the pattern every other wrapper uses: fp32 on CPU, fp16 elsewhere."""
    captured = _stub_loaders(monkeypatch)

    wrapper = Sam2(device="cuda")

    assert wrapper.torch_dtype is torch.float16
    assert captured["dtype"] is torch.float16


def test_model_is_put_in_eval_mode(monkeypatch: MonkeyPatch) -> None:
    """Inference only. Leaving a model in train mode changes dropout/batchnorm."""
    captured = _stub_loaders(monkeypatch)

    Sam2(device="cpu")

    assert captured["model"].eval_called
    assert captured["model"].device_seen == "cpu"


def test_explicit_model_id_is_forwarded(monkeypatch: MonkeyPatch) -> None:
    captured = _stub_loaders(monkeypatch)

    Sam2(model_id="facebook/sam2.1-hiera-tiny", device="cpu")

    assert captured["model_id"] == "facebook/sam2.1-hiera-tiny"
    assert captured["processor_id"] == "facebook/sam2.1-hiera-tiny"


def test_no_boxes_returns_no_masks_without_touching_the_model() -> None:
    """`spatial_relations` can legitimately find nothing to segment.

    Built via `__new__` so no model exists at all: if the short circuit ever stopped
    happening, this would raise AttributeError rather than quietly running inference.
    """
    wrapper = object.__new__(Sam2)

    assert wrapper.segment(Image.new("RGB", (10, 10)), []) == []
