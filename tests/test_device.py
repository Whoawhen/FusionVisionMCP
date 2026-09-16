"""Unit tests for device selection.

`resolve_device` decides which torch device every wrapper loads onto, so getting it
wrong is silently expensive (CPU inference on a GPU box) rather than loud. The
accelerator probes are monkeypatched, so these run anywhere.
"""

import torch
from pytest import MonkeyPatch

from fusion_vision_mcp.device import resolve_device


def _set_accelerators(monkeypatch: MonkeyPatch, *, mps: bool, cuda: bool) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


def test_explicit_override_wins_over_every_probe(monkeypatch: MonkeyPatch) -> None:
    """An explicit --device must be honoured even when an accelerator is present.

    This is the point of the override: pinning to a device index, or forcing CPU on a
    shared GPU box.
    """
    _set_accelerators(monkeypatch, mps=True, cuda=True)

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_mps_is_preferred_when_available(monkeypatch: MonkeyPatch) -> None:
    _set_accelerators(monkeypatch, mps=True, cuda=False)

    assert resolve_device() == "mps:0"


def test_cuda_is_used_when_there_is_no_mps(monkeypatch: MonkeyPatch) -> None:
    _set_accelerators(monkeypatch, mps=False, cuda=True)

    assert resolve_device() == "cuda"


def test_cpu_is_the_fallback(monkeypatch: MonkeyPatch) -> None:
    _set_accelerators(monkeypatch, mps=False, cuda=False)

    assert resolve_device() == "cpu"


def test_empty_string_falls_through_to_detection(monkeypatch: MonkeyPatch) -> None:
    """`""` is falsy, so it must behave as "not specified" rather than as a device name."""
    _set_accelerators(monkeypatch, mps=False, cuda=False)

    assert resolve_device("") == "cpu"


def test_returns_a_string_torch_accepts() -> None:
    """Every wrapper passes this straight to `.to(...)`, so it must be a valid device."""
    torch.device(resolve_device())
