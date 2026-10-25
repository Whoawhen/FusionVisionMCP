"""Checks free memory before a model loads, and releases idle models to make room.

`IdleReleased` decides *when* a model goes (after a stretch of no use). This module
decides *whether the next one fits*. Without it, a session that touched enough models
kept them all, and on 2026-10-24 the test server reached 26.8 GB of commit on a 31.5 GB
laptop: Windows logged a low-virtual-memory condition and the process died with every
client seeing only "connection closed" (Future_Ideas/memory_manager.md).

Before a load, `MemoryManager` compares the model's estimated size with live free
memory and, when it is short, releases least-recently-used idle models first. Two
limits are read, because they fail differently:

- **Commit** (Windows' RAM + pagefile allowance) is the hard one. Past it allocations
  fail and torch crashes. A load that still can't fit after releasing everything idle
  raises `ModelMemoryError`, which the client gets as a tool error.
- **RAM** is the soft one. Past it the load pages instead of failing: slow, but alive.
  Idle models are released to avoid it, and if that isn't enough the load goes ahead
  with a warning.

An optional budget (`--memory-budget`) caps the summed size of loaded models the same
soft way. Loads are serialised, so two concurrent calls can't both pass the check and
then both load.

This module is on the eager import path and must stay torch-free: sizes come from a
static table (`constants.MODEL_MEMORY_GB`), not from inspecting a model.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fusion_vision_mcp.constants import DEFAULT_MODEL_MEMORY_GB, MEMORY_RESERVE_GB, MODEL_MEMORY_GB

if TYPE_CHECKING:
    from fusion_vision_mcp.idle import IdleReleased

logger = logging.getLogger(__name__)

GB: int = 1024**3


class ModelMemoryError(MemoryError):
    """A model can't be loaded because the machine lacks the memory for it."""


@dataclass(frozen=True)
class MemoryStatus:
    """Live free memory in bytes. `None` means this platform can't report it.

    `available_commit` is the hard limit: Windows' commit (RAM + pagefile) where it is
    accounted, free RAM + free swap elsewhere, since that is what the out-of-memory
    killer acts on. `commit_label` names which one it is in messages.
    """

    available_ram: int | None
    available_commit: int | None
    commit_label: str = "commit"

    def describe(self) -> str:
        parts = []
        if self.available_ram is not None:
            parts.append(f"{self.available_ram / GB:.1f} GB RAM free")
        if self.available_commit is not None:
            parts.append(f"{self.available_commit / GB:.1f} GB {self.commit_label} free")
        return ", ".join(parts) or "free memory unknown"


if sys.platform == "win32":  # pragma: no cover - platform specific

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class _ProcessMemoryCounters(ctypes.Structure):
        """PROCESS_MEMORY_COUNTERS_EX."""

        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    # A private handle, so setting argtypes here can't disturb `idle.py`'s use of kernel32.
    _K32: Any = None
    try:
        _K32 = ctypes.WinDLL("kernel32")
        _K32.GetCurrentProcess.restype = ctypes.c_void_p
        _K32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
    except Exception:  # noqa: BLE001 - diagnostics only; the manager then sees "unknown"
        _K32 = None


def read_memory_status() -> MemoryStatus:
    """Read free RAM and free commit for the whole machine, as the OS reports them now.

    Read on every load rather than once: Windows grows a system-managed pagefile after a
    low-memory event, so the commit limit moves (80 GB of pagefile and a 111.5 GB limit
    the day after the 2026-10-24 crash). Linux overcommits by default, so its commit
    figures say nothing about what will fail; RAM + swap is the limit there.
    """
    if sys.platform == "win32":  # pragma: no cover - platform specific
        if _K32 is None:
            return MemoryStatus(None, None)
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not _K32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return MemoryStatus(None, None)
        return MemoryStatus(int(status.ullAvailPhys), int(status.ullAvailPageFile))
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            fields = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in meminfo if line.count(":") == 1}
        if "MemAvailable" in fields:
            ram = fields["MemAvailable"]
            return MemoryStatus(ram, ram + fields.get("SwapFree", 0), "RAM + swap")
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # type: ignore[import-untyped]  # optional, not a dependency

        ram = int(psutil.virtual_memory().available)
        return MemoryStatus(ram, ram + int(psutil.swap_memory().free), "RAM + swap")
    except Exception:  # noqa: BLE001 - no reading is an answer too
        return MemoryStatus(None, None)


def process_private_bytes() -> int | None:
    """This process's private memory: commit on Windows, anonymous RSS elsewhere."""
    if sys.platform == "win32":  # pragma: no cover - platform specific
        if _K32 is None:
            return None
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not _K32.K32GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PrivateUsage)
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _gb(size: float) -> str:
    return f"{size / GB:.1f} GB"


class MemoryManager:
    """Makes room before each model load, across every `IdleReleased` registered with it.

    Args:
        budget_gb: Optional cap on the summed size of loaded models. Soft: idle models are
            released to stay under it, but a load is never refused for the budget alone.
        reserve_gb: Free memory every load must leave for the OS, the client and inference.
        read_status: Source of live free memory; replaceable for tests.
        read_process: Source of this process's private bytes, used to measure each load.
    """

    def __init__(
        self,
        budget_gb: float | None = None,
        reserve_gb: float = MEMORY_RESERVE_GB,
        read_status: Callable[[], MemoryStatus] = read_memory_status,
        read_process: Callable[[], int | None] = process_private_bytes,
    ) -> None:
        self._budget = int(budget_gb * GB) if budget_gb else None
        self._reserve = int(reserve_gb * GB)
        self._read_status = read_status
        self._read_process = read_process
        self._caches: list[IdleReleased[Any]] = []
        self._measured: dict[str, int] = {}
        self._load_lock = threading.Lock()

    def register(self, cache: IdleReleased[Any]) -> None:
        self._caches.append(cache)

    def estimate(self, name: str) -> int:
        """Bytes a load of `name` is expected to need.

        A size measured on an earlier load in this process can raise the table's figure
        but never lower it: the measurement is the resident size after loading, and the
        load itself can peak higher.
        """
        table = int(MODEL_MEMORY_GB.get(name, DEFAULT_MODEL_MEMORY_GB) * GB)
        return max(table, self._measured.get(name, 0))

    def resident(self, name: str) -> int:
        """Bytes a loaded `name` is taken to hold, for the budget: measured if known."""
        if name in self._measured:
            return self._measured[name]
        return int(MODEL_MEMORY_GB.get(name, DEFAULT_MODEL_MEMORY_GB) * GB)

    def loaded(self) -> list[str]:
        return [cache.name for cache in self._caches if cache.loaded]

    @contextmanager
    def loading(self, cache: IdleReleased[Any]) -> Iterator[None]:
        """Bracket one model load: make room first, measure what it took afterwards."""
        with self._load_lock:
            need = self.estimate(cache.name)
            self._make_room(cache, need)
            before = self._read_process()
            started = time.monotonic()
            yield
            after = self._read_process()
            if before is not None and after is not None:
                took = max(after - before, 0)
                self._measured[cache.name] = took
                logger.info(
                    "Loaded %s in %.1fs: +%s private (estimate %s); now %s",
                    cache.name,
                    time.monotonic() - started,
                    _gb(took),
                    _gb(need),
                    self._read_status().describe(),
                )

    def _shortfalls(self, cache: IdleReleased[Any], need: int, status: MemoryStatus) -> list[str]:
        short = []
        if status.available_commit is not None and status.available_commit < need + self._reserve:
            short.append("commit")
        if status.available_ram is not None and status.available_ram < need + self._reserve:
            short.append("RAM")
        if self._budget is not None:
            held = sum(self.resident(c.name) for c in self._caches if c.loaded and c is not cache)
            if held + need > self._budget:
                short.append("budget")
        return short

    def _make_room(self, cache: IdleReleased[Any], need: int) -> None:
        while True:
            status = self._read_status()
            short = self._shortfalls(cache, need, status)
            if not short:
                logger.info("Loading %s (needs about %s; %s)", cache.name, _gb(need), status.describe())
                return
            released = self._release_one(exclude=cache)
            if released is None:
                break
            logger.info("Released %s to make room for %s (short on %s)", released, cache.name, ", ".join(short))

        still_loaded = ", ".join(self.loaded()) or "nothing"
        if "commit" in short:
            raise ModelMemoryError(
                f"Not enough memory to load {cache.name}: it needs about {_gb(need)} plus a {_gb(self._reserve)} "
                f"reserve, and the machine has {status.describe()}. Loaded in this server, all in use: "
                f"{still_loaded}. Free memory elsewhere (other model processes) or retry once other calls finish."
            )
        logger.warning(
            "Loading %s while short on %s (%s; loaded and in use: %s)",
            cache.name,
            ", ".join(short),
            status.describe(),
            still_loaded,
        )

    def _release_one(self, exclude: IdleReleased[Any]) -> str | None:
        """Release the least recently used idle model; its name, or None if none could go."""
        candidates = [c for c in self._caches if c is not exclude and c.loaded and c.idle]
        for cache in sorted(candidates, key=lambda c: c.last_used):
            if cache.try_release():
                return cache.name
        return None
