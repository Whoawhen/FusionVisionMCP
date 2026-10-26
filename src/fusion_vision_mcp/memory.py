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

Between loads and calls, a watcher thread (`start_watcher`) reads free memory every few
seconds and releases idle models when *something else* -- another process, another
client's server -- squeezes the machine, instead of waiting for the idle timer. See
`relieve_pressure` for its three tiers.

This module is on the eager import path and must stay torch-free: sizes come from a
static table (`constants.MODEL_MEMORY_GB`), not from inspecting a model.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fusion_vision_mcp.constants import (
    CALL_CHECK_MIN_GB,
    DEFAULT_MODEL_MEMORY_GB,
    EVENT_LOG_SIZE,
    MEMORY_RESERVE_GB,
    MEMORY_WATCH_CRITICAL_GB,
    MEMORY_WATCH_INTERVAL_S,
    MEMORY_WATCH_MIN_IDLE_S,
    MEMORY_WATCH_PAGES_OUT_PER_S,
    MEMORY_WATCH_PAGING_CHECKS,
    MEMORY_WATCH_PSI_SOME_AVG10,
    MEMORY_WATCH_TARGET_GB,
    MEMORY_WATCH_TIGHT_FRACTION,
    MEMORY_WATCH_TIGHT_IDLE_S,
    MODEL_MEMORY_GB,
    MODEL_RESIDENT_GB,
)

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
    total_ram: int | None = None

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
        return MemoryStatus(int(status.ullAvailPhys), int(status.ullAvailPageFile), total_ram=int(status.ullTotalPhys))
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            fields = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in meminfo if line.count(":") == 1}
        if "MemAvailable" in fields:
            ram = fields["MemAvailable"]
            return MemoryStatus(ram, ram + fields.get("SwapFree", 0), "RAM + swap", fields.get("MemTotal"))
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # type: ignore[import-untyped]  # optional, not a dependency

        vm = psutil.virtual_memory()
        ram = int(vm.available)
        return MemoryStatus(ram, ram + int(psutil.swap_memory().free), "RAM + swap", int(vm.total))
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


@dataclass(frozen=True)
class PagingReading:
    """One reading of how hard the OS is paging: the value, its unit, and whether it counts as high."""

    value: float
    unit: str
    high: bool


if sys.platform == "win32":  # pragma: no cover - platform specific

    class _PdhValue(ctypes.Structure):
        """PDH_FMT_COUNTERVALUE with the double member of its union (8-byte aligned after CStatus)."""

        _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


_PDH_FMT_DOUBLE = 0x00000200


class PagingMonitor:
    """Reads paging pressure: Pages Output/sec on Windows, memory PSI on Linux, nothing elsewhere.

    Free RAM alone misses the pressure Windows absorbs by paging other programs out, so the
    watcher reads this too. Pages Output/sec counts pages written out to make room, which is
    ~0 unless memory is short; Pages Input/sec would also count memory-mapped file reads,
    which every model load here is. A rate needs two samples, so the first Windows read
    returns None. Keep one monitor per thread: a PDH query is not shared.
    """

    def __init__(self) -> None:
        self._pdh: Any = None
        self._query: Any = None
        self._counter: Any = None
        self._failed = False

    def read(self) -> PagingReading | None:
        if sys.platform == "win32":  # pragma: no cover - platform specific
            return self._read_windows()
        return read_psi()

    def _read_windows(self) -> PagingReading | None:  # pragma: no cover - platform specific
        if self._failed:
            return None
        try:
            if self._query is None:
                pdh = ctypes.WinDLL("pdh")
                for fn, argtypes in (
                    ("PdhOpenQueryW", [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]),
                    (
                        "PdhAddEnglishCounterW",
                        [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
                    ),
                    ("PdhCollectQueryData", [ctypes.c_void_p]),
                    (
                        "PdhGetFormattedCounterValue",
                        [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(_PdhValue)],
                    ),
                ):
                    getattr(pdh, fn).argtypes = argtypes
                    getattr(pdh, fn).restype = ctypes.c_long
                query, counter = ctypes.c_void_p(), ctypes.c_void_p()
                if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                    raise OSError("PdhOpenQueryW failed")
                if pdh.PdhAddEnglishCounterW(query, "\\Memory\\Pages Output/sec", 0, ctypes.byref(counter)) != 0:
                    raise OSError("PdhAddEnglishCounterW failed")
                pdh.PdhCollectQueryData(query)
                self._pdh, self._query, self._counter = pdh, query, counter
                return None
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return None
            value = _PdhValue()
            if self._pdh.PdhGetFormattedCounterValue(self._counter, _PDH_FMT_DOUBLE, None, ctypes.byref(value)) != 0:
                return None
            pages = float(value.doubleValue)
            return PagingReading(pages, "pages out/s", pages >= MEMORY_WATCH_PAGES_OUT_PER_S)
        except Exception:
            logger.debug("Paging counter unavailable", exc_info=True)
            self._failed = True
            return None


def read_psi() -> PagingReading | None:
    """Linux memory pressure stall information: the share of the last 10 s some task waited on memory."""
    try:
        with open("/proc/pressure/memory", encoding="ascii") as psi:
            for line in psi:
                if line.startswith("some "):
                    avg10 = float(line.split()[1].split("=")[1])
                    return PagingReading(avg10, "% stalled (avg10)", avg10 >= MEMORY_WATCH_PSI_SOME_AVG10)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gb(size: float) -> str:
    return f"{size / GB:.1f} GB"


def _round_gb(size: float | None) -> float | None:
    return None if size is None else round(size / GB, 2)


class MemoryManager:
    """Makes room before each model load and each heavy call, across every registered `IdleReleased`.

    A load needs the model's peak (`MODEL_MEMORY_GB`). A call on a loaded model needs what
    the call adds on top of what the model holds (peak minus `MODEL_RESIDENT_GB`): SAM 3
    holds 3.6 GB but a call takes it 3.5 GB higher, on every call, not just the first. A
    call adding at least `CALL_CHECK_MIN_GB` reserves that much while it runs, so concurrent
    calls still run together when there is room, and one that would not fit waits for the
    others to finish rather than stacking its peak on theirs.

    Args:
        budget_gb: Optional cap on the summed size of loaded models. Soft: idle models are
            released to stay under it, but a load is never refused for the budget alone.
        reserve_gb: Free memory every load and heavy call must leave for the OS and client.
        read_status: Source of live free memory; replaceable for tests.
        read_process: Source of this process's private bytes, used to measure each load.
        call_wait_s: How long a heavy call that does not fit waits for others between checks.
    """

    def __init__(
        self,
        budget_gb: float | None = None,
        reserve_gb: float = MEMORY_RESERVE_GB,
        read_status: Callable[[], MemoryStatus] = read_memory_status,
        read_process: Callable[[], int | None] = process_private_bytes,
        call_wait_s: float = 5.0,
    ) -> None:
        self._budget = int(budget_gb * GB) if budget_gb else None
        self._reserve = int(reserve_gb * GB)
        self._read_status = read_status
        self._read_process = read_process
        self._call_wait_s = call_wait_s
        self._caches: list[IdleReleased[Any]] = []
        self._measured: dict[str, int] = {}
        self._load_lock = threading.Lock()
        # Extra bytes reserved by heavy calls now running, keyed by (thread id, call number).
        self._calls = threading.Condition()
        self._reserved: dict[tuple[int, int], tuple[str, int]] = {}
        self._next_call = 0
        self._watcher: threading.Thread | None = None
        self._stop_watching = threading.Event()
        # The latest paging reading and how many watcher checks in a row it has been high.
        self._paging: PagingReading | None = None
        self._paging_high_checks = 0
        # What happened lately, newest last, for `status()`: loads, releases and why, refusals.
        self._events: deque[dict[str, str]] = deque(maxlen=EVENT_LOG_SIZE)

    def register(self, cache: IdleReleased[Any]) -> None:
        self._caches.append(cache)

    def record(self, event: str, model: str, detail: str) -> None:
        """Note an event for `status()`; thread-safe (deque appends are atomic)."""
        self._events.append(
            {
                "time": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                "event": event,
                "model": model,
                "detail": detail,
            }
        )

    def status(self) -> dict[str, Any]:
        """What is loaded, what it holds, live free memory, running heavy calls and recent events.

        Backs the `server_status` tool, so a caller can see why a call was slow (a reload after
        a release) or refused, without reading the server's log.
        """
        now = time.monotonic()
        machine = self._read_status()
        with self._calls:
            running = list(self._reserved.values())
        models = []
        for cache in sorted(self._caches, key=lambda c: (not c.loaded, c.name)):
            row: dict[str, Any] = {
                "model": cache.name,
                "loaded": cache.loaded,
                "holds_gb": _round_gb(self.resident(cache.name)) if cache.loaded else 0.0,
                "peak_gb": _round_gb(self.estimate(cache.name)),
                "call_adds_gb": _round_gb(self.call_extra(cache.name)),
            }
            if cache.loaded:
                row["in_use"] = not cache.idle
                row["idle_s"] = round(now - cache.last_used)
            models.append(row)
        return {
            "machine": {
                "ram_free_gb": _round_gb(machine.available_ram),
                "ram_total_gb": _round_gb(machine.total_ram),
                "hard_limit": machine.commit_label,
                "hard_limit_free_gb": _round_gb(machine.available_commit),
            },
            "reserve_gb": _round_gb(self._reserve),
            "budget_gb": _round_gb(self._budget),
            "loaded_total_gb": _round_gb(sum(self.resident(c.name) for c in self._caches if c.loaded)),
            "models": models,
            "heavy_calls_running": [{"model": name, "reserved_gb": _round_gb(size)} for name, size in running],
            "paging": None
            if self._paging is None
            else {
                "value": round(self._paging.value, 1),
                "unit": self._paging.unit,
                "high": self._paging.high,
                "high_checks_in_a_row": self._paging_high_checks,
            },
            "recent_events": list(self._events),
        }

    def estimate(self, name: str) -> int:
        """Bytes a load of `name` is expected to need at its peak.

        A size measured on an earlier load in this process can raise the table's figure
        but never lower it: the measurement is the resident size after loading, and the
        load itself can peak higher.
        """
        table = int(MODEL_MEMORY_GB.get(name, DEFAULT_MODEL_MEMORY_GB) * GB)
        return max(table, self._measured.get(name, 0))

    def resident(self, name: str) -> int:
        """Bytes a loaded `name` is taken to hold between calls, for the budget."""
        peak = MODEL_MEMORY_GB.get(name, DEFAULT_MODEL_MEMORY_GB)
        table = int(MODEL_RESIDENT_GB.get(name, peak) * GB)
        return max(table, self._measured.get(name, 0))

    def call_extra(self, name: str) -> int:
        """Bytes one call on a loaded `name` adds at its peak, from the two tables."""
        peak = MODEL_MEMORY_GB.get(name, DEFAULT_MODEL_MEMORY_GB)
        held = MODEL_RESIDENT_GB.get(name, peak)
        return max(int((peak - held) * GB), 0)

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
            self.record("loaded", cache.name, f"in {time.monotonic() - started:.1f}s")
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

    @contextmanager
    def calling(self, cache: IdleReleased[Any]) -> Iterator[None]:
        """Bracket one call on a loaded model: reserve what it adds while it runs."""
        extra = self.call_extra(cache.name)
        if extra < CALL_CHECK_MIN_GB * GB:
            yield
            return
        key = self._reserve_call(cache, extra)
        try:
            yield
        finally:
            with self._calls:
                del self._reserved[key]
                self._calls.notify_all()

    def reserved_by_others(self) -> int:
        """Bytes reserved by heavy calls on other threads.

        Never counts this thread's own: a call made from inside another call would
        otherwise wait for its caller to finish, which it never would.
        """
        me = threading.get_ident()
        with self._calls:
            return sum(size for (thread, _), (_, size) in self._reserved.items() if thread != me)

    def _reserve_call(self, cache: IdleReleased[Any], extra: int) -> tuple[int, int]:
        with self._calls:
            while True:
                status = self._read_status()
                running = self.reserved_by_others()
                short = self._shortfalls(cache, extra + running, status, budget=False)
                if not short:
                    break
                if running:
                    # Other heavy calls hold memory this one needs; their ends free it.
                    self._calls.wait(self._call_wait_s)
                    continue
                released = self._release_one(exclude=cache)
                if released is not None:
                    logger.info(
                        "Released %s to make room for a %s call (short on %s)", released, cache.name, ", ".join(short)
                    )
                    self.record("released", released, f"to make room for a {cache.name} call (short on {short[0]})")
                    continue
                if "commit" in short:
                    self.record("refused", cache.name, f"call: {status.describe()}")
                    raise ModelMemoryError(
                        f"Not enough memory to run {cache.name}: a call adds about {_gb(extra)} plus a "
                        f"{_gb(self._reserve)} reserve, and the machine has {status.describe()}. Loaded in this "
                        f"server, all in use: {', '.join(self.loaded()) or 'nothing'}. Free memory elsewhere "
                        "(other model processes) and retry. `server_status` shows what is loaded."
                    )
                logger.warning("Calling %s while short on %s (%s)", cache.name, ", ".join(short), status.describe())
                break
            self._next_call += 1
            key = (threading.get_ident(), self._next_call)
            self._reserved[key] = (cache.name, extra)
            return key

    def _shortfalls(self, cache: IdleReleased[Any], need: int, status: MemoryStatus, budget: bool = True) -> list[str]:
        short = []
        if status.available_commit is not None and status.available_commit < need + self._reserve:
            short.append("commit")
        if status.available_ram is not None and status.available_ram < need + self._reserve:
            short.append("RAM")
        if budget and self._budget is not None:
            held = sum(self.resident(c.name) for c in self._caches if c.loaded and c is not cache)
            if held + need > self._budget:
                short.append("budget")
        return short

    def _make_room(self, cache: IdleReleased[Any], need: int) -> None:
        while True:
            status = self._read_status()
            # Heavy calls already running may not have reached their peak yet.
            short = self._shortfalls(cache, need + self.reserved_by_others(), status)
            if not short:
                logger.info("Loading %s (needs about %s; %s)", cache.name, _gb(need), status.describe())
                return
            released = self._release_one(exclude=cache)
            if released is None:
                break
            logger.info("Released %s to make room for %s (short on %s)", released, cache.name, ", ".join(short))
            self.record("released", released, f"to make room for loading {cache.name} (short on {short[0]})")

        still_loaded = ", ".join(self.loaded()) or "nothing"
        if "commit" in short:
            self.record("refused", cache.name, f"load: {status.describe()}")
            raise ModelMemoryError(
                f"Not enough memory to load {cache.name}: it needs about {_gb(need)} plus a {_gb(self._reserve)} "
                f"reserve, and the machine has {status.describe()}. Loaded in this server, all in use: "
                f"{still_loaded}. Free memory elsewhere (other model processes) or retry once other calls finish. "
                "`server_status` shows what is loaded."
            )
        logger.warning(
            "Loading %s while short on %s (%s; loaded and in use: %s)",
            cache.name,
            ", ".join(short),
            status.describe(),
            still_loaded,
        )

    def start_watcher(self, interval_s: float = MEMORY_WATCH_INTERVAL_S) -> None:
        """Start the background thread that runs `relieve_pressure` every `interval_s`."""
        if self._watcher is not None:
            return
        self._stop_watching.clear()
        self._watcher = threading.Thread(target=self._watch, args=(interval_s,), name="memory-watcher", daemon=True)
        self._watcher.start()

    def stop_watcher(self) -> None:
        if self._watcher is None:
            return
        self._stop_watching.set()
        self._watcher.join(timeout=5)
        self._watcher = None

    def _watch(self, interval_s: float) -> None:
        paging = PagingMonitor()
        while not self._stop_watching.wait(interval_s):
            try:
                self.relieve_pressure(paging=paging.read())
            except Exception:
                logger.warning("Memory watcher check failed", exc_info=True)

    def relieve_pressure(self, now: float | None = None, paging: PagingReading | None = None) -> list[str]:
        """Release idle models while the machine is short of memory; the names released.

        `paging` is this check's `PagingMonitor` reading. High on `MEMORY_WATCH_PAGING_CHECKS`
        checks in a row while free RAM is under the tight level, it releases one model idle
        `MEMORY_WATCH_MIN_IDLE_S` or longer per check: the OS is paging other programs out to
        keep RAM free, which free RAM alone does not show. It ranks below the two critical
        tiers and above "RAM tight".

        Three tiers by free memory, checked in this order, each releasing least-recently-used
        idle models until its own level is regained (so it does not flap at the boundary):

        - free **commit** under `MEMORY_WATCH_CRITICAL_GB`: any idle model, until
          `MEMORY_WATCH_TARGET_GB` is free. Allocations fail past this; age doesn't matter.
        - free **RAM** under `MEMORY_WATCH_CRITICAL_GB`: models idle `MEMORY_WATCH_MIN_IDLE_S`
          or longer, until `MEMORY_WATCH_TARGET_GB` is free. The age floor keeps it from
          dropping a model between two steps of one tool call (`caption` uses several).
        - free RAM under `MEMORY_WATCH_TIGHT_FRACTION` of the total: models idle
          `MEMORY_WATCH_TIGHT_IDLE_S` or longer, until that fraction is free again. This is
          the idle timer running faster while memory is tight.
        """
        now = time.monotonic() if now is None else now
        if paging is not None:
            self._paging = paging
            self._paging_high_checks = self._paging_high_checks + 1 if paging.high else 0
        status = self._read_status()
        tier = self._pressure_tier(status)
        if (tier is None or tier[3] == "RAM tight") and self._paging_sustained(status):
            assert self._paging is not None
            # A floor of 0 is always regained, so this releases exactly one model per check.
            tier = (
                MEMORY_WATCH_MIN_IDLE_S,
                "available_ram",
                0,
                f"paging: {self._paging.value:.0f} {self._paging.unit}",
            )
        released: list[str] = []
        while tier is not None:
            min_idle_s, field, floor, why = tier
            name = self._release_one(exclude=None, min_idle_s=min_idle_s, now=now)
            if name is None:
                break
            released.append(name)
            status = self._read_status()
            logger.info("Released %s: %s (%s)", name, why, status.describe())
            self.record("released", name, why)
            free = getattr(status, field)
            if free is None or free >= floor:
                break
        return released

    def _paging_sustained(self, status: MemoryStatus) -> bool:
        if self._paging_high_checks < MEMORY_WATCH_PAGING_CHECKS:
            return False
        if status.available_ram is None or not status.total_ram:
            return False
        return status.available_ram < status.total_ram * MEMORY_WATCH_TIGHT_FRACTION

    def _pressure_tier(self, status: MemoryStatus) -> tuple[float, str, int, str] | None:
        """(minimum idle seconds, MemoryStatus field, free bytes to regain, reason), or None."""
        critical = int(MEMORY_WATCH_CRITICAL_GB * GB)
        target = int(MEMORY_WATCH_TARGET_GB * GB)
        commit, ram = status.available_commit, status.available_ram
        if commit is not None and commit < critical:
            return 0.0, "available_commit", target, f"{status.commit_label} nearly exhausted"
        if ram is not None and ram < critical:
            return MEMORY_WATCH_MIN_IDLE_S, "available_ram", target, "RAM nearly exhausted"
        if ram is not None and status.total_ram:
            tight = int(status.total_ram * MEMORY_WATCH_TIGHT_FRACTION)
            if ram < tight:
                return MEMORY_WATCH_TIGHT_IDLE_S, "available_ram", tight, "RAM tight"
        return None

    def _release_one(
        self, exclude: IdleReleased[Any] | None, min_idle_s: float = 0.0, now: float | None = None
    ) -> str | None:
        """Release the least recently used idle model; its name, or None if none could go."""
        now = time.monotonic() if now is None else now
        candidates = [
            c for c in self._caches if c is not exclude and c.loaded and c.idle and now - c.last_used >= min_idle_s
        ]
        for cache in sorted(candidates, key=lambda c: c.last_used):
            if cache.try_release():
                return cache.name
        return None
