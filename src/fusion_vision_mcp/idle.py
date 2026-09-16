"""Keeps a model loaded while it is being used and releases it once it goes idle.

Loading Florence-2 or Moondream costs several seconds, so unloading after every
call makes each request noticeably slower. Holding them forever, on the other
hand, keeps gigabytes of resident memory tied up between bursts of work.

`IdleReleased` sits between the two: the object is built on first use, reused for
as long as calls keep arriving, and dropped once `timeout` seconds pass with no
activity. The next call transparently rebuilds it.

A `timeout` of 0 never schedules a release, holding the object for the process's
lifetime -- the fastest and most memory-hungry end of that trade-off, which the
CLI's "persistent" memory mode selects.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Resolved once. Re-assigning `argtypes` on every trim was pure overhead.
_KERNEL32 = None
if sys.platform == "win32":  # pragma: no cover - platform specific
    try:
        _KERNEL32 = ctypes.windll.kernel32
        _KERNEL32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    except Exception:  # noqa: BLE001 - optional Win32 diagnostics; never fatal
        _KERNEL32 = None


def trim_working_set() -> None:
    """Return pages freed by the allocator to the operating system.

    Dropping the last reference to a model and running a collection is not enough
    on Windows: CPython and torch hold on to the freed arenas, so the process
    keeps its resident size and the memory stays unavailable to everything else.
    Asking for a working set trim releases those pages. This is a no-op on other
    platforms, where the allocator already returns memory on its own.
    """
    if _KERNEL32 is None:
        return
    try:
        _KERNEL32.SetProcessWorkingSetSize(_KERNEL32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Could not trim the working set", exc_info=True)


class IdleReleased[T]:
    """Builds an object on demand and releases it after a period of inactivity.

    The idle countdown measures time since the last *completed* use, and a release
    never interrupts a call that is still running: `get()` only marks the object as
    in use, so callers that hold it for the duration of a call must bracket that
    work with `in_use()`. `IdleProxy` does this automatically for method calls.

    Args:
        factory: Called to build the object. May be called again after a release.
        timeout: Seconds of inactivity before the object is released.
        name: Human readable name, used in log messages.
    """

    def __init__(self, factory: Callable[[], T], timeout: float, name: str) -> None:
        self._factory = factory
        self._timeout = timeout
        self._name = name
        self._value: T | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._in_flight = 0

    def get(self) -> T:
        """Return the object, building it if necessary, and restart the idle countdown."""
        with self._lock:
            if self._value is None:
                logger.info("Loading %s", self._name)
                self._value = self._factory()
            self._last_used = time.monotonic()
            self._arm_timer()
            return self._value

    def in_use(self) -> _InUse:
        """Context manager marking a call in progress, so a release waits for it."""
        return _InUse(self)

    def _enter(self) -> None:
        with self._lock:
            self._in_flight += 1

    def _exit(self) -> None:
        with self._lock:
            self._in_flight -= 1
            self._last_used = time.monotonic()

    def release(self) -> None:
        """Drop the object now and hand its memory back to the operating system."""
        with self._lock:
            self._cancel_timer()
            if self._value is None:
                return
            self._value = None
            # Collect under the lock so a concurrent `get()` cannot rebuild the model
            # while the old one is still being collected -- that briefly doubled
            # resident memory on the very path that exists to reduce it.
            gc.collect()
            trim_working_set()
        if self._timeout > 0:
            logger.info("Released %s after %.0fs idle", self._name, self._timeout)
        else:
            logger.info("Released %s", self._name)

    def _arm_timer(self) -> None:
        """Start the idle timer if it is not already running.

        Deliberately *not* a cancel-and-recreate on every access: `IdleProxy` calls
        `get()` once per attribute lookup, so that spawned a fresh OS thread per
        lookup. One timer runs at a time and re-arms itself if the object turns out
        to still be in use when it fires.
        """
        if self._timeout <= 0 or self._timer is not None:
            return
        self._timer = threading.Timer(self._timeout, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if self._value is None:
                return
            idle_for = time.monotonic() - self._last_used
            # A call still running, or use since the timer was armed: wait out the
            # remainder rather than tearing down a model that is mid-inference.
            if self._in_flight > 0 or idle_for < self._timeout:
                self._arm_timer()
                return
        self.release()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class _InUse:
    """Marks a span during which the cached object must not be released."""

    def __init__(self, cache: IdleReleased[Any]) -> None:
        self._cache = cache

    def __enter__(self) -> None:
        self._cache._enter()

    def __exit__(self, *exc: object) -> None:
        self._cache._exit()


class IdleProxy:
    """Forwards attribute access to an `IdleReleased` object.

    Every lookup goes through `get()`, so any method call both loads the model if
    it was released and resets the idle countdown. This lets the proxy stand in
    for a `Florence2` or `Moondream` instance without restating their methods.

    Callables are wrapped so the idle timer sees the *call*, not just the lookup:
    an inference longer than the timeout would otherwise be released while still
    running.
    """

    def __init__(self, cache: IdleReleased[Any]) -> None:
        # Bypass __getattr__ for our own attribute.
        object.__setattr__(self, "_cache", cache)

    def release(self) -> None:
        """Release the cached object without loading it first."""
        self._cache.release()

    def __getattr__(self, name: str) -> Any:
        cache = self._cache
        attr = getattr(cache.get(), name)
        if not callable(attr):
            return attr

        def _tracked(*args: Any, **kwargs: Any) -> Any:
            with cache.in_use():
                return attr(*args, **kwargs)

        return _tracked
