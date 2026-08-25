#  idle.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Keeps a model loaded while it is being used and releases it once it goes idle.

Loading Florence-2 or Moondream costs several seconds, so unloading after every
call makes each request noticeably slower. Holding them forever, on the other
hand, keeps gigabytes of resident memory tied up between bursts of work.

`IdleReleased` sits between the two: the object is built on first use, reused for
as long as calls keep arriving, and dropped once `timeout` seconds pass with no
activity. The next call transparently rebuilds it.

Either end of that trade-off is still reachable. A `timeout` of 0 never schedules
a release, holding the object for the process's lifetime; an `IdleProxy` built with
`release_after_call=True` drops it the moment each call returns, which is the
lowest-memory setting available and what the CLI's "instant" memory mode selects.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def trim_working_set() -> None:
    """Return pages freed by the allocator to the operating system.

    Dropping the last reference to a model and running a collection is not enough
    on Windows: CPython and torch hold on to the freed arenas, so the process
    keeps its resident size and the memory stays unavailable to everything else.
    Asking for a working set trim releases those pages. This is a no-op on other
    platforms, where the allocator already returns memory on its own.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        kernel32.SetProcessWorkingSetSize(kernel32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Could not trim the working set", exc_info=True)


class IdleReleased[T]:
    """Builds an object on demand and releases it after a period of inactivity.

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

    def get(self) -> T:
        """Return the object, building it if necessary, and restart the idle countdown."""
        with self._lock:
            if self._value is None:
                logger.info("Loading %s", self._name)
                self._value = self._factory()
            self._schedule_release()
            return self._value

    def release(self) -> None:
        """Drop the object now and hand its memory back to the operating system."""
        with self._lock:
            self._cancel_timer()
            if self._value is None:
                return
            self._value = None
        # Collect outside the lock: a call that arrives mid-collection should be
        # free to start rebuilding rather than block on it.
        gc.collect()
        trim_working_set()
        if self._timeout > 0:
            logger.info("Released %s after %.0fs idle", self._name, self._timeout)
        else:
            logger.info("Released %s", self._name)

    def _schedule_release(self) -> None:
        self._cancel_timer()
        if self._timeout <= 0:
            return
        self._timer = threading.Timer(self._timeout, self.release)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class IdleProxy:
    """Forwards attribute access to an `IdleReleased` object.

    Every lookup goes through `get()`, so any method call both loads the model if
    it was released and resets the idle countdown. This lets the proxy stand in
    for a `Florence2` or `Moondream` instance without restating their methods.

    Args:
        cache: The `IdleReleased` holding the object to forward to.
        release_after_call: Release the object as soon as each forwarded method
            returns, rather than waiting for an idle timer. The idle timer cannot
            express this on its own: it is restarted by attribute *lookup*, so any
            timeout short enough to fire promptly would also fire mid-inference,
            while the caller still holds a live reference and nothing is freed.
    """

    def __init__(self, cache: IdleReleased[Any], release_after_call: bool = False) -> None:
        self._cache = cache
        self._release_after_call = release_after_call

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._cache.get(), name)
        if not self._release_after_call or not callable(attr):
            return attr

        @functools.wraps(attr)
        def call_and_release(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            finally:
                self._cache.release()

        return call_and_release
