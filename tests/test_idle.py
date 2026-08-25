#  test_idle.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import threading
import time

import pytest

from fusion_vision_mcp.idle import IdleProxy, IdleReleased


class Counter:
    """Stands in for a model: records how often it was built and answers calls."""

    builds = 0

    def __init__(self) -> None:
        type(self).builds += 1

    def double(self, value: int) -> int:
        return value * 2


def test_builds_lazily() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=60, name="counter")

    assert Counter.builds == 0

    cache.get()
    assert Counter.builds == 1


def test_reuses_while_active() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=60, name="counter")

    first = cache.get()
    second = cache.get()

    assert first is second
    assert Counter.builds == 1


def test_releases_after_timeout() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=0.1, name="counter")

    cache.get()
    time.sleep(0.4)

    # The next call has to build a second instance, proving the first was dropped.
    cache.get()
    assert Counter.builds == 2


def test_use_postpones_the_release() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=0.3, name="counter")

    first = cache.get()
    for _ in range(4):
        time.sleep(0.1)
        # Each call restarts the countdown, so the object should survive well past
        # the timeout as long as requests keep arriving.
        assert cache.get() is first

    assert Counter.builds == 1


def test_zero_timeout_never_releases() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=0, name="counter")

    cache.get()
    time.sleep(0.3)
    cache.get()

    assert Counter.builds == 1


def test_release_is_safe_when_nothing_is_loaded() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=60, name="counter")

    cache.release()
    cache.release()

    assert Counter.builds == 0


def test_proxy_forwards_calls_and_reloads() -> None:
    Counter.builds = 0
    proxy = IdleProxy(IdleReleased(Counter, timeout=0.1, name="counter"))

    assert proxy.double(21) == 42
    assert Counter.builds == 1

    time.sleep(0.4)

    # Released in the meantime, but the proxy rebuilds transparently.
    assert proxy.double(4) == 8
    assert Counter.builds == 2


def test_release_after_call_drops_the_object_once_each_call_returns() -> None:
    """The "instant" memory mode: no timer, released the moment the work finishes."""
    Counter.builds = 0
    proxy = IdleProxy(IdleReleased(Counter, timeout=0, name="counter"), release_after_call=True)

    assert proxy.double(21) == 42
    assert Counter.builds == 1

    # No sleep: a second call must rebuild immediately, so the first was already gone.
    assert proxy.double(4) == 8
    assert Counter.builds == 2


def test_release_after_call_still_releases_when_the_call_raises() -> None:
    """A failed inference must not strand the model in memory."""
    Counter.builds = 0
    proxy = IdleProxy(IdleReleased(Counter, timeout=0, name="counter"), release_after_call=True)

    with pytest.raises(TypeError):
        proxy.double()  # missing the required argument
    assert Counter.builds == 1

    # Having to build again proves the failed call still handed the first one back.
    assert proxy.double(3) == 6
    assert Counter.builds == 2


def test_release_after_call_leaves_plain_attributes_alone() -> None:
    """Only calls trigger a release; reading an attribute must not wrap or drop anything."""
    Counter.builds = 0
    proxy = IdleProxy(IdleReleased(Counter, timeout=0, name="counter"), release_after_call=True)

    assert proxy.builds == 1


def test_a_proxy_without_release_after_call_keeps_the_object() -> None:
    """The default has to stay unchanged: no release until the idle timer says so."""
    Counter.builds = 0
    proxy = IdleProxy(IdleReleased(Counter, timeout=60, name="counter"))

    proxy.double(1)
    proxy.double(2)

    assert Counter.builds == 1


def test_concurrent_use_builds_once() -> None:
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=60, name="counter")
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        seen.append(cache.get())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert Counter.builds == 1
    assert all(item is seen[0] for item in seen)
