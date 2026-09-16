import threading
import time

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
    # A wide margin between the sleep and the timeout (not just a large timeout alone)
    # matters here: on a loaded CI runner, a 0.2s gap was tight enough to flake when
    # scheduling jitter delayed this thread past the release timer firing.
    cache = IdleReleased(Counter, timeout=2.0, name="counter")

    first = cache.get()
    for _ in range(4):
        time.sleep(0.3)
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


def test_a_proxy_holds_the_object_across_calls_until_the_timer_fires() -> None:
    """Repeat calls inside the idle window must reuse one instance, not rebuild per call."""
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


class SlowCounter:
    """Stands in for a model whose call outlasts the idle timeout."""

    builds = 0

    def __init__(self) -> None:
        type(self).builds += 1

    def slow(self, seconds: float) -> str:
        time.sleep(seconds)
        return "done"


def test_release_waits_for_a_call_that_outlasts_the_timeout() -> None:
    """A release must not fire mid-inference.

    The timer measures idle time, but a single call can run longer than the whole
    timeout. Releasing then dropped the model and ran gc.collect() plus a Windows
    working-set trim *while torch was executing*.
    """
    SlowCounter.builds = 0
    cache = IdleReleased(SlowCounter, timeout=0.1, name="slow")
    proxy = IdleProxy(cache)

    assert proxy.slow(0.4) == "done"
    # Still resident: the call was in flight the whole time the timer was due.
    assert cache._value is not None
    assert SlowCounter.builds == 1


def test_repeated_lookups_do_not_spawn_a_timer_each_time() -> None:
    """`__getattr__` runs per attribute lookup; each one used to build a new Timer."""
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=30, name="counter")
    proxy = IdleProxy(cache)

    before = threading.active_count()
    for _ in range(25):
        proxy.double(2)
    after = threading.active_count()

    assert after - before <= 1
    cache.release()


def test_proxy_release_does_not_load_the_object() -> None:
    """`proxy.release()` used to go through __getattr__, loading the model to release it."""
    Counter.builds = 0
    cache = IdleReleased(Counter, timeout=60, name="counter")
    proxy = IdleProxy(cache)

    proxy.release()

    assert Counter.builds == 0
