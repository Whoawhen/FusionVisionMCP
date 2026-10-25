import io
import sys
import threading

import pytest

from fusion_vision_mcp import memory
from fusion_vision_mcp.idle import IdleProxy, IdleReleased
from fusion_vision_mcp.memory import GB, MemoryManager, MemoryStatus, ModelMemoryError, read_memory_status


class Model:
    """Stands in for a model. Names missing from `MODEL_MEMORY_GB` are estimated at 1 GB."""

    def answer(self) -> str:
        return "ok"


class Machine:
    """Free memory that drops by 1 GB for each loaded cache, so releases show up in the next reading."""

    def __init__(self, ram_gb: float | None, commit_gb: float | None) -> None:
        self.ram_gb = ram_gb
        self.commit_gb = commit_gb
        self.caches: list[IdleReleased[Model]] = []

    def status(self) -> MemoryStatus:
        held = sum(1 for cache in self.caches if cache.loaded)
        ram = None if self.ram_gb is None else int((self.ram_gb - held) * GB)
        commit = None if self.commit_gb is None else int((self.commit_gb - held) * GB)
        return MemoryStatus(ram, commit)


def setup(
    names: list[str], ram_gb: float | None, commit_gb: float | None, **kwargs: float
) -> tuple[MemoryManager, list[IdleReleased[Model]]]:
    machine = Machine(ram_gb, commit_gb)
    manager = MemoryManager(reserve_gb=0, read_status=machine.status, read_process=lambda: None, **kwargs)
    caches = [IdleReleased(Model, timeout=0, name=name, manager=manager) for name in names]
    machine.caches = caches
    return manager, caches


def test_loads_without_releasing_when_it_fits() -> None:
    manager, (a, b) = setup(["a", "b"], ram_gb=10, commit_gb=10)

    a.get()
    b.get()

    assert manager.loaded() == ["a", "b"]


def test_releases_the_least_recently_used_idle_model_when_ram_is_short() -> None:
    # Room for two 1 GB models; the third must push one out, and it must be the stalest.
    manager, (a, b, c) = setup(["a", "b", "c"], ram_gb=2.5, commit_gb=None)
    a.get()
    b.get()
    a.get()  # a is now more recent than b

    c.get()

    assert manager.loaded() == ["a", "c"]


def test_never_releases_a_model_mid_call() -> None:
    _, (a, b) = setup(["a", "b"], ram_gb=1.5, commit_gb=None)
    a.get()

    with a.in_use():
        b.get()  # short on RAM, but a is busy: load anyway rather than tear it down

    assert a.loaded and b.loaded


def test_refuses_a_load_that_cannot_fit_in_commit() -> None:
    _, (a, b) = setup(["a", "b"], ram_gb=None, commit_gb=1.5)
    a.get()

    with a.in_use(), pytest.raises(ModelMemoryError) as excinfo:
        b.get()

    message = str(excinfo.value)
    assert "load b" in message and "commit free" in message and "all in use: a." in message
    assert a.loaded and not b.loaded
    # Once a is idle again it can make way, and the same request succeeds.
    b.get()
    assert b.loaded and not a.loaded


def test_short_on_ram_alone_loads_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    _, (a,) = setup(["a"], ram_gb=0.5, commit_gb=10)

    a.get()

    assert a.loaded
    assert "short on RAM" in caplog.text


def test_budget_releases_to_stay_under_it() -> None:
    manager, (a, b, c) = setup(["a", "b", "c"], ram_gb=100, commit_gb=100, budget_gb=2)
    a.get()
    b.get()

    c.get()

    assert manager.loaded() == ["b", "c"]


def test_a_measured_size_raises_the_estimate_but_never_lowers_it() -> None:
    readings = iter([0, 3 * GB, 0, GB // 2])
    machine = Machine(100, 100)
    manager = MemoryManager(reserve_gb=0, read_status=machine.status, read_process=lambda: next(readings))
    big = IdleReleased(Model, timeout=0, name="big", manager=manager)
    small = IdleReleased(Model, timeout=0, name="small", manager=manager)

    big.get()
    small.get()

    assert manager.estimate("big") == 3 * GB
    assert manager.estimate("small") == GB  # measured 0.5 GB; the 1 GB table figure stands
    assert manager.resident("small") == GB // 2


def test_try_release_does_not_wait_for_a_held_lock() -> None:
    cache = IdleReleased(Model, timeout=0, name="a")
    cache.get()
    held = threading.Event()
    done = threading.Event()

    def hold() -> None:
        with cache._lock:
            held.set()
            done.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(5)
    try:
        assert cache.try_release() is False
    finally:
        done.set()
        thread.join()
    assert cache.try_release() is True
    assert not cache.loaded


def test_proxy_call_reloads_a_model_released_after_lookup() -> None:
    cache = IdleReleased(Model, timeout=0, name="a")
    method = IdleProxy(cache).answer
    cache.release()  # as if the manager made room between the lookup and the call

    assert method() == "ok"
    assert cache.loaded


def test_load_failure_releases_the_load_lock() -> None:
    def broken() -> Model:
        raise RuntimeError("download failed")

    machine = Machine(10, 10)
    manager = MemoryManager(reserve_gb=0, read_status=machine.status, read_process=lambda: None)
    bad = IdleReleased(broken, timeout=0, name="bad", manager=manager)
    good = IdleReleased(Model, timeout=0, name="good", manager=manager)

    with pytest.raises(RuntimeError):
        bad.get()
    good.get()  # would hang if the failed load had kept the lock

    assert good.loaded


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="read natively only on Windows and Linux")
def test_reads_live_memory_on_this_platform() -> None:
    status = read_memory_status()

    assert status.available_ram is not None and status.available_ram > 0
    # The hard limit is read on both: commit on Windows, RAM + swap on Linux.
    assert status.available_commit is not None and status.available_commit > 0


def test_linux_hard_limit_is_ram_plus_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    meminfo = (
        "MemTotal:       24000000 kB\n"
        "MemAvailable:    4000000 kB\n"
        "SwapTotal:       8000000 kB\n"
        "SwapFree:        2000000 kB\n"
    )
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory, "open", lambda *_a, **_k: io.StringIO(meminfo), raising=False)

    status = memory.read_memory_status()

    assert status.available_ram == 4000000 * 1024
    assert status.available_commit == 6000000 * 1024
    assert "RAM + swap free" in status.describe()
