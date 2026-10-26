import io
import sys
import threading
import time

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
    assert manager.resident("small") == GB  # nor what it is taken to hold between calls


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


# --- heavy calls: what a call adds on top of a loaded model -------------------------------------

#: "SAM 3" is in both tables, so a call on it adds peak minus resident; the test names above add nothing.
HEAVY = "SAM 3"


def heavy_setup(commit_gb: float) -> tuple[MemoryManager, IdleReleased[Model], IdleReleased[Model], Machine]:
    machine = Machine(ram_gb=None, commit_gb=100)
    manager = MemoryManager(reserve_gb=0, read_status=machine.status, read_process=lambda: None, call_wait_s=0.05)
    heavy = IdleReleased(Model, timeout=0, name=HEAVY, manager=manager)
    other = IdleReleased(Model, timeout=0, name="other", manager=manager)
    machine.caches = [heavy, other]
    heavy.get()
    other.get()
    machine.commit_gb = commit_gb  # from here on, free memory is what the test says
    return manager, heavy, other, machine


def test_light_calls_are_not_checked() -> None:
    manager, _, other, _ = heavy_setup(commit_gb=0)

    assert manager.call_extra("other") == 0
    assert IdleProxy(other).answer() == "ok"  # no memory at all, and nothing checked


def test_a_heavy_call_that_fits_reserves_its_extra_only_while_running() -> None:
    manager, heavy, _, _ = heavy_setup(commit_gb=10)
    extra = manager.call_extra(HEAVY)
    seen: list[int] = []

    class Probe(Model):
        def answer(self) -> str:
            # Another thread sees this call's reservation; this thread does not count its own.
            box: list[int] = []
            thread = threading.Thread(target=lambda: box.append(manager.reserved_by_others()))
            thread.start()
            thread.join()
            seen.append(box[0])
            seen.append(manager.reserved_by_others())
            return "ok"

    heavy._factory = Probe
    heavy.release()
    heavy.get()

    assert IdleProxy(heavy).answer() == "ok"
    assert extra > 3 * GB and seen == [extra, 0]
    assert manager.reserved_by_others() == 0


def test_a_heavy_call_releases_an_idle_model_to_fit() -> None:
    manager, heavy, other, machine = heavy_setup(commit_gb=0)
    # Loaded heavy + other hold 2 GB of the simulated machine; releasing other frees 1.
    machine.commit_gb = 2 + manager.call_extra(HEAVY) / GB - 0.5  # 0.5 GB short until other goes

    assert IdleProxy(heavy).answer() == "ok"
    assert not other.loaded and heavy.loaded


def test_a_heavy_call_that_cannot_fit_is_refused() -> None:
    _, heavy, other, _ = heavy_setup(commit_gb=2.5)  # 0.5 GB free, nowhere near a SAM 3 call

    with other.in_use(), pytest.raises(ModelMemoryError) as excinfo:
        IdleProxy(heavy).answer()

    assert "run SAM 3" in str(excinfo.value)
    assert heavy.loaded and other.loaded  # refusing the call drops nothing


def test_a_heavy_call_waits_for_another_rather_than_stacking_on_it() -> None:
    manager, heavy, _, machine = heavy_setup(commit_gb=0)
    extra_gb = manager.call_extra(HEAVY) / GB
    machine.commit_gb = 2 + extra_gb * 1.5  # room for one call's extra, not two
    first_running = threading.Event()
    let_first_finish = threading.Event()
    order: list[str] = []

    class Slow(Model):
        def answer(self) -> str:
            if threading.current_thread().name == "first":
                first_running.set()
                let_first_finish.wait(5)
            order.append(threading.current_thread().name)
            return "ok"

    heavy._factory = Slow
    heavy.release()
    heavy.get()
    proxy = IdleProxy(heavy)
    first = threading.Thread(target=proxy.answer, name="first")
    second = threading.Thread(target=proxy.answer, name="second")
    first.start()
    first_running.wait(5)
    second.start()
    time.sleep(0.3)
    assert order == []  # second is waiting, not running beside first
    let_first_finish.set()
    first.join(5)
    second.join(5)

    assert order == ["first", "second"]


def test_a_nested_heavy_call_does_not_wait_on_its_own_caller() -> None:
    manager, heavy, _, machine = heavy_setup(commit_gb=0)
    machine.commit_gb = 2 + manager.call_extra(HEAVY) / GB * 1.5
    proxy = IdleProxy(heavy)

    class Nested(Model):
        def answer(self) -> str:
            return "ok"

        def outer(self) -> str:
            return proxy.answer()  # a second heavy call from inside the first, same thread

    heavy._factory = Nested
    heavy.release()
    heavy.get()
    done: list[str] = []
    thread = threading.Thread(target=lambda: done.append(proxy.outer()))
    thread.start()
    thread.join(3)

    assert done == ["ok"]


# --- the watcher: releasing idle models when something else squeezes the machine ----------------


def watched(
    ram_gb: float | None, commit_gb: float | None, total_ram_gb: float | None = 32
) -> tuple[MemoryManager, list[IdleReleased[Model]], Machine]:
    machine = Machine(ram_gb, commit_gb)

    def status() -> MemoryStatus:
        base = machine.status()
        total = None if total_ram_gb is None else int(total_ram_gb * GB)
        return MemoryStatus(base.available_ram, base.available_commit, total_ram=total)

    manager = MemoryManager(reserve_gb=0, read_status=status, read_process=lambda: None)
    caches = [IdleReleased(Model, timeout=0, name=name, manager=manager) for name in ("old", "mid", "new")]
    machine.caches = caches
    for cache in caches:
        cache.get()
    machine.ram_gb, machine.commit_gb = ram_gb, commit_gb  # now what the test says, with all three loaded
    return manager, caches, machine


def at(cache: IdleReleased[Model], seconds_ago: float, now: float) -> None:
    cache._last_used = now - seconds_ago


def test_watcher_does_nothing_with_room_to_spare() -> None:
    manager, caches, _ = watched(ram_gb=3 + 20, commit_gb=3 + 50)

    assert manager.relieve_pressure() == []
    assert all(c.loaded for c in caches)


def test_critical_commit_releases_any_idle_model_until_the_target() -> None:
    now = 1000.0
    # 3 loaded; free commit 2 GB (critical, under 4); each release frees 1; the target is 6 free.
    manager, (old, mid, new), _ = watched(ram_gb=None, commit_gb=3 + 2)
    for cache, ago in ((old, 5), (mid, 3), (new, 1)):
        at(cache, ago, now)

    released = manager.relieve_pressure(now=now)

    assert released == ["old", "mid", "new"]  # oldest first, age no object; 5 free is still under 6


def test_critical_commit_never_releases_a_model_mid_call() -> None:
    manager, (old, _, _), _ = watched(ram_gb=None, commit_gb=3 + 2)

    with old.in_use():
        released = manager.relieve_pressure()

    assert "old" not in released and old.loaded


def test_critical_ram_spares_models_used_in_the_last_half_minute() -> None:
    now = 1000.0
    manager, (old, mid, new), _ = watched(ram_gb=3 + 2, commit_gb=None)
    at(old, 600, now)
    at(mid, 45, now)
    at(new, 10, now)  # likely between two steps of one tool call

    released = manager.relieve_pressure(now=now)

    assert released == ["old", "mid"] and new.loaded


def test_tight_ram_releases_only_models_idle_two_minutes() -> None:
    now = 1000.0
    # 32 GB total, tight under 8 GB free; 7.5 GB free is tight but not critical.
    manager, (old, mid, new), _ = watched(ram_gb=3 + 7.5, commit_gb=None)
    at(old, 300, now)
    at(mid, 200, now)
    at(new, 60, now)

    released = manager.relieve_pressure(now=now)

    assert released == ["old"]  # 8.5 GB free after one release: tight no longer


def test_tight_needs_the_total_to_be_known() -> None:
    manager, caches, _ = watched(ram_gb=3 + 7.5, commit_gb=None, total_ram_gb=None)
    for cache in caches:
        at(cache, 900, time.monotonic())

    assert manager.relieve_pressure() == []


def test_watcher_thread_runs_and_stops() -> None:
    manager, (old, _, _), _ = watched(ram_gb=None, commit_gb=3 + 2)

    manager.start_watcher(interval_s=0.02)
    deadline = time.monotonic() + 3
    while old.loaded and time.monotonic() < deadline:
        time.sleep(0.02)
    manager.stop_watcher()

    assert not old.loaded
    assert not any(t.name == "memory-watcher" for t in threading.enumerate())


# --- status: what server_status reports ---------------------------------------------------------


def test_status_reports_models_memory_and_events() -> None:
    manager, (a, b, c) = setup(["a", "b", "c"], ram_gb=2.5, commit_gb=None)
    a.get()
    b.get()
    c.get()  # pushes out the stalest, a

    status = manager.status()

    assert status["machine"]["ram_free_gb"] == 0.5
    assert status["loaded_total_gb"] == 2.0
    assert [(m["model"], m["loaded"]) for m in status["models"]] == [("b", True), ("c", True), ("a", False)]
    assert status["models"][0]["in_use"] is False and status["models"][0]["idle_s"] >= 0
    assert "idle_s" not in status["models"][2]
    events = [(e["event"], e["model"]) for e in status["recent_events"]]
    assert events == [("loaded", "a"), ("loaded", "b"), ("released", "a"), ("loaded", "c")]
    assert "to make room for loading c" in status["recent_events"][2]["detail"]


def test_status_records_refusals_and_running_heavy_calls() -> None:
    manager, heavy, other, _ = heavy_setup(commit_gb=2.5)

    with other.in_use(), pytest.raises(ModelMemoryError) as excinfo:
        IdleProxy(heavy).answer()

    assert "server_status" in str(excinfo.value)
    assert manager.status()["recent_events"][-1]["event"] == "refused"

    seen: list[list[dict[str, object]]] = []

    class Probe(Model):
        def answer(self) -> str:
            seen.append(manager2.status()["heavy_calls_running"])
            return "ok"

    manager2, heavy2, _, _ = heavy_setup(commit_gb=20)
    heavy2._factory = Probe
    heavy2.release()
    heavy2.get()
    IdleProxy(heavy2).answer()
    assert seen[0][0]["model"] == HEAVY and seen[0][0]["reserved_gb"] > 3
    assert manager2.status()["heavy_calls_running"] == []


def test_idle_timer_and_watcher_releases_are_events() -> None:
    machine = Machine(100, 100)
    manager = MemoryManager(reserve_gb=0, read_status=machine.status, read_process=lambda: None)
    cache = IdleReleased(Model, timeout=0.1, name="a", manager=manager)
    machine.caches = [cache]
    cache.get()
    deadline = time.monotonic() + 3  # generous: the full suite keeps the machine busy
    while manager.status()["recent_events"][-1]["event"] != "released" and time.monotonic() < deadline:
        time.sleep(0.05)
    last = manager.status()["recent_events"][-1]
    assert (last["event"], last["detail"]) == ("released", "idle timer")

    manager, _, _ = watched(ram_gb=None, commit_gb=3 + 2)
    manager.relieve_pressure()
    assert manager.status()["recent_events"][-1]["detail"] == "commit nearly exhausted"


# --- paging: the pressure free RAM misses ---------------------------------------------------------

HIGH = memory.PagingReading(2000.0, "pages out/s", True)
LOW = memory.PagingReading(0.0, "pages out/s", False)


def test_sustained_paging_with_ram_under_tight_releases_one_model_per_check() -> None:
    now = 1000.0
    # 32 GB total: 4.6 GB free is above the 4 GB critical level but under the 8 GB tight one, as in S3.
    manager, (old, mid, new), _ = watched(ram_gb=3 + 4.6, commit_gb=None)
    at(old, 80, now)
    at(mid, 40, now)
    at(new, 10, now)

    assert manager.relieve_pressure(now=now, paging=HIGH) == []  # one high check is not yet sustained
    assert manager.relieve_pressure(now=now, paging=HIGH) == ["old"]  # one per check, not all at once
    assert manager.relieve_pressure(now=now, paging=HIGH) == ["mid"]
    assert manager.relieve_pressure(now=now, paging=HIGH) == []  # new was used 10 s ago
    assert manager.status()["recent_events"][-1]["detail"] == "paging: 2000 pages out/s 0s ago"
    assert manager.status()["paging"]["high_checks_in_a_row"] == 4


def test_paging_with_ram_to_spare_releases_nothing() -> None:
    now = 1000.0
    manager, caches, _ = watched(ram_gb=3 + 12, commit_gb=None)  # 12 of 32 GB free: not tight
    for cache in caches:
        at(cache, 600, now)

    assert manager.relieve_pressure(now=now, paging=HIGH) == []
    assert manager.relieve_pressure(now=now, paging=HIGH) == []


def test_a_low_reading_resets_the_streak() -> None:
    now = 1000.0
    manager, caches, _ = watched(ram_gb=3 + 4.6, commit_gb=None)
    for cache in caches:
        at(cache, 90, now)  # idle past 30 s, short of the tight tier's 120 s

    for reading in (HIGH, LOW, HIGH):
        assert manager.relieve_pressure(now=now, paging=reading) == []


def test_a_burst_is_remembered_while_models_age_past_the_idle_floor() -> None:
    """The S3 re-run's shape: a 10 s burst while the models were just used, then quiet."""
    start = 1000.0
    manager, (old, mid, new), _ = watched(ram_gb=3 + 4.6, commit_gb=None)
    at(old, 5, start)
    at(mid, 3, start)
    at(new, 1, start)

    assert manager.relieve_pressure(now=start, paging=HIGH) == []
    assert manager.relieve_pressure(now=start + 5, paging=HIGH) == []  # a burst, but nothing 30 s idle yet
    assert manager.relieve_pressure(now=start + 10, paging=LOW) == []
    assert manager.relieve_pressure(now=start + 26, paging=LOW) == ["old"]  # 31 s idle; burst 21 s ago
    assert manager.relieve_pressure(now=start + 31, paging=LOW) == ["mid"]
    assert manager.relieve_pressure(now=start + 36, paging=LOW) == ["new"]


def test_a_burst_is_forgotten_after_a_minute() -> None:
    start = 1000.0
    manager, caches, _ = watched(ram_gb=3 + 4.6, commit_gb=None)
    for cache in caches:
        at(cache, 0, start)

    manager.relieve_pressure(now=start, paging=HIGH)
    manager.relieve_pressure(now=start + 5, paging=HIGH)

    # 61 s after the burst every model is 66 s idle, but the burst no longer counts (and 66 s < the tight tier's 120 s).
    assert manager.relieve_pressure(now=start + 66, paging=LOW) == []


def test_a_burst_with_ram_recovered_releases_nothing() -> None:
    start = 1000.0
    manager, caches, machine = watched(ram_gb=3 + 4.6, commit_gb=None)
    for cache in caches:
        at(cache, 0, start)
    manager.relieve_pressure(now=start, paging=HIGH)
    manager.relieve_pressure(now=start + 5, paging=HIGH)
    machine.ram_gb = 3 + 12  # the squeeze ended

    assert manager.relieve_pressure(now=start + 40, paging=LOW) == []


def test_psi_is_read_from_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    psi = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123\nfull avg10=4.00 avg60=1.00 avg300=0.50 total=45\n"
    monkeypatch.setattr(memory, "open", lambda *_a, **_k: io.StringIO(psi), raising=False)

    reading = memory.read_psi()

    assert reading == memory.PagingReading(12.5, "% stalled (avg10)", True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows performance counters")
def test_windows_paging_counter_reads_a_rate() -> None:
    monitor = memory.PagingMonitor()

    assert monitor.read() is None  # a rate needs two samples
    time.sleep(0.2)
    reading = monitor.read()
    assert reading is not None and reading.value >= 0 and reading.unit == "pages out/s"
