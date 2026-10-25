# Memory manager: a RAM budget across models, instead of only an idle timer

**Status:** 2026-10-25: diagnosed (commit exhaustion, see below); steps M1, M2, M2b and M2c done; M3-M5 and option C open. Raised by the user after a
crash on 2026-10-24.

## Diagnosis (2026-10-25): it was commit exhaustion

The Windows event logs settle candidate 1 without a reproduction:

- **System log, Resource-Exhaustion-Detector (event 2004), 2026-10-24 22:04:46 and 22:21:38:** "Windows successfully
  diagnosed a low virtual memory condition." Top consumers: one `python.exe` at **26.8 GB** of commit (then 17.1 GB at
  the second event), another at 7.9 GB, NVIDIA Overlay 4.8 GB. The 26.8 GB process fits the test server, which keeps
  every model it touches.
- **Application log, 2026-10-23 20:47 and 23:03:** two `python.exe` crashes in `c10.dll` (torch's allocator), exception
  `0xc0000005`. An access violation in the allocator is what a failed allocation under commit pressure looks like.
- **The commit limit moves.** The pagefile (`G:\pagefile.sys`) is system-managed: 80 GB allocated and a commit limit of
  111.5 GB on 2026-10-25, peak pagefile use 15.7 GB. Windows grows it *after* a low-memory event, so the limit at the
  moment of a crash can be far lower than what a later reading shows. A check has to read it live, every time.

So the hard guard is **available commit** (what Windows refuses to hand out), and the soft one is **available
physical memory** (past it, loads page instead of failing, which is slow but survivable).

## Plan

Each step is shippable alone; M1 is the one that stops the crashes.

- **M1. Load-time guard, in-process (built 2026-10-25).** A torch-free `memory.py`: `MemoryManager` reads live memory
  (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on Linux, psutil if installed elsewhere; none of them a new
  dependency) and every `IdleReleased` consults it before its factory runs. "Commit" below is Windows' commit; on
  Linux and macOS the hard limit is free RAM + free swap, which is what the OOM killer acts on. Loads are serialised (one at a time, so
  two calls can't both pass the check and then both load). Before a load of estimated size S:
  1. if available commit < S + reserve, or available RAM < S + reserve, or (with `--memory-budget`) the loaded
     models' sizes + S exceed the budget: release the least recently used *idle* model (never one in flight, never by
     blocking on its lock) and re-read; repeat;
  2. still short on **commit** with nothing left to release: raise `ModelMemoryError` ("not enough memory to load X:
     N GB of commit free, needs about M GB; loaded: ..."), which reaches the client as a tool error instead of a dead
     process;
  3. still short on RAM or budget only: load anyway, with a warning.
  This applies in `persistent` mode too: persistent means "no idle timer", not "crash rather than release". That is
  what fixes the test suite, whose session server is persistent.
  Every load logs the estimate, live RAM / commit, and the measured private-bytes delta afterwards; a measured size
  can raise the estimate for later reloads in the same process, never lower it (it is the resident size, and a load
  can peak higher). PP-OCRv6, EasyOCR and ImageQuality build their sessions on first *call*, not in the constructor,
  so their measurement reads ~0 and only their table estimate guards them; all three are 0.5 GB or less.
  Sizes start from a static table in `constants.py` (`MODEL_MEMORY_GB`), from `CLAUDE.md`'s figures where they exist
  and parameter counts otherwise.
  **Built 2026-10-25:** `memory.py`, `IdleReleased(manager=...)` + `try_release()`, `--memory-budget GB`,
  `tests/test_memory.py` (12 tests). End to end through a spawned server: the check and measurement log on load.
  **The real-model suite passes: 523 of 523 in 10 min 34 s** (2026-10-25, persistent test server, nothing else
  running, 13 GB RAM free at the start), where on 2026-10-24 it died twice. Its log: 31 loads, 25 idle models
  released to make room (all for RAM; e.g. the ONNX LFM copy for the bf16 one and back), no load that went ahead
  short, no refusal. The default never needed `--memory-budget`.
- **M2. Measure the size table (done).** One script, one fresh subprocess per model: load it, run one representative call,
  read the process's *peak* commit (`PeakPagefileUsage`, exact per process on Windows). Replaces the M1 estimates.
  **Done 2026-10-25** (VisionBench `benchmarks/model_memory.py`, `results/model_memory.json`; rule fixed before
  running: max(peak commit, peak working set) above the post-import baseline x 1.1, rounded up to 0.1 GB). New table
  (GB, old estimate in brackets): Florence-2 2.9 (2.0), Moondream 5.0 (7.5), LFM bf16 7.0 (6.4), LFM ONNX 8.7 (8.0),
  SAM2 1.5 (0.6), SAM 3 7.1 (3.5), CLIP 2.0 (1.7), Grounding DINO 2.1 (0.7), DINOv2 1.4 (1.2), EasyOCR 1.7 (0.5),
  PP-OCRv6 0.5 (0.3), ImageQuality 0.2 (0.3). Two corrections to M1's reading of its suite log: Moondream's +3.6 GB
  was right (it peaks at 4.5 GB; the 7.5 GB in CLAUDE.md was stale), not memory-mapped weights going uncounted; the
  real undercount is call-time peaks, which no after-load figure sees (SAM 3 3.2 GB loaded, 6.5 GB peak).
  Loads every model once (~10 loads, one at a time); run only when no benchmark holds a model.
- **M2b. Check heavy calls, not only loads (done 2026-10-25).** M2 showed a call can need far more than a loaded model
  holds (SAM 3 3.6 GB held, 3.5 more inside every call), and a call on a loaded model had no check at all. Now
  `constants.MODEL_RESIDENT_GB` holds M2's after-load figures beside the peaks, and every proxied call on a model whose
  call adds at least `CALL_CHECK_MIN_GB` (0.25) *reserves* that extra while it runs (`MemoryManager.calling`): it fits
  -> runs, beside other calls if there is room; doesn't fit and other heavy calls are running -> waits for them
  instead of stacking its peak on theirs; nothing running -> releases idle models, then refuses (commit) or warns
  (RAM). Loads count running calls' reservations too. A thread never waits on its own reservation, so a call made
  from inside another can't deadlock. Checked calls: SAM 3, Grounding DINO, SAM2, LFM ONNX, Moondream, Florence-2,
  LFM bf16, and the OCR/IQA engines' first-call loads are covered by their load check. Tools run in worker threads
  (`anyio.to_thread`), so a waiting call blocks only itself.
- **M2c. A watcher for pressure between calls (done 2026-10-25).** Loads and calls only check when they happen; if
  another process grows while this server sits idle, nothing reacted until the idle timer. `MemoryManager` now runs a
  daemon thread (`start_watcher`, started and stopped by `app_lifespan`, in every memory mode) that reads free memory
  every `MEMORY_WATCH_INTERVAL_S` (5 s; the read is microseconds, so polling beats Windows' low-memory notification on
  portability for no real cost) and runs `relieve_pressure`: free commit under 4 GB -> release any idle model until
  6 GB is free; free RAM under 4 GB -> models idle 30 s or more, until 6 GB; free RAM under 25% of the total -> models
  idle 2 minutes or more, until 25% again (the idle timer running faster when memory is tight; on 31.5 GB that is
  about 7.9 GB free). The 30 s floor keeps it from dropping a model between two steps of one tool call. Thresholds
  were set from the reserve and the machine, not tuned; M3 is where they get tested.
- **Option C, not started: one server shared by every client.** Claude Code, Cline and Antigravity each spawn their
  own server and can each hold an 8.7 GB LFM copy; a single local HTTP server would load it once. Changes client
  configuration, so it waits on the user.
- **M3. Stress test (the judging rule below).** VisionBench script: concurrent calls across the big models with a
  second model-loading process beside the server. Pass: no process dies, a refused load is a clear tool error, single-
  client median latency within 5%.
- **M4. Report it.** A `server_status` tool or a field in error results: loaded models, their sizes, live RAM /
  commit. Small once M1 exists.
- **M5. Cross-process lock (opt-in), only if M3 shows M1 isn't enough.** M1 already sees other processes' memory
  through the live reading; a lock only adds queueing.
- **Not a manager step but bigger than any of them:** dropping Moondream2 (~7.5 GB) once `critique_composition`'s and
  `count_objects`' questions are gated on LFM, and one LFM copy for default and `deep`.

**Original note (2026-10-24):** idea; nothing measured or built yet. Raised by the user after a crash the same day.

## Why

Today the server's only memory control is time-based. `--memory-mode` (`aggressive` 5 min / `standard` 10 min /
`persistent`) sets how long each model stays loaded after its last call (`idle.py`: `IdleReleased`, `IdleProxy`, and a
Windows working-set trim on release). Nothing looks at **how much** is loaded, or at what else is running on the machine.

The models have grown. Approximate resident sizes from `CLAUDE.md`: LFM2.5-VL-3B is loaded **twice**, the ONNX copy
(~8 GB at peak) that serves both the default caption and, since 2026-10-24 (5fac21d), `query_image`, and the bf16 torch
copy (~6.4 GB) for `caption(deep)`. Moondream2 (~7.5 GB fp32) no longer answers `query_image` but still loads for
`critique_composition`'s background and subject questions and `count_objects`' VQA estimate. Add DINOv2-large ~1.2 GB,
CLIP ViT-L/14 ~1.7 GB (now loaded on every default caption for the `Image type` line), Florence-2-large, SAM 3, SAM2,
Grounding DINO, EasyOCR and PP-OCRv6. One session that touches `caption(deep)`, `critique_composition` and
`detect_objects` within ten minutes can hold well over 20 GB on a 31.5 GB laptop. Other model processes run beside it: VisionBench harnesses,
the pytest suite's own server, other chats.

**What happened (2026-10-24).** The FusionVisionMCP pre-commit pytest run (which spawns a real server) overlapped a
VisionBench job loading Florence-2 and EasyOCR. The test server died mid-suite: 15 tests got `MCPError: Connection
closed`, 3 got empty replies. The VisionBench Florence-2 process died silently after 11 of 37 images. Both ran fine
alone. The cause was **not confirmed** (likely memory exhaustion or the OS killing a process; no log was kept). A
client sees the same thing as "the server disconnected", with no message saying why.

**It then happened with nothing else running.** A second pre-commit run the same day, alone on the machine with about
11 GB free at the start, died the same way (18 tests `Connection closed`). So overlap is not the only trigger: the test
server keeps every model it touches loaded, and since the LFM changes that set alone (both LFM copies, Moondream2, SAM 3,
CLIP, Florence-2, ...) may no longer fit. Already in place as a stopgap: `CLAUDE.md` (45aa1a0) says to run the real-model
suite only while no benchmark holds a model, and the overnight queue waits for pytest to finish (`wait_for_pytest.ps1`).
Neither helps when the suite alone is too big.

## What a memory manager could do (candidates, cheapest first)

1. **Diagnose first.** Reproduce the overlap with per-process memory logged (psutil RSS per second, plus Windows
   event log / commit-charge), to confirm it is memory and which model load tips it over. Everything below depends on
   this.
2. **Pre-load check.** Before a factory loads a model, compare its known size (a small table measured once per model
   and dtype) with free memory. If it doesn't fit, release least-recently-used idle models first (they already know how
   to release themselves). If it still doesn't fit, return a clear tool error ("not enough memory to load X: N GB
   free, needs M GB") instead of crashing the process.
3. **A budget across models**, e.g. `--memory-budget 12GB`: keep loaded models under it by LRU eviction, on top of the
   idle timer. This is the "manager" proper: the timer answers *when idle*, the budget answers *how much at once*.
4. **One heavy inference at a time.** A semaphore around the big models, so two concurrent tool calls don't load two
   multi-GB models at once.
5. **Cross-process coordination (machine level).** An opt-in lock file or named mutex that heavy jobs (the server's
   loads, VisionBench runs, pytest) take, so they queue instead of overlapping. The overnight queue
   (`VisionBench/scratch/night_queue/run_night.ps1`) already does this for scheduled jobs by running them one at a time.
6. **Report it.** Each tool result (or a `server_status` tool) could include what is loaded and its memory, so a calling
   agent or the user can see why a call was slow or refused.

## Constraints and cautions

- **The import must stay torch-free** (see `CLAUDE.md`): the manager runs on the eager path, so it can't import a model
  wrapper or torch to measure sizes. Keep a static size table, refreshed by a VisionBench script.
- **No eviction mid-inference.** `IdleProxy` already defers release while a call is in flight; a budget evictor has to
  respect the same in-flight counter.
- Re-load cost is real (LFM and Moondream take tens of seconds), so an aggressive budget trades memory for latency.
  Measure both before picking a default.
- Cross-platform: free-memory reads differ on Windows, macOS and Linux (psutil covers all three; CI runs Ubuntu).
- Not a substitute for fewer or smaller models. LFM took over `query_image` (2026-10-24); gating it for
  `critique_composition`'s two questions and `count_objects`' estimate would let Moondream2's ~7.5 GB fp32 go entirely,
  and sharing one LFM copy between the default and `deep` (if ONNX could serve `deep`) would drop ~6.4 GB more. Either
  would do more than any manager.
- The test suite is a load case of its own: a session-scoped server that keeps every model loaded is the worst case by
  construction. A budget (or releasing models between test modules) applies there too.

## How it would be judged

A reproducible stress script in VisionBench: N concurrent tool calls across the big models, with another
model-loading process beside it. The pass rule: no process dies; a call that can't fit gets a clear error; and median
latency on the normal single-client path is unchanged (within 5%).
