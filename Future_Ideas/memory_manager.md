# Memory manager: a RAM budget across models, instead of only an idle timer

**Status:** idea, 2026-10-24. Nothing measured or built yet. Raised by the user after a crash the same day.

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
