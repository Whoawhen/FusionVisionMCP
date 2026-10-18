# Handoff: C4-3b gate 1, Gemma 3 4B (4-bit GGUF) timing for the `deep` caption slot

**Model:** Sonnet 5.5, in a new session. Opus 5.5 fixed these rules on 2026-10-18 before anything ran; don't change
them. If one can't be followed, stop and report.
**Plan:** `Future_Ideas/caption_sprint_c4_plan.md`, section "C4-3b". Read it, and the "Rules carried over" section,
first.
**Time:** about 1 to 1.5 hours, most of it the 4 GB download and 40 caption calls.
**Deadline:** finish before the user's phase 11 scoreboard run tonight (2026-10-18). Never time anything while
another benchmark is running; seconds measured under load aren't comparable.

## The question

Can Gemma 3 4B, quantised to 4 bits and run through llama.cpp on CPU, caption an image fast enough for `deep`?
This gate is timing only. It doesn't score captions; gate 2 does that, and gate 2 waits for the user.

## Already checked (2026-10-18)

- Hugging Face access: the user's token (`Whoawhen`) can read `google/gemma-3-4b-it-qat-q4_0-gguf`, both
  `gemma-3-4b-it-q4_0.gguf` (3.16 GB) and `mmproj-model-f16-4B.gguf` (0.85 GB). 155 GB free on C:.
- Gate 0 (licence, publisher, runtime) passed: VisionBench-labels `worker_handoff/MODEL_RESEARCH_C4_VERIFIED.md`.

## Setup

1. **A separate venv, `C:\AI\Projects\VisionBench\.venv-gguf`.** Don't install anything into FusionVisionMCP's
   `.venv` or the uv tool: that is the live server. Don't touch `pyproject.toml` either; that happens only if gate 2
   passes and the user approves.
   - `uv venv C:\AI\Projects\VisionBench\.venv-gguf --python 3.13`
   - `llama-cpp-python==0.3.36` from its own CPU index (it has no Windows wheel on PyPI):
     `--index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`. If 3.13 has no wheel, use the newest Python
     that does; never build from source.
   - Plus `pillow`, `numpy`, `huggingface_hub`, `psutil`, and FusionVisionMCP itself as an editable install
     (`-e C:\AI\MCP\FusionVisionMCP[ocr-specialist]` without the torch extras, if that resolves; it's only for
     `analysis.ocr_pages` and `ppocr_engine`). If the editable install drags in torch, skip it: take OCR seconds
     from FusionVisionMCP's `.venv` in a second, separate pass over the same images.
2. **Download** both files with `huggingface_hub.hf_hub_download` into the normal HF cache. No re-quantisations
   from other uploaders.
3. **Runtime, in this order:** `llama_cpp.Llama(model_path=..., chat_handler=MTMDChatHandler(clip_model_path=mmproj),
   n_ctx=4096, n_threads=18, n_threads_batch=18)`. If `MTMDChatHandler` (or `Gemma3ChatHandler`) can't drive
   Gemma 3's vision, fall back to llama.cpp's prebuilt `win-cpu-x64` binaries from the GitHub releases (build
   `b11558` or newer, MIT): `llama-mtmd-cli -m ... --mmproj ... --image ... -p ... -t 18`, run as a subprocess. Say
   which one you used. The time to load the model doesn't count; report it separately.

## Fixed settings

- **Images:** the first 20 tune images, exactly as `VisionBench/benchmarks/caption_c4_1.py` picks them:
  `split_items(rv.sourced_items())["tune"][:20]` (from `caption_ab` and `run_visionbench`).
- **Prompt:** `Describe this image in detail.` (the Gemma 3 model card's), one user turn, with the image first.
- **Decoding:** greedy (`temperature=0`), `max_tokens=512`. Gate 2 sets the real decoding; this is for timing.
- **Threads:** at most 18 (the benchmark affinity rule). Nothing else heavy running.
- **One warm-up call** on `tests/sample.jpg` first, not counted.
- **Timed per image:** wall seconds for the Gemma caption, plus `ocr_pages` seconds on the same image (the `deep`
  path's `read_text` line). Record output words, and prompt and output tokens if the runtime reports them.
- **Reference, same session, not part of the gate:** Moondream2's shipped `deep` caption
  (`Moondream.caption([image], length="long")` from FusionVisionMCP's `.venv`, 18 torch threads) on the same 20
  images, run right after Gemma. Seconds measured in different sessions aren't comparable, so this gives the gap.
  If the two can't run from one script because of the venvs, run them one after the other with nothing else in
  between.
- **Memory:** peak resident memory of the Gemma process (`psutil`), and system available memory before loading
  and after the 20 images.

## Rule

- **Pass:** median (Gemma caption + OCR) <= **45 s** over the 20 images.
- **Over 45 s: stop.** C4-3b ends at gate 1. Record it; don't try other quantisations, thread counts or prompts
  (that would be fitting the gate).
- Either way, also report: the median and 90th percentile, words per caption, seconds per output token, load
  time, memory, and Moondream2's median from the same session.
- If a caption loops or is cut off at 512 tokens, count it and say so; don't rerun it.

## Outputs and records

- Script: `VisionBench/benchmarks/caption_c4_3b_gate1.py`. Commit it before the timed run, by path (never
  `git commit -a`).
- Rows: `VisionBench/benchmarks/results/caption_c4_3b_gate1.jsonl` (per image: id, kind, seconds for each part,
  words, tokens, caption text). Summary: `caption_c4_3b_gate1.json`.
- Add a short "Gate 1 result (date)" paragraph under C4-3b in the plan with the numbers and the verdict, and a
  pointer in the memory file `project_caption_c4_status.md`.
- **Don't push.** The user decides pushes; list `origin/main..HEAD` for them.
- **Don't start gate 2**, even on a pass. Report to the user: pass or fail, the numbers, and gate 2's cost (the
  tune split, 120 images at roughly the measured seconds each).

## Don'ts

- No MCP vision tools on these images and no native vision; this is timing.
- No Chinese-developed models or mirrors (the user's rule). Gemma is Google's; fetch it from Google's own repo
  only.
- Don't change FusionVisionMCP's source or its environments.
