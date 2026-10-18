# Handoff: C4-3b gate 2, Gemma 3 4B against Moondream2 for the `deep` caption, tune split

**Model:** Sonnet 5.5, in a new session. Opus 5.5 fixed these rules on 2026-10-18 before anything ran; don't change
them. If one can't be followed, stop and report.
**Plan:** `Future_Ideas/caption_sprint_c4_plan.md`, "C4-3b" (with its gate 1 result) and "Rules carried over". Read
both first, and gate 1's script `VisionBench/benchmarks/caption_c4_3b_gate1.py`.
**Time:** about 1.5 to 2 CPU hours (120 Gemma captions at ~45 s, plus OCR).
**When:** only while nothing else is running: not during C4-2a or the phase 11 scoreboard run.

## The question

Does Gemma 3 4B's caption, with the `Visible text` line, name more of the labelled key facts than the shipped `deep`
caption, at the same length and without inventing more? This is the tune split. A win here doesn't ship anything:
it goes to the user, and then to the fresh holdout.

## Gemma: exactly as gate 1

Same venv (`VisionBench/.venv-gguf`), files (Google's `gemma-3-4b-it-qat-q4_0-gguf`), runtime (`llama-cpp-python`
0.3.36, `MTMDChatHandler`), prompt (`Describe this image in detail.`, image first, one user turn), greedy
(`temperature=0`), `max_tokens=512`, `n_ctx=4096`, 18 threads, one uncounted warm-up. Reuse gate 1's code rather
than rewriting it. **Don't raise `max_tokens` or change the prompt:** gate 1 showed either pushes Gemma over 45 s.

Two clean-ups, applied the same way to both captioners before scoring, because they're what would ship:
- **Loop stop:** `fusion_vision_mcp.repetition.cut_loop` on every caption (Moondream2's shipped `deep` already
  does this live).
- **Cut-off captions:** when Gemma stops at the token limit (`finish_reason == "length"`), drop the unfinished last
  sentence: cut back to the last `.`, `!` or `?`. Count how many captions this touched.

## Baseline: the shipped `deep` caption

The cached sampled `moondream_long` rows for the 120 tune images in `VisionBench/benchmarks/results/caption_ab_tune.jsonl`
(the `deep` baseline C4-0 and C4-1 used), with `cut_loop` applied. Don't re-caption them.

## Scoring

Images: all 120 of `split_items(rv.sourced_items())["tune"]`.

Each caption is scored as the shipped `caption` returns it: the prose, a space, then
`analysis.visible_text_line(ocr_text)` at today's 80 words, with OCR from `analysis.ocr_pages` (FusionVisionMCP's
`.venv`; cache it per id, and record its seconds).

- **Recall:** caption-keyword groups found (`run_visionbench.keyword_group_hit`), Gemma minus baseline, with a
  5,000-sample image bootstrap (seed 1).
- **Matched length:** each image's two proses cut to the shorter one's word count, then the text line added to both
  and scored the same way. Same bootstrap.
- **Key phrases** (`run_visionbench.phrase_found`) on the images that have them: report only.
- `absent_object_named` (`keyword_group_hit` on the labelled absent object) for each.
- Captions with a repeated 8-word span (`caption_c4_0.has_loop`), counted after `cut_loop`.
- Median prose words, and median seconds for Gemma caption + OCR, measured in this run.
- Per kind: recall for each of the nine kinds (report only).

## Rule ("Gemma wins on tune" means all of these)

1. Recall, Gemma minus baseline: the 95% CI is above 0.
2. Matched length: the point estimate is above 0 **and** its 95% CI lower bound is above -1.0. Gemma writes about
   1.8x as many words, and keyword recall rewards length, so the length-matched gain carries most of the weight.
3. `absent_object_named`: at most 1 more than the baseline.
4. Repeated 8-word spans: at most 2 more than the baseline.
5. Speed: median Gemma caption + OCR <= 45 s over the 120, measured in this run.

## Also measure: memory with the server's models loaded (report only)

After the 120 captions, with Gemma still loaded, load Florence-2-large and Moondream2 in a second process from
FusionVisionMCP's `.venv` (their wrappers, CPU) and caption `tests/sample.jpg` with each. Report system available
memory and both processes' resident memory at that point. If available memory falls under 2 GB, stop that step
right away and say so. This tells the user what shipping Gemma beside the server's models would cost.

## Outputs and records

- Script: `VisionBench/benchmarks/caption_c4_3b_gate2.py`. Commit it before the run, by path (never
  `git commit -a`).
- Rows: `benchmarks/results/caption_c4_3b_gate2.jsonl` (per image: id, kind, Gemma caption before and after the
  clean-ups, finish reason, tokens, seconds). Summary: `caption_c4_3b_gate2.json`.
- Add a "Gate 2 result (date)" paragraph under C4-3b in the plan, and a pointer in the memory file
  `project_caption_c4_status.md`.
- **Don't push.** List `origin/main..HEAD` for the user.
- **Stop after scoring.** Whether it wins or loses, report to the user. Don't run the confirm split, the fresh
  holdout, CHAIR or anything that ships.

## Don'ts

- No MCP vision tools and no native vision on these images.
- No other prompts, token limits, decodings or quantisations (that would be fitting the tune split).
- Don't change FusionVisionMCP's source, `pyproject.toml` or its environments.
- No Chinese-developed models. Gemma comes from Google's own repo only.
