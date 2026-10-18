# Handoff: C4-2a, one beam instead of three for the default caption

**Model:** Sonnet 5.5, in a new session. Opus 5.5 fixed these rules on 2026-10-18 before anything ran; don't change
them. If one can't be followed, stop and report.
**Plan:** `Future_Ideas/caption_sprint_c4_plan.md`, "Sprint C4-2", item a. Read it, and "Rules carried over", first.
**Time:** about 1 to 1.5 CPU hours if it gets past the smoke test, 15 minutes if it doesn't.
**When:** only while nothing else is running: not during C4-3b gate 1 (Gemma timing) or the phase 11 scoreboard
run. Seconds measured under load aren't comparable.

## The question

`Florence2._run` decodes every task with `num_beams=3`. Does one beam, for the caption task only, make the default
caption meaningfully faster without making it worse? The default slot has about 1.9 s of headroom in its 15 s
budget; whatever this saves is the budget for C4-2c (an object list from dense regions). This is a speed test:
it needs to save time and lose nothing.

## How to vary the beams

Don't edit FusionVisionMCP's source; it's the live server. In the script, load `fusion_vision_mcp.florence2.Florence2`
(Florence-2-large, CPU, the way `caption_ab.py`'s `florence_detailed` does) and call its model's `generate` with
the same arguments as `_run` except `num_beams`. A small helper that copies `_run`'s steps for the
`<MORE_DETAILED_CAPTION>` task is fine. Check it first: with `num_beams=3`, the helper's caption must equal
`Florence2.caption(images, CaptionLevel.MORE_DETAILED)` on 5 images, word for word, or stop.

Use 18 torch threads (`torch.set_num_threads(18)`, as `caption_ab.py`). Run from FusionVisionMCP's `.venv`.

## Step 1: smoke, 20 images, timing

- **Images:** `split_items(rv.sourced_items())["tune"][:20]` (as `caption_c4_1.py`).
- One warm-up caption on `tests/sample.jpg`, not counted.
- Per image, in one session, alternate: 3 beams, then 1 beam, then 3 again, then 1 again. Time each call (wall
  seconds, caption only). An image's time for a setting is the mean of its two runs.
- **Stop rule:** if the median saving (3 beams minus 1 beam, per image) is **under 1.0 s**, C4-2a ends here. Record
  it and stop; don't try 2 beams or other settings.
- Also check the cached baseline: the 3-beam captions here must equal `florence_detailed` in
  `benchmarks/results/caption_ab_tune.jsonl` for the same ids. If they don't all match, step 2 re-runs the 3-beam
  captions too instead of using the cache, and you say how many differed.

## Step 2: tune split, 120 images (only if step 1 saved 1.0 s or more)

- 1-beam caption on all 120 tune images. 3-beam captions from the cache, if step 1's check passed.
- Score each caption the way the shipped `caption` returns it: the prose plus the `ocr` engine's
  `Visible text` line at today's 80 words (`analysis.visible_text_line(text)`, with OCR text from
  `analysis.ocr_pages`; cache it per id).
- **Metrics:**
  - caption-keyword recall (`run_visionbench.keyword_group_hit`), with a 5,000-sample image bootstrap (seed 1)
    on the 1-beam minus 3-beam difference;
  - matched-length recall: each 1-beam caption cut to its 3-beam caption's word count (prose only, before the
    text line is added), then scored the same way;
  - median words (prose);
  - `absent_object_named`;
  - captions with a repeated 8-word span (`caption_c4_0.has_loop`);
  - median seconds per caption.

## Step 3: CHAIR, 300 COCO images (only if step 2 passes)

`caption_hallucination.py`'s 300 images. 1-beam captions against its cached `florence_detailed` rows, scored with
its `_stats`. Store yours as a separate variant (for example `florence_detailed_b1`) in your own results file,
not in `caption_chair.jsonl`.

## Rule (all must hold)

1. Step 1: median saving >= 1.0 s.
2. Recall: 1 beam minus 3 beams >= **-1.0 point**, and matched-length recall >= -1.0 point.
3. `absent_object_named` no higher with 1 beam.
4. Captions with a repeated 8-word span: at most **2 more** than with 3 beams (beam search is partly what stops
   Florence-2's long captions looping).
5. CHAIR-i and CHAIR-s each no more than **+1.0 point** above `florence_detailed`.

If all hold, report it as **passed** with the seconds saved. Don't ship it: whether to ship one beam alone, or keep
the time for C4-2c, is the user's decision. If any fails, C4-2a stops; record which.

## Outputs and records

- Script: `VisionBench/benchmarks/caption_c4_2a.py`, steps `smoke`, `tune`, `chair`. Commit it before the first
  timed run, by path (never `git commit -a`).
- Results: `benchmarks/results/caption_c4_2a_*.jsonl` and a summary `caption_c4_2a.json`.
- Add a "C4-2a result (date)" paragraph under Sprint C4-2 in the plan, and a pointer in the memory file
  `project_caption_c4_status.md`.
- **Don't push.** List `origin/main..HEAD` for the user.

## Don'ts

- No MCP vision tools and no native vision on these images.
- Don't change any other task's beams, Florence-2's other settings, or FusionVisionMCP's source and environments.
- No Chinese-developed models.
