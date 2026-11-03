# C4-7: the text-line clean-up (cleaned `Visible text` line, 160 words, with a cut marker)

**Designed by:** Opus 5.5, 2026-10-19, before round 20 exists. The variants, metric and rules below are fixed; whoever
runs it doesn't change them. If one can't be followed, stop and report.
**Runs:** the Gemini worker sources and labels round 20 (`VisionBench/worker_handoff/ROUND20_PLAN.md`); the reviewer
builds it; a Sonnet 5.5 session runs and scores; the user spot-checks labels and decides shipping.
**User decisions already made (2026-10-19):** the cap is counted in words; the maximum is 160 words; a cut line says so.

## Why

On round 18 the shipped 80-word line keeps 57.5% of key phrases in the default caption (`deep` 63.6%) while the `ocr`
engine read 76.3%. An exploration on round 18's cached rows (Opus, scratch script, not a gate) found two levers:
- **Waste:** about a fifth of the first 80 words go on lines repeated word for word (5%), one-character lines (2%) and
  lines the prose already contains (12%). Dropping the first two at 80 words: 57.5 -> 59.6% (deep 63.6 -> 65.3).
- **Length:** half the lost phrases sit past word 148. Cleaned at 120 words: 66.7% (deep 71.3%); raw at 160: 68.9%.
Lines the prose already contains are **kept**: the quoted line is the tool's verified text and the prose is not, so
moving text out of the line would lose that guarantee. Giving every line a few words first lost 1 to 3 points.
Round 18 informed these numbers, so round 20 decides.

## The variants (fixed)

All are built from the same `ocr_pages` text as `caption` does now (lines in reading order, joined with ` / `, words
counted after the join, so each separator counts as one word, exactly as `visible_text_line` counts today).

- **baseline:** the shipped line, `visible_text_line(text)` at 80 words.
- **candidate:** clean, then cap at 160, then mark a cut.
  - **Clean:** drop a line whose normalised form (lower case, every run of non-word characters as one space,
    trimmed) is empty, has fewer than 2 characters once spaces are removed, or equals an earlier line's normalised
    form. Keep everything else in order.
  - **Cap:** the first 160 words of the cleaned, joined text; trailing ` /` trimmed as today.
  - **Marker:** if words were cut, the line ends `Visible text: "<words>…" (N more words not shown; ocr returns the
    full text).` where N is the number of cut words that aren't `/`. Uncut lines end `".` as today.
- **Report only:** clean at 80, raw at 160, and clean with no cap, each without the marker, to show where the gain
  comes from.

## The images: round 20

A fresh text-heavy set built like round 18 (`ROUND20_PLAN.md`): five kinds (app and web screenshots, documents,
charts/diagrams/infographics, game UI and HUD, signage/logos/packaging), each image with about 50 or more legible words,
transcribed and given 3-10 key phrases by the Gemini worker with round 4's rules plus round 18's "wherever it is", and
an `absent_object`. Test set: the first 30 kept per kind in manifest order (fewer if a kind comes up short). The user
spot-checks 20 random images' key phrases before any tool runs on the set (as round 18's
`round18_label_check_serve.py` did); if fewer than 90% of the checked phrases are right, stop and report.

**Amended 2026-10-19, after labelling and before any tool ran on round 20.** The worker excluded 54 of 178 images
for under 30 legible words, leaving 124 (app and web 29, documents 28, charts 22, game UI 16, signage 29). The
variants, metric and rule are unchanged. If the user tops it up (round 20b, `ROUND20B_PLAN.md`), the new images come
after round 20's in the manifest, so the test set is still the first 30 kept per kind. The 20-image spot check drawn
from the 124 stands, and about 4 of the top-up images are added to it, drawn the same way. Without a top-up, the
test runs on the 124.

## Run and metric

- Run once, shipped code, as `caption_text_cap_r18.py run` does: `caption` (default and `deep`) and `ocr(detail=true)`
  per image in one server session, resumable. Each variant is the caption's prose (its own line stripped) plus the line
  rebuilt from that `ocr` reply. **Parity:** the rebuilt baseline must equal the server's own line on every caption,
  or stop.
- **Metric:** key-phrase recall (`run_visionbench.phrase_found`) on the whole caption; 5,000-sample image bootstrap,
  seed 1, on candidate minus baseline. Also: `absent_object_named`, line words (median, p90), how often the marker
  appears, and per-kind recall.

## Rule (all must hold)

1. Default slot: candidate minus baseline >= **+3.0 points**, 95% CI above 0.
2. `deep`: candidate minus baseline not below 0.
3. `absent_object_named`: not higher in either slot.

**Then, before shipping:** COCO CHAIR with the candidate line on `caption_hallucination.py`'s 300 images (CHAIR-i and
CHAIR-s each at most +1.0 against the 80-word line, both slots); the round 11 check of `caption_text_cap_guards.py`'s
holdout step with the candidate in place of the 80-word line (the candidate names the absent object on no image where
the 80-word line doesn't); unit tests for the clean-up, the cap and the marker; and the docstring, `read_text`
description and CLAUDE.md. Shipping waits for the user's go-ahead.

## Records

- Script: `VisionBench/benchmarks/caption_text_line_r20.py` (steps `smoke`, `run`, `score`), committed before `run`.
- Results: `benchmarks/results/caption_text_line_r20*.json(l)`; a "C4-7 result" paragraph in the plan; a pointer in
  the memory file `project_caption_c4_status.md`.
- Nothing is pushed without the user.
