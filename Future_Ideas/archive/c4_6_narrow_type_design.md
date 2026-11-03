# C4-6: the narrow-type check (an `Image type:` line for five kinds, and a `suggest_deep` hint)

> **Result (2026-10-20): FAILED at the label check; closed.** 75.9% score, 16.0% wrong, 9 wrong on other kinds
> (needs 85% / 10% / 4); the caption run was skipped since it could change no decision. See the "C4-6 result" in
> `caption_sprint_c4_plan.md`.

**Designed by:** Opus 5.5, 2026-10-20, before round 21 exists. The list, threshold, rubric, metrics and rules below
are fixed; whoever runs it doesn't change them. If one can't be followed, stop and report.
**Runs:** the Gemini worker sources and labels round 21 (`VisionBench/worker_handoff/ROUND21_PLAN.md`); the reviewer
builds it; a Sonnet 5.5 session runs and scores; the user spot-checks keywords and grades the type labels blind, and
decides shipping.
**User decisions already made:** the narrowed list (2026-10-19), and test the hint as well as the type line
(2026-10-20, after being told what's below).

## Why, and what is already known

C4-5's 64-label line gained on tune (+3.1 keywords, CI 1.6 to 4.7) but failed C4-5b's blind label check on round 19
(v1 69.9% / 28% wrong, v2 77.9% / 19% wrong; bars 85% / 10%). Per kind, the labels were good on signage (100%), charts
(94%), app screenshots (90%), documents (86%) and game UI (85%), and bad on photos, artwork, illustrations and AI
images. Those strong kinds were picked after seeing round 19, so a fresh round decides.

Two explorations on cached data, done before this design (Opus, not gates; their images are not used here):
- **Type line, narrowed:** C4-5's tune rows, with only the five kinds' v1 labels printed, at 0.3 and the 160-word
  text line: default +2.5 (CI 1.1 to 3.9), `deep` +1.6; shown on 69 of 120 (62 of the 68 five-kind images, 7 of the
  52 others), against v1's 107 (42 on others).
- **Hint:** with the 160-word line in both slots, `deep` beats the default by about as much everywhere. On the 280
  held-out images (confirm half), keywords +5.2 points on the five kinds and +3.8 on the rest (tune half +12.9 and
  +4.4); on round 20's text phrases, +3.0. So the hint's question is whether *where the label fires* predicts a
  bigger `deep` gain, not whether `deep` gains. Expect that to be hard to show.

## The list (fixed): v3 = v2 with only the five kinds printed

C4-5b's `LABELS_V2` (`VisionBench/benchmarks/caption_c4_5b.py`), all 64 labels in the softmax, prompt form, CLIP
ViT-L/14 via `Aesthetic.zero_shot`, threshold **0.3**, unchanged. A line is printed only if the top label is one of
these 33 and its probability is at least 0.3:

- **Screens (6):** `website screenshot`, `mobile app screenshot`, `desktop software screenshot`,
  `code editor screenshot`, `video game screenshot`, `video game title screen`.
- **Documents (10):** `typed letter`, `handwritten note`, `paper form`, `invoice`, `receipt`, `certificate`,
  `newspaper page`, `printed page of text`, `data table`, `restaurant menu`.
- **Charts (10):** `infographic`, `bar chart`, `line chart`, `pie chart`, `scatter plot`, `flowchart`, `diagram`,
  `technical drawing`, `map`, `timeline`.
- **Signage (7):** `street sign`, `shop sign`, `information panel`, `product packaging`, `product label`, `logo`,
  `banner`.

The other 30 (the 12 photo labels, 10 art labels, and `cartoon`, `comic strip`, `comic page`, `manga page`,
`anime illustration`, `vector illustration`, `clip art`, `poster`) and `something else` are silent. The run script
defines `PRINTED_V3` and asserts it is these 33, a subset of `LABELS_V2`.

**Line form**, as C4-5: `Image type: <label>.` between the prose and the `Visible text` line.
**Hint form**, default caption only, when the type line prints: the type line becomes
`Image type: <label> (deep=true usually describes these more fully).` The hint's words are not scored.

## The images: round 21

Fresh Commons images (`ROUND21_PLAN.md`), built like round 20, labelled by the worker with round 4's `description`,
`caption_keywords` and `absent_object` rules, and nothing about image type. **Test set:** the first N kept per kind,
in manifest order:

| Kind | N |
| --- | --- |
| app and web screenshots, documents, charts, diagrams and infographics, game UI and HUD, signage, logos and packaging | 20 each (100) |
| everyday photos | 40 |
| artwork and paintings | 20 |
| illustrations, cartoons and comics | 20 |
| AI-generated images | 15 |

The "other" 95 are there to measure false labels and the hint's not-fired side. If a kind comes up short, run with
what there is and say so.

## The user's two checks (one sitting, keywords first)

1. **Keyword spot-check**, before anything runs on the set: 20 random test images (seed 20261020), each keyword
   group marked right (a correct caption of this image would mention it) or wrong. If under 90% of groups are right,
   stop and report.
2. **Blind type-label check** with C4-5b's rubric, word for word (right 1, partly 0.5, wrong 0; "Is this label a fair
   answer to *what kind of image is this?*"). One card per test image where v3 prints a label; shuffled, seed
   20261020; the card shows only the image and the label. CLIP may run once the worker's labels are committed; the
   user sees no caption.

## Run and metrics

- CLIP probabilities for every test image (seconds). Captions on the shipped code (the 160-word line): `caption`
  default and `deep`, plus `ocr(detail=true)`, one server session, resumable, about 3 CPU hours. **Parity:** the
  text line rebuilt from the `ocr` reply must equal the server's own on every caption, or stop.
- Each variant is the server's caption with the type line inserted before its `Visible text` line (or appended if it
  has none). **Keywords:** `run_visionbench.keyword_group_hit`, recall over all groups; `absent_object` via the same.
  5,000-sample image bootstrap, seed 1.

## Rules (fixed)

**Type line ships only if all hold:**
1. **Label check:** score at least **85%** and wrong at most **10%** of the cards shown.
2. **False labels:** wrong cards on the 95 other-kind images at most **5%** of those images (at most 4 of 95).
3. **Keywords, default:** type line minus no type line, 95% CI above 0.
4. **Keywords, `deep`:** not below 0.
5. **`absent_object` named:** not higher in either slot.

Then, before shipping: COCO CHAIR on `caption_hallucination.py`'s 300 images with v3 (C4-5's `chair` method with the
160-word line; CHAIR-i and CHAIR-s each at most +1.0, both slots); unit tests; docs.

**The hint ships only if the type line ships and both hold**, on the test set, both slots carrying the type line and
the 160-word line:
- **H1:** on images where v3 prints a label, `deep` minus default keywords at least **+3.0 points**, CI above 0.
- **H2:** that gain minus the same gain on images where it prints nothing, CI above 0 (bootstrap resampling images
  within each group).

If H1 holds and H2 doesn't, the finding is "`deep` gains about as much everywhere"; that goes into the docs, and no
hint ships.

**Report too:** coverage per kind; score and wrong per kind; the labels that came up most; seconds per slot; the
hint's firing rate; and H1/H2 with the v2 list's printed labels (report only).

## Records

- Script: `VisionBench/benchmarks/caption_c4_6.py` (steps `smoke`, `keyword_check`, `clip`, `deck`, `run`, `score`,
  `chair`), committed with `PRINTED_V3` before `clip`.
- Results: `benchmarks/results/caption_c4_6_*.json(l)`; a "C4-6 result" paragraph in the plan; a pointer in the
  memory file `project_caption_c4_status.md`.
- Nothing ships and nothing is pushed without the user.
