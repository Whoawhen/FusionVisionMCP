# Handoff: C4-5, an `Image type:` line on the caption from CLIP zero-shot

**Model:** Sonnet 5.5, in a new session. Opus 5.5 fixed these rules, and the label list, on 2026-10-18 before
anything ran; don't change them. If one can't be followed, stop and report.
**Plan:** `Future_Ideas/caption_sprint_c4_plan.md`: "Rules carried over", C4-0b's miss list (27 of 111 groups both
slots miss are image type or genre), and the C4-2b, C4-2a and C4-3b results. Read them first.
**When:** steps 1-2 take minutes and can run any time nothing is timing; step 4 (round 11 captions, ~3 CPU hours)
must not overlap the phase 11 scoreboard run or any other benchmark.

## The idea

Both captioners often miss what *kind* of image it is (title screen, invoice, engraving, comic, packaging).
`classify_image`'s CLIP ViT-L/14 picks between fixed labels well (gate Z2: 98.5% on species, landmarks and styles,
0.25 s). So `caption` would end its prose with `Image type: <label>.` from a fixed label list, before the
`Visible text` line. The captioners never see it; it's extra output for the caller. If it ships, the same label can
later drive a `suggest_deep` hint (report only here; whether to route is the user's call).

**Known risk, stated up front:** Opus wrote the label list after seeing C4-0b's tune miss list, so the tune split
flatters it. That's why the decision is on round 11's holdout (step 4), which no caption test has used, plus the
user's own check of the labels (step 3).

## The label list (fixed; don't add, drop or reword)

CLIP prompt for each label: `"a " + label`, or `"an " + label` before a vowel. The caption prints the label itself.

```
photograph, black and white photograph, aerial photograph, close-up photograph, portrait photograph,
landscape photograph, product photograph,
oil painting, watercolor painting, pencil drawing, ink drawing, engraving, etching, lithograph, woodcut,
fresco, mosaic, digital painting,
cartoon, comic strip, comic page, manga page, anime illustration, vector illustration, clip art,
children's illustration, poster, infographic,
website screenshot, mobile app screenshot, desktop software screenshot, code editor screenshot,
video game screenshot, video game title screen, video game menu screen, video game inventory screen,
scanned document, typed letter, handwritten note, form, invoice, receipt, certificate, newspaper page,
printed page of text, data table, restaurant menu,
bar chart, line chart, pie chart, scatter plot, flowchart, diagram, technical drawing, map, timeline,
street sign, shop sign, information panel, product packaging, product label, logo, banner,
something else
```

`something else` is the catch-all: when it wins, no line is added.

## Fixed settings

- Classifier: `Aesthetic.zero_shot` (what `classify_image` runs), on FusionVisionMCP's `.venv`, CPU.
- Show the line only when the top label's probability is at least the **threshold**, picked on tune in step 1 from
  `{0.2, 0.3, 0.4, 0.5}` only (the one with the highest default-slot recall; ties go to the higher threshold).
- Format, appended to the prose with one space: `Image type: <label>.` then the `Visible text` line as `caption`
  adds it (80 words).

## Step 1: tune, 120 images, cached captions (minutes)

- Images: `split_items(rv.sourced_items())["tune"]`. Captions: cached `florence_detailed` (default) and
  `moondream_long` (`deep`, with `repetition.cut_loop`) in `benchmarks/results/caption_ab_tune.jsonl`. OCR text:
  `caption_c4_3b_gate2_ocr.jsonl` (same 120 ids; check they all match).
- For each slot and threshold: caption-keyword recall (`keyword_group_hit`), base (prose + text line) against
  +type (prose + type line + text line), 5,000-sample image bootstrap (seed 1); `absent_object_named`; how often
  the line is shown.
- Report only: how often the label agrees with the image's labelled `kind` (map each label to the kinds it fits;
  write that map into the script before running, and commit it), and which labels came up.
- **Tune passes if,** at the picked threshold: default +type minus base, 95% CI above 0; `deep` point estimate not
  below 0; `absent_object_named` not higher in either slot. If it fails, C4-5 stops here.

## Step 2: CHAIR, 300 COCO images (minutes)

`caption_hallucination.py`'s 300 images and cached `florence_detailed` / `moondream_long` rows, with and without
the type line (picked threshold), scored with its `_stats`. Store as new variants in your own results file, not in
`caption_chair.jsonl`. **Passes if** CHAIR-i and CHAIR-s each rise by at most 1.0 point in both slots (some labels
could name a COCO object; that's what this catches).

## Step 3: the user checks the labels (blind)

Draw 40 images at random (seed 20261018) from tune images where the line is shown, and build a review page in the
pattern of `VisionBench/benchmarks/round18_label_check_serve.py`: the image, the label, and keys for right /
wrong / partly. The user does this, not you, and you don't use any vision on the images. **Passes if** at least 32
of 40 are right (partly counts as half).

## Step 4: round 11 holdout (the decision; ~3 CPU hours, only on a quiet machine)

Round 11's 235 images at labels commit `fa06afc` (as `caption_text_cap_guards.py` reads them). Caption each with
the shipped default and `deep` paths (`Florence2.caption(..., MORE_DETAILED)` and `Moondream.caption(length="long")`,
18 threads), OCR with `ocr_pages`, CLIP labels at the picked threshold. Resumable rows.
- **Ships if:** default +type minus base keyword recall, 95% CI above 0; `deep` point estimate not below 0;
  `absent_object_named` not higher in either slot.
- Report only: key-phrase recall, per-kind recall, label agreement with `kind`, and how often a label maps to a
  kind where `deep` beats the default (screenshots, documents, charts, game UI, signage): the `suggest_deep` case.

## Outputs and records

- Script: `VisionBench/benchmarks/caption_c4_5_type.py` (steps `tune`, `chair`, `review`, `holdout`, `score`).
  Commit it, with the label-to-kind map, before step 1 runs, by path (never `git commit -a`).
- Results: `benchmarks/results/caption_c4_5_*.json(l)`.
- Add a "C4-5: image-type line" section to the plan with each step's result as it lands, and a pointer in the
  memory file `project_caption_c4_status.md`.
- **Stop after step 3 and report to the user** before starting step 4 (it needs a quiet machine and their timing).
- **Don't ship and don't push.** If step 4 passes, report; shipping (constants, `caption` code and description,
  tests, CLAUDE.md) waits for the user's go-ahead.

## Don'ts

- No MCP vision tools and no native vision on these images.
- No other labels, prompts, templates or thresholds than the ones above.
- Don't change FusionVisionMCP's source or environments.
- No Chinese-developed models.
- Don't run the repo's whole pre-commit config while anything is timing: its pytest hook loads models.
