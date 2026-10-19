# C4-5b: a fresh, larger label check for the `Image type:` line (v1 against a v2 list)

**Designed by:** Opus 5.5, 2026-10-18, before any image for it exists. The rubric, the v2 list, the sample, the
threshold and the pass and pick rules below are fixed; whoever runs it doesn't change them. If one can't be
followed, stop and report.
**Runs:** the Gemini worker sources round 19 (`VisionBench/worker_handoff/ROUND19_PLAN.md`); a Sonnet 5.5 session
builds the set, computes the labels, builds the review page and tallies; the user grades.
**Why:** C4-5 step 3 (plan section "C4-5") scored v1's labels 31.5 of 40 against a bar of 32. At n = 40 that is 79%
plus or minus about 13 points, so it can't separate "good enough to state as fact" from "not". Several of the
non-right calls came from overlapping labels (portrait vs photograph, product photograph, desktop software vs video
game), and CHAIR showed v1 putting `form` (20) and `scatter plot` (11) on ordinary COCO photos. This check answers
one question: **are the labels good enough to show a caller?** Recall stays step 4's question (round 11 holdout).

## What changes from the first check, stated so the two aren't mixed

- New images (round 19), never labelled or captioned by anything here.
- A written rubric (below). The first check had none; your doubts on #10, #11, #18, #29 and #33 were about what
  "right" means.
- Two lists judged blind on the same images.
- v1's first-check score (31.5/40) stays recorded as it was. Don't re-score it under the new rubric.

## The rubric (fixed)

For each card you see an image and one label, and answer: **"Is this label a fair answer to *what kind of image is
this?*"**

- **Right (1):** true, and a person who knows the image would accept it as the kind of image it is.
- **Partly (0.5):** true but misleading or much vaguer than an obvious answer (a game screen called "desktop
  software screenshot"), or it fits only part of the image.
- **Wrong (0):** false, or it would mislead someone who can't see the image.

Judge the label you're shown, not whether a better one exists on some list you can't see. Under this rubric the
first check's #29 would have been partly; don't go back and apply it there.

## The two lists

Both use v1's prompt form (`"a " + label`, `"an "` before a vowel), CLIP ViT-L/14 via `Aesthetic.zero_shot`, and
**threshold 0.3** (picked for v1 on tune in C4-5 step 1; tune can't be used to pick anything for v2, since Opus has
seen its misses). A label is shown only if it's the top label, isn't silent or the catch-all, and its probability is
at least 0.3.

**v1:** the 64 labels in `VisionBench/benchmarks/caption_c4_5_type.py` (`LABELS`), unchanged.

**v2:** v1 with these changes, each for a reason the first check or CHAIR gave:

1. **Photo labels become silent.** They still compete in the softmax, so a photo still has photo labels to land
   on, but when one wins, no line is printed. The prose already says "photo", and photos are where the false
   labels clustered (portrait and product photograph partly or wrong 4 times; `form` and `scatter plot` on COCO
   photos). Silent: `photograph`, `black and white photograph`, `aerial photograph`, `close-up photograph`,
   `portrait photograph`, `landscape photograph`, `product photograph`, plus five new ones so ordinary scenes have
   somewhere to go: `photograph of people`, `photograph of an animal`, `photograph of food`,
   `photograph of a room`, `photograph of a street`.
2. **Dropped**, because they overlapped a better label or were wrong on the check: `children's illustration`
   (wrong twice: an oil painting and an AI image), `fresco` (wrong once; rare), `scanned document` (describes how
   it was made, not what it is; overlaps `printed page of text`), `video game menu screen` and
   `video game inventory screen` (overlap `video game screenshot`; `video game title screen` stays).
3. **Reworded:** `form` becomes `paper form`, so CLIP reads it as a document rather than a shape.

Everything else is kept as is, `something else` included. v2 has 51 printed labels, 12 silent ones and the
catch-all (64). The run script defines it as `LABELS_V2` and `SILENT_V2`, and asserts both against this file's
lists before anything runs.

## The images: round 19

120 kept images from Wikimedia Commons, sourced by the worker (`ROUND19_PLAN.md`), built and duplicate-checked by
the reviewer like round 18, with no labels of any kind from the worker beyond its kind:

| Kind | Kept |
| --- | --- |
| everyday photos | 36 |
| artwork and paintings | 12 |
| illustrations, cartoons and comics | 12 |
| signage, logos and packaging | 12 |
| app and web screenshots | 10 |
| game UI and HUD | 10 |
| documents | 10 |
| charts, diagrams and infographics | 10 |
| AI-generated images | 8 |

Photos are 30%, three times their share of the tune split, because that's where the false labels are. The
first-30-kept-per-kind rule from round 18 applies with these counts.

## The review page

One deck of cards, built before you see anything:
- For each image, each list's label if that list shows one. If both lists give the same label, it's one card that
  counts for both.
- All cards shuffled together (seed 20261019), so you don't see an image's two cards side by side and can't tell
  which list a card came from. The card shows only the image and the label.
- Keys: right / partly / wrong, as on the C4-5 page (`caption_c4_5_type.py review`'s pattern). Verdicts saved after
  each key. Expect about 150 to 200 cards: one sitting.

Images where a list shows no line aren't graded for that list; how often each list shows a line (coverage) is
reported.

## Rules (fixed)

**A list passes** if, over the cards it showed:
1. score (right 1, partly 0.5) is at least **85%**;
2. wrong is at most **10%**; and
3. on everyday photos alone, wrong is at most **10%** of the cards it showed on photos (if it showed none, this
   holds).

**Pick:**
- If both pass: the higher score. Within 1.0 point, the one with fewer wrong; still tied, v2 (less on photos).
- If one passes: that one.
- If neither passes: C4-5 ends, and the image-type line is not shipped.

**Report too:** a paired bootstrap of the v2 minus v1 score over images where both showed a label (5,000 samples,
seed 1); coverage per list and per kind; scores per kind; the labels that came up most; and how often v2's silent
labels won on photos and on non-photos.

**After a pick:** CHAIR on the 300 COCO images with the picked list (`caption_c4_5_type.py chair`'s method; at most
+1.0 on CHAIR-i and CHAIR-s in both slots), then C4-5 step 4 (round 11 holdout) as `c4_5_image_type_handoff.md`
writes it, with the picked list at 0.3. Step 4 waits for the user and a quiet machine.

## Records

- The run script extends `caption_c4_5_type.py` (new steps `labels19`, `review19`, `tally19`) or a new
  `caption_c4_5b.py`; either is committed, with `LABELS_V2` and `SILENT_V2`, before labels are computed.
- Results: `benchmarks/results/caption_c4_5b_*.json(l)`. A "C4-5b" paragraph in the plan's C4-5 section, and a
  pointer in the memory file `project_caption_c4_status.md`.
- Nothing ships and nothing is pushed without the user.
