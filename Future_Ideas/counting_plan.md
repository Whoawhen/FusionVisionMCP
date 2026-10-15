# Counting plan: gates C1 and up

`count_objects` is the second-largest 🔻 on the scoreboard (61.8% exact on 427 held-out photo counts, against an
estimated 80–90% for frontier models on counts of 3 or fewer). The problem was put to an outside model
(`counting_problem_statement.md`, 2026-10-14, answered on its low setting). This file records which of its
options are worth testing, in what order, and how. Each gate's rules are committed in its VisionBench script
before it runs, as usual.

## The outside model's options, triaged

| # | Option | Verdict | Why |
| --- | --- | --- | --- |
| 7 | Learned box-validity scorer: per box, raw score, score relative to the image's best, rank, density, containment (IoA) and IoU with higher-scoring boxes, area ratio; optional DINOv2 similarity to the best box | **Gate C1** | Replays on sprint R's saved boxes (`results/replacement_gate1_raw_grounding_dino_{tune,confirm_tf5}.jsonl`, every box down to score 0.0002): no model calls, minutes of CPU. Its central idea, judge a weaker box by whether it sits *inside* a confident one (a spot) or *beside* it (another object), is geometric, not the sub-object detail every rejected method used |
| 1 | Keep `count` at 0.425 and add `count_range` (+ `candidate_boxes`) from boxes in [0.18, 0.425) that survive a containment veto | **Gate C1, alongside 7** | Same saved boxes. Cannot break a control, because `count` is unchanged; it changes what the tool reports, which the problem statement asked for |
| 2 | SAM2 "new territory" rule: a lower-scoring box counts only if most of its SAM2 mask lies outside the confident masks | Second tier | Needs SAM2 runs (~1 h CPU). Worth it if C1 shows the misses are adjacent instances the box geometry can't separate |
| 6 | DINOv2 prototypes from the confident boxes to filter candidates in crowded scenes | Second tier | Targets the other regime (crowds, which overcounted by ~12 at 0.15). Needs DINOv2 on the images |
| 5 | Mask out the counted objects, then ask Moondream2 "is there another whole X?" | Later | Slower; only covers counts of 1–3 |
| 4 | Depth discontinuity (MiDaS, Intel, MIT; not Depth Anything, which is Chinese-developed) to veto painted spots | Later | The only genuinely new idea against the wall, but the 8 controls are flat synthetic drawings where depth means nothing, so it needs its own controls. Photos only |
| 3 | Replace Grounding DINO with OWLv2 | **Dead** | Already tested in sprint R gate 1: no threshold held more than 3 of the 8 controls; 43.0% exact at best |

**One claim to check, not assume.** The outside model reads the 85/50/24% breakdown (1, 2, 3 objects) as the
2nd and 3rd objects scoring between 0.15 and 0.425 and being dropped. That breakdown was measured *at* 0.15, when
those boxes were still counted, so the misses then were more likely merges or overcounts. C1's first step
measures which.

## Gate C1 (run 2026-10-14: option 7 failed, option 1 to the user)

- **Diagnose (tune half only):** for each 2- and 3-object count missed at 0.425, where the labelled instances'
  best boxes sit (above 0.425, in [0.15, 0.425), or not boxed at all) and whether the extra boxes on overcounts
  are contained in a confident box.
- **Candidates, fitted on tune:** option 7 with geometric and score features only (no DINOv2, so no new model
  calls), logistic regression, threshold chosen on tune; option 1's `count_range` and containment veto.
- **Controls:** all 8 synthetic negative controls must stay at exactly 1 (their raw boxes are re-run if they are
  not in the saved cache; about a minute).
- **Confirm:** round 14 (below), scored once, against the shipped 0.425 count, with a paired bootstrap 95%
  interval (10,000 resamples). Option 7 ships only if the interval is above 0 and every control holds. Option 1
  is reported by how often the true count falls inside `count_range` and how wide the range is on 1-object
  scenes.
- **Model choice:** Opus 5.5 to set the rules, Sonnet 5.5 to run.

**Result (VisionBench `count_gate_c1.py`, rules 04ec7b9, result 0f7636f).**
- *The claim, checked:* the missed 2s and 3s are undercounts. For 15 of 17 missed 3s and 8 of 16 missed 2s, enough
  distinct boxes sit in [0.15, 0.425). But the shipped count also overcounts 1-object scenes (204 of 325 exact on
  tune), so a lower floor trades one error for the other.
- *Option 7 fails.* Tau 0.30, controls 16/16, tune 60.0 vs 55.6%, all of it on 1-object scenes. Round 14: **51.5 vs
  57.3% exact (103 counts), CI −11.7 to 0.0**, losing on 2s (19 vs 21 of 35) and 3s (7 vs 11 of 21). Box geometry and
  score alone don't tell a second object from a part.
- *Option 1, for the user to decide:* a `count_range` whose upper end adds the distinct boxes in [0.18, 0.425)
  holds the truth 78.6–79.4% of the time on three sets, against 55.6–61.8% for `count` exact; mean width 1.1–1.9,
  and 47–68% of 1-object scenes get no range. Every control keeps width 0. It changes what the tool reports, not
  `count`.
- *Next if counting continues:* option 2 (SAM2 "new territory") targets exactly the 2/3 undercounts found here, or
  PaliGemma (gate P1, below).

## Round 14: the confirm set

160 (image, class) pairs from Open Images V5 test (CC BY 2.0), 104 classes, no image used anywhere else in
VisionBench; by Open Images' own box count 1: 30, 2: 40, 3: 40, 4–6: 30, 7–15: 20 (VisionBench
`build_round14_count.py`, 7d08677). The Gemini worker counts every image (`worker_handoff/ROUND14_PLAN.md`),
since Open Images boxes are often incomplete, and the user blind-checks a sample. No tool runs on these images
until the counts are committed.

**Worker counts and blind check (2026-10-14):** 134 kept (104 exact, 30 approximate), 26 excluded (mostly Open
Images naming the wrong class). The user counted 30 blind: 21 of 27 agree; the worker's `exact` counts 19 of 22 (21 of 22 after the user corrected two of their own counts: Pillow 3, Van 4), its
`approximate` ones 2 of 5, and the user could not count 3 (all approximate). So C1's confirm should be scored on the
worker's `exact` counts (with the user's counts replacing the worker's where given), and the approximate ones
reported separately, not mixed in. That goes into C1's committed rules.

## Replacement candidates (worker survey, 2026-10-14)

VisionBench `worker_handoff/RESEARCH_COUNT_DETECTORS_REPORT.md`: non-Chinese, permissive, ≤3B models that take a
text name. Licences spot-checked (GLIP MIT, Detic and MDETR Apache-2.0, FIBER MIT; FIBER's cited LICENSE URL is
wrong but GitHub reports MIT).

- **Worth a test: PaliGemma (Google, 3B).** The only candidate that counts natively ("count X", TallyQA simple
  81.7%) as well as boxing ("detect X"), and it runs in `transformers`. The newer PaliGemma 2 mix checkpoints
  (`google/paligemma2-3b-mix-448`) are the version to test. A VLM counts by knowing what the object is, which is
  the wall the detector methods hit. Costs: ~3B parameters on CPU (seconds per call, several GB), gated on
  Hugging Face, and the **Gemma Terms of Use** are a custom licence with a use policy, not Apache/MIT: the user's
  call.
- **Not worth testing:** GLIP, FIBER, Detic, RegionCLIP, ViLD, F-VLM, RO-ViT and MDETR. All 2021–2023, mostly
  weaker zero-shot than Grounding DINO, and nearly all need Detectron2, maskrcnn-benchmark or TensorFlow builds,
  which are a poor fit for a Windows CPU install. MDETR grounds a phrase to one region; it does not count.
- **Too large:** Molmo-7B (AI2, Apache-2.0), which counts by pointing.

## Status

- 2026-10-14: options triaged; round 14 built and handed to the worker; gate C1 waits for the go-ahead.
- 2026-10-14 (night): gate C1 run: option 7 failed on round 14; option 1's `count_range` waits on the user. Gate P1
  (PaliGemma 2 mix, VisionBench `count_gate_paligemma.py`, rules 1a67565) running overnight.
