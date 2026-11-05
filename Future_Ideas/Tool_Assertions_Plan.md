# Tool assertions plan: say what is measured, and give the caller ways to raise confidence

**Status:** drafted 2026-11-05 by Opus 5.5, for the user's review. Nothing here is implemented yet.

## Why

FusionVisionMCP is a set of vision tools for a calling model that makes the decisions; it is not meant to replace
that model. Its small models (3B at most) are at or above frontier level for some jobs and well below it for others
(`frontier_capability_gaps.md`). A blanket "could be wrong" disclaimer makes a tool close to worthless, because the
caller then has to check everything. What makes a tool worth calling is **knowing, by case, when its output can be
acted on**, plus a route to more confidence when it can't: another tool, a check option, or the caller's own
(non-local) vision.

**The rule this plan applies to every tool.** Every output is either:
- a **measurement or reading** with a measured reliability (text, boxes, masks, distances, scores, probabilities), or
- a **small model's opinion**, labelled as evidence with its measured reliability for that kind of case.

No output presents itself as the judgement. Where reliability is low and nothing in the server does better, the tool
says so and points to the caller's own vision; where a claim is not supported at all, the tool stops making it.
`ocr` is the model to follow: its 0.75 floor drops a line rather than garble it, so what it asserts can be trusted.

## 1. What each tool can assert today (measured)

Figures are from the gate records (`docs/history.md`) and the scoreboard; "fresh" means a set the shipped tool was
not tuned on.

| Tool | Can assert (figure) | Must not assert | Caller's next step when it matters |
|---|---|---|---|
| `ocr` | Clean screens/documents: 97.3% of lines verbatim. Real images: 76% of lines (label ceiling ~80%). Empty on text-free images (87% on phase 11). Kept spans are >= 0.75 confidence | That a missing line isn't there (small or blurred text is dropped); handwriting (none); blurred text (62% of key phrases vs 84-96% frontier); TextVQA photos 68.5% | `caption(region=...)` on a small area; native vision for blur or handwriting |
| `caption` | `Visible text` line = `ocr`'s reading (reliable); prose names 73.2% of key facts; names an absent object 12.4% (COCO) | That the prose quotes text; `Image type` is a fact (it matches the labelled kind about 7 in 10) | `ocr` for text; `query_image` or `detect_objects` for one detail |
| `query_image` | Fresh round 31, by kind: yes/no 94.9%, colour 86.9%, absent object 84.6%, "no text" 82.3%, position 84.2% (n 19), count 78.9% (n 38), reading text 77.4%, **object name 58.3%**. **"There is none" answers right 192 of 201 (95.5%).** A short text answer whose words all appear in `ocr`'s reading: right 78-86% on all five sets; when none appear: 41-43% | An object's name on its own word (58%); text it read without OCR support; that a thing is present because it answered about it (it answers a false premise 16.5% of the time on round 31) | `ocr` for text; `classify_image` when the options can be listed; `detect_objects` to corroborate an object; `check_consistency`; native vision for object names that matter |
| `classify_image` | Probability >= 0.5: right 197 of 198 (gate Z2, famous images; everyday photos 79.9% overall) | That any label fits: it always picks one | Add a catch-all label ("something else") |
| `detect_objects` | First box lands on a real instance 71-73%; absent object boxed 9% (generalist) | That a box proves presence; later boxes are instances; recall on app/web (40%) and game UI (56%) | `query_image` presence; native vision for UI elements |
| `count_objects` | Fresh round 26 photo counts, exact by true count: 1: 97.7%, 2: 64.5%, 3: 43.5%, 4-5: 38.1%. `crowd_range` holds the truth 86% on crowds of 4-30. Flat images 68.4% exact | **Presence**: on verified-absent objects it returns non-zero 69% of the time, so a count of 1 says nothing about whether the thing is there. Any exact count above 2 | `query_image` presence ("absent object" 84.6%) first; report `crowd_range` for crowds; native vision for exact counts above 2 |
| `spatial_relations` | Contact 92%, containment 84%; held-object pairs agree with the user 83% | Which instance it measured when there are several of a class (it takes the best box per name) | Distinguishing names per object |
| `dense_region_caption` | 74% box recall | That the labels are right (they misled LFM, -5.6 on object questions) | `detect_objects` for a named object |
| `score_aesthetics` | Ranking edits or shots of one subject (rho 0.49 against people's ratings) | An absolute verdict; anything about non-photographic media without `style_context` | The caller's own judgement |
| `measure_framing` | Subject box matches a labeller 84.4%; same fifth across 88-90%, down 80-87%; size class 76-82% | Whether framing is good; background blur; the prose note (about -7 vs frontier) | The caller writes the critique from the numbers |
| `query_image(structured_analysis)` defect probe | Round 10 (unseen generators): catches 44.8% of defective images, leaves 84.1% of clean ones alone | **"No defects"**: it misses more than half | Treat a flag as "look here"; native vision or `geminivision.detect_generation_errors` for a verdict |

## 2. Changes

### S. Scorer fix first (VisionBench only; no tool change)

The object figure is understated by the scorer, and every reliability figure section B puts into an output would
inherit the error. Of round 31's 55 object misses, about 22 are right or arguably right: "mini-map" against
"minimap" (twice), "ship's hull profile drawing" against "ship hull", "amphitheater" against "anfiteatro",
"merry-go-round" against "carousel", "life preserver" against "lifebuoy", "martini glass" against "cocktail glass".
On the four older sets, 17 of 65 object misses were synonyms. A fair object figure is probably near 70-75% on round
31, not 58.3%.

The fix must stay non-circular: nothing in it may be read off the tool's answers.
1. **Mechanical normalisation** in a new, versioned grader (VisionBench `benchmarks/vqa_grade_v2.py`, tests in
   `tests/test_grade_v2.py`; `run_visionbench.grade_answer` stays frozen so past gates still reproduce): case,
   hyphens and spaces inside a word ("mini-map" = "minimap"), singular/plural, leading articles. Pure string rules,
   fixed and unit-tested before any re-score.
2. **Accepted alternatives written blind to the answers.** For every object question on the five sets (433), a
   worker (Gemini 3.8 Flash; labelling, not coding) writes the other names a careful person would accept for the gold
   answer, seeing the image, question and gold answer but no tool or frontier answer. The user spot-checks a sample
   on the usual card page, as for round 31. Stored beside the questions on the labels branch, as `accept` lists.
3. **Re-score everything the same way**: every arm of Gates V5-V9 that quotes an object figure, the shipped tool by
   kind on all five sets, and the frontier 240 (`frontier_vqa_score.py`), so the gap stays like with like. No gate
   decision is re-opened: each gate's verdict was made under its own scorer and stands as recorded; the re-score
   changes the quoted figures, not what shipped.
4. Report old vs new figures side by side for every kind (normalisation changes text answers too), and update the
   scoreboard row and the figures section 1 quotes.

Rules fixed before the alternatives exist: an alternative is accepted only if the worker wrote it and the user's
spot-check passed (the same 90% floor as round 31's premise cards); the normalisation is the same function for every
arm and the frontier.

### A. Descriptions and manifest (no output changes, no gate)

1. **`manifest.json` and `server.json` are out of date and overclaim.** The server description lists Florence-2,
   Moondream2, EasyOCR and Grounding DINO but not LFM2.5-VL or SAM 3; the `ocr` entry says "backed by EasyOCR" and
   "handles printed, stylized, cursive and low-contrast text alike". There is no handwriting support. Rewrite both
   from the table above, with the framing in "Why".
2. **Each tool docstring and manifest entry opens with what it can and cannot assert**, from the table, before any
   history. History and gate names move below, or out to `docs/history.md`. Docstrings are what a calling agent
   reads at selection time, so keep each opening to 2-3 lines.
3. **README:** a short "what this server is and isn't" section near the top, in three tiers (assert / evidence /
   don't rely on it), linking the scoreboard.
4. **The user's global `CLAUDE.md` routing section** and the Cline rule are updated to match.

### B. Reliability travels with the output (metadata only; answers unchanged)

The description is read once; the output is read every time. Each opinion-type output gains a small block the
caller can act on without re-reading the description:

- `query_image`: `kind` (yes/no, colour, count, position, text, object, other: a deterministic classifier over the
  question, reusing `VQA_TEXT_QUESTION`, `question.py`'s count nouns and the colour/position words) and
  `kind_reliability` (round 31's figure for that kind). **The classifier must be checked against the labelled types
  on all five sets before the figures are quoted** (a figure attached to the wrong kind would be worse than none).
- `query_image` on routed text questions: `ocr_match: "all" | "some" | "none"` (the answer's words against the OCR
  reading the prompt used) with its figure (all 78-86%, none 41-43%). This is the strongest new signal found: it
  held on all five sets, it costs nothing, and it tells the caller exactly when to call `ocr` itself or look.
- `query_image` "none" answers: note that a "there is none" was right 95.5% on fresh questions.
- `count_objects`: `reliability` from the reported count's size (1 / 2 / 3 / 4+), and a fixed note that a non-zero
  count does not establish presence. **Needs one offline step first:** the round 26 figures are by *true* count; the
  caller sees the *reported* count, so recompute exactness by reported count from the cached rows.
- `detect_objects`: `trust: "top"` on the first box and `"candidate"` on the rest (the description already says
  this; the output does not).
- `caption`: the CLIP probability behind `Image type`, so a 0.3 guess and a 0.9 one look different.
- Defect probe: the probe probability and the two operating figures, and the wording "flagged for a look" /
  "not flagged (misses about half)" instead of a verdict.

These change no answer, so they need no VisionBench gate, but each figure quoted in an output must come from a
named, reproducible computation (a script in VisionBench) so it can be refreshed when a tool changes.

### C. Strengthening the weak tools where they can still give insight

Each item is a measurement, not a new answer path, and each gets its rule fixed before it is measured.

1. **`query_image` object names: corroboration, not correction.** Run SAM 3 on the noun the answer names. If
   "found" separates right from wrong answers (as `ocr_match` does for text), ship it as
   `corroborated_by_detector`. Offline on the cached object answers of the five sets (~600 SAM 3 calls, about an
   hour on CPU). Rule: ship only if corroborated answers are right at least 15 points more often than uncorroborated
   ones on every set with 50+ object questions.
2. **`count_objects` agreement signal.** `lfm_check` already asks LFM to count; compare its reply with the detector's
   count. If agreement separates exact counts from wrong ones, report `agreement` (cached replies exist for round 26,
   so this is offline). Same 15-point rule.
3. **`ocr` degradation flag.** A pixel measure of blur (variance of the Laplacian, or the sharpened pass's gain) to
   report `degraded: true` with the measured recall drop, so the caller knows when to lean on its own vision. Check
   on the blur rounds (25/25b) against sharp controls; rule: the flag must catch most real-blur images while firing on
   few sharp ones (thresholds fixed before the run).
4. **`count_objects` presence.** Stop implying presence: when the caller needs it, the description routes to
   `query_image`'s presence answer (absent object 84.6%) or `detect_objects`; no new code.

### D. Retirements (the user's decision)

- `score_aesthetics(artistic_judgment)`, the Ollama reasoner: interpretation is the calling model's job, and it does
  it better. **Retire: approved by the user 2026-11-05.**
- ~~`measure_framing`'s prose note~~: **kept** (user, 2026-11-05). It stays below `low_score_threshold` and is
  described as a weak opinion (about -7 against frontier critiques).

### E. Tool renames (after A and B)

The server keeps its name. **Tool renames and description changes are cleared by the user (2026-11-05)**: the user
is the server's only user, so a rename needs no transition alias. Decide after the new descriptions exist.
`query_image` is the only candidate: if, with per-kind reliability in its output, it still reads like "ask me
anything", rename it (`ask_local_vlm` or similar) and update, in the same change, the global `CLAUDE.md` routing
section, the Cline rule, client auto-approve lists, and every VisionBench script and test that calls it by name.

## 3. Order

Stop for the user's go-ahead between steps.

1. **User reviews this plan** (especially section 1's table and D).
2. **S, scorer fix** (Opus 5.5 writes the normalisation and its tests, and the worker brief; Gemini writes the
   alternatives; the user spot-checks; Opus re-scores). Figures in section 1 are refreshed from it before A quotes
   them.
3. **A, descriptions and manifest** (Opus 5.5, this session or a new one: the wording is the work). User reads the
   drafts before they ship.
4. **B, offline figures first** (Sonnet 5.5, new session + handoff): the `query_image` kind classifier checked
   against labels on five sets; count exactness by reported count; then the output fields and tests.
5. **C1-C3 measurements** (Opus 5.5 writes the rules and scripts; long runs queued for when the user is away), then
   implementation of whichever pass (Sonnet 5.5).
6. **D** (retire `artistic_judgment`, approved) goes with A. **E** (tool renames, cleared) is decided after B,
   on how the new descriptions and outputs read.
