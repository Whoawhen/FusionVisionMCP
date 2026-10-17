# Caption sprint C4 plan

Written 2026-10-10 (evening). A plan only: nothing in it has been run. Each sprint waits for a go-ahead.
Steps are labelled with the model to run them (Opus 5.5 to design or judge, Sonnet 5.5 to execute a written
procedure). Follows [caption_sprint_plan.md](caption_sprint_plan.md) (C1, C2) and the C3 work recorded in
VisionBench commits `cd382b6`, `a668780` and `644fa7a`.

**Reviewed 2026-10-11 (Opus 5.5).** The code claims were checked against the pinned Moondream2 revision
(`2025-06-21` = `9a7d402`) and our wrappers, and they hold: `DEFAULT_TEMPERATURE = 0.5`, `DEFAULT_TOP_P = 0.3`,
`temperature == 0` takes the argmax, `encode_image` returns an `EncodedImage` with the prefill KV cache, `caption()`
and `query()` both accept one, `max_crops = 12`, and `Florence2._run` uses 3 beams and 1,024 new tokens for every
task. Changes from the review: greedy and beam guards now check for repetition; the C4-2b go/no-go count moved
into C4-0; C4-3 drops two candidates that cannot be run and adds Gemma 3 4B through llama.cpp (the untried
"stronger model" route); C4-4 corrects the "no inference time" claim, adds a terms check on Gemini outputs used as
training data, and sets guards for local GPU training; the user confirmed the Decisions section (2026-10-11).

## Where `caption` stands (280 held-out images, confirm split)

| | Default: Florence-2-large + `read_text` | `deep`: Moondream2 long + `read_text` | Label ceiling |
|---|---|---|---|
| Keyword recall | 60.9% (was 53.3% before `read_text`) | 66.3% (was 61.5%) | ~75% |
| Median words | 157 | 244.5 | |
| Median seconds (CPU) | 13.1 | 35.8 | |
| Speed limit (user, 2026-10-09) | ≤ 15 s (**1.9 s left**) | ≤ 45 s (**~9 s left**; tune median was 41.3 s, so ~4 s there) | |

Weakest kinds after `read_text` (default / deep): AI images 48.5 / 52.5, illustrations 50.8 / 56.9,
game UI 54.4 / 62.2, charts 56.9 / 67.9, app/web 63.4 / 60.6. Photos (89–91%) and artwork (76–81%) are near
or above the ceiling.

The scoreboard's last note (2026-10-10): *code-only caption work has plateaued; what is left (concepts,
specialist knowledge, image types without false labels) needs a stronger model than CPU allows today.* This
plan accepts that conclusion. It has two small code-only items that use headroom C3 didn't have (Sprints C4-1
and C4-2) and then the step the "good" section names for when progress stalls: fine-tuning (C4-4).

## Already tried: do not repeat

| Idea | Result | Where |
|---|---|---|
| OCR text appended to the caption | **Shipped** as `read_text` (+7.7 default, +4.7 deep, 0 images worse) | `caption_text_grounding.py`, FusionVision `3283766` |
| Zero-shot CLIP `Image type:` label | Not shipped: +3.2 / +3.5, but labels agreed with the image's kind only 83.0% of the time (floor 90%). The wrong labels are false claims (photos called "table of data") | `caption_type_label.py` |
| Moondream2 "say what kind of image it is first" prompt (with or without OCR) | Lost on tune | `caption_text_grounding.py` |
| Language tag on the OCR line | No gain (+0.1) | same |
| Moondream2 theme question (`Theme:`) | +1.3 on tune (CI 0.4–2.5), but costs ~22 s; deep went to 63.6 s | `caption_theme.py` |
| Crop to a region before captioning | No gain (33% → 28%; small objects 21% → 9%): the crop loses the context | `region_crop_pilot.py` |
| `Florence-2-large-ft` | **36.9%** vs 55.9% on tune: 38-word captions | `caption_ab.py` `florence_ft` |
| Florence-2 fine-tunes: PromptGen v2, moredetailed, Flux, Chan | PromptGen 62.6% is mostly length and is Chinese-origin; moredetailed 61.1% at 21.8 s (over 15 s); Flux 53.0%; Chan 47.7% | `caption_ab.py` |
| Florence-2 caption + Moondream2 normal caption joined | 67.1% at 39.7 s, no better than Moondream2 long (67.8%, 39.3 s) | `caption_ab.py` |
| Other captioners (SmolVLM2, PaliGemma 2, Gemma 3 4B) | Nothing beats either slot within its speed limit | C2 gates 1–2 |

Still untried from earlier plans: Gemma 3 through a 4-bit GGUF runtime (now C4-3b); Moondream 3.1 on a GPU
(needs CUDA inference, which the GPU-crash rule limits; not planned).

## Ideas from the outside Gemini chat (2026-10-10), checked against the record

| Idea | Verdict |
|---|---|
| "61.5% → ~81.5%" | 61.5% matches Florence-2 + text on the tune split. No 81.5% figure appears in the results files I read, and it is above the ~75% label ceiling, so the labels can't reward it. |
| Tiling / sliding window for Florence-2 | Moondream2 already tiles: the pinned `2025-06-21` code cuts up to 12 crops of 378 px plus a global view (`config.py` `max_crops = 12`). For Florence-2 prose, crops lost context (row above), and 4–5 calls cannot fit the 1.9 s left. **Not planned for `caption`.** |
| IoU thresholds, NMS, box recall | Do not apply. The metric is keyword-group recall in prose (`run_visionbench.keyword_group_hit`), not boxes. |
| Decoding (`num_beams=3`, longer `max_new_tokens`) | Florence-2 already runs 3 beams with 1,024 new tokens. Captions stop at ~130 words because the task head stops, not because of the token limit. C4-2a tests fewer beams **for speed**. |
| Use `Florence-2-large-ft` | Measured: much worse for captions (row above). |
| LoRA on UI and document data | C4-4. Few public adapters exist and they're a poor fit (C4-3). |
| Chain OCR into the caption | Shipped (`read_text`). |

## Rules carried over

- Prompts, settings and thresholds are chosen on the tune split only. A pass/fail guard is written down before the run.
- Every recall figure is reported with its word count and a length-matched value, plus `absent_object_named`.
- Anything that adds words to a caption also gets the COCO CHAIR run (`caption_hallucination.py`) before it ships.
- Speed limits are hard: 15 s default, 45 s `deep`, median on the same CPU and images.
- No Chinese-developed models or base models. Each new checkpoint or adapter gets a paper check first (gate 0).
- **New:** the 280-image confirm split has now decided C1, C2, C3 and option C. Each extra use makes it less of a
  holdout. A fresh labelled holdout (C4-0c) does the final confirm for anything from C4 that ships.
- **New:** a shipped change gets permanent tests in `tests/` (user rule), not just a VisionBench run.

## Sprint C4-0: measurement hygiene (Sonnet 5.5; ~2 CPU hours)

**a. Moondream2 sampling.** The pinned code defaults to `temperature=0.5, top_p=0.3`
(`moondream.py` `DEFAULT_TEMPERATURE`), and neither `Moondream.caption` nor `caption_ab.py` passes
`settings`. So every `deep` number so far comes from a sampled caption. A rerun of one image can differ, and
part of the deep-slot CIs is sampling noise.
- Run `moondream_long+text` on tune with `settings={"temperature": 0}`, twice. Check the two runs are identical.
- Guard (fixed now): switch to greedy if recall is within the tune CI of the sampled run and `absent_object_named`
  doesn't rise. Being repeatable is reason enough if the result is a tie.
- **Added guard: repetition.** Greedy decoding on a long caption can loop ("The image also shows ... The image also
  shows ..."), and with `max_tokens = 768` a loop runs long and slow. Report median words, median seconds and the
  number of captions with a repeated 8-word span, for both runs. Greedy fails if loops appear on more than 2 of 120
  images or median seconds rise by more than 2 s, whatever the recall.
- This also has to be settled before C4-1, because the encode-once equivalence check only works on greedy output.
- Whichever setting wins becomes the `deep` baseline for every later C4 comparison. Don't compare a greedy variant
  against the old sampled numbers.
- `query_image` uses the same default sampling. Changing it is out of scope here and would need its own phase 11 rerun.

**b. Refresh the miss list.** The "122 groups missed by both" breakdown predates `read_text`. Recount on the tune
split which keyword groups `florence_detailed+text` and `moondream_long+text` both miss. Sort them into: text in
the image that OCR missed, image type, concept/theme, specialist name (game, app, artist), small object, other.
This decides whether C4-2c and C4-4 target the right misses. Output: `caption_miss_taxonomy.py` + JSON in VisionBench.
- Sonnet 5.5 builds the recount; the sorting into kinds is a judgement call, so Opus 5.5 does it or reviews it.

**b2. C4-2b's go/no-go count, moved here.** Count tune images whose OCR text runs past 40 words. It needs no model
run (the OCR text is already cached from C3), so it costs nothing and settles C4-2b before C4-2 is planned in detail.

**Results (2026-10-12, Opus 5.5; VisionBench `caption_c4_0.py`, `0c7ae2c`).**
- **a: greedy fails its guard as written, so sampling stays the `deep` baseline.** Greedy was repeatable (120/120
  identical) and tied on recall: 68.9% vs 70.7% sampled (−1.8, CI −4.3 to 0.7; 10 images better, 18 worse), with
  absent objects named 0 vs 1. The loop guard (a repeated 8-word span on at most 2 of 120) failed: 19 captions. But
  the sampled baseline scores 26 on the same check, because the check also catches ordinary reused phrases. Real
  runaway loops (the caption's last quarter mostly repeats, e.g. hundreds of "Í" or "effects, effects, …") came to
  greedy 6, sampled 8, plus 3 sampled captions at the 768-token cap. **So the shipped slot already loops on ~7% of
  images.** Speed tied in the same session, alternated on 20 tune images: sampled 32.1 s, greedy 31.9 s. The
  older 39.3 s sampled median came from a busier machine, so compare seconds only within one session. C4-1's
  equivalence check can still run greedy internally; it only needs determinism for the test, not for shipping.
- **b: 111 groups missed by both slots** (of 447). Sorted by Opus 5.5 (`caption_c4_0_misses_tune.json`):

  | Kind | Groups |
  |---|---|
  | Object or scene detail (4 are game HUD parts) | 34 |
  | Image type or genre (title screen, invoice/receipt/certificate, engraving/lithograph, comic, packaging, DAW) | 27 |
  | Concept or story | 13 |
  | Fine-grained vocabulary (kufi, quail, De Morgan's law, ostracon) | 11 |
  | Specialist name (Kdenlive, BZFlag, Nadar) | 10 |
  | Language | 7 |
  | Text OCR misread (`Meta Al`, `HACIBEKIR`) | 5 |
  | Scoring artefact ("bar graph" vs "bar chart", unfolded accents) | 4 |

  Image type is the largest kind a model could fix (the CLIP type label was ruled out for its false labels). It's
  what C4-4's training data should aim at, along with UI and HUD vocabulary. The accent and chart/graph artefacts
  are a scorer issue. Leave the 400-image numbers alone; fold them in only if the scorer is changed for every
  variant at once.
- **b2: 38 of 120** tune images have more than 40 OCR words (18 have more than 80), so C4-2b goes ahead.

**New item C4-0d (proposed, needs a go-ahead): stop the `deep` caption when it loops.** Cutting each caption where
its loop starts costs 0.0 points (sampled) and 0.2 (greedy) on cached captions, and the loops are the slow tail
(97 s and 170 s in the timing run). It ships by streaming the caption and stopping when the last 8-word span
repeats one seen earlier and the tail is mostly repeats. Needs: unit tests on the loop detector, and a timing check
on the 11 looping tune images. No recall run is needed beyond the cached replay.

**C4-0d shipped (2026-10-12, FusionVisionMCP `6538257`).** The online rule (16 consecutive repeated 8-word spans) cut 14
of 120 cached tune captions for −0.2 points. Timed on 18 tune images that had looped, alternated in one session:
median 57.3 → 40.7 s, mean 63.8 → 47.2 s, 9 loops → 0 (VisionBench `a913d48`). It's now part of the `deep` baseline
for C4-1.

**c. Fresh holdout.** Ask a worker round for 150–200 new images across the nine kinds, labelled with the
same caption-keyword procedure as phase 11 (same labeller and prompt), deduplicated against every existing
VisionBench image by perceptual hash. Not used for anything until a C4 finalist's single confirm run.

**Done (2026-10-11): round 11, 235 images** (25–27 per kind, all kept; 6 near-duplicates within the round
dropped, none matching earlier VisionBench images), in VisionBench-labels `benchmarks/holdout_round11/`. Use all
235 rather than trimming to 20 per kind: nobody has seen a result on it, and trimming would only widen the CIs.
Caveat: its keyword groups are a little looser than round 4's (3.2 groups per image vs 3.65; more groups mixing
different things, e.g. `[carriages, spectator crowd, Impressionism]`, which are easier to hit). So read it only
as a paired comparison (candidate vs baseline on the same 235), never as an absolute level against the 280 split.
Round 11b (36 images labelled again blind) measures how repeatable its labels are.

**Repeatability (2026-10-11).** Against 11b, only 62.1% of round 11's groups were hit by the blind description
plus keywords (round 4 vs 5: 85.4%): its groups were specific multi-word phrases a correct caption wouldn't use
(`[decision diamonds, swimlanes, tickets]`). Round 11c added plain 1–2-word synonyms for each group's main thing
(547 words on 273 of 757 groups after the reviewer dropped 9 loose ones: `will`, `game`, `document`), and it rose to
**80.2%** (description alone 53.4 → 75.0%). That's slightly flattered, because the 36 images in 11b were broadened
too. It's close enough to round 4 to use, still as a paired comparison only. Before C4, the frozen file is
VisionBench-labels `fa06afc`.

## Sprint C4-1: Moondream2 encode-once for `deep` (Opus 5.5 to design, Sonnet 5.5 to run)

**Why:** the theme question gained +1.3 (0 worse) but cost ~22 s, because `model.query(image, …)` encodes the
image from scratch. In the pinned revision `encode_image()` returns an `EncodedImage` (vision features plus the
text model's prefill KV cache), and both `caption()` and `query()` accept one directly (`moondream.py` L230,
L541, L625). If most of the 22 s is encoding, a second question on the same encoding costs only its ~20
generated words.

1. **Measure first, no recall run:** on 20 tune images, time `encode_image` alone, `caption(encoded, "long")`,
   and `query(encoded, THEME_PROMPT)` (greedy). Stop if `query` on a reused encoding costs > 6 s median.
2. **Equivalence check:** with greedy settings, `caption(image)` and `caption(encode_image(image))` must give
   identical text on those 20 images. If they don't, stop: reuse would change what ships. Also check the order the
   shipped code will use: `caption(encoded)` first, then `query(encoded, …)`. The caption writes KV entries past
   `encoded.pos`, and `query` reloads only up to `pos`, relying on the attention mask to hide the rest; so
   `query(encoded)` after a caption must equal `query(image)` on a fresh encode.
3. **Tune, at most two variants, fixed now:**
   - `deep+text+theme`: the existing `caption_theme.py` prompt on the reused encoding.
   - `deep+text+type_q`: `query(encoded, "What kind of image is this? Answer in a few words.")`, appended as
     `Image type: …`. This is option C's idea with Moondream2 instead of CLIP as the labeller, under **the same
     90% agreement floor**, judged the same way as `caption_type_label.py`. This differs from C3's lost variant
     (that one replaced the caption prompt; this one appends a short answer to the full caption), but C3 also
     found Moondream2's type answers short and vague, so the agreement floor is the likelier failure. Check the
     agreement on the 20 timing images first (free, they're already run); skip the variant if it's under 80%.
4. Guards: total `deep` median ≤ 45 s on both splits; recall CI above 0; 0 images worse; agreement ≥ 90% for
   `type_q`; CHAIR not worse.
5. Ship (if it passes on the fresh holdout): `Moondream.caption()` encodes once and runs the extra question on
   the same encoding. Tests: a unit test with a fake model checking `encode_image` is called once per image;
   a slow-marked test for the greedy equivalence on `tests/sample.jpg`.

**Steps 1–2 done (2026-10-16, VisionBench `caption_c4_1.py measure`, `0316fb5`; pin `5d6c926`).** Medians on 20 tune
images, greedy, 18 threads: encode 9.8 s, long caption on the encoding 18.3 s, **theme question on the same encoding
2.0 s** (stop rule: 6 s) against 12.9 s with a fresh encode, type question 0.4 s. Equivalence: caption and theme
answer identical on 20 of 20, with the theme asked after the caption as shipped. `type_q` is out: 5 of the 20 could
be judged and 2 agreed (40%, floor 80%). That leaves `deep+text+theme`. Its tune recall was measured in C3
(`caption_theme_tune.json`): +1.3, CI +0.4 to +2.5, 0 images worse, absent object named 1 vs 1. It failed then only
on time (63.6 s against the 45 s limit), and reuse removes ~11 s. Since reuse gives identical output, confirm reuses
`caption_theme.py confirm` (fresh-encode answers, same distribution) for recall. The real `deep` path's seconds are
timed separately, in one session.

Expected size: +1 to +3 points on `deep`, mostly AI images and charts. A small gain, but it costs no memory and
no new model.

## Sprint C4-2: Florence-2 default slot (Sonnet 5.5; ~3 CPU hours)

The default slot has 1.9 s of headroom, so it needs a speed-up first.

**a. Fewer beams.** `Florence2._run` uses `num_beams=3` for every task. Try `num_beams=1` for the caption task
only (no other task changes). Measure on tune: median seconds, recall, length-matched recall, words,
`absent_object_named`. Guard: recall within the tune CI, CHAIR not worse, and no rise in repeated spans (same
check as C4-0a; beam search is partly what keeps Florence-2's long captions from looping). Whatever time it saves
is the budget for c. Time the 20-image smoke first: if 1 beam saves under 1 s median, stop there.

**b. Put titles first in the OCR line.** `visible_text_line` keeps reading order and cuts at 40 words, so a long
document or busy screen can lose its title. With `ocr_pages(..., detail)` boxes, order lines by text height
(tallest first) and keep reading order among similar heights. The 40-word cap stays, so captions don't get longer.
- Go/no-go: the count from C4-0 b2. If it's under ~10, skip this step.
- Pure ordering on the OCR boxes, so it carries over unchanged when PP-OCRv6 is replaced (no tuning on a
  Chinese-developed model).
- Applies to both slots. Unit tests in `tests/test_analysis_helpers.py` next to the existing `visible_text_line` tests.

**c. Object list from `<DENSE_REGION_CAPTION>`** (only if a frees ≥ 4 s, or only in the `deep` slot). Dense
regions score 74% box recall (phase 7), and small objects are a known miss (21% named in the crop pilot). Add
`Also visible: …` with region labels the caption doesn't already mention, deduplicated, at most ~12 words.
- It's a second Florence-2 call, not the same pass. Time it on its own first.
- Biggest risk is hallucination: Florence-2's grounding boxes an absent object on every image (`detect_backend_ab`).
  So COCO CHAIR and the claims spot-check are required, not optional.
- Run in the `deep` slot it would need both Florence-2 and Moondream2 loaded for one call. They're already loaded
  under idle release, but check `--memory-mode`.

## Sprint C4-3: other checkpoints (Opus 5.5 gate 0, Sonnet 5.5 runs; ~half a day)

### C4-3a: existing adapters, paper check then tune

Low expected value; done because it's cheap and settles the "is there one I can download?" question.

| Candidate | What it is (verify at gate 0) | Why the prior is low |
|---|---|---|
| `NikshepShetty/Florence-2-DOCCI-FT` | PEFT LoRA on **Florence-2-base-ft** (0.23B), DOCCI captions | Built on `-ft` (which collapsed to 38 words for us) and on base, not large. DOCCI is photo-centric, and photos are already ~90%. PaliGemma 2 DOCCI scored 52.1% |
| `NikshepShetty/Florence-2-pixelprose` | PEFT LoRA, PixelProse captions | Same base question |
| `fal/moondream2-docci-instruct` | Moondream2 fine-tune | Which Moondream2 revision is it built on? A different one also moves `query_image` (phase 11 rerun) |

Dropped in review, no run possible or useful: Moondream2 `variant` LoRAs (the pinned code fetches them from
`api.moondream.ai`: private cloud fine-tunes, not a public catalogue, and image data would leave the machine) and
OmniParser's icon captioner (captions detector crops, not whole images).

Gate 0 checks for each: author and organisation origin, base-model origin, licence, dataset licences, which base
checkpoint (base vs large, `-ft` or not). **Run rule (no runs without a decision):** an adapter on
Florence-2-base or `-base-ft` doesn't get a tune run: `-large-ft` already collapsed to 38 words, and a 0.23B base
won't beat the 0.77B large on the same task. Only a candidate on `Florence-2-large` (not `-ft`) or on our pinned
Moondream2 revision gets one tune run with its card prompt. If it's fast, compare with the default slot; if not,
compare with `deep`. If none qualifies, C4-3a ends at the paper check.

**Gate 0 result (2026-10-11, VisionBench-labels `worker_handoff/MODEL_RESEARCH_C4_VERIFIED.md`): C4-3a ends at the paper check.** Both NikshepShetty adapters are LoRAs on
`Florence-2-base-ft`. fal's model is a full fine-tune of a May 2024 Moondream2, older than our pin, so it would
move `query_image` too. No tune runs.

### C4-3b: Gemma 3 4B through llama.cpp (Opus 5.5 gate 0, Sonnet 5.5 runs)

The scoreboard says the rest of caption needs a stronger model than CPU allows. C2 ruled out Gemma 3 4B on speed
in transformers (79 s bf16, 66 s int8), not on quality, and a 4-bit GGUF run through llama.cpp is the one untried
way to make it fit. This is a model swap, not a fine-tune, so it isn't held back by the "good" rule.

**Gate 0 result (2026-10-11, VisionBench-labels `worker_handoff/MODEL_RESEARCH_C4_VERIFIED.md`): passes.** Google's QAT build has the vision projector
(`gemma-3-4b-it-q4_0.gguf` 3.16 GB + `mmproj-model-f16-4B.gguf` 0.85 GB; gated, accept the Gemma Terms).
`llama-cpp-python` 0.3.36 has a Windows CPU wheel on its own index (not PyPI) with `MTMDChatHandler`; llama.cpp's
prebuilt `win-cpu-x64` binaries are the fallback if the handler doesn't drive Gemma 3. Try that first at gate 1.

1. **Gate 0:** Gemma licence (Gemma Terms of Use, already accepted for PaliGemma 2), the GGUF's publisher (prefer
   Google's own `-qat-q4_0-gguf` release; avoid re-quantisations from unknown uploaders), and the runtime
   (`llama-cpp-python` with Gemma 3 vision support, Windows CPU wheels). A new runtime dependency means a
   `pyproject.toml` change and a reinstall; check the torch-free import rule still holds with it lazy-loaded.
2. **Gate 1, 20 tune images, timing only:** median ≤ 45 s for the full `deep` path (caption + `read_text`), at
   most 18 threads (the benchmark affinity rule). Stop if over. Memory resident alongside Moondream2 and
   Florence-2 is reported too.
3. **Gate 2, tune split:** its card prompt for a detailed description, plus `read_text`. Compare with
   `moondream_long+text` (greedy baseline from C4-0a) at matched length. Guards as in the rules above.
4. If it wins on tune, decide with the user whether it replaces Moondream2 in `deep` (Moondream2 stays for
   `query_image` either way) before any confirm run.

Run C4-3b before C4-3a's adapter runs: it targets the gap the scoreboard names, and its gate 1 is cheap.

## Sprint C4-4: train a caption LoRA for Florence-2-large (decision point; Opus 5.5 to design)

This is the "fine-tunes" step that the "good" section reserves for when progress stalls. **It needs the user's go-ahead.**
Per the scoreboard, *the user* decides when to move to fine-tunes. The user's list is
[model_variants_and_loras.md](model_variants_and_loras.md) (confirmed 2026-10-11). Its caption item, switching to
`Florence-2-large-ft`, was measured and lost (above), and it doesn't propose a trained LoRA, so this sprint adds to
the list rather than conflicting with it.

**Target:** the default slot. Florence-2-large is 0.77B (Moondream2 is 1.9B), so it's cheaper to train. A merged
LoRA adds no inference time, but this plan keeps the adapter **unmerged** (below), and an unmerged LoRA does add
work on every forward pass. With 1.9 s of headroom that has to be measured before training anything: load a
randomly initialised adapter of the planned rank on the caption task and time 20 tune images. Aim at the C4-0b miss list: image type, game/app/UI vocabulary, chart and document
structure.

**The risk the record already shows:** `Florence-2-large-ft` was tuned on short captions and its recall
collapsed (38 words, 36.9%). Training data must be long, dense, multi-sentence captions, and median caption
length is a tracked metric during training.

**Data (check every licence and origin at gate 0; none of this is verified yet):**

| Source | Covers | Notes |
|---|---|---|
| Screen2Words (Google Research; on Rico screens) | App screen summaries | Short summaries: use them as part of a caption, not the whole thing |
| Widget Captioning (Google Research; Rico) | UI element names | Element-level |
| DOCCI (Google) | Long, detailed captions | Photo-heavy: keep its share small |
| TextCaps (Meta AI) | Captions that use the text in the image | Overlaps with what `read_text` already does |
| Distilled captions for game UI, AI images and illustrations | Weakest kinds; little public data | User-sourced images captioned by Gemini (the user already approved Gemini as a grader). **These images leave the machine.** Gate 0 must also check the Gemini API terms on using its outputs to train another model (the same open question `image_sources_to_consider.md` records for ShareGPT4V). If the terms forbid it, this row is out |

**Gate 0 result for the data (2026-10-11, VisionBench-labels `worker_handoff/MODEL_RESEARCH_C4_VERIFIED.md`):** DOCCI (CC BY 4.0, ~136-word descriptions) is the clean
first source. TextCaps is usable (CC BY 4.0 / CC BY 2.0) but short. Widget Captioning's annotations are CC BY 4.0,
but its images (and Screen2Words') are Rico's: third-party copyrighted screenshots under research terms, outside
the project's licence list. Screen2Words has no licence. The Gemini API terms say "You may not use the Services to
develop models that compete with the Services". **Two user decisions before C4-4: accept Rico's images or not,
and whether the distilled Gemini set fits that clause.** 2026-10-12: the user accepted Gemini-made captions for
the LoRA set, and accepted Rico's screenshots (Widget Captioning, plus Gemini captions on Rico screens) because the
LoRA stays local: the weights are not to be published. Revisit Rico if that ever changes.

**2026-10-12, revised: FusionVisionMCP itself will be published (GitHub, free local server), so any LoRA it loads is
published too.** That splits C4-4 in two:

- **C4-4p, proof of concept (go-ahead from the user, 2026-10-12).** Rico screenshots captioned by Gemini (long, dense,
  multi-sentence, a different prompt from the eval labels), plus a small DOCCI share so photos don't regress. Purpose:
  find out fast whether a caption LoRA improves Florence-2's captions at all, on the tune and confirm splits with the
  usual CHAIR check. **These weights are private and never ship**: not committed to this repo, not downloaded by the
  server, kept under VisionBench's ignored scratch area. Same data guards (dedup against VisionBench and the holdout).
  Because the eval labels also come from a frontier model, a recall gain without a CHAIR and claim-grading hold counts
  as style, not content.
- **C4-4s, shippable LoRA, only if C4-4p shows a gain.** Retrained on redistributable data only: DOCCI and TextCaps,
  screenshots the user takes or that are openly licensed, with **web UI as the main screenshot source (the user's
  choice, 2026-10-12)**: openly licensed sites rendered with Playwright as VisionBench's rendered-pages step already
  does. These are modern, any size, and licence-clean, unlike Rico. Add some desktop and game UI too. Public web-UI
  datasets (WebUI, Mind2Web) need a licence check first. Captioner candidates (user, 2026-10-12): the latest Gemma
  (Gemma 4 if released), after a terms check; Qwen 3.8 27B was raised too, but it is Chinese-developed, so it needs
  the user's explicit exception to the no-Chinese-models rule first. Caption with a model whose licence permits training on and
  publishing from its output (Gemma 3 is the candidate). Rico and Gemini captions stay out of it. It then faces the
  full ship gates above, including the fresh holdout. Every training row carries a `source` tag so any subset can be
  retrained.

**C4-4p status (2026-10-12):**
- **Rico screens selected:** VisionBench `c82f048`, `benchmarks/c4p_data.py rico`. From 3 Screen2Words train shards
  (1,725 apps), 1,200 screens were picked, one per app, round-robin across 27 Play categories. They're in
  `scratch/c4p/rico/batch_01..12`.
- **Worker handoff:** VisionBench-labels `975c975`, `worker_handoff/C4P_CAPTIONS_PLAN.md`.
  - Captions are 100-180 words, structured as image type → layout → UI parts → content → look.
  - At most three quotes of five words or fewer, so the model doesn't learn to invent text.
  - Blank, loading and unreadable screens are skipped.
  - `c4p_data.py check` verifies every batch.
- **Worker captions done and reviewed (labels `122c3a2`):**
  - `check` reports 0 problems: 1,186 captions plus 14 skips, median 138 words.
  - My native-vision spot check of 10 random captions: 8 accurate, 2 with small count errors (four headers for
    three, four dots for five), none invents anything. A Qwen second opinion wasn't needed.
- **Two defects the checker couldn't see:**
  - "A status bar sits across the top." was word for word in 792 captions, with variants in ~83% of them; stock
    navigation-bar sentences were similar. The handoff's "status bar only in passing" invited this.
  - 9 captions had more than 3 quotes.
- **Fix:** `c4p_data.py clean` writes `scratch/c4p/rico_clean.jsonl`.
  - It drops sentences about the phone's own status and navigation bars (app tab bars stay) and rewrites references
    to them ("Below the status bar, a toolbar ..." → "At the top, a toolbar ...").
  - It leaves out the 9 over-quoted captions. `check` now counts quotes too.
  - Result: 1,177 captions, median 129 words; 59 scattered, varied mentions remain (5%).
  - Lesson for future handoffs: never prompt for a fixed element every image has.
- **2026-10-16, built and checked (VisionBench `benchmarks/c4p_lora.py`, environment `.venv-lora`: torch 2.14.1
  cu130, transformers 5.19 as the server, peft 0.21.2):**
  - DOCCI: 400 training-split images streamed from Google's archive (~200 MB of the 7.6 GB), median 112 words.
  - Dedup (dHash, Hamming <= 4) against all 4,004 images under main VisionBench's `benchmarks/`: 2 DOCCI photos
    dropped (likely chance matches). Correction: the round 11 holdout lives only in the VisionBench-labels worktree,
    so that pass didn't cover it. Checked separately afterwards: 0 of the 1,575 rows match any of its 241 files. 1,575 rows: 1,496 train, 79 validation (5%, seeded).
  - Switch check: with a random LoRA loaded but disabled, `<OD>`, dense regions, OCR regions and phrase grounding on
    8 `tests/` fixture cases, and the caption, are byte-identical to the base model; enabled, the caption changes.
  - LoRA: rank 16, alpha 32, on the text model's attention (encoder self-attention, decoder self- and
    cross-attention q/k/v/out); vision tower untouched. 4.7M trainable weights (0.6%).
  - Overhead (`results/c4p_overhead.json`): the unmerged, untrained adapter adds a median 1.4 s to the default
    caption (8.7 → 10.0 s on 20 tune images, CPU, 18 threads, alternating order; all 20 captions identical). That's
    within the 15 s limit here. If it ships, merging into a caption-only copy of the weights would remove it, at
    the cost of the memory for a second copy.
  - Prep: 1,575 images stored as uint8 tensors that reproduce the processor's output exactly (max difference 0).
    Labels: median 157 tokens, max 367, none truncated.
  - Label check: the base model scores its own captions at loss ~0.16 and the worker's at 5.2–5.7. The labels are
    built correctly; the target style is just far from Florence-2's own.

**C4-4p rule, fixed 2026-10-16 before any training.** Train once (rank 16, 3 epochs, lr 2e-4, effective batch 8).
Score the three epoch checkpoints on tune (120) with `c4p_lora.py caption`/`score`, against the shipped
`florence_detailed+text`. Pick the checkpoint with the highest `lora+text` recall whose absent-object count is not
above the base's. Run that one checkpoint on confirm (280) once. **"LoRA helps" means, on confirm:**
1. `lora+text` minus `base+text` recall has a 95% bootstrap CI above 0;
2. the gain holds at matched length (each caption's prose cut to the shorter of the pair): point estimate above 0;
3. the absent object is named no more than 2 more times than by the base.

**Training (2026-10-16):** on the local GPU with the guards above. A 5-minute smoke run went clean, so the full run
went next: it took 15 minutes, against the 30-minute stage it would have replaced. Peak 3.7 GB VRAM, 84 °C,
~114 W, 0 new WHEA events, exit 0. Validation loss 5.48 → 2.44 / 2.30 / 2.26 by epoch.

**Tune (120, `results/c4p_score_tune.json`), against `florence_detailed+text` at 61.5%:**

| Checkpoint | Recall | CI95 | At matched length | Absent named | Median words |
|---|---|---|---|---|---|
| epoch 1 | 64.7% | -0.4 to +6.8 | 63.8% | 0 | 132 (base 122) |
| epoch 2 | 62.4% | -2.2 to +4.2 | 62.2% | 1 | 130 |
| epoch 3 | 64.0% | -0.9 to +5.7 | 63.1% | 1 | 134 |

The rule picks **epoch 1**. Its gains by kind: everyday photos 67.4 → 78.3, charts 66.0 → 72.0, signage 54.2 → 60.4,
illustrations 67.3 → 73.5, screenshots 67.9 → 71.7. AI-generated images lost (48.1 → 44.2); documents and game UI
were flat. Confirm runs epoch 1 only.

**Confirm (280, `results/c4p_score_confirm.json`): fails.** `lora+text` 61.5% against 60.9% (+0.6, CI -1.3 to
+2.6); at matched length 60.3% against 60.3% (+0.0); absent object named 2 against 1; median words 136 against 129.
Rules 1 and 2 fail and rule 3 holds, so **the LoRA route stops here: C4-4s does not go ahead**, and the web-UI data
idea stays parked. The +3.2 on tune was noise across 120 images; on 280 fresh ones it shrank to +0.6, and to zero
once length is equal. By kind it's a wash: signage 60.7 → 66.1 and illustrations 50.8 → 56.9, against everyday
photos 89.1 → 83.6, with the rest within ±2. What this tested is one recipe: 1,500 captions, mostly phone screens,
rank 16 on the text model only. It doesn't rule out every caption fine-tune. But nothing in these numbers points at
a variant worth a second run, and the plan's rule was a single shot.

If all three hold, C4-4s goes ahead (clean data, then the full ship gates: default slot <= 15 s, CHAIR, holdout).
If not, the LoRA route stops here. Speed isn't part of this verdict; it's measured separately by `overhead`. Nothing
is tuned after confirm.

Guards against fitting the test:
- No training image may match a VisionBench or fresh-holdout image (perceptual-hash dedup, logged).
- The eval labels also come from a frontier model. Distilled captions use a different prompt and are judged on
  CHAIR and claim grading as well as keyword recall. A gain only on recall is treated as learning the labeller's style.

**Keep the other Florence-2 tools unchanged:** Florence-2 also backs `dense_region_caption`, flat-image
`detect_objects`, `count_objects` consensus and `verify_text`. A merged LoRA would change all of them. So keep the
adapter **unmerged** and enable it only for the caption task (PEFT's adapter disable context). Verify this works
with the native `Florence2ForConditionalGeneration` (what `florence2.py` loads) before training, against current
PEFT docs (context7), not from memory.
Permanent test: non-caption task outputs on `tests/` fixtures are identical with the adapter loaded.

**Compute: local GPU (user decision, 2026-10-11).** The user thinks the phase 11 crashes came from RAM/VRAM swapping
and wants to train locally while steering clear of it. The record to design against: phase 11 on CUDA hard-crashed
3 of 3 times within 1–4 minutes (bugcheck 0x00020001, I/O MMU, with NVIDIA PCIe "Unsupported Request" WHEA errors).
"Prefer No Sysmem Fallback" was already on in one of those attempts and didn't stop it, and a 10-minute full-VRAM
stress test didn't crash. So swapping may not be the only trigger, and heavy host-to-GPU transfer traffic is the
other likely one. The training run guards against both:

- **No swapping:** "Prefer No Sysmem Fallback" on; peak VRAM capped well under the 8 GB
  (`torch.cuda.set_per_process_memory_fraction(0.75)`), so a spike raises an out-of-memory error instead of spilling
  into shared memory. Florence-2-large in bf16 with a LoRA, batch 1–2, gradient accumulation and gradient
  checkpointing should fit in about 4 GB. Log `torch.cuda.max_memory_allocated` every N steps.
- **Little transfer traffic:** `pin_memory=False`, `num_workers=0`. Images are preprocessed to tensors once
  before training, so the training loop does no image decoding. No CPU offload (no DeepSpeed/ZeRO offload, no
  `device_map="auto"` spilling layers to the CPU).
- **Staged, with the user present and open work saved:** a 5-minute smoke run first (past the 1–4 minute window
  the crashes hit), then 30 minutes, then full runs. Telemetry on (`VisionBench\scratch\gpu_stability\`), and save
  checkpoints often so a crash loses minutes, not the run.
- **Stop rule:** one bugcheck or a WHEA PCIe error in the event log means stop and report. Don't retry with other
  settings without the user. The fallback is a cloud GPU (the distilled images already leave the machine).
- The phase 11 CUDA ban is unchanged. This allowance covers the C4-4 training runs only. Evaluation stays on the CPU
  as usual.

**Gates:** tune split (≤ 3 checkpoints/ranks), confirm split, then the fresh holdout once. Ship if recall is
above Florence-2 + text at matched length with CI above 0, CHAIR not worse, default ≤ 15 s, phase 11 Florence-2
tools unchanged.

## Decisions Made (2026-10-10)

*Confirmed by the user 2026-10-11, including the go-ahead for C4-0 and C4-1. Item 4 is refined in C4-4, Compute
(local GPU with guards and a stop rule). C4-4 itself still needs its own go-ahead and the user's fine-tune list.*

1. **C4-0 & C4-1:** Go-ahead approved (measurement hygiene and Moondream2 encode-once optimization).
2. **Fresh Holdout (C4-0c):** Approved. Antigravity / Gemini will execute this via a separate worker plan ([ROUND11_PLAN.md](file:///C:/AI/Projects/VisionBench/worker_handoff/ROUND11_PLAN.md)).
3. **Training Datasets & Distillation:** No Chinese datasets without explicit case-by-case review and risk mitigation.
   Distillation captions generated by Gemini for user-sourced images are approved.
4. **Training Compute:** Local GPU with governed parameters (governed batch size, memory caps, thermal pacing).
5. **`query_image` Decoding:** Remains unchanged for now; moving to greedy decoding is deferred to its own phase 11 evaluation.

## Implementation touchpoints (when a step ships)

| File | Change |
|---|---|
| [moondream.py](../src/fusion_vision_mcp/moondream.py) | Pass `settings` (greedy) to `caption` only, leaving `query` sampled (Decision 5); `encode_image` once per image; optional extra question |
| [florence2.py](../src/fusion_vision_mcp/florence2.py) | Generation kwargs per task (beams for caption only); optional PEFT adapter toggled for caption only |
| [analysis.py](../src/fusion_vision_mcp/analysis.py) | `visible_text_line`: title-first ordering from OCR boxes |
| [tools/text.py](../src/fusion_vision_mcp/tools/text.py) | `caption` docstring figures updated from the fresh holdout |
| `tests/test_analysis_helpers.py`, new `tests/test_moondream.py`, `tests/test_florence2_adapter.py` | Permanent tests listed under each sprint |

## Rough cost

C4-0: ~2 CPU hours (two greedy tune runs; b and b2 reuse cached output) plus a worker round. C4-1: ~2 CPU hours
(20-image timing, then 120 tune images at ~40 s). C4-2: ~3 CPU hours. C4-3: an hour of gate 0; C4-3b gate 1 about
20 minutes, gate 2 ~1.5 CPU hours; C4-3a tune runs only for candidates on a large base. C4-4: a day to build data
and scripts, local GPU hours per run (see C4-4, Compute), then the usual tune/confirm/holdout CPU runs.

Suggested order, cheapest decisive check first: C4-0 (a, b, b2) → C4-1 → C4-3b gate 1 → C4-2a → the rest.
