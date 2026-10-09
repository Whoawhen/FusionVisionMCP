# Caption sprint plan

Written 2026-10-08 (overnight). A plan only: no model in it has been downloaded or run. Each sprint waits for a
go-ahead. Steps are labelled with the model to run them (Opus 5.5 to design or judge, Sonnet 5.5 to execute a
written procedure).

## Where `caption` stands

| | Florence-2-large detailed (default) | Moondream2 long (`deep=true`) |
|---|---|---|
| Keyword recall, 280 held-out images | 53.3% | 61.5% (+8.2, CI 5.3–11.2) |
| Same, cut to Florence's word count | — | +1.9 over Florence |
| Median words | 129 | 214 |
| Names the labelled absent object | 0.4% | 1.1% |
| Median seconds per image (CPU) | 10.7 | 33.1 |

Weakest kinds: illustrations/comics 40–49%, AI images 48–52%, app/web 50%, game UI 44–59%. Label ceiling ~75%.
Most of every gain so far has been length. The harness already exists: VisionBench `caption_ab.py` (seeded,
kind-stratified 120 tune / 280 confirm split, resumable) and `caption_confirm_analysis.py` (paired bootstrap,
length-matched).

## The rule that matters most

**Recall is not enough.** Keyword recall rewards a caption for being long and for guessing. The only
hallucination check today is one labelled absent object per image, which a caption almost never names (0.4–1.1%),
so it says little. A model that wins on recall by inventing things would pass it. So the first sprint builds a
real hallucination measure and runs it on what ships, before any new model is tried.

## Sprint C1: hallucination check (Opus 5.5 to build, Sonnet 5.5 to run)

1. **Object hallucination on COCO (CHAIR).** COCO val2017 has every instance of 80 object categories annotated,
   so a caption naming a category the image does not contain is a hallucination by construction, with no judge.
   Score CHAIR-i (share of mentioned objects that are absent) and CHAIR-s (share of captions with at least one),
   using the standard CHAIR synonym list. 300 val2017 images not used anywhere else in VisionBench. Needs the
   instances annotation file (CC BY 4.0); check what VisionBench's COCO counting sets already have before
   downloading.
2. **Length-matched.** Hallucination grows with length, so report CHAIR at full length and with each caption cut
   to Florence-2's word count on the same image, as recall already is.
3. **Generalist set spot-check.** COCO is photos only; the weak kinds are not. On a 60-image sample of the
   tune split (stratified by kind), list each caption's concrete claims (objects, text, counts, colours) and
   grade them present / absent / unsure. **Grader: Gemini (user's choice, 2026-10-09)**, via GeminiVisionMCP or
   the Gemini API, so these 60 images leave the machine. Spot-check a sample of Gemini's grades by hand before
   trusting it: on AI defects it called glitches the user did not (100% caught, 52% clean left alone on phase 11).
4. **Baselines.** Run all of it on Florence-2 detailed and Moondream2 long. This is the bar the bake-off has to
   clear, and it may show `deep=true` already hallucinates more than its recall gain is worth.

Output: `caption_hallucination.py` and a results JSON in VisionBench; a scoreboard line. No change to the server.

## Sprint C2: bake-off (Opus 5.5 to plan the prompts, Sonnet 5.5 to run)

### Candidates

All non-Chinese, including the language backbone. Facts below are from memory and get checked at gate 0.

| Model | Maker | Params | Licence (verify) | Why it is here | Watch |
|---|---|---|---|---|---|
| Gemma 3 4B-it | Google | 4.3B | Gemma Terms of Use (commercial use allowed, use policy attached; not OSI) | Strong general captioner, 896 px SigLIP with pan-and-scan for documents and UI | CPU speed in bf16; a GGUF/int4 path adds llama.cpp as a dependency. Also check Gemma 3n E2B/E4B (built for on-device) and anything newer from Google |
| PaliGemma 2 3B, DOCCI fine-tune at 448 px | Google | 3B | Gemma Terms of Use | Fine-tuned on DOCCI for long, detailed captions with few invented details: aimed squarely at this problem | Caption-only model, no chat; 448 px may be too low for screenshots |
| Phi-4-multimodal-instruct | Microsoft | 5.6B | MIT | Best licence here; good on charts, documents, screenshots | Largest; likely the slowest on CPU, ~11 GB in bf16 |
| SmolVLM2 2.2B (and 500M) | Hugging Face | 2.2B / 0.5B | Apache-2.0 | Fastest; a candidate to replace the *default* rather than `deep` | Probably weaker than Moondream2; the question is whether it beats Florence-2 at Florence's speed |
| Moondream, newer revision | Moondream (US) | 1.9B (moondream2 revisions); Moondream 3 preview is a larger mixture of experts | moondream2: Apache-2.0. Moondream 3 preview: a business source licence, check terms | Cheapest swap: the server already loads moondream2 for `query_image`, so a newer revision may lift both tools | A revision change also moves `query_image`; re-run its phase 11 numbers too |
| MolmoE-1B | Allen AI | 7B total, 1B active (OLMoE base, AI2) | Apache-2.0 | Trained on PixMo dense captions, written to be detailed and grounded | 7B of weights to hold in memory despite 1B active; transformers support needs `trust_remote_code` |

Excluded on origin: Qwen-VL, InternVL, MiniCPM-V, GLM-V, DeepSeek-VL, Ovis, Kimi-VL, and Molmo-7B-D/O (Qwen2 or
other Chinese bases).

### Gates

0. **Paper (Opus 5.5, no downloads):** licence, maker and language-backbone origin confirmed from the model
   cards; disk and memory sizes. Anything whose licence the user would not accept stops here.
1. **Smoke (20 images, CPU):** load time, peak resident memory, median seconds per image. Drop a model above
   ~12 GB resident or ~60 s per image median, unless a quantised path brings it under (record which).
2. **Tune split (120 images):** keyword recall, recall at Florence's length, median words, by kind; the C1
   hallucination measures (CHAIR on the COCO set, absent-object rate). At most three prompts per model, chosen
   here and frozen.
3. **Confirm split (280 images), finalists only, nothing changed:** paired bootstrap against Florence-2 detailed
   and Moondream2 long, full and length-matched; CHAIR; latency.

### What would ship

- **New `deep` model:** beats Moondream2 long at matched length with a confidence interval above zero, and
  hallucinates no more (CHAIR within noise or better), within the memory budget.
- **New default:** beats Florence-2 detailed at matched length at roughly its speed (≤ ~15 s/image on CPU).
  Only SmolVLM2 and maybe PaliGemma 2 have a realistic chance.
- **Nothing:** a fine result. Then the sprint's output is the hallucination numbers and a recommendation.

Memory budget: the server already holds Florence-2-large, Moondream2, CLIP ViT-L/14, SAM2 and Grounding DINO
under idle release. A new captioner that replaces one of them is cheap; one added beside them needs
`--memory-mode` thought.

## Parallel experiment: OCR text in the caption prompt (Sonnet 5.5)

Signage, documents, app/web and game UI lose recall on names the model cannot read. Pass the `ocr` engine's text
into the caption prompt ("Text visible in the image: ...") for the candidates that take a prompt (not
PaliGemma's fixed task prompt or Florence-2). Measure on the text-heavy kinds, and check the control: a
text-free image gets an empty OCR result and therefore the same prompt as before, so it cannot get worse. The
prompt only needs the text, so this carries over unchanged when PP-OCRv6 is replaced.

## Not in this plan

- LoRAs and fine-tunes: still deferred until the suite reaches "good".
- Any tuning of Grounding DINO or PP-OCRv6 (slated for replacement).
- Changing `query_image`'s model, except as a side effect of a newer Moondream revision, which gets its own
  phase 11 VQA re-run before shipping.

## Fitting-the-test guards

- Prompts and generation settings are chosen on the tune split only. The confirm split is run once per finalist.
- Every recall number is reported beside its word count and its length-matched value.
- Hallucination is measured on images no model was chosen on (COCO set) as well as the tune split.

## Rough cost

C1: an evening of building plus a few CPU hours, plus however long the claim grading takes the grader.
C2: gate 0 an hour; gate 1 about an hour of downloads (~25 GB in all) and CPU; gates 2–3 roughly 1–3 CPU hours
per model per split at the measured speeds. Two to three overnight runs in total.
