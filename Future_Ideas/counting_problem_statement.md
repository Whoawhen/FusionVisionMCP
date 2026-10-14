# Problem statement: object counting in a local vision server (for an outside model)

Paste everything below the line as the prompt. For more depth, attach `CLAUDE.md` (from this repo; the
"Counting is a separate tool" and "`count_lobes`" sections) and `Future_Ideas/frontier_capability_gaps.md`.

---

I maintain FusionVisionMCP, an MCP server that gives AI agents local computer-vision tools. One tool,
`count_objects(image, name)`, answers "how many X are in this image". It is the weakest tool I have against
frontier multimodal models, and every idea I have tried has failed the same way. I want fresh options.

## What it does now

- **Detector:** Grounding DINO tiny (open-vocabulary, text prompt = the object name), boxes kept at a score of
  0.425 or above. An "envelope filter" drops any box that swallows most of the other boxes' centres, because
  the detector also returns one high-scoring box around the whole group.
- **Second opinions, reported but never allowed to rewrite the count:** Florence-2-large dense region captions
  (a consensus count), and a silhouette measurement on a SAM2 mask (distance-transform persistence plus a
  convex-hull angular harmonic) that estimates how many repeated parts make up one blob. A `separable` flag
  says when a low count probably means "could not separate them" rather than a real tally.
- **Flat images** (clip art, logos, flat UI) are routed by a pixel-statistics test to Florence-2 instead.
- Other models already loaded in the server: Moondream2 (1.9B VLM), SAM2.1 Hiera-small, SAM 3, CLIP ViT-L/14,
  DINOv2-large, PP-OCRv6.

## How well it does

- 61.8% exact on 427 held-out photo counts, most of them 3 or fewer objects (43.8% at the old 0.15 floor).
- 58.9% exact on 400 mixed images (photos, screenshots, game UI, documents, charts, art), mean absolute error
  0.68; screenshots 49%, game UI 51%.
- Before the floor change, by true count on 75 COCO scenes of 3 or fewer: 1 object 85%, 2 objects 50%,
  3 objects 24%. Crowded photos ran high: 48 of 67 crowded COCO scenes overcounted, by about 12 on average.
- Estimated frontier models: 80–90% exact on scenes of 3 or fewer; weak on large or mixed counts.

## The controls that every change must pass

A synthetic fixture suite with exact counts: 10 positive cases (e.g. 8 separated or touching shapes, asked for
as "petal" and as a literal name like "pink circle") and **8 negative controls that are each exactly one
object**: a spotted ball, a striped ball, a rough-edged rod, a plain blob, a four-colour logo, and similar.
Currently 7/10 positives, 8/8 controls. Lower thresholds (0.10–0.125) score 10/10 positives and break the
controls (the spotted ball becomes 3–4 objects, the rod 2), so positives alone are never evidence.

## What has been tried and rejected (please don't re-propose these without saying what is different)

| Approach | Result |
| --- | --- |
| Moondream2's detect head | Collapses to 1 when shapes are named by colour ("pink circle") instead of "petal" |
| CountGD (ported onto stock HF Grounding DINO, exemplar conv included) | 8/10 positives but 4/8 controls; 3x the error; a flower counted 11, 4 and 5 at three resolutions |
| Tiled inference, 2x2 and 3x3 overlapping crops merged by IoU | +1 point on heavy overlap, 5–12x latency, controls drop to 7/8 then 5/8 |
| Visual-exemplar prompting | Breaks 5 of 6 controls (spotted ball → 31) |
| Interior colour boundaries (CIELAB) | Striped ball → 7, four-colour logo → 4 |
| SAM 3 as the counter | Returns no box at all for "ball", "rod", "shape", "blob" at any threshold; counted photos worse |
| Lower score floors | See controls above |

**The common wall:** everything that raises recall does it by attending to sub-object detail (zooming,
appearance matching, interior contrast), and a spot, a stripe, a logo segment and a petal look the same to
it. One test image, a paper flower with low-contrast overlapping petals (silhouette solidity 0.984), returns 1
from every method; I have accepted that one as not countable from pixels.

## Constraints

- Runs locally on a laptop CPU (Intel Core Ultra 9 275HX, 32 GB RAM); a few seconds per call is the budget
  (currently ~7 s). No image leaves the machine.
- **No Chinese-developed models.** Grounding DINO (IDEA Research) is in use but slated for replacement, so I
  don't want heavy investment tuned to it specifically. Permissive licences only (AGPL is a problem).
- Fine-tuning and LoRAs are deferred, but you may propose one if you think it is the only real route; say so
  and say what labelled data it would need.
- The calling agent is a frontier model. The tool may legitimately return candidates plus uncertainty and let
  the caller decide, as long as it says so; it must never present a guess as a tally.
- Every change is judged on a tune/confirm split of held-out photo counts with paired bootstrap intervals, and
  must hold all 8 controls.

## What I want from you

Five to eight concrete options, ranked by expected gain per unit of effort. For each:

1. The mechanism, specifically enough to implement (models, prompts, thresholds, merge logic).
2. **Why it would not turn the spotted ball into many objects.** This is the first thing I will test.
3. Expected effect on small counts (1–3) versus crowded scenes, and which of the two it targets.
4. CPU latency and memory cost, model origin and licence.
5. The cheapest experiment that would disprove it, ideally under an hour on existing data.

Include at least one option that changes what the tool returns (e.g. a calibrated range or a confidence)
rather than the count itself, and say plainly if you think part of this problem is not solvable locally.
