# FusionVisionMCP v0.8.1 — Sprint Plan ("Third Vision Pass")

Follow-on to `FusionVisionMCP_v0.8.0_Sprint_Plan.md`, in the same shape and under the same rules.
Where v0.7.1 was the "Second Vision Pass" — fixes for gaps a head-to-head comparison against
Claude's native vision exposed — this release is the third such pass, run against shipped v0.8.0
on 2026-09-11.

The difference this time: the pass did not find a missing capability. It found a **shipped feature
reporting fabricated evidence**, and a test suite structurally incapable of catching it. That makes
this a correctness release, not a feature release.

All work happens in a fresh worktree at `C:\AI\MCP\FusionVisionMCP-v0.8.1-dev` (branch
`v0.8.1-dev`, cut from `main` at `008a28c`). The v0.8.0 worktree has already been removed; the
checkout at `C:\AI\MCP\FusionVisionMCP` stays on `main` and **is the live install the user runs
daily** — changing its branch changes what's live immediately. Create the worktree, don't reuse
the main checkout:

```powershell
git worktree add C:\AI\MCP\FusionVisionMCP-v0.8.1-dev -b v0.8.1-dev main
```

A sprint is "**built**" and "**verified**" by exactly the definitions in the v0.8.0 plan's
"Division of labor and verification" section — that section is not restated here and still governs.
Read it before picking up any sprint.

---

## What the third vision pass measured

Four documented v0.8.0 claims were tested live through the real `fusionvision` MCP server, each
alongside Claude's native vision on the identical image. Three reproduced exactly as documented.
The fourth did not.

| Case | Fixture | Result |
|---|---|---|
| `caption` misreads embedded text | `FusionVisionMCP-Dark.jpg` | **As documented.** Caption still says "FusionVisionMP"; `verify_text=true` returns verbatim "FusionVisionMCP", `caption_text_warning: true`, and `caption_corrected` substitutes correctly. Mitigation works. |
| Petal collapse + ambiguity reporting | `tests/sample.jpg` | **As documented.** `count: 1`, `separable: "no"`, `semantic_ambiguity: true`, `count_semantics: "minimum_visible_instances"`, `estimates.outline: 8`. Native vision counts 8 petals directly — each is a different pastel color, an appearance cue a shape-only detector does not use. |
| Generic-noun full-frame guard | `tests/detect_blank_canvas.png` | **As documented.** `detect_objects("object")` returns `count: 0`, `bboxes: []`. |
| **Structured anatomical inspection** | `tests/defect_test6.jpg` | **Fails.** See below. |

### The failure, stated precisely

`query_image(structured_analysis=true)` on `tests/defect_test6.jpg` returned three anomalies, each
`severity_estimate: "high"`, each carrying an `Observation` with `corroborated: true`:

```
"Counted 10 arms for 2 people"
"Counted 9 legs for 2 people"
"Counted 5 heads for 2 people"
```

Two things are true about that image at once, and both matter:

1. **It is genuinely AI-generated.** The red cap, the t-shirt graphic, the black sweatshirt and the
   clipboard all carry garbled pseudo-text — the most recognizable generation artifact there is.
   Flagging the image as anomalous is the right verdict, and the CHANGELOG is right to treat it as
   a defect fixture.
2. **None of the reported evidence is real.** The anatomy is ordinary: two foreground men with two
   arms and one head each, plus at least one blurred background figure. There are not 10 arms, 9
   legs, or 5 heads. The numbers are detector noise that happened to clear a threshold.

So the feature produced a defensible conclusion from invented measurements, and labelled those
measurements `corroborated: true`. By this project's own standard — "measure, don't judge", and
`spatial_relations`' founding rule that the tool reports numbers and stops — that is worse than
returning nothing. A caller acting on `corroborated: true` is acting on a literal.

Meanwhile the one artifact that is unmistakably present and directly measurable — garbled text —
is not reported at all, in a codebase that has had two independent OCR engines wired in since
Sprint 7/8.

### Root causes, all confirmed in source

| # | Cause | Location |
|---|---|---|
| 1 | `corroborated=True` is a **hardcoded literal**. The whole "Defect Corroboration Strategy" rests on a field that is asserted, never computed. | `inspection.py:68`, `:85`, `:100`, `:117` |
| 2 | Parts are counted **globally across the frame**, with no spatial association to any person box. Background figures and partial bodies inflate the numerator while `persons` — the denominator — counts only confidently-detected whole people. | `inspection.py:42-56` |
| 3 | **No dedup between part boxes.** A bent elbow yields upper-arm, forearm and whole-arm detections; nothing merges them, unlike `count_objects`, which at least has the group-envelope filter. | `inspection.py:48-52` |
| 4 | Thresholds (`persons * 2 + 1`) were chosen against **stubbed integers**, never against measured detector output on a real image. | `inspection.py:64` |
| 5 | **Zero real-image coverage.** `tests/test_inspection.py` is entirely stubs on a blank 10×10 image. `tests/test_server.py:781` runs `structured_analysis` on the *flower*, which contains no people, so `persons == 0` returns early and the assertion is vacuous. | both files |

Cause 5 is why this shipped. The rule this project already wrote for itself after v0.6.0 — *"a flag
meant to catch a known failure mode has to be tested against that exact failure mode, not just
checked for producing a valid-shaped value"* — was not applied to Sprint 10. The stub tests assert
that `10 > 2*2+1` evaluates true. They cannot fail on any real image, because they never see one.

---

## Model grade: which model to hand each sprint to

New in this plan, layered on top of — not replacing — the v0.8.0 plan's junior/senior framework.
That framework's four junior-safe conditions and its senior-required triggers still define the
split; this section only says which model each side of that split should be routed to.

| Grade | Model | Corresponds to |
|---|---|---|
| **Sonnet grade** | Sonnet 5 | The v0.8.0 plan's **junior-safe** bar: all four conditions met — ground truth already exists, a close existing pattern to follow, success mechanically checkable, contained blast radius. |
| **Opus grade** | Opus 5 | The v0.8.0 plan's **senior-required** triggers: inventing or validating what ground truth means; first integration of an unfamiliar model or signal; choosing a threshold or tradeoff not purely read off gathered data; becoming a shared default across tools; a refactor whose equivalence isn't mechanically checkable. |

Three rules about applying it, each earned by something that already went wrong on this project:

- **Grade the task, not the builder.** Sprint 4's failure was not carelessness — the sweep ran, the
  numbers were real. It was a *senior* problem (invent a valid methodology; be skeptical of your own
  result) handed over shaped like a *junior* one (fill in this table), and the builder defaulted to
  making the number look good. Sprint 10 repeated the same shape: "wire up an anomaly check" reads
  mechanical, but deciding what counts as corroborating evidence is a judgment call with no
  mechanical answer. Route on the judgment load the task actually carries.

- **Verification is always Opus grade, regardless of who built it.** The v0.8.0 plan already
  requires independent verification of every sprint including junior ones; this release is the
  proof of why. Sprint 10 was correctly tagged *senior* and still shipped a hardcoded `True` —
  because verification checked that the tests passed, and the tests were stubs. Verification's job
  is to ask whether the test could ever have failed, which is the exact skeptical-judgment work
  that defines the Opus tier.

- **A Sonnet-grade sprint that turns surprising is a stop, not a judgment call.** Same boundary
  Sprints 3b, 15 and 16 drew: if the numbers come back very high or very low, or the pre-specified
  definition stops fitting the data, stop and escalate rather than tuning. Crossing that line is
  what converts a Sonnet-grade sprint into an Opus-grade one mid-flight.

---

## Status

- [ ] Sprint 17 — The missing negative control (Sonnet)
- [ ] Sprint 18 — Associate parts with people; dedup part boxes (Opus)
- [ ] Sprint 19 — Compute `corroborated`; drop unearned severity (Sonnet)
- [ ] Sprint 20 — Garbled-text artifact detection (Opus)
- [ ] Sprint 21 — Correct the record: CHANGELOG + CLAUDE.md (Opus, with a Sonnet-able part)
- [ ] Sprint 22 — Release checkpoint and merge (Opus, unconditionally)

Nothing is started. Checkboxes are the verifier's to tick, never the builder's.

---

## Sprints

### Sprint 17 — The missing negative control (2 hrs) — junior / **Sonnet grade**

Build the test that should have existed before Sprint 10 shipped, and run it against **unmodified
`main`** first so the failure is recorded as a measured baseline rather than assumed.

A real photograph of two or more people, with bent arms and ordinary occlusion, must produce **zero
anomalies**. No new fixture curation is required: `benchmarks/coco_images/` already holds 35
CC-BY, license-audited real photographs with COCO's own ground-truth `person` annotations from
Sprint 15. Filter to the person-bearing images, run `analyze_inspection` against each, and record
per-image: COCO's ground-truth person count, the detector's `person`/`arm`/`leg`/`head`/`finger`
counts, and how many anomalies were raised. Output a CSV to `benchmarks/results/` in the same shape
as `open_images_spatial_agreement.csv`.

Add the same as a real regression test — not a stub — asserting zero anomalies across the
person-bearing COCO set. Expect it to fail on current `main`; commit it failing-and-marked
(`xfail` with a reason pointing at this plan), so Sprint 18 has a green/red signal to work against.

This is `neg_spotted_ball` applied to inspection: the cheapest possible disproof, which the feature
never had.

*Sonnet grade because*: ground truth is COCO's own professional annotation, not invented; the
statistic to record and the CSV format are fully pre-specified above; success is mechanically
checkable (a count of anomalies, which must be zero); and it adds a test without touching shipped
behavior. *Stop and escalate if*: a materially non-zero share of COCO person photos raise anomalies
even after Sprint 18 — that would mean the per-person ratios themselves are wrong, which is a
threshold judgment and therefore Opus work.

### Sprint 18 — Associate parts with people; dedup part boxes (4–5 hrs) — senior / **Opus grade**

The actual fix for causes 2 and 3. Two changes to `_check_anatomy`:

**Association.** Stop tallying the frame. Require each part box to be contained within (or overlap
past a chosen fraction of) a detected `person` box, and tally **per person**. `geometry.py` already
measures containment for `spatial_relations` — reuse it rather than writing a second containment
notion. The output shape should express what an artifact actually looks like: *this specific person
has three arms*, which a global frame tally is structurally incapable of saying. A part that
belongs to no detected person is evidence about detection coverage, not about anatomy, and must not
enter a ratio.

**Dedup.** Merge overlapping detections of one limb by IoU before counting, mirroring the
group-envelope logic already in `grounding_dino.py`. A bent arm producing upper-arm + forearm +
whole-arm boxes is one arm.

Re-run Sprint 17's CSV after each change separately, so the contribution of association and dedup
are measured independently rather than as one lump. Flip the `xfail` to a passing assertion only
when the COCO set is genuinely clean.

*Opus grade because*: the IoU cutoff and the containment fraction are thresholds that cannot be
read directly off gathered data — they are the same category of choice as `count_objects`' 0.15 box
threshold and `count_lobes`' `_SMOOTH_FLOOR_PX`, both of which this project found by sweep against
positive *and* negative controls. It also becomes shared default behavior for every
`structured_analysis` caller (rule 4). *The specific trap to avoid*: over-correcting until nothing
ever flags. `tests/defect_test6.jpg` must still be reported as anomalous at the end of this sprint —
but on Sprint 20's text evidence, not on invented limb counts. Losing the positive case to fix the
negative one is the failure mode; both controls hold or the sprint isn't done.

### Sprint 19 — Compute `corroborated`; drop unearned severity (2 hrs) — junior / **Sonnet grade**

Fix cause 1, once Sprint 18 has made a real definition available.

`corroborated` becomes a computed boolean with exactly this pre-specified meaning: the part count
**survived dedup**, **was associated with a specific person box**, and **violates that person's own
ratio**. All three, or it is `false`. A bare global tally reports `corroborated: false` and carries
its raw numbers in `evidence` where a caller can see them for what they are.

Delete `severity_estimate`, or reduce it to a field that reports which check fired. "high" is a
judgment, it was never measured, and this project's standing rule is that the tool measures and the
calling model judges — the same rule that keeps `spatial_relations` from rendering verdicts.
Removing it is a breaking change to the payload; note it in the CHANGELOG rather than preserving a
field whose only content is unearned confidence.

Rewrite `tests/test_inspection.py` so the stubs test arithmetic only and are labelled as such, and
so no stub test can ever stand in for real-image coverage again. Replace the vacuous
`test_server.py:781` flower assertion with one that actually exercises a person-bearing image.

*Sonnet grade because*: the definition of `corroborated` is fully pre-specified in the paragraph
above, the pattern to follow (a computed flag backed by evidence fields) is well established in
`count_objects`' `separable`/`consensus` and `ambiguity.py`, and success is mechanically checkable.
This is only Sonnet-safe **after** Sprint 18 — attempted before it, deciding what corroboration
means is precisely the judgment call that makes it Opus work.

### Sprint 20 — Garbled-text artifact detection (4–5 hrs) — senior / **Opus grade**

Measure the artifact that is actually present. Nonsense rendered text is the most reliable visible
signature of image generation, and the components are already here: Florence-2 OCR, Granite-Docling
specialist OCR (Sprint 7), and `textmatch.py` (Sprint 8).

The signal: run both OCR engines over each detected text region; strong disagreement between the
two on the same region, **plus** tokens that fail a lexicon check, is measurable evidence of
synthetic text. Report both transcriptions and the disagreement score as `evidence` on an
`Observation` — an observation with real numbers behind it, never a verdict about the image's
provenance.

Negative controls are the whole sprint, and must be held before it ships:
- `FusionVisionMCP-Dark.jpg` — real rendered text, both engines agree exactly. Must not flag.
- A COCO photo carrying real signage — must not flag.
- A photo with no text at all — must not flag, and must not load the OCR models needlessly.

Positive control: `tests/defect_test6.jpg`, where four separate regions carry garbled pseudo-text.

*Opus grade because*: this defines a new kind of measurement, which the junior-safe framework rules
out categorically — nobody has established what "ground truth" means for synthetic text here. *Where
the false positives will hide*: the lexicon. Brand names, logos, foreign-language signage, stylized
type and legitimate acronyms are all non-dictionary words on real photographs, and a naive
dictionary check will flag every one of them. Expect the lexicon design, not the OCR plumbing, to
consume this sprint. If the negative controls cannot be held, **ship nothing and record the negative
result** — the project has done exactly that before with CountGD, tiling, exemplar prompting and the
CIELAB interior-colour experiment, and a documented disproof is a real deliverable here.

### Sprint 21 — Correct the record (2–3 hrs) — senior / **Opus grade** (one part Sonnet-able)

**The CHANGELOG correction (Opus).** The Sprint 10 entry presents "successfully flagging the 10 arms
generated for 2 people" as validation. It is detector noise that cleared a threshold on an image
whose real defect is elsewhere. Rewrite it with the measured numbers from Sprints 17–20. Precedent
for the tone already exists in this file's own history — the Sprint 4 entry that opens *"The first
version of this sweep measured nothing real, and that was caught before shipping."* Match it.

**The CLAUDE.md refresh (Opus).** `CLAUDE.md` stops at the v0.6.0 section and is missing Sprints
6–16 entirely. It is the file that steers every future agent on this repo, and it currently hands
them a gap list that v0.8.0 already moved past, with no mention of Granite-Docling, the routing
cascade, adaptive thresholding, the COCO and Open Images ground-truth suites, the IQA model, the
reasoner, or `structured_analysis`. Bring it current, in its existing voice: the *why* behind each
decision and the negative results worth not re-investigating, not a feature list.

**The v0.8.0 doc-vs-reality audit (Sonnet-able).** Every remaining behavioral claim in the v0.8.0
CHANGELOG that has not been live-tested since it was written, listed with a pass/fail against a
real fixture call. Mechanical, pre-specified, and the finding that motivated this whole release
argues strongly for doing it: one of the four claims spot-checked on 2026-09-11 was wrong, and
nothing rules out others. *Escalate anything that fails* rather than fixing it inside this sprint.

*Opus grade because*: synthesizing ten sprints of history into `CLAUDE.md`'s why-focused voice is a
judgment-heavy writing task, and getting the CHANGELOG correction's severity right — honest without
overstating — is the same category of call.

### Sprint 22 — Release checkpoint and merge (2–3 hrs) — senior / **Opus grade, unconditionally**

Version bump to `0.8.1` across `pyproject.toml` / `manifest.json` / `server.json`; full regression
and benchmark run (counting fixtures' 9/10 positives and 8/8 negatives must hold, plus Sprint 17's
new COCO inspection control); final CHANGELOG v0.8.1 entry with real fixture evidence; update
`README.md`'s "Why FusionVisionMCP?" section, which currently claims v0.7.1 closed the last vision
pass's gaps and says nothing about this one; merge `v0.8.1-dev` into `main`; re-run
`uv tool install --editable . --force ...` **only if dependencies changed** (Sprint 20's lexicon may
add one); confirm the live server through real `mcp__fusionvision__*` calls.

*Opus grade, unconditionally*: this is the merge into `main`, which is the live-install boundary.
Rule 4 of the junior-safe framework rules it out categorically, independent of how mechanical the
individual steps look — exactly as Sprint 14 was, every time.

---

## Dependencies between sprints

```
Sprint 17 ── independent, do first (establishes the failing baseline everything else is measured against)
Sprint 17 → Sprint 18 (needs the control to know when the fix is real)
Sprint 18 → Sprint 19 (corroboration can't be defined until association and dedup exist)
Sprint 20 ── independent (but Sprint 18's "defect_test6 must still flag" requirement depends on it landing)
Sprint 21 ── depends on 17-20's measured numbers for the CHANGELOG correction;
             the CLAUDE.md refresh and the doc audit are independent and can start any time
Sprint 22 ── last, depends on everything else in scope
```

Sprints 17, 20, and Sprint 21's CLAUDE.md/audit portions have no upstream blocker and can run in
parallel with the 18 → 19 chain.

**Grade tally, for quick assignment:** Sonnet — 17, 19, and Sprint 21's doc-vs-reality audit.
Opus — 18, 20, 21, 22, and **every verification pass without exception**.

Opus outnumbers Sonnet here, which tracks: a correctness release whose central finding is *"the
tests could not have failed"* is mostly judgment work. The genuinely mechanical parts are the
control that should have existed (17), the flag that should have been computed (19), and the audit
of claims nobody has re-checked (21c) — and all three are only mechanical because this plan
pre-specifies what correct means for them.

---

## The one-line version

v0.8.0's `structured_analysis` returns `corroborated: true` next to numbers it invented, on an image
whose real defect it never looked for, and no test in the suite could have caught it. Sprints 17–19
make the flag mean something, Sprint 20 measures the defect that is actually there, and Sprint 21
corrects what the docs claim was verified.
