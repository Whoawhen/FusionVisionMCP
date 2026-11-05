# Future_Ideas: plans and where they live

Every plan for this repo lives in this folder. Reorganised 2026-11-02.

## Live

| File | What it is |
| --- | --- |
| [Critique_Composition_Improvement_Plan.md](Critique_Composition_Improvement_Plan.md) | `critique_composition` plan; R3 step 0 next |
| [Open_Ideas_By_Tool.md](Open_Ideas_By_Tool.md) | Every still-valid idea from the archived plans, by tool, with its source |
| [frontier_capability_gaps.md](frontier_capability_gaps.md) | The scoreboard against frontier models (a living reference, not a plan) |
| [image_sources_to_consider.md](image_sources_to_consider.md) | Candidate image sources for VisionBench (reference) |

A new plan goes here at the top level. When it is finished, superseded or closed, `git mv` it into `archive/`,
add a row below, and move any idea it leaves open into `Open_Ideas_By_Tool.md`.

## Archive

Finished, superseded or parked. Older commits, VisionBench scripts and notes cite these at their old paths; the
"Was at" column maps them.

| File in `archive/` | Was at | Tool | Outcome |
| --- | --- | --- | --- |
| FusionVisionMCP_v0.8.0_Agentic_Implementation.md | repo root | all | v0.8.0 shipped |
| FusionVisionMCP_v0.8.0_Sprint_Plan.md | repo root | all | v0.8.0 shipped |
| HANDOFF.md | repo root | all | v0.8.1 worktree handoff; released |
| FusionVisionMCP_v0.8.1_Sprint_Plan.md | `docs/` | all | v0.8.1 shipped |
| ADAPTIVE_THRESHOLD_RISK_ANALYSIS.md | `docs/` | `count_objects` | Analysis; prevalence question carried to the backlog |
| gemini-optional-backend.md | `docs/plans/` | `caption`, `query_image` | Built on an unmerged branch; parked |
| Caption_Ideas.md | `docs/plans/` | `caption` | Folded into the C4 plan 2026-10-17 |
| caption_sprint_plan.md | `Future_Ideas/` | `caption` | C1-C2 done |
| caption_sprint_c4_plan.md | `Future_Ideas/` | `caption` | C4 done; C4-4 LoRA dropped |
| c4_2a_handoff.md, c4_3b_gate1_handoff.md, c4_3b_gate2_handoff.md, c4_5_image_type_handoff.md | `Future_Ideas/` | `caption` | Run |
| c4_5b_label_check_design.md, c4_6_narrow_type_design.md, c4_7_text_line_design.md | `Future_Ideas/` | `caption` | Run; C4-7 shipped |
| Caption_Improvement_Plan.md | `Future_Ideas/` | `caption` | LFM default and `deep` shipped |
| caption_overnight_workplan_2026-10-22.md | `Future_Ideas/` | `caption` | Run |
| OCR-TextVQA_Improvement_Plan.md | `Future_Ideas/` | `ocr` | Ended with nothing to ship (0.18) |
| ocr_blur_improvement_plan.md | `Future_Ideas/` | `ocr` | Sharpened pass and `lfm_read` shipped 2026-10-30 |
| COUNTING_IMPLEMENTATION_PLAN.md (local only, gitignored) | repo root | `count_objects` | Revision 2 of 2026-08-25; superseded by everything below |
| counting_problem_statement.md, counting_plan.md | `Future_Ideas/` | `count_objects` | Superseded by the improvement plan |
| Counting_Improvement_Plan.md | `Future_Ideas/` | `count_objects` | Closed by the user 2026-10-31 |
| detector_ocr_replacement_plan.md | `Future_Ideas/` | `detect_objects`, `count_objects`, `ocr` | Sprint R done; SAM 3 detect and the 0.425 count floor shipped |
| sam3_expansion_plan.md | `Future_Ideas/` | `detect_objects`, `spatial_relations` | Flat detection shipped; spatial reverted |
| Query_Image_Track1_handoff.md | `Future_Ideas/` | `query_image` | Track 1 shipped 2026-10-31 |
| Query_Image_Improvement_Plan.md | `Future_Ideas/` | `query_image` | Done 2026-11-05: V5, presence check and OCR cleaning/excerpts shipped; routing regex failed Gate V9 |
| memory_manager.md | `Future_Ideas/` | server | M1-M4 done; M5 conditional |
| model_variants_and_loras.md | `Future_Ideas/` | all | Matrix measured or ruled out; remaining ideas in the backlog |
