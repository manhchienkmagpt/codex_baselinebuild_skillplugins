---
name: cv-baseline-workflow
description: Build optimized, minimal-preprocessing basic and inference-only hardcore computer vision notebooks plus a Vietnamese proposal for image/video tasks from descriptions or Excel task tables; include concise basic/hardcore manual prompt guides for independent Gemini sessions.
---

# CV Baseline Workflow


Resolve dataset/version, sample schema, labels, task metric and image/video/sequence split unit. Default `outputs/cv/<STT>_<Tên bài toán>/` contains baseline.ipynb, baseline_basic.ipynb, baseline_hardcore.ipynb, huong_giai_quyet.md, baseline_basic_promting.md and baseline_hardcore_promting.md. Apply the [output naming convention](references/output_naming.md), even in a new chat without a naming request: preserve supplied STT; otherwise reuse a matching task directory or allocate the next available number. Explicit user paths take precedence. Respect report-only/notebook-only/selected-variant requests.

Optimized uses strong task-specific preprocessing and a self-built task model, optionally with an eligible pretrained backbone. Basic uses only minimal task-correct image/video conversion and a simpler scratch model with full training/evaluation. Hardcore loads raw data, defines one untrained scratch model and directly infers, with mechanical conversion inside the model only. Preserve existing optimized content during companion updates.

Read the companion engineer/architect skills. Apply the shared contracts below, inspect actual data and closest fewshot sources, generate independent notebooks, then validate each against the proposal using scripts/validate_artifacts.py <notebook> <report>. Manually review geometry, split leakage, metrics and pretrained dataflow; static validation is insufficient. Smoke-test variant-appropriate behavior and report real execution status. Clean only task-created temporary assets; preserve user files.


## Required references

Read the [three-variant contract](../cv-baseline-workflow/references/dual_notebook_contract.md), [CV task guide](../cv-baseline-workflow/references/cv_tasks.md), [backbone policy](../cv-baseline-workflow/references/pretraining_policy.md), [hardcore contract](../cv-baseline-workflow/references/hardcore_contract.md), [runtime settings](../cv-notebook-engineer/references/runtime_config.md) and [fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Read the closest packaged source notebooks. Spreadsheet input follows the [Excel contract](../cv-baseline-workflow/references/excel_input_contract.md).

Support image classification/pairs, detection, semantic/instance segmentation, keypoints, restoration, video classification, action recognition and tracking. Route OCR/captioning/VQA and other language-output tasks to the NLP skills; do not duplicate those policies here. Plan for one 32 GB GPU, fixed editable constants and ordinary PyTorch precision; no AMP, VRAM probes, adaptive batches/precision or OOM controllers. Sources and helpers are packaged locally; runtime notebooks never depend on this skill's files.

## Manual prompt companions

Use responsibility-based format 4 under the [manual prompt guide contract](references/prompting_contract.md). Basic focuses on preprocessing, model and hyperparameters, reusing the existing training pipeline. Hardcore adds data loading/splitting and loss/optimizer/training, while the delivered hardcore notebook itself remains inference-only. Implement the optimized method's idea with compatible I/O rather than reproducing its notebook. Split only into complete work items when token fit requires it, never every helper or partial function.

Attach exact relevant basic code; hardcore generation uses prose/context. Include imports and short placement comments with returned code. Reuse stable interfaces directly; request a summary only when a later step consumes it, at most 3 lines and a target of 80 tokens, shorter than its input. Shared settings have one owner and flow through context. Keep data/metric/alignment and pretraining constraints, while choosing suitable tunable values. Do not add standalone imports, audit, unchanged-helper, metric, export or reload sessions.

Each chat is independent. Target at most 1800 tokens with reserve within the platform limit input + output < 2000. Code output may exceed input; only summaries have the shorter-output requirement. The user's web chatbot cannot enforce a separate output cap, so do not claim prompt wording guarantees it. Token verification remains unverified without real counters. Preserve notebooks/reports during guide updates.

Use `scripts/migrate_prompt_guides.py <task_dir>` to preview formats 1-3 to 4, `--write` to save and `--refresh` to rebuild format 4. Validate with `scripts/validate_prompt_guides.py <task_dir>` (optional `--variant`). Run `tests/test_validate_prompt_guides.py` and `tests/test_focused_prompt_guides.py` for legacy compatibility and focused-guide regression checks.

## Validator regression checks

Run `python <skill_dir>/tests/test_validate_artifacts.py` from any writable workspace to exercise offline variant, training/pretraining and prediction-contract checks. Tests do not download models or datasets. Also run `python <skill_dir>/tests/test_validate_prompt_guides.py` for guide references, dependency transfer and token-budget arithmetic. Run `python <skill_dir>/tests/test_focused_prompt_guides.py` for format-4 grouping, short-summary and migration checks.

## Training and checkpoint style

Follow the [training style contract](../cv-notebook-engineer/references/training_style.md). For optimized/basic, use inline epoch/batch loops for straightforward training and concise inference-only saves in the validation-improvement branch. train_epoch and save_checkpoint are not mandatory. Use helpers only for real reuse/readability; add resume state only when explicitly requested.

## Focused solution report

Any requested huong_giai_quyet.md follows the [five-part report contract](../cv-solution-architect/references/report_contract.md). Focus on the optimized solution and its reasoning; three notebooks do not require three report chapters. Put detailed execution/check status in the handoff. Notebook-only requests still need no new report.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods. The packaged visual samples are image-to-text references for style/components; coverage gaps and CV adaptation boundaries in the guide are binding.
