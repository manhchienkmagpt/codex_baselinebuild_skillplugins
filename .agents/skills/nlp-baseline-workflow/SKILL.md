---
name: nlp-baseline-workflow
description: Build optimized, basic and inference-only hardcore NLP or image-to-text notebooks plus a Vietnamese proposal from descriptions, Excel cells, tables or workbooks. Use packaged fewshot, fixed parameters and task-specific pretraining rules; include concise basic/hardcore manual prompt guides for independent Gemini sessions.
---

# NLP Baseline Workflow

Generate three independent notebooks, a shared Vietnamese proposal and two manual prompt guides per task. Respect report-only, notebook-only and explicit variant selections. Use Vietnamese explanations and logs; normal code identifiers remain English.

## Required reading

Read the [three-variant contract](references/dual_notebook_contract.md), [pretraining policy](references/pretraining_policy.md), companion [engineer](../nlp-notebook-engineer/SKILL.md) and [architect](../nlp-solution-architect/SKILL.md). Before authoring, read the [fewshot guide](../nlp-notebook-engineer/references/fewshot_guide.md) and the closest packaged notebook sources. Spreadsheet inputs follow the [Excel contract](references/excel_input_contract.md).

## Workflow

1. Resolve each selected task, dataset/version, real schema, labels, raw splits, grouping unit and official metric. A workbook is task metadata, not training data. Prefer supplied data; do not silently substitute a dataset.
2. Default to `outputs/<STT>_<Tên bài toán>/` under the [output naming convention](references/output_naming.md), even in a new chat without a naming request. Preserve supplied STT; otherwise reuse a matching task directory or allocate the next available number. Deliver baseline.ipynb, baseline_basic.ipynb, baseline_hardcore.ipynb, huong_giai_quyet.md, baseline_basic_promting.md and baseline_hardcore_promting.md. Honor explicit user paths, narrower requested subsets and unrelated existing files.
3. Keep the optimized preprocessing and model quality. When updating an existing task, preserve baseline.ipynb unless the user requests an optimized change. Basic uses minimal task-correct preprocessing and a materially simpler scratch model, with complete training/evaluation. Hardcore has only data loading and one scratch model with direct inference, no training or separate preprocessing pipeline.
4. Preserve common dataset/version, sample membership, IDs, label semantics and metric protocol. Basic may change features, not raw splits. Hardcore uses test when available, otherwise the declared validation/held-out; its untrained predictions are an execution example, not a trained quality baseline.
5. Write the shared report under the [five-part report contract](../nlp-solution-architect/references/report_contract.md): explain the problem, data/approaches, optimized preprocessing, selected model and justified hyperparameters. Keep it focused on the optimized method; do not add required three-way tables, runtime chapters or history. Never invent optimality or measured gains.
6. Generate natural Markdown/code cells. Optimized/basic have full training sections; hardcore follows the [two-part contract](references/hardcore_contract.md). Use one 32 GB GPU as a design target, fixed editable constants and ordinary PyTorch precision. No AMP, VRAM probes, adaptive batch/precision/accumulation or automatic OOM retries.
7. Run scripts/validate_artifacts.py separately for each notebook against the shared report. The CLI remains: python <skill_dir>/scripts/validate_artifacts.py <notebook> <report>. Review leakage, masks/alignment, component eligibility and protocol manually; a static pass cannot prove them.
8. Where data/dependencies permit, smoke-test optimized/basic forward/backward, parameter updates, checkpoint reload and metric; test hardcore fresh-kernel inference, unchanged parameters and output IDs/schema. Report actual checks per variant. Full training requires appropriate scope and resources.
9. Clean only task-created verification/runtime assets. Retain the six requested primary files, preserve user data and pre-existing files, and verify no dependency on removed helpers, other variants, original workbook or fewshot at runtime. Notebooks may create their own runtime artifacts normally.

## Missing data

Still deliver configured notebooks with explicit real-source paths/schema and actionable errors. Do not invent labels, successful execution or scores. Distinguish static validation, fixture smoke tests, real-data smoke tests and full training.

## Manual prompt companions

Use responsibility-based format 4 under the [manual prompt guide contract](references/prompting_contract.md). Basic focuses on preprocessing, model and hyperparameters, reusing the existing training pipeline. Hardcore adds data loading/splitting and loss/optimizer/training, while the delivered hardcore notebook itself remains inference-only. Implement the optimized method's idea with compatible I/O rather than reproducing its notebook. Split only into complete work items when token fit requires it, never every helper or partial function.

Attach exact relevant basic code; hardcore generation uses prose/context. Include imports and short placement comments with returned code. Reuse stable interfaces directly; request a summary only when a later step consumes it, at most 3 lines and a target of 80 tokens, shorter than its input. Shared settings have one owner and flow through context. Keep data/metric/alignment and pretraining constraints, while choosing suitable tunable values. Do not add standalone imports, audit, unchanged-helper, metric, export or reload sessions.

Each chat is independent. Target at most 1800 tokens with reserve within the platform limit input + output < 2000. Code output may exceed input; only summaries have the shorter-output requirement. The user's web chatbot cannot enforce a separate output cap, so do not claim prompt wording guarantees it. Token verification remains unverified without real counters. Preserve notebooks/reports during guide updates.

Use `scripts/migrate_prompt_guides.py <task_dir>` to preview formats 1-3 to 4, `--write` to save and `--refresh` to rebuild format 4. Validate with `scripts/validate_prompt_guides.py <task_dir>` (optional `--variant`). Run `tests/test_validate_prompt_guides.py` and `tests/test_focused_prompt_guides.py` for legacy compatibility and focused-guide regression checks.

## Validator regression checks

Run `python <skill_dir>/tests/test_validate_artifacts.py` from any writable workspace to exercise offline variant, training/pretraining and prediction-contract checks. Tests do not download models or datasets. Also run `python <skill_dir>/tests/test_validate_prompt_guides.py` for guide references, dependency transfer and token-budget arithmetic. Run `python <skill_dir>/tests/test_focused_prompt_guides.py` for format-4 grouping, short-summary and migration checks.

## Training and checkpoint style

Follow the [training style contract](../nlp-notebook-engineer/references/training_style.md). For optimized/basic, use inline epoch/batch loops for straightforward training and concise inference-only saves in the validation-improvement branch. train_epoch and save_checkpoint are not mandatory. Use helpers only for real reuse/readability; add resume state only when explicitly requested.

## Focused solution report

Any requested huong_giai_quyet.md follows the [five-part report contract](../nlp-solution-architect/references/report_contract.md). Focus on the optimized solution and its reasoning; three notebooks do not require three report chapters. Put detailed execution/check status in the handoff. Notebook-only requests still need no new report.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](../nlp-notebook-engineer/references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods.
