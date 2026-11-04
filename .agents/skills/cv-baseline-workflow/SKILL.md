---
name: cv-baseline-workflow
description: Build optimized, minimal-preprocessing basic and inference-only hardcore computer vision notebooks plus a Vietnamese proposal for image/video tasks from descriptions or Excel task tables.
---

# CV Baseline Workflow


Resolve dataset/version, sample schema, labels, task metric and image/video/sequence split unit. Default outputs/cv/<task_slug>/ contains baseline.ipynb, baseline_basic.ipynb, baseline_hardcore.ipynb and huong_giai_quyet.md. Respect report-only/notebook-only/selected-variant requests.

Optimized uses strong task-specific preprocessing and a self-built task model, optionally with an eligible pretrained backbone. Basic uses only minimal task-correct image/video conversion and a simpler scratch model with full training/evaluation. Hardcore loads raw data, defines one untrained scratch model and directly infers, with mechanical conversion inside the model only. Preserve existing optimized content during companion updates.

Read the companion engineer/architect skills. Apply the shared contracts below, inspect actual data and closest fewshot sources, generate independent notebooks, then validate each against the proposal using scripts/validate_artifacts.py <notebook> <report>. Manually review geometry, split leakage, metrics and pretrained dataflow; static validation is insufficient. Smoke-test variant-appropriate behavior and report real execution status. Clean only task-created temporary assets; preserve user files.


## Required references

Read the [three-variant contract](../cv-baseline-workflow/references/dual_notebook_contract.md), [CV task guide](../cv-baseline-workflow/references/cv_tasks.md), [backbone policy](../cv-baseline-workflow/references/pretraining_policy.md), [hardcore contract](../cv-baseline-workflow/references/hardcore_contract.md), [runtime settings](../cv-notebook-engineer/references/runtime_config.md) and [fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Read the closest packaged source notebooks. Spreadsheet input follows the [Excel contract](../cv-baseline-workflow/references/excel_input_contract.md).

Support image classification/pairs, detection, semantic/instance segmentation, keypoints, restoration, video classification, action recognition and tracking. Route OCR/captioning/VQA and other language-output tasks to the NLP skills; do not duplicate those policies here. Plan for one 32 GB GPU, fixed editable constants and ordinary PyTorch precision; no AMP, VRAM probes, adaptive batches/precision or OOM controllers. Sources and helpers are packaged locally; runtime notebooks never depend on this skill's files.

## Validator regression checks

Run `python <skill_dir>/tests/test_validate_artifacts.py` from any writable workspace to exercise offline variant, training/pretraining and prediction-contract checks. Tests do not download models or datasets.

## Training and checkpoint style

Follow the [training style contract](../cv-notebook-engineer/references/training_style.md). For optimized/basic, use inline epoch/batch loops for straightforward training and concise inference-only saves in the validation-improvement branch. train_epoch and save_checkpoint are not mandatory. Use helpers only for real reuse/readability; add resume state only when explicitly requested.

## Focused solution report

Any requested huong_giai_quyet.md follows the [five-part report contract](../cv-solution-architect/references/report_contract.md). Focus on the optimized solution and its reasoning; three notebooks do not require three report chapters. Put detailed execution/check status in the handoff. Notebook-only requests still need no new report.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods. The packaged visual samples are image-to-text references for style/components; coverage gaps and CV adaptation boundaries in the guide are binding.
