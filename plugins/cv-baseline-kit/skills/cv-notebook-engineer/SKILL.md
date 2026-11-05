---
name: cv-notebook-engineer
description: Generate or repair Vietnamese computer vision notebooks for image/video classification, detection, segmentation, keypoints, restoration and tracking in optimized, basic and untrained hardcore variants.
---

# CV Notebook Engineer

Apply the [output naming convention](../cv-baseline-workflow/references/output_naming.md): default `outputs/cv/<STT>_<Tên bài toán>/`, preserve explicit STT or reuse/allocate the task number even in a fresh chat. Explicit user paths take precedence.


Generate the requested subset of baseline.ipynb, baseline_basic.ipynb and baseline_hardcore.ipynb. Use independent fresh-kernel Run All, Vietnamese natural sections, flexible cells and fixed constants. Notebook-only requests do not require a new report.

Optimized may use backbone pretraining only; new heads and task-specific decoding/association are self-built. Basic is scratch with minimal color/resize/tensor conversion and required target geometry; no augmentation or engineered features. Both have full correct training, validation-only checkpoint selection and complete reload before evaluation. Hardcore has only raw data loading and one scratch model with direct inference, no preprocessing helpers or training/checkpoints.

Read actual schema and relevant packaged sources before implementation. Preserve IDs/order, class mappings, boxes/masks/keypoints coordinates, image/clip/sequence membership and task-specific output schema. Do not truncate evaluation silently. Follow temporal order and sequence resets for tracking. Smoke-test forward/backward/update/reload/metrics for full pipelines; direct inference/unchanged parameters/ID schema for hardcore. Never claim unmeasured GPU fit or quality. Clear outputs/counts and clean only task-created intermediates.


## Required references

Read the [three-variant contract](../cv-baseline-workflow/references/dual_notebook_contract.md), [CV task guide](../cv-baseline-workflow/references/cv_tasks.md), [backbone policy](../cv-baseline-workflow/references/pretraining_policy.md), [hardcore contract](../cv-baseline-workflow/references/hardcore_contract.md), [runtime settings](../cv-notebook-engineer/references/runtime_config.md) and [fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Read the closest packaged source notebooks. Spreadsheet input follows the [Excel contract](../cv-baseline-workflow/references/excel_input_contract.md).

Support image classification/pairs, detection, semantic/instance segmentation, keypoints, restoration, video classification, action recognition and tracking. Route OCR/captioning/VQA and other language-output tasks to the NLP skills; do not duplicate those policies here. Plan for one 32 GB GPU, fixed editable constants and ordinary PyTorch precision; no AMP, VRAM probes, adaptive batches/precision or OOM controllers. Sources and helpers are packaged locally; runtime notebooks never depend on this skill's files.

## Training and checkpoint style

Follow the [training style contract](references/training_style.md). For optimized/basic, use inline epoch/batch loops for straightforward training and concise inference-only saves in the validation-improvement branch. train_epoch and save_checkpoint are not mandatory. Use helpers only for real reuse/readability; add resume state only when explicitly requested.

## Focused solution report

Any requested huong_giai_quyet.md follows the [five-part report contract](../cv-solution-architect/references/report_contract.md). Focus on the optimized solution and its reasoning; three notebooks do not require three report chapters. Put detailed execution/check status in the handoff. Notebook-only requests still need no new report.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods. The packaged visual samples are image-to-text references for style/components; coverage gaps and CV adaptation boundaries in the guide are binding.
