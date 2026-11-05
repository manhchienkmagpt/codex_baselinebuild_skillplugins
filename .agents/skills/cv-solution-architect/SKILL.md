---
name: cv-solution-architect
description: Analyze image/video computer vision tasks and write a learner-friendly Vietnamese report on problem understanding, data and approaches, optimized preprocessing, model choice and justified hyperparameters.
---

# CV Solution Architect

Write huong_giai_quyet.md in Vietnamese for a learner. Follow the [five-part report contract](references/report_contract.md). Architect-only invocation creates only the report; do not claim implementation or training that did not happen.

Apply the [output naming convention](../cv-baseline-workflow/references/output_naming.md): default `outputs/cv/<STT>_<Tên bài toán>/`, preserve explicit STT or reuse/allocate the task number even in a fresh chat. Explicit user paths take precedence.

## Required references

Read the [decision playbook](references/decision_playbook.md), [pretraining policy](../cv-baseline-workflow/references/pretraining_policy.md) and [fixed runtime settings](../cv-notebook-engineer/references/runtime_config.md). Spreadsheet input follows the [Excel contract](../cv-baseline-workflow/references/excel_input_contract.md). The [three-variant notebook contract](../cv-baseline-workflow/references/dual_notebook_contract.md) still applies when notebooks are requested, but does not impose three-way report chapters. Use relevant [fewshot sources](../cv-notebook-engineer/references/fewshot_guide.md) to understand methods when helpful, not as report-layout requirements.

## Method decisions

1. Inspect the requested task, real dataset/schema/version, labels, raw splits/grouping and official metric. Preserve user data scope and distinguish verified observations from assumptions.
2. Compare 2–4 feasible approaches, choose one optimized method within data/policy/compute constraints and explain why its preprocessing and architecture address the observed difficulties. For existing artifacts, read baseline.ipynb and match its actual implementation.
3. Explain task-specific fixed hyperparameters with values, roles, reasons and sensitivity. Proposed settings are not proven optimal; only actual comparable validation results justify a measured-best claim. Do not use held-out test for model/hyperparameter selection.
4. Optimized CV may use eligible feature backbones only; task heads/pipelines are newly implemented. Route OCR, captioning, VQA and other language-output tasks to the NLP skills. Use a 32 GB GPU as a design target with fixed editable settings and ordinary PyTorch precision; no AMP or hardware-adaptive controls. Sparse CPU methods need no invented neural training settings.
5. Produce exactly five main report sections under the report contract. Keep relevant source links and evidence. No required notebook map, three-variant table, upgrade checklist, hashes, execution log or historical appendix.
6. Keep actual execution/check status in the final handoff or concise notebook notes. Preserve existing notebooks and user inputs during report-only updates; remove only task-created temporary resources. Never invent scores, timings or GPU fit.

## Training and checkpoint style

When explaining a requested implementation, follow the [training style contract](../cv-notebook-engineer/references/training_style.md): straightforward training uses inline loops, inference checkpoints are concise and resume state is saved only when explicitly requested. This is not a required report chapter.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](../cv-notebook-engineer/references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods. The packaged visual samples are image-to-text references for style/components; coverage gaps and CV adaptation boundaries in the guide are binding.
