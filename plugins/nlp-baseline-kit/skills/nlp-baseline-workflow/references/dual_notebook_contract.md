# Three-variant baseline contract

This historical filename is retained for link compatibility. All three companion skills share this contract.

## Deliverables

| File | Preprocessing | Learning / purpose |
|---|---|---|
| baseline.ipynb | Optimized task-specific pipeline | Optimized complete method; preserve existing content unless explicitly asked to change it |
| baseline_basic.ipynb | Minimal task-correct conversion, simple tokenization/vocabulary/padding | Simpler scratch model with complete meaningful training and evaluation |
| baseline_hardcore.ipynb | No separate preprocessing pipeline; minimal raw-input conversion inside model only | One untrained scratch model, immediate inference; learner designs preprocessing/model/training |
| huong_giai_quyet.md | Shared Vietnamese report | Explain the optimized solution in five focused parts |
| baseline_basic_promting.md | Task-specific manual prompts with selected basic code | Upgrade basic toward the optimized method |
| baseline_hardcore_promting.md | Task-specific text-only prompts with interface summaries | Build components toward the optimized method across independent sessions |

Respect report-only and explicit single-variant requests. Notebook engineer includes the matching guide for each selected basic/hardcore variant unless the user explicitly requests only notebook files. Optimized-only/report-only requests need no new guide. Follow the [manual prompt guide contract](prompting_contract.md); do not create unrelated variants just to complete the list.

## Basic boundaries

Basic is simpler, not incorrect or deliberately undertrained. Use word/character tokenization and train-only vocabulary instead of BPE/SentencePiece; sparse models may use unigram counts instead of optimized TF-IDF/feature unions. Images use color/resize/tensor, with geometry/padding retained where task-required (e.g. CTC). No augmentation, advanced feature engineering or unused modalities/features. Keep masks, alignment, valid targets, official metric normalization and full training/checkpoint/reload. Preserve the simpler architecture already selected; adapt only its input dimensions when necessary. Never corrupt labels, remove most data or force a score gap.

## Protocol and independence

- Same real dataset/version, raw sample membership, IDs, seed, grouping, label semantics and evaluation protocol. Preserve official split ordering/IDs. Missing validation comes from train, not test. Fit new basic/optimized artifacts on train only.
- Hardcore follows the [two-part contract](hardcore_contract.md). Prefer test, otherwise explicitly named validation/held-out; an unlabeled test still permits inference. Do not infer on train silently or pass gold targets into the model.
- Each notebook runs from a fresh kernel and raw data independently; no loading another variant's notebook/checkpoint/tokenizer. Packaged fewshot/workbooks are authoring inputs only.
- Set metadata.nlp_baseline.variant to optimized/basic/hardcore; declare pretraining using the [policy](pretraining_policy.md). Basic/hardcore require scratch and no pretrained components. Keep existing metadata readers compatible.
- Runtime directories are separate (artifacts/<task>/{optimized,basic,hardcore}). Hardcore may write predictions, but has no training/checkpoint state.
- Use fixed settings and default PyTorch precision under the [runtime contract](../../nlp-notebook-engineer/references/runtime_config.md): 32 GB GPU design target, simple cuda/cpu choice, no AMP/VRAM probes/adaptive batching or OOM retries.

## Style, report and verification

Read the [fewshot guide](../../nlp-notebook-engineer/references/fewshot_guide.md) and relevant sources. Natural Vietnamese Markdown/code, flexible cell count, no required six-cell layout or English markers. Optimized/basic have complete data, preprocessing, model, training, checkpoint and evaluation sections; hardcore has only its two content sections.

Follow the [five-part report contract](../../nlp-solution-architect/references/report_contract.md): problem explanation, data/approaches, optimized preprocessing, model choice and justified hyperparameters. Do not require three-way tables, notebook maps, learner roadmaps or historical appendices. Mention a variant or experimental limit briefly only when relevant to method choice.

Run the validator separately on all selected notebooks. Review split equality, leakage, metric correctness and component eligibility manually. Smoke-test basic/optimized forward/backward, parameter updates, complete checkpoint reload and metrics where feasible. Hardcore tests fresh-kernel inference, unchanged parameters, IDs/order/count and prediction schema; no checkpoint/backward requirement.

## Clean handoff

Keep only the requested primary subset (six files for full workflow). Track and remove only agent-created temporary fixtures, scripts, logs/caches and runtime assets after validation; preserve user inputs, datasets and unrelated pre-existing files. Check resolved cleanup paths before recursive removal. Do not delete user checkpoints during notebook execution. Clear notebook outputs/counts and verify no dependency on removed assets. Summarize execution/check status in the final handoff or concise notebook notes; retain only methodological evidence in the five-part report.

## Training layout and minimal checkpoints

Use the [training style contract](../../nlp-notebook-engineer/references/training_style.md): inline loops by default for simple tasks; helpers only when useful; concise inference checkpoints saved directly on validation improvement. No mandatory train_epoch/save_checkpoint, no optimizer/scheduler/RNG serialization unless resume is explicitly requested. Fewshot helper boundaries and verbose save dictionaries are not requirements.
