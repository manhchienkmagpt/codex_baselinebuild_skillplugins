# Three-variant notebook contract

Apply the [shared contract](../../cv-baseline-workflow/references/dual_notebook_contract.md) separately to optimized, basic and hardcore. Use natural Vietnamese Markdown/code with flexible cells, no six-cell rule or required English markers. Read the [fewshot guide](fewshot_guide.md) and closest sources first.

## Setup and data

Keep imports, paths, seed and fixed uppercase constants easy to edit under [runtime_config.md](runtime_config.md). No AMP or hardware-adaptive controls. Optional guarded Colab mounting is allowed only when useful. Declare actual variant/task/pretraining metadata; basic/hardcore are scratch. Use separate runtime directories.

Load actual data/schema with actionable errors. Preserve dataset/version, raw splits, sample membership, IDs, grouping and metric semantics. Fit learned preprocessing only on train after splitting. Preserve task masks, coordinates, label IDs, instance masks and frame/sequence IDs. Do not copy personal paths, leakage, empty/debug cells or sample scores.

## Optimized and basic

Optimized retains strong preprocessing/model choices. Basic retains its simpler scratch architecture with minimal image/video conversion and required target geometry; no BPE/SentencePiece, augmentation or advanced feature engineering. Image conversion is color/resize/tensor only, plus geometry/padding essential to the task. Remove unused image/character features. Keep target encoding, official evaluation normalization and correct learning.

Both have complete readable Dataset/collation, model, a readable training loop (inline by default), validation-only best checkpoint selection, full pipeline save/reload and final evaluation. Use a meaningful fixed training budget; no artificial undertraining or score gap. Reload all model/vocabulary/config/decoding state before held-out evaluation. Do not select thresholds/decoding/checkpoints on test.

## Hardcore

Follow the [two-part hardcore contract](../../cv-baseline-workflow/references/hardcore_contract.md) instead of the full training layout. Only basic raw loading and one scratch model with direct inference. Minimal conversion occurs inside that model, with no fitted features, separate preprocessing, training functions, optimizer, backward, checkpoints or TODO. Prefer test, otherwise declared validation/held-out; assert IDs/count/schema and disclose untrained weights.

## Delivery

Valid nbformat 4, Markdown/nonempty code cells only, clear outputs/counts, fresh-kernel execution and no dependencies on fewshot, removed helpers or other variants. Validate each notebook against the shared proposal and smoke-test variant-appropriate behavior when feasible. Distinguish static, fixture, real-data and full-training status. Follow the [cleanup rule](../../cv-baseline-workflow/references/dual_notebook_contract.md), retaining only requested primary deliverables and preserving user files.

## Training layout and minimal checkpoints

Use the [training style contract](training_style.md): inline loops by default for simple tasks; helpers only when useful; concise inference checkpoints saved directly on validation improvement. No mandatory train_epoch/save_checkpoint, no optimizer/scheduler/RNG serialization unless resume is explicitly requested. Fewshot helper boundaries and verbose save dictionaries are not requirements.
