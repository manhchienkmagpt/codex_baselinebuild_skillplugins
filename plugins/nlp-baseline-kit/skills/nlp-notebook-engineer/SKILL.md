---
name: nlp-notebook-engineer
description: Generate or repair Vietnamese NLP/image-to-text notebooks in optimized, minimal-preprocessing basic and untrained inference-only hardcore variants. Follow packaged fewshot, flexible cells and fixed settings for Jupyter/Colab and a 32 GB GPU target; include concise basic/hardcore manual prompt guides for independent Gemini sessions.
---

# NLP Notebook Engineer

Create baseline.ipynb, baseline_basic.ipynb and baseline_hardcore.ipynb, unless the request selects fewer variants. Every notebook runs independently from a fresh kernel. Notebook-only requests do not require a new proposal. Use Vietnamese headings, comments and logs.

Apply the [output naming convention](../nlp-baseline-workflow/references/output_naming.md): default `outputs/<STT>_<Tên bài toán>/`, preserve explicit STT or reuse/allocate the task number even in a fresh chat. Explicit user paths take precedence.

Read the [three-variant contract](../nlp-baseline-workflow/references/dual_notebook_contract.md), [notebook contract](references/notebook_contract.md), [runtime settings](references/runtime_config.md), [pretraining policy](../nlp-baseline-workflow/references/pretraining_policy.md) and [hardcore contract](../nlp-baseline-workflow/references/hardcore_contract.md). Read the [fewshot guide](references/fewshot_guide.md) and relevant packaged sources before authoring. Spreadsheet input uses the [Excel contract](../nlp-baseline-workflow/references/excel_input_contract.md).

## Variant behavior

- Optimized: retain strong task-specific preprocessing and modeling. Preserve existing optimized notebooks during companion-only updates.
- Basic: minimal word/character tokenization, train-only vocabulary, required padding/masks/alignment; no BPE/SentencePiece, feature unions, augmentation or learned advanced features. Images need only color conversion, resize and tensor conversion; keep aspect-ratio padding if required for OCR/CTC. Retain the existing simpler model and a complete meaningful training/evaluation pipeline. Remove unused image/character features when the model does not consume them. Do not remove task-required target encoding or official metric normalization.
- Hardcore: exactly two content sections, flexible cells: (1) basic data loading and inference split selection; (2) one scratch model definition/instantiation and immediate inference using that model. The model may internally convert raw inputs minimally; no separately defined preprocessing/tokenizer, learned vocabulary, training/loss/optimizer/backward, epoch loop, checkpoint dependency or unfinished TODO. Explicitly disclose random untrained weights.

## Correctness and delivery

Preserve dataset/version, raw sample membership, grouping, label semantics and official metric across variants. Fit basic/optimized artifacts on train after splitting; select checkpoints and thresholds on validation only. Text NLP is scratch-only; optimized image-to-text follows eligible proper-subtask policy. Basic/hardcore are scratch.

Keep fixed uppercase constants, default PyTorch precision and simple cuda/cpu device selection. No VRAM/precision automation, AMP or OOM controllers. Preserve task masks, span offsets, CTC lengths and valid losses. Optimized/basic reload complete best pipelines; hardcore has no checkpoint requirement.

Validate syntax, metadata and policy per notebook; smoke-test actual data where feasible. Keep separate runtime paths. Clear notebook outputs/counts before handoff. Remove only this task's temporary files; keep requested primary notebooks/proposal and user inputs. Return paths, independent Run All instructions, actual check status and learner next steps; never claim unmeasured training quality.

## Manual prompt companions

Use responsibility-based format 4 under the [manual prompt guide contract](../nlp-baseline-workflow/references/prompting_contract.md). Basic focuses on preprocessing, model and hyperparameters, reusing the existing training pipeline. Hardcore adds data loading/splitting and loss/optimizer/training, while the delivered hardcore notebook itself remains inference-only. Implement the optimized method's idea with compatible I/O rather than reproducing its notebook. Split only into complete work items when token fit requires it, never every helper or partial function.

For existing-code edits, first analyze the current code, related callers, reference material and prerequisite contexts. Extract names/signatures to preserve and constraints needed for the proposed idea, then pass only that analysis, the short idea and current editable code to the edit chat. Learners do not type module names or assemble technical requirement checklists. Prefer exact relevant report excerpts; add optimized example code when the description is absent or insufficient. Keep current-code interfaces distinct from example-code names. New components consume the idea, reference material and existing context directly, without an extra analysis chat. Preserve data/metric/alignment and pretraining requirements in the attached sources. Do not require complete-code/import/placement-comment boilerplate.

Summarize only consumed results, keeping signatures, I/O (shape/mask where relevant), configuration, state, constraints and main logic for equivalent implementation or callers. No fixed line/token cap, output/input ratio or next-step placeholder. Each chat targets at most 1800 tokens including reserve within input + output < 2000. Extraction and summary defaults are input=1100, output=500, reserve=200; split complete components when needed rather than dropping required context. Budgets are allocations, not enforced output caps or measured usage. Preserve notebooks/reports during guide updates; do not add separate imports, audit, unchanged-helper, metric, export or reload sessions.

Use `scripts/migrate_prompt_guides.py <task_dir>` to preview formats 1-3 to 4, `--write` to save and `--refresh` to rebuild format 4. Validate with `scripts/validate_prompt_guides.py <task_dir>` (optional `--variant`). Run `tests/test_validate_prompt_guides.py` and `tests/test_focused_prompt_guides.py` for legacy compatibility and focused-guide regression checks.

## Training and checkpoint style

Follow the [training style contract](references/training_style.md). For optimized/basic, use inline epoch/batch loops for straightforward training and concise inference-only saves in the validation-improvement branch. train_epoch and save_checkpoint are not mandatory. Use helpers only for real reuse/readability; add resume state only when explicitly requested.

## Focused solution report

Any requested huong_giai_quyet.md follows the [five-part report contract](../nlp-solution-architect/references/report_contract.md). Focus on the optimized solution and its reasoning; three notebooks do not require three report chapters. Put detailed execution/check status in the handoff. Notebook-only requests still need no new report.

## Current curated fewshot

When using examples, select the closest task family and the requested optimized/basic/hardcore variant from the [current fewshot guide](references/fewshot_guide.md). Each group includes its own five-part report. Read only the relevant sources; examples are authoring references, not runtime dependencies or guaranteed best methods.
