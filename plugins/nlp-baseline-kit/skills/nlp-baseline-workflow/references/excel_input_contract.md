# Excel task input contract

Read this contract whenever the request supplies spreadsheet cells, a pasted table, or an Excel file. It applies to all three companion skills. The spreadsheet describes **tasks and dataset sources**; its rows are not training examples. Keep ordinary natural-language requests working too.

## Accepted input and selection

- A single pasted cell can contain a complete task description, a multiline `field: value` record, a dataset URL/identifier, or a table. Use available context and inspect the referenced source to infer missing fields. A lone value such as `Dễ` cannot identify a task: ask one focused question for the task or dataset rather than inventing it.
- Accept multiple pasted cells as Excel TSV, CSV, or a Markdown table (including the user's eleven-column format below). Preserve quoted cells containing tabs/newlines. A headerless row with exactly eleven values uses the column order below; twelve values are accepted when a positive STT precedes those eleven. With headers, STT may appear in any column. If a partial headerless selection cannot be mapped unambiguously, ask for column names or inspect the supplied workbook.
- For an attached/local `.xlsx` or `.xlsm`, read all relevant worksheets and task rows unless the user specifies sheets, rows, or A1 ranges. `Sheet1!B2` means the task represented by that row: inspect the row's headers and neighboring columns in the available workbook. Multiple selected cells on the same row represent one task. Do not silently process tasks outside selected rows. Reading neighboring columns is allowed to reconstruct the selected task, but preserve the original selection in provenance.
- A whole workbook means all nonempty task rows on every task sheet, not only the active sheet. Ignore cover/notes sheets without a recognizable task table; report them. Keep duplicate tasks as distinct records with their row provenance; do not silently overwrite them.
- Preserve visible text, actual hyperlink targets, multiline notes, and explicit merged-cell values. Do not forward-fill arbitrary blanks or merge independent task rows. Read cached formula values without executing formulas, macros, shell commands, or loader expressions from cells. If a required value has no formula cache, ask for its value or a recalculated workbook.
- For legacy `.xls`/`.xlsb`, use an available reader that supports the actual format, or ask for an `.xlsx` export if unavailable. Do not rename an unsupported file and claim it was read.

## Normalize columns

Match headers independent of case, Vietnamese accents, Markdown emphasis, whitespace, and line breaks. Keep the original values for provenance. The bundled reader understands these canonical columns and common English aliases:

| Excel column, in positional order | Normalized key | How it controls the workflow |
|---|---|---|
| Nhóm bài toán | `task_group` | Input/output modality and task family; image-to-text remains an image task. |
| Bài toán cụ thể | `task` | Concrete objective; infer only from unambiguous source/context if absent. |
| Dataset khuyến nghị | `dataset` | Requested dataset identity, not permission to replace it silently. |
| Mức ưu tiên nguồn | `source_priority` | Preferred acquisition source if multiple equivalent sources exist. |
| Thư viện / cách lấy | `loader` | Loading library or acquisition method; validate compatibility. |
| Link nguồn | `source_url` | Source to inspect, including actual hyperlink targets. |
| Lệnh tải / khởi tạo | `load_command` | A loading hint to inspect and translate into explicit code; never `eval`/`exec` a cell. |
| Split có sẵn | `splits` | Claimed splits; inspect actual names, sizes, schema and leakage groups. |
| Metric chính | `metric` | Metrics and optimize directions; preserve normalization and evaluation notes. |
| Lưu ý đánh giá & tiền xử lý | `notes` | Dataset scope, language, granularity, preprocessing and evaluation constraints. |
| Mức độ triển khai | `difficulty` | Planning context; it does not override score, validity, or the task-specific pretraining policy. |

Only a task/dataset/source sufficient to identify the work is needed to start. Infer other missing settings from inspected data and document assumptions. Explicit user instructions take precedence over table defaults; mention material conflicts. A `load_dataset("Teklia/IAM-line")` hint identifies a raw dataset, not a pretrained model.

Optional `STT`, `Số thứ tự`, `task number` or `sequence number` headers map to `task_number`. Preserve the supplied positive integer for output naming; do not replace it with the worksheet row index. Follow the [output naming convention](output_naming.md) when no task number is supplied.

## Reading helper

Use `<workflow_dir>/scripts/read_task_table.py`, resolving `<workflow_dir>` from this contract's parent skill directory. It uses Python's standard library and prints JSON; it needs no pandas/openpyxl installation. It handles `.xlsx`/`.xlsm`, UTF-8 TSV/CSV/Markdown, and multiline key-value cells. For connected spreadsheet tools, direct reading is also allowed if it preserves the same selection and provenance.

```text
python "<workflow_dir>/scripts/read_task_table.py" "tasks.xlsx"
python "<workflow_dir>/scripts/read_task_table.py" "tasks.xlsx" --sheet "Tasks" --range B2
python "<workflow_dir>/scripts/read_task_table.py" "tasks.xlsx" --sheet "Tasks" --range A2:K4 --range B7
python "<workflow_dir>/scripts/read_task_table.py" "pasted_cells.tsv"
python "<workflow_dir>/scripts/read_task_table.py" --text "Dataset khuyến nghị: Teklia/IAM-line"
```

The helper emits `tasks`, per-record `_source`, `_warnings`, `_output_slug`, `_task_number`, `_output_dirname`, and workbook-level `warnings`. `_output_slug` is an internal provenance identifier, not the final directory name. `_task_number` is the parsed explicit STT; `_output_dirname` suggests the numbered basename when STT and task name are present, otherwise it is null and the agent resolves numbering under the output naming convention. Resolve invalid-STT warnings before assigning an output directory. Review other warnings; parsing a row is not verification that the referenced dataset is accessible or correctly described. For a header on an unusual line, `--header-row N` is available. `--sheet` and `--range` may repeat; ranges apply to the selected sheets. The helper reports unsupported or unidentifiable input rather than inventing a schema. Free prose and isolated dataset URLs/identifiers can be interpreted by the agent directly; the structured helper is not mandatory for them.

## From records to deliverables

1. Normalize the supplied inputs, then briefly state the resolved task(s), dataset(s), primary metric/direction, selected sheet/rows, and any material assumptions in Vietnamese. Proceed without asking for confirmation when identifiable.
2. Inspect each referenced dataset/card/evaluation implementation. Verify the row's claimed split/schema/language and acquisition requirements. Keep source limitations in the report. A gated or unavailable dataset does not justify invented samples or reported scores; produce the configured runnable artifacts with clear access diagnostics as the existing contracts require.
3. If several metrics appear in one cell, follow an explicit ranking or official primary metric. Otherwise use the first listed metric as a **declared initial assumption**, report all listed metrics, and select checkpoints by that primary metric. Preserve `↓` as `min` and `↑` as `max`; infer known metric directions if absent. Never average CER and WER or tune on test.
4. One distinct task row gets one directory, normally `outputs/<STT>_<Tên bài toán>/`, under the [output naming convention](output_naming.md). Preserve supplied task numbers; otherwise reuse the same task/dataset's existing directory or allocate the next number in the output root. Add dataset or source sheet/row only to disambiguate a genuine collision. User output paths take precedence; multiple tasks still need separate subdirectories. Existing outputs must be inspected before revising them. Apply the [shared pretraining policy](pretraining_policy.md), fewshot-style flexible Markdown/code sections, fixed configuration, Vietnamese explanations, validation-only model selection, and honest execution reporting for **every** task. Identify the actual overall task before permitting pretrained subcomponents; do not infer permission merely from a Hugging Face dataset URL.
5. Apply the [dual-notebook contract](dual_notebook_contract.md) per record: `$nlp-baseline-workflow` creates optimized `baseline.ipynb`, practice `baseline_basic.ipynb`, inference-only `baseline_hardcore.ipynb` and shared `huong_giai_quyet.md`. `$nlp-notebook-engineer` creates/repairs the three notebooks; `$nlp-solution-architect` creates only the five-part report explaining the optimized solution. Honor explicit single-variant requests. Process all requested rows with their own four artifacts/separate runtime asset directories; report each task's paths and any blocked item. A workbook does not authorize automatically running two full training jobs per row; follow the requested scope and actual budget.
6. Persist the normalized source record and sheet/row provenance in the notebook Configuration and/or proposal assumptions, without requiring a new primary deliverable. Cross-check all generated settings against that record.
7. Before handing over each task, apply the dual-notebook contract's cleanup rule: retain only its requested primary notebooks/proposal, removing agent-created verification workspaces, intermediate exports, scripts/logs and runtime artifact folders after checks. Do not leave a task-reader JSON export or `_verification/` beside the final files. Preserve the supplied workbook, raw datasets and unrelated user files; the notebooks must not depend on deleted process files.

## User's OCR example

For `Image-to-Text | OCR (Optical Character Recognition) | IAM-line (Teklia) | Hugging Face | HF datasets | https://huggingface.co/datasets/Teklia/IAM-line | load_dataset("Teklia/IAM-line") | train / validation / test | CER ↓; WER ↓ | OCR chữ viết tay tiếng Anh ở mức dòng; không đại diện OCR tiếng Việt hoặc văn bản in. | Dễ`:

- Resolve **English handwritten line OCR**, source identifier `Teklia/IAM-line`, claimed train/validation/test, provisional primary CER minimized and supplementary WER minimized. Verify the actual source before asserting its schema or availability.
- The scope note is binding: do not describe this as Vietnamese OCR or printed-document OCR. Inspect image/transcription fields, aspect ratio, writer/document split leakage and transcription normalization.
- Compare a randomly initialized CNN + sequence encoder + CTC with an eligible pretrained image encoder + scratch recognizer using the decision playbook and shared policy. ImageNet backbones or HF image-only encoders are allowed; a complete pretrained TrOCR/OCR model is prohibited for this standalone OCR task. Keep CTC lengths, repeated-label feasibility, blank/PAD separation and train-only character vocabulary correct. A separate cell-OCR model can be permitted for a different overall task such as table extraction; that does not change this row into table extraction. Hosted OCR APIs remain disallowed.
- Implement the basic companion with a small custom CNN and simple scratch RNN/CTC recognizer when appropriate. Preserve raw splits, CER/WER normalization, alignment correctness and meaningful training. Explain the basic method's limitations and upgrades toward the optimized recognizer; do not invent a CER/WER gap. Each notebook uses its own checkpoint/tokenizer directory.
- Read the packaged table/OCR pipeline fewshot for organization, adapting it to standalone line OCR. Use natural configuration, data, image/target processing, Dataset, architecture, training/checkpoint and evaluation sections; no fixed cell count. Keep source settings/metric direction visible and validation-CER checkpointing separate from held-out CER/WER evaluation.

Minimal invocations:

```text
$nlp-baseline-workflow
<paste one cell or a selection of cells here>

$nlp-baseline-workflow tasks.xlsx

$nlp-baseline-workflow tasks.xlsx, sheet Tasks, cells A2:K4

$nlp-solution-architect tasks.xlsx, sheet Tasks, cell B2

$nlp-notebook-engineer
<paste the eleven-column Markdown table here>
```
