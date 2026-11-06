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
| Nhóm bài toán | `task_group` | Input/output modality and task family; route language-output image tasks to the NLP skills. |
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
4. One distinct task row gets one directory, normally `outputs/cv/<STT>_<Tên bài toán>/`, under the [output naming convention](output_naming.md). Preserve supplied task numbers; otherwise reuse the same task/dataset's existing directory or allocate the next number in the output root. Add dataset or source sheet/row only to disambiguate a genuine collision. User output paths take precedence; multiple tasks still need separate subdirectories. Existing outputs must be inspected before revising them. Apply the [shared pretraining policy](pretraining_policy.md), fewshot-style flexible Markdown/code sections, fixed configuration, Vietnamese explanations, validation-only model selection, and honest execution reporting for **every** task. Identify the actual overall task before permitting pretrained subcomponents; do not infer permission merely from a Hugging Face dataset URL.
5. Apply the [dual-notebook contract](dual_notebook_contract.md) per record: `$cv-baseline-workflow` creates optimized `baseline.ipynb`, practice `baseline_basic.ipynb`, inference-only `baseline_hardcore.ipynb`, shared `huong_giai_quyet.md` and the two matching `baseline_basic_promting.md` / `baseline_hardcore_promting.md` guides. `$cv-notebook-engineer` creates/repairs the selected notebooks and matching basic/hardcore prompt guides unless explicitly limited to notebook files; `$cv-solution-architect` creates only the five-part report explaining the optimized solution. Honor explicit single-variant requests. Process all requested rows with their own six artifacts/separate runtime asset directories; report each task's paths and any blocked item. A workbook does not authorize automatically running two full training jobs per row; follow the requested scope and actual budget.
6. Persist the normalized source record and sheet/row provenance in the notebook Configuration and/or proposal assumptions, without requiring a new primary deliverable. Cross-check all generated settings against that record.
7. Before handing over each task, apply the dual-notebook contract's cleanup rule: retain only its requested primary notebooks/proposal, removing agent-created verification workspaces, intermediate exports, scripts/logs and runtime artifact folders after checks. Do not leave a task-reader JSON export or `_verification/` beside the final files. Preserve the supplied workbook, raw datasets and unrelated user files; the notebooks must not depend on deleted process files.

## Domain routing

Rows about OCR, captioning, VQA or other language-output tasks use the NLP workflow. CV handles image/video classification, image pairs, detection, segmentation, keypoints, restoration, action recognition and tracking. Preserve dataset-specific subject/video/sequence splits and the official task metric; see [cv_tasks.md](cv_tasks.md).

Minimal invocations:

```text
$cv-baseline-workflow
<paste one cell or a selection of cells here>

$cv-baseline-workflow tasks.xlsx

$cv-baseline-workflow tasks.xlsx, sheet Tasks, cells A2:K4

$cv-solution-architect tasks.xlsx, sheet Tasks, cell B2

$cv-notebook-engineer
<paste the eleven-column Markdown table here>
```
