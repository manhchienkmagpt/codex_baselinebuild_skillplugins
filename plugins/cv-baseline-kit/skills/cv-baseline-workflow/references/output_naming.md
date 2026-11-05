# Numbered task output directories

Apply this convention to workflow, notebook-only and report-only requests, including a fresh chat with no naming instructions. Explicit user output paths or naming instructions take precedence.

- Default CV destination: `outputs/cv/<STT>_<Tên bài toán>/`, for example `outputs/cv/1_Image Classification/`.
- Use the concrete task name, preserving readable spaces, case, accents and parentheses. Do not use the task group or dataset as the primary name. Replace Windows-invalid characters (`< > : " / \ | ? *` and control characters) with ` - `, collapse repeated whitespace, and trim trailing spaces/dots. Example: `2_Object Detection - Localization`.
- Use the task number explicitly supplied by the user or a table's STT/sequence-number column. Preserve gaps and the supplied numbering; an Excel row number or a selected-row position is not the task number. Write ordinary positive integers without added zero padding.
- Without an explicit number, first inspect existing outputs for the same concrete task and dataset, using notebook metadata/report content when needed. Reuse the matching directory and number for updates, including a pre-existing unnumbered directory; do not create a duplicate just because its name differs from the convention.
- For a new task with no number or matching output, inspect direct child directories of the chosen output root and use one greater than the highest leading integer followed by `_`; start at 1 if none exist. Assign successive numbers in input order for a batch. NLP and CV have separate default output roots and numbering sequences.
- Resolve a genuine destination collision before writing. Preserve unrelated files; if the same number/task is supplied for distinct datasets, append the dataset name (or source sheet/row if still ambiguous) to the readable task portion. Do not renumber or rename existing outputs unless requested.
- Keep machine identifiers such as `TASK`, dataset IDs, vocabulary IDs and data/cache/artifact slugs independent of the display folder number. Do not rewrite existing runtime paths just to mirror STT. Quote paths containing spaces in shell commands.

The table reader's `_output_slug` remains an internal provenance identifier, not the final directory name. `_task_number` is the parsed explicit STT; `_output_dirname` is a suggested numbered directory basename when the table supplies both STT and a concrete task name. If either is missing, resolve it using the rules above rather than using a machine slug as the output directory.
