# Manual prompt guides

## Purpose and scope

Deliver `baseline_basic_promting.md` with basic and `baseline_hardcore_promting.md` with hardcore unless explicitly restricted to notebook files. Keep the spelling `promting`. Full workflow delivery has three notebooks, the proposal and these two guides. Report-only/optimized-only requests need no new guides. Updating a guide does not modify notebooks/reports. Retain dataset/version, split/ID membership, labels, metrics, target method and pretraining eligibility; do not promise measured scores or memory fit.

Write concise Vietnamese guides showing only **which context is needed → which prompt to send → which code to attach**. Use format 3 for new guides. Explain fresh sessions and reusable summaries in one short opening paragraph. Do not add a long task overview, Gemini/token explanation, standalone imports/package-installation/global-configuration steps, a `Ghép kết quả` section, visible notebook line references or caller inventories. Keep machine references and budgets in HTML comments.

## Modules and copyable prompts

Do not create fingerprint/audit-digest generation or summary steps, named context slots, or prompts requesting such checks. Data steps retain the actual samples, IDs, labels, split protocol and duplicate/leakage checks. Hash/key helpers needed for grouping are still valid; do not mistake them for optional audit utilities. Existing basic Python attachments remain exact even if they contain fingerprint calls. This exception applies only to actual code under Code gửi kèm, not headings, prose, prompt fences or future-code paste slots. Refreshing must remove obsolete dependencies, renumber steps and retain all remaining source inputs.

Select meaningful task functions, methods and processing blocks. Each generation prompt is one `text` fence containing the goal, necessary API/constraints and response requirements. Request code with its required imports, without prerequisite import-summary sessions. Explain behavior in natural language; do not enumerate every AST call, temporary variable, branch or baseline constant. For a method or multi-part callable, retain only a short scope/order instruction inside the prompt; never run an incomplete class or block.

A standalone module goes directly to its prompt. A dependent module has one `### Context cần chuẩn bị` section listing links to the required summary prompts, in order. Its main prompt includes named slots such as `Context tokenizer — Tokenizer:` and `[Dán tóm tắt tokenizer]`. Reuse checked summaries while code/API remains unchanged. Do not carry unrelated context/history or links to removed infrastructure steps.

Put actual basic code input in a separate `python` fence under `### Code gửi kèm`, telling learners to substitute current code after edits. Keep Source references hidden. A missing basic counterpart explicitly says `Module bổ sung mới`. If an existing basic module already fulfills the intended behavior, provide only `### Prompt tóm tắt module` with that real code, not an unnecessary generation step.

Only add `### Tóm tắt context để dùng tiếp` when another real step consumes the result. Give a copyable summary prompt and a Python slot for code just generated/edited. Keep signatures, I/O, shape/dtype/device, selected parameters and necessary state; return prose, flag missing facts and do not reconstruct code. Include already checked prerequisite contexts only when needed to interpret external interfaces. In summary-only steps, place the context anchor before the main summary prompt; its response is already the context, with no second summary session.

Hardcore generation uses prose and dependency summaries only. Hardcore summary sessions may attach generated code. Keep code inputs outside text fences. Class-interface merging may use `### Prompt gộp context` with method summaries. Finish with a small integration/check prompt covering I/O, split/IDs, metric, appropriate sparse fit/predict or neural forward/loss/update, and consistent reload; do not ask to regenerate the notebook.

## Module-owned configuration

The chatbot **chooses suitable tunable parameters**, explains them briefly and generates configuration with the owning module. Do not copy baseline hyperparameter values as requirements. This policy applies to manual prompt guides; the existing notebook runtime contract is unchanged.

- Data-source/version and split constraints belong in data loading. Tokenizer/image settings belong in preprocessing. Architecture dimensions/dropout belong with model construction. Batch settings belong with DataLoader. Optimizer, learning rate, epochs and patience belong with training.
- Declare a shared setting once at its first meaningful use; transfer its chosen value through the module's context. Resolve definitions using actual dependency order, not notebook text order, to avoid cycles. Derived model config uses the vocabulary/label sizes already established upstream.
- Consumers reuse the selected configuration. Loss, decoder, training and reload must agree with it; use symbolic/derived dimensions instead of stale baseline widths/defaults. Save or carry model configuration and tokenizer state for reload. Treat model/batch argument types as interface dependencies even when their class names do not occur in the function body.
- Keep structural invariants: dataset/label catalogs, special IDs mandated by the task, official metrics, CTC blank/stride alignment, masks and authorized pretraining policy. Do not turn tunable defaults into such invariants. An architecture target stays the same approach while its tunable settings may change.
- Preserve ordinary precision and the 32 GB GPU design target for neural modules; no AMP, VRAM probing or automatic batch changes. Choose explicit settings once, not a runtime adaptation system.

## Hidden metadata and offline checks

The header `<!-- prompt-guide` contains `Format: 3`, `Variant: basic` or `hardcore`, `Model: gemini-3.8-flash`, `Thinking: low`, `Token verification: unverified` and `Context merge budget: input=1200; output=500; reserve=300`, followed by `-->`.

Use consecutive `## Bước N: <meaningful name>` headings. Each `<!-- module` comment retains `ID`, `Depends` (unique earlier IDs or `none`), `Kind` (`code`, `summary`, `integration`), `Budget` and `Summary` (`yes` only for consumed results). Consumed code steps also have `Summary budget`. `Target` points to the real optimized unit; basic Source/Python excerpts exactly match the initial local code. New modules can omit Source. `Summary input: code` identifies basic summary-only source steps. References use one-based cells including Markdown and local line ranges, but remain hidden.

Record relocated settings in `Config owns` and `Config uses` as comma-separated names. Each shared setting has one owner; users of it depend directly or transitively on that owner's summary. Do not create a configuration-only module to satisfy these fields. Context anchors remain `<a id="context-ID"></a>`; every consumed result has a valid summary prompt.

Internally preserve independent fresh Gemini sessions and the 2000-token total allowance including system instructions, prompt, source, summaries, output and thinking. Generation defaults: input 800 + output/thinking 1000 + reserve 200. Summary/merge: input 1200 + output/thinking 500 + reserve 300. Adjust/split meaningful units before sending; use actual Gemini counting/caps when available. Offline word/character checks and arithmetic are not measured token evidence. Do not display the budget tutorial in the guide and do not call Gemini to author/validate it.

Run `python <workflow_skill>/scripts/validate_prompt_guides.py <task_dir>` or select `--variant basic` / `--variant hardcore`. Check context links/slots, ordering, real source excerpts/references, independent budgets, one configuration owner and inheritance, response imports and final coverage. Ignore apparent headings/metadata inside fences. Formats 1 and 2 remain readable with their original checks; format 3 does not require the removed headings. Scientific correctness and semantically compatible code still need manual review.

Keep usage unverified unless real Gemini evidence exists for every generation and separate summary session. Measured header `Measurement` JSON uses module IDs or `ID:summary`, model, method `Gemini countTokens + usageMetadata`, input and total tokens including thinking; common merge uses `context-merge`. Offline validation checks arithmetic, not authenticity. No credentials/network/paid calls are needed.

For maintenance, `python <workflow_skill>/scripts/migrate_prompt_guides.py <task_dir>` previews formats 1/2 → 3 without writing; `--write` applies validated results. Add `--refresh` to apply current presentation policy to format-3 guides while preserving attached code; repeat refreshes leave compliant files unchanged. Editing prompts invalidates earlier token measurements. The converter inspects local AST for boundaries, dependencies and config ownership, but renders behavioral prompts rather than AST dumps. Review domain semantics after migration.
