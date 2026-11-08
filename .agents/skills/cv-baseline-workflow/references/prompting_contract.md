# Source-guided manual prompt guides (format 4)

## Goal and work groups

Help learners implement the optimized method's idea with compatible code. Learners supply a basic idea and existing code/reference material; the chatbot derives the needed interfaces and constraints. Do not make learners remember or assemble technical checklists.

Deliver the selected `baseline_basic_promting.md` / `baseline_hardcore_promting.md`, preserving the spelling `promting`. Keep notebooks and reports unchanged during guide refreshes. The delivered hardcore notebook remains inference-only; its guide teaches adding training.

Basic groups are preprocessing, model and hyperparameters, reusing the existing training pipeline. Hardcore groups are data loading/splitting, preprocessing, model, hyperparameters and loss/optimizer/training. Traditional estimators use fit/predict. Split complete work items only when necessary for the session limit, never partial function bodies. Do not add separate imports, audit/fingerprint, unchanged-helper, metric, export, reload or final-integration sessions.

## Ideas, analysis and editing

Fill a short, ordinary-language idea into each prompt. Do not reproduce the optimized implementation's entire configuration or task rules inside the idea. Existing names/signatures come from the current code; reference code describes the target method, not replacement names.

For existing-code edits, separate `Code cần sửa`, `Code liên quan` and `Nguồn tham khảo`. Current and related code excerpts resolve exactly to the initial basic notebook; learners substitute current versions after edits. Related callers/providers are reference inputs outside the edit scope. Each chat is independent.

The analysis chat receives the idea, current/related code, reference material and prerequisite contexts:

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
[Ý tưởng cơ bản.]

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

The edit chat receives the same idea, current editable code, analysis result and prerequisite contexts. Reattach reference material only for unresolved details:

```text
Sửa phần “Code cần sửa” theo ý tưởng:
[Ý tưởng cơ bản.]

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.
```

Do not add a `Yêu cầu bài toán` checklist for manual pasting, repeat every interface condition, or force complete-code/import/comment boilerplate. Existing bugs are not requirements to preserve. Names, arguments and I/O semantics must remain compatible unless a necessary change is identified first. Shared configuration has one owner and flows through checked context.

New components use the short idea, reference material and checked context directly, without an extra analysis session. Hardcore may receive example code as reference material when report descriptions are insufficient; this is distinct from attaching existing editable basic code. Preserve actual hardcore interfaces through prose context.

## Reference material

Prefer exact relevant narrative paragraphs, inline in the guide: data uses sections 1–2, preprocessing section 3, model section 4, hyperparameters section 5 and training the relevant method/configuration paragraphs in sections 4–5. Avoid generic definitions, unrelated alternatives, execution-history notes and long comparison/configuration tables. Hyperparameter references supplement prose with actual tunable declarations, inheriting architecture settings from context. Keep complete paragraphs, including necessary constraints at their ends. If a task needs a table as evidence, attach complete relevant units rather than a truncated row.

Paths and line references are provenance, not substitutes for attached contents: an independent web chatbot cannot read local files from a link. Use Markdown fences for report excerpts, extending fence length when the excerpt contains backticks. Add complete corresponding optimized code when report descriptions are missing or too brief; review semantic sufficiency separately from static checks. If method information is still missing, disclose that and request unclear points be marked rather than invented.

Attach actual optimized declarations carrying dataset/version and split settings at the first data/preprocessing work item, then inherit them through context. Include labels/special IDs, metric/direction and pretraining policy where the work consumes them. Do not repeat the entire global configuration at every step or turn it into a checklist the learner must retype. Task policy remains binding: keep raw membership, IDs, gold/metric semantics, train-only fitting, alignment/masks and eligible pretrained components. Hyperparameter values in references are suggestions, not exact-reproduction requirements. Preserve configuration and representation state needed by reload/callers. Use fixed editable settings, ordinary precision and a 32 GB GPU design target; no AMP or adaptive memory/batch controllers.

## Summaries and session budgets

Inline short stable interfaces; summarize only results consumed by later steps, using matching context anchors and named paste slots. Keep this prompt unchanged:

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

Do not impose a fixed line/token cap, output/input ratio or next-step placeholder. Reference saved vocabulary/tensor state rather than dumping its contents.

The backend's hard limit remains input + output < 2000, with a safe session target of 1800 including reserve. Prose/new-code sessions default to input=650, output=950, reserve=200; existing-code generation uses input=1000, output=600, reserve=200; analysis and summary each use input=1100, output=500, reserve=200. Count all attachments and pasted contexts. Adjust allocations or split complete work items when needed instead of dropping required information. These are allocations, not enforced response caps. Do not call Gemini/backend services to author or validate guides.

## Metadata and validation

Keep Format: 4 and existing model/thinking settings. New headers include Constraint workflow: extract-before-edit and Reference workflow: report-first. Work items retain ID, Group, Kind: code, Depends, Budget, Summary, configuration ownership and Source/Related source/Target references. Set Constraints: yes with its own budget exactly for edits of existing code; new components use Constraints: no. Analysis and summary sessions use `ID:constraints` and `ID:summary` for token evidence.

Each work item has Reference report fields (`huong_giai_quyet.md#lines=start-stop`), Reference code fields using the existing optimized-notebook reference syntax, and Reference missing: yes/no. Report/Python fences under `Nguồn tham khảo` must exactly match those references. Include a `references-ID` anchor consumed by analysis or new-module generation, plus the existing `constraints-ID` analysis-to-edit handoff and `context-ID` dependency handoffs.

Keep Token verification: unverified without actual model counters. Measured evidence uses the existing Measurement JSON and must cover all generation/analysis/summary sessions within their allocations, without a response/input ratio rule. Offline character checks are warnings, not token measurements; editing a prompt invalidates previous evidence.

Use `migrate_prompt_guides.py <task_dir>` to preview, `--write` to save, and `--refresh` to rebuild format 4. Refreshes must be idempotent. Missing unselected companions are skipped. Older formats and earlier format-4 workflows remain readable under their historical structural rules.

Run `validate_prompt_guides.py <task_dir>` with optional `--variant`, both prompt-guide test suites and skill validation. Check report/code provenance, analysis handoff, missing references, context links, grouping and budgets. Review representative traditional, sequence and image-to-text examples. Verify notebook/report hashes remain unchanged. Static checks do not prove semantic completeness, actual token fit or successful execution of generated code.
