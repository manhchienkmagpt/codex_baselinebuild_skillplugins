# Focused manual prompt guides (format 4)

## Goal and work groups

Help learners implement the optimized method's idea from basic/hardcore with compatible code. Correct data, method and runnable interfaces matter; identical notebook structure, helper names and tuning values do not.

Deliver the selected `baseline_basic_promting.md` / `baseline_hardcore_promting.md` unless explicitly restricted to notebooks. Keep the spelling `promting`. Do not change notebooks or reports when refreshing guides. The delivered hardcore notebook remains inference-only; its guide teaches adding training.

Basic has only preprocessing, model and hyperparameter groups. Reuse existing loaders, splits, metrics, training and save/reload code; include only necessary call-site adjustments within the relevant group. Hardcore has data loading/splitting, preprocessing, model, hyperparameters and loss/optimizer/training groups, in that order. Traditional estimators use fit/predict instead of artificial neural training code.

Choose complete work items within each group. Split only when the expected input and code response need a smaller task: for example, representation versus batching, or a complete CRF/encoder versus the model that consumes it. Never slice a function body or replay every helper in the optimized notebook. Do not add separate imports, audit/fingerprint, unchanged-helper, metric, export, reload or final-integration sessions. Incorporate indispensable compatibility changes and small checks into the work that needs them.

## Prompt and context

Write Vietnamese prompts that specify the change, the target method, compatible I/O, necessary current code/context and the requested result. Use the report and optimized implementation to establish the idea; use actual basic interfaces and exact source excerpts to identify replacement code. Do not dump AST calls or all baseline constants. Keep source locations and machine metadata hidden; tell the learner where returned code replaces or extends the current notebook.

Basic attaches only relevant current code in separate Python fences under the source attachment heading. Source references must resolve exactly to the initial basic notebook. After edits, substitute current code. A new component needs no old-code attachment. Hardcore generation takes prose and checked interface summaries only; summary sessions may attach code just generated. Request complete code with required imports and short placement comments, not a rewritten notebook or long explanation.

Keep data/version, raw split membership, IDs, labels, metric and direction, alignment, masks, special IDs and eligible pretrained components. Choose suitable tunable values, rather than reproducing baseline defaults. Each shared setting has one owner; consumers inherit it through context. Preserve full preprocessing/model configuration for existing or added reload behavior. Neural work uses ordinary precision and a 32 GB GPU design target, without AMP or adaptive memory/batch controllers.

Each chat is stateless. Inline stable, short interfaces instead of asking for summaries of unchanged helpers. Link only summaries that a later step really consumes, with an anchor and matching named paste slot. A summary prompt requests at most 3 lines, a target of no more than 80 tokens, and a response shorter than its input: only callable signatures, I/O, shared configuration and state references. No explanations, code repetition or guessed values. Summaries carry interfaces/state references, not entire vocabularies or tensors. Summarize only the needed code; if it cannot fit, split by complete components.

## Token policy and hidden metadata

The user's chatbot uses an Antigravity CLI backend, with independent chats and a hard platform limit of input + output below 2000 tokens. It exposes no separate output cap. Design every generation and summary session for a total allowance of at most 1800, including a reserve of 200 for overhead/uncertainty. Account for actual system instructions, code, contexts and any thinking charged by the backend. Do not call Gemini or the backend to author/validate guides.

Code output may be larger than input. Per-step input/output allowances describe a planning allocation, not an enforced code-output cap. Default prose-only work allocates input=650, output=950, reserve=200; basic source work allocates input=1000, output=600, reserve=200. Adjust or split complete responsibilities when the actual input and expected response do not fit. Summary allowances use input=1400, output=80, reserve=200; only summaries require output smaller than input. Prompt wording is a soft request and cannot guarantee tokens. Never pad inputs just to satisfy a ratio.

The hidden `prompt-guide` header has Format: 4, Variant, the existing target model/thinking profile, Token verification, Session target: 1800 and Session limit: 2000. Use consecutive numbered step headings. Each hidden `module` comment has ID, Group, Kind: code, Depends, Budget and Summary. Group is data, preprocessing, model, hyperparameters or training, as applicable above. IDs/dependencies must be unique and refer backwards. Include Summary budget only for consumed results; Source/Target references identify actual local excerpts. Config owns/uses record one owner and dependency inheritance. Keep a short visible opening about fresh chats, the safe target and unverified token fit; do not repeat budget instructions in every prompt.

Keep Token verification: unverified without real target-model counters for every generation and summary. Measured evidence retains the existing Measurement JSON with step, model, method, input_tokens and total_tokens; method is `Gemini countTokens + usageMetadata` or `chatbot token counters`. It must fit the declared allowances, and summary output must also be smaller than its actual input. Offline word/character heuristics are warnings, never token proof. Editing prompts invalidates prior evidence.

## Maintenance and validation

`python <workflow_skill>/scripts/migrate_prompt_guides.py <task_dir> [...]` previews formats 1-3 to 4 without writing. `--write` saves validated guides; `--refresh` rebuilds existing format-4 guides from current notebooks/report. Repeat refreshes are idempotent. Missing unselected companions are skipped. `focused_prompts.py` is the current generator; `prompt_modules.py` remains available for legacy format-3 support.

Run `python <workflow_skill>/scripts/validate_prompt_guides.py <task_dir>` with optional `--variant basic` or `--variant hardcore`. Formats 1-3 remain readable under their original rules. Format 4 checks group coverage/order, source excerpts, context links/slots, unique configuration ownership, summary consumption and the new budget policy. The validator ignores headings/metadata inside fences. Scientific correctness, response length and successful generated-code execution require separate evidence.

Run both `tests/test_validate_prompt_guides.py` and `tests/test_focused_prompt_guides.py`. Review representative traditional, neural, sequence and image-to-text guides; verify migration idempotence and that notebooks/reports remain unchanged.
