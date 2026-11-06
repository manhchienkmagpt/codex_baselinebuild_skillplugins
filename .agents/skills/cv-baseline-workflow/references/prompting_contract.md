# Manual prompt guides

## Delivery and scope

Full workflow delivery has six primary files: the three notebooks, huong_giai_quyet.md, baseline_basic_promting.md and baseline_hardcore_promting.md. Preserve the requested spelling `promting`. Notebook engineer produces the guide matching each selected basic/hardcore variant unless the user explicitly restricts output to notebook files. Optimized-only and report-only requests need no new guide. Preserve unrelated existing files and retain requested guides during cleanup.

Read the actual basic and optimized notebooks before writing a guide; when optimized is not supplied or requested, state the proposed optimized target from the task specification instead of claiming correspondence to an existing notebook. Preserve dataset/version, raw splits, IDs, labels, official metric, pretraining eligibility and fixed runtime constraints. Do not promise equal scores without comparable measurements. These guides describe learner upgrades; they do not add preprocessing/training to the inference-only hardcore notebook being delivered.

## Author the task-specific sequence

Write in Vietnamese. Basic upgrades selected code from baseline_basic.ipynb toward the method in baseline.ipynb; identify an exact cell (one-based index including Markdown cells), line range, expected edit and merge location. Select only code needed for that operation, not a whole long cell. Prefer compatible signatures and outputs. If an interface must change, explicitly order all affected producers/consumers and their migration before integration. For later steps, tell the learner to use the current edited code at that location, not overwrite changes with the original excerpt.

Hardcore constructs preprocessing, model components, training, checkpoint reload and evaluation toward the optimized method using natural-language prompts and dependency summaries only. Never require source code, a notebook upload, a prior chat URL or access to other sessions as input. Describe actual function signatures, tensor shapes/dtypes, masks/coordinates and state needed by the next component in the summary. Eligible optimized visual backbones remain governed by pretraining policy; task modules are newly implemented. Sparse methods use fit/selection instead of invented epochs/backpropagation.

Derive steps from actual module boundaries and dependencies, not a universal fixed number of steps. Split large classes into constructor, methods and assembly; split large preprocessing/training operations similarly. Every generated part must have a clear insertion point, imports/config requirements and a bounded output; no unfinished placeholder may remain after assembly. Include a final integration/check sequence preserving data membership, leakage prevention and official evaluation. Do not ask for a whole notebook or pipeline in one response.

Each ready-to-type prompt is one continuous paragraph of about 2–4 sentences, without headings, lists, Markdown emphasis, code fences or backticks. Use concise natural language and precise technical terms. Prefer fewer than 100 Gemini tokens for the prompt itself; necessary requirements take priority over that preference. Retain only goal, specified method, essential constraints and response format; do not repeat facts already in attached code or the supplied summary. Allow the chatbot to choose a method only when the task/target does not specify one. Put labels, instructions, source locations, budgets and dependency lists outside the prompt paragraph.

## Independent sessions and transfer

Target `gemini-3.8-flash` with thinking level `low`. Each step uses a fresh session with one request/response. Input is the prompt plus only relevant dependency summaries and, for basic code steps, the indicated current code excerpt. No step may assume another session's memory. Ask code-producing responses to include a short natural-language handoff containing exact interfaces, I/O shapes/types, config/state and invariants required downstream, within the same output allowance. Keep generated code locally and assemble it manually; do not resend accumulated code for hardcore.

Supply a separate short summary prompt for merging multiple dependency summaries when needed. It follows the same paragraph and budget rules, accepts natural-language summaries and returns only the combined interface contract, never reconstructed code. Carry forward only dependencies actually needed for the next step. Include instructions to compare a new summary with the generated code/API; do not use a lossy summary to guess missing critical signatures or constants. Correct such omissions in a separately budgeted session.

## Token accounting

The hard per-session limit is 2000 tokens, not the model's native context limit. Count prompt, source excerpt, transferred summaries, system instructions, generated code/text, handoff summary and thinking. Use Gemini countTokens for the exact input; configure the remaining output cap when the platform allows it. max_output_tokens includes thinking and visible response. Low thinking does not disable thinking or guarantee an exact response length. Do not substitute an OpenAI tokenizer or word/character counts as verified Gemini counts.

For every step state input allowance, combined thinking/output allowance and reserve, summing to at most 2000. Reserve covers uncounted overhead; count known system instructions as input. An illustrative allocation is 600 input + 1100 output/thinking + 300 reserve. Re-count the actual edited code/summaries before sending; if input exceeds allowance, split the operation and recalculate. Allocate both code and the handoff within output. If the response is truncated, keep only complete units and redo the unfinished unit in a fresh smaller step with its necessary summary; never blindly append a continuation without context. If the interface exposes no output cap/counting, label compliance unverified; a prompt request alone cannot enforce a hard token limit.

Set Token verification to unverified unless actual Gemini measurements are available. Never fabricate counts or claim a checked total from budget arithmetic alone. Verified input requires a matching Gemini countTokens result; end-to-end compliance requires actual usage including thinking. No API credentials, paid generation or network calls are required to author or run the offline validator. Keep measurement evidence in the guide when provided, without secrets.

Sources: [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [token counting](https://firebase.google.com/docs/ai-logic/count-tokens?hl=en), [thinking/output limits](https://ai.google.dev/gemini-api/docs/thinking).

## Guide structure and offline validation

Use this lightweight format so the packaged validator can inspect guides without executing notebooks. Header fields are `Variant: basic` or `Variant: hardcore`, `Model: gemini-3.8-flash`, `Thinking: low`, and `Token verification: unverified`. Put human instructions before the steps. Use Vietnamese prose for instructions; the stable field labels are for validation.

Each step is `## Bước N: <name>` and has these fields on separate lines:

- `ID: <unique identifier>` and `Depends: none` or comma-separated earlier step IDs.
- `Kind: code`, `Kind: summary` or `Kind: integration`.
- `Input: <exact text/summaries and basic code excerpt to send>`.
- `Source: baseline_basic.ipynb#cell=N; lines=A-B` for basic code/integration steps. Multiple Source lines are allowed. Use original source line coordinates and identify the current edited counterpart in Input/Merge. Omit Source for hardcore and summary-only steps.
- `Target: baseline.ipynb#cell=N; lines=A-B` when the target notebook exists; multiple target references are allowed. Target references are authoring/learner inspection references, never code attachments required for hardcore.
- `Merge: <where/how to insert the complete returned unit, or where to pass a summary>`.
- `Budget: input=600; output=1100; reserve=300` (choose a feasible per-step allocation).
- `Handoff: <precise downstream interface/config/invariants to retain, including external dependency signatures>`.
- `### Prompt` followed by the single plain paragraph. No other content in that subsection; the next step starts a new level-two heading.

Run `python <workflow_skill>/scripts/validate_prompt_guides.py <task_dir>` for both guides, or add `--variant basic` / `--variant hardcore` for a selected variant. An explicitly notebook-only/report-only/optimized-only delivery does not invoke this guide validator. It checks structure, valid local notebook locations, dependency order, summary transfer and budget arithmetic, not scientific correctness or actual Gemini usage. Sentence/token brevity and semantic compatibility need manual review. If actual usage is supplied, use `Token verification: measured` and one header line `Measurement: {"step":"step-id","model":"gemini-3.8-flash","method":"Gemini countTokens + usageMetadata","input_tokens":500,"total_tokens":1500}` per step. These numbers are format examples, not measured values; total_tokens must include thinking. The offline check validates supplied evidence arithmetic, not its authenticity. Existing `validate_artifacts.py <notebook> <report>` remains compatible.
