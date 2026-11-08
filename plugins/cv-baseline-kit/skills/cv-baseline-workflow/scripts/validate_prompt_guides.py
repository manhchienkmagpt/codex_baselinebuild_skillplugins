#!/usr/bin/env python3
"""Offline checks for manual prompt guides; no API calls or tokenizer estimates as proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


STEP = re.compile(r"^## Bước (\d+):[^\n]+$", re.MULTILINE)
REFERENCE = re.compile(r"(baseline(?:_basic)?\.ipynb)#cell=(\d+); lines=(\d+)-(\d+)")
REQUIRED = ("ID", "Depends", "Kind", "Input", "Merge", "Budget", "Handoff")


def field_values(text: str, key: str) -> list[str]:
    return re.findall(r"^" + re.escape(key) + r":[ \t]*(.*)$", text, re.MULTILINE)


def single_field(text: str, key: str, errors: list[str], label: str) -> str:
    values = field_values(text, key)
    if len(values) != 1 or not values[0].strip():
        errors.append(f"{label}: require one nonempty {key} field")
        return ""
    return values[0].strip()


def scan_reference(value: str, directory: Path, expected: str,
                   errors: list[str], label: str) -> str:
    match = REFERENCE.fullmatch(value)
    if not match or match[1] != expected:
        errors.append(f"{label}: invalid notebook reference {value!r}")
        return ""
    try:
        notebook = json.loads((directory / match[1]).read_text(encoding="utf-8"))
        cell = notebook["cells"][int(match[2]) - 1]
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        lines = source.splitlines()
        start, stop = int(match[3]), int(match[4])
        if int(match[2]) < 1 or cell["cell_type"] != "code" or not 1 <= start <= stop <= len(lines):
            raise ValueError("cell must be code and line bounds must exist")
        return "\n".join(lines[start - 1:stop])
    except (OSError, UnicodeError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        errors.append(f"{label}: cannot resolve reference: {exc}")
        return ""


def scan_report_reference(value: str, directory: Path, errors: list[str], label: str) -> str:
    match = re.fullmatch(r"huong_giai_quyet\.md#lines=(\d+)-(\d+)", value)
    if not match:
        errors.append(f"{label}: invalid report reference {value!r}")
        return ""
    try:
        lines = (directory / "huong_giai_quyet.md").read_text(encoding="utf-8").splitlines()
        start, stop = map(int, match.groups())
        if not 1 <= start <= stop <= len(lines):
            raise ValueError("report line bounds must exist")
        return "\n".join(lines[start - 1:stop])
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{label}: cannot resolve report reference: {exc}")
        return ""


def check_prompt(prompt: str, errors: list[str], warnings: list[str], label: str) -> None:
    if not prompt:
        errors.append(f"{label}: empty prompt")
        return
    if "\n" in prompt:
        errors.append(f"{label}: prompt must be a single continuous paragraph on one line")
    if (re.search(r"^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>)", prompt)
            or re.search(r"`|\*\*|__|\[[^\]]+\]\(|<[^>]+>", prompt)
            or re.search(r"(?<!\w)[*_][^*_]+[*_](?!\w)", prompt)):
        errors.append(f"{label}: prompt contains Markdown or list formatting")
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", prompt) if part.strip()]
    if not 2 <= len(sentences) <= 4 or not re.search(r"[.!?]$", prompt):
        warnings.append(f"{label}: manually review the 2–4 sentence requirement (technical punctuation may vary)")
    if len(prompt.split()) > 100:
        warnings.append(f"{label}: long prompt; prefer fewer than 100 Gemini tokens (not measured offline)")


def scan_legacy_guide(path: Path, variant: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read guide: {exc}"], warnings
    if not content.strip():
        return ["Guide is empty"], warnings
    steps = list(STEP.finditer(content))
    header = content[:steps[0].start()] if steps else content
    for key, expected in (("Variant", variant), ("Model", "gemini-3.8-flash"), ("Thinking", "low")):
        if single_field(header, key, errors, "Guide") != expected:
            errors.append(f"Guide: {key} must be {expected}")
    verification = single_field(header, "Token verification", errors, "Guide")
    if verification not in ("unverified", "measured"):
        errors.append("Guide: Token verification must be unverified or measured")
    if verification == "unverified":
        warnings.append("Gemini token usage is unverified; budgets are allowances, not measured counts")
    if not steps:
        return errors + ["Guide has no numbered steps"], warnings
    known: dict[str, set[str]] = {}
    summary_ids: set[str] = set()
    budgets: dict[str, tuple[int, int]] = {}
    kinds: list[str] = []
    for index, heading in enumerate(steps):
        label = f"Step {index + 1}"
        if int(heading[1]) != index + 1:
            errors.append(f"{label}: numbering must be consecutive from 1")
        end = steps[index + 1].start() if index + 1 < len(steps) else len(content)
        block = content[heading.end():end]
        pieces = re.split(r"^### Prompt[ \t]*$", block, flags=re.MULTILINE)
        if len(pieces) != 2:
            errors.append(f"{label}: require one ### Prompt subsection")
            continue
        metadata, prompt = pieces[0], pieces[1].strip()
        values = {key: single_field(metadata, key, errors, label) for key in REQUIRED}
        identifier = values["ID"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) or identifier in known:
            errors.append(f"{label}: invalid or duplicate ID {identifier!r}")
        depends = [] if values["Depends"] == "none" else [d.strip() for d in values["Depends"].split(",")]
        if len(depends) != len(set(depends)) or any(dep not in known for dep in depends):
            errors.append(f"{label}: dependencies must be unique earlier step IDs")
        ancestors = set(depends)
        for dependency in depends:
            ancestors.update(known.get(dependency, set()))
        kind = values["Kind"]
        kinds.append(kind)
        if kind not in ("code", "summary", "integration"):
            errors.append(f"{label}: unsupported Kind {kind!r}")
        if kind == "summary":
            summary_ids.add(identifier)
            if not depends:
                errors.append(f"{label}: summary needs earlier dependency contracts")
        budget = re.fullmatch(r"input=(\d+); output=(\d+); reserve=(\d+)", values["Budget"])
        input_limit = 0
        if not budget:
            errors.append(f"{label}: malformed Budget")
        else:
            input_limit, output_limit, reserve = map(int, budget.groups())
            if input_limit <= 0 or output_limit <= 0 or sum((input_limit, output_limit, reserve)) > 2000:
                errors.append(f"{label}: budget must have positive input/output and total <= 2000")
            budgets[identifier] = (input_limit, output_limit)
        sources = field_values(metadata, "Source")
        if variant == "basic" and kind != "summary" and not sources:
            errors.append(f"{label}: basic code/integration requires a Source reference")
        if (variant == "hardcore" or kind == "summary") and sources:
            errors.append(f"{label}: hardcore/summary input cannot require source code")
        if variant == "hardcore" and re.search(r"\.ipynb#|```|def \w+\(|class \w+[:(]", values["Input"]):
            errors.append(f"{label}: hardcore Input must contain only prose/summary instructions")
        excerpts = [scan_reference(source, path.parent, "baseline_basic.ipynb", errors, label) for source in sources]
        for target in field_values(metadata, "Target"):
            scan_reference(target, path.parent, "baseline.ipynb", errors, label)
        # A warning only: bytes/words/characters are NOT Gemini token counts.
        attached = prompt + "\n".join(excerpts)
        if input_limit and len(attached) > input_limit * 3:
            warnings.append(f"{label}: large input excerpt; count with Gemini and split before sending if over allowance")
        if depends and not re.search(r"tóm tắt|summary", values["Input"], re.IGNORECASE):
            errors.append(f"{label}: fresh session Input must explicitly transfer dependency summaries")
        check_prompt(prompt, errors, warnings, label)
        if index == len(steps) - 1:
            if kind != "integration":
                errors.append("Final step must be integration")
            if not set(known).issubset(ancestors):
                errors.append("Final integration must transitively cover all earlier steps")
        known[identifier] = ancestors
    if not summary_ids:
        errors.append("Guide must include a separately budgeted summary prompt")
    if not any(kind == "code" for kind in kinds):
        errors.append("Guide must include code-producing steps")
    if verification == "measured":
        measured: set[str] = set()
        for line in field_values(header, "Measurement"):
            try:
                evidence = json.loads(line)
                identifier = evidence["step"]
                input_tokens = evidence["input_tokens"]
                total_tokens = evidence["total_tokens"]
                if evidence["model"] != "gemini-3.8-flash" or evidence["method"] != "Gemini countTokens + usageMetadata":
                    raise ValueError("evidence must use target Gemini model and counters")
                if identifier in measured or identifier not in budgets:
                    raise ValueError("unknown/duplicate step evidence")
                if type(input_tokens) is not int or type(total_tokens) is not int:
                    raise ValueError("measured counts must be integers")
                input_limit, output_limit = budgets[identifier]
                if not 0 < input_tokens <= input_limit or not input_tokens <= total_tokens <= 2000:
                    raise ValueError("measured input/total exceeds budget or is invalid")
                if total_tokens - input_tokens > output_limit:
                    raise ValueError("measured thinking/output exceeds allowance")
                measured.add(identifier)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid Measurement: {exc}")
        if measured != set(known):
            errors.append("Measured verification requires actual Gemini input/total evidence for every step")
        warnings.append("Offline validation checks supplied evidence arithmetic, not authenticity of API measurements")
    return errors, warnings


def outside_fences(content: str) -> str:
    """Mask fenced blocks without changing offsets, including headings in examples."""
    result = []
    fence = None
    for line in content.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence is None and match:
            fence = match[1]
            result.append(" " * len(line.rstrip("\r\n")) + line[len(line.rstrip("\r\n")):])
        elif fence is not None:
            result.append(" " * len(line.rstrip("\r\n")) + line[len(line.rstrip("\r\n")):])
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
        else:
            result.append(line)
    return "".join(result)


def fenced_blocks(content: str, language: str) -> list[str]:
    blocks = []
    fence = None
    capture = False
    lines = []
    for line in content.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is None and match:
            fence = match[1]
            capture = match[2].strip() == language
            lines = []
        elif fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                if capture:
                    blocks.append("\n".join(lines))
                fence = None
            else:
                lines.append(line)
    return blocks


def subsection(block: str, heading: str) -> str:
    masked = outside_fences(block)
    match = re.search(r"^### " + re.escape(heading) + r"[ \t]*$", masked, re.MULTILINE)
    if not match:
        return ""
    end = re.search(r"^### ", masked[match.end():], re.MULTILINE)
    return block[match.end():match.end() + end.start()] if end else block[match.end():]


FINGERPRINT_WORD = re.compile(r"fingerprint|dấu vân tay|vân tay dữ liệu", re.IGNORECASE)


def source_attachment_spans(content: str) -> list[tuple[int, int]]:
    """Exempt supplied code/reference fences, never generated prompt instructions."""
    masked = outside_fences(content)
    spans = []
    for heading in re.finditer(r"^### (Code gửi kèm|Code cần sửa|Code liên quan|Nguồn tham khảo)[ \t]*$", masked, re.MULTILINE):
        following = re.search(r"^#{2,3} ", masked[heading.end():], re.MULTILINE)
        stop = heading.end() + following.start() if following else len(content)
        position, start, fence, language = heading.end(), 0, None, ""
        for line in content[heading.end():stop].splitlines(keepends=True):
            match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
            if fence is None and match:
                start, fence, language = position, match[1], match[2].strip()
            elif fence is not None and match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                if language == "python" or (heading[1] == "Nguồn tham khảo" and language == "markdown"):
                    spans.append((start, position + len(line)))
                fence = None
            position += len(line)
    return spans


def instruction_text(content: str) -> str:
    for start, stop in reversed(source_attachment_spans(content)):
        content = content[:start] + content[stop:]
    return content


def checked_budget(value: str, errors: list[str], label: str) -> tuple[int, int]:
    match = re.fullmatch(r"input=(\d+); output=(\d+); reserve=(\d+)", value)
    if not match:
        errors.append(f"{label}: malformed Budget")
        return (0, 0)
    inp, out, reserve = map(int, match.groups())
    if min(inp, out) <= 0 or inp + out + reserve > 2000:
        errors.append(f"{label}: budget must have positive input/output and total <= 2000")
    return inp, out


def scan_modular_guide(path: Path, variant: str, content: str) -> tuple[list[str], list[str]]:
    if re.search(r"^Format: 4\s*$", outside_fences(content).split("## Bước ", 1)[0], re.M):
        return scan_focused_guide(path, variant, content)
    errors: list[str] = []
    warnings: list[str] = []
    masked = outside_fences(content)
    steps = list(STEP.finditer(masked))
    header = masked[:steps[0].start()] if steps else masked
    formats = field_values(header, "Format")
    compact = formats == ["3"]
    if compact and FINGERPRINT_WORD.search(instruction_text(content)):
        errors.append("Guide: fingerprint steps/contracts are excluded; only original basic source attachments may contain them")
    for key, expected in (("Format", "3" if compact else "2"), ("Variant", variant), ("Model", "gemini-3.8-flash"), ("Thinking", "low")):
        if single_field(header, key, errors, "Guide") != expected:
            errors.append(f"Guide: {key} must be {expected}")
    verification = single_field(header, "Token verification", errors, "Guide")
    if verification not in ("unverified", "measured"):
        errors.append("Guide: Token verification must be unverified or measured")
    if verification == "unverified":
        warnings.append("Gemini token usage is unverified; budgets are allowances, not measured counts")
    if not steps:
        return errors + ["Guide has no numbered steps"], warnings
    merge_budget = single_field(header, "Context merge budget", errors, "Guide")
    merge_limits = checked_budget(merge_budget, errors, "Guide context merge")
    known: dict[str, set[str]] = {}
    consumed: set[str] = set()
    summarized: set[str] = set()
    budgets: dict[str, tuple[int, int]] = {}
    config_owners: dict[str, str] = {}
    kinds = []
    for index, heading in enumerate(steps):
        label = f"Step {index + 1}"
        if int(heading[1]) != index + 1:
            errors.append(f"{label}: numbering must be consecutive from 1")
        end = steps[index + 1].start() if index + 1 < len(steps) else len(content)
        block = content[heading.end():end]
        metadata = re.findall(r"<!-- module\s*\n(.*?)-->", outside_fences(block), re.DOTALL)
        if len(metadata) != 1:
            errors.append(f"{label}: require one hidden module metadata block")
            continue
        metadata = metadata[0]
        values = {key: single_field(metadata, key, errors, label)
                  for key in ("ID", "Depends", "Kind", "Budget", "Summary")}
        identifier, kind = values["ID"], values["Kind"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) or identifier in known:
            errors.append(f"{label}: invalid or duplicate ID {identifier!r}")
        depends = [] if values["Depends"] == "none" else [part.strip() for part in values["Depends"].split(",")]
        if len(depends) != len(set(depends)) or any(dep not in known for dep in depends):
            errors.append(f"{label}: dependencies must be unique earlier step IDs")
        consumed.update(depends)
        ancestors = set(depends)
        for dep in depends:
            ancestors.update(known.get(dep, set()))
        known[identifier] = ancestors
        kinds.append(kind)
        if kind not in ("code", "summary", "integration"):
            errors.append(f"{label}: unsupported Kind {kind!r}")
        budgets[identifier] = checked_budget(values["Budget"], errors, label)
        for heading_text in (() if compact else ("Thông tin cần có", "Chuẩn bị context", "Ghép kết quả")):
            if not subsection(block, heading_text).strip():
                errors.append(f"{label}: missing {heading_text} instructions")
        sources = field_values(metadata, "Source")
        summary_input = field_values(metadata, "Summary input")
        if summary_input and (kind != "summary" or summary_input != ["code"] or variant != "basic" or not sources):
            errors.append(f"{label}: code-summary input requires basic source references")
        prompt_heading = {"code": "Prompt tạo hoặc sửa module", "summary": "Prompt tóm tắt module" if summary_input else "Prompt gộp context",
                          "integration": "Prompt ghép và kiểm tra"}.get(kind, "")
        prompt_section = subsection(block, prompt_heading)
        prompts = fenced_blocks(prompt_section, "text")
        if len(prompts) != 1 or not prompts[0].strip():
            errors.append(f"{label}: require one nonempty text prompt")
        prompt = prompts[0] if prompts else ""
        for dep in depends:
            if f"[Dán tóm tắt {dep}]" not in prompt:
                errors.append(f"{label}: missing context slot for {dep}")
            if f"(#context-{dep})" not in subsection(block, "Context cần chuẩn bị" if compact else "Chuẩn bị context"):
                errors.append(f"{label}: missing summary prompt link for {dep}")
        if compact:
            if subsection(block, "Ghép kết quả").strip():
                errors.append(f"{label}: compact guides must not contain merge-location instructions")
            owns = [x.strip() for line in field_values(metadata, "Config owns") for x in line.split(",") if x.strip()]
            uses = [x.strip() for line in field_values(metadata, "Config uses") for x in line.split(",") if x.strip()]
            for name in owns:
                if name in config_owners:
                    errors.append(f"{label}: configuration {name} already owned by {config_owners[name]}")
                config_owners[name] = identifier
            for name in set(uses) - set(owns):
                if config_owners.get(name) not in ancestors:
                    errors.append(f"{label}: configuration {name} needs its owner's context")
            if kind == "code" and "imports" not in prompt.lower():
                errors.append(f"{label}: generated module must include required imports")
        slots = set(re.findall(r"\[Dán tóm tắt ([a-z][a-z0-9_-]*)\]", prompt))
        if slots != set(depends):
            errors.append(f"{label}: context slots must match direct dependencies")
        excerpts = [scan_reference(source, path.parent, "baseline_basic.ipynb", errors, label) for source in sources]
        targets = field_values(metadata, "Target")
        for target in targets:
            scan_reference(target, path.parent, "baseline.ipynb", errors, label)
        if kind == "code" and not targets:
            errors.append(f"{label}: code module requires a target location")
        attached = fenced_blocks(subsection(block, "Code gửi kèm"), "python")
        if variant == "hardcore" and (sources or attached or fenced_blocks(prompt_section, "python")):
            errors.append(f"{label}: hardcore generation cannot attach source code")
        if variant == "hardcore" and re.search(r"(?:dán|gửi|đính kèm|đọc)\s+(?:đoạn\s+)?(?:code|mã nguồn)|code gửi kèm", prompt, re.I):
            errors.append(f"{label}: hardcore generation must use prose and summaries only")
        if variant == "basic" and sources and attached != excerpts:
            errors.append(f"{label}: attached code must match the referenced notebook excerpt")
        if attached and not sources:
            errors.append(f"{label}: attached code requires Source references")
        if variant == "basic" and kind == "code" and not sources and "bổ sung mới" not in subsection(block, "Code gửi kèm").lower():
            errors.append(f"{label}: new basic module must explicitly say bổ sung mới")
        summary = subsection(block, "Tóm tắt context để dùng tiếp")
        if values["Summary"] not in ("yes", "no"):
            errors.append(f"{label}: Summary must be yes or no")
        if values["Summary"] == "yes":
            summarized.add(identifier)
            if kind != "summary":
                budget = single_field(metadata, "Summary budget", errors, label)
                budgets[f"{identifier}:summary"] = checked_budget(budget, errors, label + " summary")
            if f'<a id="context-{identifier}"></a>' not in block:
                errors.append(f"{label}: missing summary anchor")
            texts = fenced_blocks(summary, "text")
            if kind == "summary":
                if not summary.strip() and not compact:
                    errors.append(f"{label}: missing combined context instructions")
            elif len(texts) != 1 or not fenced_blocks(summary, "python"):
                errors.append(f"{label}: summary requires a text prompt and current-code slot")
            elif set(re.findall(r"\[Dán tóm tắt ([a-z][a-z0-9_-]*)\]", texts[0])) - set(depends):
                errors.append(f"{label}: summary references unknown prerequisite context")
        elif summary.strip():
            errors.append(f"{label}: unexpected summary subsection")
        if budgets[identifier][0] and len(prompt + "\n".join(excerpts)) > budgets[identifier][0] * 3:
            warnings.append(f"{label}: large input; count with Gemini and split before sending if over allowance")
        if index == len(steps) - 1:
            if kind != "integration":
                errors.append("Final step must be integration")
            if not (set(known) - {identifier}).issubset(ancestors):
                errors.append("Final integration must transitively cover all earlier steps")
    if consumed - summarized:
        errors.append("Missing summary prompts for consumed modules: " + ", ".join(sorted(consumed - summarized)))
    if "code" not in kinds and variant != "basic":
        errors.append("Guide must include code-producing steps")
    if verification == "measured":
        measured = set()
        for raw in field_values(header, "Measurement"):
            try:
                evidence = json.loads(raw)
                session = evidence["step"]
                if session == "context-merge":
                    budgets[session] = merge_limits
                inp, total = evidence["input_tokens"], evidence["total_tokens"]
                if session not in budgets or session in measured:
                    raise ValueError("unknown/duplicate session")
                if evidence["model"] != "gemini-3.8-flash" or evidence["method"] != "Gemini countTokens + usageMetadata":
                    raise ValueError("evidence must use target Gemini counters")
                if type(inp) is not int or type(total) is not int or not 0 < inp <= budgets[session][0] or not inp <= total <= 2000 or total - inp > budgets[session][1]:
                    raise ValueError("invalid or excessive measured usage")
                measured.add(session)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid Measurement: {exc}")
        if measured != set(budgets):
            errors.append("Measured verification requires evidence for every generation and summary session")
    return errors, warnings


def scan_focused_guide(path: Path, variant: str, content: str) -> tuple[list[str], list[str]]:
    """Check format 4 independently; formats 1–3 retain their historical rules."""
    errors, warnings = [], []
    if FINGERPRINT_WORD.search(instruction_text(content)):
        errors.append("Guide: audit/fingerprint sessions are excluded from focused work")
    masked = outside_fences(content)
    steps = list(STEP.finditer(masked))
    header = masked[:steps[0].start()] if steps else masked
    for key, expected in (("Format", "4"), ("Variant", variant), ("Model", "gemini-3.8-flash"),
                          ("Thinking", "low"), ("Session target", "1800"), ("Session limit", "2000")):
        if single_field(header, key, errors, "Guide") != expected:
            errors.append(f"Guide: {key} must be {expected}")
    verification = single_field(header, "Token verification", errors, "Guide")
    if verification not in ("unverified", "measured"):
        errors.append("Guide: invalid Token verification")
    if verification == "unverified":
        warnings.append("Gemini token usage is unverified; budgets are allowances, not measured counts")
    workflow = field_values(header, "Constraint workflow")
    extracted = workflow == ["extract-before-edit"]
    if workflow and not extracted:
        errors.append("Guide: invalid Constraint workflow")
    reference_workflow = field_values(header, "Reference workflow")
    reference_guided = reference_workflow == ["report-first"]
    if reference_workflow and (not reference_guided or not extracted):
        errors.append("Guide: report-first references require the extraction workflow")
    groups = ("preprocessing", "model", "hyperparameters") if variant == "basic" else (
        "data", "preprocessing", "model", "hyperparameters", "training")
    seen_groups, known, consumed, summarized, owners, budgets = [], {}, set(), set(), {}, {}

    def budget(raw, label):
        inp, out = checked_budget(raw, errors, label)
        match = re.fullmatch(r"input=(\d+); output=(\d+); reserve=(\d+)", raw)
        if match and sum(map(int, match.groups())) > 1800:
            errors.append(f"{label}: format-4 allowance must total <= 1800")
        return inp, out

    for index, heading in enumerate(steps):
        label = f"Step {index + 1}"
        if int(heading[1]) != index + 1:
            errors.append(f"{label}: numbering must be consecutive")
        stop = steps[index + 1].start() if index + 1 < len(steps) else len(content)
        block = content[heading.end():stop]
        metadata = re.findall(r"<!-- module\s*\n(.*?)-->", outside_fences(block), re.S)
        if len(metadata) != 1:
            errors.append(f"{label}: require one module metadata block")
            continue
        meta = metadata[0]
        values = {k: single_field(meta, k, errors, label) for k in (
            "ID", "Group", "Kind", "Depends", "Budget", "Summary")}
        identifier, group = values["ID"], values["Group"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) or identifier in known:
            errors.append(f"{label}: invalid/duplicate ID")
        if values["Kind"] != "code" or group not in groups:
            errors.append(f"{label}: work must belong to an allowed responsibility group")
        if not seen_groups or group != seen_groups[-1]:
            seen_groups.append(group)
        deps = [] if values["Depends"] == "none" else [d.strip() for d in values["Depends"].split(",")]
        if len(set(deps)) != len(deps) or any(d not in known for d in deps):
            errors.append(f"{label}: dependencies must be unique earlier steps")
        ancestors = set(deps)
        for dep in deps:
            ancestors.update(known.get(dep, set()))
        known[identifier] = ancestors
        consumed.update(deps)
        budgets[identifier] = budget(values["Budget"], label)
        prompts = fenced_blocks(subsection(block, "Prompt tạo hoặc sửa module"), "text")
        if len(prompts) != 1 or not prompts[0].strip():
            errors.append(f"{label}: require one complete text prompt")
        prompt = prompts[0] if prompts else ""
        if set(re.findall(r"\[Dán tóm tắt ([a-z][a-z0-9_-]*)\]", prompt)) != set(deps):
            errors.append(f"{label}: context slots must match dependencies")
        for dep in deps:
            if f"(#context-{dep})" not in subsection(block, "Context cần chuẩn bị"):
                errors.append(f"{label}: missing context link for {dep}")
        own = [n.strip() for v in field_values(meta, "Config owns") for n in v.split(",")]
        uses = [n.strip() for v in field_values(meta, "Config uses") for n in v.split(",")]
        for name in own:
            if not name or name in owners:
                errors.append(f"{label}: duplicate/empty configuration owner {name}")
            owners[name] = identifier
        for name in set(uses) - set(own):
            if owners.get(name) not in ancestors:
                errors.append(f"{label}: configuration {name} lacks owner context")
        sources = field_values(meta, "Source")
        excerpts = [scan_reference(s, path.parent, "baseline_basic.ipynb", errors, label) for s in sources]
        source_heading = "Code cần sửa" if extracted else "Code gửi kèm"
        attached = fenced_blocks(subsection(block, source_heading), "python")
        if attached != excerpts:
            errors.append(f"{label}: basic attachment differs from its referenced source")
        if variant == "basic" and not sources and "bổ sung mới" not in subsection(block, source_heading).lower():
            errors.append(f"{label}: a source-free basic work item must explicitly be new")
        if variant == "hardcore" and (sources or attached or fenced_blocks(subsection(block, "Prompt tạo hoặc sửa module"), "python")):
            errors.append(f"{label}: hardcore generation accepts prose/context only")
        for target in field_values(meta, "Target"):
            scan_reference(target, path.parent, "baseline.ipynb", errors, label)
        reference_input = ""
        if reference_guided:
            material = subsection(block, "Nguồn tham khảo")
            report_refs = field_values(meta, "Reference report")
            code_refs = field_values(meta, "Reference code")
            report_parts = [scan_report_reference(s, path.parent, errors, label) for s in report_refs]
            code_parts = [scan_reference(s, path.parent, "baseline.ipynb", errors, label) for s in code_refs]
            if fenced_blocks(material, "markdown") != report_parts or fenced_blocks(material, "python") != code_parts:
                errors.append(f"{label}: reference attachments must match actual report/code excerpts")
            missing = single_field(meta, "Reference missing", errors, label)
            if missing not in ("yes", "no") or (missing == "no" and not (report_refs or code_refs)):
                errors.append(f"{label}: references must exist or explicitly be marked missing")
            if missing == "yes" and "chưa mô tả đủ" not in material:
                errors.append(f"{label}: insufficient references must be disclosed")
            if f'<a id="references-{identifier}"></a>' not in material:
                errors.append(f"{label}: missing reference-material anchor")
            if subsection(block, "Yêu cầu bài toán").strip() or "yêu cầu của bài toán.]" in prompt:
                errors.append(f"{label}: source-guided work must not require a manual requirements checklist")
            reference_input = material
        if extracted:
            constraints = single_field(meta, "Constraints", errors, label)
            extraction = subsection(block, "Prompt trích ràng buộc")
            related = subsection(block, "Code liên quan")
            related_refs = field_values(meta, "Related source")
            related_code = [scan_reference(s, path.parent, "baseline_basic.ipynb", errors, label)
                            for s in related_refs]
            if fenced_blocks(related, "python") != related_code:
                errors.append(f"{label}: related attachment differs from its referenced source")
            expected = "yes" if sources else "no"
            if constraints != expected:
                errors.append(f"{label}: extraction is required exactly for existing-code edits")
            if constraints == "yes":
                budgets[identifier + ":constraints"] = budget(
                    single_field(meta, "Constraints budget", errors, label), label + " constraints")
                texts = fenced_blocks(extraction, "text")
                if (len(texts) != 1 or "tên/chữ ký" not in texts[0]
                        or "ràng buộc" not in texts[0] or "chưa sửa code" not in texts[0]):
                    errors.append(f"{label}: extraction must request existing names/signatures and constraints without editing")
                if f'<a id="constraints-{identifier}"></a>' not in extraction:
                    errors.append(f"{label}: missing constraint extraction anchor")
                handoff = subsection(block, "Prompt tạo hoặc sửa module")
                if (f"(#constraints-{identifier})" not in handoff
                        or ("[Dán kết quả phân tích.]" if reference_guided else
                            "[Dán tên/chữ ký, các ràng buộc đã trích xuất và yêu cầu của bài toán.]") not in prompt):
                    errors.append(f"{label}: edit must consume extracted signatures and constraints")
                if not related.strip() or (not reference_guided and not subsection(block, "Yêu cầu bài toán").strip()):
                    errors.append(f"{label}: edits need related code and task reference inputs")
                interface_terms = (("Code cần sửa", "Giữ tương thích", "báo trước") if reference_guided else
                                   ("Code cần sửa", "Giữ nguyên tên", "input/output", "ngoài phạm vi"))
                if not all(s in prompt for s in interface_terms):
                    errors.append(f"{label}: edit must preserve interfaces and identify scope changes")
                if reference_guided and (f"(#references-{identifier})" not in extraction
                                         or not texts or "nguồn tham khảo" not in texts[0]):
                    errors.append(f"{label}: analysis must consume current code and reference material")
            elif extraction.strip() or field_values(meta, "Constraints budget") or related_refs:
                errors.append(f"{label}: a new module must use context without an extraction session")
            elif reference_guided and f"(#references-{identifier})" not in subsection(block, "Prompt tạo hoặc sửa module"):
                errors.append(f"{label}: new modules must consume reference material directly")
        summary = subsection(block, "Tóm tắt context để dùng tiếp")
        if values["Summary"] == "yes":
            summarized.add(identifier)
            budgets[identifier + ":summary"] = budget(
                single_field(meta, "Summary budget", errors, label), label + " summary")
            texts = fenced_blocks(summary, "text")
            if (f'<a id="context-{identifier}"></a>' not in summary or len(texts) != 1
                    or not fenced_blocks(summary, "python")):
                errors.append(f"{label}: summary needs an anchor, prompt and current-code input")
            if extracted and texts and not all(s in texts[0] for s in ("chữ ký", "I/O", "cấu hình", "trạng thái", "ràng buộc", "logic")):
                errors.append(f"{label}: summary must retain interfaces, configuration, state, constraints and main logic")
        elif values["Summary"] != "no" or summary.strip() or field_values(meta, "Summary budget"):
            errors.append(f"{label}: unexpected summary or invalid Summary field")
        generation_input = prompt + "\n".join(excerpts)
        if reference_guided and not sources:
            generation_input += reference_input
        elif extracted and sources and not reference_guided:
            generation_input += subsection(block, "Yêu cầu bài toán")
            # The extraction response will be pasted here; its actual size is unknown.
        if budgets[identifier][0] and len(generation_input) > budgets[identifier][0] * 3:
            warnings.append(f"{label}: large input; token fit requires real counters and possibly a smaller complete work item")
        if extracted and sources:
            extraction_input = extraction + "\n".join(excerpts + related_code) + subsection(block, "Context cần chuẩn bị") + reference_input
            if len(extraction_input) > budgets.get(identifier + ":constraints", (0, 0))[0] * 3:
                warnings.append(f"{label}: large extraction input; count tokens and split complete components if needed")
    if seen_groups != list(groups):
        errors.append("Guide: missing, repeated or out-of-order responsibility groups")
    if consumed != summarized:
        errors.append("Guide: summaries must exist exactly for consumed results")
    if verification == "measured":
        measured = set()
        for raw in field_values(header, "Measurement"):
            try:
                evidence = json.loads(raw)
                session, inp, total = evidence["step"], evidence["input_tokens"], evidence["total_tokens"]
                if session not in budgets or session in measured:
                    raise ValueError("unknown/duplicate session")
                if evidence["model"] != "gemini-3.8-flash" or evidence["method"] not in (
                        "Gemini countTokens + usageMetadata", "chatbot token counters"):
                    raise ValueError("real target model counters are required")
                if (type(inp) is not int or type(total) is not int or not 0 < inp <= budgets[session][0]
                        or not inp <= total <= 1800 or total >= 2000 or total - inp > budgets[session][1]):
                    raise ValueError("usage exceeds the declared allowance")
                measured.add(session)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid Measurement: {exc}")
        if measured != set(budgets):
            errors.append("Measured verification requires every generation, extraction and summary session")
    return errors, warnings


def scan_guide(path: Path, variant: str) -> tuple[list[str], list[str]]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read guide: {exc}"], []
    header = outside_fences(content).split("## Bước ", 1)[0]
    if re.search(r"^Format: [234]\s*$", header, re.MULTILINE):
        return scan_modular_guide(path, variant, content)
    if re.search(r"^Format:", header, re.MULTILINE):
        return ["Unsupported prompt guide format"], []
    return scan_legacy_guide(path, variant)


def scan_directory(directory: Path, variants: tuple[str, ...] = ("basic", "hardcore")) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for variant in variants:
        name = f"baseline_{variant}_promting.md"
        found_errors, found_warnings = scan_guide(directory / name, variant)
        errors.extend(f"{name}: {message}" for message in found_errors)
        warnings.extend(f"{name}: {message}" for message in found_warnings)
    return errors, warnings


def main() -> int:
    # Windows consoles/pipes may default to a codepage without Vietnamese characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--variant", choices=("basic", "hardcore"), help="Validate only the selected companion")
    args = parser.parse_args()
    variants = (args.variant,) if args.variant else ("basic", "hardcore")
    errors, warnings = scan_directory(args.task_dir, variants)
    for message in warnings:
        print(f"WARN: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    print(f"Prompt validation: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
