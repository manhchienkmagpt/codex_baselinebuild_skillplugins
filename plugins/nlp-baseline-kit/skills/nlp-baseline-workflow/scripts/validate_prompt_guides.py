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


def scan_guide(path: Path, variant: str) -> tuple[list[str], list[str]]:
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
