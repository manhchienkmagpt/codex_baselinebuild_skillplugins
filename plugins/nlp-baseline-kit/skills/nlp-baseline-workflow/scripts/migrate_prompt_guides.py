#!/usr/bin/env python3
"""Re-author legacy prompt guides from their local notebooks (offline, opt-in).

Usage: python migrate_prompt_guides.py TASK_DIR [TASK_DIR ...] [--write]
Without --write, build and validate the proposed guides without changing files.
Formats 1 and 2 are upgraded to format 3. Use --refresh to apply current
presentation policy to existing format-3 guides without changing source inputs.
Notebook code is inspected, never executed.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
from pathlib import Path
import re

from validate_prompt_guides import scan_modular_guide


BUDGET = "input=800; output=1000; reserve=200"
SUMMARY_BUDGET = "input=1200; output=500; reserve=300"
REF = re.compile(r"(baseline(?:_basic)?\.ipynb)#cell=(\d+); lines=(\d+)-(\d+)")
BUILTINS = set(dir(builtins)) | {"self", "cls"}


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def fields(text, name):
    return re.findall(r"^" + re.escape(name) + r": (.*)$", text, re.M)


def legacy_steps(content):
    steps = []
    for block in re.split(r"^## Bước \d+: ", content, flags=re.M)[1:]:
        targets = [REF.fullmatch(value) for value in fields(block, "Target")]
        prompt = block.split("### Prompt\n", 1)[-1].strip()
        prompt = re.sub(r"(?:Chỉ |Trả riêng |Trả )(?:trả )?code.*?(?:phiên sau\.|pipeline\.)$", "", prompt).strip()
        inp = fields(block, "Input")[0]
        if "Hợp đồng cần giữ:" in inp:
            spec = inp.split("Hợp đồng cần giữ:", 1)[1]
        elif "đặc tả chữ:" in inp:
            spec = inp.split("đặc tả chữ:", 1)[1]
        else:
            spec = re.sub(r"^.*?; ", "", inp, count=1)
        spec = re.sub(r"Không gửi mã nguồn\.?", "", spec).strip()
        steps.append({"id": fields(block, "ID")[0], "title": block.splitlines()[0],
                      "prompt": prompt, "spec": spec,
                      "targets": [(int(m[2]), int(m[3]), int(m[4])) for m in targets if m]})
    return steps


def defined(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split(".")[0] for a in node.names if a.name != "*"}
    class Bindings(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, item):
            if isinstance(item.ctx, ast.Store):
                self.names.add(item.id)

        def visit_FunctionDef(self, item):
            self.names.add(item.name)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_ListComp(self, item):
            # Comprehension targets do not leak into the enclosing Python scope.
            pass

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp
        visit_Lambda = visit_ListComp

    visitor = Bindings()
    visitor.visit(node)
    return visitor.names


def free_names(nodes, arguments=None):
    """Find external names, respecting nested function and comprehension scopes."""
    class Loads(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, item):
            if isinstance(item.ctx, ast.Load):
                self.names.add(item.id)

        def visit_FunctionDef(self, item):
            args = [a.arg for a in ast.walk(item.args) if isinstance(a, ast.arg)]
            self.names.update(free_names(item.body, args) - {item.name})
            for expr in item.decorator_list + item.args.defaults + [x for x in item.args.kw_defaults if x]:
                self.visit(expr)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ListComp(self, item):
            inner = Loads()
            inner.generic_visit(item)
            bound = set().union(*(defined(g.target) for g in item.generators))
            self.names.update(inner.names - bound)

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

        def visit_For(self, item):
            self.visit(item.iter)
            inner = Loads()
            for stmt in item.body + item.orelse:
                inner.visit(stmt)
            self.names.update(inner.names - defined(item.target))

        visit_AsyncFor = visit_For

    loaded, bound = set(), set(arguments or ())
    for node in nodes:
        visitor = Loads()
        visitor.visit(node)
        loaded.update(visitor.names - bound if arguments is None else visitor.names)
        bound.update(defined(node))
    return (loaded - bound if arguments is not None else loaded) - BUILTINS


def mutations(nodes):
    """Track in-place writes so a later consumer sees the updated object."""
    class Writes(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Subscript(self, item):
            if isinstance(item.ctx, ast.Store):
                base = item.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    self.names.add(base.id)
            self.generic_visit(item)

        def visit_FunctionDef(self, item):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

    visitor = Writes()
    for node in nodes:
        visitor.visit(node)
    return visitor.names


def brief(expr, limit=360):
    value = ast.unparse(expr)
    return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0] + " …"


def safe_parts(node, lines, limit=1500):
    """Split long function bodies at statement boundaries, never in strings."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    if len("\n".join(lines[start - 1:node.end_lineno])) <= limit or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [(start, node.end_lineno, [node])]
    result, first, group = [], start, []
    for stmt in node.body:
        if group and len("\n".join(lines[first - 1:stmt.end_lineno])) > limit:
            result.append((first, group[-1].end_lineno, group))
            first, group = stmt.lineno, []
        group.append(stmt)
    if group:
        result.append((first, group[-1].end_lineno, group))
    return result


def units(cells):
    result = []
    section = "Chuẩn bị"
    for cell_no, cell in enumerate(cells, 1):
        code = source(cell)
        if cell["cell_type"] != "code":
            if code.strip():
                section = re.sub(r"^#+\s*(?:\d+\.\s*)?", "", code.splitlines()[0]).strip()
            continue
        # Packaged notebooks use Python-only cells; fail rather than silently omit magics.
        tree = ast.parse(code)
        lines = code.splitlines()
        group = []

        def add(nodes, start, stop, title, symbol="", owner="", signature="", part=1, parts=1):
            result.append({"cell": cell_no, "start": start, "stop": stop, "nodes": nodes,
                           "title": title, "symbol": symbol, "owner": owner,
                           "signature": signature, "part": part, "parts": parts,
                           "section": section, "code": "\n".join(lines[start - 1:stop]),
                           "exports": set().union(*(defined(n) for n in nodes)) | mutations(nodes),
                           "uses": free_names(nodes), "kind": "code", "deps": set()})

        def flush():
            if not group:
                return
            exports = set().union(*(defined(n) for n in group))
            imports = all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in group)
            names = sorted(exports)
            if imports:
                label = "Imports — " + ", ".join(names[:5]) + ("…" if len(names) > 5 else "")
            elif {"train_loader", "val_loader"} & exports:
                label = "Tạo DataLoader cho các split"
            elif "char_vocab" in exports:
                label = "Bảng ký tự và trạng thái tokenizer"
            elif "MODEL_CONFIG" in exports:
                label = "Cấu hình kiến trúc mô hình"
            elif any(isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch" for n in group):
                label = "Vòng huấn luyện, validation và lưu checkpoint"
            elif {"train_records", "val_records"} & exports:
                label = "Chuẩn hóa và chia dữ liệu"
            else:
                label = section + (" — " + ", ".join(names[:3]) + ("…" if len(names) > 3 else "") if names else "")
            add(list(group), group[0].lineno, group[-1].end_lineno, label)
            group.clear()

        def add_function(node, owner=""):
            parts = safe_parts(node, lines)
            signature = f"{owner + '.' if owner else ''}{node.name}({ast.unparse(node.args)})"
            for i, (start, stop, nodes) in enumerate(parts, 1):
                title = signature + (f" — phần {i}/{len(parts)}" if len(parts) > 1 else "")
                add(nodes, start, stop, title, node.name, owner, signature, i, len(parts))
                # The full function supplies argument scope for sliced bodies.
                result[-1]["uses"] = free_names([node])
                result[-1]["exports"] = {node.name} if not owner and i == len(parts) else set()
                result[-1]["function"] = node

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                flush()
                if isinstance(node, ast.ClassDef):
                    class_start = len(result)
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            add_function(member, node.name)
                        elif not (isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant)):
                            add([member], member.lineno, member.end_lineno,
                                f"Thuộc tính class {node.name}", owner=node.name)
                    for part in result[class_start:]:
                        part["bases"] = ", ".join(ast.unparse(base) for base in node.bases) or "object"
                    result.append({"title": f"Ghép giao diện class {node.name}", "kind": "summary",
                                   "exports": {node.name}, "uses": set(), "deps": set(),
                                   "class_parts": list(range(class_start, len(result))),
                                   "symbol": node.name, "section": section,
                                   "cell": cell_no, "start": node.lineno, "stop": node.end_lineno,
                                   "signature": ", ".join(ast.unparse(base) for base in node.bases), "nodes": []})
                else:
                    add_function(node)
            else:
                if group and (len("\n".join(lines[group[0].lineno - 1:node.end_lineno])) > 1300
                              or isinstance(node, (ast.Import, ast.ImportFrom)) != isinstance(group[-1], (ast.Import, ast.ImportFrom))):
                    flush()
                group.append(node)
        flush()
    for i, unit in enumerate(result, 1):
        unit["id"] = f"m{i:03d}"
    return result


def dependencies(items):
    providers = {}
    for i, unit in enumerate(items):
        for symbol in unit["exports"]:
            providers.setdefault(symbol, []).append(i)
    for i, unit in enumerate(items):
        if unit["kind"] == "summary":
            unit["deps"].update(unit["class_parts"])
            continue
        for symbol in unit["uses"]:
            choices = [j for j in providers.get(symbol, []) if j != i]
            before = [j for j in choices if j < i]
            if symbol in unit["exports"] and not before:
                continue
            if before or choices:
                unit["deps"].add(max(before) if before else min(choices))
        # Preserve intermediate state when one function is authored in slices.
        if unit["part"] > 1:
            unit["deps"].add(i - 1)
        if unit["owner"]:
            previous = [j for j in range(i) if items[j].get("owner") == unit["owner"] and items[j]["cell"] == unit["cell"]]
            constructors = [j for j in previous if items[j]["symbol"] == "__init__"]
            unit["deps"].update(constructors)
            for node in unit["nodes"]:
                for attr in ast.walk(node):
                    if isinstance(attr, ast.Attribute) and isinstance(attr.value, ast.Name) and attr.value.id in ("self", "cls"):
                        unit["deps"].update(j for j in previous if items[j]["symbol"] == attr.attr)
        if unit.get("function"):
            unit["uses"].discard(unit["function"].name)
    # Definition order may differ from call order (e.g. metrics needed before fitting).
    ordered, remaining = [], set(range(len(items)))
    while remaining:
        ready = [i for i in sorted(remaining) if not (items[i]["deps"] & remaining)]
        if not ready:
            detail = {items[i]["title"]: [items[j]["title"] for j in items[i]["deps"] & remaining] for i in sorted(remaining)}
            raise ValueError("Cyclic notebook dependencies: " + repr(detail))
        i = ready[0]
        ordered.append(items[i])
        remaining.remove(i)
    for unit in ordered:
        unit["deps"] = [items[j]["id"] for j in sorted(unit["deps"])]
    return ordered


def reference(unit, name="baseline.ipynb"):
    return f"{name}#cell={unit['cell']}; lines={unit['start']}-{unit['stop']}"


def old_spec(unit, old):
    matching = []
    for step in old:
        if step["id"] in ("summary", "integration", "loaders", "factory"):
            continue
        if any(c == unit["cell"] and a <= unit["stop"] and b >= unit["start"] for c, a, b in step["targets"]):
            matching.append(step)
    if not unit.get("symbol"):
        # Actual constants and operation details below supersede broad setup handoffs.
        matching = [s for s in matching if s["id"] not in ("config", "data", "splits", "training", "reload")]
    sentences = []
    for step in matching:
        spec = re.split(r"Thao tác cần dùng:|Trạng thái/biến phần này:", step["spec"])[0].strip()
        for sentence in [step["prompt"] if not step["id"].startswith("s0") else "", spec]:
            sentence = sentence.strip()
            if sentence and sentence not in sentences:
                sentences.append(sentence)
    return " ".join(sentences)


def details(unit, has_spec=False):
    nodes = unit["nodes"]
    all_nodes = [n for node in nodes for n in ast.walk(node)]
    result = []
    if unit.get("signature"):
        result.append("Chữ ký cần giữ: " + unit["signature"] + ".")
    if unit.get("function") and unit["function"].decorator_list:
        result.append("Decorators cần giữ: " + ", ".join(ast.unparse(d) for d in unit["function"].decorator_list) + ".")
    constants = []
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper() and node.value is not None:
                    constants.append(f"{target.id}: {ast.unparse(node.value)}")
    if constants:
        result.append("Giá trị cấu hình: " + "; ".join(constants) + ".")
    imports = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.extend(a.name + (f" với tên {a.asname}" if a.asname else "") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module}: " + ", ".join(a.name + (f" với tên {a.asname}" if a.asname else "") for a in node.names))
    if imports:
        result.append("Thư viện và tên cần dùng: " + "; ".join(imports) + ".")
    def describe_call(n):
        name = brief(n.func, 80)
        params = [brief(a, 150) for a in n.args if not isinstance(a, (ast.GeneratorExp, ast.Lambda))]
        params += [f"{k.arg or '**'}: {brief(k.value, 150)}" for k in n.keywords]
        return name + (" nhận " + ", ".join(params) if params else "")
    calls = list(dict.fromkeys(describe_call(n) for n in all_nodes if isinstance(n, ast.Call)))
    if has_spec:
        calls = []
    if calls:
        result.append("Thao tác chính: " + "; ".join(calls[:10]) + ".")
    checks = list(dict.fromkeys(brief(n.test, 150) for n in all_nodes if isinstance(n, (ast.If, ast.While, ast.Assert))))
    if checks:
        result.append("Điều kiện cần xử lý: " + "; ".join(checks[:5]) + ".")
    returns = list(dict.fromkeys(brief(n.value, 500) for n in all_nodes if isinstance(n, ast.Return) and n.value))
    if returns:
        result.append("Đầu ra theo hợp đồng: " + "; ".join(returns[:4]) + ".")
    if not unit.get("symbol") and unit["exports"]:
        result.append("Biến cần cung cấp: " + ", ".join(sorted(unit["exports"])) + ".")
    return " ".join(result)


def contexts(depends, by_id, consumer=None):
    slots = []
    for dep in depends:
        required = sorted(by_id[dep].get("exports", set()) & consumer.get("uses", set())) if consumer else []
        note = "\nChỉ cần phần: " + ", ".join(required) + "." if required else ""
        slots.append(f"Context {dep} — {by_id[dep]['title']}:{note}\n[Dán tóm tắt {dep}]")
    return "\n\n".join(slots)


def render_v2(directory, variant, old_content):
    cells = notebook(directory / "baseline.ipynb")
    basic_units = units(notebook(directory / "baseline_basic.ipynb"))
    items = dependencies(units(cells))
    old = legacy_steps(old_content)
    consumers = {item["id"]: [] for item in items}
    for item in items:
        for dep in item["deps"]:
            consumers[dep].append(item["id"])
    sinks = [item["id"] for item in items if not consumers[item["id"]]]
    final = {"id": "integration", "title": "Ghép và kiểm tra toàn bộ bản làm việc", "kind": "integration", "deps": sinks}
    for dep in sinks:
        consumers[dep].append("integration")
    items.append(final)
    by_id = {item["id"]: item for item in items}
    title = old_content.splitlines()[0]
    intro = next((p for p in old_content.split("\n\n") if p.startswith("Đích:")), "")
    text = [title, f"<!-- prompt-guide\nFormat: 2\nVariant: {variant}\nModel: gemini-3.8-flash\nThinking: low\nToken verification: unverified\nContext merge budget: {SUMMARY_BUDGET}\n-->"]
    text.append("Mục tiêu: nâng bản làm việc tới phương pháp của baseline.ipynb; không cam kết cùng điểm số. "
                "Notebook và báo cáo gốc được giữ nguyên. Cell được đếm từ 1, gồm cả Markdown; dòng được tính trong cell. "
                "Vị trí tham chiếu dùng cho người học đối chiếu, chatbot không cần truy cập file.")
    if intro:
        text.append(intro)
    text.append("Cách làm: mỗi phiên mới chỉ gửi khối prompt và các nội dung đính kèm được chỉ định. "
                "Để tạo module có phụ thuộc, lần lượt lấy tóm tắt ở các liên kết trong mục Chuẩn bị context, rồi dán vào đúng ô trong prompt. "
                "Tóm tắt một lần và dùng lại khi code/API chưa đổi; khi sửa module, kiểm lại context của các caller bị ảnh hưởng.")
    text.append(("Basic: gửi thêm code Python hiện tại để sửa; đoạn nhúng là bản ban đầu, hãy thay bằng code đang dùng nếu đã chỉnh sửa. "
                 if variant == "basic" else
                 "Hardcore: chỉ gửi prompt và context chữ khi tạo module; được gửi code vừa tạo trong phiên riêng để lấy tóm tắt. Notebook hardcore gốc vẫn chỉ inference với trọng số chưa huấn luyện. ")
                + "Khối Python có ghi chỗ dán là ô nhập, không phải code để chạy. Giữ code trả về cục bộ; ghép các phần đúng scope trước khi thực thi.")
    text.append("Dùng Gemini 3.8 Flash, thinking low. Toàn phiên tối đa 2000 tokens, gồm system instructions, prompt, code, context, câu trả lời và thinking. "
                "Ngân sách tạo/sửa mặc định: input 800, output/thinking 1000, dự phòng 200; phiên tóm tắt: input 1200, output/thinking 500, dự phòng 300. "
                "Đây là hạn mức chưa đo. Dùng Gemini countTokens cho input thực tế và đặt trần output nếu nền tảng hỗ trợ; không coi số từ/ký tự là số đo token. "
                "Nếu vượt hạn mức, tách tiếp theo statement/phương thức và kiểm lại; không gửi tràn hoặc nối mù câu trả lời bị cắt. "
                "Giữ seed/cấu hình cố định, default precision, không AMP, tự dò VRAM hay tự giảm batch; fit train-only, chọn trên validation, giữ held-out để đánh giá cuối.")
    text.append("Nếu nhiều context quá dài, mở phiên riêng và gửi prompt gộp dưới đây cùng các tóm tắt liên quan. "
                "Giữ nhãn từng module trong kết quả để điền lại các ô; đối chiếu code lưu cục bộ, không đoán chữ ký hay hằng số bị thiếu.")
    text.append("```text\nRút gọn các context dưới đây theo từng module, giữ chữ ký, đầu vào/đầu ra, shape/dtype/device, "
                "cấu hình và điều kiện mà caller còn dùng. Chỉ trả tóm tắt chữ có nhãn module, tối đa 250 tokens; "
                "không dựng lại code và báo rõ thông tin thiếu hoặc mâu thuẫn.\n\n[Dán các tóm tắt cần rút gọn]\n```")
    for number, unit in enumerate(items, 1):
        identifier, kind, deps = unit["id"], unit["kind"], unit["deps"]
        need_summary = kind != "integration"
        meta = [f"ID: {identifier}", "Depends: " + (", ".join(deps) or "none"), f"Kind: {kind}",
                f"Budget: {SUMMARY_BUDGET if kind == 'summary' else BUDGET}", "Summary: " + ("yes" if need_summary else "no")]
        if need_summary and kind != "summary":
            meta.append(f"Summary budget: {SUMMARY_BUDGET}")
        if kind != "integration":
            meta.append("Target: " + reference(unit))
        sources = []
        if variant == "basic" and kind == "code":
            if unit.get("symbol"):
                candidates = [u for u in basic_units if u["kind"] == "code" and u.get("symbol") == unit["symbol"] and u.get("owner") == unit["owner"]]
                if candidates:
                    # Match corresponding slices, retaining the callable's original signature.
                    candidate = candidates[min(unit["part"] - 1, len(candidates) - 1)]
                    sources = [candidate]
            else:
                names = unit["exports"]
                candidates = [u for u in basic_units if u["kind"] == "code" and not u.get("symbol") and names & u["exports"]]
                if candidates:
                    sources = [max(candidates, key=lambda u: len(names & u["exports"]))]
            for s in sources:
                meta.append("Source: " + reference(s, "baseline_basic.ipynb"))
            if len(sources) == 1 and unit["code"].strip() == sources[0]["code"].strip():
                kind = "summary"
                meta = [line for line in meta if not line.startswith("Summary budget:")]
                meta = ["Kind: summary" if line.startswith("Kind:") else
                        f"Budget: {SUMMARY_BUDGET}" if line.startswith("Budget:") else line for line in meta]
                meta.append("Summary input: code")
        text.append(f"## Bước {number}: {unit['title']}\n\n<!-- module\n" + "\n".join(meta) + "\n-->")
        purpose = "Để tóm tắt chính xác phần code đã có, cần context của: " if kind == "summary" and sources else "Để tạo được module này, cần context của: "
        standalone = "Code phần này đã có sẵn; chỉ cần tóm tắt, không cần tạo lại." if kind == "summary" and sources else "Module này độc lập, có thể tạo ngay từ đặc tả bên dưới."
        text.append("### Thông tin cần có\n\n" + (purpose + "; ".join(by_id[d]["title"] for d in deps) + "." if deps else standalone))
        if deps:
            instructions = ["Nên lần lượt tạo tóm tắt context bằng các prompt sau trước. Nếu đã có tóm tắt khớp code hiện tại thì dùng lại.\n"]
            instructions.extend(f"{i}. [{by_id[d]['title']} — prompt context](#context-{d})." for i, d in enumerate(deps, 1))
            text.append("### Chuẩn bị context\n\n" + "\n".join(instructions))
        else:
            text.append("### Chuẩn bị context\n\nKhông cần tóm tắt module trước.")
        if kind == "summary" and sources:
            prompt = f"Tóm tắt giao diện của {unit['title']} từ phần Python hiện tại gửi kèm; phần này đã giống bản tối ưu nên giữ code đang có. "
            prompt += "Giữ tên/chữ ký, đầu vào/đầu ra, shape/dtype/device, imports/cấu hình và trạng thái caller còn dùng. "
            prompt += "Chỉ trả tóm tắt chữ tối đa 150 tokens; đối chiếu context phụ thuộc và báo rõ thông tin thiếu, không viết lại code."
            heading = "Prompt tóm tắt module"
        elif kind == "summary":
            prompt = f"Gộp các context thành giao diện class {unit['symbol']}, kế thừa {unit['signature'] or 'object'}. "
            prompt += "Giữ constructor, tên và chữ ký methods, đầu vào/đầu ra, trạng thái dùng chung và điều kiện để gọi class. "
            prompt += "Chỉ trả tóm tắt chữ, không sinh lại class; báo rõ API thiếu hoặc xung đột."
            heading = "Prompt gộp context"
        elif kind == "integration":
            prompt = "Viết checklist và chỉ phần glue/assertions còn thiếu để nối các module theo context dưới đây. "
            prompt += "Kiểm imports/cấu hình, thứ tự định nghĩa, producer/consumer, split/IDs và leakage, một batch forward/loss/update khi có huấn luyện, "
            prompt += "reload tương đương và metric chính thức. Không sinh lại toàn bộ notebook hoặc tự chạy train/test; báo rõ hợp đồng thiếu."
            heading = "Prompt ghép và kiểm tra"
        else:
            heading = "Prompt tạo hoặc sửa module"
            action = "Sửa phần Python gửi kèm hoặc bổ sung phần còn thiếu" if variant == "basic" and sources else "Viết riêng module"
            prompt = f"{action} cho {unit['title']}, thuộc phần {unit['section']}. "
            spec = old_spec(unit, old)
            prompt += spec + " " + details(unit, bool(spec))
            if unit.get("owner"):
                prompt += f" Đặt phần này trong class {unit['owner']} kế thừa {unit['bases']}; giữ các methods đã có, không trả lại cả class."
            if unit.get("parts", 1) > 1:
                prompt += f" Đây là phần {unit['part']}/{unit['parts']} của cùng callable; nối đúng thứ tự và giữ biến trung gian, chưa chạy khi thiếu phần."
            prompt += " Chỉ trả code của phần này và hướng dẫn chèn cần thiết; dùng đúng API trong các context, báo rõ hợp đồng thiếu thay vì đoán."
        if deps:
            prompt += "\n\n" + contexts(deps, by_id, unit)
        anchor = f'<a id="context-{identifier}"></a>\n\n' if kind == "summary" else ""
        text.append(anchor + f"### {heading}\n\n```text\n{prompt.strip()}\n```")
        if variant == "basic" and (kind == "code" or sources):
            attachment = "### Code gửi kèm\n\n"
            if not sources:
                attachment += "Module bổ sung mới: không có phần tương ứng trong bản basic để đính kèm. Dùng đặc tả và context ở trên."
            for s in sources:
                attachment += f"Nguồn ban đầu: {reference(s, 'baseline_basic.ipynb')} — {s.get('signature') or s['title']}. "
                attachment += "Nếu đã chỉnh sửa, thay khối dưới bằng đúng phần hiện tại; không chép bản cũ đè thay đổi.\n\n```python\n" + s["code"] + "\n```\n\n"
            text.append(attachment.rstrip())
        if kind == "summary":
            merge = ("Giữ nguyên code module hiện tại. Lưu bản tóm tắt để dùng ở các liên kết context của caller; không thay code bằng bản tóm tắt."
                     if sources else "Lưu kết quả làm context của class; giữ code các methods đã ghép cục bộ, không thay code bằng bản tóm tắt.")
        elif kind == "integration":
            merge = "Trong bản làm việc riêng, chỉ thêm glue/assertions còn thiếu sau các định nghĩa liên quan. "
            merge += "Hoàn tất mọi class/khối nhiều phần rồi mới Run All từ kernel sạch; không thay bằng toàn bộ pipeline mới."
        else:
            names = unit.get("signature") or ", ".join(sorted(unit["exports"])) or unit["section"]
            merge = f"Vị trí đối chiếu: {reference(unit)} — {names}. "
            merge += "Trong bản làm việc, thay đúng callable/khối tương ứng hoặc chèn phần mới tại đây; giữ các phần khác. "
            if unit.get("owner") or unit.get("parts", 1) > 1:
                merge += "Giữ đúng class/scope và thứ tự các phần; chỉ chạy sau khi ghép đủ. "
                if unit.get("parts", 1) > 1:
                    merge += "Lưu riêng lát trả về rồi nối trong cùng callable; không thay cả hàm/class bằng một lát chưa hoàn tất. "
            callers = consumers[identifier]
            if callers:
                merge += "Kiểm và cập nhật caller/API tại: " + "; ".join(by_id[d]["title"] for d in callers) + "."
        text.append("### Ghép kết quả\n\n" + merge)
        if need_summary:
            summary = '### Tóm tắt context để dùng tiếp\n\n'
            if kind != "summary":
                summary += f'<a id="context-{identifier}"></a>\n\n'
            if kind == "summary":
                summary += "Kết quả của prompt " + ("tóm tắt" if sources else "gộp") + " ngay phía trên chính là context này. Dùng lại khi code/API chưa thay đổi."
            else:
                summary += "Sau khi hoàn tất phần này, mở phiên mới và gửi prompt dưới cùng code hiện tại vào ô Python. "
                summary += "Đối chiếu kết quả với code, lưu context theo tên module để các bước sau dùng lại.\n\n"
                summary += f"```text\nĐọc code của module {unit['title']} dưới đây và tóm tắt giao diện cho caller. "
                summary += "Giữ tên/chữ ký, đầu vào/đầu ra, shape/dtype/device, imports/cấu hình, trạng thái hoặc biến trung gian và điều kiện bắt buộc còn dùng. "
                summary += "Chỉ trả chữ tối đa 150 tokens, không viết lại code hoặc đoán thông tin thiếu; nếu là một lát callable, ghi rõ phần nào chưa hoàn tất."
                if deps:
                    summary += " Dùng các context đã đối chiếu dưới đây để xác định API ngoài và trạng thái từ phần trước; chỉ giữ thông tin caller còn cần.\n\n" + contexts(deps, by_id, unit)
                summary += "\n```\n\n"
                summary += f"```python\n# Dán code hiện tại của {unit.get('signature') or unit['title']} tại đây.\n# Dùng kết quả vừa ghép/sửa, không dùng lại code basic cũ.\n```"
            text.append(summary)
    return "\n\n".join(text) + "\n"


def render(directory, variant, old_content):
    from prompt_modules import render as compact_render
    return compact_render(directory, variant, old_content)


def migrate(directory, write=False, refresh=False):
    proposed = []
    for variant in ("basic", "hardcore"):
        path = directory / f"baseline_{variant}_promting.md"
        old = path.read_text(encoding="utf-8")
        if "\nFormat: 3\n" in old:
            if not refresh:
                continue
            from prompt_modules import remove_fingerprint_steps
            content = remove_fingerprint_steps(old)
        else:
            content = render(directory, variant, old)
        errors, warnings = scan_modular_guide(path, variant, content)
        if errors:
            raise ValueError(str(path) + ": " + "\n".join(errors))
        if content != old:
            proposed.append((path, content, len(warnings)))
    for path, content, warnings in proposed:
        if write:
            path.write_text(content, encoding="utf-8", newline="\n")
        print(f"{'WROTE' if write else 'CHECKED'} {path}: {content.count('<!-- module')} modules, {warnings} warnings")
    return len(proposed)


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dirs", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Refresh format-3 guides while preserving attached code")
    args = parser.parse_args()
    for directory in args.task_dirs:
        migrate(directory, args.write, args.refresh)


if __name__ == "__main__":
    main()
