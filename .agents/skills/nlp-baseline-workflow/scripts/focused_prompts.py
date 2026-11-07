"""Format-4 guides: plan changes by responsibility, never replay an entire notebook.

Only local Markdown and Python ASTs are read. Token allowances are not measurements.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re


GROUPS = {
    "basic": ("preprocessing", "model", "hyperparameters"),
    "hardcore": ("data", "preprocessing", "model", "hyperparameters", "training"),
}
TITLES = {
    "data": "Load dữ liệu và chia train/validation/test",
    "preprocessing": "Tiền xử lí dữ liệu",
    "model": "Xử lí mô hình",
    "hyperparameters": "Siêu tham số",
    "training": "Loss, optimizer và training loop",
}
SUMMARY_BUDGET = "input=1400; output=80; reserve=200"
SUMMARY_PROMPT = (
    "Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. "
    "Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. "
    "Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu."
)
PREPROCESS_NAMES = {
    "normalize_text", "clean_text", "words_with_offsets", "context_tokens", "question_words",
    "prepare", "strip_accent", "case_name", "punctuation_name", "source_upos",
}
TRAIN_SETTINGS = re.compile(
    r"(?:.*LEARNING_RATE|.*BATCH_SIZE|EPOCHS|PATIENCE|WEIGHT_DECAY|GRAD_CLIP|"
    r"WARMUP.*|LABEL_SMOOTHING|C_VALUES|MAX_ITER|TOLERANCE|NUM_WORKERS)$"
)


def text(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def clean(value):
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[`*]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def sentences(value, limit):
    """Keep whole sentences/clauses, without slicing source code or identifiers."""
    parts = re.split(r"(?<=[.!?;])\s+", clean(value))
    kept = []
    for part in parts:
        if re.search(r"fingerprint|đã đo|lần chạy|tốt nhất|32 GB|hai notebook", part, re.I):
            continue
        if len(" ".join(kept + [part])) > limit:
            if not kept:
                clauses = re.split(r"; |, (?=không|giữ|chỉ|toàn bộ)", part)
                kept = [clauses[0]]
            break
        kept.append(part)
    return " ".join(kept)


def inspect_notebook(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    section, nodes, constants = "", [], {}
    for cell_no, cell in enumerate(cells, 1):
        code = text(cell)
        if cell["cell_type"] != "code":
            if code.strip():
                section = clean(code.splitlines()[0].lstrip("# "))
            continue
        lines = code.splitlines()
        for node in ast.parse(code).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            nodes.append({"node": node, "cell": cell_no, "start": start, "stop": node.end_lineno,
                          "code": "\n".join(lines[start - 1:node.end_lineno]), "section": section})
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            constants.setdefault(target.id, ast.literal_eval(node.value))
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        try:
                            values = ast.literal_eval(node.value)
                            if len(target.elts) == len(values):
                                for item, value in zip(target.elts, values):
                                    if isinstance(item, ast.Name) and item.id.isupper():
                                        constants.setdefault(item.id, value)
                        except (ValueError, TypeError):
                            pass
    return nodes, constants


def signature(node):
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args if a.arg not in ("self", "cls")]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return node.name + "(" + ", ".join(names) + ")"


def interface(nodes):
    """Describe only externally consumed interfaces, not every helper or AST call."""
    result = []
    called = {n.func.attr for u in nodes for n in ast.walk(u["node"])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.value.id in ("model", "best_model", "reloaded_model")}
    for unit in nodes:
        node = unit["node"]
        if isinstance(node, ast.ClassDef) and any(
                isinstance(n, ast.FunctionDef) and n.name == "forward" for n in node.body):
            methods = [signature(n) for n in node.body if isinstance(n, ast.FunctionDef)
                       and n.name in ({"__init__", "forward", "loss", "generate", "decode"} | called)]
            result.append(node.name + ": " + "; ".join(methods))
        elif isinstance(node, ast.FunctionDef) and (node.name.startswith("collate") or node.name == "make_classifier"):
            returns = [n.value for n in ast.walk(node) if isinstance(n, ast.Return)]
            keys = sorted({str(k.value) for n in returns if isinstance(n, ast.Dict)
                           for k in n.keys if isinstance(k, ast.Constant)})
            result.append(signature(node) + (" -> batch{" + ", ".join(keys) + "}" if keys else ""))
    return "; ".join(result)


def report_sections(directory):
    path = directory / "huong_giai_quyet.md"
    if not path.exists():
        return {}
    sections = {}
    for match in re.finditer(r"^##\s+(\d+)[.)]?[^\n]*\n(.*?)(?=^## |\Z)",
                             path.read_text(encoding="utf-8"), re.M | re.S):
        paragraphs = [p.strip() for p in match[2].split("\n\n") if p.strip()]
        # Exclude generic definitions, tables, execution history and cross-variant prose.
        paragraphs = [p for p in paragraphs if not p.startswith(("|", "Nguồn đối chiếu:", "Token là", "TF-IDF biến"))
                      and not re.search(r"fingerprint|^ID nguồn|^Đây là|^Các thống kê|^\*\*Bằng chứng", p)]
        sections[int(match[1])] = " ".join(paragraphs)
    return sections


def method_brief(sections, nodes):
    details = re.split(r"(?<=[.!?])\s+", clean(sections.get(4, "")))
    details = [p for p in details if not re.search(r"ví dụ|chẳng hạn|để hiểu|nghĩa là|là bộ giải mã", p, re.I)]
    model = sentences(" ".join(details), 470)
    if model:
        return model
    # Notebook-only inputs can identify a method without a report. Do not reuse old
    # guide wording: doing so makes refreshes depend on their own previous output.
    calls = {ast.unparse(n.func).split(".")[-1] for u in nodes for n in ast.walk(u["node"]) if isinstance(n, ast.Call)}
    if "LinearSVC" in calls:
        return "TF-IDF từ/ký tự và LinearSVC; chỉ fit biểu diễn trên train, chọn điều chuẩn bằng validation."
    classes = [u["code"] for u in nodes if isinstance(u["node"], ast.ClassDef)
               and any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in u["node"].body)]
    components = sorted(calls & {"LSTM", "GRU", "Conv1d", "Conv2d", "Transformer", "TransformerEncoder", "MultiheadAttention"})
    if classes and components:
        return "Kết hợp " + ", ".join(components) + " theo luồng encoder → head của notebook tối ưu; khởi tạo và nối đúng I/O."
    functions = [u["node"] for u in nodes if isinstance(u["node"], ast.FunctionDef)]
    if functions:
        return "Triển khai phép biến đổi " + signature(functions[0]) + ": " + clean(ast.unparse(functions[0].body[-1])) + "."
    raise ValueError("Cannot identify a target method; provide a report or an explicit task model")


def profile(directory, optimized, constants):
    sections = report_sections(directory)
    source = constants.get("SOURCE", "")
    if isinstance(source, dict):
        source = {k: v for k, v in source.items() if k not in ("row", "url", "task")}
    source = json.dumps(source, ensure_ascii=False, separators=(",", ":")) if source else ""
    metric = str(constants.get("PRIMARY_METRIC", "metric đã khai báo trong notebook"))
    if metric.lower() in ("cer", "wer", "loss", "perplexity"):
        metric += " (càng thấp càng tốt)"
    data = sentences(sections.get(2, ""), 750)
    policy = constants.get("TASK_PRETRAINING_POLICY", constants.get("PRETRAINING_POLICY", "scratch"))
    components = constants.get("TASK_PRETRAINED_COMPONENTS", constants.get("PRETRAINED_COMPONENTS", []))
    pretrained = "Không dùng pretrained cho tokenizer/embedding/model tác vụ."
    if policy == "image_components":
        allowed = [str(c.get("model_id", "")) + "/" + str(c.get("revision", "")) for c in components]
        pretrained = "Chỉ cho pretrained backbone ảnh " + ", ".join(allowed) + "; phần ngôn ngữ/head học mới."
    calls = {ast.unparse(n.func).split(".")[-1] for u in optimized for n in ast.walk(u["node"]) if isinstance(n, ast.Call)}
    losses = []
    if calls & {"CTCLoss", "ctc_loss"}:
        losses.append("CTC với blank của tokenizer; kiểm độ dài feature/target và nhãn lặp, không che mẫu lỗi bằng zero_infinity")
    if calls & {"cross_entropy", "CrossEntropyLoss"}:
        losses.append("cross-entropy theo head/target của bài; chỉ bỏ padding khi target có padding")
    if calls & {"binary_cross_entropy_with_logits", "BCEWithLogitsLoss"}:
        losses.append("BCE-with-logits cho soft/multi-label target")
    if calls & {"nll_loss", "NLLLoss"}:
        losses.append("NLL trên log-probabilities đã chuẩn hóa")
    if any(isinstance(u["node"], ast.FunctionDef) and u["node"].name == "compute_loss"
           and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "gather"
                   for n in ast.walk(u["node"])) for u in optimized):
        losses.append("token NLL từ log-probabilities, target dịch một bước; bỏ PAD, smoothing theo cấu hình")
    if any("CRF" in getattr(u["node"], "name", "") for u in optimized):
        losses.append("CRF NLL với transition/mask hợp lệ; decode Viterbi")
    special = {k: v for k, v in constants.items() if k in ("PAD", "UNK", "BOS", "EOS", "BLANK", "IGNORE_INDEX")
               or (k.endswith(("_ID", "_IDS")) and isinstance(v, (int, list, dict)))}
    split = {k: constants[k] for k in ("SEED", "VAL_FRACTION", "VALIDATION_FRACTION", "INFERENCE_SPLIT") if k in constants}
    return {
        "method": method_brief(sections, optimized),
        "preprocessing": sentences(sections.get(3, ""), 800) or "Fit biểu diễn chỉ trên train; giữ đúng căn chỉnh input/target và mask.",
        "source": source,
        "data": data,
        "metric": metric,
        "policy": pretrained,
        "loss": "; ".join(losses),
        "special": json.dumps(special, ensure_ascii=False, separators=(",", ":")) if special else "",
        "split": json.dumps(split, ensure_ascii=False, separators=(",", ":")) if split else "",
    }


def role(unit):
    node, section = unit["node"], unit["section"].lower()
    name = getattr(node, "name", "")
    if name in PREPROCESS_NAMES:
        return "representation"
    if isinstance(node, ast.ClassDef):
        if any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in node.body):
            return "model"
        if "dataset" in name.lower():
            return "batch"
        if any(word in section for word in ("kiến trúc", "mô hình")):
            return "model"
    if name.startswith(("collate", "make_loader", "make_nli_loader")):
        return "batch"
    if name == "make_classifier":
        return "model"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = [n.id for n in ast.walk(node.targets[0] if isinstance(node, ast.Assign) else node.target) if isinstance(n, ast.Name)]
        if names and all(n.isupper() for n in names):
            return "hyperparameters" if any(TRAIN_SETTINGS.fullmatch(n) for n in names) else ""
    if any(word in section for word in ("vocabulary", "vocab", "tokenizer", "alphabet", "đặc trưng", "biểu diễn", "căn chỉnh", "padding")):
        if name.startswith("package_"):
            return ""
        return "batch" if "loader" in unit["code"] and not name else "representation"
    return ""


def reference(unit, notebook="baseline_basic.ipynb"):
    return f"{notebook}#cell={unit['cell']}; lines={unit['start']}-{unit['stop']}"


def packs(units, limit=1450):
    """Pack adjacent complete definitions; never cut a function/class body."""
    result, current = [], []
    for unit in units:
        if current and sum(len(u["code"]) for u in current) + len(unit["code"]) > limit:
            result.append(current)
            current = []
        current.append(unit)
    if current:
        if result and sum(len(u["code"]) for u in current) < 300:
            result[-1].extend(current)
        else:
            result.append(current)
    return result


def coalesce(units, directory, name):
    """Present adjacent source statements in one exact excerpt, not tiny fences."""
    cells = json.loads((directory / name).read_text(encoding="utf-8"))["cells"]
    merged = []
    for unit in units:
        if merged and unit["cell"] == merged[-1]["cell"] and unit["start"] <= merged[-1]["stop"] + 2:
            last = merged[-1]
            last["stop"] = unit["stop"]
            last["code"] = "\n".join(text(cells[unit["cell"] - 1]).splitlines()[last["start"] - 1:unit["stop"]])
        else:
            merged.append(dict(unit))
    return merged


def stages(variant, optimized, basic):
    steps = []

    def add(group, title, goal, sources=(), targets=(), deps=(), owns=()):
        identifier = f"s{len(steps) + 1:02d}"
        steps.append(dict(id=identifier, group=group, title=title, goal=goal,
                          sources=list(sources), targets=list(targets), deps=list(dict.fromkeys(deps)), owns=list(owns)))
        return identifier

    data = []
    if variant == "hardcore":
        data = [add("data", TITLES["data"],
                    "Bổ sung loader train/validation/test, giữ nguồn/schema và split chính thức hoặc quy tắc nhóm/seed của bài. "
                    "Giữ ID, nhãn/references và held-out gốc; kiểm nhóm trùng, chưa fit biểu diễn.", owns=("splits",))]
    preprocessing = []
    for kind, title in (("representation", "Biểu diễn và căn chỉnh"), ("batch", "Dataset và batch")):
        target = [u for u in optimized if role(u) == kind]
        original = [u for u in basic if role(u) == kind] if variant == "basic" else []
        # Separate representation from batching, not one chat per helper. Source
        # packs are bounded hints; only actual counters can prove token fit.
        bundles = packs(original) if original else [[]]
        if not target and not original:
            continue
        for index, bundle in enumerate(bundles, 1):
            names = ", ".join(getattr(u["node"], "name", "") for u in bundle if getattr(u["node"], "name", ""))
            goal = ("Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn."
                    if kind == "representation" else
                    "Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.")
            if names:
                goal += " Phạm vi sửa: " + names + "."
            deps = data + preprocessing[-1:]
            identifier = add("preprocessing", title + (f" — phần việc {index}" if len(bundles) > 1 else ""), goal,
                             bundle, target, deps, ("preprocess_" + str(len(preprocessing) + 1),))
            preprocessing.append(identifier)
    if not preprocessing:
        preprocessing = [add("preprocessing", TITLES["preprocessing"],
                             "Bổ sung biểu diễn train-only theo phương pháp; giữ I/O hiện có.", deps=data, owns=("preprocess_1",))]

    target_models = [u for u in optimized if role(u) == "model"]
    source_models = [u for u in basic if role(u) == "model"] if variant == "basic" else []
    helpers = [u for u in target_models if isinstance(u["node"], ast.ClassDef)
               and not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in u["node"].body)]
    # CRF/position encodings are complete reusable components. Large composite
    # models also get an independent encoder, rather than sliced methods.
    helper_ids = []
    for helper in helpers:
        name = helper["node"].name
        helper_ids.append(add("model", "Thành phần " + name,
                              "Viết trọn thành phần " + name + " phục vụ phương pháp bên dưới; định nghĩa API nhận tensor/mask và trả kết quả để model chính dùng. "
                              "Với CRF: gồm NLL và Viterbi, transition hợp lệ, bỏ padding; không chỉ tạo khung lớp.",
                              targets=[helper], deps=preprocessing[-1:], owns=("component_" + name,)))
    main_targets = [u for u in target_models if u not in helpers]
    if sum(len(u["code"]) for u in main_targets) > 2900:
        helper_ids.append(add("model", "Encoder độc lập",
                              "Viết một encoder hoàn chỉnh cho kiến trúc bên dưới, gồm embedding/backbone và mask/lengths. "
                              "Xuất memory và trạng thái/shape cần cho attention/head; chưa viết decoder hay training loop.",
                              targets=main_targets, deps=preprocessing[-1:], owns=("encoder",)))
    model = add("model", "Model và giao diện sử dụng",
                "Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. "
                "Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.",
                source_models, main_targets, preprocessing[-1:] + helper_ids, ("architecture",))
    settings = [u for u in basic if role(u) == "hyperparameters"] if variant == "basic" else []
    hyper = add("hyperparameters", TITLES["hyperparameters"],
                "Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. "
                "Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. "
                "Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. "
                + ("Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng."
                   if variant == "basic" else "Trả cấu hình cố định để bước training dùng; chưa viết vòng train."),
                settings, (), preprocessing[-1:] + [model], ("training_config",))
    if variant == "hardcore":
        add("training", TITLES["training"],
            "Bổ sung huấn luyện thật vào notebook hardcore: loss đúng output/mask/target, optimizer chỉ cho tham số được học, train/eval và chọn checkpoint bằng metric validation. "
            "Tận dụng loader/model/config trong context; không tạo lại chúng. Với sparse dùng fit/predict và chọn tham số validation, không bịa optimizer/backward. "
            "Với neural kiểm forward/loss hữu hạn, backward và một bước update; reload config/weights và trạng thái biểu diễn rồi dùng lại inference hiện có. "
            "Giữ ID, metric và chiều tốt hơn của metric; test chỉ chấm sau khi chọn xong, không chọn tham số bằng test.",
            deps=data + preprocessing[-1:] + [model, hyper])
    return steps


def render(directory: Path, variant: str, old_content=""):
    optimized, constants = inspect_notebook(directory / "baseline.ipynb")
    basic, _ = inspect_notebook(directory / "baseline_basic.ipynb") if variant == "basic" else ([], {})
    facts = profile(directory, optimized, constants)
    stable_interface = interface(basic if variant == "basic" else inspect_notebook(directory / "baseline_hardcore.ipynb")[0]) \
        if (variant == "basic" or (directory / "baseline_hardcore.ipynb").exists()) else ""
    steps = stages(variant, optimized, basic)
    by_id = {s["id"]: s for s in steps}
    consumed = {d for s in steps for d in s["deps"]}
    cells = json.loads((directory / "baseline.ipynb").read_text(encoding="utf-8"))["cells"]
    title = clean(next((text(c).splitlines()[0].lstrip("# ") for c in cells if c["cell_type"] == "markdown" and text(c).strip()), directory.name))
    result = [f"# Prompt {variant}: {title}\n\n<!-- prompt-guide\nFormat: 4\nVariant: {variant}\n"
              "Model: gemini-3.8-flash\nThinking: low\nToken verification: unverified\nSession target: 1800\nSession limit: 2000\n-->",
              "Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. "
              "Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. "
              "Chỉ tóm tắt ở bước có liên kết dùng tiếp; web không đặt trần output riêng nên giới hạn độ dài tóm tắt là yêu cầu mềm."]
    for number, step in enumerate(steps, 1):
        sid, deps, group = step["id"], step["deps"], step["group"]
        step["sources"] = coalesce(step["sources"], directory, "baseline_basic.ipynb") if step["sources"] else []
        step["targets"] = coalesce(step["targets"], directory, "baseline.ipynb") if step["targets"] else []
        budget = "input=1000; output=600; reserve=200" if step["sources"] else "input=650; output=950; reserve=200"
        metadata = ["ID: " + sid, "Group: " + group, "Kind: code", "Depends: " + (", ".join(deps) or "none"),
                    "Budget: " + budget, "Summary: " + ("yes" if sid in consumed else "no")]
        if sid in consumed:
            metadata.append("Summary budget: " + SUMMARY_BUDGET)
        metadata += ["Source: " + reference(u) for u in step["sources"]]
        metadata += ["Target: " + reference(u, "baseline.ipynb") for u in step["targets"]]
        if step["owns"]:
            metadata.append("Config owns: " + ", ".join(step["owns"]))
        uses = [name for d in deps for name in by_id[d]["owns"]]
        if uses:
            metadata.append("Config uses: " + ", ".join(uses))
        heading = TITLES[group] if step["title"] == TITLES[group] else TITLES[group] + " — " + step["title"]
        result.append(f"## Bước {number}: {heading}\n\n<!-- module\n" + "\n".join(metadata) + "\n-->")
        if deps:
            result.append("### Context cần chuẩn bị\n\n" + "\n".join(
                f"- [{by_id[d]['title']}](#context-{d})" for d in deps))
        prompt = step["goal"]
        if group == "data":
            prompt += "\nNguồn/schema/split: " + facts["source"] + ". " + facts["data"]
            if facts["split"]:
                prompt += "\nThiết lập split cần giữ: " + facts["split"] + "."
        elif group == "preprocessing":
            prompt += "\nÝ tưởng tiền xử lí: " + facts["preprocessing"]
            if facts["special"]:
                prompt += "\nSpecial IDs cố định: " + facts["special"] + "."
        elif group == "model":
            prompt += "\nÝ tưởng mô hình: " + facts["method"] + " " + facts["policy"]
        else:
            prompt += "\nPhương pháp: " + sentences(facts["method"], 320) + " Metric: " + facts["metric"] + "."
            if group == "training" and facts["loss"]:
                prompt += "\nObjective cần nối: " + facts["loss"] + "."
        if group == "preprocessing" and not deps and facts["source"]:
            prompt += "\nNguồn: " + facts["source"] + "."
        if stable_interface and group == "model" and not step["sources"]:
            prompt += "\nAPI hiện có: " + stable_interface + "."
        if group in ("hyperparameters", "training"):
            prompt += " Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch."
        prompt += "\nChỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo."
        if deps:
            prompt += "\n\n" + "\n\n".join(f"Context {d} — {by_id[d]['title']}:\n[Dán tóm tắt {d}]" for d in deps)
        result.append("### Prompt tạo hoặc sửa module\n\n```text\n" + prompt + "\n```")
        if step["sources"]:
            result.append("### Code gửi kèm\n\nĐính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:\n\n" +
                          "\n\n".join("```python\n" + u["code"] + "\n```" for u in step["sources"]))
        elif variant == "basic":
            result.append("### Code gửi kèm\n\nPhần bổ sung mới; dùng giao diện hiện có và context ở trên.")
        if sid in consumed:
            result.append(f'### Tóm tắt context để dùng tiếp\n\n<a id="context-{sid}"></a>\n\n```text\n'
                          + SUMMARY_PROMPT + "\n```\n\n```python\n# Dán code hiện tại vừa tạo/sửa của phần việc này.\n```")
    return "\n\n".join(result) + "\n"
