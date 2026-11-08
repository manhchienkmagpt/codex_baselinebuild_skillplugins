"""Format-4 guides: plan changes by responsibility, never replay an entire notebook.

Only local Markdown and Python ASTs are read. Token allowances are not measurements.
"""
from __future__ import annotations

import ast
import copy
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
SUMMARY_BUDGET = "input=1100; output=500; reserve=200"
CONSTRAINT_BUDGET = SUMMARY_BUDGET
CONSTRAINT_PROMPT = (
    "Đọc code hiện tại và nguồn tham khảo theo ý tưởng:\n{idea}\n\n"
    "Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.\n"
    "Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code."
)
CONSTRAINT_SLOT = "[Dán kết quả phân tích.]"
SUMMARY_PROMPT = (
    "Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.\n"
    "Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,\n"
    "ràng buộc và logic xử lý chính."
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
    args = copy.deepcopy(node.args)
    positional = args.posonlyargs or args.args
    if positional and positional[0].arg in ("self", "cls"):
        # Drop only the bound receiver; retain defaults, annotations and / or *.
        if len(args.defaults) == len(args.posonlyargs) + len(args.args):
            args.defaults.pop(0)
        positional.pop(0)
    result = node.name + "(" + ast.unparse(args) + ")"
    if node.returns:
        result += " -> " + ast.unparse(node.returns)
    return result


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






def defined_names(unit):
    node = unit["node"]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


def related_sources(units, sources, limit=1450):
    """Attach small complete providers/callers; list larger neighbors for the learner."""
    selected = {(u["cell"], u["start"], u["stop"]) for u in sources}
    names = set().union(*(defined_names(u) for u in sources))
    loads = {n.id for u in sources for n in ast.walk(u["node"])
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    related, extra, size = [], [], 0
    for unit in units:
        if (unit["cell"], unit["start"], unit["stop"]) in selected:
            continue
        used = {n.id for n in ast.walk(unit["node"])
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if not (defined_names(unit) & loads or used & names):
            continue
        if size + len(unit["code"]) <= limit:
            related.append(unit)
            size += len(unit["code"])
        else:
            node = unit["node"]
            extra.append(signature(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         else node.name if isinstance(node, ast.ClassDef)
                         else f"cell {unit['cell']}, dòng {unit['start']}–{unit['stop']}")
    return related, extra


def report_excerpts(directory, sections):
    """Return exact, complete report paragraphs with reproducible line references."""
    path = directory / "huong_giai_quyet.md"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result, section, start = [], None, None
    for index in range(len(lines) + 1):
        line = lines[index] if index < len(lines) else ""
        heading = re.match(r"^##\s+(\d+)[.)]?\s", line)
        boundary = not line.strip() or heading or line.startswith("## ")
        if start is not None and boundary:
            excerpt = "\n".join(lines[start:index])
            # Data alternatives, generic definitions and execution-history notes
            # are not the selected implementation's description.
            skip = re.match(r"(?:Siêu tham số là|Đây là lựa chọn thiết kế|Các thống kê|Nguồn đối chiếu:)", excerpt)
            if section in sections and not skip and not excerpt.startswith("|"):
                result.append({"ref": f"huong_giai_quyet.md#lines={start + 1}-{index}", "text": excerpt})
            start = None
        if heading:
            section = int(heading[1])
        elif line.startswith("## "):
            section = None
        elif line.strip() and start is None:
            start = index
    return result


def reference_material(directory, step, optimized):
    sections = {"data": (1, 2), "preprocessing": (3,), "model": (4,),
                "hyperparameters": (5,), "training": (4, 5)}[step["group"]]
    reports = report_excerpts(directory, sections)
    code = []
    fallback = not reports or sum(len(p["text"]) for p in reports) < 100
    if fallback:
        code.extend(step["targets"])
        if not code:
            roles = {"data": set(), "preprocessing": {"representation", "batch"},
                     "model": {"model"}, "hyperparameters": {"hyperparameters"}, "training": set()}[step["group"]]
            code.extend(u for u in optimized if role(u) in roles)
            if step["group"] == "training":
                code.extend(u for u in optimized if getattr(u["node"], "name", "") in
                            {"compute_loss", "train_epoch", "fit", "fit_model"}
                            or any(w in u["section"].lower() for w in ("huấn luyện", "training", "loss")))
            elif step["group"] == "data":
                code.extend(u for u in optimized if any(w in u["section"].lower()
                            for w in ("load", "chia tập", "dữ liệu thô")))
    if step["group"] == "training" and not re.search(
            r"loss|NLL|cross.entropy|CTC|CRF|objective|likelihood", " ".join(p["text"] for p in reports), re.I):
        code.extend(u for u in optimized if getattr(u["node"], "name", "") in
                    {"compute_loss", "objective", "compute_objective"})
    substantive = bool((reports and not fallback) or code)
    # Actual declarations carry version, labels, special IDs and policy without
    # requiring the learner to transcribe a technical checklist.
    source_names = {
        "SOURCE", "DATASET_ID", "HF_REVISION", "DATASET_VERSION", "DATASET_REVISION",
        "SEED", "VAL_FRACTION", "VALIDATION_FRACTION", "INFERENCE_SPLIT",
    }
    metric_names = {"PRIMARY_METRIC", "SELECTION_METRIC", "DIRECTION"}
    policy_names = {"PRETRAINING_POLICY", "PRETRAINED_COMPONENTS", "TASK_PRETRAINING_POLICY",
                    "TASK_PRETRAINED_COMPONENTS", "ALLOW_PRETRAINED_EVALUATORS"}
    label_names = {"LABELS", "LABEL_NAMES", "ENTITY_TYPES", "SPECIALS", "PAD", "UNK", "BOS", "EOS", "BLANK", "IGNORE_INDEX"}
    metadata_names = set()
    if step["group"] == "data" or not step["deps"]:
        metadata_names |= source_names | metric_names | policy_names
    if step["group"] in ("model", "training"):
        metadata_names |= policy_names
    if step["group"] in ("hyperparameters", "training"):
        metadata_names |= metric_names
    if step["group"] in ("data", "preprocessing", "model", "training"):
        metadata_names |= label_names
    if step["group"] == "hyperparameters":
        code.extend(u for u in optimized if role(u) == "hyperparameters")
    for unit in optimized:
        names = defined_names(unit)
        if isinstance(unit["node"], (ast.Assign, ast.AnnAssign)) and any(
                n in metadata_names or (step["group"] != "hyperparameters" and
                n.endswith(("_ID", "_IDS")) and not n.startswith(("DATASET_", "TASK_"))) for n in names):
            code.append(unit)
    unique = {(u["cell"], u["start"], u["stop"]): u for u in code}
    code = coalesce(sorted(unique.values(), key=lambda u: (u["cell"], u["start"])), directory, "baseline.ipynb")
    return reports, code, not substantive


def work_idea(step):
    group = step["group"]
    if group == "data":
        return "Load dữ liệu và chia tập theo mô tả bài toán."
    if group == "preprocessing":
        return ("Nối Dataset, batch và padding với biểu diễn đã chốt."
                if "Dataset và batch" in step["title"] else
                "Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.")
    if group == "model":
        if step["title"] != "Model và giao diện sử dụng":
            return "Viết " + step["title"] + " hoàn chỉnh để mô hình chính sử dụng."
        return "Triển khai mô hình theo nguồn tham khảo và context hiện có."
    if group == "hyperparameters":
        return "Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo."
    return "Bổ sung huấn luyện cho mô hình và dữ liệu đã có."


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
            goal = ("Cập nhật biểu diễn và căn chỉnh theo mục tiêu tiền xử lí."
                    if kind == "representation" else
                    "Tạo Dataset, collate và DataLoader dùng biểu diễn đã chốt. "
                    "Thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.")
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
    optimized, _ = inspect_notebook(directory / "baseline.ipynb")
    basic, _ = inspect_notebook(directory / "baseline_basic.ipynb") if variant == "basic" else ([], {})
    stable_interface = interface(basic if variant == "basic" else inspect_notebook(directory / "baseline_hardcore.ipynb")[0]) \
        if (variant == "basic" or (directory / "baseline_hardcore.ipynb").exists()) else ""
    steps = stages(variant, optimized, basic)
    by_id = {s["id"]: s for s in steps}
    consumed = {d for s in steps for d in s["deps"]}
    cells = json.loads((directory / "baseline.ipynb").read_text(encoding="utf-8"))["cells"]
    title = clean(next((text(c).splitlines()[0].lstrip("# ") for c in cells if c["cell_type"] == "markdown" and text(c).strip()), directory.name))
    result = [f"# Prompt {variant}: {title}\n\n<!-- prompt-guide\nFormat: 4\nVariant: {variant}\n"
              "Model: gemini-3.8-flash\nThinking: low\nToken verification: unverified\nSession target: 1800\nSession limit: 2000\n"
              "Constraint workflow: extract-before-edit\nReference workflow: report-first\n-->",
              "Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. "
              "Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. "
              "Lượt phân tích dùng Code cần sửa, Code liên quan, Nguồn tham khảo và context. "
              "Lượt sửa chỉ cần Code cần sửa, kết quả phân tích và context; gửi thêm nguồn khi còn điều chưa rõ. "
              "Module mới dùng ý tưởng, nguồn tham khảo và context trực tiếp. Tham số trong nguồn là gợi ý để chọn phù hợp. "
              "Chỉ tóm tắt ở bước có liên kết dùng tiếp; không giới hạn cứng số dòng/token của bản tóm tắt. "
              "Nếu cả phiên không đủ chỗ, chia theo thành phần hoàn chỉnh."]
    for number, step in enumerate(steps, 1):
        sid, deps, group = step["id"], step["deps"], step["group"]
        related, extra = related_sources(basic, step["sources"]) if step["sources"] else ([], [])
        related = coalesce(related, directory, "baseline_basic.ipynb") if related else []
        step["sources"] = coalesce(step["sources"], directory, "baseline_basic.ipynb") if step["sources"] else []
        step["targets"] = coalesce(step["targets"], directory, "baseline.ipynb") if step["targets"] else []
        reports, reference_code, missing_reference = reference_material(directory, step, optimized)
        budget = "input=1000; output=600; reserve=200" if step["sources"] else "input=650; output=950; reserve=200"
        metadata = ["ID: " + sid, "Group: " + group, "Kind: code", "Depends: " + (", ".join(deps) or "none"),
                    "Budget: " + budget, "Summary: " + ("yes" if sid in consumed else "no"),
                    "Constraints: " + ("yes" if step["sources"] else "no")]
        if step["sources"]:
            metadata.append("Constraints budget: " + CONSTRAINT_BUDGET)
        if sid in consumed:
            metadata.append("Summary budget: " + SUMMARY_BUDGET)
        metadata += ["Source: " + reference(u) for u in step["sources"]]
        metadata += ["Related source: " + reference(u) for u in related]
        metadata += ["Target: " + reference(u, "baseline.ipynb") for u in step["targets"]]
        metadata += ["Reference report: " + item["ref"] for item in reports]
        metadata += ["Reference code: " + reference(u, "baseline.ipynb") for u in reference_code]
        metadata.append("Reference missing: " + ("yes" if missing_reference else "no"))
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
        if stable_interface and group == "model" and not step["sources"]:
            result.append("### Code liên quan\n\nGiao diện hiện có để nối với module mới: " + stable_interface + ".")
        idea = work_idea(step)
        source_parts = [f'### Nguồn tham khảo\n\n<a id="references-{sid}"></a>\n\n'
                        "Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. "
                        "Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp."]
        for item in reports:
            fence = "`" * max(3, max((len(m[0]) + 1 for m in re.finditer(r"`+", item["text"])), default=3))
            source_parts.append("Trích báo cáo:\n\n" + fence + "markdown\n" + item["text"] + "\n" + fence)
        if reference_code:
            source_parts.append("Code/cấu hình tham khảo trích từ notebook tối ưu:\n\n" + "\n\n".join(
                "```python\n" + u["code"] + "\n```" for u in reference_code))
        if missing_reference:
            source_parts.append("Nguồn hiện có chưa mô tả đủ phần việc này; đánh dấu điều chưa rõ, không tự đoán phần thiếu.")
        result.append("\n\n".join(source_parts))
        if step["sources"]:
            result.append(f'### Prompt trích ràng buộc\n\n<a id="constraints-{sid}"></a>\n\n'
                          f'Gửi cùng [nguồn tham khảo](#references-{sid}), code hiện tại và context phụ thuộc.\n\n```text\n'
                          + CONSTRAINT_PROMPT.format(idea=idea) + "\n```")
            prompt = "Sửa phần “Code cần sửa” theo ý tưởng:\n" + idea
            prompt += "\n\nDựa trên phân tích:\n" + CONSTRAINT_SLOT
            prompt += "\n\nGiữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm."
        else:
            prompt = "Tạo module theo ý tưởng:\n" + idea
            prompt += "\n\nDùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ."
        if deps:
            prompt += "\n\n" + "\n\n".join(f"Context {d} — {by_id[d]['title']}:\n[Dán tóm tắt {d}]" for d in deps)
        handoff = (f"Dán kết quả [phân tích code và nguồn](#constraints-{sid}) vào ô bên dưới; gửi cùng Code cần sửa và context.\n\n"
                   if step["sources"] else f"Gửi cùng [nguồn tham khảo](#references-{sid}) và context ở trên.\n\n")
        result.append("### Prompt tạo hoặc sửa module\n\n" + handoff + "```text\n" + prompt + "\n```")
        if step["sources"]:
            result.append("### Code cần sửa\n\nĐính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:\n\n" +
                          "\n\n".join("```python\n" + u["code"] + "\n```" for u in step["sources"]))
            neighbors = "Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ."
            if related:
                neighbors += "\n\n" + "\n\n".join("```python\n" + u["code"] + "\n```" for u in related)
            if extra:
                neighbors += "\n\nĐối chiếu thêm code hiện tại của: " + "; ".join(dict.fromkeys(extra)) + ". Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên."
            if stable_interface:
                neighbors += "\n\nGiao diện dùng trong notebook: " + stable_interface + "."
            result.append("### Code liên quan\n\n" + neighbors)
        elif variant == "basic":
            result.append("### Code cần sửa\n\nPhần bổ sung mới; dùng giao diện hiện có và context ở trên.")
        if sid in consumed:
            result.append(f'### Tóm tắt context để dùng tiếp\n\n<a id="context-{sid}"></a>\n\n```text\n'
                          + SUMMARY_PROMPT + "\n```\n\n```python\n# Dán code hiện tại vừa tạo/sửa của phần việc này.\n```")
    return "\n\n".join(result) + "\n"
