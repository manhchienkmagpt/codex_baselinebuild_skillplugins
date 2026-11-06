"""Compact, dependency-aware manual prompts. Inspect notebooks; never execute them."""
from __future__ import annotations

import ast
import re

from migrate_prompt_guides import (
    BUDGET, SUMMARY_BUDGET, REF, defined, dependencies, fields, free_names,
    mutations, notebook, reference, source, units,
)
from validate_prompt_guides import (
    STEP, FINGERPRINT_WORD, fenced_blocks, outside_fences, source_attachment_spans,
    subsection, instruction_text,
)


FIXED = re.compile(r"(?:SEED|RANDOM_STATE|VAL(?:IDATION)?_FRACTION|TEST_FRACTION|DATA.*|SOURCE|.*REVISION|.*POLICY|.*PRETRAIN.*|ALLOW_.*|RUN_.*|USE_NB_RATIO|(?:BOX|BBOX|COORD)_SCALE|PRIMARY_METRIC|DIRECTION|LABEL_NAMES|NUM_CLASSES|NUM_LABELS|.*_ID|.*_IDS)$")
DERIVED = {"MODEL_CONFIG", "PREPROCESS_CONFIG", "TOKENIZER_STATE", "SPLIT_INFO", "SPLIT_FINGERPRINTS"}
SETUP = {"REQUIRED_PACKAGES", "NEEDED_PACKAGES", "missing"}
FIELDS = re.compile(r"Chữ ký cần giữ:|Decorators cần giữ:|Giá trị cấu hình:|Thư viện và tên cần dùng:|Thao tác chính:|Điều kiện cần xử lý:|Đầu ra theo hợp đồng:|Biến cần cung cấp:|Constants cố định phải khai báo/cập nhật:")
HELPERS = {
    "sha": "Tạo hash ổn định của văn bản để nhận diện nhóm trùng lặp.",
    "document_key": "Tạo khóa tài liệu từ nội dung words và boxes để phát hiện tài liệu trùng giữa các split, thay vì chỉ dựa ID.",
    "read_jsonl": "Đọc UTF-8 JSONL, giữ thứ tự bản ghi và báo rõ lỗi thiếu file hoặc dữ liệu không hợp lệ.",
    "clean_text": "Chuẩn hóa Unicode NFC và khoảng trắng theo quy ước của dữ liệu; không làm mất nhãn hoặc references.",
    "normalize_text": "Chuẩn hóa văn bản theo quy ước nhãn của bài toán; dùng nhất quán khi xử lý dữ liệu và đánh giá.",
    "decode_utf8": "Giải mã chuỗi hoặc bytes từ nguồn dữ liệu thành văn bản UTF-8.",
    "open_rgb": "Đọc ảnh từ path, bytes hoặc PIL và chuyển RGB; báo rõ ảnh hỏng.",
    "to_device": "Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần.",
    "move_batch": "Chuyển batch sang device, giữ metadata và các tensor độ dài theo đúng API của model.",
    "make_loader": "Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0.",
    "restore_tokenizer": "Khôi phục đầy đủ tokenizer/vocabulary và special IDs từ trạng thái đã lưu; không fit lại khi reload.",
    "edit_distance": "Tính Levenshtein bằng quy hoạch động cho ký tự hoặc danh sách từ.",
    "ctc_decode": "Giải mã CTC: gộp nhãn lặp liền, blank đặt lại trạng thái lặp; giữ gold đầy đủ khi chấm CER/WER.",
    "classification_metrics": "Tính metric phân loại chính thức với đầy đủ nhãn và xử lý lớp không có dự đoán.",
    "lr_multiplier": "Tạo hệ số lịch learning rate theo số bước và warmup đã chọn trong context huấn luyện.",
}


def original_specs(content):
    """Read v2 prompts without reusing their scaffolding or AST call dumps."""
    masked = outside_fences(content)
    headings = list(STEP.finditer(masked))
    result = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        block = content[heading.end():end]
        targets = [REF.fullmatch(x) for x in fields(block, "Target")]
        prompt = fenced_blocks(subsection(block, "Prompt tạo hoặc sửa module"), "text")
        if not prompt:
            continue
        result.append(([(int(m[2]), int(m[3]), int(m[4])) for m in targets if m], prompt[0].split("\n\nContext ", 1)[0]))
    return result


def literal_configs(cells):
    configs = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse(source(cell)).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        configs.setdefault(target.id, node.value)
    return configs


def infrastructure(unit, configs):
    if unit["kind"] == "summary" or unit.get("owner"):
        return False
    if unit.get("symbol"):
        return unit["symbol"] == "package_versions"
    if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in unit["nodes"]):
        return True
    if unit["exports"] & SETUP:
        return True
    if unit["exports"] and unit["exports"] <= set(configs):
        return True
    if not unit["exports"] and any(word in unit["section"].lower() for word in ("cấu hình", "môi trường", "thư viện")):
        return True
    return False


def semantic_signature(unit):
    node = unit.get("function")
    if not node:
        return ""
    # Defaults that tune architecture/decoding come from the owning context.
    args = []
    defaults = [None] * (len(node.args.posonlyargs + node.args.args) - len(node.args.defaults)) + node.args.defaults
    for arg, default in zip(node.args.posonlyargs + node.args.args, defaults):
        suffix = ""
        if isinstance(default, ast.Constant) and isinstance(default.value, bool):
            suffix = "=" + repr(default.value)
        args.append(arg.arg + suffix)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    args.extend(a.arg for a in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return (unit.get("owner", "") + "." if unit.get("owner") else "") + node.name + "(" + ", ".join(args) + ")"


def prepare(cells):
    raw = units(cells)
    priority = {u["id"]: i for i, u in enumerate(dependencies(units(cells)))}
    configs = literal_configs(cells)
    kept = []
    old_to_new = {}
    removed_names = set()
    for old_index, unit in enumerate(raw):
        if infrastructure(unit, configs):
            removed_names.update(unit["exports"])
            continue
        unit = dict(unit, deps=set())
        # Imports that share a cell/block with task code belong in that module.
        unit["nodes"] = [n for n in unit["nodes"] if not isinstance(n, (ast.Import, ast.ImportFrom))]
        if not unit["nodes"] and unit["kind"] != "summary":
            removed_names.update(unit["exports"])
            continue
        if unit["kind"] == "code" and not unit.get("symbol"):
            # A model config and its instantiation may share a processing block.
            model_config_nodes = [n for n in unit["nodes"] if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_CONFIG" for t in n.targets)]
            unit["nodes"] = [n for n in unit["nodes"] if n not in model_config_nodes]
            if model_config_nodes:
                removed_names.add("MODEL_CONFIG")
            if not unit["nodes"]:
                continue
            unit["exports"] = set().union(*(defined(n) for n in unit["nodes"])) | mutations(unit["nodes"])
            unit["uses"] = free_names(unit["nodes"])
        unit["signature"] = semantic_signature(unit) or unit.get("signature", "")
        if not unit.get("symbol") and unit["exports"] & {"PREPROCESS_CONFIG", "TOKENIZER_STATE"}:
            # Old notebooks bundle decode settings into preprocessing metadata.
            # In new prompts, these are chosen with decoding, not before the model.
            unit["uses"] = unit["uses"] - {"BEAM_SIZE", "LENGTH_PENALTY", "MAX_DECODE_TOKENS"}
        if unit.get("symbol"):
            unit["title"] = unit["signature"] + (f" — phần {unit['part']}/{unit['parts']}" if unit.get("parts", 1) > 1 else "") if unit["kind"] == "code" else unit["title"]
        old_to_new[old_index] = len(kept)
        kept.append(unit)
    for unit in kept:
        if unit["kind"] == "summary":
            unit["class_parts"] = [old_to_new[j] for j in unit["class_parts"] if j in old_to_new]
    relocated = {name: value for name, value in configs.items() if name in removed_names}
    # Architecture config belongs with its cfg-taking constructor, not a new step.
    owners = {}
    constructors = [u for u in kept if u.get("symbol") == "__init__" and u.get("part", 1) == 1
                    and u.get("function") and any(a.arg == "cfg" for a in u["function"].args.args)]
    main_class = constructors[-1].get("owner") if constructors else None
    has_collate = any(u.get("symbol") == "collate_batch" for u in kept)
    # Argument types are also contracts: loss/optimizer/evaluate need the model
    # interface even when the concrete class name is absent from their AST.
    for unit in kept:
        fn = unit.get("function")
        args = {a.arg for a in fn.args.args} if fn else set()
        if main_class and "model" in args and not unit.get("owner"):
            unit["uses"] = unit["uses"] | {main_class}
        if has_collate and unit.get("symbol") != "collate_batch" and ("batch" in args or
                unit.get("owner") == main_class and bool(args & {"images", "source", "target"})):
            unit["uses"] = unit["uses"] | {"collate_batch"}
    if "MODEL_CONFIG" in relocated and constructors:
        owners["MODEL_CONFIG"] = constructors[-1]
    for unit in kept:
        if unit["kind"] != "code":
            continue
        unit["owns"] = set()
    # Resolve parameters recursively: a derived MODEL_CONFIG needs its vocabulary
    # provider and architecture parameters; those parameters share its owner.
    def assign(name, unit, visiting=frozenset()):
        if name not in relocated or name in visiting:
            return
        owner = owners.setdefault(name, unit)
        if name in owner["owns"]:
            return
        owner["owns"].add(name)
        owner["exports"].add(name)
        expr = relocated[name]
        refs = free_names([ast.Expr(value=expr)]) if expr else set()
        owner["uses"] = owner["uses"] | refs
        for ref in refs & set(relocated):
            assign(ref, owner, visiting | {name})
    for name, owner in list(owners.items()):
        assign(name, owner)
    for unit in kept:
        fn = unit.get("function")
        if not fn or unit.get("part", 1) != 1:
            continue
        positional = fn.args.posonlyargs + fn.args.args
        for arg, default in zip(positional[-len(fn.args.defaults):], fn.args.defaults):
            name = arg.arg.upper()
            if name in {"BEAM_SIZE", "LENGTH_PENALTY"} and name in relocated and name not in owners and isinstance(default, ast.Constant) and not isinstance(default.value, bool):
                assign(name, unit)
    for unit in sorted(kept, key=lambda u: priority[u["id"]]):
        if unit["kind"] == "code":
            for name in sorted(unit["uses"] & set(relocated)):
                assign(name, unit)
    for unit in kept:
        if unit["kind"] == "code":
            unit["config_uses"] = unit["uses"] & set(owners)
    ordered = dependencies(kept)
    return ordered, relocated, owners


def task_sections(content, configs):
    result = {}
    for label, next_label, key in (("Nguồn/giới hạn bài toán:", "Kiến trúc:", "data"),
                                  ("Kiến trúc:", "Đánh giá:", "model")):
        if label in content[:6500]:
            result[key] = content.split(label, 1)[1].split(next_label, 1)[0].split("\n\n", 1)[0].strip()
    match = re.search(r"Đánh giá:\s*([^\n]+)", content[:6500])
    if match:
        result["metrics"] = match[1]
    if "data" not in result:
        value = configs.get("SOURCE")
        if isinstance(value, ast.Dict):
            entries = dict(zip([k.value for k in value.keys if isinstance(k, ast.Constant)], value.values))
            details = [f"{key}: {ast.literal_eval(entries[key])}" for key in ("dataset", "tfds_builder", "revision") if key in entries and isinstance(entries[key], ast.Constant)]
            result["data"] = "Nguồn dữ liệu " + "; ".join(details) + ". Giữ split, ID, nhãn và references của nguồn."
    return result


def clean_goal(unit, specs):
    goals = []
    for refs, prompt in specs:
        if not any(c == unit["cell"] and a <= unit["stop"] and b >= unit["start"] for c, a, b in refs):
            continue
        marker = "thuộc phần " + unit["section"] + "."
        if marker in prompt:
            prompt = prompt.split(marker, 1)[1]
        prompt = FIELDS.split(prompt, 1)[0]
        prompt = re.split(r"Chỉ trả code|Đặt phần này trong class|Đây là phần \d+/", prompt, maxsplit=1)[0]
        prompt = re.sub(r"\b\w+(?:\.\w+)? nhận \([^)]*\)\.\s*", "", prompt)
        prompt = prompt.strip()
        sentences = re.split(r"(?<!\d)(?<=[.!?])\s+", prompt)
        sentences = [s.strip() for s in sentences if s.strip() and
                     not re.search(r"->|\b[A-Z][A-Z_]+\s*=|\[(?:B|N|S|T)[,\]]|\bcfg\s+\w+=", s)]
        prompt = " ".join(sentences[:3])
        if prompt and "Thiết lập imports" not in prompt and prompt not in goals:
            goals.append(prompt)
    return " ".join(goals)


def flexible_text(text):
    # Keep dataset versions, special token IDs, metrics and structural factors.
    # Replace only explicitly tunable named settings and tunable tensor widths.
    setting = r"(?:[A-Z][A-Z_]+|hidden_dim|embed_dim|embedding_dim|dropout|epochs|batch|patience|clip|beam_size|length_penalty|num_layers|layers|heads|ff_dim|lr|learning_rate|weight_decay|max_iter|min_df|max_features|tol|alpha)"
    text = re.sub(r"\b(" + setting + r")\s*[=:]\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", r"\1 theo cấu hình đã chọn", text)
    text = re.sub(r"\b(?:hai|ba|bốn|\d+) lớp\b", "số lớp được chọn", text)
    text = re.sub(r"\bsmoothing\s+\d+(?:\.\d+)?", "smoothing dương được chọn phù hợp", text)
    text = re.sub(r"\bcao\s*\d+\b", "chiều cao theo cấu hình", text)
    text = re.sub(r"\[((?=[^\]]*\b(?:B|N|S|T|H|W|C|V)\b)[^\]\n]*)\]", lambda m: "[" + re.sub(r"(?<!\w)(?:[89]|\d{2,})(?!\w)", "D_context", m[1]) + "]", text)
    return text


def algorithm_goal(unit):
    calls = {ast.unparse(n.func).split(".")[-1] for node in unit["nodes"] for n in ast.walk(node) if isinstance(n, ast.Call)}
    role = unit.get("symbol", "")
    if role == "document_key":
        return HELPERS[role]
    call_paths = {ast.unparse(n.func) for node in unit["nodes"] for n in ast.walk(node) if isinstance(n, ast.Call)}
    if any("SentencePieceTrainer" in name for name in call_paths):
        return "Fit SentencePiece BPE scratch với byte fallback chỉ trên train; giữ special IDs đã thống nhất, lưu tokenizer và trạng thái tiền xử lý để reload. PREPROCESS_CONFIG chỉ lưu thiết lập mã hóa/chuẩn hóa, không chứa cấu hình giải mã."
    if "char_vocab" in unit.get("exports", set()):
        return "Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá."
    if "word_vocab" in unit.get("exports", set()):
        return "Xây vocabulary từ train, giữ special IDs và ánh xạ nhãn; lưu đầy đủ trạng thái để dùng nhất quán khi reload."
    if "CTCLoss" in calls:
        return "Khởi tạo CTC loss với blank=0, reduction mean và zero_infinity=False; kiểm độ dài hợp lệ."
    if "FeatureUnion" in calls and "TfidfVectorizer" in calls:
        return "Kết hợp TF-IDF word và char_wb bằng FeatureUnion, chỉ fit train; tự chọn ngram_range và trọng số hai nhánh phù hợp."
    classifiers = calls & {"LinearSVC", "LogisticRegression", "MultinomialNB", "SGDClassifier", "ComplementNB"}
    if classifiers:
        return "Tạo classifier " + ", ".join(sorted(classifiers)) + " theo API đã chốt; kiểm hội tụ và dùng cấu hình lựa chọn trên validation."
    if role == "make_optimizer" or {"AdamW", "Adam", "SGD"} & calls:
        optimizers = ", ".join(sorted(calls & {"AdamW", "Adam", "SGD"})) or "phù hợp"
        groups = " Tách nhóm backbone và task head theo cấu trúc model." if "BACKBONE_LEARNING_RATE" in unit.get("owns", set()) else ""
        return "Khởi tạo optimizer " + optimizers + " cho tham số có thể học." + groups
    if "fit" in calls and any(isinstance(n, ast.For) for n in unit["nodes"]):
        return "Fit pipeline trên train theo các ứng viên siêu tham số được chọn; chọn bằng metric validation và lưu đủ biểu diễn, classifier và nhãn để reload. Không fit lại trên test hoặc train+validation."
    if any(isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch" for n in unit["nodes"]):
        return "Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume."
    if any(isinstance(n, ast.Call) and ast.unparse(n.func) in {"torch.load", "joblib.load"} for node in unit["nodes"] for n in ast.walk(node)):
        return "Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số."
    if "train_records" in unit.get("exports", set()) and not unit.get("symbol"):
        return "Chuẩn hóa các record và split theo giao thức nguồn, giữ đúng mẫu, ID, nhãn/references và quy tắc chia train/validation/test; kiểm nhóm trùng, chỉ fit biểu diễn trên train."
    return ""


def output_contract(unit):
    returns = [n.value for node in unit["nodes"] for n in ast.walk(node) if isinstance(n, ast.Return) and n.value]
    keys = sorted({str(k.value) for value in returns if isinstance(value, ast.Dict) for k in value.keys if isinstance(k, ast.Constant)})
    if keys:
        return "Kết quả giữ các trường " + ", ".join(keys) + "."
    tuples = [v for v in returns if isinstance(v, ast.Tuple)]
    if tuples:
        names = [v.id if isinstance(v, ast.Name) else "kết quả " + str(i) for i, v in enumerate(tuples[-1].elts, 1)]
        return "Giữ thứ tự đầu ra: " + ", ".join(names) + "."
    return ""


def special_ids(unit):
    values = []
    if unit.get("symbol"):
        return ""
    for node in unit["nodes"]:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PAD_ID", "UNK_ID", "BOS_ID", "EOS_ID", "BLANK_ID", "PAD", "UNK", "BOS", "EOS"}:
                    values.append(target.id + "=" + repr(node.value.value))
    return "Giữ special IDs: " + ", ".join(values) + "." if values else ""


def config_instructions(unit, configs):
    owned = sorted(unit.get("owns", ()))
    fixed, chosen, derived = [], [], []
    for name in owned:
        expr = configs[name]
        if name == "MODEL_CONFIG" and isinstance(expr, ast.Dict):
            derived.append("MODEL_CONFIG có các khóa " + ", ".join(str(k.value) for k in expr.keys if isinstance(k, ast.Constant)) + "; suy ra vocabulary/số lớp từ context dữ liệu")
        elif name == "SOURCE":
            derived.append("SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu")
        elif FIXED.fullmatch(name) and expr is not None:
            fixed.append(name + "=" + ast.unparse(expr))
        elif expr is not None and (any(isinstance(n, ast.Call) for n in ast.walk(expr)) or name in DERIVED or isinstance(expr, (ast.Name, ast.BinOp, ast.Attribute, ast.Dict))):
            derived.append(name + " được suy ra từ dữ liệu/context hoặc API tương ứng")
        else:
            chosen.append(name)
    parts = []
    if fixed:
        parts.append("Giữ ràng buộc: " + "; ".join(fixed) + ".")
    if chosen:
        parts.append("Tự chọn " + ", ".join(chosen) + " phù hợp và khai báo ngay trong module này; nêu ngắn lý do.")
    if derived:
        parts.append("Khai báo cùng phần này: " + "; ".join(derived) + ".")
    if unit.get("symbol") == "__init__" and "nn.Module" in unit.get("bases", ""):
        parts.append(("Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt" if not chosen else "Đặt") + " cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg.")
        parts.append("Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi.")
    inherited = sorted(unit.get("config_uses", set()) - set(owned))
    if inherited:
        parts.append("Dùng lại " + ", ".join(inherited) + " từ context, không chọn hoặc khai báo lại giá trị.")
    return " ".join(parts)


def context_slots(deps, by_id):
    return "\n\n".join(f"Context {dep} — {by_id[dep]['title']}:\n[Dán tóm tắt {dep}]" for dep in deps)


def render(directory, variant, old_content):
    if not re.search(r"^Format: [23]$", old_content, re.M):
        from migrate_prompt_guides import render_v2
        old_content = render_v2(directory, variant, old_content)
    cells = notebook(directory / "baseline.ipynb")
    items, configs, owners = prepare(cells)
    basic = units(notebook(directory / "baseline_basic.ipynb"))
    specs = original_specs(old_content)
    sections = task_sections(old_content, literal_configs(cells))
    consumers = {u["id"]: [] for u in items}
    for unit in items:
        for dep in unit["deps"]:
            consumers[dep].append(unit["id"])
    sinks = [u["id"] for u in items if not consumers[u["id"]]]
    items.append({"id": "integration", "title": "Kiểm tra kết nối các module", "kind": "integration", "deps": sinks})
    for dep in sinks:
        consumers[dep].append("integration")
    by_id = {u["id"]: u for u in items}
    result = [old_content.splitlines()[0], f"<!-- prompt-guide\nFormat: 3\nVariant: {variant}\nModel: gemini-3.8-flash\nThinking: low\nToken verification: unverified\nContext merge budget: {SUMMARY_BUDGET}\n-->",
              "Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi."]
    for number, unit in enumerate(items, 1):
        identifier, kind, deps = unit["id"], unit["kind"], unit["deps"]
        consumed = bool(consumers.get(identifier))
        sources = []
        if variant == "basic" and kind == "code":
            if unit.get("symbol"):
                candidates = [u for u in basic if u["kind"] == "code" and u.get("symbol") == unit["symbol"] and u.get("owner") == unit["owner"]]
                if candidates:
                    sources = [candidates[min(unit["part"] - 1, len(candidates) - 1)]]
            else:
                candidates = [u for u in basic if u["kind"] == "code" and not u.get("symbol") and unit["exports"] & u["exports"]]
                if candidates:
                    sources = [max(candidates, key=lambda u: len(unit["exports"] & u["exports"]))]
            if len(sources) == 1 and sources[0]["code"].strip() == unit["code"].strip() and not unit.get("owns") and not (unit.get("symbol") == "__init__" and "nn.Module" in unit.get("bases", "")):
                kind = "summary"
        metadata = [f"ID: {identifier}", "Depends: " + (", ".join(deps) or "none"), f"Kind: {kind}",
                    "Budget: " + (SUMMARY_BUDGET if kind == "summary" else BUDGET), "Summary: " + ("yes" if consumed else "no")]
        if kind == "summary" and sources:
            metadata.append("Summary input: code")
        if kind != "integration":
            metadata.append("Target: " + reference(unit))
        if consumed and kind != "summary":
            metadata.append("Summary budget: " + SUMMARY_BUDGET)
        if unit.get("owns"):
            metadata.append("Config owns: " + ", ".join(sorted(unit["owns"])))
        if unit.get("config_uses"):
            metadata.append("Config uses: " + ", ".join(sorted(unit["config_uses"])))
        metadata.extend("Source: " + reference(s, "baseline_basic.ipynb") for s in sources)
        result.append(f"## Bước {number}: {unit['title']}\n\n<!-- module\n" + "\n".join(metadata) + "\n-->")
        if deps:
            result.append("### Context cần chuẩn bị\n\nLần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:\n\n" +
                          "\n".join(f"{i}. [{by_id[d]['title']}](#context-{d})" for i, d in enumerate(deps, 1)))
        if kind == "summary" and sources:
            heading = "Prompt tóm tắt module"
            prompt = f"Tóm tắt giao diện {unit['title']} từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code."
        elif kind == "summary":
            heading = "Prompt gộp context"
            prompt = f"Gộp các context thành giao diện class {unit['symbol']} kế thừa {unit['signature'] or 'object'}. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số."
        elif kind == "integration":
            heading = "Prompt ghép và kiểm tra"
            prompt = "Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số."
        else:
            heading = "Prompt tạo hoặc sửa module"
            goal = algorithm_goal(unit) or clean_goal(unit, specs)
            if not goal:
                goal = HELPERS.get(unit.get("symbol"), "")
            role = unit.get("symbol", "")
            if not goal and role == "__init__":
                goal = "Khởi tạo các thành phần và trạng thái cần cho class."
            if not goal and role:
                goal = "Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context."
            if not goal and not role:
                if any(word in unit["section"].lower() for word in ("tải lại", "nạp lại", "held-out", "đánh giá test")):
                    goal = "Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test."
                elif "token" in unit["section"].lower() or "vocab" in unit["section"].lower():
                    goal = "Chuẩn bị mã hóa và trạng thái tokenizer từ train theo phương pháp đã chốt; giữ special IDs, masks và khả năng reload."
            if not role and {"train_loader", "val_loader"} & unit.get("exports", set()):
                goal = "Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá."
            if "data" in sections and (role in {"load_raw", "load_raw_data", "load_native", "load_raw_data", "load_raw", "load_native", "adapt", "adapt_split", "canonical_rows", "group_rows", "coco_records"} or "train_records" in unit.get("exports", set())):
                if sections["data"].rstrip(".") not in goal:
                    goal += " " + sections["data"]
            if role == "__init__" and "nn.Module" in unit.get("bases", ""):
                if sections.get("model", "").rstrip(".") not in goal:
                    goal += " " + sections.get("model", "")
            if role in {"score_corpus", "evaluate", "classification_metrics"}:
                if sections.get("metrics", "").rstrip(".") not in goal:
                    goal += " " + sections.get("metrics", "")
            if role == "__init__":
                goal = re.sub(r"^(?:SentencePiece|BPE|Tokenizer)[^,;]+[,;]\s*(?=BiGRU|BiLSTM|GRU|LSTM|Transformer|CNN)", "", goal)
            if goal and not goal.rstrip().endswith((".", "!", "?")):
                goal = goal.rstrip() + "."
            action = "Sửa hoặc bổ sung" if sources else "Viết"
            prompt = f"{action} {unit['title']}. " + flexible_text(goal) + " " + output_contract(unit) + " " + special_ids(unit) + " " + config_instructions(unit, configs)
            if unit.get("owner"):
                prompt += f" Viết method trong class {unit['owner']} kế thừa {unit.get('bases', 'object')}; giữ phần đã có."
            if unit.get("parts", 1) > 1:
                prompt += f" Đây là phần {unit['part']}/{unit['parts']} của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần."
            prompt += " Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context."
            if "nn.Module" in unit.get("bases", "") and role == "__init__" or "epoch" in unit.get("exports", set()):
                prompt += " Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch."
            prompt = re.sub(r" +", " ", prompt).strip()
        if deps:
            prompt += "\n\n" + context_slots(deps, by_id)
        anchor = f'<a id="context-{identifier}"></a>\n\n' if kind == "summary" and consumed else ""
        result.append(anchor + f"### {heading}\n\n```text\n{prompt}\n```")
        if sources:
            result.append("### Code gửi kèm\n\nDùng code hiện tại nếu đã chỉnh sửa:\n\n" +
                          "\n\n".join("```python\n" + s["code"] + "\n```" for s in sources))
        elif variant == "basic" and kind == "code":
            result.append("### Code gửi kèm\n\nModule bổ sung mới; không cần gửi code cũ.")
        if consumed and kind != "summary":
            summary = f'### Tóm tắt context để dùng tiếp\n\n<a id="context-{identifier}"></a>\n\n```text\nTóm tắt giao diện {unit["title"]} từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.'
            if deps:
                summary += "\n\n" + context_slots(deps, by_id)
            summary += f"\n```\n\n```python\n# Dán code hiện tại của {unit['title']} tại đây.\n```"
            result.append(summary)
    return remove_fingerprint_steps("\n\n".join(result) + "\n")


def remove_fingerprint_steps(content):
    """Project format-3 guides onto task steps, preserving attached source bytes."""
    masked = outside_fences(content)
    headings = list(STEP.finditer(masked))
    if not headings:
        return content
    steps = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        block = content[heading.end():end]
        meta = re.search(r"<!-- module\s*\n(.*?)-->", outside_fences(block), re.S)
        if not meta:
            raise ValueError("Cannot refresh a guide without module metadata")
        def value(key):
            matches = fields(meta[1], key)
            return matches[0] if matches else ""
        title = heading[0].split(": ", 1)[1]
        steps.append({"id": value("ID"), "kind": value("Kind"), "title": title,
                      "deps": [] if value("Depends") == "none" else value("Depends").split(", "),
                      "block": block, "heading": heading[0]})
    removed = {s["id"] for s in steps if FINGERPRINT_WORD.search(s["title"].split("(", 1)[0])
               and "(" in s["title"]}
    # A hash helper used exclusively by a removed audit is not another task step.
    # Hashes used for grouping records or duplicate checks retain their consumers.
    while True:
        orphaned = set()
        for step in steps:
            if step["id"] in removed or not re.match(r"(?:sha|sha256|stable_hash|hash_text)\(", step["title"]):
                continue
            consumers = [s for s in steps if s["id"] not in removed and s["kind"] != "integration" and step["id"] in s["deps"]]
            if not consumers:
                orphaned.add(step["id"])
        if not orphaned:
            break
        removed.update(orphaned)
    kept = [s for s in steps if s["id"] not in removed]
    renamed = {}
    for step in kept:
        if FINGERPRINT_WORD.search(step["title"]):
            renamed[step["title"]] = step["title"].split(" — ", 1)[0]
        step["deps"] = [d for d in step["deps"] if d not in removed]
    # Final coverage applies to remaining work, including a newly orphaned task
    # result; it must not reintroduce removed helper contexts.
    for step in kept:
        if step["kind"] == "integration":
            referenced = {d for s in kept if s["kind"] != "integration" for d in s["deps"]}
            step["deps"] = list(dict.fromkeys(step["deps"] + [s["id"] for s in kept
                                      if s["kind"] != "integration" and s["id"] not in referenced]))
    consumed = {d for s in kept for d in s["deps"]}
    header = content[:headings[0].start()]
    # Usage evidence for a removed session must not remain in a refreshed guide.
    header = re.sub(r"^Measurement: (.*)\n", lambda m: "" if any(f'\"{key}\"' in m[1] or f'\"{key}:summary\"' in m[1] for key in removed) else m[0], header, flags=re.M)
    result = [header.rstrip()]
    for number, step in enumerate(kept, 1):
        block = step["block"]
        protected = []
        for start, stop in reversed(source_attachment_spans(block)):
            token = f"\x00SOURCE_{len(protected)}\x00"
            protected.append((token, block[start:stop]))
            block = block[:start] + token + block[stop:]
        for old, new in sorted(renamed.items(), key=lambda pair: -len(pair[0])):
            block = block.replace(old, new)
        # Remove both dependency links and the corresponding named paste slots.
        block = re.sub(r"^\d+\. \[[^\n]*\]\(#context-([a-z][a-z0-9_-]*)\)\.?\n?",
                       lambda m: "" if m[1] in removed else m[0], block, flags=re.M)
        block = re.sub(r"\n\nContext ([a-z][a-z0-9_-]*)[^\n]*:\n(?:[^\n]*\n)*?\[Dán tóm tắt \1\]",
                       lambda m: "" if m[1] in removed else m[0], block)
        block = re.sub(r"^Depends: .*$", "Depends: " + (", ".join(step["deps"]) or "none"), block, flags=re.M)
        for field in ("Config owns", "Config uses"):
            block = re.sub(r"^" + field + r": (.*)\n", lambda m: (
                field + ": " + ", ".join(x for x in m[1].split(", ") if not FINGERPRINT_WORD.search(x)) + "\n"
                if any(not FINGERPRINT_WORD.search(x) for x in m[1].split(", ")) else ""), block, flags=re.M)
        block = block.replace(" và lưu fingerprint membership", "")
        # Generated schema descriptions exclude audit-only fields; original
        # dictionaries/statements in attached code remain untouched above.
        block = re.sub(r"Kết quả giữ các trường ([^.]+)\.", lambda m: "Kết quả giữ các trường " + ", ".join(
            x.strip() for x in m[1].split(",") if not FINGERPRINT_WORD.search(x)) + ".", block)
        if step["title"].startswith("document_key("):
            block = block.replace("Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context.", HELPERS["document_key"])
        # Legacy prose can describe the same audit in other words. Discard only
        # its sentence, not the containing module/data/schema requirements.
        def clean_paragraph(match):
            paragraph = match[0]
            if not FINGERPRINT_WORD.search(paragraph):
                return paragraph
            parts = re.split(r"(?<=[.!?])\s+", paragraph)
            return " ".join(p for p in parts if not FINGERPRINT_WORD.search(p))
        block = re.sub(r"[^\n]+", clean_paragraph, block)
        if not step["deps"]:
            masked_block = outside_fences(block)
            section = re.search(r"^### Context cần chuẩn bị[^\n]*\n", masked_block, re.M)
            if section:
                following = re.search(r"^(?:### |<a id=)", masked_block[section.end():], re.M)
                stop = section.end() + following.start() if following else len(block)
                block = block[:section.start()] + block[stop:]
        else:
            section = subsection(block, "Context cần chuẩn bị")
            counter = iter(range(1, len(step["deps"]) + 1))
            updated = re.sub(r"^\d+\. ", lambda m: str(next(counter)) + ". ", section, flags=re.M)
            block = block.replace(section, updated, 1)
        if step["id"] not in consumed:
            summary = subsection(block, "Tóm tắt context để dùng tiếp")
            if summary:
                block = block.replace("### Tóm tắt context để dùng tiếp" + summary, "", 1)
            block = re.sub(r"^Summary: yes$", "Summary: no", block, flags=re.M)
            block = re.sub(r"^Summary budget: .*\n", "", block, flags=re.M)
        for token, code in protected:
            block = block.replace(token, code)
        title = renamed.get(step["title"], step["title"])
        result.append(f"## Bước {number}: {title}" + block.rstrip())
    output = "\n\n".join(result) + "\n"
    if output != content:
        first_step = STEP.search(outside_fences(output))
        boundary = first_step.start() if first_step else len(output)
        revised_header = re.sub(r"^Token verification: measured$", "Token verification: unverified", output[:boundary], flags=re.M)
        revised_header = re.sub(r"^Measurement: .*\n", "", revised_header, flags=re.M)
        output = revised_header + output[boundary:]
    if FINGERPRINT_WORD.search(instruction_text(output)):
        raise ValueError("Fingerprint text remains outside an original source attachment")
    return output
