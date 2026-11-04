#!/usr/bin/env python3
"""Dependency-free structural/static validator for three NLP baseline variants.

Usage:
    python validate_artifacts.py path/to/baseline.ipynb path/to/huong_giai_quyet.md

Exit 0 = no static errors. This does not prove runtime correctness, metric
validity, or pretrained component eligibility; review model cards and dataflow.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPORT_HEADINGS = (
    "# Phân tích bài toán và đề xuất hướng giải quyết",
    "## 1. Giải thích bài toán",
    "## 2. Phân tích dữ liệu và các hướng tiếp cận",
    "## 3. Tiền xử lý dữ liệu tối ưu",
    "## 4. Lựa chọn mô hình tối ưu",
    "## 5. Siêu tham số và lý do lựa chọn",
)

METADATA_KEY = "nlp_baseline"
COMPONENT_POLICY = "image_components"

TASK_ROLES = {
    "text_only": set(),
    "image_captioning": {"image_encoder"},
    "ocr": {"image_encoder"},
    "table_extraction": {"image_encoder", "table_structure", "cell_ocr"},
    "image_to_text": {"image_encoder", "vision_component"},
}
COMPONENT_FIELDS = ("source", "model_id", "revision", "role", "input", "output", "training")
UNRESOLVED = object()


def pretraining_metadata(notebook: dict, errors: list[str], warnings: list[str]) -> dict:
    """Default old notebooks to scratch; malformed metadata never opens an exception."""
    fallback = {"pretraining_policy": "scratch", "task_type": "text_only", "pretrained_components": []}
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append("Notebook metadata must be an object")
        return fallback
    declaration = metadata.get(METADATA_KEY, {})
    if not isinstance(declaration, dict):
        errors.append(f"metadata.{METADATA_KEY} must be an object")
        return fallback
    policy = declaration.get("pretraining_policy", "scratch")
    task = declaration.get("task_type", "text_only")
    components = declaration.get("pretrained_components", [])
    if not isinstance(policy, str) or policy not in ("scratch", COMPONENT_POLICY):
        errors.append(f"pretraining_policy must be scratch or {COMPONENT_POLICY}")
        return fallback
    if not isinstance(task, str) or task not in TASK_ROLES:
        errors.append("task_type must be one of: " + ", ".join(TASK_ROLES))
        return fallback
    if not isinstance(components, list):
        errors.append("pretrained_components must be a list")
        return fallback
    if policy == "scratch":
        if components:
            errors.append("scratch policy cannot declare pretrained_components")
        return {**fallback, "task_type": task}
    if task == "text_only":
        errors.append("image_components policy cannot be used for text_only NLP")
    if not components:
        errors.append("image_components requires a nonempty pretrained_components list")
    valid = []
    seen = set()
    for index, component in enumerate(components, 1):
        prefix = f"Pretrained component {index}"
        if not isinstance(component, dict) or any(
            not isinstance(component.get(field), str) or not component[field].strip()
            for field in COMPONENT_FIELDS
        ):
            errors.append(f"{prefix}: require nonempty string fields: {', '.join(COMPONENT_FIELDS)}")
            continue
        if component["source"] not in ("huggingface", "torchvision"):
            errors.append(f"{prefix}: source must be huggingface or torchvision")
        if component["training"] not in ("frozen", "finetune"):
            errors.append(f"{prefix}: training must be frozen or finetune")
        if component["role"] not in TASK_ROLES[task]:
            errors.append(f"{prefix}: role {component['role']!r} is not allowed for {task}")
        if component["source"] == "torchvision" and component["role"] != "image_encoder":
            errors.append(f"{prefix}: torchvision ImageNet weights require image_encoder role")
        key = (component["source"], component["model_id"])
        if key in seen:
            errors.append(f"{prefix}: duplicate source/model_id declaration")
        seen.add(key)
        valid.append(component)
    warnings.append("Pretrained component eligibility requires model-card and actual pipeline review; "
                    "static checks cannot prove that a model solves only a subtask")
    return {"pretraining_policy": policy, "task_type": task, "pretrained_components": valid}


class PretrainingScanner(ast.NodeVisitor):
    """Conservative static checks only; never execute notebook code or resolve remote models."""

    def __init__(self, declaration: dict, errors: list[str], warnings: list[str]):
        self.declaration = declaration
        self.errors, self.warnings = errors, warnings
        self.aliases: dict[str, str] = {}
        self.values: dict[str, object] = {}
        self.bindings: dict[str, dict] = {}
        self.loaded: dict[int, dict] = {}
        self.stage = 0

    @property
    def scratch(self) -> bool:
        return self.declaration["pretraining_policy"] == "scratch"

    def error(self, message: str) -> None:
        self.errors.append(f"Cell {self.stage}: {message}")

    def warning(self, message: str) -> None:
        self.warnings.append(f"Cell {self.stage}: {message}")

    def name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.name(node.value)}.{node.attr}"
        return ""

    def value(self, node: ast.AST | None) -> object:
        if isinstance(node, ast.Name):
            return self.values.get(node.id, UNRESOLVED)
        if isinstance(node, ast.Dict):
            return {self.value(k): self.value(v) for k, v in zip(node.keys, node.values)
                    if isinstance(self.value(k), (str, int))}
        if isinstance(node, ast.Subscript):
            mapping, key = self.value(node.value), self.value(node.slice)
            if isinstance(mapping, dict) and isinstance(key, (str, int)):
                return mapping.get(key, UNRESOLVED)
        try:
            return ast.literal_eval(node) if node is not None else UNRESOLVED
        except (ValueError, TypeError):
            return UNRESOLVED

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            self.check_import(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.check_import(node.module or "")

    def check_import(self, module: str) -> None:
        if module.split(".")[0] == "sentence_transformers":
            self.error("SentenceTransformers pretrained text components are disallowed")
        if self.scratch and module.split(".")[0] == "transformers":
            self.error("Hugging Face transformers import is disallowed under scratch policy")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        component = next((self.loaded[id(child)] for child in ast.walk(node.value)
                          if id(child) in self.loaded), None)
        if isinstance(node.value, ast.Name):
            component = self.bindings.get(node.value.id, component)
        elif (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and node.value.func.attr in ("to", "eval", "train", "requires_grad_", "half", "float")):
            component = self.bindings.get(self.name(node.value.func.value), component)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.values[target.id] = self.value(node.value)
                resolved = self.name(node.value)
                self.aliases.pop(target.id, None)
                self.bindings.pop(target.id, None)
                if resolved:
                    self.aliases[target.id] = resolved
                if component:
                    self.bindings[target.id] = component

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

    def component(self, node: ast.Call, source: str, model_id: object) -> dict | None:
        if not isinstance(model_id, str):
            self.warning("Unresolved pretrained model ID; inspect dynamic loader and match it to component metadata")
            return None
        component = next((item for item in self.declaration["pretrained_components"]
                          if item["source"] == source and item["model_id"] == model_id), None)
        if component is None:
            self.error(f"Undeclared pretrained component: {source}:{model_id}")
        else:
            self.loaded[id(node)] = component
        return component

    def check_huggingface(self, node: ast.Call, api: str, model_id: object, pipeline: bool) -> None:
        if self.scratch:
            self.error("Pretrained model/tokenizer loader is disallowed under scratch policy")
            return
        component = self.component(node, "huggingface", model_id)
        cell_ocr = (self.declaration["task_type"] == "table_extraction"
                    and component is not None and component["role"] == "cell_ocr")
        # A known generating/text model is incompatible with an image-only role.
        # The same OCR generator can be valid as an intermediate table-cell stage.
        text_or_generator = bool(re.search(
            r"Tokenizer|LMHeadModel|ForCausalLM|ForSeq2SeqLM|ForConditionalGeneration|"
            r"VisionEncoderDecoderModel|ForVision2Seq|ForImageTextToText|"
            r"(?:Bert|Roberta|GPT2|T5|CLIP)Model$|(?:CLIP|Blip|TrOCR|Donut)Processor$", api
        ))
        if (pipeline or text_or_generator) and not cell_ocr:
            self.error("Pretrained text/full-task model or pipeline requires an eligible table cell_ocr role; "
                       "captioning and standalone OCR require an image encoder and scratch decoder/tokenizer")
        if component and api == "AutoProcessor" and not cell_ocr:
            self.warning("Verify AutoProcessor is image-only; pretrained text processing is not allowed for this role")

    def visit_Call(self, node: ast.Call) -> None:
        name = self.name(node.func)
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        api = name.split(".")[-1]
        if name in ("torch.hub.load", "huggingface_hub.InferenceClient", "huggingface_hub.AsyncInferenceClient"):
            self.error("torch.hub and hosted inference APIs are disallowed")
        if name in ("spacy.load", "fasttext.load_model"):
            self.error("Pretrained text model loader is disallowed")
        if api == "from_pretrained":
            if not name.startswith("transformers.") and not self.scratch:
                self.error("Only Hugging Face transformers components and torchvision ImageNet weights are allowed")
            else:
                model_id = self.value(node.args[0] if node.args else kwargs.get("pretrained_model_name_or_path"))
                owner = name.split(".")[-2] if "." in name else name
                self.check_huggingface(node, owner, model_id, False)
                component = self.loaded.get(id(node))
                if component:
                    revision = self.value(kwargs.get("revision"))
                    if revision is UNRESOLVED:
                        self.warning("Hugging Face loader has no statically pinned revision; verify checkpoint reproducibility")
                    elif revision != component["revision"]:
                        self.error("Hugging Face selected revision does not match component revision")
        elif name == "transformers.pipeline":
            model_id = self.value(kwargs.get("model") or (node.args[1] if len(node.args) > 1 else None))
            self.check_huggingface(node, "pipeline", model_id, True)

        weights = kwargs.get("weights")
        weight_value = self.value(weights)
        weight_name = self.name(weights) if weights is not None else ""
        legacy = self.value(kwargs.get("pretrained")) is True
        torchvision = name.startswith("torchvision.models.")
        selected_weights = weights is not None and (weight_value is not None) and (
            torchvision or "_Weights." in weight_name or weight_value == "DEFAULT"
        )
        if legacy or selected_weights:
            if self.scratch:
                self.error("Pretrained weights are disallowed under scratch policy")
            elif not torchvision:
                self.error("Pretrained vision weights must come from torchvision ImageNet builders")
            else:
                model_id = (self.value(node.args[0] if node.args else kwargs.get("name"))
                            if api == "get_model" else api)
                component = self.component(node, "torchvision", model_id)
                if component and "_Weights." in weight_name:
                    chosen = weight_name.rsplit(".", 2)[-2:]
                    if chosen[-1] != "DEFAULT" and ".".join(chosen) != component["revision"]:
                        self.error("torchvision selected weights do not match component revision")
                elif component and isinstance(weight_value, str) and weight_value != "DEFAULT":
                    if weight_value != component["revision"].rsplit(".", 1)[-1]:
                        self.error("torchvision selected weights do not match component revision")
                if legacy or weight_value == "DEFAULT" or weight_name.endswith(".DEFAULT"):
                    self.warning("Resolve DEFAULT/legacy pretrained selection to an exact ImageNet weight enum for reproduction")

        if isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
            component = self.bindings.get(self.name(node.func.value))
            if component and component["role"] != "cell_ocr":
                self.error("Text generation by a pretrained component is allowed only for table cell_ocr")
        self.generic_visit(node)


def source_text(source: object) -> str:
    if isinstance(source, list):
        return "".join(map(str, source))
    return str(source or "")



PREDICTION_TYPES = {"string", "integer", "spans", "entities", "classification", "detection",
                    "segmentation", "keypoints", "restoration", "video_classification", "tracking"}


def validate_predictions(rows: list, expected_ids: list, contract: dict) -> list[str]:
    """Validate actual inference results; no execution/download is performed by the CLI.

    For dense task outputs, the caller must additionally check actual array shapes,
    coordinate ranges and official serialization against the real dataset.
    """
    errors = []
    if not isinstance(contract, dict):
        return ["Prediction contract must be an object"]
    id_key, pred_key = contract.get("id_field"), contract.get("prediction_field")
    kind = contract.get("prediction_type")
    if not isinstance(id_key, str) or not isinstance(pred_key, str) or kind not in PREDICTION_TYPES:
        return ["Prediction contract has invalid fields/type"]
    if not isinstance(rows, list) or not isinstance(expected_ids, list) or not expected_ids:
        return ["Predictions and nonempty expected IDs must be lists"]
    if len(rows) != len(expected_ids):
        errors.append("Prediction count differs from inference sample count")
    actual_ids = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or id_key not in row or pred_key not in row:
            errors.append(f"Prediction {i}: missing ID or prediction field")
            continue
        actual_ids.append(row[id_key])
        value = row[pred_key]
        if kind == "string" and not isinstance(value, str):
            errors.append(f"Prediction {i}: expected string")
        elif kind in ("integer", "classification", "video_classification") and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"Prediction {i}: expected integer label")
        elif kind in ("spans", "entities"):
            if not isinstance(value, list):
                errors.append(f"Prediction {i}: expected a span/entity list")
            else:
                previous = -1
                for span in value:
                    start_key,end_key = ("start_word","end_word") if kind == "entities" and isinstance(span,dict) and "start_word" in span else ("start","end")
                    if not isinstance(span, dict) or not all(k in span for k in (start_key, end_key)):
                        errors.append(f"Prediction {i}: invalid span/entity object")
                        continue
                    a, b = span[start_key], span[end_key]
                    if (not isinstance(a,int) or isinstance(a,bool) or not isinstance(b,int) or isinstance(b,bool)
                            or a < 0 or b <= a or a < previous):
                        errors.append(f"Prediction {i}: invalid or unordered span offsets")
                    else:
                        previous = b
                    if not any(isinstance(span.get(k),str) for k in ("polarity","type","label")):
                        errors.append(f"Prediction {i}: span/entity lacks a string label")
        elif kind in ("detection", "tracking", "keypoints") and not isinstance(value, (list,dict)):
            errors.append(f"Prediction {i}: expected structured spatial/temporal output")
        elif kind in ("segmentation", "restoration") and not isinstance(value, (list,dict,str)):
            errors.append(f"Prediction {i}: expected array data or explicit output reference")
    if actual_ids != expected_ids:
        errors.append("Prediction IDs/order differ from inference IDs")
    if len({json.dumps(x,sort_keys=True) for x in actual_ids}) != len(actual_ids):
        errors.append("Prediction IDs are duplicated")
    return errors


def variant_metadata(notebook: dict, path: Path, errors: list[str]) -> tuple[str, dict]:
    metadata = notebook.get("metadata", {})
    raw = metadata.get(METADATA_KEY, {}) if isinstance(metadata,dict) else {}
    declaration = raw if isinstance(raw,dict) else {}
    expected = {"baseline.ipynb":"optimized", "baseline_basic.ipynb":"basic", "baseline_hardcore.ipynb":"hardcore"}.get(path.name)
    variant = declaration.get("variant", expected or "optimized")
    if variant not in ("optimized", "basic", "hardcore"):
        errors.append("variant must be optimized/basic/hardcore")
        return "optimized", declaration
    if expected and expected != variant:
        errors.append("Notebook filename and variant metadata disagree")
    if variant in ("basic","hardcore") and (declaration.get("pretraining_policy","scratch") != "scratch" or declaration.get("pretrained_components", [])):
        errors.append(f"{variant} requires scratch policy and no pretrained components")
    if variant == "hardcore":
        if declaration.get("variant") != "hardcore" or declaration.get("training_status") != "untrained":
            errors.append("Hardcore must explicitly declare variant=hardcore and training_status=untrained")
        contract = declaration.get("inference")
        if not isinstance(contract,dict):
            errors.append("Hardcore requires an inference contract")
        else:
            if contract.get("split") not in ("test","validation","val","held_out","held_out_val2014","testdev","public_test","private_test"):
                errors.append("Hardcore inference split must identify test or validation/held-out, never train")
            if any(not isinstance(contract.get(k),str) or not contract[k] for k in ("id_field","prediction_field")):
                errors.append("Hardcore requires nonempty inference ID/prediction fields")
            if contract.get("prediction_type") not in PREDICTION_TYPES:
                errors.append("Hardcore prediction_type is invalid")
    return variant, declaration


def scan_hardcore(trees: list[ast.AST], prose: list[str], errors: list[str], warnings: list[str]) -> None:
    # Conservative AST checks: real dataflow and actual output semantics still need a smoke test.
    combined = ast.Module(body=[node for tree in trees for node in tree.body],type_ignores=[])
    aliases = {"nn":"torch.nn", "torch":"torch"}
    for node in ast.walk(combined):
        if isinstance(node,ast.Import):
            for item in node.names: aliases[item.asname or item.name.split('.')[0]]=item.name
        if isinstance(node,ast.ImportFrom):
            for item in node.names: aliases[item.asname or item.name]=f"{node.module}.{item.name}"
    def name(node):
        if isinstance(node,ast.Name): return aliases.get(node.id,node.id)
        if isinstance(node,ast.Attribute): return name(node.value)+"."+node.attr
        return ""
    classes=[node for node in combined.body if isinstance(node,ast.ClassDef)]
    if len(classes)!=1 or not any(name(base)=="torch.nn.Module" for cls in classes for base in cls.bases):
        errors.append("Hardcore must define exactly one scratch nn.Module class")
    for node in combined.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and not node.name.startswith(("load_","read_","select_")):
            errors.append("Hardcore external helpers may only load/read/select raw data; preprocessing/training belongs to the learner")
    model_classes={cls.name for cls in classes}
    model_variables=set()
    for node in combined.body:
        if isinstance(node,ast.Assign) and any(isinstance(child,ast.Call) and name(child.func) in model_classes for child in ast.walk(node.value)):
            model_variables.update(target.id for target in node.targets if isinstance(target,ast.Name))
    if not any(isinstance(node,ast.Call) and (name(node.func) in model_variables or (isinstance(node.func,ast.Attribute) and node.func.attr=="forward" and name(node.func.value) in model_variables)) for node in ast.walk(combined)):
        errors.append("Hardcore must directly call the instantiated model for inference")
    for node in ast.walk(combined):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and (node.name in ("loss","train_epoch","training_step","configure_optimizers") or node.name.startswith(("train_","fit_")) or node.name.endswith("_loss")):
            errors.append("Hardcore cannot define training/loss helpers, even if unused")
    calls=[name(node.func) for node in ast.walk(combined) if isinstance(node,ast.Call)]
    for call in calls:
        api=call.rsplit('.',1)[-1]
        if (api in ("backward","fit","fit_transform","partial_fit","zero_grad","train","load_state_dict") or api.endswith("_loss")
            or call.startswith(("torch.optim.","torch.nn.functional.cross_entropy","torch.nn.functional.ctc_loss"))
            or re.search(r"(?:Loss|Scheduler|Scaler)$",api)
            or call in ("torch.save","torch.load","joblib.dump","joblib.load","pickle.dump","pickle.load")):
            errors.append(f"Hardcore forbids training/loss/checkpoint operations: {call}")
    for node in ast.walk(combined):
        if isinstance(node,(ast.For,ast.AsyncFor)) and any(isinstance(x,ast.Name) and 'epoch' in x.id.lower() for x in ast.walk(node.target)):
            errors.append("Hardcore cannot contain an epoch loop")
        if isinstance(node,ast.Call) and name(node.func).rsplit('.',1)[-1] in ('step','update') and isinstance(node.func,ast.Attribute):
            owner=name(node.func.value).lower()
            if any(x in owner for x in ('optim','schedul','scaler')):
                errors.append("Hardcore cannot update optimizer/scheduler/scaler state")
    if not any(c.endswith('.eval') for c in calls): errors.append("Hardcore must use model.eval()")
    if not any(c in ("torch.inference_mode","torch.no_grad") for c in calls): errors.append("Hardcore must disable gradients for inference")
    if not any(isinstance(n,ast.Assert) for n in ast.walk(combined)): errors.append("Hardcore must assert prediction count/ID/schema at runtime")
    headings=re.findall(r"^##\s+[12]\.","\n".join(prose),re.MULTILINE)
    if len(headings)!=2 or re.search(r"^##\s+[3-9]\.","\n".join(prose),re.MULTILINE):
        errors.append("Hardcore must have exactly two numbered content sections")
    warnings.append("Hardcore is untrained; review raw-input conversion, actual forward dataflow, split IDs and runtime output schema")


def scan_notebook(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Cannot read notebook JSON: {exc}"], warnings

    if not isinstance(notebook, dict):
        return ["Notebook JSON must be an object"], warnings
    variant, variant_declaration = variant_metadata(notebook, path, errors)
    declaration = pretraining_metadata(notebook, errors, warnings)
    scanner = PretrainingScanner(declaration, errors, warnings)
    if notebook.get("nbformat") != 4:
        errors.append("Notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return ["Notebook is missing a valid 'cells' list"], warnings
    sources: list[str] = []
    prose: list[str] = []
    code_cells = 0
    trees = []
    for index, cell in enumerate(cells):
        stage = index + 1
        if not isinstance(cell, dict):
            errors.append(f"Cell {stage} is not an object")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in ("code", "markdown"):
            errors.append(f"Cell {stage}: unsupported cell type {cell_type!r}; expected code or markdown")
            continue
        raw_source = cell.get("source")
        if not (isinstance(raw_source, str) or
                isinstance(raw_source, list) and all(isinstance(line, str) for line in raw_source)):
            errors.append(f"Cell {stage}: source must be a string or a list of strings")
            continue
        src = source_text(raw_source)
        if cell_type == "markdown":
            prose.append(src)
            continue  # Prose, code fences and API names here are not executable Python.
        code_cells += 1
        sources.append(src)
        if not src.strip():
            errors.append(f"Cell {stage}: code cell is empty")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            warnings.append(f"Cell {stage} contains execution output; consider clearing before delivery")

        code_lines = [line for line in src.splitlines() if not line.lstrip().startswith("#")]
        # Notebook magics/commands are valid in Jupyter but not parseable by ast.parse.
        has_ipython_magic = any(line.lstrip().startswith(("%", "!")) for line in code_lines)
        if has_ipython_magic:
            warnings.append(f"Cell {stage} uses Jupyter magic; command lines are not analyzed, prefer Python-compatible setup")
        parse_source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                                 for line in src.splitlines()) if has_ipython_magic else src
        try:
            tree = ast.parse(parse_source, filename=f"{path.name}:cell{stage}")
        except SyntaxError as exc:
            errors.append(f"Cell {stage}: Python syntax error line {exc.lineno}: {exc.msg}")
        else:
            trees.append(tree)
            scanner.stage = stage
            # Bindings survive cells; AST-node identities must not (Python may reuse IDs).
            scanner.loaded.clear()
            scanner.visit(tree)

    if not code_cells:
        errors.append("Notebook must contain at least one code cell")
    combined = "\n".join(sources)
    if not re.search(r"[à-ỹÀ-ỸđĐ]", "\n".join(prose + sources)):
        warnings.append("Notebook may lack Vietnamese explanations/comments/print messages")
    if not re.search(r"\b(?:cuda|torch\.cuda)\b", combined):
        warnings.append("GPU/CUDA detection not evident; verify the selected method and hardware")
    if variant != "hardcore" and not re.search(r"\b(?:torch\.save|joblib\.dump|pickle\.dump)\b", combined):
        warnings.append("Checkpoint persistence not evident; verify best-model saving")
    if variant != "hardcore" and not re.search(r"\b(?:torch\.load|joblib\.load|pickle\.load)\b", combined):
        warnings.append("Best-checkpoint reloading not evident")
    if not re.search(r"\b(?:val(?:idation)?|dev)\b", combined, re.IGNORECASE):
        warnings.append("Validation split or metric is not evident")
    if not re.search(r"\b(?:test|evaluation)\b", combined, re.IGNORECASE):
        warnings.append("Testing/evaluation logic is not evident")
    if variant == "basic":
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node,ast.Import) and any(alias.name.split('.')[0]=="sentencepiece" for alias in node.names):
                    errors.append("Basic requires simple word/character tokenization, not SentencePiece/BPE")
                if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0]=="sentencepiece":
                    errors.append("Basic cannot import SentencePiece")
                if isinstance(node,ast.Call):
                    api=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
                    if api in ("TfidfVectorizer","FeatureUnion","BPE","BpeTrainer","SentencePieceBPETokenizer","RandomHorizontalFlip","RandomVerticalFlip","RandomRotation","ColorJitter","RandomResizedCrop","RandAugment","AutoAugment"):
                        errors.append(f"Basic forbids optimized preprocessing: {api}")
    if variant == "hardcore":
        scan_hardcore(trees, prose, errors, warnings)
    return errors, warnings


def scan_report(path: Path) -> tuple[list[str], list[str]]:
    """Validate the five-part report structure, not scientific content/optimality."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read report: {exc}"], warnings

    # Code examples and comments are not report headings or explanatory prose.
    visible = []
    fence = None
    fence_length = 0
    for line in content.splitlines():
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if re.match(r"^ {0,3}" + re.escape(fence) + "{" + str(fence_length) + r",}[ \t]*$", line):
                fence = None
            continue
        if opening:
            fence, fence_length = opening.group(1)[0], len(opening.group(1))
            continue
        visible.append(line)
    plain = re.sub(r"<!--.*?-->", "", "\n".join(visible), flags=re.DOTALL)
    title_matches = list(re.finditer(r"^#[ \t]+[^#\n][^\n]*$", plain, flags=re.MULTILINE))
    if len(title_matches) != 1 or title_matches[0].group().strip() != REPORT_HEADINGS[0]:
        errors.append("Report must have exactly one required Vietnamese title")
    sections = list(re.finditer(r"^##[ \t]+[^#\n][^\n]*$", plain, flags=re.MULTILINE))
    actual = [match.group().strip() for match in sections]
    expected = list(REPORT_HEADINGS[1:])
    for heading in expected:
        count = actual.count(heading)
        if not count:
            errors.append(f"Missing required Markdown heading: {heading}")
        elif count > 1:
            errors.append(f"Duplicate required Markdown heading: {heading}")
    if actual != expected:
        errors.append("Report must contain exactly the five required sections in order; legacy nine-part/extra sections are not supported")
    if title_matches and sections and title_matches[0].start() > sections[0].start():
        errors.append("Report title must precede the five sections")
    for index, section in enumerate(sections):
        stop = sections[index + 1].start() if index + 1 < len(sections) else len(plain)
        body = plain[section.end():stop]
        body = re.sub(r"^#{3,6}[ \t]+[^\n]*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", body, flags=re.MULTILINE)
        if not body.strip():
            errors.append(f"Empty report section: {section.group().strip()}")
    if not re.search(r"[à-ỹÀ-ỸđĐ]", plain):
        errors.append("Markdown proposal is not visibly Vietnamese")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Jupyter baseline.ipynb")
    parser.add_argument("report", type=Path, help="Vietnamese huong_giai_quyet.md")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    for label, path, inspector in (
        ("Notebook", args.notebook, scan_notebook),
        ("Report", args.report, scan_report),
    ):
        if not path.is_file():
            errors.append(f"{label}: missing file {path}")
            continue
        e, w = inspector(path)
        errors.extend(f"{label}: {msg}" for msg in e)
        warnings.extend(f"{label}: {msg}" for msg in w)

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validation: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
