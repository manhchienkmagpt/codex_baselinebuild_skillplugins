"""Offline regression tests: python tests/test_validate_prompt_guides.py."""
import importlib.util
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid


SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts/validate_prompt_guides.py"
spec = importlib.util.spec_from_file_location("prompt_validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
sys.path.insert(0, str(SKILL / "scripts"))
import migrate_prompt_guides as migration
from prompt_modules import render as compact_render


def step(number, identifier, kind="code", depends="none", variant="basic"):
    source = "Source: baseline_basic.ipynb#cell=1; lines=1-2\n" if variant == "basic" and kind != "summary" else ""
    return f"""## Bước {number}: {identifier}
ID: {identifier}
Depends: {depends}
Kind: {kind}
Input: Prompt và bản tóm tắt interface của các bước phụ thuộc.
{source}Target: baseline.ipynb#cell=1; lines=1-2
Merge: Chèn hàm trả về tại vị trí được chỉ định và kiểm tra interface.
Budget: input=600; output=1100; reserve=300
Handoff: Giữ tên hàm, dtype và shape cho phiên mới.
### Prompt
Viết hàm xử lý theo interface đã tóm tắt, giữ nguyên kiểu đầu vào và đầu ra. Trả code hoàn chỉnh cùng tóm tắt chữ ngắn cho phiên sau.

"""


def guide(variant="basic"):
    return (f"# Hướng dẫn\nVariant: {variant}\nModel: gemini-3.8-flash\nThinking: low\n"
            "Token verification: unverified\n\nMỗi bước mở phiên mới; đếm input bằng Gemini trước khi gửi.\n\n"
            + step(1, "data", variant=variant)
            + step(2, "model", depends="data", variant=variant)
            + step(3, "summary", "summary", "data, model", variant)
            + step(4, "assemble", "integration", "summary", variant))


class PromptGuides(unittest.TestCase):
    def setUp(self):
        self.directory = Path.cwd().resolve() / (".prompt-test-" + uuid.uuid4().hex)
        self.directory.mkdir()
        value = {"cells": [{"cell_type": "code", "source": ["def transform(x):\n", "    return x\n"]}]}
        for name in ("baseline", "baseline_basic"):
            (self.directory / f"{name}.ipynb").write_text(json.dumps(value), encoding="utf-8")

    def tearDown(self):
        target = self.directory.resolve()
        if target.parent != Path.cwd().resolve() or not target.name.startswith(".prompt-test-"):
            raise RuntimeError("Unexpected cleanup path")
        shutil.rmtree(target)

    def scan(self, value=None, variant="basic"):
        path = self.directory / f"baseline_{variant}_promting.md"
        path.write_text(guide(variant) if value is None else value, encoding="utf-8")
        return validator.scan_guide(path, variant)

    def test_independent_sessions_and_multiple_dependencies(self):
        for variant in ("basic", "hardcore"):
            errors, warnings = self.scan(variant=variant)
            self.assertFalse(errors, errors)
            self.assertTrue(any("unverified" in warning for warning in warnings))

    def test_missing_and_empty_files(self):
        self.assertTrue(validator.scan_directory(self.directory)[0])
        self.assertTrue(self.scan("")[0])

    def test_selected_variant_does_not_require_other_guide(self):
        self.scan()
        self.assertFalse(validator.scan_directory(self.directory, ("basic",))[0])
        self.assertTrue(validator.scan_directory(self.directory)[0])

    def test_cli_selected_variant(self):
        self.scan(variant="hardcore")
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.directory), "--variant", "hardcore"],
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_basic_requires_resolvable_code_excerpt(self):
        for replacement in ("", "Source: baseline_basic.ipynb#cell=8; lines=1-2\n",
                            "Source: baseline_basic.ipynb#cell=1; lines=0-2\n",
                            "Source: ../baseline_basic.ipynb#cell=1; lines=1-2\n"):
            with self.subTest(replacement=replacement):
                self.assertTrue(self.scan(guide().replace("Source: baseline_basic.ipynb#cell=1; lines=1-2\n", replacement))[0])

    def test_reference_must_be_code_cell(self):
        path = self.directory / "baseline_basic.ipynb"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cells"][0]["cell_type"] = "markdown"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertTrue(self.scan()[0])

    def test_malformed_notebook_reference_is_diagnostic(self):
        (self.directory / "baseline_basic.ipynb").write_text('{"cells":[{"cell_type":"code","source":null}]}', encoding="utf-8")
        self.assertTrue(self.scan()[0])

    def test_hardcore_does_not_request_code(self):
        value = guide("hardcore").replace("Kind: code\n", "Kind: code\nSource: baseline_basic.ipynb#cell=1; lines=1-2\n", 1)
        self.assertTrue(self.scan(value, "hardcore")[0])
        value = guide("hardcore").replace("Input: Prompt", "Input: baseline_hardcore.ipynb#cell=1 và prompt", 1)
        self.assertTrue(self.scan(value, "hardcore")[0])

    def test_forward_unknown_and_duplicate_dependencies(self):
        for value in (guide().replace("Depends: data\n", "Depends: assemble\n", 1),
                      guide().replace("Depends: data\n", "Depends: unknown\n", 1),
                      guide().replace("Depends: data, model", "Depends: data, data"),
                      guide().replace("ID: model", "ID: data")):
            self.assertTrue(self.scan(value)[0])

    def test_no_implicit_memory_between_sessions(self):
        value = guide().replace("Input: Prompt và bản tóm tắt interface của các bước phụ thuộc.", "Input: Tiếp tục phiên trước.")
        self.assertTrue(self.scan(value)[0])

    def test_budget_overflow_zero_and_malformed(self):
        for budget in ("input=1000; output=1100; reserve=0", "input=0; output=1100; reserve=300", "2000"):
            value = guide().replace("input=600; output=1100; reserve=300", budget)
            self.assertTrue(self.scan(value)[0])

    def test_large_module_warns_to_count_and_split(self):
        long_source = {"cells": [{"cell_type": "code", "source": ["#" + "x" * 3000 + "\n", "result = 1\n"]}]}
        (self.directory / "baseline_basic.ipynb").write_text(json.dumps(long_source), encoding="utf-8")
        errors, warnings = self.scan()
        self.assertFalse(errors, errors)
        self.assertTrue(any("split" in warning for warning in warnings))

    def test_prompt_is_one_paragraph_without_markdown(self):
        for prompt in ("Viết hàm.\n\nTrả code.", "**Viết hàm**. Trả code.", "- Viết hàm. Trả code.",
                       "Viết `transform`. Trả code.", "Viết hàm.\n### Ghi chú\nTrả code."):
            value = guide().replace("Viết hàm xử lý theo interface đã tóm tắt, giữ nguyên kiểu đầu vào và đầu ra. Trả code hoàn chỉnh cùng tóm tắt chữ ngắn cho phiên sau.", prompt)
            self.assertTrue(self.scan(value)[0])

    def test_sentence_count_is_reviewed_without_miscounting_interfaces(self):
        value = guide().replace("Viết hàm xử lý theo interface đã tóm tắt, giữ nguyên kiểu đầu vào và đầu ra. Trả code hoàn chỉnh cùng tóm tắt chữ ngắn cho phiên sau.", "Viết hàm transform.")
        errors, warnings = self.scan(value)
        self.assertFalse(errors)
        self.assertTrue(any("sentence" in warning for warning in warnings))

    def test_summary_and_complete_integration_are_required(self):
        for value in (guide().replace("Kind: summary", "Kind: code"),
                      guide().replace("Kind: integration", "Kind: code"),
                      guide().replace("Depends: summary", "Depends: model")):
            self.assertTrue(self.scan(value)[0])

    def test_unavailable_counter_never_claims_verified_tokens(self):
        errors, warnings = self.scan()
        self.assertFalse(errors)
        self.assertTrue(any("not measured counts" in warning for warning in warnings))
        self.assertTrue(self.scan(guide().replace("Token verification: unverified", "Token verification: measured"))[0])

    def test_measured_evidence_includes_thinking_and_every_step(self):
        evidence = "\n".join("Measurement: " + json.dumps({"step": name, "model": "gemini-3.8-flash",
                              "method": "Gemini countTokens + usageMetadata", "input_tokens": 500,
                              "total_tokens": 1500}) for name in ("data", "model", "summary", "assemble"))
        value = guide().replace("Token verification: unverified", "Token verification: measured\n" + evidence)
        self.assertFalse(self.scan(value)[0])
        self.assertTrue(self.scan(value.replace('"total_tokens": 1500', '"total_tokens": 2100'))[0])
        self.assertTrue(self.scan(value.replace('"input_tokens": 500', '"input_tokens": 900'))[0])
        self.assertTrue(self.scan(value.replace('"model": "gemini-3.8-flash"', '"model": "other"'))[0])


class ModularGuides(unittest.TestCase):
    tearDown = PromptGuides.tearDown

    def setUp(self):
        PromptGuides.setUp(self)
        value = {"cells": [{"cell_type": "code", "source": [
            "import math\n", "SEED = 42\n", "def transform(x):\n",
            "    return math.sqrt(x) + SEED\n", "result = transform(4)\n"]}]}
        for name in ("baseline", "baseline_basic"):
            content = json.dumps(value)
            if name == "baseline_basic":
                content = content.replace("math.sqrt(x) + SEED", "x + SEED")
            (self.directory / f"{name}.ipynb").write_text(content, encoding="utf-8")

    def scan(self, value=None, variant="basic"):
        path = self.directory / f"baseline_{variant}_promting.md"
        content = migration.render_v2(self.directory, variant, guide(variant)) if value is None else value
        path.write_text(content, encoding="utf-8")
        return validator.scan_guide(path, variant)

    def rendered(self, variant="basic"):
        return migration.render_v2(self.directory, variant, guide(variant))

    def test_both_variants_have_resolvable_code_and_summary_context(self):
        for variant in ("basic", "hardcore"):
            errors, warnings = self.scan(variant=variant)
            self.assertFalse(errors, errors)
            self.assertTrue(any("unverified" in w for w in warnings))

    def test_context_slot_link_and_summary_are_required(self):
        value = self.rendered()
        for modified in (value.replace("[Dán tóm tắt m001]", "[Thiếu context]"),
                         value.replace("(#context-m001)", "(#missing)"),
                         value.replace('<a id="context-m001"></a>', ""),
                         value.replace("### Tóm tắt context để dùng tiếp", "### Thiếu prompt tóm tắt", 1)):
            with self.subTest(modified=modified[:50]):
                self.assertTrue(self.scan(modified)[0])

    def test_dependencies_cannot_be_unknown_or_forward(self):
        value = self.rendered().replace("Depends: m001", "Depends: unknown", 1)
        self.assertTrue(self.scan(value)[0])

    def test_headings_and_metadata_inside_fences_do_not_create_steps(self):
        value = self.rendered().replace("# Dán code hiện tại", "## Bước 99: fake\n<!-- module\nID: evil\n-->\n# Dán code hiện tại", 1)
        self.assertFalse(self.scan(value)[0])
        # Exercise the fence tokenizer separately: four-backtick examples can contain triples.
        block = "````text\n## Bước 77: example\n```python\nprint(1)\n```\n````\n"
        self.assertEqual(len(validator.fenced_blocks(block, "text")), 1)
        self.assertNotIn("Bước 77", validator.outside_fences(block))

    def test_basic_excerpts_are_real_and_cannot_silently_change(self):
        value = self.rendered().replace("return x + SEED\n```", "return x\n```", 1)
        self.assertTrue(self.scan(value)[0])
        value = self.rendered().replace("Source: baseline_basic.ipynb#cell=1", "Source: baseline_basic.ipynb#cell=99", 1)
        self.assertTrue(self.scan(value)[0])

    def test_hardcore_allows_summary_code_but_not_generation_attachments(self):
        value = self.rendered("hardcore")
        self.assertFalse(self.scan(value, "hardcore")[0])
        value = value.replace("### Ghép kết quả", "### Code gửi kèm\n\n```python\nx = 1\n```\n\n### Ghép kết quả", 1)
        self.assertTrue(self.scan(value, "hardcore")[0])
        value = self.rendered("hardcore").replace("Viết riêng module", "Đọc code gửi kèm và viết riêng module", 1)
        self.assertTrue(self.scan(value, "hardcore")[0])

    def test_each_summary_session_has_its_own_budget(self):
        value = self.rendered().replace("Summary budget: input=1200; output=500; reserve=300", "Summary budget: input=1500; output=900; reserve=300", 1)
        self.assertTrue(self.scan(value)[0])
        self.assertTrue(self.scan(self.rendered().replace("Token verification: unverified", "Token verification: measured"))[0])

    def test_measured_v2_requires_every_generation_and_separate_summary(self):
        import re
        value = self.rendered("hardcore")
        sessions = []
        for block in re.findall(r"<!-- module\n(.*?)-->", value, re.S):
            identifier = validator.field_values(block, "ID")[0]
            sessions.append(identifier)
            if validator.field_values(block, "Summary budget"):
                sessions.append(identifier + ":summary")
        evidence = "\n".join("Measurement: " + json.dumps({
            "step": session, "model": "gemini-3.8-flash",
            "method": "Gemini countTokens + usageMetadata",
            "input_tokens": 500, "total_tokens": 900}) for session in sessions + ["context-merge"])
        value = value.replace("Token verification: unverified", "Token verification: measured\n" + evidence)
        self.assertFalse(self.scan(value, "hardcore")[0])
        value = re.sub(r'^Measurement: .*?"step": "m001:summary".*\n', '', value, flags=re.M)
        self.assertTrue(self.scan(value, "hardcore")[0])

    def test_new_basic_modules_have_explicit_add_instructions(self):
        path = self.directory / 'baseline.ipynb'
        nb = json.loads(path.read_text(encoding='utf-8'))
        nb['cells'].append({'cell_type': 'code', 'source': 'def new_helper(x):\n    return x * 2\n'})
        path.write_text(json.dumps(nb), encoding='utf-8')
        value = self.rendered()
        self.assertIn('Module bổ sung mới', value)
        self.assertFalse(self.scan(value)[0])
        self.assertTrue(self.scan(value.replace('Module bổ sung mới', 'Không có code'))[0])

    def test_migration_is_opt_in_and_idempotent(self):
        for variant in ("basic", "hardcore"):
            (self.directory / f"baseline_{variant}_promting.md").write_text(guide(variant), encoding="utf-8")
        originals = {p: p.read_bytes() for p in self.directory.glob("*.md")}
        migration.migrate(self.directory)
        self.assertTrue(all(p.read_bytes() == value for p, value in originals.items()))
        self.assertEqual(migration.migrate(self.directory, write=True), 2)
        self.assertEqual(migration.migrate(self.directory, write=True), 0)

    def test_python_scope_and_mutated_state_drive_dependencies(self):
        tree = ast.parse("items = [encode(word) for word in records]\n")
        self.assertEqual(migration.defined(tree.body[0]), {"items"})
        self.assertEqual(migration.free_names(tree.body), {"encode", "records"})
        tree = ast.parse("native['train'] = [r for r in native['train'] if keep(r)]\n")
        self.assertEqual(migration.mutations(tree.body), {"native"})
        self.assertEqual(migration.free_names(tree.body), {"native", "keep"})
        cells = [{"cell_type": "code", "source": "native = {}\n"},
                 {"cell_type": "code", "source": "native['train'] = []\n"},
                 {"cell_type": "code", "source": "train_records = native['train']\n"}]
        items = migration.dependencies(migration.units(cells))
        self.assertEqual(items[1]["deps"], [items[0]["id"]])
        self.assertEqual(items[2]["deps"], [items[1]["id"]])


class CompactGuides(unittest.TestCase):
    setUp = ModularGuides.setUp
    tearDown = PromptGuides.tearDown

    def rendered(self, variant="hardcore"):
        old = migration.render_v2(self.directory, variant, guide(variant))
        return compact_render(self.directory, variant, old)

    def scan(self, value=None, variant="hardcore"):
        path = self.directory / f"baseline_{variant}_promting.md"
        path.write_text(self.rendered(variant) if value is None else value, encoding="utf-8")
        return validator.scan_guide(path, variant)

    def test_formats_1_2_and_3_remain_supported(self):
        for variant in ("basic", "hardcore"):
            for old in (guide(variant), migration.render_v2(self.directory, variant, guide(variant))):
                value = compact_render(self.directory, variant, old)
                self.assertIn("Format: 3", value)
                self.assertFalse(self.scan(value, variant)[0])

    def test_imports_and_global_setup_are_not_context_modules(self):
        import re
        value = self.rendered()
        headings = re.findall(r"^## Bước \d+: (.*)$", value, re.M)
        self.assertFalse(any(h.startswith("Imports") for h in headings))
        self.assertNotIn("### Ghép kết quả", value)
        self.assertNotIn("### Thông tin cần có", value)
        # SEED is declared with transform, without a synthetic imports/config dependency.
        self.assertIn("Config owns: SEED", value)
        self.assertNotIn("[Dán tóm tắt m001]", value)
        self.assertFalse(self.scan(value)[0])

    def test_imports_are_required_in_generated_responses(self):
        value = self.rendered().replace("imports cần dùng", "thành phần cần dùng")
        self.assertTrue(self.scan(value)[0])

    def test_compact_summary_links_are_still_validated(self):
        value = self.rendered().replace("(#context-", "(#missing-")
        self.assertTrue(self.scan(value)[0])

    def test_config_ownership_reaches_training_and_reload(self):
        import re
        import prompt_modules
        nb = {"cells": [{"cell_type": "code", "source": code} for code in [
            "from torch import nn\nimport torch\nHIDDEN_DIM=256\nDROPOUT=0.3\nLEARNING_RATE=0.001\n",
            "class Model(nn.Module):\n    def __init__(self,cfg):\n        super().__init__()\n        self.head=nn.Linear(cfg['hidden_dim'],1)\n    def forward(self,x):\n        return self.head(x)\n",
            "MODEL_CONFIG={'hidden_dim':HIDDEN_DIM,'dropout':DROPOUT}\n",
            "model=Model(MODEL_CONFIG)\noptimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)\n",
            "best_model=Model(MODEL_CONFIG)\nbest_model.load_state_dict(model.state_dict())\n"]]}
        for name in ('baseline', 'baseline_basic'):
            (self.directory / f'{name}.ipynb').write_text(json.dumps(nb), encoding='utf-8')
        value = self.rendered()
        self.assertFalse(self.scan(value)[0])
        self.assertIn('Config owns: DROPOUT, HIDDEN_DIM, MODEL_CONFIG', value)
        self.assertNotIn('HIDDEN_DIM=256', value)
        self.assertNotIn('LEARNING_RATE=0.001', value)
        items, configs, owners = prompt_modules.prepare(nb['cells'])
        self.assertEqual(owners['HIDDEN_DIM']['id'], owners['MODEL_CONFIG']['id'])
        alternate = json.loads(json.dumps(nb))
        alternate['cells'][0]['source'] = alternate['cells'][0]['source'].replace('256', '192').replace('0.3', '0.1')
        (self.directory / 'baseline.ipynb').write_text(json.dumps(alternate), encoding='utf-8')
        # Tuning defaults in a reference must not constrain the new prompts.
        self.assertEqual(value, self.rendered())
        invalid = value.replace('Config owns: LEARNING_RATE', 'Config owns: HIDDEN_DIM, LEARNING_RATE')
        self.assertTrue(self.scan(invalid)[0])

    def test_format_2_migration_is_applied_once(self):
        for variant in ('basic', 'hardcore'):
            path = self.directory / f'baseline_{variant}_promting.md'
            path.write_text(migration.render_v2(self.directory, variant, guide(variant)), encoding='utf-8')
        self.assertEqual(migration.migrate(self.directory, write=True), 2)
        self.assertEqual(migration.migrate(self.directory, write=True), 0)

    def test_loss_inherits_model_and_batch_contracts(self):
        import prompt_modules
        cells = [{'cell_type': 'code', 'source': code} for code in [
            "from torch import nn\nHIDDEN_DIM=256\n",
            "def collate_batch(items):\n    return {'x':items}\n",
            "class Model(nn.Module):\n    def __init__(self,cfg):\n        self.head=nn.Linear(cfg['hidden_dim'],1)\n    def forward(self,batch):\n        return self.head(batch['x'])\n",
            "MODEL_CONFIG={'hidden_dim':HIDDEN_DIM}\n",
            "def compute_loss(model,batch):\n    return model(batch).sum()\n"]]
        items, _, _ = prompt_modules.prepare(cells)
        loss = next(u for u in items if u.get('symbol') == 'compute_loss')
        model = next(u for u in items if u['kind'] == 'summary' and u.get('symbol') == 'Model')
        batch = next(u for u in items if u.get('symbol') == 'collate_batch')
        self.assertIn(model['id'], loss['deps'])
        self.assertIn(batch['id'], loss['deps'])

    def test_decoder_choices_do_not_create_a_tokenizer_cycle(self):
        import prompt_modules
        cells = [{'cell_type': 'code', 'source': code} for code in [
            "from torch import nn\nHIDDEN_DIM=256\nBEAM_SIZE=3\nLENGTH_PENALTY=0.6\n",
            "vocab=['pad','a']\nTOKENIZER_STATE={'vocab':vocab}\nPREPROCESS_CONFIG={'beam_size':BEAM_SIZE,'length_penalty':LENGTH_PENALTY}\n",
            "class Model(nn.Module):\n    def __init__(self,cfg):\n        self.head=nn.Linear(cfg['hidden_dim'],len(vocab))\n    def generate(self,source,beam_size=3,length_penalty=0.6):\n        return source\n",
            "MODEL_CONFIG={'hidden_dim':HIDDEN_DIM}\n",
            "def evaluate(model,source):\n    return model.generate(source,beam_size=BEAM_SIZE,length_penalty=LENGTH_PENALTY)\n"]]
        items, _, owners = prompt_modules.prepare(cells)
        tokenizer = next(u for u in items if 'PREPROCESS_CONFIG' in u['exports'])
        self.assertEqual(owners['BEAM_SIZE']['symbol'], 'generate')
        self.assertNotIn('BEAM_SIZE', tokenizer.get('config_uses', set()))
        self.assertNotIn(owners['BEAM_SIZE']['id'], tokenizer['deps'])

    def test_tuning_does_not_change_special_ids_or_box_coordinates(self):
        import prompt_modules
        text = prompt_modules.flexible_text('bbox trong [0,1000]; hidden [B,T,256]; hidden_dim=256')
        self.assertIn('[0,1000]', text)
        self.assertNotIn('[B,T,256]', text)
        self.assertNotIn('hidden_dim=256', text)
        unit = {'symbol': '', 'nodes': ast.parse('BLANK_ID=0\nUNK_ID=1\n').body}
        self.assertIn('BLANK_ID=0', prompt_modules.special_ids(unit))
        self.assertIn('UNK_ID=1', prompt_modules.special_ids(unit))
        settings = {'USE_NB_RATIO': ast.Constant(value=True), 'NB_RATIO_ALPHA': ast.Constant(value=1.0)}
        unit = {'owns': set(settings), 'nodes': [], 'symbol': ''}
        instruction = prompt_modules.config_instructions(unit, settings)
        self.assertIn('USE_NB_RATIO=True', instruction)
        self.assertNotIn('NB_RATIO_ALPHA=1', instruction)


class FingerprintPolicy(unittest.TestCase):
    tearDown = PromptGuides.tearDown

    def setUp(self):
        from unittest.mock import patch
        PromptGuides.setUp(self)
        self.notebook = {"cells": [{"cell_type": "code", "source": code} for code in [
            "import hashlib\nimport json\n",
            "def sha(text):\n    return hashlib.sha256(text.encode()).hexdigest()\n",
            "def fingerprint(rows):\n    return sha(json.dumps(rows,sort_keys=True))\n",
            "def group_rows(rows):\n    return [{'text':r['text'],'key':sha(r['text'])} for r in rows]\n",
            "def document_key(row):\n    return fingerprint({'words':row['words'],'boxes':row['boxes']})\n",
            "records=[{'id':'one','text':'same','words':['same'],'boxes':[[0,0,1,1]]}]\ntrain_records=group_rows(records)\nval_records=[]\ntest_records=[]\n",
            "SPLIT_FINGERPRINTS={'train':fingerprint(train_records)}\n",
            "keys={document_key(r) for r in records}\nresults={'split_fingerprints':SPLIT_FINGERPRINTS,'count':len(keys)}\n"]]}
        self.write_notebooks()
        self.originals = {}
        # Simulate existing format-3 guides authored before the new policy.
        with patch('prompt_modules.remove_fingerprint_steps', side_effect=lambda content: content):
            for variant in ('basic', 'hardcore'):
                self.originals[variant] = compact_render(self.directory, variant, guide(variant))

    def write_notebooks(self):
        for name in ('baseline', 'baseline_basic'):
            (self.directory / f'{name}.ipynb').write_text(json.dumps(self.notebook), encoding='utf-8')

    def refreshed(self, variant='basic'):
        import prompt_modules
        return prompt_modules.remove_fingerprint_steps(self.originals[variant])

    def scan(self, value, variant='basic'):
        path = self.directory / f'baseline_{variant}_promting.md'
        path.write_text(value, encoding='utf-8')
        return validator.scan_guide(path, variant)

    def test_sources_stay_exact_while_prompts_and_dependencies_are_clean(self):
        import re
        for variant in ('basic', 'hardcore'):
            before, after = self.originals[variant], self.refreshed(variant)
            self.assertFalse(validator.FINGERPRINT_WORD.search(validator.instruction_text(after)))
            self.assertFalse(self.scan(after, variant)[0])
            def sources_by_id(content):
                headings = list(validator.STEP.finditer(validator.outside_fences(content)))
                found = {}
                for i,h in enumerate(headings):
                    block = content[h.end():headings[i+1].start() if i+1<len(headings) else len(content)]
                    found[validator.field_values(block, 'ID')[0]] = validator.fenced_blocks(validator.subsection(block,'Code gửi kèm'),'python')
                return found
            old_sources, new_sources = sources_by_id(before), sources_by_id(after)
            for identifier, code in new_sources.items():
                self.assertEqual(code, old_sources[identifier])
            if variant == 'basic':
                self.assertTrue(any('fingerprint(' in c for blocks in new_sources.values() for c in blocks))

    def test_deduplication_helpers_and_their_context_are_kept(self):
        value = self.refreshed('hardcore')
        self.assertIn('sha(text)', value)
        self.assertIn('group_rows(rows)', value)
        self.assertIn('document_key(row)', value)
        self.assertIn('nội dung words và boxes', value)
        self.assertFalse(self.scan(value, 'hardcore')[0])

    def test_hash_used_only_for_removed_audit_is_not_a_separate_step(self):
        from unittest.mock import patch
        import prompt_modules
        self.notebook['cells'][3]['source'] = 'def group_rows(rows):\n    return rows\n'
        self.write_notebooks()
        with patch('prompt_modules.remove_fingerprint_steps', side_effect=lambda content: content):
            old = compact_render(self.directory, 'hardcore', guide('hardcore'))
        value = prompt_modules.remove_fingerprint_steps(old)
        self.assertNotIn('sha(text)', value)
        self.assertIn('document_key(row)', value)
        self.assertFalse(self.scan(value, 'hardcore')[0])

    def test_validator_exempts_only_actual_basic_attachments(self):
        value = self.refreshed()
        self.assertFalse(self.scan(value)[0])
        value = value.replace('### Prompt tóm tắt module', 'fingerprint\n\n### Prompt tóm tắt module', 1)
        self.assertTrue(self.scan(value)[0])
        value = self.refreshed('hardcore').replace('# Dán code hiện tại', '# fingerprint\n# Dán code hiện tại', 1)
        self.assertTrue(self.scan(value, 'hardcore')[0])

    def test_refresh_previews_writes_and_is_idempotent(self):
        for variant, content in self.originals.items():
            (self.directory / f'baseline_{variant}_promting.md').write_text(content, encoding='utf-8')
        self.assertEqual(migration.migrate(self.directory, refresh=True), 2)
        for variant, content in self.originals.items():
            self.assertEqual((self.directory / f'baseline_{variant}_promting.md').read_text(encoding='utf-8'), content)
        self.assertEqual(migration.migrate(self.directory, write=True, refresh=True), 2)
        self.assertEqual(migration.migrate(self.directory, write=True, refresh=True), 0)

    def test_changed_prompts_invalidate_old_token_measurements(self):
        import prompt_modules
        old = self.originals['hardcore'].replace('Token verification: unverified',
              'Token verification: measured\nMeasurement: {"step":"old-session","input_tokens":123}')
        value = prompt_modules.remove_fingerprint_steps(old)
        self.assertIn('Token verification: unverified', value)
        self.assertNotIn('Measurement:', value)
        self.assertEqual(value, prompt_modules.remove_fingerprint_steps(value))


class PackagedGuides(unittest.TestCase):
    def test_examples_resolve_actual_notebook_cells_and_cover_dependencies(self):
        domain = SKILL.name.split("-", 1)[0]
        examples = SKILL.parent / f"{domain}-notebook-engineer/assets/fewshot"
        directories = sorted(path for path in examples.iterdir() if path.is_dir())
        self.assertTrue(directories)
        for directory in directories:
            with self.subTest(task=directory.name):
                errors, warnings = validator.scan_directory(directory)
                self.assertFalse(errors, errors)
                # Without live Gemini measurements the examples must remain explicitly unverified.
                self.assertEqual(sum("usage is unverified" in warning for warning in warnings), 2)


if __name__ == "__main__":
    unittest.main()
