"""Behavioral checks for responsibility-based guides; no model/network calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

import test_validate_prompt_guides as legacy

guide = legacy.guide

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import focused_prompts as focused
import migrate_prompt_guides as migration
import validate_prompt_guides as validator


class FocusedGuides(unittest.TestCase):
    setUp = legacy.ModularGuides.setUp
    tearDown = legacy.PromptGuides.tearDown

    def render(self, variant="basic"):
        return migration.render(self.directory, variant, guide(variant))

    def scan(self, content, variant="basic"):
        return validator.scan_focused_guide(self.directory / f"baseline_{variant}_promting.md", variant, content)

    def measured(self, content, summary_input=300, summary_output=50):
        records = []
        for meta in re.findall(r"<!-- module\n(.*?)-->", content, re.S):
            identifier = validator.field_values(meta, "ID")[0]
            records.append((identifier, 200, 700))  # Code output may exceed input.
            if validator.field_values(meta, "Summary") == ["yes"]:
                records.append((identifier + ":summary", summary_input, summary_input + summary_output))
            if validator.field_values(meta, "Constraints") == ["yes"]:
                records.append((identifier + ":constraints", 300, 500))
        evidence = "\n".join("Measurement: " + json.dumps({
            "step": name, "model": "gemini-3.8-flash", "method": "chatbot token counters",
            "input_tokens": inp, "total_tokens": total}) for name, inp, total in records)
        return content.replace("Token verification: unverified", "Token verification: measured\n" + evidence)

    def test_basic_reuses_training_and_hardcore_adds_it(self):
        for variant in ("basic", "hardcore"):
            value = self.render(variant)
            errors, warnings = self.scan(value, variant)
            self.assertFalse(errors, errors)
            self.assertTrue(any("unverified" in w for w in warnings))
            groups = validator.field_values(validator.outside_fences(value), "Group")
            self.assertEqual(tuple(dict.fromkeys(groups)), focused.GROUPS[variant])
            self.assertEqual(groups.count("training"), int(variant == "hardcore"))
            if variant == "hardcore":
                self.assertNotIn("Source:", value)

    def test_no_output_input_ratio_rule_for_code(self):
        value = self.render("hardcore")
        self.assertIn("input=650; output=950", value)
        errors, _ = self.scan(self.measured(value), "hardcore")
        self.assertFalse(errors, errors)

    def test_summary_preserves_information_without_a_length_or_ratio_rule(self):
        value = self.render("hardcore")
        # A complete summary can exceed 80 tokens and its actual input.
        self.assertFalse(self.scan(self.measured(value, 100, 150), "hardcore")[0])
        self.assertFalse(self.scan(self.measured(value, 300, 500), "hardcore")[0])
        self.assertTrue(self.scan(self.measured(value, 300, 501), "hardcore")[0])
        measured = self.measured(value)
        incomplete = re.sub(r"^Measurement: .*:summary.*\n", "", measured, count=1, flags=re.M)
        self.assertTrue(self.scan(incomplete, "hardcore")[0])

    def test_budget_target_and_platform_limit_are_distinct(self):
        value = self.render("hardcore")
        for changed in (
            value.replace("input=650; output=950; reserve=200", "input=650; output=951; reserve=200", 1),
            value.replace("input=1100; output=500; reserve=200", "input=1100; output=501; reserve=200", 1),
            self.measured(value).replace('"total_tokens": 700', '"total_tokens": 2000', 1),
        ):
            self.assertTrue(self.scan(changed, "hardcore")[0])

    def add_editable_source(self):
        for name in ("baseline", "baseline_basic"):
            path = self.directory / f"{name}.ipynb"
            nb = json.loads(path.read_text(encoding="utf-8"))
            nb["cells"].append({"cell_type": "code", "source":
                "def normalize_text(value):\n    return value.strip()\n"
                "normalized = normalize_text('example')\n"})
            path.write_text(json.dumps(nb), encoding="utf-8")

    def test_existing_edits_extract_signatures_and_constraints_before_generation(self):
        self.add_editable_source()
        value = self.render()
        self.assertFalse(self.scan(value)[0])
        self.assertIn("Related source:", value)
        self.assertIn("normalized = normalize_text('example')", value)
        self.assertNotIn("[phần cần sửa]", value)
        for changed in (
            value.replace("Rút ra tên/chữ ký cần giữ", "Chỉ rút ràng buộc", 1),
            value.replace(focused.CONSTRAINT_SLOT, "[Thiếu kết quả trích xuất]", 1),
            value.replace("(#constraints-s01)", "(#missing)", 1),
            value.replace("Constraints: yes", "Constraints: no", 1),
            value.replace("normalized = normalize_text('example')", "normalized = None", 1),
        ):
            self.assertTrue(self.scan(changed)[0])

    def test_new_modules_use_context_without_extraction(self):
        for variant in ("basic", "hardcore"):
            value = self.render(variant)
            self.assertFalse(self.scan(value, variant)[0])
            self.assertNotIn("### Prompt trích ràng buộc", value)
            self.assertNotIn(focused.CONSTRAINT_SLOT, value)
            self.assertNotIn("Constraints: yes", value)

    def test_older_format_four_guides_remain_readable(self):
        value = self.render("hardcore").replace("Constraint workflow: extract-before-edit\n", "").replace("Reference workflow: report-first\n", "")
        self.assertFalse(self.scan(value, "hardcore")[0])

    def test_extraction_sessions_need_their_own_budget_and_measured_evidence(self):
        self.add_editable_source()
        value = self.render()
        measured = self.measured(value)
        self.assertFalse(self.scan(measured)[0])
        incomplete = re.sub(r"^Measurement: .*:constraints.*\n", "", measured, count=1, flags=re.M)
        self.assertTrue(self.scan(incomplete)[0])
        self.assertTrue(self.scan(value.replace("Constraints budget:", "Missing budget:", 1))[0])

    def test_task_constraints_beyond_the_old_prose_cap_are_retained(self):
        self.add_editable_source()
        report = "## 3. Tiền xử lí\n\n" + "Chuẩn hóa văn bản bằng NFC. " * 45
        report += "CTC cần đủ số bước cho target và các cặp ký tự lặp.\n"
        (self.directory / "huong_giai_quyet.md").write_text(report, encoding="utf-8")
        value = self.render()
        self.assertIn("CTC cần đủ số bước cho target và các cặp ký tự lặp.", value)
        self.assertFalse(self.scan(value)[0])

    def test_missing_context_and_duplicate_config_owner_are_rejected(self):
        value = self.render("hardcore")
        for changed in (
            value.replace("(#context-s01)", "(#missing)", 1),
            value.replace("[Dán tóm tắt s01]", "[Thiếu context]", 1),
            value.replace("Depends: s01", "Depends: unknown", 1),
            value.replace("Config owns: architecture", "Config owns: splits", 1),
            value.replace("Group: model", "Group: reload", 1),
            value.replace('<a id="context-s01"></a>', "", 1),
        ):
            self.assertTrue(self.scan(changed, "hardcore")[0])

    def test_source_excerpts_and_fence_scopes_are_checked(self):
        # Give the model a real basic source attachment.
        path = self.directory / "baseline_basic.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"].append({"cell_type": "markdown", "source": "## Mô hình"})
        nb["cells"].append({"cell_type": "code", "source": "def make_classifier(c):\n    return c\n"})
        path.write_text(json.dumps(nb), encoding="utf-8")
        value = self.render()
        self.assertFalse(self.scan(value)[0])
        self.assertTrue(self.scan(value.replace("    return c\n```", "    return c + 1\n```"))[0])
        scoped = value.replace("# Dán code hiện tại", "## Bước 99: example\n<!-- module\nID: false\n-->\n# Dán code hiện tại", 1)
        self.assertFalse(self.scan(scoped)[0])

    def test_preview_refresh_and_notebook_preservation(self):
        for variant in ("basic", "hardcore"):
            (self.directory / f"baseline_{variant}_promting.md").write_text(guide(variant), encoding="utf-8")
        protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.directory.glob("*.ipynb")}
        original = {p: p.read_bytes() for p in self.directory.glob("*.md")}
        self.assertEqual(migration.migrate(self.directory), 2)
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        self.assertEqual(migration.migrate(self.directory, write=True), 2)
        self.assertEqual(migration.migrate(self.directory, write=True, refresh=True), 0)
        self.assertEqual(protected, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
        # A source change is picked up by refresh, without depending on previous guide prose.
        p = self.directory / "baseline.ipynb"
        p.write_text(p.read_text(encoding="utf-8").replace("math.sqrt(x)", "math.log(x)"), encoding="utf-8")
        (self.directory / "huong_giai_quyet.md").write_text("## 3. Tiền xử lí\n\nĐổi phép biến đổi sang log theo mô tả mới.\n", encoding="utf-8")
        self.assertEqual(migration.migrate(self.directory, refresh=True), 2)

    def test_selected_companion_does_not_require_the_other_file(self):
        (self.directory / "baseline_basic_promting.md").write_text(guide(), encoding="utf-8")
        self.assertEqual(migration.migrate(self.directory, write=True), 1)
        self.assertFalse(validator.scan_directory(self.directory, ("basic",))[0])

    def test_atomic_packing_never_slices_a_class_or_creates_a_tiny_tail(self):
        code = "class Encoder:\n    def forward(self, batch):\n        return batch\n"
        nb = {"cells": [{"cell_type": "markdown", "source": "## Mô hình"},
                        {"cell_type": "code", "source": code}]}
        p = self.directory / "baseline.ipynb"
        p.write_text(json.dumps(nb), encoding="utf-8")
        units, _ = focused.inspect_notebook(p)
        bundles = focused.packs(units, limit=10)
        self.assertEqual(bundles[0][0]["code"], code.rstrip())
        self.assertEqual(len(bundles), 1)

    def test_fixed_token_ids_and_manual_nll_are_carried_into_the_guide(self):
        path = self.directory / "baseline.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"].append({"cell_type": "code", "source":
            "PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3\n"
            "def compute_loss(model, batch):\n"
            "    logp = model(batch['source'], batch['target'][:, :-1])\n"
            "    gold = batch['target'][:, 1:]\n"
            "    return -logp.gather(-1, gold.unsqueeze(-1)).mean()\n"})
        path.write_text(json.dumps(nb), encoding="utf-8")
        value = self.render("hardcore")
        self.assertIn("PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3", value)
        self.assertIn("gold = batch['target'][:, 1:]", value)
        self.assertIn("-logp.gather(-1, gold.unsqueeze(-1)).mean()", value)
        self.assertFalse(self.scan(value, "hardcore")[0])

    def test_report_excerpts_are_inline_exact_and_consumed_by_analysis(self):
        self.add_editable_source()
        paragraph = ("Chuẩn hóa NFC và gộp khoảng trắng. Giữ hoa/thường, ký tự khó, split và nhãn. "
                     "CTC cần đủ số bước cho target và ký tự lặp. Không fit alphabet trên held-out.")
        (self.directory / "huong_giai_quyet.md").write_text("## 3. Tiền xử lí\n\n" + paragraph + "\n", encoding="utf-8")
        value = self.render()
        self.assertIn("```markdown\n" + paragraph + "\n```", value)
        self.assertNotIn("### Yêu cầu bài toán", value)
        self.assertNotIn("cùng yêu cầu bài toán vào", value)
        self.assertFalse(self.scan(value)[0])
        for changed in (
            value.replace("CTC cần đủ số bước", "CTC bỏ kiểm độ dài", 1),
            value.replace("huong_giai_quyet.md#lines=3-3", "huong_giai_quyet.md#lines=99-99", 1),
            value.replace("(#references-s01)", "(#missing)", 1),
        ):
            self.assertTrue(self.scan(changed)[0])

    def test_report_first_adds_example_code_only_when_description_is_insufficient(self):
        self.add_editable_source()
        report = self.directory / "huong_giai_quyet.md"
        paragraph = "Dùng NFC và gộp khoảng trắng, giữ case và dấu câu. " * 3
        report.write_text("## 3. Tiền xử lí\n\n" + paragraph + "\n", encoding="utf-8")
        value = self.render("hardcore")
        self.assertNotIn("def normalize_text(value):", value)
        report.write_text("## 3. Tiền xử lí\n\nMô tả chưa đầy đủ.\n", encoding="utf-8")
        fallback = self.render("hardcore")
        self.assertIn("def normalize_text(value):", fallback)
        self.assertFalse(self.scan(fallback, "hardcore")[0])

    def test_missing_references_are_disclosed_without_inventing_requirements(self):
        value = self.render("hardcore")
        self.assertIn("Reference missing: yes", value)
        self.assertIn("Nguồn hiện có chưa mô tả đủ", value)
        self.assertFalse(self.scan(value, "hardcore")[0])
        self.assertTrue(self.scan(value.replace("Nguồn hiện có chưa mô tả đủ", "Nguồn đầy đủ"), "hardcore")[0])

    def test_reference_text_is_evidence_not_an_instruction_for_an_audit_session(self):
        report = "## 3. Tiền xử lí\n\nCode nguồn từng in fingerprint để đối chiếu ID. Giữ split và fit vocabulary chỉ trên train.\n"
        (self.directory / "huong_giai_quyet.md").write_text(report, encoding="utf-8")
        value = self.render("hardcore")
        self.assertFalse(self.scan(value, "hardcore")[0])
        self.assertTrue(self.scan(value.replace("Tạo module theo ý tưởng:", "Tạo fingerprint theo ý tưởng:", 1), "hardcore")[0])

    def test_model_contract_keeps_methods_used_by_existing_training(self):
        units_path = self.directory / "baseline.ipynb"
        nb = {"cells": [{"cell_type": "code", "source":
            "class Model:\n"
            "    def forward(self, batch):\n        return batch\n"
            "    def objective(self, logits, batch):\n        return logits\n"
            "loss = model.objective(model(batch), batch)\n"}]}
        units_path.write_text(json.dumps(nb), encoding="utf-8")
        units, _ = focused.inspect_notebook(units_path)
        self.assertIn("objective(logits, batch)", focused.interface(units))

    def test_signatures_keep_defaults_annotations_and_argument_kinds(self):
        node = ast.parse("def forward(self, x: int, /, *, mask=None, **kwargs) -> tuple:\n    pass\n").body[0]
        self.assertEqual(focused.signature(node), "forward(x: int, /, *, mask=None, **kwargs) -> tuple")


class RealMethods(unittest.TestCase):
    def test_traditional_sequence_and_image_methods_keep_their_ideas(self):
        domain = SKILL.name.split("-", 1)[0]
        examples = SKILL.parent / f"{domain}-notebook-engineer/assets/fewshot"
        expected = {"sentiment_imdb_reviews": "LinearSVC", "ocr_iam_line": "CTC",
                    "document_funsd": "CRF", "mt_ted_pt_en": "Transformer"}
        for task, method in expected.items():
            directory = examples / task
            if not directory.exists():
                continue
            for variant in ("basic", "hardcore"):
                value = focused.render(directory, variant)
                self.assertIn(method, value)
                errors, _ = validator.scan_focused_guide(directory / f"baseline_{variant}_promting.md", variant, value)
                self.assertFalse(errors, errors)
                self.assertLess(len(validator.STEP.findall(validator.outside_fences(value))), 12)


if __name__ == "__main__":
    unittest.main()
