"""Behavioral checks for responsibility-based guides; no model/network calls."""
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

    def test_summary_checks_actual_ratio_and_complete_evidence(self):
        value = self.render("hardcore")
        # Both fit the allocated 80 output tokens, but only the first is shorter
        # than its actual input. Arithmetic is checked without claiming authenticity.
        self.assertFalse(self.scan(self.measured(value, 300, 50), "hardcore")[0])
        self.assertTrue(self.scan(self.measured(value, 30, 40), "hardcore")[0])
        self.assertTrue(self.scan(self.measured(value, 300, 81), "hardcore")[0])
        measured = self.measured(value)
        incomplete = re.sub(r"^Measurement: .*:summary.*\n", "", measured, count=1, flags=re.M)
        self.assertTrue(self.scan(incomplete, "hardcore")[0])

    def test_budget_target_and_platform_limit_are_distinct(self):
        value = self.render("hardcore")
        for changed in (
            value.replace("input=650; output=950; reserve=200", "input=650; output=951; reserve=200", 1),
            value.replace("input=1400; output=80; reserve=200", "input=1400; output=81; reserve=200", 1),
            self.measured(value).replace('"total_tokens": 700', '"total_tokens": 2000', 1),
        ):
            self.assertTrue(self.scan(changed, "hardcore")[0])

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
        self.assertIn('"EOS_ID":3', value)
        self.assertIn("token NLL", value)
        self.assertFalse(self.scan(value, "hardcore")[0])

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
