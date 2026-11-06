"""Offline regression tests: python tests/test_validate_prompt_guides.py."""
import importlib.util
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
