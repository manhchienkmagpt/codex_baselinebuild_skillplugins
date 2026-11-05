"""Offline checks for explicit STT and backward-compatible task-table reading."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from zipfile import ZipFile

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "read_task_table.py"
spec = importlib.util.spec_from_file_location("task_table_reader", SCRIPT)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class TaskNumberReading(unittest.TestCase):
    def test_header_preserves_explicit_number_instead_of_row(self):
        result = reader.read_tasks(text="STT\tBài toán cụ thể\tDataset\n14\tIntent Classification\tBANKING77")
        task, = result["tasks"]
        self.assertEqual(task["_source"]["row"], 2)
        self.assertEqual(task["task_number"], "14")
        self.assertEqual(task["_task_number"], 14)
        self.assertEqual(task["_output_dirname"], "14_Intent Classification")

    def test_header_alias_and_column_position(self):
        for header in ("Số thứ tự", "Sequence number", "Task number", "No."):
            with self.subTest(header=header):
                task, = reader.read_tasks(text=f"Task,Dataset,{header}\nTopic Classification,20 Newsgroups,013.0")["tasks"]
                self.assertEqual(task["_task_number"], 13)
                self.assertEqual(task["_output_dirname"], "13_Topic Classification")

    def test_headerless_twelve_columns_and_markdown_separator(self):
        values = ["Text Classification", "Intent Classification", "BANKING77", "HF", "datasets",
                  "https://huggingface.co/datasets/PolyAI/banking77", "load_dataset", "train/test", "Accuracy", "notes", "Dễ"]
        first = "| 14 | " + " | ".join(values) + " |"
        second = "| 24 | " + " | ".join(values) + " |"
        separator = "| :-: | " + " | ".join(["---"] * 11) + " |"
        tasks = reader.read_tasks(text="\n".join([first, separator, second]))["tasks"]
        self.assertEqual([r["_task_number"] for r in tasks], [14, 24])
        self.assertEqual([r["task"] for r in tasks], ["Intent Classification"] * 2)
        self.assertEqual(tasks[0]["metric"], "Accuracy")

    def test_original_eleven_columns_do_not_invent_stt(self):
        text = "\t".join(["Sequence Labeling", "NER", "CoNLL-2003", "HF", "datasets", "https://example.org",
                          "load_dataset", "train/validation/test", "F1", "notes", "Dễ"])
        task, = reader.read_tasks(text=text)["tasks"]
        self.assertEqual(task["dataset"], "CoNLL-2003")
        self.assertIsNone(task["_task_number"])
        self.assertIsNone(task["_output_dirname"])
        self.assertIn("_output_slug", task)

    def test_missing_number_remains_for_output_root_allocation(self):
        task, = reader.read_tasks(text="Task: Image Classification\nDataset: CIFAR-10")["tasks"]
        self.assertIsNone(task["_task_number"])
        self.assertIsNone(task["_output_dirname"])

    def test_invalid_number_warns_without_using_source_row(self):
        for number in ("0", "-2", "14.5", "not a number"):
            with self.subTest(number=number):
                task, = reader.read_tasks(text=f"STT\tTask\tDataset\n{number}\tNER\tCoNLL-2003")["tasks"]
                self.assertIsNone(task["_task_number"])
                self.assertIsNone(task["_output_dirname"])
                self.assertTrue(any("STT" in warning for warning in task["_warnings"]))

    def test_readable_windows_safe_title(self):
        name = reader.output_dirname(18, "  Chunking / Shallow Parsing: tiếng Việt.  ")
        self.assertEqual(name, "18_Chunking - Shallow Parsing - tiếng Việt")
        self.assertEqual(reader.output_dirname(3, "VQA (Visual Question Answering)"), "3_VQA (Visual Question Answering)")
        self.assertIsNone(reader.output_dirname(14, ""))

    def test_embedded_record_keeps_task_number(self):
        text = '"STT: 24\nTask: ABSA Extraction\nDataset: SemEval 2014"'
        task, = reader.read_tasks(text=text)["tasks"]
        self.assertEqual(task["_task_number"], 24)
        self.assertEqual(task["_output_dirname"], "24_ABSA Extraction")

    def test_selected_excel_cell_keeps_neighboring_stt(self):
        # Real minimal XLSX, with STT=24 on worksheet row 9; selection is only the task cell.
        workbook = '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Tasks" sheetId="1" r:id="rId1"/></sheets></workbook>'
        rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/></Relationships>'
        sheet = '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>STT</t></is></c><c r="B1" t="inlineStr"><is><t>Task</t></is></c><c r="C1" t="inlineStr"><is><t>Dataset</t></is></c></row><row r="9"><c r="A9"><v>24</v></c><c r="B9" t="inlineStr"><is><t>ABSA Extraction</t></is></c><c r="C9" t="inlineStr"><is><t>SemEval 2014</t></is></c></row></sheetData></worksheet>'
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", rels)
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        buffer.seek(0)
        (name, rows, formulas, links), = reader.read_workbook(buffer)
        tasks, _ = reader.extract_tasks(rows, source="tasks.xlsx", sheet=name, selections=["B9"], formulas=formulas, links=links)
        task, = tasks
        self.assertEqual(task["_source"]["row"], 9)
        self.assertEqual(task["_source"]["selection"], ["B9"])
        self.assertEqual(task["_task_number"], 24)
        self.assertEqual(task["_output_dirname"], "24_ABSA Extraction")


if __name__ == "__main__":
    unittest.main()
