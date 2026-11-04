"""Read task specifications from Excel or pasted cells; never execute cell content."""
from __future__ import annotations

import argparse
import csv
import io
import json
import posixpath
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile


COLUMNS = (
    ("task_group", "Nhóm bài toán", "task group", "group"),
    ("task", "Bài toán cụ thể", "task", "problem"),
    ("dataset", "Dataset khuyến nghị", "dataset", "recommended dataset"),
    ("source_priority", "Mức ưu tiên nguồn", "source priority"),
    ("loader", "Thư viện / cách lấy", "loader", "library", "acquisition"),
    ("source_url", "Link nguồn", "source url", "source link", "url"),
    ("load_command", "Lệnh tải / khởi tạo", "load command", "download command"),
    ("splits", "Split có sẵn", "splits", "available splits"),
    ("metric", "Metric chính", "metric", "primary metric"),
    ("notes", "Lưu ý đánh giá & tiền xử lý", "notes", "evaluation notes"),
    ("difficulty", "Mức độ triển khai", "difficulty", "implementation level"),
)
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN}
IDENTITY = {"task", "dataset", "source_url", "load_command"}


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value).casefold().replace("đ", "d"))
    return " ".join(re.findall(r"[a-z0-9]+", value))


ALIASES = {normalize(alias): col[0] for col in COLUMNS for alias in col}


def column_number(label):
    number = 0
    for char in label.upper():
        number = number * 26 + ord(char) - 64
    return number


def address(value):
    match = re.fullmatch(r"\$?([A-Za-z]+)\$?([1-9][0-9]*)", value)
    if not match:
        raise ValueError(f"Địa chỉ ô không hợp lệ: {value}")
    return int(match[2]), column_number(match[1])


def range_bounds(value):
    pieces = value.split(":")
    if len(pieces) > 2:
        raise ValueError(f"Vùng ô không hợp lệ: {value}")
    r1, c1 = address(pieces[0])
    r2, c2 = address(pieces[-1])
    if r2 < r1 or c2 < c1:
        raise ValueError(f"Vùng ô bị đảo ngược: {value}")
    return r1, c1, r2, c2


def slug(value):
    return normalize(value).replace(" ", "-")[:90].strip("-") or "task"


def text_value(node):
    return "".join(t.text or "" for t in node.findall(".//m:t", NS))


def relationship_targets(archive, part):
    relpart = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
    if relpart not in archive.namelist():
        return {}
    return {
        item.attrib["Id"]: item.attrib["Target"]
        for item in ET.fromstring(archive.read(relpart)).findall(f"{{{PKG_REL}}}Relationship")
    }


def read_workbook(path):
    sheets = []
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [text_value(si) for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        targets = relationship_targets(archive, "xl/workbook.xml")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            target = targets[sheet.attrib[f"{{{REL}}}id"]]
            part = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            xml = ET.fromstring(archive.read(part))
            rows, formulas, links = {}, set(), {}
            for cell in xml.findall("m:sheetData/m:row/m:c", NS):
                row, col = address(cell.attrib["r"])
                kind = cell.attrib.get("t")
                value = cell.findtext("m:v", default="", namespaces=NS)
                if kind == "s" and value:
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = text_value(cell)
                if cell.find("m:f", NS) is not None and not value:
                    formulas.add((row, col))
                rows.setdefault(row, {})[col] = value.strip()
            rels = relationship_targets(archive, part)
            for link in xml.findall("m:hyperlinks/m:hyperlink", NS):
                target = rels.get(link.attrib.get(f"{{{REL}}}id", ""), "")
                if target:
                    r1, c1, r2, c2 = range_bounds(link.attrib["ref"])
                    for row in range(r1, r2 + 1):
                        for col in range(c1, c2 + 1):
                            links[row, col] = target
            for merged in xml.findall("m:mergeCells/m:mergeCell", NS):
                r1, c1, r2, c2 = range_bounds(merged.attrib["ref"])
                value = rows.get(r1, {}).get(c1, "")
                for row in range(r1, r2 + 1):
                    for col in range(c1, c2 + 1):
                        if not rows.setdefault(row, {}).get(col):
                            rows[row][col] = value
            sheets.append((sheet.attrib["name"], rows, formulas, links))
    return sheets


def read_text(content):
    content = content.lstrip("\ufeff").strip()
    if not content:
        raise ValueError("Không có nội dung ô hoặc bảng để đọc.")
    # A single copied Excel cell can be quoted and contain a whole table/record.
    unwrapped = list(csv.reader(io.StringIO(content), delimiter="\t"))
    if len(unwrapped) == 1 and len(unwrapped[0]) == 1 and unwrapped[0][0] != content:
        content = unwrapped[0][0]
    lines = content.splitlines()
    pairs = {}
    for line in lines:
        label, sep, value = line.partition(":")
        key = ALIASES.get(normalize(label)) if sep else None
        if key:
            pairs[key] = value.strip()
        elif pairs and line.strip():
            last = next(reversed(pairs))
            pairs[last] += "\n" + line.strip()
    if pairs:
        keys = list(pairs)
        return {1: {i: key for i, key in enumerate(keys, 1)}, 2: {i: pairs[key] for i, key in enumerate(keys, 1)}}
    if lines and lines[0].lstrip().startswith("|"):
        table = []
        for line in lines:
            if not line.strip():
                continue
            values = re.split(r"(?<!\\)\|", line.strip().strip("|"))
            if all(re.fullmatch(r"\s*:?-{3,}:?\s*", v) for v in values):
                continue
            table.append([v.strip().replace(r"\|", "|") for v in values])
    else:
        try:
            dialect = csv.Sniffer().sniff(content, delimiters="\t,;")
        except csv.Error:
            dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
        table = list(csv.reader(io.StringIO(content), dialect=dialect))
    return {r: {c: value.strip() for c, value in enumerate(row, 1)} for r, row in enumerate(table, 1)}


def extract_tasks(rows, *, source, sheet, selections=(), header_row=None, formulas=(), links=None, allow_embedded=True):
    links = links or {}
    bounds = [range_bounds(selection) for selection in selections]
    headers = []
    for row, cells in sorted(rows.items()):
        mapped = {col: ALIASES[normalize(value)] for col, value in cells.items() if normalize(value) in ALIASES}
        recognized = len(set(mapped.values())) >= 2 or (len(mapped) == 1 and sum(bool(v) for v in cells.values()) == 1)
        if (header_row is None or row == header_row) and recognized and IDENTITY.intersection(mapped.values()):
            if len(mapped.values()) != len(set(mapped.values())):
                raise ValueError(f"Tiêu đề trùng trường ở {sheet}, dòng {row}.")
            headers.append((row, mapped))
    # Single field: value records are explicitly labeled and need no guess.
    if not headers and len(rows) == 2:
        labeled = {col: ALIASES[normalize(value)] for col, value in rows.get(1, {}).items() if normalize(value) in ALIASES}
        if labeled and IDENTITY.intersection(labeled.values()) and len(labeled) == len(rows[1]):
            headers = [(1, labeled)]
    if not headers and header_row is not None:
        raise ValueError(f"Không nhận diện được tiêu đề ở {sheet}, dòng {header_row}.")
    positional = not headers and rows and all(len(cells) == len(COLUMNS) for cells in rows.values() if any(cells.values()))
    if positional:
        headers = [(0, {i: col[0] for i, col in enumerate(COLUMNS, 1)})]
    if not headers:
        embedded_tasks, embedded_warnings = [], []
        if allow_embedded:
            for row, cells in sorted(rows.items()):
                for col, value in cells.items():
                    if bounds and not any(r1 <= row <= r2 and c1 <= col <= c2 for r1, c1, r2, c2 in bounds):
                        continue
                    structured = '\n' in value or '\t' in value or '|' in value or normalize(value.partition(':')[0]) in ALIASES
                    if not value or not structured:
                        continue
                    records, _ = extract_tasks(read_text(value), source=source, sheet=sheet, allow_embedded=False)
                    for index, record in enumerate(records, 1):
                        record['_source'].update(row=row, cell_column=col, embedded_record=index, selection=list(selections))
                        record['_output_slug'] += f'-cell-r{row}c{col}-{index}'
                    embedded_tasks.extend(records)
        if embedded_tasks:
            return embedded_tasks, embedded_warnings
        return [], [f"Bỏ qua {sheet}: không tìm thấy bảng bài toán; cần tiêu đề hoặc đúng 11 cột."]
    tasks, warnings = [], []
    for row, cells in sorted(rows.items()):
        if bounds and not any(r1 <= row <= r2 for r1, _, r2, _ in bounds):
            continue
        preceding = [(r, m) for r, m in headers if r < row]
        if not preceding or row in {r for r, _ in headers}:
            continue
        _, mapping = preceding[-1]
        record = {key: "" for key, *_ in COLUMNS}
        for col, key in mapping.items():
            record[key] = cells.get(col, "")
            markdown_link = re.fullmatch(r"\[([^\]]*)\]\(([^)]+)\)", record[key]) if key == "source_url" else None
            if markdown_link and (row, col) not in links:
                links[row, col] = markdown_link[2].strip().strip('<>')
            if key == "source_url" and (row, col) in links:
                record[key] = links[row, col]
        if not any(record.values()):
            continue
        if not any(record[key] for key in IDENTITY):
            warnings.append(f"Bỏ qua {sheet}, dòng {row}: thiếu bài toán/dataset/nguồn nhận diện.")
            continue
        record["_source"] = {"file": source, "sheet": sheet, "row": row, "selection": list(selections)}
        record["_source"]["hyperlinks"] = {
            key: {"text": cells.get(col, ""), "target": links[row, col]}
            for col, key in mapping.items() if (row, col) in links
        }
        record["_warnings"] = []
        missing_cache = [col for col in mapping if (row, col) in formulas]
        if missing_cache:
            record["_warnings"].append(f"Ô công thức chưa có giá trị lưu đệm ở cột {missing_cache}; không thực thi công thức.")
        if positional:
            record["_warnings"].append("Không có tiêu đề: đã dùng thứ tự 11 cột đã quy định.")
        base = slug(record["task"] + " " + record["dataset"])
        record["_output_slug"] = f"{base}-{slug(sheet)}-r{row}"
        tasks.append(record)
    return tasks, warnings


def read_tasks(path=None, *, text=None, sheets=(), selections=(), header_row=None):
    if (path is None) == (text is None):
        raise ValueError("Cung cấp đúng một file hoặc --text.")
    for selection in selections:
        range_bounds(selection)
    if header_row is not None and header_row < 1:
        raise ValueError("Dòng tiêu đề phải lớn hơn 0.")
    source = str(Path(path).resolve()) if path else "pasted cells"
    if path and Path(path).suffix.lower() in {".xlsx", ".xlsm"}:
        data = read_workbook(path)
    else:
        if path and Path(path).suffix.lower() not in {".csv", ".tsv", ".txt", ".md"}:
            raise ValueError("Định dạng chưa hỗ trợ. Dùng .xlsx/.xlsm hoặc xuất TSV/CSV; .xls/.xlsb cần trình đọc riêng.")
        if sheets:
            raise ValueError("--sheet chỉ áp dụng cho file Excel.")
        content = Path(path).read_text(encoding="utf-8-sig") if path else text
        data = [("pasted" if text is not None else Path(path).stem, read_text(content), set(), {})]
    unknown = set(sheets) - {name for name, *_ in data}
    if unknown:
        raise ValueError(f"Không tìm thấy sheet: {', '.join(sorted(unknown))}")
    tasks, warnings = [], []
    for name, rows, formulas, links in data:
        if sheets and name not in sheets:
            continue
        records, messages = extract_tasks(rows, source=source, sheet=name, selections=selections,
                                          header_row=header_row, formulas=formulas, links=links)
        tasks.extend(records)
        warnings.extend(messages)
    if not tasks:
        raise ValueError("Không tìm thấy dòng bài toán trong vùng đã chọn. Cần tiêu đề, dòng 11 cột hoặc ô có nhãn bài toán/dataset. " + " ".join(warnings))
    return {"tasks": tasks, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description="Đọc bảng mô tả bài toán từ Excel/ô đã sao chép thành JSON.")
    parser.add_argument("file", nargs="?")
    parser.add_argument("--text", help="Nội dung một ô hoặc bảng được dán; không chạy lệnh trong nội dung.")
    parser.add_argument("--sheet", action="append", default=[], help="Tên sheet; có thể lặp lại.")
    parser.add_argument("--range", dest="selections", action="append", default=[], help="Ô/vùng A1; đọc các dòng được chọn cùng tiêu đề và cột liên quan.")
    parser.add_argument("--header-row", type=int)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = read_tasks(args.file, text=args.text, sheets=args.sheet,
                            selections=args.selections, header_row=args.header_row)
    except (ValueError, OSError, KeyError, BadZipFile, ET.ParseError, csv.Error) as exc:
        parser.exit(2, f"Không đọc được bảng: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
