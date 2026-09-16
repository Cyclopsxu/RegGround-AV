"""盲标工作簿的数据结构、artifact-tool 写出入口与回收读取器。"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

STYLE_BODY = 0
STYLE_HEADER = 1
STYLE_MONO = 2
STYLE_EXAMPLE = 3

BUILDER = Path(__file__).with_name("build_workbook.mjs")


@dataclass
class Sheet:
    """一个工作表：首行为表头，其余为数据行。"""

    name: str
    header: list[str]
    rows: list[list[str]] = field(default_factory=list)
    column_widths: list[float] = field(default_factory=list)
    row_styles: dict[int, int] = field(default_factory=dict)
    mono_columns: tuple[str, ...] = ()
    freeze_header: bool = True


def write_workbook(
    path: Path,
    sheets: list[Sheet],
    *,
    node_executable: Path,
    node_modules: Path,
    preview_dir: Path,
) -> None:
    """经 Codex 随附的 artifact-tool runtime 写出并渲染检查工作簿。"""
    if not sheets:
        raise ValueError("工作簿至少需要一个 sheet")
    for sheet in sheets:
        width = len(sheet.header)
        for row_index, row in enumerate(sheet.rows):
            if len(row) != width:
                raise ValueError(
                    f"sheet {sheet.name!r} 第 {row_index + 1} 行列数 {len(row)} "
                    f"与表头 {width} 不一致"
                )
    if not node_executable.is_file():
        raise FileNotFoundError(f"artifact-tool Node 不存在：{node_executable}")
    if not node_modules.is_dir():
        raise FileNotFoundError(f"artifact-tool node_modules 不存在：{node_modules}")
    if not BUILDER.is_file():
        raise FileNotFoundError(f"工作簿构建器不存在：{BUILDER}")

    path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sheets": [
            {
                **asdict(sheet),
                "mono_columns": list(sheet.mono_columns),
            }
            for sheet in sheets
        ]
    }

    with tempfile.TemporaryDirectory(prefix="rag-av1-workbook-") as tmp:
        workdir = Path(tmp)
        spec_path = workdir / "workbook_spec.json"
        builder_path = workdir / BUILDER.name
        temporary_output = workdir / path.name
        spec_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copy2(BUILDER, builder_path)
        (workdir / "node_modules").symlink_to(node_modules, target_is_directory=True)
        command = [
            str(node_executable),
            str(builder_path),
            "--spec",
            str(spec_path),
            "--output",
            str(temporary_output),
            "--preview-dir",
            str(preview_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"artifact-tool 工作簿写出失败：{details}")
        shutil.copy2(temporary_output, path)


def read_workbook(path: Path) -> dict[str, list[list[str]]]:
    """读取 xlsx 单元格文本，供回收校验和生成后逐字核验使用。"""
    import re
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        target_by_id = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in rels.findall(f"{rel_ns}Relationship")
        }
        r_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            table = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in table.findall(f"{ns}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{ns}t")))

        result: dict[str, list[list[str]]] = {}
        sheets = workbook.find(f"{ns}sheets")
        if sheets is None:
            raise ValueError(f"{path} 缺少 sheets")
        for sheet in sheets.findall(f"{ns}sheet"):
            target = target_by_id[sheet.attrib[f"{r_ns}id"]].lstrip("/")
            name = target if target.startswith("xl/") else f"xl/{target}"
            root = ET.fromstring(archive.read(name))
            rows: list[list[str]] = []
            for row in root.iter(f"{ns}row"):
                cells: dict[int, str] = {}
                for cell in row.findall(f"{ns}c"):
                    ref = cell.attrib.get("r", "")
                    letters = re.match(r"([A-Z]+)", ref)
                    if letters is None:
                        continue
                    index = 0
                    for char in letters.group(1):
                        index = index * 26 + (ord(char) - ord("A") + 1)
                    index -= 1
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        node = cell.find(f"{ns}is")
                        text = (
                            "".join(part.text or "" for part in node.iter(f"{ns}t"))
                            if node is not None
                            else ""
                        )
                    elif cell_type == "s":
                        value = cell.find(f"{ns}v")
                        text = (
                            shared[int(value.text)]
                            if value is not None and value.text
                            else ""
                        )
                    else:
                        value = cell.find(f"{ns}v")
                        text = value.text or "" if value is not None else ""
                    cells[index] = text
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(i, "") for i in range(width)])
            result[sheet.attrib["name"]] = rows
    return result


__all__ = [
    "STYLE_BODY",
    "STYLE_EXAMPLE",
    "STYLE_HEADER",
    "STYLE_MONO",
    "Sheet",
    "read_workbook",
    "write_workbook",
]
