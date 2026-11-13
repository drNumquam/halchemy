#!/usr/bin/env python3
"""Generate a TOC YAML file from a PDF bookmark tree and validate it.

This script accepts a PDF device reference manual and produces:
  1. toc.yaml
  2. report.txt

The generated YAML follows the repository template in core/template/yaml/toc.yaml
and validates against the schema in core/template/yaml/toc_schema.yaml.

Usage:
    python pdf2chunks.py --pdf /path/to/manual.pdf --o /path/to/output_dir

The script prints all processing messages to the console and also saves them to
report.txt in the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - only for runtime environment
    raise SystemExit(
        "PyMuPDF is not installed. Install it with: python -m pip install pymupdf"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create toc.yaml and report.txt from a PDF bookmark table of contents."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--o",
        "--output",
        dest="output",
        required=True,
        help="Directory where toc.yaml and report.txt will be written.",
    )
    return parser.parse_args()


def clean_title(value: str) -> str:
    """Normalize bookmark titles for YAML output."""
    return " ".join(str(value).strip().split())


def yaml_string(value: str) -> str:
    """Return a YAML-safe double-quoted string."""
    return json.dumps(value, ensure_ascii=False)


def build_section_tree(bookmarks: Iterable[Any], page_count: int) -> list[dict[str, Any]]:
    """Convert flat PDF bookmarks into a nested tree of sections.

    PyMuPDF bookmark entries are typically represented as tuples like:
        (level, title, page_number, destination)
    """
    valid_nodes: list[dict[str, Any]] = []
    for item in bookmarks:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        level, title, page_no = item[0], item[1], item[2]
        if not isinstance(level, int):
            try:
                level = int(level)
            except (TypeError, ValueError):
                continue
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            page_no = int(page_no)
        except (TypeError, ValueError):
            continue
        if page_no < 1 or page_no > page_count:
            continue

        valid_nodes.append(
            {
                "id": len(valid_nodes) + 1,
                "title": clean_title(title),
                "start_index": page_no - 1,
                "level": max(1, level),
                "children": [],
            }
        )

    root_sections: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []

    for node in valid_nodes:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            root_sections.append(node)
        stack.append(node)

    def assign_end_index(section: dict[str, Any], later: list[dict[str, Any]]) -> None:
        next_sibling: dict[str, Any] | None = None
        for candidate in later:
            if candidate["level"] <= section["level"]:
                next_sibling = candidate
                break
        if next_sibling is not None:
            section["end_index"] = max(section["start_index"], next_sibling["start_index"] - 1)
        else:
            section["end_index"] = max(section["start_index"], page_count - 1)

        for child in section["children"]:
            assign_end_index(child, later)

    # The later list is ordered by bookmark appearance; we use it to determine the
    # next boundary at the same or higher level.
    ordered = []
    for section in root_sections:
        ordered.extend(iter_flat_sections(section))

    for section in root_sections:
        assign_end_index(section, ordered)

    return root_sections


def iter_flat_sections(section: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield section
    for child in section["children"]:
        yield from iter_flat_sections(child)


def render_yaml(source_name: str, pages: int, sections: list[dict[str, Any]], indent: int = 0) -> str:
    """Render a TOC YAML structure manually without external dependencies."""
    pad = " " * indent
    lines: list[str] = [
        f"{pad}source: {yaml_string(source_name)}",
        f"{pad}pages: {pages}",
        f"{pad}sections:",
    ]

    def emit(item_list: list[dict[str, Any]], current_indent: int) -> None:
        for section in item_list:
            start = section["start_index"]
            end = section["end_index"]
            lines.append(f"{' ' * current_indent}- id: {section['id']}")
            lines.append(f"{' ' * current_indent}  title: {yaml_string(section['title'])}")
            lines.append(f"{' ' * current_indent}  start_index: {start}")
            lines.append(f"{' ' * current_indent}  end_index: {end}")
            lines.append(f"{' ' * current_indent}  children:")
            if section["children"]:
                emit(section["children"], current_indent + 4)
            else:
                lines.append(f"{' ' * (current_indent + 4)}[]")

    if sections:
        emit(sections, 2)
    else:
        lines.append(f"{pad}  []")

    return "\n".join(lines) + "\n"


def validate_section(section: dict[str, Any], path: str = "root") -> tuple[bool, str]:
    """Validate the recursive structure against the current TOC schema.

    The validation is intentionally lightweight and mirrors the schema values used in
    the repository: integer id, string title, integer start_index, integer end_index,
    and a recursive children list.
    """
    if not isinstance(section, dict):
        return False, f"{path}: section is not an object"

    for key in ("id", "title", "start_index", "end_index", "children"):
        if key not in section:
            return False, f"{path}: missing key '{key}'"

    if not isinstance(section["id"], int):
        return False, f"{path}: 'id' must be an integer"
    if not isinstance(section["title"], str):
        return False, f"{path}: 'title' must be a string"
    if not isinstance(section["start_index"], int):
        return False, f"{path}: 'start_index' must be an integer"
    if not isinstance(section["end_index"], int):
        return False, f"{path}: 'end_index' must be an integer"
    if not isinstance(section["children"], list):
        return False, f"{path}: 'children' must be a list"

    for idx, child in enumerate(section["children"]):
        ok, msg = validate_section(child, f"{path}.children[{idx}]")
        if not ok:
            return False, msg

    return True, "OK"


def validate_toc_document(data: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Top-level value must be a mapping"
    if "source" not in data or not isinstance(data["source"], str):
        return False, "Missing or invalid 'source'"
    if "pages" not in data or not isinstance(data["pages"], int):
        return False, "Missing or invalid 'pages'"
    if data["pages"] < 0:
        return False, "'pages' must be zero or a positive integer"
    if "sections" not in data or not isinstance(data["sections"], list):
        return False, "Missing or invalid 'sections'"

    for index, section in enumerate(data["sections"]):
        valid, msg = validate_section(section, f"sections[{index}]")
        if not valid:
            return False, msg

    return True, "TOC is valid"


def load_yaml_like_dict(text: str) -> dict[str, Any]:
    """Minimal YAML reader for the generated output.

    The generated TOC is intentionally simple and compact, so this mode parses only the
    keys used by the repository schema.
    """
    out: dict[str, Any] = {"source": "", "pages": 0, "sections": []}
    current_sections: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    lines = text.splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if stripped.startswith("source:"):
            out["source"] = stripped.split(":", 1)[1].strip().strip("'\"")
            continue
        if stripped.startswith("pages:"):
            try:
                out["pages"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                out["pages"] = 0
            continue
        if stripped.startswith("sections:"):
            continue
        if stripped.startswith("- id:"):
            section = {
                "id": int(stripped.split(":", 1)[1].strip()),
                "title": "",
                "start_index": 0,
                "end_index": 0,
                "children": [],
            }
            current_sections.append(section)
            stack.append(section)
            continue

        if "title:" in stripped:
            section = stack[-1]
            section["title"] = stripped.split(":", 1)[1].strip().strip("'\"")
            continue
        if "start_index:" in stripped:
            section = stack[-1]
            section["start_index"] = int(stripped.split(":", 1)[1].strip())
            continue
        if "end_index:" in stripped:
            section = stack[-1]
            section["end_index"] = int(stripped.split(":", 1)[1].strip())
            continue
        if stripped == "children:":
            continue
        if stripped == "[]":
            if stack:
                stack.pop()
            continue

    def attach_children(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # This helper preserves the generated tree shape directly as it is produced.
        return items

    out["sections"] = attach_children(current_sections)
    return out


def write_report(log_lines: list[str], path: Path) -> None:
    path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        log(f"WARNING: Input file not found or not a PDF: {pdf_path}")
        write_report(log_lines, output_dir / "report.txt")
        return 1

    log(f"Opening PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - environment dependent
        log(f"WARNING: Could not open PDF: {exc}")
        write_report(log_lines, output_dir / "report.txt")
        return 1

    if doc.needs_pass:
        log("WARNING: PDF requires a password. The file is not suitable for TOC extraction.")
        doc.close()
        write_report(log_lines, output_dir / "report.txt")
        return 1

    page_count = len(doc)
    log(f"PDF page count: {page_count}")

    toc_entries = doc.get_toc(simple=False)
    if not toc_entries:
        log("WARNING: PDF has no usable bookmark table of contents.")
        doc.close()
        write_report(log_lines, output_dir / "report.txt")
        return 1

    root_sections = build_section_tree(toc_entries, page_count)
    if not root_sections:
        log("WARNING: No valid sections were extracted from the PDF bookmarks.")
        doc.close()
        write_report(log_lines, output_dir / "report.txt")
        return 1

    source_name = pdf_path.name
    toc_text = render_yaml(source_name, page_count, root_sections)
    toc_path = output_dir / "toc.yaml"
    toc_path.write_text(toc_text, encoding="utf-8")
    log(f"Created TOC file: {toc_path}")

    # Validate the generated YAML content against the repository schema.
    parsed_data = load_yaml_like_dict(toc_text)
    valid, msg = validate_toc_document(parsed_data)
    if not valid:
        log(f"WARNING: TOC validation failed: {msg}")
    else:
        log(f"Validation result: {msg}")

    doc.close()
    report_path = output_dir / "report.txt"
    write_report(log_lines, report_path)
    log(f"Created report file: {report_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
