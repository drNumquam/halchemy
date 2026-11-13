#!/usr/bin/env python3
"""Generate toc.yaml from a PDF's hierarchical bookmarks.

Usage:
    python pdf2toc.py --pdf /path/to/manual.pdf [--schema /path/to/schema.yaml] --o /path/to/output

Requirements:
    python -m pip install pymupdf

Install PyYAML and jsonschema as well when using --schema:
    python -m pip install pyyaml jsonschema

The output directory receives toc.yaml and report.txt. Page indexes are zero-based
physical PDF pages, and end_index is inclusive. The source field contains only the
input PDF filename. Schema validation is skipped unless --schema is specified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Section:
    """One bookmark in the nested table of contents."""

    id: int
    title: str
    start_index: int
    level: int
    end_index: int = 0
    children: list["Section"] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Create toc.yaml and report.txt from a PDF bookmark table of contents."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--schema", help="Optional YAML or JSON Schema file used to validate toc.yaml."
    )
    parser.add_argument(
        "--o", "--output", dest="output", required=True,
        help="Directory where toc.yaml and report.txt will be written.",
    )
    return parser.parse_args()


def build_sections(bookmarks: list[list[Any]], page_count: int) -> list[Section]:
    """Convert flat PDF bookmarks to nested sections with physical page indexes."""
    nodes: list[Section] = []
    for bookmark in bookmarks:
        if len(bookmark) < 3:
            continue
        level, title, page_number = bookmark[:3]
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            level = int(level)
            page_number = int(page_number)
        except (TypeError, ValueError):
            continue
        if level < 1 or page_number < 1 or page_number > page_count:
            continue
        # Convert PyMuPDF's one-based page number to a zero-based physical index.
        nodes.append(Section(
            id=len(nodes) + 1,
            title=" ".join(title.split()),
            start_index=page_number - 1,
            level=level,
        ))

    roots: list[Section] = []
    stack: list[Section] = []
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    # A section ends just before the next bookmark at the same or a higher level.
    # This lets parent sections cover their nested children.
    for index, node in enumerate(nodes):
        next_peer_or_ancestor = next(
            (candidate for candidate in nodes[index + 1:]
             if candidate.level <= node.level),
            None,
        )
        end_index = (
            next_peer_or_ancestor.start_index - 1
            if next_peer_or_ancestor is not None
            else page_count - 1
        )
        node.end_index = max(node.start_index, end_index)
    return roots


def walk(sections: list[Section]):
    """Yield every section recursively in document order."""
    for section in sections:
        yield section
        yield from walk(section.children)


def render_yaml(source_filename: str, page_count: int, sections: list[Section]) -> str:
    """Render the template structure without requiring a YAML library."""
    lines = [
        f"source: {json.dumps(source_filename, ensure_ascii=False)}",
        f"pages: {page_count}",
        "sections:",
    ]

    def emit(items: list[Section], indent: int) -> None:
        pad = " " * indent
        for section in items:
            lines.extend([
                f"{pad}- id: {section.id}",
                f"{pad}  title: {json.dumps(section.title, ensure_ascii=False)}",
                f"{pad}  start_index: {section.start_index}",
                f"{pad}  end_index: {section.end_index}",
            ])
            if section.children:
                lines.append(f"{pad}  children:")
                emit(section.children, indent + 4)
            else:
                lines.append(f"{pad}  children: []")

    if sections:
        emit(sections, 2)
    else:
        lines[-1] += " []"
    return "\n".join(lines) + "\n"


class Reporter:
    """Print messages and collect them for the process report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def validate_toc(toc_path: Path, schema_path: Path) -> tuple[bool, str]:
    """Validate the YAML output against an optional JSON Schema written as YAML."""
    try:
        import yaml
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        return False, (
            "Schema validation requires PyYAML and jsonschema. Install them with "
            "python -m pip install pyyaml jsonschema. "
            f"Details: {exc}"
        )

    if not schema_path.is_file():
        return False, f"Schema file does not exist: {schema_path}"
    try:
        with toc_path.open("r", encoding="utf-8") as stream:
            toc_data = yaml.safe_load(stream)
        with schema_path.open("r", encoding="utf-8") as stream:
            schema = yaml.safe_load(stream)
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(toc_data),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        return False, f"Could not parse YAML or validate against schema: {exc}"

    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "root"
        extra = f" ({len(errors)} errors total)" if len(errors) > 1 else ""
        return False, f"{location}: {first.message}{extra}"
    return True, "toc.yaml is valid against the specified schema."


def main() -> int:
    """Run the PDF-to-TOC workflow and record its console output."""
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser()
    output_dir = Path(args.output).expanduser()
    reporter = Reporter()
    report_path: Path | None = None
    exit_code = 0

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.txt"

        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            reporter.log(f"WARNING: The input file does not exist or is not a PDF: {pdf_path}")
            exit_code = 2
        else:
            try:
                import pymupdf
            except ImportError:
                try:
                    import fitz as pymupdf
                except ImportError:
                    pymupdf = None
            if pymupdf is None:
                reporter.log(
                    "WARNING: PyMuPDF is not installed. Install it with: "
                    "python -m pip install pymupdf"
                )
                exit_code = 2
            else:
                reporter.log(f"Input PDF: {pdf_path}")
                reporter.log(f"Output directory: {output_dir}")
                with pymupdf.open(pdf_path) as document:
                    page_count = len(document)
                    if document.needs_pass:
                        bookmarks = []
                        reporter.log("WARNING: The PDF is password-protected; no TOC was created.")
                        exit_code = 2
                    else:
                        bookmarks = document.get_toc(simple=True)

                sections = build_sections(bookmarks, page_count) if exit_code == 0 else []
                if exit_code == 0 and not sections:
                    reporter.log(
                        "WARNING: The PDF has no usable table of contents (bookmarks). "
                        "No toc.yaml was created."
                    )
                    exit_code = 1
                elif exit_code == 0:
                    toc_path = output_dir / "toc.yaml"
                    toc_path.write_text(
                        render_yaml(pdf_path.name, page_count, sections), encoding="utf-8"
                    )
                    reporter.log(f"Created TOC file: {toc_path}")
                    reporter.log(f"PDF pages: {page_count}")
                    reporter.log(f"Usable TOC entries: {sum(1 for _ in walk(sections))}")
                    reporter.log("Page indexes are zero-based physical PDF indexes; end_index is inclusive.")

                    if args.schema:
                        schema_path = Path(args.schema).expanduser()
                        valid, message = validate_toc(toc_path, schema_path)
                        reporter.log(("Validation passed: " if valid else "WARNING: Validation failed: ") + message)
                        if not valid:
                            exit_code = 2
                    else:
                        reporter.log("Schema validation skipped: no --schema file was specified.")

                    if exit_code == 0:
                        reporter.log("Process completed successfully.")
    except Exception as exc:
        reporter.log(f"WARNING: Processing failed: {exc}")
        exit_code = 2
    finally:
        if report_path is not None:
            reporter.log(f"Process finished with exit code {exit_code}.")
            reporter.save(report_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
