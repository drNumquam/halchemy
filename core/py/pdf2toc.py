#!/usr/bin/env python3
"""Create a YAML table of contents from a PDF's bookmarks.

The input PDF must contain a usable, hierarchical table of contents (bookmarks).
The generated page indexes are zero-based and end_index is inclusive.

Requires PyMuPDF: python -m pip install pymupdf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


@dataclass
class Section:
    """A PDF bookmark represented as a node in the output TOC tree."""

    id: int
    title: str
    start_index: int
    end_index: int = 0
    children: list["Section"] = field(default_factory=list)
    level: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate toc.yaml and report.txt from a bookmarked PDF."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--o", "--output", dest="output", required=True,
        help="Directory where toc.yaml and report.txt will be written.",
    )
    return parser.parse_args()


def yaml_string(value: str) -> str:
    """Encode a string as a valid YAML double-quoted scalar."""
    # JSON string quoting is compatible with YAML double-quoted strings.
    return json.dumps(value, ensure_ascii=False)


def build_sections(bookmarks: list[list], page_count: int) -> list[Section]:
    """Convert PyMuPDF's flat bookmark list into a nested section tree."""
    entries: list[tuple[int, str, int]] = []
    for item in bookmarks:
        level, title, page = item[:3]
        if not isinstance(title, str) or not title.strip():
            continue
        # PyMuPDF uses one-based page numbers; the schema uses zero-based indexes.
        if not isinstance(page, int) or page < 1 or page > page_count:
            continue
        entries.append((int(level), " ".join(title.split()), page - 1))

    nodes = [
        Section(id=index, title=title, start_index=page, level=level)
        for index, (level, title, page) in enumerate(entries, start=1)
    ]

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

    # A section ends on the page immediately before the next sibling/ancestor-level
    # entry. The final section runs to the end of the PDF. Page indexes are inclusive.
    for index, node in enumerate(nodes):
        next_boundary = next(
            (candidate for candidate in nodes[index + 1:]
             if candidate.level <= node.level),
            None,
        )
        last_page = next_boundary.start_index - 1 if next_boundary else page_count - 1
        node.end_index = max(node.start_index, last_page)

    return roots


def render_yaml(source: str, page_count: int, sections: list[Section]) -> str:
    """Render the current toc.yaml structure without requiring a YAML package."""
    lines = [
        f"source: {yaml_string(source)}",
        f"pages: {page_count}",
        "sections:",
    ]

    def emit(items: list[Section], indent: int) -> None:
        pad = " " * indent
        for section in items:
            lines.extend([
                f"{pad}- id: {section.id}",
                f"{pad}  title: {yaml_string(section.title)}",
                f"{pad}  start_index: {section.start_index}",
                f"{pad}  end_index: {section.end_index}",
                f"{pad}  children:",
            ])
            if section.children:
                emit(section.children, indent + 4)
            else:
                lines[-1] += " []"

    if sections:
        emit(sections, 2)
    else:
        lines[-1] += " []"
    return "\n".join(lines) + "\n"


class Reporter:
    """Print messages and retain the same output for report.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def run(pdf_path: Path, output_dir: Path, reporter: Reporter) -> int:
    """Process the PDF and write the requested output files."""
    try:
        import pymupdf
    except ImportError:
        reporter.write(
            "WARNING: PyMuPDF is not installed. Install it with: "
            "python -m pip install pymupdf"
        )
        return 2

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        reporter.write(f"WARNING: The input PDF file does not exist: {pdf_path}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                reporter.write("WARNING: The PDF is password-protected; no TOC was created.")
                return 2

            page_count = len(document)
            bookmarks = document.get_toc(simple=True)
            sections = build_sections(bookmarks, page_count)

            reporter.write(f"Input PDF: {pdf_path}")
            reporter.write(f"Pages: {page_count}")
            reporter.write(f"Usable TOC entries: {sum(1 for _ in walk(sections))}")

            if not sections:
                reporter.write(
                    "WARNING: This PDF has no usable table of contents (bookmarks). "
                    "No toc.yaml was created."
                )
                return 1

            yaml_text = render_yaml(str(pdf_path.resolve()), page_count, sections)
            (output_dir / "toc.yaml").write_text(yaml_text, encoding="utf-8")
            reporter.write(f"Created TOC file: {output_dir / 'toc.yaml'}")
            reporter.write("Page indexes in toc.yaml are zero-based; end_index is inclusive.")
            reporter.write("Process completed successfully.")
            return 0
    except Exception as exc:
        reporter.write(f"WARNING: PDF processing failed: {exc}")
        return 2
    finally:
        # Save all console messages, including warnings and completion status.
        reporter.save(report_path)
        if report_path.exists():
            print(f"Created process report: {report_path}")


def walk(sections: list[Section]):
    """Yield every section recursively."""
    for section in sections:
        yield section
        yield from walk(section.children)


def main() -> int:
    """Run the command-line application."""
    args = parse_args()
    reporter = Reporter()
    return run(Path(args.pdf).expanduser(), Path(args.output).expanduser(), reporter)


if __name__ == "__main__":
    sys.exit(main())
