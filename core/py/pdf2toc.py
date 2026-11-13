#!/usr/bin/env python3
"""Generate and validate toc.yaml from a PDF's hierarchical bookmarks.

Usage:
    python pdf2toc.py --pdf /path/to/manual.pdf --o /path/to/output_directory

Requirements:
    python -m pip install pymupdf pyyaml jsonschema

The output directory receives toc.yaml and report.txt. Section page numbers are
one-based physical PDF pages, matching the repository TOC schema. The source
field contains only the input PDF filename.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+(\S.*)$")


@dataclass
class Section:
    """One PDF bookmark in the hierarchical TOC."""

    id: int
    number: str
    title: str
    start_page: int
    start_y: float
    level: int
    end_page: int = 0
    end_y: float = 0.0
    pdf: str = ""
    keywords: list[str] = field(default_factory=list)
    references: list[dict[str, int]] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Create and validate toc.yaml from a PDF bookmark table of contents."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--o", "--output", dest="output", required=True,
        help="Directory where toc.yaml and report.txt will be written.",
    )
    return parser.parse_args()


def clean_title(raw_title: str) -> tuple[str, str]:
    """Separate an optional leading section number from a bookmark title."""
    title = " ".join(raw_title.split())
    match = NUMBER_RE.match(title)
    if match:
        return match.group(1), match.group(2).strip()
    return "", title


def bookmark_y(destination: Any) -> float:
    """Read a bookmark's vertical destination coordinate when available."""
    try:
        point = destination.get("to") if isinstance(destination, dict) else None
        value = float(point.y) if point is not None else 0.0
        return value if value >= 0 else 0.0
    except (AttributeError, TypeError, ValueError):
        return 0.0


def build_sections(bookmarks: list[list[Any]], page_count: int) -> list[Section]:
    """Build a nested section tree and assign page/Y boundaries."""
    nodes: list[Section] = []
    for item in bookmarks:
        if len(item) < 3:
            continue
        level, raw_title, page = item[:3]
        if not isinstance(raw_title, str) or not raw_title.strip():
            continue
        try:
            level, page = int(level), int(page)
        except (TypeError, ValueError):
            continue
        if level < 1 or page < 1 or page > page_count:
            continue
        number, title = clean_title(raw_title)
        destination = item[3] if len(item) > 3 else None
        nodes.append(Section(
            id=len(nodes) + 1,
            number=number,
            title=title,
            start_page=page,
            start_y=bookmark_y(destination),
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

    # The next same-level or ancestor-level bookmark ends this section. A parent
    # therefore includes its child sections, while each child ends at its sibling
    # or the next section above it in the TOC hierarchy.
    for index, node in enumerate(nodes):
        following = next(
            (candidate for candidate in nodes[index + 1:]
             if candidate.level <= node.level),
            None,
        )
        if following is None:
            node.end_page = page_count
            node.end_y = 0.0
        elif following.start_page > node.start_page:
            node.end_page = following.start_page - 1
            node.end_y = 0.0
        else:
            node.end_page = node.start_page
            node.end_y = following.start_y
        if node.end_page < node.start_page:
            node.end_page = node.start_page
    return roots


def walk(sections: list[Section]):
    """Yield all sections in document order."""
    for section in sections:
        yield section
        yield from walk(section.children)


def yaml_string(value: str) -> str:
    """Encode a string using JSON quoting, compatible with YAML double quotes."""
    return json.dumps(value, ensure_ascii=False)


def render_yaml(source: str, page_count: int, sections: list[Section]) -> str:
    """Render all fields required by the repository's toc.yaml schema."""
    lines = [f"source: {yaml_string(source)}", f"pages: {page_count}", "sections:"]

    def emit(items: list[Section], indent: int) -> None:
        pad = " " * indent
        for section in items:
            lines.extend([
                f"{pad}- id: {section.id}",
                f"{pad}  number: {yaml_string(section.number)}",
                f"{pad}  title: {yaml_string(section.title)}",
                f"{pad}  start_page: {section.start_page}",
                f"{pad}  end_page: {section.end_page}",
                f"{pad}  start_y: {section.start_y:.3f}",
                f"{pad}  end_y: {section.end_y:.3f}",
                f"{pad}  pdf: {yaml_string(section.pdf)}",
                f"{pad}  keywords: []",
                f"{pad}  references: []",
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
    """Print console messages and keep a copy for report.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def validate_toc(toc_path: Path, schema_path: Path) -> tuple[bool, str]:
    """Validate the generated YAML against toc_schema.yaml."""
    try:
        import yaml
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        return False, (
            "Validation dependencies are missing; install them with "
            "python -m pip install pyyaml jsonschema. "
            f"Details: {exc}"
        )

    if not schema_path.is_file():
        return False, f"Schema file was not found: {schema_path}"
    try:
        with toc_path.open("r", encoding="utf-8") as stream:
            toc_data = yaml.safe_load(stream)
        with schema_path.open("r", encoding="utf-8") as stream:
            schema = yaml.safe_load(stream)
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(toc_data),
            key=lambda error: list(map(str, error.absolute_path)),
        )
    except Exception as exc:
        return False, f"Could not parse or validate YAML/schema: {exc}"

    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        suffix = f" ({len(errors)} validation errors total)" if len(errors) > 1 else ""
        return False, f"{location}: {error.message}{suffix}"
    return True, "toc.yaml is valid against toc_schema.yaml."


def main() -> int:
    """Run the PDF-to-TOC workflow."""
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
                    reporter.log(
                        "WARNING: PyMuPDF is not installed. Install it with: "
                        "python -m pip install pymupdf"
                    )
                    exit_code = 2
                    pymupdf = None

            if pymupdf is not None:
                reporter.log(f"Input PDF: {pdf_path}")
                reporter.log(f"Output directory: {output_dir}")
                with pymupdf.open(pdf_path) as document:
                    if document.needs_pass:
                        reporter.log("WARNING: The PDF is password-protected; no TOC was created.")
                        exit_code = 2
                        sections = []
                        page_count = len(document)
                    else:
                        page_count = len(document)
                        sections = build_sections(document.get_toc(simple=False), page_count)

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
                    reporter.log(f"TOC entries: {sum(1 for _ in walk(sections))}")
                    schema_path = (
                        Path(__file__).resolve().parents[1]
                        / "template" / "yaml" / "toc_schema.yaml"
                    )
                    valid, message = validate_toc(toc_path, schema_path)
                    reporter.log(("Validation passed: " if valid else "WARNING: Validation failed: ") + message)
                    exit_code = 0 if valid else 2
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
