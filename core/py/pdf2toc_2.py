#!/usr/bin/env python3
"""Generate ``toc.yaml`` from the hierarchical bookmarks of a PDF document.

Output contract (matches the ``schema.yaml`` used by the downstream tooling)::

    source: "<pdf filename>"
    pages: <page count>
    sections:
      - id: <int>                 # unique, assigned in document (pre-order) order
        title: "<title>"          # title without the leading section number
        number: "<1.2.3>"         # printed number, or positional fallback
        span_start: <int>         # first page, 1-based, inclusive
        span_end: <int>           # last page, 1-based, inclusive
        references: []            # not derivable from bookmarks (see notes)
        children: [...]

Notes:
    * Page numbers are 1-based physical PDF pages; ``span_end`` is inclusive.
    * Granularity is the page: a section spans every page it touches, so
      adjacent spans may overlap by one page. A section ends on the start page
      of the next bookmark of the same or a higher level, unless that bookmark
      begins at the very top of its page (see ``--span-end``), in which case
      the section ends on the previous page.
    * ``references`` is always emitted as an empty list: cross-references
      cannot be inferred from bookmarks and need a separate analysis step.
    * When a title carries no printed number (e.g. "Deployment Guidelines"),
      ``number`` falls back to its positional outline number.

Usage:
    python pdf2toc.py --pdf manual.pdf -o out_dir [--schema schema.yaml]
                      [--span-end {auto,inclusive,exclusive}] [--top-ratio 0.15] [-v]

Requirements:
    python -m pip install pymupdf
    python -m pip install pyyaml jsonschema      # only with --schema

Exit codes:
    0  success
    1  the PDF has no usable bookmarks (nothing written)
    2  error (bad input, missing dependency, I/O failure, schema violation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

LOGGER = logging.getLogger("pdf2toc")

TOC_FILENAME = "toc.yaml"
REPORT_FILENAME = "report.txt"
DEFAULT_SCHEMA_HINT = "./schema.yaml"
DEFAULT_TOP_RATIO = 0.15

# Matches "2", "2.1.1", "2.1.1." and appendix-style "A.1" followed by the title.
NUMBER_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?\s+(?P<title>\S.*)$"
)


class ExitCode(IntEnum):
    """Process exit codes."""

    OK = 0
    NO_TOC = 1
    FAILURE = 2


class SpanEnd(str, Enum):
    """How a section's last page is derived from the next bookmark."""

    AUTO = "auto"            # shared page unless the next bookmark starts at page top
    INCLUSIVE = "inclusive"  # always include the next bookmark's start page
    EXCLUSIVE = "exclusive"  # never include it (spans never overlap)


class TocError(Exception):
    """Raised for expected, user-reportable failures."""


@dataclass(frozen=True)
class Bookmark:
    """A validated bookmark with a resolved 1-based page number."""

    level: int
    title: str
    page: int
    top_ratio: float | None = None  # destination y / page height; None if unknown


@dataclass
class Section:
    """One node of the nested table of contents."""

    id: int
    raw_title: str
    level: int
    start_page: int
    end_page: int = 0
    top_ratio: float | None = None
    number: str = ""
    title: str = ""
    children: list[Section] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Command line and logging
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Create toc.yaml and report.txt from a PDF bookmark tree."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--schema",
        help="Optional YAML/JSON Schema file used to validate toc.yaml.",
    )
    parser.add_argument(
        "-o", "--output", "--o",
        dest="output",
        required=True,
        help="Directory where toc.yaml and report.txt are written.",
    )
    parser.add_argument(
        "--span-end",
        type=SpanEnd,
        choices=list(SpanEnd),
        default=SpanEnd.AUTO,
        help="Last page of a section: 'auto' (default) includes the next section's "
             "start page unless it starts at the top of that page; 'inclusive' "
             "always includes it; 'exclusive' never does.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=DEFAULT_TOP_RATIO,
        help="With --span-end auto: a section whose destination lies within this "
             f"fraction of the page height from the top counts as starting at the "
             f"top of the page (default: {DEFAULT_TOP_RATIO}).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def configure_console_logging(verbose: bool) -> None:
    """Send log records to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


def attach_report_handler(report_path: Path) -> logging.Handler:
    """Mirror every log record into the report file (overwritten each run)."""
    handler = logging.FileHandler(report_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    return handler


# --------------------------------------------------------------------------- #
# PDF reading
# --------------------------------------------------------------------------- #
def load_pymupdf() -> ModuleType:
    """Import PyMuPDF under either of its module names."""
    try:
        import pymupdf  # type: ignore[import-not-found]
        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore[import-not-found]
        return fitz
    except ImportError as exc:
        raise TocError(
            "PyMuPDF is not installed. Install it with: python -m pip install pymupdf"
        ) from exc


def read_bookmarks(pymupdf: ModuleType, pdf_path: Path) -> tuple[int, list[list[Any]]]:
    """Return ``(page_count, raw_bookmarks)`` for the given PDF.

    Each raw bookmark is ``[level, title, page, top_ratio]`` where ``top_ratio`` is
    the destination's vertical offset from the top of the page as a fraction of
    the page height (``None`` when the destination carries no position).
    """
    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise TocError("The PDF is password-protected; no TOC was created.")
            page_count = document.page_count
            heights: dict[int, float] = {}
            raw: list[list[Any]] = []
            for level, title, page, *rest in document.get_toc(simple=False):
                ratio: float | None = None
                destination = rest[0] if rest and isinstance(rest[0], dict) else {}
                point = destination.get("to")
                if point is not None and isinstance(page, int) and 1 <= page <= page_count:
                    if page not in heights:
                        heights[page] = float(document[page - 1].rect.height)
                    if heights[page] > 0:
                        ratio = min(max(float(point.y) / heights[page], 0.0), 1.0)
                raw.append([level, title, page, ratio])
            return page_count, raw
    except TocError:
        raise
    except Exception as exc:  # PyMuPDF raises several unrelated exception types.
        raise TocError(f"Cannot read the PDF: {exc}") from exc


# --------------------------------------------------------------------------- #
# TOC construction
# --------------------------------------------------------------------------- #
def normalize_bookmarks(raw: Sequence[Sequence[Any]], page_count: int) -> list[Bookmark]:
    """Validate raw bookmarks and resolve unresolved destinations.

    Entries with an unresolvable destination (page < 1) inherit the page of the
    following valid bookmark, so that their children stay nested under them.
    Entries that are malformed or point beyond the last page are dropped.
    """
    candidates: list[tuple[int, str, int | None, float | None]] = []
    for position, entry in enumerate(raw, start=1):
        if len(entry) < 3:
            LOGGER.warning("Bookmark #%d is malformed and was skipped.", position)
            continue
        level_raw, title_raw, page_raw = entry[:3]
        ratio = entry[3] if len(entry) > 3 and isinstance(entry[3], (int, float)) else None
        title = " ".join(title_raw.split()) if isinstance(title_raw, str) else ""
        try:
            level, page = int(level_raw), int(page_raw)
        except (TypeError, ValueError):
            LOGGER.warning("Bookmark #%d has non-numeric level/page; skipped.", position)
            continue
        if not title or level < 1:
            LOGGER.warning("Bookmark #%d has an empty title or invalid level; skipped.", position)
            continue
        if page > page_count:
            LOGGER.warning(
                "Bookmark %r points to page %d, beyond the last page (%d); skipped.",
                title, page, page_count,
            )
            continue
        candidates.append((level, title, page if page >= 1 else None, ratio))

    resolved: list[Bookmark] = []
    next_page: int | None = None
    for level, title, page, ratio in reversed(candidates):
        if page is None:
            if next_page is None:
                LOGGER.warning("Bookmark %r has no destination and no successor; skipped.", title)
                continue
            LOGGER.warning(
                "Bookmark %r has no destination; using page %d of the next bookmark.",
                title, next_page,
            )
            page, ratio = next_page, None
        next_page = page
        resolved.append(Bookmark(level=level, title=title, page=page, top_ratio=ratio))
    resolved.reverse()
    return resolved


def last_page(
    closed: Section, successor: Section, mode: SpanEnd, top_ratio: float
) -> int:
    """Last page of ``closed`` given the bookmark that terminates it."""
    if mode is SpanEnd.EXCLUSIVE:
        shares_page = False
    elif mode is SpanEnd.INCLUSIVE:
        shares_page = True
    else:  # AUTO: unknown position is treated conservatively as a shared page.
        shares_page = successor.top_ratio is None or successor.top_ratio > top_ratio
    end = successor.start_page if shares_page else successor.start_page - 1
    return max(closed.start_page, end)


def nest_sections(
    bookmarks: Sequence[Bookmark],
    page_count: int,
    mode: SpanEnd = SpanEnd.AUTO,
    top_ratio: float = DEFAULT_TOP_RATIO,
) -> list[Section]:
    """Nest bookmarks by level and compute inclusive 1-based page spans (single pass)."""
    roots: list[Section] = []
    stack: list[Section] = []
    previous_page = 0
    next_id = 0

    for bookmark in bookmarks:
        next_id += 1
        node = Section(
            id=next_id,
            raw_title=bookmark.title,
            level=bookmark.level,
            start_page=bookmark.page,
            top_ratio=bookmark.top_ratio,
        )
        if bookmark.page < previous_page:
            LOGGER.warning(
                "Bookmark %r (page %d) precedes the previous bookmark (page %d); "
                "its span is clamped to a single page.",
                bookmark.title, bookmark.page, previous_page,
            )
        previous_page = bookmark.page

        # Close every open section at the same or a deeper level.
        while stack and stack[-1].level >= node.level:
            closed = stack.pop()
            closed.end_page = last_page(closed, node, mode, top_ratio)
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    for closed in stack:
        closed.end_page = page_count
    return roots


def assign_numbers(sections: list[Section], prefix: str = "") -> int:
    """Split printed numbers from titles; fall back to positional numbering.

    Returns the number of sections that needed the positional fallback.
    """
    fallbacks = 0
    for position, section in enumerate(sections, start=1):
        match = NUMBER_PATTERN.match(section.raw_title)
        if match:
            section.number = match.group("number")
            section.title = match.group("title")
        else:
            section.number = f"{prefix}{position}"
            section.title = section.raw_title
            fallbacks += 1
        fallbacks += assign_numbers(section.children, prefix=f"{section.number}.")
    return fallbacks


def walk(sections: Sequence[Section]) -> Iterator[Section]:
    """Yield every section in document (pre-order) order."""
    for section in sections:
        yield section
        yield from walk(section.children)


def check_consistency(
    sections: Sequence[Section], page_count: int, parent: Section | None = None
) -> int:
    """Log structural anomalies and return their count."""
    issues = 0
    for section in sections:
        label = f"{section.number} {section.title!r}"
        if not 1 <= section.start_page <= section.end_page <= page_count:
            LOGGER.warning("Section %s has an invalid span %d-%d.",
                           label, section.start_page, section.end_page)
            issues += 1
        if parent and not (
            parent.start_page <= section.start_page and section.end_page <= parent.end_page
        ):
            LOGGER.warning("Section %s (%d-%d) lies outside its parent %s (%d-%d).",
                           label, section.start_page, section.end_page,
                           parent.number, parent.start_page, parent.end_page)
            issues += 1
        issues += check_consistency(section.children, page_count, section)
    return issues


def build_sections(
    raw: Sequence[Sequence[Any]],
    page_count: int,
    mode: SpanEnd = SpanEnd.AUTO,
    top_ratio: float = DEFAULT_TOP_RATIO,
) -> list[Section]:
    """Full pipeline: raw bookmarks -> numbered, consistent section tree."""
    bookmarks = normalize_bookmarks(raw, page_count)
    roots = nest_sections(bookmarks, page_count, mode, top_ratio)
    fallbacks = assign_numbers(roots)
    if fallbacks:
        LOGGER.info("%d section(s) had no printed number; positional numbers were used.", fallbacks)
    issues = check_consistency(roots, page_count)
    if issues:
        LOGGER.warning("%d structural anomaly(ies) detected; review the TOC.", issues)
    return roots


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def schema_hint(output_dir: Path, schema_path: Path | None) -> str:
    """Path of the schema relative to the output directory, for the YAML header."""
    if schema_path is None:
        return DEFAULT_SCHEMA_HINT
    relative = Path(os.path.relpath(schema_path.resolve(), output_dir.resolve())).as_posix()
    return relative if relative.startswith(".") else f"./{relative}"


def render_yaml(
    source_filename: str, page_count: int, sections: Sequence[Section], schema_ref: str
) -> str:
    """Render the TOC without requiring a YAML library."""
    quote = lambda value: json.dumps(value, ensure_ascii=False)  # noqa: E731
    lines = [
        f"# yaml-language-server: $schema={schema_ref}",
        f"source: {quote(source_filename)}",
        f"pages: {page_count}",
        "sections:",
    ]

    def emit(items: Sequence[Section], indent: int) -> None:
        pad = " " * indent
        for section in items:
            lines.extend([
                f"{pad}- id: {section.id}",
                f"{pad}  title: {quote(section.title)}",
                f"{pad}  number: {quote(section.number)}",
                f"{pad}  span_start: {section.start_page}",
                f"{pad}  span_end: {section.end_page}",
                f"{pad}  references: []",
            ])
            if section.children:
                lines.append(f"{pad}  children:")
                emit(section.children, indent + 4)
            else:
                lines.append(f"{pad}  children: []")

    emit(sections, 2)
    return "\n".join(lines) + "\n"


def write_text_atomic(path: Path, content: str) -> None:
    """Write via a temporary file so a failure never leaves a truncated output."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_toc(toc_path: Path, schema_path: Path) -> tuple[bool, str]:
    """Validate the generated YAML against a JSON Schema (YAML or JSON file)."""
    try:
        import yaml
        from jsonschema.validators import validator_for
    except ImportError as exc:
        return False, (
            "Schema validation requires PyYAML and jsonschema "
            f"(python -m pip install pyyaml jsonschema). Details: {exc}"
        )
    if not schema_path.is_file():
        return False, f"Schema file does not exist: {schema_path}"
    try:
        toc_data = yaml.safe_load(toc_path.read_text(encoding="utf-8"))
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        errors = sorted(
            validator_class(schema).iter_errors(toc_data),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        return False, f"Could not parse YAML or validate against schema: {exc}"

    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "root"
        extra = f" ({len(errors)} errors total)" if len(errors) > 1 else ""
        return False, f"{location}: {first.message}{extra}"
    return True, f"{TOC_FILENAME} is valid against the specified schema."


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, output_dir: Path) -> ExitCode:
    """Execute the PDF -> toc.yaml workflow."""
    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise TocError(f"The input file does not exist or is not a PDF: {pdf_path}")

    pymupdf = load_pymupdf()
    LOGGER.info("Input PDF: %s", pdf_path)
    LOGGER.info("Output directory: %s", output_dir)

    page_count, raw_bookmarks = read_bookmarks(pymupdf, pdf_path)
    if not 0.0 <= args.top_ratio <= 1.0:
        raise TocError("--top-ratio must be between 0 and 1.")
    sections = build_sections(raw_bookmarks, page_count, args.span_end, args.top_ratio)
    if not sections:
        LOGGER.warning("The PDF has no usable bookmarks. No %s was created.", TOC_FILENAME)
        return ExitCode.NO_TOC

    schema_path = Path(args.schema).expanduser() if args.schema else None
    toc_path = output_dir / TOC_FILENAME
    write_text_atomic(
        toc_path,
        render_yaml(pdf_path.name, page_count, sections, schema_hint(output_dir, schema_path)),
    )
    LOGGER.info("Created TOC file: %s", toc_path)
    LOGGER.info("PDF pages: %d", page_count)
    LOGGER.info("Usable TOC entries: %d", sum(1 for _ in walk(sections)))
    LOGGER.info(
        "Spans are 1-based physical PDF pages; span_end is inclusive "
        "(--span-end %s; adjacent spans may overlap by one page).",
        args.span_end.value,
    )

    if schema_path is None:
        LOGGER.info("Schema validation skipped: no --schema file was specified.")
    else:
        valid, message = validate_toc(toc_path, schema_path)
        if not valid:
            LOGGER.error("Validation failed: %s", message)
            return ExitCode.FAILURE
        LOGGER.info("Validation passed: %s", message)

    LOGGER.info("Process completed successfully.")
    return ExitCode.OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    configure_console_logging(args.verbose)
    output_dir = Path(args.output).expanduser()
    report_handler: logging.Handler | None = None
    exit_code = ExitCode.FAILURE

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_handler = attach_report_handler(output_dir / REPORT_FILENAME)
        exit_code = run(args, output_dir)
    except TocError as exc:
        LOGGER.error("%s", exc)
    except Exception as exc:  # Last-resort guard: always report and return a code.
        LOGGER.error("Processing failed: %s", exc, exc_info=args.verbose)
    finally:
        LOGGER.info("Process finished with exit code %d.", int(exit_code))
        if report_handler is not None:
            LOGGER.removeHandler(report_handler)
            report_handler.close()
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
