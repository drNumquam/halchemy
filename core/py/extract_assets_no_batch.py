#!/usr/bin/env python3
"""Extract sections, tables and figures from a technical-manual PDF with Docling.

The PDF is converted in a single pass. If the conversion reports errors or
warnings, the extractor retries page by page to isolate problematic pages.

Output:
  * one Markdown file per section;
  * one Markdown and JSON file per table;
  * one PNG per figure;
  * a master JSON index;
  * a human-readable report;
  * standalone PDF copies of problematic pages.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import pymupdf
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    DocItemLabel,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MIN_IMAGE_PX = 48
CHECKPOINT_VERSION = 3
EXCERPT_CHARS = 160

INTRO_SECTION_ID = "sec_00000_front_matter"

TOP_MAX_FRAC = 0.20
BOTTOM_MIN_FRAC = 0.80

CONFIDENCE_RANK = {"medium": 1, "high": 2}

HIGH_CONFIDENCE_REASONS = frozenset(
    {"header_repeated", "caption_marked_continued", "sequential_row_keys"}
)

CONTINUED_MARKER_RE = re.compile(
    r"\(?\s*(?:continued|cont\.?)\s*\)?\s*$",
    re.IGNORECASE,
)

SKIPPED_TEXT_LABELS = frozenset(
    {
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CAPTION,
    }
)

HEADING_NUMBER_RE = re.compile(
    r"^(?:(?:Chapter|Section|Appendix)\s+([A-Z0-9]+(?:\.[0-9]+)*)"
    r"|([0-9]+(?:\.[0-9]+)*))\b",
    re.IGNORECASE,
)

TABLE_NUMBER_RE = re.compile(
    r"(?:Table|Tbl\.?)\s+([A-Z0-9]+(?:[.\-][0-9]+)*)",
    re.IGNORECASE,
)

FIGURE_NUMBER_RE = re.compile(
    r"(?:Figure|Fig\.?)\s+([A-Z0-9]+(?:[.\-][0-9]+)*)",
    re.IGNORECASE,
)

FOOTNOTE_NUMBER_RE = re.compile(r"^\[?(\d+)\]?[.)\s]")

log = logging.getLogger("docling_extractor")


# --------------------------------------------------------------------------- #
# Logging with PDF page context
# --------------------------------------------------------------------------- #


class _PageContextFilter(logging.Filter):
    """Add the currently processed PDF page or page range to log records."""

    current = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.pages = _PageContextFilter.current
        return True


@contextlib.contextmanager
def page_context(label: str) -> Iterator[None]:
    previous = _PageContextFilter.current
    _PageContextFilter.current = label

    try:
        yield
    finally:
        _PageContextFilter.current = previous


class _WarningCollector(logging.Handler):
    """Collect third-party warnings emitted during a Docling conversion."""

    MAX_KEPT = 20
    MAX_LEN = 300

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == log.name or len(self.messages) >= self.MAX_KEPT:
            return

        message = record.getMessage()[: self.MAX_LEN]

        if message not in self.messages:
            self.messages.append(message)


@contextlib.contextmanager
def collect_warnings(enabled: bool = True) -> Iterator[list[str]]:
    collector = _WarningCollector()
    root = logging.getLogger()

    if enabled:
        root.addHandler(collector)

    try:
        yield collector.messages
    finally:
        if enabled:
            root.removeHandler(collector)


class _BlankAwareFormatter(logging.Formatter):
    """Render empty log messages as empty separator lines."""

    def format(self, record: logging.LogRecord) -> str:
        if not record.getMessage():
            return ""

        return super().format(record)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.addFilter(_PageContextFilter())

    handler.setFormatter(
        _BlankAwareFormatter(
            "[%(asctime)s] %(levelname)-8s [%(pages)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )

    logging.captureWarnings(True)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def slugify(text: str, fallback: str) -> str:
    clean = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        text.strip().lower(),
    ).strip("_")

    return clean[:60] if clean else fallback


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_name, path)

    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)

        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )


def _format_duration(seconds: float) -> str:
    total = int(seconds)

    return (
        f"{total // 3600:02d}:"
        f"{total % 3600 // 60:02d}:"
        f"{total % 60:02d}"
    )


def count_pdf_pages(pdf_path: Path) -> int:
    with pymupdf.open(str(pdf_path)) as doc:
        return len(doc)


def describe_page(pdf_path: Path, page_no: int) -> str:
    """Return a short page-text excerpt. Never raises."""
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            text = doc[page_no - 1].get_text("text")

    except Exception as exc:  # noqa: BLE001
        return f"<page text unavailable: {exc}>"

    flat = " ".join(text.split())

    return flat[:EXCERPT_CHARS] if flat else "<no extractable text>"


# --------------------------------------------------------------------------- #
# Output layout and extraction state
# --------------------------------------------------------------------------- #


@dataclass
class OutputLayout:
    root: Path

    @property
    def sections(self) -> Path:
        return self.root / "sections"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def failed_pages(self) -> Path:
        return self.root / "failed_pages"

    @property
    def tables_merged(self) -> Path:
        return self.root / "tables_merged"

    @property
    def checkpoint_file(self) -> Path:
        return self.checkpoints / "state.json"

    @property
    def index_file(self) -> Path:
        return self.root / "manual_index_raw.json"

    @property
    def report_file(self) -> Path:
        return self.root / "report.txt"

    def create(self) -> None:
        for directory in (
            self.sections,
            self.tables,
            self.images,
            self.checkpoints,
            self.failed_pages,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def _new_index(manual_name: str) -> dict[str, Any]:
    return {
        "manual_name": manual_name,
        "sections": [],
        "tables": [],
        "pictures": [],
        "footnotes": [],
        "issues": [],
    }


def _new_section(
    sec_id: str,
    toc_number: Optional[str],
    title: str,
    level: int,
    page_start: int,
) -> dict[str, Any]:
    return {
        "id": sec_id,
        "toc_number": toc_number,
        "title": title,
        "level": level,
        "page_start": page_start,
        "buffer": [],
    }


@dataclass
class ExtractionState:
    """Serializable extraction state retained for final post-processing."""

    version: int
    fingerprint: dict[str, Any]
    next_page: int
    counters: dict[str, int]
    current_section: Optional[dict[str, Any]]
    index: dict[str, Any]
    finished: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fresh(cls, fingerprint: dict[str, Any]) -> "ExtractionState":
        return cls(
            version=CHECKPOINT_VERSION,
            fingerprint=fingerprint,
            next_page=1,
            counters={
                "section": 0,
                "table": 0,
                "picture": 0,
                "picture_skipped": 0,
                "footnote": 0,
            },
            current_section=_new_section(
                INTRO_SECTION_ID,
                "0",
                "Front Matter & Introduction",
                1,
                1,
            ),
            index=_new_index(fingerprint["pdf_name"]),
        )


# --------------------------------------------------------------------------- #
# Table continuation detection and merging
# --------------------------------------------------------------------------- #


def table_geometry(
    doc: Any,
    item: Any,
    page_no: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "page_height": None,
        "top_frac": None,
        "bottom_frac": None,
        "num_rows": 0,
        "num_cols": 0,
        "header_rows": 0,
        "header_signature": [],
        "column_x_signature": [],
        "first_data_key": None,
        "last_data_key": None,
    }

    try:
        data = item.data

        info["num_rows"] = int(data.num_rows)
        info["num_cols"] = int(data.num_cols)

        header = [
            cell
            for cell in data.table_cells
            if cell.column_header
        ]

        if header:
            info["header_rows"] = max(
                cell.end_row_offset_idx
                for cell in header
            )

            ordered = sorted(
                header,
                key=lambda c: (
                    c.start_row_offset_idx,
                    c.start_col_offset_idx,
                ),
            )

            info["header_signature"] = [
                cell.text.strip()
                for cell in ordered
            ]

        by_col: dict[int, list[tuple[float, float]]] = {}

        for cell in data.table_cells:
            if cell.column_header:
                bbox = getattr(cell, "bbox", None)

                if bbox is not None:
                    by_col.setdefault(
                        int(cell.start_col_offset_idx),
                        [],
                    ).append(
                        (
                            float(bbox.l),
                            float(bbox.r),
                        )
                    )

        if by_col:
            info["column_x_signature"] = [
                [
                    round(min(x[0] for x in by_col[c]), 1),
                    round(max(x[1] for x in by_col[c]), 1),
                ]
                for c in sorted(by_col)
            ]

        data_rows: dict[int, dict[int, str]] = {}

        for cell in data.table_cells:
            if not cell.column_header:
                r = int(cell.start_row_offset_idx)
                c = int(cell.start_col_offset_idx)

                data_rows.setdefault(r, {})[c] = str(
                    cell.text
                ).strip()

        if data_rows:
            row_ids = sorted(data_rows)

            info["first_data_key"] = (
                data_rows[row_ids[0]].get(0) or None
            )

            info["last_data_key"] = (
                data_rows[row_ids[-1]].get(0) or None
            )

    except Exception:  # noqa: BLE001
        log.debug(
            "No table data on page %d",
            page_no,
            exc_info=True,
        )

    try:
        prov = item.prov[0]
        height = float(doc.pages[prov.page_no].size.height)
        box = prov.bbox.to_top_left_origin(height)

        info["page_height"] = height

        info["top_frac"] = round(
            min(max(box.t / height, 0.0), 1.0),
            4,
        )

        info["bottom_frac"] = round(
            min(max(box.b / height, 0.0), 1.0),
            4,
        )

    except Exception:  # noqa: BLE001
        log.debug(
            "No table geometry on page %d",
            page_no,
            exc_info=True,
        )

    return info


def _norm_header(signature: list[str]) -> list[str]:
    return [
        " ".join(text.lower().split())
        for text in signature
    ]


def _same_column_layout(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> bool:
    a = prev.get("column_x_signature") or []
    b = curr.get("column_x_signature") or []

    if not a or len(a) != len(b):
        return False

    try:
        return all(
            abs(float(x1) - float(y1)) <= 3.0
            and abs(float(x2) - float(y2)) <= 3.0
            for left, right in zip(a, b)
            for x1, x2, y1, y2 in [
                (left[0], left[1], right[0], right[1])
            ]
        )

    except (TypeError, ValueError, IndexError):
        return False


def _sequential_row_keys(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> bool:
    a = prev.get("last_data_key")
    b = curr.get("first_data_key")

    if not a or not b:
        return False

    ma = re.fullmatch(r"(.*?)(\d+)", str(a).strip())
    mb = re.fullmatch(r"(.*?)(\d+)", str(b).strip())

    if (
        not ma
        or not mb
        or ma.group(1).casefold() != mb.group(1).casefold()
    ):
        return False

    na, nb = ma.group(2), mb.group(2)

    return len(na) == len(nb) and int(nb) == int(na) + 1


def _continuation_reasons(
    prev: dict[str, Any],
    curr: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> Optional[list[str]]:
    if curr["page"] != prev["page"] + 1:
        return None

    prev_bottom = prev.get("bottom_frac")
    curr_top = curr.get("top_frac")
    cols = curr.get("num_cols") or 0

    if cols == 0 or cols != prev.get("num_cols"):
        return None

    edge_match = (
        prev_bottom is not None
        and curr_top is not None
        and prev_bottom >= BOTTOM_MIN_FRAC
        and curr_top <= TOP_MAX_FRAC
    )

    curr_header = _norm_header(
        curr.get("header_signature") or []
    )

    prev_header = _norm_header(
        prev.get("header_signature") or []
    )

    structural_match = (
        bool(curr_header)
        and curr_header == prev_header
        and _same_column_layout(prev, curr)
        and _sequential_row_keys(prev, curr)
    )

    if not edge_match and not structural_match:
        return None

    reasons = ["next_page", "same_cols"]

    if edge_match:
        reasons.append("bottom_to_top")

    if structural_match:
        reasons.extend(
            [
                "header_repeated",
                "same_column_layout",
                "sequential_row_keys",
            ]
        )

    caption = (curr.get("caption") or "").strip()

    if caption:
        if CONTINUED_MARKER_RE.search(caption):
            reasons.append("caption_marked_continued")

        elif (
            curr.get("is_numbered")
            and prev.get("is_numbered")
            and curr.get("label_number") == prev.get("label_number")
        ):
            reasons.append("caption_repeated")

        else:
            return None

    root = prev
    seen = {prev["id"]}

    while (
        root.get("continues_from") in by_id
        and root["continues_from"] not in seen
    ):
        root = by_id[root["continues_from"]]
        seen.add(root["id"])

    root_sig = _norm_header(
        root.get("header_signature") or []
    )

    if (
        curr_header
        and curr_header == root_sig
        and "header_repeated" not in reasons
    ):
        reasons.append("header_repeated")

    return reasons


def detect_continuations(
    tables: list[dict[str, Any]],
) -> dict[str, int]:
    by_id = {
        table["id"]: table
        for table in tables
    }

    counts = {
        "high": 0,
        "medium": 0,
    }

    for position, curr in enumerate(tables):
        curr["continues_from"] = None
        curr.pop("continuation_confidence", None)
        curr.pop("continuation_reasons", None)

        if position == 0:
            continue

        reasons = _continuation_reasons(
            tables[position - 1],
            curr,
            by_id,
        )

        if reasons is None:
            continue

        confidence = (
            "high"
            if HIGH_CONFIDENCE_REASONS & set(reasons)
            else "medium"
        )

        curr["continues_from"] = tables[position - 1]["id"]
        curr["continuation_confidence"] = confidence
        curr["continuation_reasons"] = reasons

        counts[confidence] += 1

    return counts


def _add_issue(
    index: dict[str, Any],
    page: int,
    severity: str,
    stage: str,
    message: str,
) -> None:
    index["issues"].append(
        {
            "page": page,
            "severity": severity,
            "stage": stage,
            "message": message,
            "page_excerpt": "",
            "page_pdf": None,
        }
    )


def _accepted_links(
    tables: list[dict[str, Any]],
    min_confidence: str,
) -> tuple[dict[str, str], list[tuple[int, str]]]:
    by_id = {
        table["id"]: table
        for table in tables
    }

    order = {
        table["id"]: n
        for n, table in enumerate(tables)
    }

    min_rank = CONFIDENCE_RANK[min_confidence]

    parent_of: dict[str, str] = {}
    claimed: dict[str, str] = {}
    warnings: list[tuple[int, str]] = []

    for table in tables:
        parent = table.get("continues_from")

        if not parent:
            continue

        tid = table["id"]

        if parent not in by_id or parent == tid:
            warnings.append(
                (
                    table["page"],
                    f"{tid}: continues_from {parent!r} is not a known table; link ignored",
                )
            )
            continue

        confidence = table.get("continuation_confidence")

        if (
            confidence is not None
            and CONFIDENCE_RANK.get(confidence, 0) < min_rank
        ):
            continue

        if order[parent] >= order[tid]:
            warnings.append(
                (
                    table["page"],
                    f"{tid}: continues_from {parent!r} must be an earlier table; link ignored",
                )
            )
            continue

        if parent in claimed:
            warnings.append(
                (
                    table["page"],
                    f"{tid}: {parent!r} is already continued by {claimed[parent]!r}; link ignored",
                )
            )
            continue

        parent_of[tid] = parent
        claimed[parent] = tid

    return parent_of, warnings


def _build_grid(
    rows: int,
    cols: int,
    cells: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grid: list[list[Optional[dict[str, Any]]]] = [
        [None] * cols
        for _ in range(rows)
    ]

    for cell in cells:
        for r in range(
            cell["start_row_offset_idx"],
            min(cell["end_row_offset_idx"], rows),
        ):
            for c in range(
                cell["start_col_offset_idx"],
                min(cell["end_col_offset_idx"], cols),
            ):
                grid[r][c] = cell

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                grid[r][c] = {
                    "bbox": None,
                    "row_span": 1,
                    "col_span": 1,
                    "start_row_offset_idx": r,
                    "end_row_offset_idx": r + 1,
                    "start_col_offset_idx": c,
                    "end_col_offset_idx": c + 1,
                    "text": "",
                    "column_header": False,
                    "row_header": False,
                    "row_section": False,
                    "fillable": False,
                }

    return grid  # type: ignore[return-value]


def merge_table_data(
    parts: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    first_meta, first_data = parts[0]

    first_sig = _norm_header(
        first_meta.get("header_signature") or []
    )

    cells: list[dict[str, Any]] = []
    rows = 0

    cols = max(
        int(data.get("num_cols", 0))
        for _, data in parts
    )

    info: list[dict[str, Any]] = []

    for position, (meta, data) in enumerate(parts):
        skip = 0

        if (
            position
            and first_sig
            and _norm_header(meta.get("header_signature") or []) == first_sig
        ):
            skip = int(meta.get("header_rows") or 0)

        shift = rows - skip

        for cell in data.get("table_cells", []):
            if cell["start_row_offset_idx"] < skip:
                continue

            merged_cell = dict(cell)

            merged_cell["start_row_offset_idx"] += shift
            merged_cell["end_row_offset_idx"] += shift

            if position:
                merged_cell["column_header"] = False

            merged_cell["source_table"] = meta["id"]
            merged_cell["source_page"] = meta["page"]

            cells.append(merged_cell)

        rows += int(data.get("num_rows", 0)) - skip

        info.append(
            {
                "id": meta["id"],
                "page": meta["page"],
                "header_rows_dropped": skip,
            }
        )

    merged = {
        key: value
        for key, value in first_data.items()
        if key not in (
            "table_cells",
            "grid",
            "num_rows",
            "num_cols",
        )
    }

    merged.update(
        num_rows=rows,
        num_cols=cols,
        table_cells=cells,
        grid=_build_grid(rows, cols, cells),
        merge_info={
            "parts": info,
            "page_start": parts[0][0]["page"],
            "page_end": parts[-1][0]["page"],
        },
    )

    return merged


def _md_cell(text: str) -> str:
    return " ".join(
        str(text).split()
    ).replace("|", "\\|")


def merged_table_markdown(
    merged: dict[str, Any],
    caption: str,
) -> str:
    rows = merged["num_rows"]
    cols = merged["num_cols"]

    matrix = [
        [""] * cols
        for _ in range(rows)
    ]

    for cell in merged["table_cells"]:
        r = cell["start_row_offset_idx"]
        c = cell["start_col_offset_idx"]

        if r < rows and c < cols:
            matrix[r][c] = _md_cell(
                cell.get("text", "")
            )

    lines = []

    if caption:
        info = merged["merge_info"]

        lines.append(
            f"**{caption} (pages {info['page_start']}-{info['page_end']}, "
            f"merged from {len(info['parts'])} parts)**\n"
        )

    if rows:
        lines.append(
            "| " + " | ".join(matrix[0]) + " |"
        )

        lines.append(
            "|" + "|".join(["---"] * cols) + "|"
        )

        lines.extend(
            "| " + " | ".join(row) + " |"
            for row in matrix[1:]
        )

    return "\n".join(lines) + "\n"


def merge_tables(
    layout: OutputLayout,
    index: dict[str, Any],
    min_confidence: str = "high",
) -> int:
    tables = index["tables"]

    for table in tables:
        table.pop("merged_into", None)

    index["merged_tables"] = []

    parent_of, warnings = _accepted_links(
        tables,
        min_confidence,
    )

    for page, message in warnings:
        log.warning(
            "Table merge: PDF page %d | %s",
            page,
            message,
        )

        _add_issue(
            index,
            page,
            "warning",
            "merge",
            message,
        )

    by_id = {
        table["id"]: table
        for table in tables
    }

    child_of = {
        parent: child
        for child, parent in parent_of.items()
    }

    merged_count = 0

    roots = (
        table
        for table in tables
        if table["id"] not in parent_of
        and table["id"] in child_of
    )

    for root in roots:
        chain = [root]

        while chain[-1]["id"] in child_of:
            chain.append(
                by_id[child_of[chain[-1]["id"]]]
            )

        try:
            parts = [
                (
                    meta,
                    json.loads(
                        (
                            layout.root / meta["json_path"]
                        ).read_text(encoding="utf-8")
                    ),
                )
                for meta in chain
            ]

        except (OSError, ValueError) as exc:
            message = (
                f"cannot merge chain starting at {root['id']}: {exc}"
            )

            log.warning(
                "Table merge: PDF page %d | %s",
                root["page"],
                message,
            )

            _add_issue(
                index,
                root["page"],
                "warning",
                "merge",
                message,
            )

            continue

        merged = merge_table_data(parts)

        merged_id = f"{root['id']}_merged"

        layout.tables_merged.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_file = (
            layout.tables_merged / f"{merged_id}.json"
        )

        md_file = (
            layout.tables_merged / f"{merged_id}.md"
        )

        atomic_write_json(json_file, merged)

        atomic_write_text(
            md_file,
            merged_table_markdown(
                merged,
                root.get("caption", ""),
            ),
        )

        index["merged_tables"].append(
            {
                "id": merged_id,
                "root_table": root["id"],
                "parts": [
                    meta["id"]
                    for meta in chain
                ],
                "page_start": chain[0]["page"],
                "page_end": chain[-1]["page"],
                "num_rows": merged["num_rows"],
                "num_cols": merged["num_cols"],
                "caption": root.get("caption", ""),
                "md_path": layout.rel(md_file),
                "json_path": layout.rel(json_file),
            }
        )

        for meta in chain:
            meta["merged_into"] = merged_id

        merged_count += 1

        log.info(
            "[TABLE MERGED] %d parti | pagine %d-%d | %s",
            len(chain),
            chain[0]["page"],
            chain[-1]["page"],
            merged_id,
        )

    return merged_count


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _continuation_report_lines(
    index: dict[str, Any],
    sub: str,
) -> list[str]:
    tables = index["tables"]

    links = [
        table
        for table in tables
        if table.get("continues_from")
    ]

    high = sum(
        1
        for table in links
        if table.get("continuation_confidence") == "high"
    )

    medium = sum(
        1
        for table in links
        if table.get("continuation_confidence") == "medium"
    )

    merged = index.get("merged_tables", [])

    lines = [
        "TABLE CONTINUATIONS",
        sub,
        f"Links found    : {len(links)} (high: {high}, medium: {medium})",
        f"Merged tables  : {len(merged)}",
    ]

    for entry in merged[:50]:
        lines.append(
            f"  {entry['id']}: pages {entry['page_start']}-{entry['page_end']}, "
            f"{len(entry['parts'])} parts, "
            f"{entry['num_rows']}x{entry['num_cols']}"
        )

    unmerged = [
        table
        for table in links
        if not table.get("merged_into")
    ]

    if unmerged:
        lines.append(
            f"Links NOT merged ({len(unmerged)}; review or lower --merge-min-confidence):"
        )

        for table in unmerged[:50]:
            lines.append(
                f"  page {table['page'] - 1} -> {table['page']} "
                f"[{table.get('continuation_confidence', 'manual')}] "
                f"{table['continues_from']} -> {table['id']}"
            )

    return lines


def build_report(
    state: ExtractionState,
    layout: OutputLayout,
    pdf_path: Path,
    min_image_px: int,
    elapsed_s: float,
    aborted: bool,
    note: Optional[str] = None,
) -> str:
    total_pages = state.fingerprint["total_pages"]

    issues = sorted(
        state.index["issues"],
        key=lambda issue: (
            issue["page"],
            issue["severity"],
        ),
    )

    errors = [
        issue
        for issue in issues
        if issue["severity"] == "error"
    ]

    warnings_ = [
        issue
        for issue in issues
        if issue["severity"] == "warning"
    ]

    if aborted or not state.finished:
        status = (
            "INCOMPLETE (single-pass extraction interrupted; "
            "no resumable intermediate checkpoint)"
        )

    elif errors:
        status = "COMPLETE WITH ERRORS"

    else:
        status = "COMPLETE"

    rule = "=" * 72
    sub = "-" * 72

    lines: list[str] = [
        rule,
        "DOCLING EXTRACTION REPORT",
        rule,
        f"Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status         : {status}",
        "",
        "INPUT / SETTINGS",
        sub,
        f"PDF            : {pdf_path}",
        f"Total pages    : {total_pages}",
        f"Pages processed: {total_pages if state.finished else 'incomplete'}",
        (
            note
            if note is not None
            else f"This run       : single-pass conversion in {_format_duration(elapsed_s)}"
        ),
        "Conversion     : entire PDF in one pass, with page-level retry on problems",
        f"Min figure size: {min_image_px}px",
        "",
        "TOTALS",
        sub,
        f"Sections       : {len(state.index['sections'])}",
        f"Tables         : {len(state.index['tables'])}",
        f"Figures        : {len(state.index['pictures'])}"
        f" (skipped as too small: {state.counters['picture_skipped']})",
        f"Footnotes      : {len(state.index['footnotes'])}",
        "",
        *_continuation_report_lines(state.index, sub),
        "",
        "ISSUES",
        sub,
        f"Errors         : {len(errors)}",
        f"Warnings       : {len(warnings_)}",
        (
            "Error pages    : "
            + (
                ", ".join(
                    str(page)
                    for page in sorted(
                        {issue["page"] for issue in errors}
                    )
                )
                or "-"
            )
        ),
        (
            "Warning pages  : "
            + (
                ", ".join(
                    str(page)
                    for page in sorted(
                        {issue["page"] for issue in warnings_}
                    )
                )
                or "-"
            )
        ),
    ]

    if issues:
        lines += [
            "",
            "ISSUE DETAILS (sorted by PDF page)",
            sub,
        ]

        for n, issue in enumerate(issues, start=1):
            lines.append(
                f"{n:>3}. PDF page {issue['page']} | "
                f"{issue['severity'].upper()} | {issue['stage']}"
            )

            lines.append(
                textwrap.fill(
                    f"Message  : {issue['message']}",
                    width=100,
                    initial_indent="     ",
                    subsequent_indent="              ",
                )
            )

            lines.append(
                textwrap.fill(
                    f"Page text: {issue.get('page_excerpt') or '-'}",
                    width=100,
                    initial_indent="     ",
                    subsequent_indent="              ",
                )
            )

            lines.append(
                f"     Page copy: {issue.get('page_pdf') or '<not saved>'}"
            )

    lines += [
        "",
        "ARTIFACTS",
        sub,
        f"Output root    : {layout.root}",
        f"Sections       : {layout.rel(layout.sections)}/",
        f"Tables         : {layout.rel(layout.tables)}/",
        f"Images         : {layout.rel(layout.images)}/",
        f"Failed pages   : {layout.rel(layout.failed_pages)}/",
        f"Checkpoint     : {layout.rel(layout.checkpoint_file)}",
        f"Index          : {layout.rel(layout.index_file)}",
    ]

    if state.index.get("merged_tables"):
        lines.append(
            f"Merged tables  : {layout.rel(layout.tables_merged)}/"
        )

    lines += [
        rule,
        "",
    ]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Memory management
# --------------------------------------------------------------------------- #


def release_memory() -> None:
    gc.collect()

    try:
        import torch  # noqa: WPS433

    except ImportError:
        return

    mps = getattr(torch.backends, "mps", None)

    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


class ManualExtractor:
    """Convert one PDF in a single Docling pass and extract its contents."""

    def __init__(
        self,
        pdf_path: Path,
        layout: OutputLayout,
        min_image_px: int,
        locate_warnings: bool = True,
        merge_tables: bool = False,
        merge_min_confidence: str = "high",
    ) -> None:
        self.pdf_path = pdf_path
        self.layout = layout
        self.min_image_px = min_image_px
        self.locate_warnings = locate_warnings
        self.merge_tables_enabled = merge_tables
        self.merge_min_confidence = merge_min_confidence

        self.total_pages = 0
        self.converter = self._create_converter()

        self.state: ExtractionState
        self._started_at = time.monotonic()

    @staticmethod
    def _create_converter() -> DocumentConverter:
        opts = PdfPipelineOptions()

        opts.do_ocr = False
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.generate_picture_images = True
        opts.images_scale = 1.5

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts
                )
            }
        )

    def run(self) -> int:
        """Run extraction. Returns 0 on success and 3 if errors were recorded."""
        self.layout.create()

        self.total_pages = count_pdf_pages(self.pdf_path)

        fingerprint = {
            "pdf_name": self.pdf_path.name,
            "pdf_size": self.pdf_path.stat().st_size,
            "total_pages": self.total_pages,
        }

        self.state = ExtractionState.fresh(fingerprint)
        self._started_at = time.monotonic()

        self.state.settings = {
            "conversion_mode": "single_pass",
            "min_image_px": self.min_image_px,
            "locate_warnings": self.locate_warnings,
        }

        log.info("=" * 72)
        log.info("DOCLING ASSET EXTRACTOR")
        log.info("=" * 72)
        log.info("PDF        : %s", self.pdf_path)
        log.info("Pages      : %d", self.total_pages)
        log.info("Conversion : entire document in one pass")
        log.info("OCR        : disabled")
        log.info("Tables     : accurate mode")
        log.info("Images     : enabled")
        log.info("Output     : %s", self.layout.root)
        log.info("=" * 72)

        try:
            with page_context(f"p1-{self.total_pages}"):
                self._process_document()

            self.state.next_page = self.total_pages + 1
            self._finalize()

        except BaseException:
            log.critical(
                "Run aborted. The single-pass conversion has no intermediate "
                "checkpoint; restart the extraction from page 1."
            )

            self._write_report(aborted=True)
            raise

        has_errors = any(
            issue["severity"] == "error"
            for issue in self.state.index["issues"]
        )

        return 3 if has_errors else 0

    def _convert(
        self,
        first: Optional[int] = None,
        last: Optional[int] = None,
    ) -> tuple[Any, Optional[str]]:
        """Convert the whole PDF, or one page when a range is supplied."""
        with collect_warnings(self.locate_warnings) as warned:
            try:
                if first is None and last is None:
                    result = self.converter.convert(
                        str(self.pdf_path)
                    )

                elif first is not None and last is not None:
                    result = self.converter.convert(
                        str(self.pdf_path),
                        page_range=(first, last),
                    )

                else:
                    raise ValueError(
                        "Both first and last page must be supplied together"
                    )

            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "Docling conversion raised",
                    exc_info=True,
                )

                return None, f"{type(exc).__name__}: {exc}"

        problems: list[str] = []

        if result.status != ConversionStatus.SUCCESS:
            problems.append(
                f"status={result.status.name}"
            )

        for err in getattr(result, "errors", None) or []:
            problems.append(
                str(
                    getattr(err, "error_message", None)
                    or repr(err)
                )
            )

        problems.extend(
            f"docling warning: {message}"
            for message in warned
        )

        return result, (
            "; ".join(problems)
            or None
        )

    def _process_document(self) -> None:
        """Convert the whole PDF; retry page by page if conversion reports problems."""
        log.info(
            "[CONVERSION START] Avvio conversione integrale: %d pagine",
            self.total_pages,
        )

        log.info(
            "[CONVERSION] Docling sta analizzando il PDF. "
            "L'avanzamento interno per pagina non è esposto da Docling."
        )

        started = time.monotonic()

        result, problem = self._convert()

        conversion_time = time.monotonic() - started

        if problem is None and result is not None:
            log.info(
                "[CONVERSION COMPLETE] Documento convertito in %s",
                _format_duration(conversion_time),
            )

            log.info(
                "[EXTRACTION START] Inizio estrazione degli elementi"
            )

            extraction_started = time.monotonic()

            self._consume(
                result,
                default_page=1,
            )

            log.info(
                "[EXTRACTION COMPLETE] Estrazione completata in %s",
                _format_duration(
                    time.monotonic() - extraction_started
                ),
            )

            del result
            release_memory()

            return

        log.warning(
            "[CONVERSION WARNING] Conversione integrale problematica: %s",
            (problem or "nessun risultato di conversione")[:500],
        )

        log.warning(
            "[RECOVERY START] Avvio conversione individuale delle %d pagine",
            self.total_pages,
        )

        if result is not None:
            del result
            release_memory()

        recovery_started = time.monotonic()

        successful_pages = 0
        failed_pages = 0

        for page in range(1, self.total_pages + 1):
            page_started = time.monotonic()

            with page_context(
                f"p{page}/{self.total_pages}"
            ):
                log.info(
                    "[PAGE %d/%d] Conversione pagina in corso...",
                    page,
                    self.total_pages,
                )

                page_result, page_problem = self._convert(
                    page,
                    page,
                )

                if page_result is None:
                    failed_pages += 1

                    self._handle_page_result(
                        page,
                        page_result,
                        page_problem,
                    )

                else:
                    successful_pages += 1

                    self._handle_page_result(
                        page,
                        page_result,
                        page_problem,
                    )

                log.info(
                    "[PAGE %d/%d] Completata in %.1f secondi",
                    page,
                    self.total_pages,
                    time.monotonic() - page_started,
                )

            release_memory()

        log.info(
            "[RECOVERY COMPLETE] Pagine riuscite: %d/%d | fallite: %d | durata: %s",
            successful_pages,
            self.total_pages,
            failed_pages,
            _format_duration(
                time.monotonic() - recovery_started
            ),
        )

    def _save_checkpoint(self) -> None:
        """Persist the final state for post-processing, not for resuming extraction."""
        atomic_write_json(
            self.layout.checkpoint_file,
            asdict(self.state),
        )

    def _finalize(self) -> None:
        """Flush the final section and write index, checkpoint and report."""
        self._flush_section()

        self.state.current_section = None
        self.state.finished = True

        log.info("[FINALIZATION] Analisi delle continuazioni delle tabelle")

        counts = detect_continuations(
            self.state.index["tables"]
        )

        log.info(
            "Table continuations detected: %d high, %d medium",
            counts["high"],
            counts["medium"],
        )

        merged = 0

        if self.merge_tables_enabled:
            log.info("[FINALIZATION] Fusione delle tabelle continuate")

            merged = merge_tables(
                self.layout,
                self.state.index,
                self.merge_min_confidence,
            )

        log.info("[FINALIZATION] Scrittura checkpoint, indice e report")

        self._save_checkpoint()

        atomic_write_json(
            self.layout.index_file,
            self.state.index,
        )

        self._write_report(aborted=False)

        counters = self.state.counters
        issues = self.state.index["issues"]

        errors = [
            issue
            for issue in issues
            if issue["severity"] == "error"
        ]

        elapsed = time.monotonic() - self._started_at

        log.info("")
        log.info("=" * 56)
        log.info("FINAL SUMMARY")
        log.info("=" * 56)
        log.info("Pages    : %d", self.total_pages)
        log.info("Duration : %s", _format_duration(elapsed))
        log.info("Sections : %d", counters["section"])
        log.info("Tables   : %d", counters["table"])

        log.info(
            "Figures  : %d (skipped as too small: %d)",
            counters["picture"],
            counters["picture_skipped"],
        )

        log.info("Footnotes: %d", counters["footnote"])
        log.info("Tables merged: %d", merged)
        log.info(
            "Issues   : %d (%d errors)",
            len(issues),
            len(errors),
        )

        for issue in errors:
            log.info(
                "  - PDF page %d [%s]: %s",
                issue["page"],
                issue["stage"],
                issue["message"],
            )

        log.info("Index    : %s", self.layout.index_file)
        log.info("Report   : %s", self.layout.report_file)
        log.info("=" * 56)

    def _write_report(self, aborted: bool) -> None:
        """Write the report on a best-effort basis."""
        try:
            atomic_write_text(
                self.layout.report_file,
                build_report(
                    state=self.state,
                    layout=self.layout,
                    pdf_path=self.pdf_path,
                    min_image_px=self.min_image_px,
                    elapsed_s=time.monotonic() - self._started_at,
                    aborted=aborted,
                ),
            )

        except Exception:  # noqa: BLE001
            log.exception(
                "Could not write report %s",
                self.layout.report_file,
            )

    # -- conversion result handling -------------------------------------- #

    def _handle_page_result(
        self,
        page: int,
        result: Any,
        problem: Optional[str],
    ) -> None:
        """Consume one page; record conversion failures without stopping the run."""
        if result is None:
            self._report_issue(
                page,
                "error",
                "convert",
                problem or "conversion failed",
            )

            self._append_block(
                f"[PAGE {page} COULD NOT BE CONVERTED]"
            )

            return

        if problem:
            self._report_issue(
                page,
                "warning",
                "convert",
                problem,
            )

        self._consume(
            result,
            default_page=page,
        )

        del result

    # -- item handling ---------------------------------------------------- #

    def _consume(
        self,
        result: Any,
        default_page: int,
    ) -> None:
        """Walk a Docling document and log page and item-level progress."""
        doc = result.document

        last_page = default_page

        page_items = 0
        page_tables = 0
        page_pictures = 0
        page_texts = 0
        page_sections = 0

        processed_pages: set[int] = set()

        def finish_page(page_no: int) -> None:
            log.info(
                "[PAGE %d/%d COMPLETE] Elementi: %d | sezioni: %d | "
                "testi: %d | tabelle: %d | figure: %d",
                page_no,
                self.total_pages,
                page_items,
                page_sections,
                page_texts,
                page_tables,
                page_pictures,
            )

        for item, _depth in doc.iterate_items():
            prov = getattr(item, "prov", None)

            page_no = (
                prov[0].page_no
                if prov
                else last_page
            )

            if page_no != last_page and page_items:
                finish_page(last_page)

                page_items = 0
                page_tables = 0
                page_pictures = 0
                page_texts = 0
                page_sections = 0

            last_page = page_no

            if page_no not in processed_pages:
                processed_pages.add(page_no)

                log.info(
                    "[PAGE %d/%d START] Analisi degli elementi...",
                    page_no,
                    self.total_pages,
                )

            if isinstance(item, SectionHeaderItem):
                item_type = "SECTION"
                page_sections += 1

            elif isinstance(item, TableItem):
                item_type = "TABLE"
                page_tables += 1

            elif isinstance(item, PictureItem):
                item_type = "FIGURE"
                page_pictures += 1

            elif isinstance(item, TextItem):
                item_type = "TEXT"
                page_texts += 1

            else:
                item_type = type(item).__name__.upper()

            page_items += 1

            description = ""

            if isinstance(item, (SectionHeaderItem, TextItem)):
                description = " ".join(
                    str(getattr(item, "text", "")).split()
                )[:100]

            elif isinstance(item, TableItem):
                description = "tabella rilevata"

            elif isinstance(item, PictureItem):
                description = "figura rilevata"

            with page_context(
                f"p{page_no}/{self.total_pages}"
            ):
                log.info(
                    "[%s] %s%s",
                    item_type,
                    description,
                    "..." if len(description) >= 100 else "",
                )

                try:
                    self._handle_item(
                        doc,
                        item,
                        page_no,
                    )

                except Exception as exc:  # noqa: BLE001
                    self._report_issue(
                        page_no,
                        "error",
                        "item",
                        (
                            f"{type(item).__name__} "
                            f"({getattr(item, 'label', '?')}): "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        exc=exc,
                    )

        if page_items:
            finish_page(last_page)

        log.info(
            "[EXTRACTION PROGRESS] Pagine con elementi elaborati: %d/%d",
            len(processed_pages),
            self.total_pages,
        )

    def _handle_item(
        self,
        doc: Any,
        item: Any,
        page_no: int,
    ) -> None:
        if isinstance(item, SectionHeaderItem):
            self._handle_section_header(
                item,
                page_no,
            )

        elif isinstance(item, TextItem):
            self._handle_text(
                item,
                page_no,
            )

        elif isinstance(item, TableItem):
            self._handle_table(
                doc,
                item,
                page_no,
            )

        elif isinstance(item, PictureItem):
            self._handle_picture(
                doc,
                item,
                page_no,
            )

    def _next_counter(self, kind: str) -> int:
        self.state.counters[kind] += 1
        return self.state.counters[kind]

    def _append_block(self, block: str) -> None:
        assert self.state.current_section is not None

        self.state.current_section["buffer"].append(block)

    def _flush_section(self) -> None:
        sec = self.state.current_section

        if (
            sec is None
            or (
                not sec["buffer"]
                and sec["id"] == INTRO_SECTION_ID
            )
        ):
            return

        md_path = (
            self.layout.sections / f"{sec['id']}.md"
        )

        md_path.write_text(
            f"# {sec['title']}\n\n"
            + "\n\n".join(sec["buffer"])
            + "\n",
            encoding="utf-8",
        )

        self.state.index["sections"].append(
            {
                "id": sec["id"],
                "toc_number": sec["toc_number"],
                "title": sec["title"],
                "level": sec["level"],
                "page_start": sec["page_start"],
                "content_path": self.layout.rel(md_path),
            }
        )

        log.info(
            "[SECTION SAVED] %s | pagina iniziale: %d | blocchi: %d",
            sec["title"],
            sec["page_start"],
            len(sec["buffer"]),
        )

    def _handle_section_header(
        self,
        item: SectionHeaderItem,
        page_no: int,
    ) -> None:
        self._flush_section()

        n = self._next_counter("section")
        heading = item.text.strip()

        match = HEADING_NUMBER_RE.match(heading)

        toc_number = (
            (match.group(1) or match.group(2))
            if match
            else None
        )

        sec_id = (
            f"sec_{n:05d}_"
            f"{slugify(toc_number or heading, 'untitled')}"
        )

        self.state.current_section = _new_section(
            sec_id,
            toc_number,
            heading,
            int(getattr(item, "level", 1) or 1),
            page_no,
        )

        log.info(
            "[SECTION #%d] Pagina %d | %s",
            n,
            page_no,
            heading,
        )

    def _handle_text(
        self,
        item: TextItem,
        page_no: int,
    ) -> None:
        if item.label in SKIPPED_TEXT_LABELS:
            return

        text = item.text.strip()

        if not text:
            return

        if item.label == DocItemLabel.FOOTNOTE:
            n = self._next_counter("footnote")

            match = FOOTNOTE_NUMBER_RE.match(text)

            fn_num = (
                match.group(1)
                if match
                else str(n)
            )

            assert self.state.current_section is not None

            self.state.index["footnotes"].append(
                {
                    "id": f"fn_{n:05d}_p{page_no}",
                    "footnote_number": fn_num,
                    "page": page_no,
                    "parent_section": self.state.current_section["id"],
                    "text": text,
                }
            )

            self._append_block(
                f"> **[Footnote {fn_num}]**: {text}"
            )

            log.info(
                "[FOOTNOTE #%d] Pagina %d | numero=%s",
                n,
                page_no,
                fn_num,
            )

        elif isinstance(item, ListItem):
            self._append_block(
                f"- {text}"
            )

        else:
            self._append_block(text)

    @staticmethod
    def _bbox(item: Any) -> list[float]:
        prov = getattr(item, "prov", None)

        return (
            list(prov[0].bbox.as_tuple())
            if prov
            else []
        )

    def _handle_table(
        self,
        doc: Any,
        item: TableItem,
        page_no: int,
    ) -> None:
        n = self._next_counter("table")

        caption = item.caption_text(doc).strip()

        match = TABLE_NUMBER_RE.search(caption)

        label = (
            match.group(1).replace("-", ".")
            if match
            else f"p{page_no}"
        )

        tbl_id = (
            f"tbl_{n:05d}_{slugify(label, 'x')}"
        )

        md_file = (
            self.layout.tables / f"{tbl_id}.md"
        )

        json_file = (
            self.layout.tables / f"{tbl_id}.json"
        )

        data = item.data

        log.info(
            "[TABLE #%d] Pagina %d | ID=%s | righe=%s | colonne=%s | caption=%s",
            n,
            page_no,
            tbl_id,
            getattr(data, "num_rows", "?"),
            getattr(data, "num_cols", "?"),
            caption or "<senza didascalia>",
        )

        body = item.export_to_markdown(doc)

        md_file.write_text(
            (
                f"**{caption}**\n\n"
                if caption
                else ""
            )
            + body
            + "\n",
            encoding="utf-8",
        )

        atomic_write_json(
            json_file,
            item.data.model_dump(),
        )

        log.info(
            "[TABLE SAVED] Markdown: %s | JSON: %s",
            md_file.name,
            json_file.name,
        )

        assert self.state.current_section is not None

        self.state.index["tables"].append(
            {
                "id": tbl_id,
                "label_number": label,
                "is_numbered": bool(match),
                "page": page_no,
                "bbox": self._bbox(item),
                "parent_section": self.state.current_section["id"],
                "caption": caption,
                "caption_continued": bool(
                    CONTINUED_MARKER_RE.search(caption)
                ),
                **table_geometry(
                    doc,
                    item,
                    page_no,
                ),
                "continues_from": None,
                "md_path": self.layout.rel(md_file),
                "json_path": self.layout.rel(json_file),
            }
        )

        self._append_block(
            f"[TABLE REF: {tbl_id} -> {md_file.name}]"
        )

    def _handle_picture(
        self,
        doc: Any,
        item: PictureItem,
        page_no: int,
    ) -> None:
        pil_img = item.get_image(doc)

        if (
            pil_img is not None
            and min(pil_img.size) < self.min_image_px
        ):
            self.state.counters["picture_skipped"] += 1

            log.info(
                "[FIGURE SKIPPED] Pagina %d | dimensioni=%sx%s px | minimo=%d px",
                page_no,
                pil_img.width,
                pil_img.height,
                self.min_image_px,
            )

            return

        n = self._next_counter("picture")

        caption = item.caption_text(doc).strip()

        match = FIGURE_NUMBER_RE.search(caption)

        label = (
            match.group(1).replace("-", ".")
            if match
            else f"p{page_no}"
        )

        fig_id = (
            f"fig_{n:05d}_{slugify(label, 'x')}"
        )

        img_file = (
            self.layout.images / f"{fig_id}.png"
        )

        if pil_img is not None:
            pil_img.save(
                img_file,
                format="PNG",
            )

        log.info(
            "[FIGURE #%d SAVED] Pagina %d | ID=%s | file=%s | dimensioni=%s",
            n,
            page_no,
            fig_id,
            img_file.name,
            (
                f"{pil_img.width}x{pil_img.height}px"
                if pil_img
                else "immagine non disponibile"
            ),
        )

        assert self.state.current_section is not None

        self.state.index["pictures"].append(
            {
                "id": fig_id,
                "label_number": label,
                "is_numbered": bool(match),
                "page": page_no,
                "bbox": self._bbox(item),
                "parent_section": self.state.current_section["id"],
                "caption": caption,
                "image_path": (
                    self.layout.rel(img_file)
                    if pil_img is not None
                    else None
                ),
            }
        )

        self._append_block(
            f"[FIGURE REF: {fig_id} -> {img_file.name}]"
        )

    # -- diagnostics ------------------------------------------------------ #

    def _export_page(
        self,
        page_no: int,
    ) -> Optional[str]:
        """Save a standalone copy of a problematic PDF page."""
        dest = (
            self.layout.failed_pages
            / f"page_{page_no:05d}.pdf"
        )

        try:
            if not dest.exists():
                with (
                    pymupdf.open(str(self.pdf_path)) as src,
                    pymupdf.open() as out,
                ):
                    out.insert_pdf(
                        src,
                        from_page=page_no - 1,
                        to_page=page_no - 1,
                    )

                    out.save(str(dest))

            return self.layout.rel(dest)

        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Could not export page %d: %s",
                page_no,
                exc,
            )

            return None

    def _report_issue(
        self,
        page_no: int,
        severity: str,
        stage: str,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        excerpt = describe_page(
            self.pdf_path,
            page_no,
        )

        saved = self._export_page(page_no)

        log.log(
            (
                logging.ERROR
                if severity == "error"
                else logging.WARNING
            ),
            (
                "PDF PAGE %d | stage=%s | %s | "
                "page text: %r | page copy: %s"
            ),
            page_no,
            stage,
            message,
            excerpt,
            saved or "<not saved>",
            exc_info=exc,
        )

        self.state.index["issues"].append(
            {
                "page": page_no,
                "severity": severity,
                "stage": stage,
                "message": message,
                "page_excerpt": excerpt,
                "page_pdf": saved,
            }
        )


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def _enrich_continuation_signals(
    layout: OutputLayout,
    tables: list[dict[str, Any]],
) -> None:
    """Backfill structural continuation signals from table JSON files."""
    for meta in tables:
        rel = meta.get("json_path")

        if not rel:
            continue

        path = layout.root / rel

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            cells = data.get("table_cells", [])

            header = [
                cell
                for cell in cells
                if cell.get("column_header")
            ]

            if header and not meta.get("column_x_signature"):
                by_col: dict[int, list[tuple[float, float]]] = {}

                for cell in header:
                    bbox = cell.get("bbox") or {}

                    if "l" in bbox and "r" in bbox:
                        by_col.setdefault(
                            int(
                                cell.get(
                                    "start_col_offset_idx",
                                    0,
                                )
                            ),
                            [],
                        ).append(
                            (
                                float(bbox["l"]),
                                float(bbox["r"]),
                            )
                        )

                if by_col:
                    meta["column_x_signature"] = [
                        [
                            round(min(x[0] for x in by_col[c]), 1),
                            round(max(x[1] for x in by_col[c]), 1),
                        ]
                        for c in sorted(by_col)
                    ]

            rows: dict[int, dict[int, str]] = {}

            for cell in cells:
                if not cell.get("column_header"):
                    r = int(
                        cell.get("start_row_offset_idx", 0)
                    )

                    c = int(
                        cell.get("start_col_offset_idx", 0)
                    )

                    rows.setdefault(r, {})[c] = str(
                        cell.get("text", "")
                    ).strip()

            if rows:
                ordered_rows = sorted(rows)

                meta.setdefault(
                    "first_data_key",
                    rows[ordered_rows[0]].get(0) or None,
                )

                meta.setdefault(
                    "last_data_key",
                    rows[ordered_rows[-1]].get(0) or None,
                )

        except (OSError, ValueError, TypeError):
            log.debug(
                "Could not read table JSON for continuation signals: %s",
                path,
                exc_info=True,
            )


def run_postprocess(
    layout: OutputLayout,
    min_confidence: str,
    redetect: bool,
) -> int:
    """Merge continuation tables without running PDF conversion."""
    ckpt = layout.checkpoint_file
    index_file = layout.index_file

    if not ckpt.exists() or not index_file.exists():
        log.error(
            "Nothing to post-process: %s and %s must both exist "
            "(run the extraction first).",
            ckpt,
            index_file,
        )

        return 1

    try:
        state = ExtractionState(
            **json.loads(
                ckpt.read_text(encoding="utf-8")
            )
        )

        state.index = json.loads(
            index_file.read_text(encoding="utf-8")
        )

    except (OSError, ValueError, TypeError) as exc:
        log.error(
            "Cannot read the extraction output in %s: %s",
            layout.root,
            exc,
        )

        return 1

    if state.version != CHECKPOINT_VERSION:
        log.error(
            "Output was produced by an incompatible script version "
            "(checkpoint v%s, expected v%s); re-run the extraction.",
            state.version,
            CHECKPOINT_VERSION,
        )

        return 1

    if not state.finished:
        log.error(
            "Extraction is not finished. Run the extraction again "
            "before post-processing."
        )

        return 1

    log.info("=== TABLE POST-PROCESSING ===")
    log.info("Output: %s", layout.root)

    if redetect:
        _enrich_continuation_signals(
            layout,
            state.index["tables"],
        )

        counts = detect_continuations(
            state.index["tables"]
        )

        log.info(
            "Re-detected continuations: %d high, %d medium",
            counts["high"],
            counts["medium"],
        )

    merged = merge_tables(
        layout,
        state.index,
        min_confidence,
    )

    atomic_write_json(
        index_file,
        state.index,
    )

    atomic_write_json(
        ckpt,
        asdict(state),
    )

    settings = state.settings or {}

    atomic_write_text(
        layout.report_file,
        build_report(
            state=state,
            layout=layout,
            pdf_path=Path(state.fingerprint["pdf_name"]),
            min_image_px=int(
                settings.get(
                    "min_image_px",
                    DEFAULT_MIN_IMAGE_PX,
                )
            ),
            elapsed_s=0.0,
            aborted=False,
            note=(
                "This run       : table post-processing only "
                "(no pages converted)"
            ),
        ),
    )

    log.info(
        "Tables merged: %d | Report: %s",
        merged,
        layout.report_file,
    )

    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(
    argv: list[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sections, tables and figures from a PDF with Docling."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        help="input PDF file (not needed with --postprocess-only)",
    )

    parser.add_argument(
        "--output",
        "--out-dir",
        dest="output",
        type=Path,
        default=Path("./output_manuale"),
        metavar="DIR",
        help=(
            "output root directory for all artifacts, created if missing "
            "(alias: --out-dir; default: ./output_manuale)"
        ),
    )

    parser.add_argument(
        "--min-image-px",
        type=int,
        default=DEFAULT_MIN_IMAGE_PX,
        metavar="PX",
        help=(
            "skip figures with width or height below PX "
            f"(default: {DEFAULT_MIN_IMAGE_PX})"
        ),
    )

    parser.add_argument(
        "--no-locate-warnings",
        action="store_true",
        help=(
            "do not retry the PDF page by page when Docling reports warnings; "
            "warnings will be attributed to the whole document"
        ),
    )

    parser.add_argument(
        "--merge-tables",
        action="store_true",
        help="after extraction, merge tables that continue across pages",
    )

    parser.add_argument(
        "--merge-min-confidence",
        choices=sorted(
            CONFIDENCE_RANK,
            key=CONFIDENCE_RANK.get,
            reverse=True,
        ),
        default="high",
        help=(
            "weakest continuation link that is merged "
            "(default: high)"
        ),
    )

    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "do not convert: merge continuation tables of the finished "
            "output in --output and refresh index and report"
        ),
    )

    parser.add_argument(
        "--redetect",
        action="store_true",
        help=(
            "with --postprocess-only: recompute continues_from from "
            "stored geometry, overwriting hand edits"
        ),
    )

    args = parser.parse_args(argv)

    if args.redetect and not args.postprocess_only:
        parser.error(
            "--redetect requires --postprocess-only"
        )

    if not args.postprocess_only and args.pdf is None:
        parser.error(
            "the PDF argument is required unless --postprocess-only is used"
        )

    if args.min_image_px < 0:
        parser.error("--min-image-px must be >= 0")

    return args


def main(
    argv: Optional[list[str]] = None,
) -> int:
    args = parse_args(
        sys.argv[1:]
        if argv is None
        else argv
    )

    configure_logging()

    if args.output.exists() and not args.output.is_dir():
        log.error(
            "--output exists but is not a directory: %s",
            args.output,
        )

        return 1

    if args.postprocess_only:
        if args.pdf is not None:
            log.warning(
                "--postprocess-only: the PDF argument is ignored"
            )

        try:
            return run_postprocess(
                OutputLayout(args.output),
                args.merge_min_confidence,
                args.redetect,
            )

        except Exception:  # noqa: BLE001
            log.exception(
                "Fatal error during post-processing"
            )

            return 1

    if not args.pdf.is_file():
        log.error(
            "Input PDF not found: %s",
            args.pdf,
        )

        return 1

    extractor = ManualExtractor(
        pdf_path=args.pdf,
        layout=OutputLayout(args.output),
        min_image_px=args.min_image_px,
        locate_warnings=not args.no_locate_warnings,
        merge_tables=args.merge_tables,
        merge_min_confidence=args.merge_min_confidence,
    )

    try:
        return extractor.run()

    except KeyboardInterrupt:
        log.warning("Interruzione richiesta dall'utente.")
        return 130

    except Exception:  # noqa: BLE001
        log.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
