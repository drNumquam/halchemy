#!/usr/bin/env python3
"""Batch extractor: split a large technical-manual PDF into sections, tables and figures.

Purpose
-------
Convert a (potentially very large) PDF with Docling, in page batches, into:
  * one Markdown file per section,
  * one Markdown + JSON file per table,
  * one PNG per figure,
  * a master JSON index (``manual_index_raw.json``) tying everything together,
  * a human-readable run report (``report.txt``) next to the index.

Usage
-----
    python docling_extractor.py MANUAL.pdf [--batchsize N] [--output DIR]
                                           [--min-image-px PX] [--no-locate-warnings]
                                           [--no-resume] [--merge-tables]
                                           [--merge-min-confidence {high,medium}]
    python docling_extractor.py --postprocess-only --output DIR
                                           [--merge-min-confidence {high,medium}]
                                           [--redetect]

    --batchsize N      Pages per batch. ``0`` converts the whole document in a
                       single pass (fastest, but no intermediate checkpoints and
                       highest memory use). Otherwise N must be >= 10.
                       Default: 100.
    --output DIR       Output root directory; ALL artifacts (sections, tables,
                       images, failed pages, checkpoint, index) are written
                       beneath it. Created (with parents) if missing. May be
                       relative or absolute. ``--out-dir`` is accepted as an
                       alias. Default: ./output_manuale
                       Resume is per output directory: the checkpoint lives
                       inside it, so reuse the same DIR to resume and use a
                       different DIR for an independent run of the same PDF.
    --no-locate-warnings
                       Skip page-level isolation of Docling warnings (faster;
                       warnings stay attributed to the whole batch range only).
                       See "Warning attribution" below.
    --min-image-px PX  Figures whose width or height is below PX pixels are
                       skipped (icons, bullets, rules). Default: 48.
    --merge-tables     After extraction, merge tables that continue across pages
                       into single tables (see "Table continuation" below).
    --merge-min-confidence {high,medium}
                       Weakest continuation link that is merged. Default: high.
    --postprocess-only Do not convert anything: read the finished output in
                       --output (no PDF argument needed), merge continuation
                       tables, and refresh the index and report. The index is
                       the source of truth, so hand-edited ``continues_from``
                       values are honoured.
    --redetect         With --postprocess-only: recompute ``continues_from`` from
                       the stored geometry, overwriting hand edits.
    --no-resume        Ignore any existing checkpoint and start from page 1.
                       Existing output files are overwritten, never deleted.

Output layout (under --output)
-------------------------------
    sections/<sec_id>.md         section text, blocks separated by blank lines
    tables/<tbl_id>.{md,json}    table rendering + raw Docling cell data
    images/<fig_id>.png          figure raster
    failed_pages/page_NNNNN.pdf  standalone copy of every page that raised an
                                 error or warning (for reproduction / bug reports)
    checkpoints/state.json       full resumable state, rewritten after every batch
    manual_index_raw.json        master index (written once, at completion)
    tables_merged/<id>_merged.*  merged multi-page tables (only with merging)
    report.txt                   human-readable run report, same folder as the
                                 index (see "Report" below)

Contract
--------
Guarantees:
  G1. IDs are unique and stable: ``<kind>_<counter:05d>_<slug>``. The counter is
      global and monotonic, so two items with identical titles/captions can never
      overwrite each other's files.
  G2. Text is never silently dropped: items whose processing raises are logged,
      recorded in ``index["issues"]`` and skipped individually; the rest of the
      page and document continue to be processed.
  G3. Every error or warning is attributed to a PDF page (1-based, absolute).
      That page is logged with a text excerpt and saved as a standalone PDF
      under ``failed_pages/``.
  G4. Crash safety: ``checkpoints/state.json`` is replaced atomically after each
      completed batch and contains the master index, the open section and all
      counters. A crash, kill or Ctrl-C loses at most the batch in progress;
      re-running the same command resumes from the next unprocessed page.
  G5. Deterministic resume: files written by an interrupted batch are
      overwritten with identical names when that batch is redone.
  G7. Non-destructive merging: merging only adds files under ``tables_merged/``
      and index fields; part tables, section files and ids stay untouched, and
      re-running the merge recomputes everything from the parts.
  G6. Output confinement: every file this tool creates lives under the
      directory given by --output; nothing is written elsewhere. The input
      PDF is only read. Existing files in that directory are never deleted.

Table continuation:
  A table that runs over a page break is emitted by the converter as one table per
  page. At the end of extraction every table is compared with the one before it
  and, when it looks like a continuation, receives in the index
      ``continues_from``          id of the previous part (null otherwise)
      ``continuation_confidence`` "high" | "medium"
      ``continuation_reasons``    list of the rules that fired
  Rules (all required): the new table is on the next page; the previous one ends
  in the bottom 20% of its page and the new one starts in the top 20%; same
  column count; and it carries no new, different table caption (no caption, a
  "(continued)" caption, or the same table number repeated). Confidence is
  "high" when additionally the header row is repeated or the caption marks a
  continuation, otherwise "medium". Detection reads the geometry stored per
  table (``page_height``, ``top_frac``, ``bottom_frac``, ``num_cols``,
  ``header_signature``), so it works identically across batch boundaries and
  without the PDF. Thresholds: TOP_MAX_FRAC / BOTTOM_MIN_FRAC.
  With --merge-tables (or --postprocess-only) chains of accepted links are fused:
  rows of later parts are appended below the first part, a repeated header is
  dropped, every cell records ``source_table`` / ``source_page``, and the result
  is written to ``tables_merged/``; ``index["merged_tables"]`` lists the merges
  and each part gets ``merged_into``. A link without ``continuation_confidence``
  (hand-written) is always accepted. Original table files are never modified.

Warning attribution:
  Docling reports table/layout oddities as plain log warnings with no page
  number. While converting a batch, the extractor collects such warnings; if any
  fired, the batch is re-converted page by page (a log line announces this) and
  each warning is attributed to the exact PDF page that triggers it. That page is
  logged with a text excerpt, saved under ``failed_pages/`` and recorded in
  ``index["issues"]`` (severity "warning", stage "convert") and in the report.
  Trade-off: affected batches are converted twice, roughly doubling their time,
  and the same Docling warning is therefore logged twice (first with the batch
  range, then with the exact page). ``--no-locate-warnings`` disables this.

Report (``report.txt``):
  Plain UTF-8 text, always in the output root next to ``manual_index_raw.json``.
  Sections: run status, input/settings, totals, problem-page overview, and a
  per-issue list sorted by PDF page (severity, stage, message, page text
  excerpt, path of the saved page copy). Status is one of COMPLETE,
  COMPLETE WITH ERRORS, or INCOMPLETE (run aborted or interrupted; the report
  then reflects the in-memory state at abort time and the last checkpointed
  page is stated). It is rewritten atomically on every run end, including an
  abort, and is best-effort: a failure to write it is logged and never masks
  the original error or changes the exit code. Elapsed time covers the
  current invocation only (a resumed run reports its own duration).

Failure semantics:
  * A batch that fails or reports problems is re-converted page by page to
    isolate the culprit(s). Pages that still fail are recorded with
    severity "error", replaced by a placeholder in the section text, and
    processing continues.
  * Unexpected exceptions (disk full, OOM, Ctrl-C) abort the run after logging
    the last safely checkpointed page.

Exit codes:
  0    completed with no errors (warnings may exist)
  1    fatal error (including: input PDF missing, or --output exists and is
       not a directory); resume with the same command where applicable
  2    invalid command-line arguments
  3    completed, but one or more pages/items had errors (see ``issues``)
  130  interrupted by the user; resume with the same command

Known limitations:
  * Continuation detection is heuristic. "medium" links are reported but not
    merged by default; merged cells with spans crossing a dropped repeated
    header are discarded; section files still reference the individual parts.
  * Stale files in ``tables_merged/`` from a previous, different merge are not
    deleted.
  * With ``--batchsize 0`` there is a single checkpoint, written at the end.
  * Section detection relies on Docling's section-header classification; the
    numbering regex is only used to extract a ``toc_number`` when present.

Dependencies: docling, docling-core, pymupdf.
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
from datetime import datetime
from dataclasses import asdict, dataclass, field
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

MIN_BATCH_SIZE = 10
DEFAULT_BATCH_SIZE = 100
DEFAULT_MIN_IMAGE_PX = 48
CHECKPOINT_VERSION = 2
EXCERPT_CHARS = 160
INTRO_SECTION_ID = "sec_00000_front_matter"

# Table-continuation heuristics (fractions of page height, measured from the top).
TOP_MAX_FRAC = 0.20
BOTTOM_MIN_FRAC = 0.80
CONFIDENCE_RANK = {"medium": 1, "high": 2}
HIGH_CONFIDENCE_REASONS = frozenset({"header_repeated", "caption_marked_continued"})
CONTINUED_MARKER_RE = re.compile(r"\(?\s*(?:continued|cont\.?)\s*\)?\s*$", re.IGNORECASE)

# Labels that never belong to the running section text.
SKIPPED_TEXT_LABELS = frozenset(
    {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.CAPTION}
)

HEADING_NUMBER_RE = re.compile(
    r"^(?:(?:Chapter|Section|Appendix)\s+([A-Z0-9]+(?:\.[0-9]+)*)"
    r"|([0-9]+(?:\.[0-9]+)*))\b",
    re.IGNORECASE,
)
TABLE_NUMBER_RE = re.compile(
    r"(?:Table|Tbl\.?)\s+([A-Z0-9]+(?:[.\-][0-9]+)*)", re.IGNORECASE
)
FIGURE_NUMBER_RE = re.compile(
    r"(?:Figure|Fig\.?)\s+([A-Z0-9]+(?:[.\-][0-9]+)*)", re.IGNORECASE
)
FOOTNOTE_NUMBER_RE = re.compile(r"^\[?(\d+)\]?[.)\s]")

log = logging.getLogger("docling_extractor")


# --------------------------------------------------------------------------- #
# Logging with PDF page context
# --------------------------------------------------------------------------- #


class _PageContextFilter(logging.Filter):
    """Inject the PDF page range currently being processed into every log record.

    Also covers records emitted by Docling or by ``warnings.warn`` (captured via
    ``logging.captureWarnings``), so third-party warnings are attributable to the
    pages that were being converted when they fired.
    """

    current = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.pages = _PageContextFilter.current
        return True


@contextlib.contextmanager
def page_context(label: str) -> Iterator[None]:
    """Temporarily set the page label shown in log lines (e.g. ``"p101-200"``)."""
    previous = _PageContextFilter.current
    _PageContextFilter.current = label
    try:
        yield
    finally:
        _PageContextFilter.current = previous


class _WarningCollector(logging.Handler):
    """Collect WARNING+ records emitted by third-party loggers (e.g. Docling).

    Docling reports table/layout oddities as plain log warnings that carry no page
    number. Collecting them during a conversion lets the extractor notice that a
    page range triggered one and then isolate the exact page (see
    ``ManualExtractor._process_batch``). Records from this module's own logger are
    ignored so our diagnostics are never mistaken for conversion warnings.
    """

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
    """Yield a list that is filled with third-party warnings logged inside the block.

    Args:
        enabled: When False nothing is attached and the yielded list stays empty.
    """
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
    """Formatter that renders an empty message as a truly empty line.

    Lets ``log.info("")`` act as a visual separator without a dangling
    ``[time] LEVEL [pages]`` prefix, while still going through the same handler
    (so ordering relative to other records is preserved).
    """

    def format(self, record: logging.LogRecord) -> str:
        if not record.getMessage():
            return ""
        return super().format(record)


def configure_logging() -> None:
    """Configure root logging: ``HH:MM:SS LEVEL [pages] message``, warnings captured."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_PageContextFilter())
    handler.setFormatter(
        _BlankAwareFormatter(
            "[%(asctime)s] %(levelname)-8s [%(pages)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.captureWarnings(True)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def slugify(text: str, fallback: str) -> str:
    """Return a filesystem-safe lowercase slug of at most 60 chars.

    Args:
        text: Arbitrary text to slugify.
        fallback: Returned when the slug would be empty.
    """
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip().lower()).strip("_")
    return clean[:60] if clean else fallback


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so that ``path`` is always either the old or the new full file.

    Writes to a temp file in the same directory, fsyncs, then ``os.replace``s it.
    A crash mid-write therefore can never leave a truncated checkpoint or report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
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
    """Atomically write ``payload`` as indented UTF-8 JSON (see ``atomic_write_text``)."""
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _continuation_report_lines(index: dict[str, Any], sub: str) -> list[str]:
    """Report block describing detected table continuations and merges."""
    tables = index["tables"]
    links = [t for t in tables if t.get("continues_from")]
    high = sum(1 for t in links if t.get("continuation_confidence") == "high")
    medium = sum(1 for t in links if t.get("continuation_confidence") == "medium")
    merged = index.get("merged_tables", [])
    lines = [
        "TABLE CONTINUATIONS", sub,
        f"Links found    : {len(links)} (high: {high}, medium: {medium})",
        f"Merged tables  : {len(merged)}",
    ]
    for entry in merged[:50]:
        lines.append(
            f"  {entry['id']}: pages {entry['page_start']}-{entry['page_end']}, "
            f"{len(entry['parts'])} parts, {entry['num_rows']}x{entry['num_cols']}"
        )
    unmerged = [t for t in links if not t.get("merged_into")]
    if unmerged:
        lines.append(f"Links NOT merged ({len(unmerged)}; review or lower --merge-min-confidence):")
        for t in unmerged[:50]:
            lines.append(
                f"  page {t['page'] - 1} -> {t['page']} [{t.get('continuation_confidence', 'manual')}] "
                f"{t['continues_from']} -> {t['id']}"
            )
    return lines


def build_report(
    state: "ExtractionState",
    layout: "OutputLayout",
    pdf_path: Path,
    batch_size: int,
    min_image_px: int,
    started_from_page: int,
    elapsed_s: float,
    aborted: bool,
    note: Optional[str] = None,
) -> str:
    """Render the human-readable run report.

    Pure function (no I/O): safe to unit-test and to call from an abort handler.

    Args:
        state: Extraction state to report on.
        layout: Output layout (used for artifact paths).
        pdf_path: Input PDF.
        batch_size: Configured batch size (0 = whole document).
        min_image_px: Configured minimum figure side in pixels.
        started_from_page: First page processed by this invocation (>1 on resume).
        elapsed_s: Wall-clock seconds of this invocation.
        aborted: True when the run did not reach the end of the document.
        note: When given, replaces the "This run" line (used by --postprocess-only,
            where no pages are converted).

    Returns:
        The full report text, ending with a newline.
    """
    total_pages = state.fingerprint["total_pages"]
    issues = sorted(state.index["issues"], key=lambda i: (i["page"], i["severity"]))
    errors = [i for i in issues if i["severity"] == "error"]
    warnings_ = [i for i in issues if i["severity"] == "warning"]
    last_done = state.next_page - 1

    if aborted or not state.finished:
        status = f"INCOMPLETE (stopped after page {last_done} of {total_pages}; re-run to resume)"
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
        f"Pages done     : {min(last_done, total_pages)} of {total_pages}",
        note if note is not None else
        f"This run       : pages {started_from_page}-{min(last_done, total_pages)}"
        f" in {_format_duration(elapsed_s)}",
        f"Batch size     : {'whole document' if batch_size == 0 else batch_size}",
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
        f"Error pages    : {', '.join(str(p) for p in sorted({i['page'] for i in errors})) or '-'}",
        f"Warning pages  : {', '.join(str(p) for p in sorted({i['page'] for i in warnings_})) or '-'}",
    ]

    if issues:
        lines += ["", "ISSUE DETAILS (sorted by PDF page)", sub]
        for n, issue in enumerate(issues, start=1):
            lines.append(
                f"{n:>3}. PDF page {issue['page']} | {issue['severity'].upper()} | {issue['stage']}"
            )
            lines.append(textwrap.fill(
                f"Message  : {issue['message']}", width=100,
                initial_indent="     ", subsequent_indent="              "))
            lines.append(textwrap.fill(
                f"Page text: {issue.get('page_excerpt') or '-'}", width=100,
                initial_indent="     ", subsequent_indent="              "))
            lines.append(f"     Page copy: {issue.get('page_pdf') or '<not saved>'}")

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
        *([f"Merged tables  : {layout.rel(layout.tables_merged)}/"] if state.index.get("merged_tables") else []),
        rule,
        "",
    ]
    return "\n".join(lines)


def release_memory() -> None:
    """Run the GC and release the Apple MPS cache if torch/MPS is in use."""
    gc.collect()
    try:
        import torch  # noqa: WPS433 (lazy: only needed for the MPS cache)
    except ImportError:
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in ``pdf_path``."""
    with pymupdf.open(str(pdf_path)) as doc:
        return len(doc)


def describe_page(pdf_path: Path, page_no: int) -> str:
    """Return a one-line text excerpt of a PDF page for diagnostics.

    Never raises: diagnostics must not mask the original problem.

    Args:
        pdf_path: Source PDF.
        page_no: 1-based page number.
    """
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            text = doc[page_no - 1].get_text("text")
    except Exception as exc:  # noqa: BLE001
        return f"<page text unavailable: {exc}>"
    flat = " ".join(text.split())
    return flat[:EXCERPT_CHARS] if flat else "<no extractable text>"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass
class OutputLayout:
    """Filesystem layout of all produced artifacts."""

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
        """Run report; always in the same folder as ``index_file``."""
        return self.index_file.parent / "report.txt"

    def create(self) -> None:
        """Create all output directories (idempotent)."""
        for directory in (
            self.sections,
            self.tables,
            self.images,
            self.checkpoints,
            self.failed_pages,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        """Return ``path`` relative to the output root, as a POSIX string."""
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
    sec_id: str, toc_number: Optional[str], title: str, level: int, page_start: int
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
    """Complete resumable state; JSON-serialisable via ``dataclasses.asdict``.

    Invariant: ``next_page`` is the first page NOT yet reflected in ``index``,
    ``current_section`` and ``counters``.
    """

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
        """Create the initial state for a document starting at page 1."""
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
                INTRO_SECTION_ID, "0", "Front Matter & Introduction", 1, 1
            ),
            index=_new_index(fingerprint["pdf_name"]),
        )


# --------------------------------------------------------------------------- #
# Table continuation: detection and merging
# --------------------------------------------------------------------------- #


def table_geometry(doc: Any, item: Any, page_no: int) -> dict[str, Any]:
    """Return the page-relative geometry and header data used to detect continuations.

    Never raises: a table whose geometry cannot be read simply gets ``None``
    fractions and is never considered a continuation.

    Args:
        doc: Docling document owning ``item`` (provides page sizes).
        item: A Docling ``TableItem``.
        page_no: 1-based page of the table (only used for diagnostics).

    Returns:
        Dict with ``page_height``, ``top_frac`` and ``bottom_frac`` (fractions of the
        page height measured from the TOP edge, 0 = top), ``num_rows``,
        ``num_cols``, ``header_rows`` and ``header_signature`` (texts of the cells
        flagged as column headers, in reading order).
    """
    info: dict[str, Any] = {
        "page_height": None, "top_frac": None, "bottom_frac": None,
        "num_rows": 0, "num_cols": 0, "header_rows": 0, "header_signature": [],
    }
    try:
        data = item.data
        info["num_rows"], info["num_cols"] = int(data.num_rows), int(data.num_cols)
        header = [cell for cell in data.table_cells if cell.column_header]
        if header:
            info["header_rows"] = max(cell.end_row_offset_idx for cell in header)
            ordered = sorted(
                header, key=lambda c: (c.start_row_offset_idx, c.start_col_offset_idx)
            )
            info["header_signature"] = [cell.text.strip() for cell in ordered]
    except Exception:  # noqa: BLE001
        log.debug("No table data on page %d", page_no, exc_info=True)
    try:
        prov = item.prov[0]
        height = float(doc.pages[prov.page_no].size.height)
        box = prov.bbox.to_top_left_origin(height)
        info["page_height"] = height
        info["top_frac"] = round(min(max(box.t / height, 0.0), 1.0), 4)
        info["bottom_frac"] = round(min(max(box.b / height, 0.0), 1.0), 4)
    except Exception:  # noqa: BLE001
        log.debug("No table geometry on page %d", page_no, exc_info=True)
    return info


def _norm_header(signature: list[str]) -> list[str]:
    """Normalise header texts (case/whitespace) for comparison."""
    return [" ".join(text.lower().split()) for text in signature]


def _continuation_reasons(
    prev: dict[str, Any], curr: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> Optional[list[str]]:
    """Decide whether ``curr`` continues ``prev``; return the fired rules or None."""
    if curr["page"] != prev["page"] + 1:
        return None
    prev_bottom, curr_top = prev.get("bottom_frac"), curr.get("top_frac")
    if prev_bottom is None or curr_top is None:
        return None
    if prev_bottom < BOTTOM_MIN_FRAC or curr_top > TOP_MAX_FRAC:
        return None
    cols = curr.get("num_cols") or 0
    if cols == 0 or cols != prev.get("num_cols"):
        return None

    reasons = ["next_page", "bottom_to_top", "same_cols"]
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
            return None  # a different, new caption: this is a new table

    root, seen = prev, {prev["id"]}
    while root.get("continues_from") in by_id and root["continues_from"] not in seen:
        root = by_id[root["continues_from"]]
        seen.add(root["id"])
    curr_sig = _norm_header(curr.get("header_signature") or [])
    if curr_sig and curr_sig == _norm_header(root.get("header_signature") or []):
        reasons.append("header_repeated")
    return reasons


def detect_continuations(tables: list[dict[str, Any]]) -> dict[str, int]:
    """Annotate index table entries with ``continues_from`` links (in place).

    Tables are compared in document order, each with the table immediately before
    it. Existing ``continues_from`` / ``continuation_*`` values are overwritten.

    Args:
        tables: ``index["tables"]`` in document order.

    Returns:
        ``{"high": n, "medium": n}`` number of links found per confidence.
    """
    by_id = {t["id"]: t for t in tables}
    counts = {"high": 0, "medium": 0}
    for position, curr in enumerate(tables):
        curr["continues_from"] = None
        curr.pop("continuation_confidence", None)
        curr.pop("continuation_reasons", None)
        if position == 0:
            continue
        reasons = _continuation_reasons(tables[position - 1], curr, by_id)
        if reasons is None:
            continue
        confidence = "high" if HIGH_CONFIDENCE_REASONS & set(reasons) else "medium"
        curr["continues_from"] = tables[position - 1]["id"]
        curr["continuation_confidence"] = confidence
        curr["continuation_reasons"] = reasons
        counts[confidence] += 1
    return counts


def _add_issue(index: dict[str, Any], page: int, severity: str, stage: str, message: str) -> None:
    """Append an issue that has no PDF page copy (e.g. raised during post-processing)."""
    index["issues"].append(
        {"page": page, "severity": severity, "stage": stage, "message": message,
         "page_excerpt": "", "page_pdf": None}
    )


def _accepted_links(
    tables: list[dict[str, Any]], min_confidence: str
) -> tuple[dict[str, str], list[tuple[int, str]]]:
    """Validate ``continues_from`` links and keep those that may be merged.

    A link is dropped (with a warning when it is malformed) if its target is
    unknown, not earlier in document order, already continued by another table,
    or weaker than ``min_confidence``. Links without ``continuation_confidence``
    (written by hand) are always accepted. Requiring earlier targets makes cycles
    impossible.

    Returns:
        ``(child_id -> parent_id, [(page, warning message), ...])``.
    """
    by_id = {t["id"]: t for t in tables}
    order = {t["id"]: n for n, t in enumerate(tables)}
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
            warnings.append((table["page"], f"{tid}: continues_from {parent!r} is not a known table; link ignored"))
            continue
        confidence = table.get("continuation_confidence")
        if confidence is not None and CONFIDENCE_RANK.get(confidence, 0) < min_rank:
            continue
        if order[parent] >= order[tid]:
            warnings.append((table["page"], f"{tid}: continues_from {parent!r} must be an earlier table; link ignored"))
            continue
        if parent in claimed:
            warnings.append((table["page"], f"{tid}: {parent!r} is already continued by {claimed[parent]!r}; link ignored"))
            continue
        parent_of[tid] = parent
        claimed[parent] = tid
    return parent_of, warnings


def _build_grid(rows: int, cols: int, cells: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Rebuild a Docling-style ``grid`` (every covered position -> its cell)."""
    grid: list[list[Optional[dict[str, Any]]]] = [[None] * cols for _ in range(rows)]
    for cell in cells:
        for r in range(cell["start_row_offset_idx"], min(cell["end_row_offset_idx"], rows)):
            for c in range(cell["start_col_offset_idx"], min(cell["end_col_offset_idx"], cols)):
                grid[r][c] = cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                grid[r][c] = {
                    "bbox": None, "row_span": 1, "col_span": 1,
                    "start_row_offset_idx": r, "end_row_offset_idx": r + 1,
                    "start_col_offset_idx": c, "end_col_offset_idx": c + 1,
                    "text": "", "column_header": False, "row_header": False,
                    "row_section": False, "fillable": False,
                }
    return grid  # type: ignore[return-value]


def merge_table_data(parts: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Fuse the raw table data of consecutive parts into one table.

    Rows of each later part are shifted below the previous ones. When a later
    part repeats the first part's header, its header rows are dropped (cells
    starting inside them are discarded, including spans that cross the boundary).
    Cells of later parts lose their ``column_header`` flag, and every cell gains
    ``source_table`` / ``source_page`` because its original ``bbox`` belongs to a
    different page.

    Args:
        parts: ``[(index_entry, table_json_dict), ...]`` in page order.

    Returns:
        A table dict in the same shape as the per-table JSON files, plus
        ``merge_info`` describing the parts.
    """
    first_meta, first_data = parts[0]
    first_sig = _norm_header(first_meta.get("header_signature") or [])
    cells: list[dict[str, Any]] = []
    rows, cols = 0, max(int(data.get("num_cols", 0)) for _, data in parts)
    info: list[dict[str, Any]] = []
    for position, (meta, data) in enumerate(parts):
        skip = 0
        if position and first_sig and _norm_header(meta.get("header_signature") or []) == first_sig:
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
        info.append({"id": meta["id"], "page": meta["page"], "header_rows_dropped": skip})

    merged = {k: v for k, v in first_data.items()
              if k not in ("table_cells", "grid", "num_rows", "num_cols")}
    merged.update(
        num_rows=rows, num_cols=cols, table_cells=cells,
        grid=_build_grid(rows, cols, cells),
        merge_info={"parts": info, "page_start": parts[0][0]["page"], "page_end": parts[-1][0]["page"]},
    )
    return merged


def _md_cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def merged_table_markdown(merged: dict[str, Any], caption: str) -> str:
    """Render a merged table as GitHub-flavoured Markdown (spans are not representable).

    The first row is used as the Markdown header line. Cells spanning several
    positions appear only at their top-left position.
    """
    rows, cols = merged["num_rows"], merged["num_cols"]
    matrix = [[""] * cols for _ in range(rows)]
    for cell in merged["table_cells"]:
        r, c = cell["start_row_offset_idx"], cell["start_col_offset_idx"]
        if r < rows and c < cols:
            matrix[r][c] = _md_cell(cell.get("text", ""))
    lines = []
    if caption:
        info = merged["merge_info"]
        lines.append(f"**{caption} (pages {info['page_start']}-{info['page_end']}, "
                     f"merged from {len(info['parts'])} parts)**\n")
    if rows:
        lines.append("| " + " | ".join(matrix[0]) + " |")
        lines.append("|" + "|".join(["---"] * cols) + "|")
        lines.extend("| " + " | ".join(row) + " |" for row in matrix[1:])
    return "\n".join(lines) + "\n"


def merge_tables(layout: "OutputLayout", index: dict[str, Any], min_confidence: str = "high") -> int:
    """Merge chains of continuation tables; mutate ``index``, write ``tables_merged/``.

    Recomputes everything from the part tables each time (idempotent): previous
    ``merged_tables`` / ``merged_into`` are cleared first. Part files are only read.
    A chain whose part files cannot be read is skipped and reported as a warning
    issue attributed to the part's page.

    Args:
        layout: Output layout (``tables_merged`` is created on demand).
        index: Master index dict (``tables`` must be in document order).
        min_confidence: ``"high"`` or ``"medium"``, weakest link to merge.

    Returns:
        Number of merged tables written.
    """
    tables = index["tables"]
    for table in tables:
        table.pop("merged_into", None)
    index["merged_tables"] = []

    parent_of, warnings = _accepted_links(tables, min_confidence)
    for page, message in warnings:
        log.warning("Table merge: PDF page %d | %s", page, message)
        _add_issue(index, page, "warning", "merge", message)

    by_id = {t["id"]: t for t in tables}
    child_of = {parent: child for child, parent in parent_of.items()}
    merged_count = 0
    for root in (t for t in tables if t["id"] not in parent_of and t["id"] in child_of):
        chain = [root]
        while chain[-1]["id"] in child_of:
            chain.append(by_id[child_of[chain[-1]["id"]]])
        try:
            parts = [
                (meta, json.loads((layout.root / meta["json_path"]).read_text(encoding="utf-8")))
                for meta in chain
            ]
        except (OSError, ValueError) as exc:
            message = f"cannot merge chain starting at {root['id']}: {exc}"
            log.warning("Table merge: PDF page %d | %s", root["page"], message)
            _add_issue(index, root["page"], "warning", "merge", message)
            continue

        merged = merge_table_data(parts)
        merged_id = f"{root['id']}_merged"
        layout.tables_merged.mkdir(parents=True, exist_ok=True)
        json_file = layout.tables_merged / f"{merged_id}.json"
        md_file = layout.tables_merged / f"{merged_id}.md"
        atomic_write_json(json_file, merged)
        md_file.write_text(merged_table_markdown(merged, root.get("caption", "")), encoding="utf-8")

        index["merged_tables"].append({
            "id": merged_id,
            "root_table": root["id"],
            "parts": [m["id"] for m in chain],
            "page_start": chain[0]["page"],
            "page_end": chain[-1]["page"],
            "num_rows": merged["num_rows"],
            "num_cols": merged["num_cols"],
            "caption": root.get("caption", ""),
            "md_path": layout.rel(md_file),
            "json_path": layout.rel(json_file),
        })
        for meta in chain:
            meta["merged_into"] = merged_id
        merged_count += 1
        log.info("Merged %d parts (pages %d-%d) -> %s",
                 len(chain), chain[0]["page"], chain[-1]["page"], merged_id)
    return merged_count


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


class ManualExtractor:
    """Drive Docling over a PDF in batches and persist results incrementally.

    Args:
        pdf_path: Input PDF.
        layout: Output layout (created on ``run``).
        batch_size: Pages per batch; ``0`` means the whole document at once.
        min_image_px: Minimum width/height (px) for a figure to be kept.
        resume: Resume from an existing checkpoint when one is present.
        locate_warnings: Attribute Docling warnings to exact PDF pages by
            re-converting the affected batch page by page (slower).
        merge_tables: Merge continuation tables at the end of the run.
        merge_min_confidence: Weakest continuation link to merge ("high"/"medium").
    """

    def __init__(
        self,
        pdf_path: Path,
        layout: OutputLayout,
        batch_size: int,
        min_image_px: int,
        resume: bool,
        locate_warnings: bool = True,
        merge_tables: bool = False,
        merge_min_confidence: str = "high",
    ) -> None:
        self.pdf_path = pdf_path
        self.layout = layout
        self.batch_size = batch_size
        self.min_image_px = min_image_px
        self.resume = resume
        self.locate_warnings = locate_warnings
        self.merge_tables = merge_tables
        self.merge_min_confidence = merge_min_confidence
        self.converter = self._create_converter()
        self.state: ExtractionState  # set in run()
        self._started_at = time.monotonic()
        self._started_from_page = 1

    # -- setup ------------------------------------------------------------- #

    @staticmethod
    def _create_converter() -> DocumentConverter:
        """Build the Docling converter (no OCR, accurate tables, figure images)."""
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.generate_picture_images = True
        opts.images_scale = 1.5
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def _load_or_create_state(self, total_pages: int) -> ExtractionState:
        """Load a compatible checkpoint or create a fresh state.

        Raises:
            RuntimeError: Checkpoint is unreadable, from another version, or
                belongs to a different PDF. The message says how to proceed.
        """
        fingerprint = {
            "pdf_name": self.pdf_path.name,
            "pdf_size": self.pdf_path.stat().st_size,
            "total_pages": total_pages,
        }
        ckpt = self.layout.checkpoint_file
        if not self.resume or not ckpt.exists():
            if ckpt.exists():
                log.warning("--no-resume: ignoring existing checkpoint %s", ckpt)
            return ExtractionState.fresh(fingerprint)

        try:
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            state = ExtractionState(**data)
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Checkpoint {ckpt} is unreadable ({exc}). "
                "Fix or remove it, or re-run with --no-resume."
            ) from exc
        if state.version != CHECKPOINT_VERSION or state.fingerprint != fingerprint:
            raise RuntimeError(
                f"Checkpoint {ckpt} does not match this PDF/script version "
                f"(checkpoint={state.fingerprint}, current={fingerprint}). "
                "Use a different --output or re-run with --no-resume."
            )
        log.info(
            "Resuming from checkpoint: next page = %d of %d%s",
            state.next_page,
            total_pages,
            " (already complete)" if state.finished else "",
        )
        return state

    # -- orchestration ----------------------------------------------------- #

    def run(self) -> int:
        """Process the whole document.

        Returns:
            Process exit code: 0 clean, 3 completed with errors.

        Raises:
            RuntimeError: Incompatible or corrupt checkpoint.
            Exception: Any unexpected failure (after logging the resume point).
            KeyboardInterrupt: Propagated after logging the resume point.
        """
        self.layout.create()
        total_pages = count_pdf_pages(self.pdf_path)
        self.state = self._load_or_create_state(total_pages)
        self._started_at = time.monotonic()
        self._started_from_page = self.state.next_page
        self.state.settings = {"batch_size": self.batch_size, "min_image_px": self.min_image_px}
        batch = total_pages if self.batch_size == 0 else self.batch_size

        log.info("=== DOCLING ASSET EXTRACTOR ===")
        log.info("PDF        : %s", self.pdf_path)
        log.info("Pages      : %d", total_pages)
        log.info("Batch size : %s", "ALL" if self.batch_size == 0 else self.batch_size)

        try:
            for first in range(self.state.next_page, total_pages + 1, batch):
                last = min(first + batch - 1, total_pages)
                with page_context(f"p{first}-{last}"):
                    self._process_batch(first, last)
                    self.state.next_page = last + 1
                    self._save_checkpoint()
                    log.info(
                        "[BATCH DONE] sections=%d tables=%d figures=%d footnotes=%d issues=%d",
                        len(self.state.index["sections"]),
                        len(self.state.index["tables"]),
                        len(self.state.index["pictures"]),
                        len(self.state.index["footnotes"]),
                        len(self.state.index["issues"]),
                    )
                release_memory()
            self._finalize()
        except BaseException:
            log.critical(
                "Run aborted. Work is safe up to page %d (checkpoint: %s). "
                "Re-run the same command to resume.",
                self.state.next_page - 1,
                self.layout.checkpoint_file,
            )
            self._write_report(aborted=True)
            raise

        has_errors = any(i["severity"] == "error" for i in self.state.index["issues"])
        return 3 if has_errors else 0

    def _save_checkpoint(self) -> None:
        atomic_write_json(self.layout.checkpoint_file, asdict(self.state))

    def _finalize(self) -> None:
        """Flush the open section, write the final index and print the summary."""
        self._flush_section()
        self.state.current_section = None
        self.state.finished = True

        counts = detect_continuations(self.state.index["tables"])
        log.info("Table continuations detected: %d high, %d medium",
                 counts["high"], counts["medium"])
        merged = 0
        if self.merge_tables:
            merged = merge_tables(self.layout, self.state.index, self.merge_min_confidence)

        self._save_checkpoint()
        atomic_write_json(self.layout.index_file, self.state.index)
        self._write_report(aborted=False)

        counters = self.state.counters
        issues = self.state.index["issues"]
        errors = [i for i in issues if i["severity"] == "error"]
        log.info("")
        log.info("=" * 40)
        log.info("FINAL SUMMARY")
        log.info("Sections : %d", counters["section"])
        log.info("Tables   : %d", counters["table"])
        log.info("Figures  : %d (skipped as too small: %d)",
                 counters["picture"], counters["picture_skipped"])
        log.info("Footnotes: %d", counters["footnote"])
        log.info("Tables merged: %d", merged)
        log.info("Issues   : %d (%d errors)", len(issues), len(errors))
        for issue in errors:
            log.info("  - PDF page %d [%s]: %s", issue["page"], issue["stage"], issue["message"])
        log.info("Index    : %s", self.layout.index_file)
        log.info("Report   : %s", self.layout.report_file)
        log.info("=" * 40)

    def _write_report(self, aborted: bool) -> None:
        """Write ``report.txt`` atomically; best-effort, never raises.

        Args:
            aborted: True when called from the abort path.
        """
        try:
            atomic_write_text(
                self.layout.report_file,
                build_report(
                    state=self.state,
                    layout=self.layout,
                    pdf_path=self.pdf_path,
                    batch_size=self.batch_size,
                    min_image_px=self.min_image_px,
                    started_from_page=self._started_from_page,
                    elapsed_s=time.monotonic() - self._started_at,
                    aborted=aborted,
                    note=None,
                ),
            )
        except Exception:  # noqa: BLE001
            log.exception("Could not write report %s", self.layout.report_file)

    # -- conversion -------------------------------------------------------- #

    def _convert(self, first: int, last: int) -> tuple[Any, Optional[str]]:
        """Convert pages ``first..last`` (inclusive, 1-based).

        Third-party warnings logged during the conversion (when
        ``locate_warnings`` is on) are folded into ``problem`` so the caller can
        attribute them to a precise page.

        Returns:
            ``(result, problem)``. ``result`` is None if Docling raised.
            ``problem`` is None only when the conversion was fully clean.
        """
        with collect_warnings(self.locate_warnings) as warned:
            try:
                result = self.converter.convert(
                    str(self.pdf_path), page_range=(first, last)
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("Docling raised for pages %d-%d", first, last, exc_info=True)
                return None, f"{type(exc).__name__}: {exc}"

        problems: list[str] = []
        if result.status != ConversionStatus.SUCCESS:
            problems.append(f"status={result.status.name}")
        for err in getattr(result, "errors", None) or []:
            problems.append(str(getattr(err, "error_message", None) or repr(err)))
        problems.extend(f"docling warning: {message}" for message in warned)
        return result, ("; ".join(problems) or None)

    def _process_batch(self, first: int, last: int) -> None:
        """Convert and consume one batch, isolating bad pages on any problem."""
        log.info("")
        log.info("[BATCH START] pages=%d-%d", first, last)
        result, problem = self._convert(first, last)

        if first == last:
            self._handle_page_result(first, result, problem)
            return

        if problem is None:
            self._consume(result, first)
            del result
            return

        del result
        log.warning(
            "Batch %d-%d reported a problem (%.300s). Re-converting page by page "
            "to locate the exact PDF page(s).", first, last, problem,
        )
        for page in range(first, last + 1):
            with page_context(f"p{page}"):
                self._handle_page_result(page, *self._convert(page, page))
            release_memory()

    def _handle_page_result(
        self, page: int, result: Any, problem: Optional[str]
    ) -> None:
        """Consume one page's conversion result; never raises.

        A raised conversion (``result is None``) is an error and leaves a
        placeholder in the section text. Any other problem is a warning attributed
        to ``page``; the page content is kept.
        """
        if result is None:
            self._report_issue(page, "error", "convert", problem or "conversion failed")
            self._append_block(f"[PAGE {page} COULD NOT BE CONVERTED]")
            return
        if problem:
            self._report_issue(page, "warning", "convert", problem)
        self._consume(result, page)
        del result

    # -- item handling ----------------------------------------------------- #

    def _consume(self, result: Any, default_page: int) -> None:
        """Walk a Docling document and dispatch each item, isolating item failures.

        Items without provenance inherit the last page seen in this document, so
        issue reports stay as close as possible to the real page.
        """
        doc = result.document
        last_page = default_page
        for item, _depth in doc.iterate_items():
            prov = getattr(item, "prov", None)
            page_no = prov[0].page_no if prov else last_page
            last_page = page_no
            try:
                self._handle_item(doc, item, page_no)
            except Exception as exc:  # noqa: BLE001
                self._report_issue(
                    page_no, "error", "item",
                    f"{type(item).__name__} ({getattr(item, 'label', '?')}): "
                    f"{type(exc).__name__}: {exc}",
                    exc=exc,
                )

    def _handle_item(self, doc: Any, item: Any, page_no: int) -> None:
        if isinstance(item, SectionHeaderItem):
            self._handle_section_header(item, page_no)
        elif isinstance(item, TextItem):
            self._handle_text(item, page_no)
        elif isinstance(item, TableItem):
            self._handle_table(doc, item, page_no)
        elif isinstance(item, PictureItem):
            self._handle_picture(doc, item, page_no)

    def _next_counter(self, kind: str) -> int:
        self.state.counters[kind] += 1
        return self.state.counters[kind]

    def _append_block(self, block: str) -> None:
        assert self.state.current_section is not None
        self.state.current_section["buffer"].append(block)

    def _flush_section(self) -> None:
        """Write the open section to disk and register it in the index.

        No-op for an empty front-matter section or when no section is open.
        """
        sec = self.state.current_section
        if sec is None or (not sec["buffer"] and sec["id"] == INTRO_SECTION_ID):
            return
        md_path = self.layout.sections / f"{sec['id']}.md"
        md_path.write_text(
            f"# {sec['title']}\n\n" + "\n\n".join(sec["buffer"]) + "\n", encoding="utf-8"
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

    def _handle_section_header(self, item: SectionHeaderItem, page_no: int) -> None:
        self._flush_section()
        n = self._next_counter("section")
        heading = item.text.strip()
        match = HEADING_NUMBER_RE.match(heading)
        toc_number = (match.group(1) or match.group(2)) if match else None
        sec_id = f"sec_{n:05d}_{slugify(toc_number or heading, 'untitled')}"
        self.state.current_section = _new_section(
            sec_id, toc_number, heading, int(getattr(item, "level", 1) or 1), page_no
        )

    def _handle_text(self, item: TextItem, page_no: int) -> None:
        if item.label in SKIPPED_TEXT_LABELS:
            return
        text = item.text.strip()
        if not text:
            return

        if item.label == DocItemLabel.FOOTNOTE:
            n = self._next_counter("footnote")
            match = FOOTNOTE_NUMBER_RE.match(text)
            fn_num = match.group(1) if match else str(n)
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
            self._append_block(f"> **[Footnote {fn_num}]**: {text}")
        elif isinstance(item, ListItem):
            self._append_block(f"- {text}")
        else:
            self._append_block(text)

    @staticmethod
    def _bbox(item: Any) -> list[float]:
        prov = getattr(item, "prov", None)
        return list(prov[0].bbox.as_tuple()) if prov else []

    def _handle_table(self, doc: Any, item: TableItem, page_no: int) -> None:
        n = self._next_counter("table")
        caption = item.caption_text(doc).strip()
        match = TABLE_NUMBER_RE.search(caption)
        label = match.group(1).replace("-", ".") if match else f"p{page_no}"
        tbl_id = f"tbl_{n:05d}_{slugify(label, 'x')}"
        md_file = self.layout.tables / f"{tbl_id}.md"
        json_file = self.layout.tables / f"{tbl_id}.json"

        body = item.export_to_markdown(doc)
        md_file.write_text(
            (f"**{caption}**\n\n" if caption else "") + body + "\n", encoding="utf-8"
        )
        atomic_write_json(json_file, item.data.model_dump())

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
                "caption_continued": bool(CONTINUED_MARKER_RE.search(caption)),
                **table_geometry(doc, item, page_no),
                "continues_from": None,
                "md_path": self.layout.rel(md_file),
                "json_path": self.layout.rel(json_file),
            }
        )
        self._append_block(f"[TABLE REF: {tbl_id} -> {md_file.name}]")

    def _handle_picture(self, doc: Any, item: PictureItem, page_no: int) -> None:
        pil_img = item.get_image(doc)
        if pil_img is not None and min(pil_img.size) < self.min_image_px:
            self.state.counters["picture_skipped"] += 1
            return

        n = self._next_counter("picture")
        caption = item.caption_text(doc).strip()
        match = FIGURE_NUMBER_RE.search(caption)
        label = match.group(1).replace("-", ".") if match else f"p{page_no}"
        fig_id = f"fig_{n:05d}_{slugify(label, 'x')}"
        img_file = self.layout.images / f"{fig_id}.png"
        if pil_img is not None:
            pil_img.save(img_file, format="PNG")

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
                "image_path": self.layout.rel(img_file) if pil_img is not None else None,
            }
        )
        self._append_block(f"[FIGURE REF: {fig_id} -> {img_file.name}]")

    # -- diagnostics ------------------------------------------------------- #

    def _export_page(self, page_no: int) -> Optional[str]:
        """Save a standalone copy of ``page_no``; return its relative path or None."""
        dest = self.layout.failed_pages / f"page_{page_no:05d}.pdf"
        try:
            if not dest.exists():
                with pymupdf.open(str(self.pdf_path)) as src, pymupdf.open() as out:
                    out.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
                    out.save(str(dest))
            return self.layout.rel(dest)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not export page %d: %s", page_no, exc)
            return None

    def _report_issue(
        self,
        page_no: int,
        severity: str,
        stage: str,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Log an issue with its PDF page, save that page, and record it.

        Args:
            page_no: 1-based absolute PDF page the issue is attributed to.
            severity: ``"warning"`` (output kept) or ``"error"`` (content lost).
            stage: ``"convert"`` (Docling) or ``"item"`` (post-processing).
            message: Human-readable description.
            exc: Original exception, logged with traceback when given.
        """
        excerpt = describe_page(self.pdf_path, page_no)
        saved = self._export_page(page_no)
        log.log(
            logging.ERROR if severity == "error" else logging.WARNING,
            "PDF PAGE %d | stage=%s | %s | page text: %r | page copy: %s",
            page_no, stage, message, excerpt, saved or "<not saved>",
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
# CLI
# --------------------------------------------------------------------------- #


def run_postprocess(layout: OutputLayout, min_confidence: str, redetect: bool) -> int:
    """Merge continuation tables of a finished extraction, without converting anything.

    The index file is the source of truth (hand-edited ``continues_from`` values are
    honoured unless ``redetect`` is set). Updates the index, the checkpoint and the
    report so the three stay consistent.

    Args:
        layout: Output layout of a completed extraction.
        min_confidence: Weakest continuation link to merge.
        redetect: Recompute ``continues_from`` from stored geometry first.

    Returns:
        0 on success, 1 when the output is missing, incomplete or incompatible.
    """
    ckpt, index_file = layout.checkpoint_file, layout.index_file
    if not ckpt.exists() or not index_file.exists():
        log.error("Nothing to post-process: %s and %s must both exist "
                  "(run the extraction first).", ckpt, index_file)
        return 1
    try:
        state = ExtractionState(**json.loads(ckpt.read_text(encoding="utf-8")))
        state.index = json.loads(index_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        log.error("Cannot read the extraction output in %s: %s", layout.root, exc)
        return 1
    if state.version != CHECKPOINT_VERSION:
        log.error("Output was produced by an incompatible script version "
                  "(checkpoint v%s, expected v%s); re-run the extraction.",
                  state.version, CHECKPOINT_VERSION)
        return 1
    if not state.finished:
        log.error("Extraction is not finished (stopped after page %d). Re-run the same "
                  "extraction command to resume it first.", state.next_page - 1)
        return 1

    log.info("=== TABLE POST-PROCESSING ===")
    log.info("Output: %s", layout.root)
    if redetect:
        counts = detect_continuations(state.index["tables"])
        log.info("Re-detected continuations: %d high, %d medium", counts["high"], counts["medium"])
    merged = merge_tables(layout, state.index, min_confidence)

    atomic_write_json(index_file, state.index)
    atomic_write_json(ckpt, asdict(state))
    settings = state.settings or {}
    atomic_write_text(
        layout.report_file,
        build_report(
            state=state, layout=layout, pdf_path=Path(state.fingerprint["pdf_name"]),
            batch_size=int(settings.get("batch_size", 0)),
            min_image_px=int(settings.get("min_image_px", DEFAULT_MIN_IMAGE_PX)),
            started_from_page=1, elapsed_s=0.0, aborted=False,
            note="This run       : table post-processing only (no pages converted)",
        ),
    )
    log.info("")
    log.info("Tables merged: %d | Report: %s", merged, layout.report_file)
    return 0


def _batch_size_arg(value: str) -> int:
    """argparse type: ``0`` (single pass) or an integer >= ``MIN_BATCH_SIZE``."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if number == 0 or number >= MIN_BATCH_SIZE:
        return number
    raise argparse.ArgumentTypeError(
        f"must be 0 (whole document in one pass) or >= {MIN_BATCH_SIZE}, got {number}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments (exits with code 2 on invalid input)."""
    parser = argparse.ArgumentParser(
        description="Extract sections, tables and figures from a large PDF with Docling."
    )
    parser.add_argument("pdf", type=Path, nargs="?",
                        help="input PDF file (not needed with --postprocess-only)")
    parser.add_argument(
        "--batchsize", type=_batch_size_arg, default=DEFAULT_BATCH_SIZE, metavar="N",
        help=f"pages per batch; 0 = whole document in one pass; otherwise >= "
             f"{MIN_BATCH_SIZE} (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output", "--out-dir", dest="output", type=Path,
        default=Path("./output_manuale"), metavar="DIR",
        help="output root directory for all artifacts, created if missing "
             "(alias: --out-dir; default: ./output_manuale)",
    )
    parser.add_argument(
        "--min-image-px", type=int, default=DEFAULT_MIN_IMAGE_PX, metavar="PX",
        help=f"skip figures with width or height below PX (default: {DEFAULT_MIN_IMAGE_PX})",
    )
    parser.add_argument(
        "--no-locate-warnings", action="store_true",
        help="do not re-convert batches page by page to locate Docling warnings "
             "(faster; warnings are then only attributed to the batch range)",
    )
    parser.add_argument(
        "--merge-tables", action="store_true",
        help="after extraction, merge tables that continue across pages",
    )
    parser.add_argument(
        "--merge-min-confidence", choices=sorted(CONFIDENCE_RANK, key=CONFIDENCE_RANK.get, reverse=True),
        default="high",
        help="weakest continuation link that is merged (default: high)",
    )
    parser.add_argument(
        "--postprocess-only", action="store_true",
        help="do not convert: merge continuation tables of the finished output in "
             "--output and refresh index and report (hand-edited continues_from "
             "values are honoured)",
    )
    parser.add_argument(
        "--redetect", action="store_true",
        help="with --postprocess-only: recompute continues_from from the stored "
             "geometry, overwriting hand edits",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore an existing checkpoint and restart from page 1 "
             "(existing output files are overwritten, never deleted)",
    )
    args = parser.parse_args(argv)
    if args.redetect and not args.postprocess_only:
        parser.error("--redetect requires --postprocess-only")
    if not args.postprocess_only and args.pdf is None:
        parser.error("the PDF argument is required unless --postprocess-only is used")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point. Returns the process exit code documented in the module header."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging()

    if args.output.exists() and not args.output.is_dir():
        log.error("--output exists but is not a directory: %s", args.output)
        return 1

    if args.postprocess_only:
        if args.pdf is not None:
            log.warning("--postprocess-only: the PDF argument is ignored")
        try:
            return run_postprocess(OutputLayout(args.output), args.merge_min_confidence, args.redetect)
        except Exception:  # noqa: BLE001
            log.exception("Fatal error during post-processing")
            return 1

    if not args.pdf.is_file():
        log.error("Input PDF not found: %s", args.pdf)
        return 1

    extractor = ManualExtractor(
        pdf_path=args.pdf,
        layout=OutputLayout(args.output),
        batch_size=args.batchsize,
        min_image_px=args.min_image_px,
        resume=not args.no_resume,
        locate_warnings=not args.no_locate_warnings,
        merge_tables=args.merge_tables,
        merge_min_confidence=args.merge_min_confidence,
    )
    try:
        return extractor.run()
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001
        log.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
