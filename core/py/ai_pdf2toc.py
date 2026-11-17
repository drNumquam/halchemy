#!/usr/bin/env python3
"""
Generate ``toc.yaml`` from the hierarchical bookmarks of a PDF document and
enrich the resulting TOC with AI-generated cross-references between sections.

Processing pipeline:

PDF
  -> bookmark extraction
  -> TOC hierarchy generation
  -> cross-references enrichment
  -> schema validation
  -> toc.yaml

Output contract (matches the ``schema.yaml`` used by the downstream tooling)::

    source: "<pdf filename>"
    pages: <page count>
    sections:
      - id: "<1.2.3>"             # printed number or positional fallback
        title: "<title>"          # title without the leading section number
        span_start: <int>         # first page, 1-based, inclusive
        span_end: <int>           # last page, 1-based, inclusive
        next: <int>               # start page of the next section in TOC preorder traversal
        references:
          - id: "<target section id>"
            type: "<semantic|structural|external>"
            confidence: <float>   # 0.0 - 1.0
        children: [...]

Notes:
    * Page numbers are 1-based physical PDF pages.
    * Granularity is the page: a section spans every page it touches, so the
      start page of a section is also included in the immediately preceding
      section. Adjacent sections overlap by one page (or more, when several
      sections start on the same page). A section ends on the start page of the
      next section at the same or a shallower level, hence a parent section
      spans all of its descendants; the last section ends on the last page.
      Footnotes are not analysed separately: since the granularity is the page,
      a footnote belongs to the section(s) spanning the page it sits on, and any
      cross-section reference it contains is outside this tool's scope.
    * ``references`` contains AI-generated cross-references between TOC nodes.
      They are inferred ONLY from the metadata of the generated TOC (ids, titles,
      hierarchy, page spans); the page text is never read. Types:
        - ``structural``: relation implied by the document layout (overview and
          detail, prerequisite, appendix of a chapter, ...);
        - ``semantic``: topically related sections a reader would consult together;
        - ``external``: reference to a document outside this one, with id = ""
          (at most one external reference per section).
      ``confidence`` is in [0, 1]; references below ``--min-confidence`` are
      dropped. Targets must be ids present in the TOC; self-references are
      discarded, duplicates (same id and type) are merged, and at most 10
      references are kept per section. A section may have zero or more references.
      The inference is not deterministic and the section titles are sent to the
      selected provider (GitHub Copilot, or the local Qwen server).
    * ``next`` always points to the start page of the next TOC node in document
      order. For the last node, next is -1.
    * When a title carries no printed number (e.g. "Deployment Guidelines"),
      ``id`` falls back to its positional outline number. Duplicate ids are
      reported as warnings (references to a duplicated id are ambiguous).
    * A report.txt is generated: it mirrors the log of the run and ends with the
      errors and warnings of each TOC entry that has any, both from the TOC
      extraction and from the AI cross-references inference. Failures of single
      AI requests are warnings/errors of that entry only.

Usage:
    python ai_pdf2toc.py --pdf <manual.pdf> --output <out_dir> [--schema <schema.yaml>]
                         [--no-ai] [--provider copilot|qwen] [--model <name>]
                         [--base-url <url>] [--concurrency <n>] [--retries <n>]
                         [--min-confidence <0-1>] [--context "<document description>"]
                         [-v]

Options:
    --pdf PATH              input PDF (required)
    -o, --output DIR        directory for toc.yaml and report.txt (required)
    --schema PATH           YAML/JSON Schema used to validate toc.yaml before writing
    --no-ai                 skip the cross-references inference (references: [])
    --provider NAME         copilot (default) | qwen
    --model NAME            model name (default: "auto" for copilot,
                            "qwen/qwen3.5-9b" for qwen)
    --base-url URL          OpenAI-compatible endpoint, qwen only
                            (default: http://localhost:1234/v1)
    --concurrency N         parallel AI requests (default 4)
    --retries N             retries per section after a failed request (default 2)
    --min-confidence C      drop references with confidence below C (default 0.0)
    --context TEXT          short description of the document added to the prompt
    -v, --verbose           debug logging

Examples:
    python ai_pdf2toc.py --pdf manual.pdf -o out --no-ai
    python ai_pdf2toc.py --pdf manual.pdf -o out --schema schema.yaml \
        --provider qwen --model "qwen/qwen3.5-9b" --concurrency 2 \
        --context "ARM Cortex-R Technical Reference Manual"

Requirements:
    python -m pip install pymupdf
    python -m pip install pyyaml jsonschema     # only with --schema
    python -m pip install httpx openai          # only with --provider qwen
    the Copilot SDK providing ``copilot.CopilotClient``   # only with --provider copilot

Exit codes:
    0  success (some single AI requests may have failed: see report.txt)
    1  the PDF has no usable bookmarks (nothing written)
    2  error (bad input, missing dependency, I/O failure, schema violation,
       or every AI request failed; nothing written)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Iterator, Sequence

LOGGER = logging.getLogger("pdf2toc")

TOC_FILENAME = "toc.yaml"
REPORT_FILENAME = "report.txt"
DEFAULT_SCHEMA_HINT = "./schema.yaml"

# --- LLM configuration ------------------------------------------------------ #
DEFAULT_PROVIDER = "copilot"            # "copilot" or "qwen"
COPILOT_MODEL_NAME = "auto"
QWEN_MODEL_NAME = "qwen/qwen3.5-9b"
QWEN_BASE_URL = "http://localhost:1234/v1"
QWEN_API_KEY = "lm-studio"
MAX_CONCURRENT_REQUESTS = 1
DEFAULT_RETRIES = 2
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 600.0
MAX_REFERENCES = 10
REFERENCE_TYPES = ("semantic", "structural", "external")

# Matches "2", "2.1.1", "2.1.1." and appendix-style "A.1" followed by the title.
NUMBER_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?\s+(?P<title>\S.*)$"
)


class ExitCode(IntEnum):
    """Process exit codes."""

    OK = 0
    NO_TOC = 1
    FAILURE = 2


class TocError(Exception):
    """Raised for expected, user-reportable failures."""


@dataclass(frozen=True)
class Bookmark:
    """A validated bookmark with a resolved 1-based page number."""

    level: int
    title: str
    page: int
    notes: tuple[str, ...] = ()     # per-entry warnings found while normalizing


@dataclass
class Section:
    """One node of the nested table of contents."""

    id: str
    raw_title: str
    level: int
    start_page: int
    end_page: int = 0
    next: int = -1
    title: str = ""
    parent: Section | None = field(default=None, repr=False)
    children: list[Section] = field(default_factory=list, repr=False)
    references: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # per-entry warnings/errors


# --------------------------------------------------------------------------- #
# Command line and logging
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Create toc.yaml and report.txt from a PDF bookmark tree, "
                    "enriched with AI-generated cross-references."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--schema",
        help="Optional YAML/JSON Schema file used to validate toc.yaml.",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        required=True,
        help="Directory where toc.yaml and report.txt are written.",
    )
    ai = parser.add_argument_group("AI cross-references")
    ai.add_argument("--no-ai", action="store_true",
                    help="Skip the cross-references inference (references: []).")
    ai.add_argument("--provider", choices=("copilot", "qwen"), default=DEFAULT_PROVIDER,
                    help=f"LLM provider (default: {DEFAULT_PROVIDER}).")
    ai.add_argument("--model",
                    help=f"Model name (default: '{COPILOT_MODEL_NAME}' for copilot, "
                         f"'{QWEN_MODEL_NAME}' for qwen).")
    ai.add_argument("--base-url", default=QWEN_BASE_URL,
                    help=f"OpenAI-compatible endpoint for qwen (default: {QWEN_BASE_URL}).")
    ai.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_REQUESTS,
                    help=f"Parallel requests (default: {MAX_CONCURRENT_REQUESTS}).")
    ai.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                    help=f"Retries per section after a failed request (default: {DEFAULT_RETRIES}).")
    ai.add_argument("--min-confidence", type=float, default=0.0,
                    help="Drop references with a lower confidence (default: 0.0).")
    ai.add_argument("--context",
                    help="Short description of the document, added to the prompt "
                         "(e.g. 'ARM Cortex-R Technical Reference Manual').")

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

    Each raw bookmark is ``[level, title, page]``; ``page`` is 1-based and is
    ``-1`` (or lower than 1) when the destination cannot be resolved.
    """
    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise TocError("The PDF is password-protected; no TOC was created.")
            return document.page_count, document.get_toc(simple=True)
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
    Entries that are malformed or point beyond the last page are dropped (they
    have no TOC entry, so they are reported in the general log).
    """
    candidates: list[tuple[int, str, int | None]] = []
    for position, entry in enumerate(raw, start=1):
        if len(entry) < 3:
            LOGGER.warning("Bookmark #%d is malformed and was skipped.", position)
            continue
        level_raw, title_raw, page_raw = entry[:3]
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
        candidates.append((level, title, page if page >= 1 else None))

    resolved: list[Bookmark] = []
    next_page: int | None = None
    for level, title, page in reversed(candidates):
        notes: tuple[str, ...] = ()
        if page is None:
            if next_page is None:
                LOGGER.warning("Bookmark %r has no destination and no successor; skipped.", title)
                continue
            notes = (f"WARNING: bookmark has no destination; page {next_page} of the "
                     "next bookmark was used.",)
            page = next_page
        next_page = page
        resolved.append(Bookmark(level=level, title=title, page=page, notes=notes))
    resolved.reverse()
    return resolved


def nest_sections(bookmarks: Sequence[Bookmark], page_count: int) -> list[Section]:
    """Nest bookmarks by level and compute inclusive 1-based page spans (single pass).

    A section ends on the start page of the next bookmark at the same or a
    shallower level: that page is shared with the following section.
    """
    roots: list[Section] = []
    stack: list[Section] = []
    previous_page = 0

    for bookmark in bookmarks:
        node = Section(
            id="",  # assigned later by assign_numbers
            raw_title=bookmark.title,
            level=bookmark.level,
            start_page=bookmark.page,
            notes=list(bookmark.notes),
        )
        if bookmark.page < previous_page:
            node.notes.append(
                f"WARNING: page {bookmark.page} precedes the previous bookmark "
                f"(page {previous_page}); its span is clamped to a single page."
            )
        previous_page = bookmark.page

        # Close every open section at the same or a deeper level.
        while stack and stack[-1].level >= node.level:
            closed = stack.pop()
            closed.end_page = max(closed.start_page, node.start_page)
        if stack:
            node.parent = stack[-1]
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
            section.id = match.group("number")
            section.title = match.group("title")
        else:
            section.id = f"{prefix}{position}"
            section.title = section.raw_title
            fallbacks += 1
        fallbacks += assign_numbers(section.children, prefix=f"{section.id}.")
    return fallbacks


def walk(sections: Sequence[Section]) -> Iterator[Section]:
    """Yield every section in document (pre-order) order."""
    for section in sections:
        yield section
        yield from walk(section.children)


def check_consistency(
    sections: Sequence[Section], page_count: int, parent: Section | None = None
) -> int:
    """Record structural anomalies as per-entry warnings and return their count."""
    issues = 0
    for section in sections:
        if not 1 <= section.start_page <= section.end_page <= page_count:
            section.notes.append(
                f"WARNING: invalid span {section.start_page}-{section.end_page}."
            )
            issues += 1
        if parent and not (
            parent.start_page <= section.start_page and section.end_page <= parent.end_page
        ):
            section.notes.append(
                f"WARNING: span {section.start_page}-{section.end_page} lies outside "
                f"its parent {parent.id} ({parent.start_page}-{parent.end_page})."
            )
            issues += 1
        issues += check_consistency(section.children, page_count, section)
    return issues


def check_duplicate_ids(sections: Sequence[Section]) -> dict[str, Section]:
    """Warn about duplicate ids; return a lookup id -> first section using it."""
    first: dict[str, Section] = {}
    for section in walk(sections):
        if section.id in first:
            section.notes.append(
                f"WARNING: duplicate id {section.id!r} (also used by "
                f"{first[section.id].title!r}); references to it are ambiguous."
            )
        else:
            first[section.id] = section
    return first


def assign_next(sections: Sequence[Section]) -> None:
    """Set ``next`` to the start page of the following node in pre-order (-1 for the last)."""
    nodes = list(walk(sections))
    for index, node in enumerate(nodes):
        node.next = nodes[index + 1].start_page if index + 1 < len(nodes) else -1


def build_sections(raw: Sequence[Sequence[Any]], page_count: int) -> list[Section]:
    """Full pipeline: raw bookmarks -> numbered, consistent section tree."""
    bookmarks = normalize_bookmarks(raw, page_count)
    roots = nest_sections(bookmarks, page_count)
    fallbacks = assign_numbers(roots)
    assign_next(roots)
    if fallbacks:
        LOGGER.info("%d section(s) had no printed number; positional numbers were used.", fallbacks)
    issues = check_consistency(roots, page_count)
    if issues:
        LOGGER.warning("%d structural anomaly(ies) detected; review the TOC.", issues)
    return roots


# --------------------------------------------------------------------------- #
# LLM backends (copilot / qwen)
# --------------------------------------------------------------------------- #
def strip_thinking(text: str | None) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


class LLMBackend:
    """Async context manager exposing ``complete`` for a numbered worker slot."""

    async def __aenter__(self) -> LLMBackend:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def complete(self, slot: int, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


def normalize_base_url(url: str) -> str:
    """Append ``/v1`` when the endpoint has no path (OpenAI-compatible servers)."""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        raise TocError(f"--base-url must look like http://host:port/v1, got {url!r}.")
    if parts.path in ("", "/"):
        fixed = urlunsplit((parts.scheme, parts.netloc, "/v1", "", ""))
        LOGGER.info("--base-url has no path; using %s", fixed)
        return fixed
    return url.strip().rstrip("/")


class QwenBackend(LLMBackend):
    """Local OpenAI-compatible server (e.g. LM Studio); blocking calls run in threads."""

    def __init__(self, model: str, base_url: str) -> None:
        try:
            import httpx
            import openai
        except ImportError as exc:
            raise TocError(
                "The qwen provider requires httpx and openai: "
                "python -m pip install httpx openai"
            ) from exc
        self._openai = openai
        self.model = model
        self._use_reasoning_effort = False #True
        self._client = openai.OpenAI(
            base_url=normalize_base_url(base_url),
            api_key=QWEN_API_KEY,
            timeout=LLM_TIMEOUT,
            http_client=httpx.Client(trust_env=False, timeout=LLM_TIMEOUT),
        )

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=False,
        )
        if self._use_reasoning_effort:
            try:
                response = self._client.chat.completions.create(
                    reasoning_effort="none", **kwargs)
                return strip_thinking(response.choices[0].message.content)
            except (TypeError, self._openai.BadRequestError,
                    self._openai.UnprocessableEntityError) as exc:
                # Retry without the parameter only if it is the one being rejected;
                # any other error (e.g. server failure) must not double the requests.
                if not isinstance(exc, TypeError) and "reasoning" not in str(exc).lower():
                    raise
                self._use_reasoning_effort = False
                LOGGER.info("The server does not accept 'reasoning_effort'; "
                            "continuing without it.")
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content

        if not content:
            raise RuntimeError(
                f"Empty response from Qwen. "
                f"Finish reason: {response.choices[0].finish_reason}. "
                f"Reasoning: {getattr(message, 'reasoning_content', '')[:200]}"
            )

        return strip_thinking(content)

    async def complete(self, slot: int, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self._call, system_prompt, user_prompt)

    async def __aexit__(self, *exc_info: object) -> None:
        self._client.close()


class CopilotBackend(LLMBackend):
    """GitHub Copilot: one session per worker slot."""

    def __init__(self, model: str, slots: int) -> None:
        self.model = model
        self.slots = slots
        self._stack: AsyncExitStack | None = None
        self._sessions: list[Any] = []

    async def __aenter__(self) -> CopilotBackend:
        try:
            from copilot import CopilotClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TocError(
                "The copilot provider requires the Copilot SDK (module 'copilot')."
            ) from exc
        self._stack = AsyncExitStack()
        try:
            client = await self._stack.enter_async_context(CopilotClient())
            self._sessions = [
                await self._stack.enter_async_context(
                    await client.create_session(model=self.model)
                )
                for _ in range(self.slots)
            ]
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def complete(self, slot: int, system_prompt: str, user_prompt: str) -> str:
        response = await self._sessions[slot].send_and_wait(f"{system_prompt}\n\n{user_prompt}")
        if response is None or not response.data.content:
            raise RuntimeError("GitHub Copilot returned no content.")
        return strip_thinking(response.data.content)


def create_backend(args: argparse.Namespace, slots: int) -> LLMBackend:
    """Instantiate the backend selected on the command line."""
    if args.provider == "qwen":
        return QwenBackend(args.model or QWEN_MODEL_NAME, args.base_url)
    return CopilotBackend(args.model or COPILOT_MODEL_NAME, slots)


# --------------------------------------------------------------------------- #
# AI cross-references
# --------------------------------------------------------------------------- #
def build_system_prompt(context: str | None) -> str:
    """Instructions shared by every request."""
    subject = f" ({context})" if context else ""
    return (
        f"You are an expert technical reader analysing the table of contents of a "
        f"technical document{subject}.\n"
        "You receive the full list of sections (id, parent id, title) and ONE target "
        "section with its page span. Using ONLY this metadata, list the OTHER sections "
        "that the target most likely cross-references.\n\n"
        "REFERENCE TYPES:\n"
        "- structural: relation implied by the document layout (overview and detail, "
        "prerequisite, summary, appendix of a chapter, ...).\n"
        "- semantic: topically related sections that a reader would consult together.\n"
        "- external: the target probably cites a document outside this one "
        '(standards, other manuals). Use id "".\n\n'
        "RULES:\n"
        "- Copy target ids exactly from the list. Never reference the target itself.\n"
        "- Be conservative: returning no references is fine.\n"
        "- confidence is a float from 0.00 to 1.00: how likely the reference is real.\n"
        f"- At most {MAX_REFERENCES} references.\n"
        "- Output ONLY a JSON object, with no Markdown fences and no other text:\n"
        '{"references": [{"id": "<section id>", "type": "semantic|structural|external", '
        '"confidence": 0.0}]}'
    )


def build_listing(nodes: Sequence[Section]) -> str:
    """Compact list of every section; identical for all requests (cache friendly)."""
    return "\n".join(
        json.dumps(
            {"id": n.id, "parent": n.parent.id if n.parent else None, "title": n.title},
            ensure_ascii=False,
        )
        for n in nodes
    )


def build_user_prompt(listing: str, node: Section) -> str:
    """Per-section prompt: shared listing first, then the target."""
    target = {
        "id": node.id,
        "title": node.title,
        "parent": node.parent.id if node.parent else None,
        "span_start": node.start_page,
        "span_end": node.end_page,
        "next": node.next,
    }
    return (
        f"ALL_NODES (one JSON object per line):\n{listing}\n\n"
        f"TARGET:\n{json.dumps(target, ensure_ascii=False)}"
    )


def parse_reference_payload(content: str) -> list[Any]:
    """Extract the ``references`` list from an LLM answer; raise ValueError if absent."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("references"), list):
            return data["references"]
        if isinstance(data, list):
            return data
    raise ValueError("the answer does not contain a valid JSON 'references' list")


def clean_references(
    raw: Sequence[Any],
    node: Section,
    by_id: dict[str, Section],
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate LLM references; return ``(references, problems)``."""
    best: dict[tuple[str, str], float] = {}
    problems: list[str] = []
    for ref in raw:
        if not isinstance(ref, dict):
            problems.append(f"discarded malformed reference {ref!r}")
            continue
        ref_type = str(ref.get("type", "")).strip().lower()
        ref_id = "" if ref.get("id") is None else str(ref.get("id")).strip()
        if ref_type not in REFERENCE_TYPES:
            problems.append(f"discarded reference {ref!r}: invalid type")
            continue
        if ref_type == "external":
            ref_id = ""
        elif ref_id == node.id:
            problems.append(f"discarded self-reference to {ref_id!r}")
            continue
        elif ref_id not in by_id:
            problems.append(f"discarded reference to unknown id {ref_id!r}")
            continue
        try:
            confidence = float(ref.get("confidence", 0.0))
        except (TypeError, ValueError):
            problems.append(f"discarded reference {ref!r}: invalid confidence")
            continue
        if not math.isfinite(confidence):
            problems.append(f"discarded reference {ref!r}: invalid confidence")
            continue
        confidence = max(0.0, min(1.0, confidence))
        if confidence < min_confidence:
            continue
        key = (ref_id, ref_type)
        best[key] = max(best.get(key, 0.0), confidence)

    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    references = [
        {"id": key[0], "type": key[1], "confidence": round(confidence, 2)}
        for key, confidence in ordered[:MAX_REFERENCES]
    ]
    return references, problems


async def enrich_references(
    sections: Sequence[Section], by_id: dict[str, Section], args: argparse.Namespace
) -> tuple[int, int]:
    """Fill ``Section.references`` with parallel LLM requests.

    Worker ``k`` owns session/slot ``k`` and handles nodes k, k+W, k+2W, ...
    Returns ``(failed_nodes, total_nodes)``.
    """
    nodes = list(walk(sections))
    system_prompt = build_system_prompt(args.context)
    listing = build_listing(nodes)
    slots = max(1, min(args.concurrency, len(nodes)))
    attempts = args.retries + 1
    approx_tokens = (len(system_prompt) + len(build_user_prompt(listing, nodes[0]))) // 3
    LOGGER.info("Each request is about %d tokens (the full TOC is sent every time); "
                "the model context must exceed this by at least %d, for each of the "
                "%d parallel requests.", approx_tokens, LLM_MAX_TOKENS, slots)
    state = {"failed": 0, "done": 0}

    async with create_backend(args, slots) as backend:

        async def worker(slot: int) -> None:
            for index in range(slot, len(nodes), slots):
                node = nodes[index]
                LOGGER.debug("Processing node %d/%d (ID %s)...", index + 1, len(nodes), node.id)
                user_prompt = build_user_prompt(listing, node)
                started = time.monotonic()
                error: Exception | None = None
                for attempt in range(1, attempts + 1):
                    try:
                        content = await backend.complete(slot, system_prompt, user_prompt)
                        raw = parse_reference_payload(content)
                    except Exception as exc:  # network, auth, empty/invalid answer...
                        error = exc
                        if attempt < attempts:
                            LOGGER.debug("Retry %d/%d for node %s: %s",
                                         attempt, args.retries, node.id, exc)
                            await asyncio.sleep(min(attempt, 5))
                        continue
                    node.references, problems = clean_references(
                        raw, node, by_id, args.min_confidence
                    )
                    node.notes.extend(f"WARNING: {problem}" for problem in problems)
                    error = None
                    break
                if error is not None:
                    state["failed"] += 1
                    node.notes.append(
                        f"ERROR: AI inference failed after {attempts} attempt(s): {error}"
                    )
                state["done"] += 1
                LOGGER.info("[%d/%d] section %s: %s (%.1fs)", state["done"], len(nodes),
                            node.id,
                            "FAILED" if error is not None
                            else f"{len(node.references)} reference(s)",
                            time.monotonic() - started)

        await asyncio.gather(*(worker(slot) for slot in range(slots)))
    return state["failed"], len(nodes)


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
                f"{pad}- id: {quote(section.id)}",
                f"{pad}  title: {quote(section.title)}",
                f"{pad}  span_start: {section.start_page}",
                f"{pad}  span_end: {section.end_page}",
                f"{pad}  next: {section.next}",
            ])
            if section.references:
                lines.append(f"{pad}  references:")
                for ref in section.references:
                    lines.extend([
                        f"{pad}    - id: {quote(ref['id'])}",
                        f"{pad}      type: {quote(ref['type'])}",
                        f"{pad}      confidence: {ref['confidence']:.2f}",
                    ])
            else:
                lines.append(f"{pad}  references: []")
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


def validate_toc(toc_text: str, schema_path: Path) -> tuple[bool, str]:
    """Validate the generated YAML text against a JSON Schema (YAML or JSON file)."""
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
        toc_data = yaml.safe_load(toc_text)
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


def log_entry_report(sections: Sequence[Section]) -> None:
    """Log the warnings/errors of each TOC entry (they end up in report.txt)."""
    nodes = list(walk(sections))
    flagged = [node for node in nodes if node.notes]
    if not flagged:
        LOGGER.info("Per-entry report: no warnings or errors (%d entries).", len(nodes))
        return
    LOGGER.info("Per-entry report: %d of %d entries have warnings or errors.",
                len(flagged), len(nodes))
    for node in flagged:
        LOGGER.info("  [%s] %s (pp. %d-%d)", node.id, node.title,
                    node.start_page, node.end_page)
        for note in node.notes:
            LOGGER.info("      %s", note)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def check_arguments(args: argparse.Namespace) -> None:
    """Reject out-of-range numeric options."""
    if not 0.0 <= args.min_confidence <= 1.0:
        raise TocError("--min-confidence must be between 0 and 1.")
    if args.concurrency < 1:
        raise TocError("--concurrency must be at least 1.")
    if args.retries < 0:
        raise TocError("--retries must not be negative.")


def run(args: argparse.Namespace, output_dir: Path) -> ExitCode:
    """Execute the PDF -> toc.yaml workflow."""
    check_arguments(args)
    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise TocError(f"The input file does not exist or is not a PDF: {pdf_path}")

    pymupdf = load_pymupdf()
    LOGGER.info("Input PDF: %s", pdf_path)
    LOGGER.info("Output directory: %s", output_dir)

    page_count, raw_bookmarks = read_bookmarks(pymupdf, pdf_path)
    sections = build_sections(raw_bookmarks, page_count)
    if not sections:
        LOGGER.warning("The PDF has no usable bookmarks. No %s was created.", TOC_FILENAME)
        return ExitCode.NO_TOC

    by_id = check_duplicate_ids(sections)
    LOGGER.info("PDF pages: %d", page_count)
    LOGGER.info("Usable TOC entries: %d", sum(1 for _ in walk(sections)))
    LOGGER.info("Spans are 1-based physical PDF pages, inclusive; a section also "
                "includes the start page of the next one (adjacent spans overlap).")

    try:
        if args.no_ai:
            LOGGER.info("AI cross-references skipped (--no-ai).")
        else:
            LOGGER.info("Inferring cross-references with provider '%s'...", args.provider)
            failed, total = asyncio.run(enrich_references(sections, by_id, args))
            if failed == total:
                LOGGER.error(
                    "All %d AI requests failed; %s was not written. "
                    "Check the provider or use --no-ai.", total, TOC_FILENAME)
                return ExitCode.FAILURE
            if failed:
                LOGGER.warning("%d of %d AI requests failed; those sections have "
                               "no references.", failed, total)
            LOGGER.info("References generated: %d in %d section(s).",
                        sum(len(n.references) for n in walk(sections)),
                        sum(1 for n in walk(sections) if n.references))

        schema_path = Path(args.schema).expanduser() if args.schema else None
        toc_text = render_yaml(pdf_path.name, page_count, sections,
                               schema_hint(output_dir, schema_path))

        # Validate before writing, so an invalid TOC never reaches the disk.
        if schema_path is None:
            LOGGER.info("Schema validation skipped: no --schema file was specified.")
        else:
            valid, message = validate_toc(toc_text, schema_path)
            if not valid:
                LOGGER.error("Validation failed: %s. %s was not written.", message, TOC_FILENAME)
                return ExitCode.FAILURE
            LOGGER.info("Validation passed: %s", message)

        toc_path = output_dir / TOC_FILENAME
        write_text_atomic(toc_path, toc_text)
        LOGGER.info("Created TOC file: %s", toc_path)
        LOGGER.info("Process completed successfully.")
        return ExitCode.OK
    finally:
        log_entry_report(sections)


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
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user; %s was not updated.", TOC_FILENAME)
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

