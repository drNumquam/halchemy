# =============================================================================
# CONTRACT
# =============================================================================
# Purpose:
#   Export a PDF (or a range of its pages) to a complete JSON document using
#   Docling (layout analysis, tables, optional OCR, picture extraction).
#   The console output is also mirrored to a report file (report.txt).
#
# Syntax:
#   python pdf_to_json_docling.py --pdf <PDF_FILE> --output <OUTPUT_DIR>
#                                 [--range MIN MAX] [--no_ocr]
#
# Arguments:
#   --pdf     (required) Path of the input PDF file. Must exist, be a file and
#             have a .pdf extension. "~" is expanded, the path is resolved.
#   --output  (required) Destination folder. Created if missing; it must not
#             be an existing non-directory path.
#   --range   (optional) Two integers MIN MAX: first and last page to process
#             (1-based, inclusive). Constraints: 1 <= MIN <= MAX <= total pages.
#             If omitted, the whole PDF is processed.
#   --no_ocr  (optional) Disable OCR. Use only if the PDF is fully native
#             (not scanned). OCR is enabled by default.
#
# Examples:
#   python pdf_to_json_docling.py --pdf doc.pdf --output out
#   python pdf_to_json_docling.py --pdf doc.pdf --output out --range 2 6
#   python pdf_to_json_docling.py --pdf doc.pdf --output out --no_ocr
#
# Output (inside --output):
#   <pdf_stem>.json              full run
#   <pdf_stem>_p<MIN>-<MAX>.json when --range is used
#   report.txt                   copy of stdout/stderr and Docling INFO logs
#                                (overwritten at every run)
#   Pictures are exported as referenced image files (ImageRefMode.REFERENCED),
#   handled by Docling next to the JSON file, and rendered at scale 2.0.
#
# Exit codes:
#   0  success
#   1  validation error (missing/invalid file, invalid --range, invalid --output)
#   2  unexpected error during the Docling conversion
#
# Dependencies:
#   - Python >= 3.10 (uses the "X | None" type-hint syntax)
#   - docling        (DocumentConverter, PdfPipelineOptions, InputFormat)
#   - docling-core   (docling_core.types.doc.ImageRefMode; installed with docling)
#   - pypdfium2      (reads the total page count of the PDF)
#   - Standard library: argparse, logging, json, io, sys, pathlib, contextlib
#   Installation:
#       pip install docling pypdfium2
#   Note: on the first run Docling may download its AI models (network needed).
# =============================================================================

import argparse
import logging
import json
import io
import re
import sys
from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, HeadingHierarchyOptions
from docling_core.types.doc import (
    ImageRefMode,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)


class Tee(io.TextIOBase):
    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    def isatty(self) -> bool:
        return bool(
            hasattr(self.console_stream, "isatty")
            and self.console_stream.isatty()
        )

    def write(self, s: str) -> int:
        if not s:
            return 0
        self.console_stream.write(s)
        self.console_stream.flush()
        self.file_stream.write(s)
        self.file_stream.flush()
        return len(s)

    def flush(self) -> None:
        self.console_stream.flush()
        self.file_stream.flush()


@contextmanager
def tee_output_to_file(report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the real references to the console
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with report_path.open("w", encoding="utf-8") as report_file:
        tee_out = Tee(orig_stdout, report_file)
        tee_err = Tee(orig_stderr, report_file)

        # Handler only for Docling/Python logs going to the file (the screen already receives stderr via tee_err)
        log_handler = logging.StreamHandler(report_file)
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(
            logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        )

        docling_logger = logging.getLogger("docling")
        prev_propagate = docling_logger.propagate
        prev_level = docling_logger.level
        if docling_logger.level == logging.NOTSET or docling_logger.level > logging.INFO:
            docling_logger.setLevel(logging.INFO)
        docling_logger.addHandler(log_handler)

        sys.stdout = tee_out
        sys.stderr = tee_err
        try:
            yield report_path
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            docling_logger.removeHandler(log_handler)
            docling_logger.setLevel(prev_level)
            docling_logger.propagate = prev_propagate
            log_handler.close()


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the total number of pages of the PDF using pypdfium2."""
    pdf_doc = None
    try:
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf_doc)
        if page_count <= 0:
            raise ValueError("The PDF does not contain any valid pages.")
        return page_count
    except Exception as exc:
        raise ValueError(f"Unable to read the PDF '{pdf_path}': {exc}") from exc
    finally:
        if pdf_doc is not None:
            pdf_doc.close()


def validate_inputs(
    pdf_raw: str,
    output_raw: str,
    range_values: list[int] | None,
) -> tuple[Path, Path, tuple[int, int] | None, int]:
    """Validate --pdf, --output and --range (1-based, inclusive)."""
    pdf_path = Path(pdf_raw).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"The PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"The --pdf path does not point to a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"The specified file does not have a .pdf extension: {pdf_path}")

    total_pages = get_pdf_page_count(pdf_path)

    validated_range: tuple[int, int] | None = None
    if range_values is not None:
        if len(range_values) != 2:
            raise ValueError("--range requires exactly two integers: min max (e.g. --range 1 5).")

        start_page, end_page = range_values

        if start_page < 1:
            raise ValueError(
                f"The start page (min={start_page}) must be greater than or equal to 1."
            )
        if end_page < 1:
            raise ValueError(
                f"The end page (max={end_page}) must be greater than or equal to 1."
            )
        if start_page > end_page:
            raise ValueError(
                f"Invalid range: min ({start_page}) cannot be greater than max ({end_page})."
            )
        if end_page > total_pages:
            raise ValueError(
                f"The end page (max={end_page}) exceeds the total number of pages in the PDF ({total_pages})."
            )

        validated_range = (start_page, end_page)

    output_dir = Path(output_raw).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"The --output path already exists but is not a folder: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return pdf_path, output_dir, validated_range, total_pages


def export_pdf_to_json(
    pdf_path: Path,
    output_dir: Path,
    page_range: tuple[int, int] | None,
    no_ocr: bool = False
) -> Path:
    """Convert the PDF (or the given range) with Docling and save the all-inclusive JSON."""
#    converter = DocumentConverter()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0  # optional (default ~1.0)
    pipeline_options.do_ocr = not no_ocr
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True,
        use_bookmarks=True,
        use_numbering=True,
        use_style=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
        show_progress=True
    )

    if page_range is not None:
        conv_result = converter.convert(source=str(pdf_path), page_range=page_range)
        output_filename = f"{pdf_path.stem}_p{page_range[0]}-{page_range[1]}.json"
    else:
        conv_result = converter.convert(source=str(pdf_path))
        output_filename = f"{pdf_path.stem}.json"

    doc = conv_result.document
    json_path = output_dir / output_filename
    json_path.parent.mkdir(parents=True, exist_ok=True)

    doc.save_as_json(
        json_path,
        image_mode=ImageRefMode.REFERENCED,
    )

    build_and_save_toc(doc, output_dir)

    return json_path


def build_and_save_toc(doc, output_dir: Path) -> None:
    """Build indented TOC (toc_ai.txt) and ID-based index (sections_by_id.json) using Docling's HeadingHierarchy."""
    # Patterns for non-section elements sometimes misclassified by layout detection
    bullet_pattern = re.compile(r"^[\u00b7\u2022\-\*\•\·]\s+")
    noise_heading_pattern = re.compile(r"^(?:Where:|Table\s+\d+\.|Figure\s+\d+\.|Equation\s+\d+:)\b", re.IGNORECASE)

    def clean_heading_text(raw_title: str) -> str:
        cleaned = re.sub(r"^[^\w\s\(\)\[\]]+\s*", "", raw_title.strip())
        return cleaned.strip()

    mapped_sections = {}
    current_section = None
    id_counter = 0

    def create_section(title: str, level: int, page: int | str) -> dict:
        nonlocal id_counter
        sec_id = f"SEC_{id_counter:04d}"
        id_counter += 1
        return {
            "id": sec_id,
            "level": level,
            "title": title,
            "page": page,
            "text": [],
            "tables_md": [],
            "images": [],
        }

    for item, tree_level in doc.iterate_items():
        page = getattr(item.prov[0], "page_no", "?") if getattr(item, "prov", None) else "?"

        if isinstance(item, SectionHeaderItem):
            raw_text = item.text.strip()

            # 1. Convert bullet points (e.g. "· Sleep mode") into markdown list items inside current section
            if bullet_pattern.match(raw_text):
                if current_section is None:
                    current_section = create_section("Preamble / Initial Content", 1, page)
                bullet_label = bullet_pattern.sub("", raw_text).strip()
                current_section["text"].append(f"- **{bullet_label}**")
                continue

            clean_title = clean_heading_text(raw_text)

            # 2. Convert captions/formulas (e.g. "Where:", "Table 52.", "Equation 1:") into inline bold text
            if noise_heading_pattern.match(clean_title):
                if current_section is None:
                    current_section = create_section("Preamble / Initial Content", 1, page)
                current_section["text"].append(f"**{clean_title}**")
                continue

            if current_section is not None:
                mapped_sections[current_section["id"]] = current_section

            # Read the hierarchical level computed by HeadingHierarchyOptions (stored on item.level)
            heading_level = max(1, getattr(item, "level", tree_level))
            current_section = create_section(clean_title, heading_level, page)
        else:
            if current_section is None:
                current_section = create_section("Preamble / Initial Content", 1, page)

            if isinstance(item, TextItem):
                current_section["text"].append(item.text)
            elif isinstance(item, TableItem):
                current_section["tables_md"].append(item.export_to_markdown(doc))
            elif isinstance(item, PictureItem):
                uri = str(item.image.uri) if (item.image and item.image.uri) else None
                current_section["images"].append({
                    "uri": uri,
                    "caption": item.caption_text(doc),
                })

    if current_section is not None:
        mapped_sections[current_section["id"]] = current_section

    toc_lines = []
    for sec_id, data in mapped_sections.items():
        indentation = "  " * max(0, data["level"] - 1)
        n_tab = len(data["tables_md"])
        n_img = len(data["images"])

        badges = []
        if n_tab > 0:
            badges.append(f"T:{n_tab}")
        if n_img > 0:
            badges.append(f"I:{n_img}")
        badge_str = f" [{' '.join(badges)}]" if badges else ""

        toc_lines.append(
            f"{indentation}[{sec_id}] {data['title']} (Page {data['page']}){badge_str}"
        )

    (output_dir / "toc_ai.txt").write_text("\n".join(toc_lines), encoding="utf-8")
    with open(output_dir / "sections_by_id.json", "w", encoding="utf-8") as f:
        json.dump(mapped_sections, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PDF (or a range of pages) to a complete JSON using Docling."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Full path of the input PDF file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination folder where the resulting JSON file will be saved.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=None,
        help="Start and end page (1-based, inclusive), e.g. --range 2 6. If omitted, the whole PDF is processed.",
    )
    parser.add_argument(
        "--no_ocr",
        action="store_true",
        default=False,
        help="Disable OCR if the input PDF is entirely native (not scanned).",
    )   
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pdf_path, output_dir, page_range, total_pages = validate_inputs(
            pdf_raw=args.pdf,
            output_raw=args.output,
            range_values=args.range,
        )

        report_path = output_dir / "report.txt"

        with tee_output_to_file(report_path):
            print("*** Invoking Docling ***")
            print("PDF file: ", end="")
            print(pdf_path)
            print(f"Pages: {total_pages}")
            if page_range is not None:
                print(f"Range: {page_range[0]} - {page_range[1]}")
            if args.no_ocr:
                print("OCR DISABLED")
            print()

            out_file = export_pdf_to_json(
                pdf_path=pdf_path,
                output_dir=output_dir,
                page_range=page_range,
                no_ocr=args.no_ocr
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error during conversion with Docling: {exc}", file=sys.stderr)
        return 2

    scope = f"pages {page_range[0]}-{page_range[1]} of {total_pages}" if page_range else f"all {total_pages} pages"
    print()
    print(f"Table Of Contents (TOC) exported to: {output_dir / 'toc_ai.txt'}")
    print(f"Sections by ID exported to: {output_dir / 'sections_by_id.json'}")
    print(f"JSON exported SUCCESSFULLY ({scope}): {out_file}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
