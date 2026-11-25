import re
from pathlib import Path
import argparse
from typing import Any
import sys

# Docling related imports.
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    DocItemLabel,
    ImageRefMode,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# PyMuPDF related imports.
import pymupdf as fitz



'''
Section headers
'''
# Matches section identifiers at the beginning of a heading
# (e.g. "1", "1.2.3", "A.1", "SEC-01", or "Appendix").
SECTION_NAME_PATTERN = re.compile(
    r"^(?:"
    r"appendix(?=\s|$|:|\)|\]|–|—|-)"
    r"|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*(?=\s|$|:|\)|\]|–|—|-)"
    r")",
    re.IGNORECASE,
)

# Strip whitespaces at the beginning of the input string.
def strip_whitespaces_at_beginning(text: str) -> str:
    return text.lstrip()

# Returns True if the formatted input is a string starting with a valid section identifier.
def is_valid_section_name(text: Any) -> bool:
    if not isinstance(text, str):
        return False

    # Strip whitespaces at the beginning.
    text = strip_whitespaces_at_beginning(text)

    return SECTION_NAME_PATTERN.match(text) is not None



'''
PyMuPDF
'''
# Return the total number of pages of the PDF using PyMuPDF.
def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        with fitz.open(pdf_path) as pdf_doc:
            page_count = pdf_doc.page_count

            if page_count <= 0:
                raise ValueError("The PDF does not contain any valid pages.")

            return page_count

    except Exception as exc:
        raise ValueError(f"Unable to read the PDF '{pdf_path}': {exc}") from exc



'''
Docling
'''
# Converts PDF (or given page range) to Docling document.
def pdf_to_docling_document(
    pdf_path: Path,
    page_range: tuple[int, int] | None,
    no_ocr: bool = False,
    parallel_threads: int = 1) -> DoclingDocument:

    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,  # Generate images for PictureItem objects.
        images_scale=2.0,              # Render images at higher resolution (default 1.0).
        do_ocr=not no_ocr,             # Disable OCR: use with native PDFs (not scanned).
    )

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=parallel_threads,   # From 1 (default) up to 4 parallel working threads.
        device=AcceleratorDevice.AUTO   # CPU / CUDA / MPS  or AUTO.
    )
    
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
        show_progress=True, # Log on console.
    )

    if page_range is not None:
        conv_result = converter.convert(source=str(pdf_path), page_range=page_range)
    else:
        conv_result = converter.convert(source=str(pdf_path))

    return conv_result.document



'''
Command line args
'''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PDF (or a range of pages) to a complete JSON and TOC using Docling."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Full path of the input PDF file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination folder where the resulting JSON and TOC files will be saved.",
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
        "--no-ocr",
        action="store_true",
        default=False,
        help="Disable OCR if the input PDF is entirely native (not scanned).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel working threads to speed up Docling: default 1, up to 4.",
    )
    return parser.parse_args()

# Validate --pdf, --output, --range (1-based, inclusive) and --threads.
def validate_inputs(
    pdf_raw: str,
    output_raw: str,
    range_values: list[int] | None,
    threads_value: int) -> tuple[Path, Path, tuple[int, int] | None, int, int]:

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

    if threads_value < 1 or threads_value > 4:
        raise ValueError(
            f"Invalid threads number: default is 1, maximum value is 4."
        )

    return pdf_path, output_dir, validated_range, threads_value, total_pages



'''
TOC
'''
def build_json_and_toc()




'''
Main
'''
def main() -> int:
    args = parse_args()
    try:
        pdf_path, output_dir, page_range, threads, total_pages = validate_inputs(
            pdf_raw=args.pdf,
            output_raw=args.output,
            range_values=args.range,
            threads_value=args.threads
        )
        
        script_stem = Path(__file__).stem
        report_path = output_dir / f"{script_stem}_report.txt"

        #with tee_output_to_file(report_path):
        print()
        print("PDF file: ", end="")
        print(pdf_path)
        print(f"Pages: {total_pages}")
        print(f"Range: {page_range[0]} - {page_range[1]}" if page_range is not None else "Range: all")
        print("OCR: disabled" if args.no_ocr else "OCR: enabled")
        print(f"Working threads: {args.threads}")
        print()
        print("Invoking Docling...")
        print()

        doc = pdf_to_docling_document(
            pdf_path,
            page_range,
            args.no_ocr,
            threads)

        if page_range is not None:
            range_suffix = f"_p{page_range[0]}-{page_range[1]}"
        else:
            range_suffix = ""

        output_filename = f"{pdf_path.stem}{range_suffix}"
        docling_output_filename = f"{output_filename}_docling.json"
        json_path = output_dir / docling_output_filename
        json_path.parent.mkdir(parents=True, exist_ok=True)

        doc.save_as_json(
            json_path,
            image_mode=ImageRefMode.REFERENCED,
        )

        # CALL JSON & TOC GENERATION HERE .....

        toc_filename = f"toc{range_suffix}.tsv"
    
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error during conversion with Docling: {exc}", file=sys.stderr)
        return 2

    return 0


'''
Entry point
'''
if __name__ == "__main__":
    sys.exit(main())