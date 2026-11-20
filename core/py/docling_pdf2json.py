import argparse
import logging
import json
import io
import sys
from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


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

    # Manteniamo i riferimenti reali alla console
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with report_path.open("w", encoding="utf-8") as report_file:
        tee_out = Tee(orig_stdout, report_file)
        tee_err = Tee(orig_stderr, report_file)

        # Handler solo per i log di Docling/Python verso il file (lo schermo già riceve stderr tramite tee_err)
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
    """Restituisce il numero totale di pagine del PDF tramite pypdfium2."""
    pdf_doc = None
    try:
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf_doc)
        if page_count <= 0:
            raise ValueError("Il PDF non contiene pagine valide.")
        return page_count
    except Exception as exc:
        raise ValueError(f"Impossibile leggere il PDF '{pdf_path}': {exc}") from exc
    finally:
        if pdf_doc is not None:
            pdf_doc.close()


def validate_inputs(
    pdf_raw: str,
    output_raw: str,
    range_values: list[int] | None,
) -> tuple[Path, Path, tuple[int, int] | None, int]:
    """Valida --pdf, --output e --range (1-based, inclusivo)."""
    pdf_path = Path(pdf_raw).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Il file PDF non esiste: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Il percorso --pdf non punta a un file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Il file specificato non ha estensione .pdf: {pdf_path}")

    total_pages = get_pdf_page_count(pdf_path)

    validated_range: tuple[int, int] | None = None
    if range_values is not None:
        if len(range_values) != 2:
            raise ValueError("--range richiede esattamente due interi: min max (es. --range 1 5).")

        start_page, end_page = range_values

        if start_page < 1:
            raise ValueError(
                f"La pagina iniziale (min={start_page}) deve essere maggiore o uguale a 1."
            )
        if end_page < 1:
            raise ValueError(
                f"La pagina finale (max={end_page}) deve essere maggiore o uguale a 1."
            )
        if start_page > end_page:
            raise ValueError(
                f"Intervallo non valido: min ({start_page}) non può essere maggiore di max ({end_page})."
            )
        if end_page > total_pages:
            raise ValueError(
                f"La pagina finale (max={end_page}) supera il numero totale di pagine del PDF ({total_pages})."
            )

        validated_range = (start_page, end_page)

    output_dir = Path(output_raw).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Il percorso --output esiste già ma non è una cartella: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return pdf_path, output_dir, validated_range, total_pages


def export_pdf_to_json(
    pdf_path: Path,
    output_dir: Path,
    page_range: tuple[int, int] | None,
    no_ocr: bool = False
) -> Path:
    """Converte il PDF (o il range indicato) con Docling e salva il JSON omnicomprensivo."""
#    converter = DocumentConverter()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0  # opzionale (default ~1.0)
    pipeline_options.do_ocr = not no_ocr

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
    #doc_dict = doc.export_to_dict(image_mode=ImageRefMode.REFERENCED)
    #json_path.write_text(
    #    json.dumps(doc_dict, indent=2, ensure_ascii=False),
    #    encoding="utf-8",
    #)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esporta un PDF (o un intervallo di pagine) in JSON completo tramite Docling."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Percorso completo del file PDF di input.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Cartella di destinazione in cui salvare il file JSON risultante.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=None,
        help="Pagina iniziale e finale (1-based, inclusive), es. --range 2 6. Se omesso, elabora tutto il PDF.",
    )
    parser.add_argument(
        "--no_ocr",
        action="store_true",
        default=False,
        help="Disabilita l'OCR se il PDF in input è completamente nativo).",
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
        print(f"Errore di validazione: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Errore durante la conversione con Docling: {exc}", file=sys.stderr)
        return 2

    scope = f"pagine {page_range[0]}-{page_range[1]} su {total_pages}" if page_range else f"tutte le {total_pages} pagine"
    print()
    print(f"JSON exported SUCCESSFULLY ({scope}): {out_file}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
