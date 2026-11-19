import gc
import json
import re
from pathlib import Path
import sys
import pymupdf  # oppure import fitz
import time
#import torch # only for MAC

def log(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling_core.types.doc import (
    SectionHeaderItem,
    TextItem,
    TableItem,
    PictureItem,
)

INPUT_PDF = "arm_cortex_r82_processor_trm_102670_06_en.pdf"
TOTAL_PAGES = 2000
BATCH_SIZE = 100

OUT_DIR = Path("./output_manuale")
SEC_DIR = OUT_DIR / "sections"
TBL_DIR = OUT_DIR / "tables"
IMG_DIR = OUT_DIR / "images"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

for d in (SEC_DIR, TBL_DIR, IMG_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

def get_pdf_page_count(pdf_path: str) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()

def create_converter() -> DocumentConverter:
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


def slugify(text: str, fallback: str) -> str:
    clean = re.sub(
        r"[^a-zA-Z0-9\._-]+",
        "_",
        text.strip().lower()
    ).strip("_")
    return clean[:60] if clean else fallback


def process_in_batches(pdf_path: str):

    converter = create_converter()

    log("=== DOCLING ASSET EXTRACTOR ===")
    log(f"PDF={pdf_path}")
    log(f"BATCH_SIZE={BATCH_SIZE}")

    total_pages = get_pdf_page_count(pdf_path)

    print(f"PDF: {pdf_path}")
    print(f"Totale pagine: {total_pages}")

    master_index = {
        "manual_name": pdf_path,
        "sections": [],
        "tables": [],
        "pictures": [],
        "footnotes": []
    }

    current_sec = {
        "id": "sec_0_intro",
        "toc_number": "0",
        "title": "Front Matter & Introduction",
        "page_start": 1,
        "buffer": []
    }

    sec_counter = 0
    tbl_counter = 0
    pic_counter = 0
    fn_counter = 0

    def flush_section(sec_obj: dict):

        if (
            not sec_obj["buffer"]
            and sec_obj["id"] == "sec_0_intro"
        ):
            return

        md_path = SEC_DIR / f"{sec_obj['id']}.md"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {sec_obj['title']}\n\n")
            f.write(" ".join(sec_obj["buffer"]))

        master_index["sections"].append({
            "id": sec_obj["id"],
            "toc_number": sec_obj["toc_number"],
            "title": sec_obj["title"],
            "page_start": sec_obj["page_start"],
            "content_path": str(
                md_path.relative_to(OUT_DIR)
            )
        })

    for start_p in range(
        1,
        total_pages  + 1,
        BATCH_SIZE
    ):

        end_p = min(
            start_p + BATCH_SIZE - 1,
            total_pages 
        )

        ckpt_file = (
            CHECKPOINT_DIR /
            f"batch_{start_p}_{end_p}.json"
        )

        print(
            f"[Fase 1] Elaborazione pagine "
            f"{start_p} - {end_p}..."
        )

        batch_start = time.time()

        log(
            f"[BATCH START] "
            f"pages={start_p}-{end_p}"
        )

        try:
            result = converter.convert(
                pdf_path,
                page_range=(start_p, end_p)
            )
        except Exception as ex:
            log(
                f"[ERROR] Docling convert "
                f"pages={start_p}-{end_p}"
            )
            log(repr(ex))
            raise

        elapsed = time.time() - batch_start

        print(
            f"[BATCH] Conversione completata "
            f"in {elapsed:.1f}s"
        )

        dl_doc = result.document

        batch_sections = 0
        batch_tables = 0
        batch_figures = 0
        batch_footnotes = 0

        for item, _ in dl_doc.iterate_items():

            page_no = (
                item.prov[0].page_no
                if item.prov
                else start_p
            )

            bbox = (
                list(item.prov[0].bbox.as_tuple())
                if item.prov
                else []
            )

            if isinstance(item, SectionHeaderItem):

                batch_sections += 1

                flush_section(current_sec)

                sec_counter += 1

                heading = item.text.strip()

                m = re.match(
                    r"^(?:Chapter|Section|Appendix)?\s*"
                    r"([A-Z0-9]+(?:\.[0-9]+)*)",
                    heading,
                    re.I
                )

                toc_num = (
                    m.group(1)
                    if m
                    else str(sec_counter)
                )

                sec_id = (
                    f"sec_{slugify(toc_num, str(sec_counter))}"
                )

                current_sec = {
                    "id": sec_id,
                    "toc_number": toc_num,
                    "title": heading,
                    "page_start": page_no,
                    "buffer": []
                }

            elif isinstance(item, TextItem):

                txt = item.text.strip()

                if not txt:
                    continue

                if item.label == "footnote":
                    
                    batch_footnotes += 1
                    fn_counter += 1

                    m_fn = re.match(
                        r"^\[?(\d+)\]?[\.\)\s]",
                        txt
                    )

                    fn_num = (
                        m_fn.group(1)
                        if m_fn
                        else str(fn_counter)
                    )

                    master_index["footnotes"].append({
                        "id": (
                            f"fn_{fn_num}_p{page_no}"
                        ),
                        "footnote_number": fn_num,
                        "page": page_no,
                        "parent_section":
                            current_sec["id"],
                        "text": txt
                    })

                    current_sec["buffer"].append(
                        f"> **[Footnote {fn_num}]**: {txt}"
                    )

                elif item.label != "caption":

                    current_sec["buffer"].append(txt)

            elif isinstance(item, TableItem):

                batch_tables += 1
                tbl_counter += 1

                cap_text = (
                    item.caption_text(dl_doc).strip()
                )

                m_tbl = re.search(
                    r"(?:Table|Tbl\.?)\s+"
                    r"([A-Z0-9]+(?:[\.-][0-9]+)*)",
                    cap_text,
                    re.I
                )

                raw_num = (
                    m_tbl.group(1).replace("-", ".")
                    if m_tbl
                    else f"p{page_no}_{tbl_counter}"
                )

                tbl_id = (
                    f"tbl_{slugify(raw_num, str(tbl_counter))}"
                )

                md_file = TBL_DIR / f"{tbl_id}.md"
                json_file = TBL_DIR / f"{tbl_id}.json"

                with open(
                    md_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    if cap_text:
                        f.write(f"**{cap_text}**\n\n")

                    f.write(
                        item.export_to_markdown(dl_doc)
                    )

                with open(
                    json_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        item.data.model_dump(),
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                master_index["tables"].append({
                    "id": tbl_id,
                    "label_number": raw_num,
                    "is_numbered": bool(m_tbl),
                    "page": page_no,
                    "bbox": bbox,
                    "parent_section":
                        current_sec["id"],
                    "caption": cap_text,
                    "md_path": str(
                        md_file.relative_to(OUT_DIR)
                    ),
                    "json_path": str(
                        json_file.relative_to(OUT_DIR)
                    )
                })

                current_sec["buffer"].append(
                    f"[TABLE REF: {tbl_id} -> "
                    f"{md_file.name}]"
                )

            elif isinstance(item, PictureItem):

                batch_figures += 1
                pic_counter += 1

                cap_text = (
                    item.caption_text(dl_doc).strip()
                )

                m_fig = re.search(
                    r"(?:Figure|Fig\.?)\s+"
                    r"([A-Z0-9]+(?:[\.-][0-9]+)*)",
                    cap_text,
                    re.I
                )

                raw_num = (
                    m_fig.group(1).replace("-", ".")
                    if m_fig
                    else f"p{page_no}_{pic_counter}"
                )

                fig_id = (
                    f"fig_{slugify(raw_num, str(pic_counter))}"
                )

                img_file = IMG_DIR / f"{fig_id}.png"

                pil_img = item.get_image(dl_doc)

                if pil_img is not None:
                    pil_img.save(
                        img_file,
                        format="PNG"
                    )

                master_index["pictures"].append({
                    "id": fig_id,
                    "label_number": raw_num,
                    "is_numbered": bool(m_fig),
                    "page": page_no,
                    "bbox": bbox,
                    "parent_section":
                        current_sec["id"],
                    "caption": cap_text,
                    "image_path": (
                        str(
                            img_file.relative_to(
                                OUT_DIR
                            )
                        )
                        if pil_img
                        else None
                    )
                })

                current_sec["buffer"].append(
                    f"[FIGURE REF: {fig_id} -> "
                    f"{img_file.name}]"
                )

        print(
            f"[BATCH SUMMARY] "
            f"pages={start_p}-{end_p} "
            f"sections={batch_sections} "
            f"tables={batch_tables} "
            f"figures={batch_figures} "
            f"footnotes={batch_footnotes}"
        )

        with open(
            ckpt_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "last_page": end_p,
                    "sections_count":
                        len(master_index["sections"])
                },
                f
            )

        print(
            f"[BATCH SUMMARY] "
            f"pages={start_p}-{end_p} "
            f"sections={len(master_index['sections'])} "
            f"tables={len(master_index['tables'])} "
            f"figures={len(master_index['pictures'])} "
            f"footnotes={len(master_index['footnotes'])}"
        )

        del result, dl_doc

        gc.collect()

        #if torch.backends.mps.is_available():
        #    torch.mps.empty_cache()

    flush_section(current_sec)

    with open(
        OUT_DIR / "manual_index_raw.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            master_index,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("================================")
    print("FINAL SUMMARY")
    print("================================")
    print(f"Sections : {sec_counter}")
    print(f"Tables   : {tbl_counter}")
    print(f"Figures  : {pic_counter}")
    print(f"Footnotes: {fn_counter}")
    print("================================")
    print("Fase 1 completata!")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <input.pdf>"
        )
        sys.exit(1)

    process_in_batches(sys.argv[1])
