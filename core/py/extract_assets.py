#!/usr/bin/env python3

import re
import fitz
#import pymupdf_layout
import yaml
import sys

# ---------------------------------------------------------------------
# CAPTION PATTERNS
# ---------------------------------------------------------------------

FIGURE_RE = re.compile(
    r"(Figure|Fig\.)\s+([A-Za-z0-9.\-]+)\s*(.*)",
    re.IGNORECASE
)

TABLE_RE = re.compile(
    r"(Table)\s+([A-Za-z0-9.\-]+)\s*(.*)",
    re.IGNORECASE
)

FOOTNOTE_RE = re.compile(
    r"^(\d+|\*|\†|\‡)\s+(.+)"
)

VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg, flush=True)

# ---------------------------------------------------------------------
# ID EXTRACTION
# ---------------------------------------------------------------------

def build_asset_id(asset_type, caption):

    caption = " ".join(caption.split())

    if asset_type == "figure":

        m = re.match(
            r"^\s*(?:Figure|Fig\.)\s+([A-Za-z0-9.\-]+)",
            caption,
            re.IGNORECASE
        )

        if m:
            return f"Figure {m.group(1)}"

    elif asset_type == "table":

        m = re.match(
            r"^\s*Table\s+([A-Za-z0-9.\-]+)",
            caption,
            re.IGNORECASE
        )

        if m:
            return f"Table {m.group(1)}"

    elif asset_type == "footnote":

        m = re.match(
            r"^\s*(\d+|\*|†|‡)",
            caption
        )

        if m:
            return f"Footnote {m.group(1)}"

    return None


# ---------------------------------------------------------------------
# CAPTION SEARCH
# ---------------------------------------------------------------------

def find_nearby_caption(page, bbox):

    if not isinstance(bbox, fitz.Rect):
        bbox = fitz.Rect(bbox)

    blocks = page.get_text("blocks")

    best_distance = 1e9
    best_caption = ""

    for block in blocks:

        x0, y0, x1, y1, text, *_ = block

        text = text.strip()

        if not text:
            continue

        if (
            FIGURE_RE.match(text)
            or TABLE_RE.match(text)
        ):

            log(
                f"[CAPTION] Candidate: "
                f"'{text[:120]}'"
            )

            distance = min(
                abs(y0 - bbox.y1),
                abs(y1 - bbox.y0)
            )

            if distance < best_distance:
                best_distance = distance
                best_caption = text

    return best_caption


# ---------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------

def extract_figures(doc):

    log("\n[INFO] Scanning figures...")

    assets = []

    for page_no in range(len(doc)):

        page = doc[page_no]

        log(
            f"[PAGE {page_no+1}] "
            f"Searching images..."
        )

        try:
            images = page.get_image_info(xrefs=True)
        except Exception as e:
            log(f"[WARN] {e}")
            continue

        log(
            f"[PAGE {page_no+1}] "
            f"Found {len(images)} image candidates"
        )

        for img in images:

            bbox = fitz.Rect(img["bbox"])

            caption = find_nearby_caption(
                page,
                bbox
            )

            if not caption:
                continue

            asset_id = build_asset_id(
                "figure",
                caption
            )
            if asset_id is None:
                log(
                    f"[WARN] Cannot derive asset id "
                    f"from caption: {caption}"
                )
                continue

            log(
                f"[FIGURE] page={page_no+1} "
                f"id='{asset_id}'"
            )

            assets.append(
                {
                    "id": asset_id,
                    "type": "figure",
                    "page": page_no + 1,
                    "caption": caption,
                    "confidence": 0.90,
                    "references": []
                }
            )

    return assets


# ---------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------

def extract_tables(doc):

    log("\n[INFO] Scanning tables...")

    assets = []

    for page_no in range(len(doc)):

        page = doc[page_no]

        try:
            tables = page.find_tables()
        except Exception as e:
            log(
                f"[PAGE {page_no+1}] "
                f"Table detection failed: {e}"
            )
            continue

        log(
            f"[PAGE {page_no+1}] "
            f"Found {len(tables.tables)} table candidates"
        )

        for table in tables.tables:

            caption = find_nearby_caption(
                page,
                fitz.Rect(table.bbox)
            )

            if not caption:
                continue

            asset_id = build_asset_id(
                "table",
                caption
            )
            if asset_id is None:
                log(
                    f"[WARN] Cannot derive asset id "
                    f"from caption: {caption}"
                )
                continue

            log(
                f"[TABLE] page={page_no+1} "
                f"id='{asset_id}'"
            )

            assets.append(
                {
                    "id": asset_id,
                    "type": "table",
                    "page": page_no + 1,
                    "caption": caption,
                    "confidence": 0.90,
                    "references": []
                }
            )

    return assets


# ---------------------------------------------------------------------
# FOOTNOTES
# ---------------------------------------------------------------------

def extract_footnotes(doc):

    log("\n[INFO] Scanning footnotes...")

    assets = []

    for page_no in range(len(doc)):

        page = doc[page_no]

        page_height = page.rect.height

        for block in page.get_text("blocks"):

            x0, y0, x1, y1, text, *_ = block

            text = text.strip()

            if not text:
                continue

            if y0 < page_height * 0.75:
                continue

            if not FOOTNOTE_RE.match(text):
                continue

            asset_id = build_asset_id(
                "footnote",
                text
            )
            if asset_id is None:
                log(
                    f"[WARN] Cannot derive asset id "
                    f"from caption: {caption}"
                )
                continue
                
            log(
                f"[FOOTNOTE] "
                f"page={page_no+1} "
                f"id='{asset_id}'"
            )

            assets.append(
                {
                    "id": asset_id,
                    "type": "footnote",
                    "page": page_no + 1,
                    "caption": text,
                    "confidence": 0.80,
                    "references": []
                }
            )

    return assets


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main(pdf_file, output_yaml):

    log(f"[INFO] Opening PDF: {pdf_file}")

    doc = fitz.open(pdf_file)

    log(f"[INFO] Total pages: {len(doc)}")

    assets = []

    assets.extend(extract_figures(doc))
    assets.extend(extract_tables(doc))
    assets.extend(extract_footnotes(doc))

    data = {
        "source": pdf_file,
        "pages": len(doc),
        "assets": assets
    }

    log(
        f"[INFO] Writing output YAML: "
        f"{output_yaml}"
    )

    with open(
        output_yaml,
        "w",
        encoding="utf-8"
    ) as fp:

        yaml.safe_dump(
            data,
            fp,
            allow_unicode=True,
            sort_keys=False
        )

    figures = sum(
        1 for a in assets
        if a["type"] == "figure"
    )

    tables = sum(
        1 for a in assets
        if a["type"] == "table"
    )

    footnotes = sum(
        1 for a in assets
        if a["type"] == "footnote"
    )

    log("")
    log("=====================================")
    log("EXTRACTION SUMMARY")
    log("=====================================")
    log(f"Figures   : {figures}")
    log(f"Tables    : {tables}")
    log(f"Footnotes : {footnotes}")
    log(f"Total     : {len(assets)}")
    log("=====================================")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python extract_assets.py input.pdf assets.yaml"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
