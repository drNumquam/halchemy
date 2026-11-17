#!/usr/bin/env python3
"""Generate ``toc.yaml`` from the hierarchical bookmarks of a PDF document.

Output contract (matches the ``schema.yaml`` used by the downstream tooling)::

    source: "<pdf filename>"
    pages: <page count>
    sections:
      - id: "<1.2.3>"             # printed number or positional fallback
        title: "<title>"          # title without the leading section number
        span_start: <int>         # first page, 1-based, inclusive
        span_end: <int>           # last page, 1-based, inclusive
        next: <int>               # next node page, 1-based, inclusive
        references: 
          - id: "<target section id>"
            type: "<semantic|structural|external>"
            confidence: <float>
        children: [...]

Notes:
    * Page numbers are 1-based physical PDF pages; ``span_end`` is inclusive.
    * Granularity is the page: a section spans every page it touches, so
      adjacent spans may overlap by one page. A section ends on the start page
      of the next bookmark of the same or a higher level, unless that bookmark
      begins at the very top of its page (see ``--span-end``), in which case
      the section ends on the previous page.
    * ``references`` contains AI-generated cross_reference between TOC nodes. 
      References are inferred from the section hierarchy and metadata available in the generated TOC structure.
    * When a title carries no printed number (e.g. "Deployment Guidelines"),
      ``id`` falls back to its positional outline number.

Usage:
    python ai_pdf2toc.py --pdf <manual.pdf> -o <out_dir> [--schema <schema.yaml>]
                      [--span-end {auto,inclusive,exclusive}] [--top-ratio 0.15] [-v]

Requirements:
    python -m pip install pymupdf
    python -m pip install pyyaml jsonschema      # only with --schema

Exit codes:
    0  success
    1  the PDF has no usable bookmarks (nothing written)
    2  error (bad input, missing dependency, I/O failure, schema violation)
"""
