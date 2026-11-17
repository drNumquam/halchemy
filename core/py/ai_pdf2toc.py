#!/usr/bin/env python3
"""
Generate ``toc.yaml`` from the hierarchical bookmarks of a PDF document and 
enrich the resulting TOC with AI-generated cross-references between sections

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
            confidence: <float>
        children: [...]

Notes:
    * Page numbers are 1-based physical PDF pages.
    * Granularity is the page: a section spans every page it touches. The start 
      page of a section is also included in the immediately preceding section.
      Adjacent sections may overlap by one page.
    * ``references`` contains AI-generated cross_references between TOC nodes. 
      References are inferred from the section hierarchy and metadata available 
      in the generated TOC structure. A section may contain zero or more references.
    * next always points to the next start page of the next TOC node in document order.
      For the last node, next is -1.
    * When a title carries no printed number (e.g. "Deployment Guidelines"),
      ``id`` falls back to its positional outline number.

Usage:
    python ai_pdf2toc.py --pdf <manual.pdf> --output <out_dir> [--schema <schema.yaml>]

Requirements:
    python -m pip install pymupdf
    python -m pip install pyyaml jsonschema     
    python -m pip install httpx
    python -m pip install openai

Exit codes:
    0  success
    1  the PDF has no usable bookmarks (nothing written)
    2  error (bad input, missing dependency, I/O failure, schema violation)
"""
