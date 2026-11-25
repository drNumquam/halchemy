import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
import pymupdf
from docling.document_converter import DocumentConverter

LLM_PROVIDER = "qwen"  # "qwen" or "copilot"
QWEN_BASE_URL = "http://localhost:1234/v1"
QWEN_MODEL_NAME = "qwen/qwen3.5-9b"
COPILOT_MODEL_NAME = "auto"

FAKE_HEADER_RE = re.compile(
    r"^(?:[\W_]*\s*)?(?:notes?|notice|warning|caution|tip|important|where|example|table\s+\d+|figure\s+\d+|equation\s+\d+)(?:\s+on\b.*)?\s*[:：]?\s*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are an expert in technical document structure analysis.
You will receive:
1. A sequential list of document section headers. Numbered sections already show their exact `[LEVEL=N]`. Unnumbered sections are marked `[TO_CLASSIFY]`.
2. A reference bookmark outline from the PDF metadata (PyMuPDF).

Assign the true hierarchical depth level (integer >= 1) ONLY to the section IDs marked `[TO_CLASSIFY]`:
- Standalone front/back matter sections (e.g., document title, "Features", "Contents", "List of tables", "IMPORTANT NOTICE...") -> Level 1.
- Unnumbered sub-headings appearing inside a numbered section MUST have a level strictly greater than their immediate parent section (e.g., an unnumbered sub-heading inside a `[LEVEL=3]` section like "6.3.5" MUST be Level 4; inside a `[LEVEL=2]` section like "7.1" MUST be Level 3).

Return ONLY a compact single-line JSON object containing ONLY the `[TO_CLASSIFY]` IDs: {"SEC_0000":1,"SEC_0062":4}"""


def call_qwen(user_prompt: str) -> str:
    import httpx
    from openai import OpenAI

    client = OpenAI(
        base_url=QWEN_BASE_URL,
        api_key="lm-studio",
        timeout=600.0,
        http_client=httpx.Client(trust_env=False, timeout=600.0),
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    try:
        res = client.chat.completions.create(model=QWEN_MODEL_NAME, messages=msgs, temperature=0.1, max_tokens=2048, reasoning_effort="none")
    except Exception:
        res = client.chat.completions.create(model=QWEN_MODEL_NAME, messages=msgs, temperature=0.1, max_tokens=2048)
    first_choice = next(iter(res.choices))
    return re.sub(r"<think>.*?</think>", "", first_choice.message.content or "", flags=re.DOTALL).strip()


async def call_llm(user_prompt: str) -> str:
    if LLM_PROVIDER == "qwen":
        return await asyncio.to_thread(call_qwen, user_prompt)
    from copilot import CopilotClient
    async with CopilotClient() as cc, await cc.create_session(model=COPILOT_MODEL_NAME) as session:
        res = await session.send_and_wait(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
        return re.sub(r"<think>.*?</think>", "", res.data.content or "", flags=re.DOTALL).strip()


def get_pymupdf_toc(pdf_path: Path) -> str:
    toc = pymupdf.open(str(pdf_path)).get_toc()
    if not toc:
        return "No embedded bookmarks found."
    return "\n".join(f"{chr(9) * max(0, lvl - 1)}{re.sub(r'\s+', ' ', title).strip()} (Page {pg})" for lvl, title, pg in toc)


def deterministic_level(title: str) -> int | None:
    # Top-level chapters with or without trailing dot (e.g., "1. Overview" or "1 Introduction") or "Appendix A"
    if re.match(r"^(?:\d+\.?|Appendix\s+[A-Z]\.?)\s+[A-Za-z]", title, re.IGNORECASE):
        return 1
    # Multi-level numbered sections with or without trailing dot (e.g., "3.1 Arm", "3.5.1 Power", "B.1 Must-Read")
    m = re.match(r"^(?:[A-Z]|\d+)((?:\.\d+)+)\.?\s+\S", title)
    if m:
        return 1 + m.group(1).count(".")
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.is_file():
        sys.exit(f"Error: PDF file '{pdf_path}' not found.")

    print(f"Converting '{pdf_path.name}' with Docling...", flush=True)
    doc_dict = DocumentConverter().convert(str(pdf_path)).document.export_to_dict()

    headers = []
    for item in doc_dict.get("texts", []):
        if item.get("label") == "section_header":
            clean_txt = re.sub(r"\s+", " ", item.get("text", "")).strip()
            if FAKE_HEADER_RE.match(clean_txt) or clean_txt.startswith("·"):
                item["label"] = "text"
                item.pop("level", None)
            else:
                headers.append(item)

    if not headers:
        sys.exit("Error: No valid section headers found.")

    flat_lines = []
    unnumbered_count = 0
    for i, h in enumerate(headers):
        sec_id = f"SEC_{i:04d}"
        title = re.sub(r"\s+", " ", h.get("text", "")).strip()
        page = next(iter(h.get("prov", [])), {}).get("page_no", "?")
        det_lvl = deterministic_level(title)
        if det_lvl is not None:
            tag = f" [LEVEL={det_lvl}]"
        else:
            tag = " [TO_CLASSIFY]"
            unnumbered_count += 1
        flat_lines.append(f"[{sec_id}]{tag} {title} (Page {page})")

    levels = {}
    if unnumbered_count > 0:
        prompt = "### DOCLING SECTIONS ###\n" + "\n".join(flat_lines) + f"\n\n### PYMUPDF REFERENCE OUTLINE ###\n{get_pymupdf_toc(pdf_path)}"
        print(f"Classifying {unnumbered_count}/{len(headers)} unnumbered sections via {LLM_PROVIDER}...", flush=True)
        raw = await call_llm(prompt)
        levels = {k: int(v) for k, v in re.findall(r'"(SEC_\d+)"\s*:\s*(\d+)', raw)}

    out_lines = []
    for i, h in enumerate(headers):
        sec_id = f"SEC_{i:04d}"
        title = re.sub(r"\s+", " ", h.get("text", "")).strip()
        det_lvl = deterministic_level(title)
        new_lvl = det_lvl if det_lvl is not None else max(1, levels.get(sec_id, h.get("level", 1)))
        h["level"] = new_lvl
        page = next(iter(h.get("prov", [])), {}).get("page_no", "?")
        out_lines.append(f"[{sec_id}] {chr(9) * (new_lvl - 1)}{title} (Page {page})")

    print("\n".join(out_lines), flush=True)

    out_path = Path(args.output).resolve() if args.output else pdf_path.with_name(f"{pdf_path.stem}_fixed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path.with_suffix(".txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nSaved fixed Docling JSON to '{out_path}' and TXT to '{out_path.with_suffix('.txt')}'.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
