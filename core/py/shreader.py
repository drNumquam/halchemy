import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SECTION_NAME_PATTERN = re.compile(
    r"^(?:"
    r"appendix(?=\s|$|:|\)|\]|–|—|-)"
    r"|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*(?=\s|$|:|\)|\]|–|—|-)"
    r")",
    re.IGNORECASE,
)

NOTE_PREFIX_PATTERN = re.compile(r"^\s*(?:notes?|notices?)", re.IGNORECASE)


def is_note_line(text: str) -> bool:
    return NOTE_PREFIX_PATTERN.match(text.lstrip()) is not None


def is_valid_section_name(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return SECTION_NAME_PATTERN.match(text.lstrip()) is not None


def starts_with_unicode(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    return ord(stripped[0]) > 127


def strip_unicode_chars(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) <= 127)


def format_output_line(text: str) -> str:
    cleaned = text.lstrip()
    if starts_with_unicode(cleaned):
        cleaned = strip_unicode_chars(cleaned).lstrip()
    return cleaned


def get_level_indent(level: Any) -> str:
    try:
        level_num = int(level)
    except (TypeError, ValueError):
        return ""
    return "\t" * max(0, level_num - 1)


def extract_section_headers(data: Any) -> list[tuple[str, Any, bool]]:
    results: list[tuple[str, Any, bool]] = []

    if isinstance(data, dict):
        if data.get("label") == "section_header" and "text" in data:
            val = data["text"]
            level = data.get("level", "")
            if isinstance(val, str):
                cleaned = val.lstrip()
                if cleaned and not is_note_line(cleaned):
                    results.append((cleaned, level, is_valid_section_name(cleaned)))

        for value in data.values():
            results.extend(extract_section_headers(value))

    elif isinstance(data, list):
        for item in data:
            results.extend(extract_section_headers(item))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Estrae il campo "text" dagli oggetti con "label": "section_header" da un file JSON.'
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Percorso del file JSON di input",
    )

    args = parser.parse_args()

    if not args.json_file.is_file():
        print(f"Errore: il file '{args.json_file}' non esiste.", file=sys.stderr)
        sys.exit(1)

    try:
        with args.json_file.open("r", encoding="utf-8") as f:
            content = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Errore nel parsing del JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    seq_num = 0
    for text, level, _is_section in extract_section_headers(content):
        if is_note_line(text):
            continue
        formatted = format_output_line(text)
        if not formatted or is_note_line(formatted):
            continue
        indent = get_level_indent(level)
        print(f"[SEC_{seq_num:04d}]\t{indent}{formatted}\t{level}")
        seq_num += 1


if __name__ == "__main__":
    main()