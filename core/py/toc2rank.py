"""
CONTRACT & SPECIFICATION
======================
Script: toc2rank.py
Purpose: Parse a hierarchical Table of Contents (TOC) from a tsv file, score each
         section's relevance (0.00 - 1.00) via an LLM, sort results by descending
         confidence, and export a ranked TSV file.

CLI Syntax:
  python toc_ranker.py --toc <path_to_toc.txt> --prompt <path_to_prompt.txt> --output <path_to_output.txt> [--concurrent <1-4>]

Arguments:
  --toc        (Required) Path to the input tsv file containing TOC lines.
  --prompt     (Required) Path to the system prompt text file (default: prompt.txt).
  --output     (Required) Path to the output ranked text file, or an existing
               directory (default: toc_ranked.txt).
  --concurrent (Optional) Maximum number of concurrent LLM requests, integer from
               1 to 4 (default: 1).

Input Format (--toc):
  Lines matching: <indent>[<ID>] <Title> (Page <N>) [<optional_tags>...]
  Example:        [SEC_0004] 4. Peripherals (Page 15)

Dependencies:
  - Standard library: argparse, asyncio, json, os, re, sys, contextlib
  - Third-party (for LLM_PROVIDER="qwen"): pyyaml (yaml), openai, httpx
  - Third-party (for LLM_PROVIDER="copilot"): pyyaml (yaml), copilot (CopilotClient)

Outputs (saved in the same directory as the resolved TXT output):
  1. toc_ranked.tsv           : Flattened lines "[<id>] <title> — confidence: <0.00-1.00>"
  2. <script_stem>_report.txt : Console output log (overwritten at every run)
"""

import argparse
import asyncio
import io
import os
import re
import sys
from contextlib import AsyncExitStack, contextmanager
from pathlib import Path
import yaml

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
        if not self.file_stream.closed:
            self.file_stream.write(s)
            self.file_stream.flush()
        return len(s)

    def flush(self) -> None:
        self.console_stream.flush()
        if not self.file_stream.closed:
            self.file_stream.flush()


@contextmanager
def tee_output_to_file(report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with report_path.open("w", encoding="utf-8") as report_file:
        sys.stdout = Tee(orig_stdout, report_file)
        sys.stderr = Tee(orig_stderr, report_file)
        try:
            yield report_path
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


# --- 1. MODEL CONFIGURATION ---
LLM_PROVIDER = "qwen"  # "qwen" per endpoint OpenAI-compatibile/locale, "copilot" per GitHub Copilot

QWEN_BASE_URL = "http://localhost:1234/v1"
QWEN_MODEL_NAME = "qwen/qwen3.5-9b"
DEFAULT_PROMPT_FILE = "prompt.txt"
DEFAULT_OUTPUT_FILE = "toc_ranked.tsv"

COPILOT_MODEL_NAME = "auto"
DEFAULT_MAX_CONCURRENT = 1
MAX_ALLOWED_CONCURRENT = 4

client = None


def load_system_prompt(prompt_path: str) -> str:
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file '{prompt_path}' does not exist.")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    if not system_prompt:
        raise ValueError(f"Prompt file '{prompt_path}' is empty.")
    return system_prompt


def init_llm_client():
    global client
    if LLM_PROVIDER == "qwen":
        import httpx
        from openai import OpenAI

        client = OpenAI(
            base_url=QWEN_BASE_URL,
            api_key="lm-studio",
            timeout=600.0,
            http_client=httpx.Client(trust_env=False, timeout=600.0),
        )
        return QWEN_MODEL_NAME
    elif LLM_PROVIDER == "copilot":
        return COPILOT_MODEL_NAME
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# --- 2. EXTRACT HIERARCHICAL NODES FROM TXT ---
LINE_PATTERN = re.compile(
    r"^(?P<indent>\s*)\[(?P<id>[A-Za-z0-9_-]+)\]\s+(?P<body>.+?)\s+\(Page\s+(?P<page>\d+|\?)\)(?:\s+(?P<tags>\[[^\]]+\](?:\s+\[[^\]]+\])*))?\s*$"
)
NUM_PREFIX_PATTERN = re.compile(
    r"^(?P<num>(?:Appendix\s+[A-Z]\.?|\d+(?:\.\d+)*\.?|[A-Z]\.\d+(?:\.\d+)*\.?))\s+(?P<rest>.+)$"
)


def _infer_level(num_str: str | None, indent_len: int, stack: list[dict]) -> int:
    if num_str:
        clean_num = num_str.rstrip(".")
        if clean_num.startswith("Appendix "):
            return 1
        return len(clean_num.split("."))

    if stack:
        return min(stack[-1]["level"] + 1, 4)
    return 1


def extract_toc_nodes_from_txt(txt_content: str) -> list:
    raw_items = []
    stack: list[dict] = []

    for line in txt_content.splitlines():
        if not line.strip():
            continue

        match = LINE_PATTERN.match(line)
        if not match:
            continue

        node_id = match.group("id").strip()
        body = match.group("body").strip()
        raw_page = match.group("page")
        page = int(raw_page) if raw_page.isdigit() else 0
        indent_len = len(match.group("indent").expandtabs(2))

        num_match = NUM_PREFIX_PATTERN.match(body)
        number = num_match.group("num").strip() if num_match else ""
        title = body

        level = _infer_level(number if number else None, indent_len, stack)

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        parent_titles = [item["title"] for item in stack]
        item = {
            "id": node_id,
            "number": number,
            "title": title,
            "level": level,
            "parents": parent_titles,
            "start_page": page,
            "end_page": None,
            "stop_before_id": None,
            "text_end_before": None,
            "references": [],
        }
        raw_items.append(item)
        stack.append(item)

    toc_nodes = []
    for idx, item in enumerate(raw_items):
        next_item = raw_items[idx + 1] if idx + 1 < len(raw_items) else None
        if next_item is not None:
            item["end_page"] = max(item["start_page"], next_item["start_page"])
            item["stop_before_id"] = next_item["id"]
            item["text_end_before"] = next_item["title"]
            pages_text = (
                f"{item['start_page']} - {item['end_page']}"
                if item["start_page"] != item["end_page"]
                else f"{item['start_page']}"
            )
        else:
            pages_text = str(item["start_page"])

        toc_nodes.append({
            "id": item["id"],
            "number": item["number"],
            "title": item["title"],
            "parents": item["parents"],
            "pages": pages_text,
            "stop_before_id": item["stop_before_id"],
            "text_end_before": item["text_end_before"],
            "references": item["references"],
        })

    return toc_nodes


# --- 3. LLM CALL AND RESPONSE PARSING ---
def build_node_prompt(node: dict) -> str:
    payload = {
        "id": node["id"],
        "title": node["title"],
    }
    if node.get("parents"):
        payload["hierarchy_path"] = " > ".join(node["parents"])

    node_yaml = yaml.dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
    )
    return f"Node:\n{node_yaml}"


def call_qwen_llm(system_prompt: str, node: dict, model_name: str) -> str:
    user_prompt = build_node_prompt(node)

    if client is None:
        raise RuntimeError("The Qwen client is not configured.")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=256,
            reasoning_effort="none",
            stream=False,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=256,
            stream=False,
        )

    first_choice, *_ = response.choices
    content = first_choice.message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


async def call_llm(
    system_prompt: str,
    node: dict,
    model_name: str,
    copilot_session=None,
) -> str:
    if LLM_PROVIDER == "qwen":
        return await asyncio.to_thread(call_qwen_llm, system_prompt, node, model_name)

    if copilot_session is None:
        raise RuntimeError("The GitHub Copilot session has not been initialized.")

    user_prompt = build_node_prompt(node)
    response = await copilot_session.send_and_wait(
        f"{system_prompt}\n\n{user_prompt}"
    )
    if response is None or not response.data.content:
        raise RuntimeError("GitHub Copilot returned no content.")

    return re.sub(
        r"<think>.*?</think>",
        "",
        response.data.content,
        flags=re.DOTALL,
    ).strip()


def _resolve_node_id(raw_id: str, original_nodes: dict) -> str | None:
    if raw_id in original_nodes:
        return raw_id
    for known_id in original_nodes:
        if known_id.endswith(f"_{raw_id}") or known_id.lstrip("0") == raw_id.lstrip("0"):
            return known_id
    return None


def extract_scores(llm_content: str, original_nodes: dict) -> dict:
    scores = {}
    for line in llm_content.splitlines():
        match = re.match(
            r"^\|\s*([A-Za-z0-9_-]+)\s*\|.*?\|\s*([0-9]*\.?[0-9]+)\s*\|$",
            line.strip(),
        )
        if match:
            resolved_id = _resolve_node_id(match.group(1).strip(), original_nodes)
            if resolved_id is not None:
                score = max(0.0, min(1.0, float(match.group(2))))
                scores[resolved_id] = round(score, 4)
    return scores


# --- 4. SORTED OUTPUT (DESCENDING BY CONFIDENCE) ---
def build_sorted_sections(sorted_scores: list, original_nodes: dict) -> list[dict]:
    sections = []
    for node_id, score in sorted_scores:
        node_info = original_nodes[node_id]
        text_boundary = (
            f"From this title up to (excluding) {node_info['text_end_before']}"
            if node_info["text_end_before"] is not None
            else "From this title through section end"
        )
        sections.append({
            "id": str(node_id),
            "title": node_info["title"],
            "confidence": round(float(score), 2),
            "hierarchy": node_info.get("parents", []),
            "pages": node_info["pages"],
            "stop_before_id": node_info.get("stop_before_id"),
            "text_boundary": text_boundary,
        })
    return sections


def format_tsv_lines(sections: list[dict]) -> list[str]:
    lines = [
        "id\ttitle\tconfidence\tpages"
    ]
    for sec in sections:
        lines.append(
            f"{sec['id']}\t"
            f"{sec['title']}\t"
            f"{sec['confidence']:.2f}\t"
            f"{sec['pages']}"
        )
    return lines


def print_sorted_sections(sections: list[dict]) -> None:
    if not sections:
        print("No sections scored.")
        return

    print("Sections sorted by descending confidence (0.00 - 1.00):")
    for idx, line in enumerate(format_tsv_lines(sections), start=1):
        print(f"{idx:2d}. {line}")


def save_ranked_sections_as_tsv(output_path: str, sections: list[dict]) -> str:
    if os.path.isdir(output_path):
        target_path = os.path.join(output_path, DEFAULT_OUTPUT_FILE)
    else:
        target_path = output_path

    output_dir = os.path.dirname(target_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lines = format_tsv_lines(sections)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return target_path


def save_sorted_sections_as_json(json_path: str, sections: list[dict]) -> None:
    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)


# --- 5. PER-NODE ANALYSIS WITH COMPLETE VALIDATION ---
async def analyze_peripheral_nodes(
    toc_nodes: list,
    system_prompt: str,
    model_name: str,
    output_path: str = DEFAULT_OUTPUT_FILE,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    copilot_sessions=None,
) -> list[dict]:
    original_nodes = {
        str(node["id"]): {
            "title": node["title"],
            "parents": node.get("parents", []),
            "pages": node["pages"],
            "stop_before_id": node.get("stop_before_id"),
            "text_end_before": node["text_end_before"],
        }
        for node in toc_nodes
    }
    total_scores = {}

    async def score_nodes(nodes: list, attempt=None) -> dict:
        if not nodes:
            return {}

        worker_count = min(max_concurrent, len(nodes))
        worker_scores = {}

        async def worker(worker_number: int) -> None:
            session = (
                copilot_sessions[worker_number]
                if copilot_sessions is not None
                else None
            )
            for node_index in range(worker_number, len(nodes), worker_count):
                node = nodes[node_index]
                node_id = str(node["id"])
                if attempt is None:
                    print(
                        f"  Processing node {node_index + 1}/{len(nodes)} "
                        f"(ID {node_id})..."
                    )
                else:
                    print(
                        f"  Retry {attempt}/2 for node ID {node_id}: "
                        "no valid score was returned."
                    )

                llm_output = await call_llm(system_prompt, node, model_name, session)
                worker_scores.update(
                    extract_scores(
                        llm_output,
                        {node_id: original_nodes[node_id]},
                    )
                )

        await asyncio.gather(
            *(worker(worker_number) for worker_number in range(worker_count))
        )
        return worker_scores

    total_scores.update(await score_nodes(toc_nodes))

    for attempt in range(1, 3):
        missing_nodes = [
            node for node in toc_nodes if str(node["id"]) not in total_scores
        ]
        if not missing_nodes:
            break

        total_scores.update(await score_nodes(missing_nodes, attempt))

    order_index = {str(node["id"]): idx for idx, node in enumerate(toc_nodes)}
    sorted_pairs = sorted(
        ((score, order_index[node_id], node_id) for node_id, score in total_scores.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    sorted_scores = [(node_id, score) for score, _, node_id in sorted_pairs]
    sorted_sections = build_sorted_sections(sorted_scores, original_nodes)

    print(f"\nEvaluated {len(sorted_sections)}/{len(toc_nodes)} sections:\n")
    print_sorted_sections(sorted_sections)
    print(f"\nSaved flattened ranked TOC to '{saved_tsv_path}'.")
    return sorted_sections


# --- 6. PARAMETER HANDLING ---
async def main():
    parser = argparse.ArgumentParser(
        description="Classify TOC nodes from a tsv file and sort them descending by confidence."
    )
    parser.add_argument(
        "--toc",
        dest="toc_path",
        required=True,
        help="Path to the tsv file containing a hierarchical table of contents",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_path",
        required=True,
        default=DEFAULT_PROMPT_FILE,
        help="Path to the prompt.txt file containing the system prompt",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        required=True,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the output text file (default: toc_ranked.txt)",
    )
    parser.add_argument(
        "--concurrent",
        dest="max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum number of concurrent LLM requests (1-{MAX_ALLOWED_CONCURRENT}, default: {DEFAULT_MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    if not (1 <= args.max_concurrent <= MAX_ALLOWED_CONCURRENT):
        print(f"Error: --concurrent must be between 1 and {MAX_ALLOWED_CONCURRENT}.")
        sys.exit(1)

    script_stem = Path(__file__).stem
    out_target = Path(args.output_path).expanduser().resolve()
    report_dir = out_target if out_target.is_dir() else out_target.parent
    report_path = report_dir / f"{script_stem}_report.txt"

    with tee_output_to_file(report_path):
        try:
            system_prompt = load_system_prompt(args.prompt_path)
            model_name = init_llm_client()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        if not os.path.isfile(args.toc_path):
            print(f"Error: TOC file '{args.toc_path}' does not exist.")
            sys.exit(1)

        with open(args.toc_path, "r", encoding="utf-8") as f:
            txt_content = f.read()

        toc_nodes = extract_toc_nodes_from_txt(txt_content)

        if not toc_nodes:
            print(f"Error: No TOC nodes found in '{args.toc_path}'.")
            sys.exit(1)

        print(
            f"Extracted {len(toc_nodes)} nodes from '{args.toc_path}'. "
            f"Sending one request per node with up to {args.max_concurrent} concurrent workers..."
        )
        if LLM_PROVIDER == "copilot":
            from copilot import CopilotClient

            async with CopilotClient() as copilot_client:
                async with AsyncExitStack() as session_stack:
                    session_count = min(args.max_concurrent, len(toc_nodes))
                    copilot_sessions = [
                        await session_stack.enter_async_context(
                            await copilot_client.create_session(model=model_name)
                        )
                        for _ in range(session_count)
                    ]
                    await analyze_peripheral_nodes(
                        toc_nodes,
                        system_prompt,
                        model_name,
                        output_path=args.output_path,
                        max_concurrent=args.max_concurrent,
                        copilot_sessions=copilot_sessions,
                    )
        else:
            await analyze_peripheral_nodes(
                toc_nodes,
                system_prompt,
                model_name,
                output_path=args.output_path,
                max_concurrent=args.max_concurrent,
            )



if __name__ == "__main__":
    asyncio.run(main())
