import argparse
import asyncio
import os
import re
import json
import sys
from contextlib import AsyncExitStack
import yaml

# --- 1. MODEL CONFIGURATION ---
LLM_PROVIDER = "copilot"  # Set to "qwen" to use the local Qwen model.

QWEN_MODEL_NAME = "qwen/qwen3.5-9b"

COPILOT_MODEL_NAME = "auto"
MAX_CONCURRENT_REQUESTS = 4

client = None
if LLM_PROVIDER == "qwen":
    import httpx2 as httpx
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        timeout=600.0,
        http_client=httpx.Client(trust_env=False, timeout=600.0)
    )
    MODEL_NAME = QWEN_MODEL_NAME
elif LLM_PROVIDER == "copilot":
    MODEL_NAME = COPILOT_MODEL_NAME
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# --- 2. EXTRACT ALL NODES FROM THE YAML FILE ---
def extract_toc_nodes(toc_data: dict) -> list:
    sections = toc_data.get("sections", [])
    toc_nodes = []

    def visit(nodes: list) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue

            children = node.get("children", [])
            if "id" in node and "title" in node:
                start_index = node.get("span_start")
                end_index = node.get("span_end")
                text_end_before = None

                if (
                    isinstance(children, list)
                    and children
                    and isinstance(children[0], dict)
                    and children[0].get("span_start") is not None
                ):
                    end_index = children[0]["span_start"]
                    text_end_before = children[0].get("title")

                if start_index is not None and end_index is not None:
                    pages_text = (
                        f"{start_index} - {end_index}"
                        if start_index != end_index
                        else f"{start_index}"
                    )
                elif start_index is not None:
                    pages_text = str(start_index)
                else:
                    pages_text = "N/A"

                toc_nodes.append({
                    "id": node["id"],
                    "number": node.get("number", ""),
                    "title": node["title"],
                    "pages": pages_text,
                    "text_end_before": text_end_before,
                    "references": node.get("references", []),
                })

            if isinstance(children, list):
                visit(children)

    if isinstance(sections, list):
        visit(sections)
    return toc_nodes


# --- 3. LLM CALL AND RESPONSE PARSING ---
def build_node_prompt(node: dict) -> str:
    node_yaml = yaml.dump(
        {"id": node["id"], "title": node["title"]},
        allow_unicode=True,
        sort_keys=False,
    )
    return (
        "Evaluate this single TOC node for how likely its title is to indicate "
        "information about ARM peripherals. Score only this node.\n\n"
        f"Node:\n{node_yaml}"
    )


def call_qwen_llm(system_prompt: str, node: dict) -> str:
    user_prompt = build_node_prompt(node)

    if client is None:
        raise RuntimeError("The Qwen client is not configured.")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=256,
            reasoning_effort="none",
            stream=False
        )
    except Exception:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=256,
            stream=False
        )

    first_choice, *_ = response.choices
    content = first_choice.message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


async def call_llm(
    system_prompt: str,
    node: dict,
    copilot_session=None,
) -> str:
    if LLM_PROVIDER == "qwen":
        return await asyncio.to_thread(call_qwen_llm, system_prompt, node)

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


def extract_scores(llm_content: str, original_nodes: dict) -> dict:
    scores = {}
    for line in llm_content.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|.*?\|\s*([0-9]*\.?[0-9]+)\s*\|$", line.strip())
        if match:
            node_id, score = match.group(1), float(match.group(2))
            if node_id in original_nodes:
                scores[node_id] = score
    return scores


# --- 4. VERTICALLY ALIGNED TABLE FORMATTING ---
def print_aligned_table(sorted_scores: list, original_nodes: dict):
    id_header, title_header, pages_header, boundary_header, confidence_header = (
        "Node ID",
        "Title",
        "Pages",
        "Text boundary",
        "Confidence",
    )
    rows = []

    for node_id, score in sorted_scores:
        node_info = original_nodes[node_id]
        text_boundary = (
            f"From this title up to (excluding) {node_info['text_end_before']}"
            if node_info["text_end_before"] is not None
            else "From this title through section end"
        )
        rows.append((
            str(node_id),
            node_info["title"],
            node_info["pages"],
            text_boundary,
            f"{score:.2f}"
        ))

    id_column, title_column, pages_column, boundary_column, confidence_column = zip(*rows)

    id_width = max(len(id_header), *(len(value) for value in id_column), 4)
    title_width = max(len(title_header), *(len(value) for value in title_column), 4)
    pages_width = max(len(pages_header), *(len(value) for value in pages_column), 4)
    boundary_width = max(
        len(boundary_header),
        *(len(value) for value in boundary_column),
        4,
    )
    confidence_width = max(
        len(confidence_header),
        *(len(value) for value in confidence_column),
        4,
    )

    print(f"| {id_header.ljust(id_width)} | {title_header.ljust(title_width)} | {pages_header.center(pages_width)} | {boundary_header.ljust(boundary_width)} | {confidence_header.center(confidence_width)} |")
    print(f"| :{'-' * (id_width - 1)} | :{'-' * (title_width - 1)} | :{'-' * (pages_width - 2)}: | :{'-' * (boundary_width - 1)} | :{'-' * (confidence_width - 2)}: |")

    for row_id, row_title, row_pages, row_boundary, row_confidence in rows:
        print(f"| {row_id.ljust(id_width)} | {row_title.ljust(title_width)} | {row_pages.center(pages_width)} | {row_boundary.ljust(boundary_width)} | {row_confidence.center(confidence_width)} |")


# EXPORT TABLE TO JSON FILE
def save_result_table_as_json(
    json_path: str,
    sorted_scores: list,
    original_nodes: dict,
) -> None:
    """
    Export to JSON file the same information displayed in the final table.
    """

    rows = []

    for node_id, score in sorted_scores:
        node_info = original_nodes[node_id]

        text_boundary = (
            f"From this title up to (excluding) {node_info['text_end_before']}"
            if node_info["text_end_before"] is not None
            else "From this title through section end"
        )

        rows.append({
            "node_id": node_id,
            "title": node_info["title"],
            "pages": node_info["pages"],
            "text_boundary": text_boundary,
            "confidence": round(score, 2)
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# --- 5. PER-NODE ANALYSIS WITH COMPLETE VALIDATION ---
async def analyze_peripheral_nodes(toc_nodes: list, copilot_sessions=None):
    system_prompt = (
        "You are an expert in embedded architectures, ARM processors, and microcontrollers. "
        "You will be given exactly one TOC node (id and title) from an ARM Technical Reference Manual per request.\n\n"
        "TASK:\n"
        "Assign this node an independent confidence score from 0.00 to 1.00 indicating how likely "
        "its title suggests information about ARM PERIPHERALS or peripheral/bus interfaces "
        "(e.g., GIC/interrupt controller, Timer, external buses/APB/Utility bus, LLPP/SPP peripheral ports, "
        "Debug/ETM/PMU, PPU, external registers).\n\n"
        "SCORING SCALE:\n"
        "- 0.85 - 1.00: Dedicated peripheral or peripheral bus/register chapter (e.g., GIC, Timer, External registers, ETM, PMU, Debug).\n"
        "- 0.40 - 0.84: System/bus/power/clock interfaces interacting with peripherals (e.g., PPUs, Memory system ports, Clocks/resets).\n"
        "- 0.00 - 0.15: CPU core internals, instructions, or document metadata (e.g., Contents, Introduction, Programmers' model, SIMD, Revisions).\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY a Markdown table with columns: `| Node ID | Title | Confidence |`\n"
        "- Return exactly one data row for the provided node; preserve its ID and title exactly.\n"
        "- No introductory or concluding text."
    )

    original_nodes = {
        str(node["id"]): {
            "title": node["title"],
            "pages": node["pages"],
            "text_end_before": node["text_end_before"],
        }
        for node in toc_nodes
    }
    total_scores = {}

    async def score_nodes(nodes: list, attempt=None) -> dict:
        if not nodes:
            return {}

        worker_count = min(MAX_CONCURRENT_REQUESTS, len(nodes))
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

                llm_output = await call_llm(system_prompt, node, session)
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

    # Sort by descending confidence, then by ascending node ID to break ties.
    sorted_pairs = sorted(
        (
            (score, int(node_id), node_id)
            for node_id, score in total_scores.items()
        ),
        key=lambda pair: (-pair[0], pair[1])
    )
    sorted_scores = [(node_id, score) for score, _, node_id in sorted_pairs]

    print(
        f"\nNode results ({len(sorted_scores)}/{len(toc_nodes)} evaluated):\n"
    )
    print_aligned_table(sorted_scores, original_nodes)
    print()
    save_result_table_as_json(
    "peripheral_nodes.json",
    sorted_scores,
    original_nodes
    )


# --- 6. PARAMETER HANDLING ---
async def main():
    parser = argparse.ArgumentParser(
        description="Classify TOC nodes from a YAML file by their relevance to ARM peripherals."
    )
    parser.add_argument(
        "--yaml",
        dest="yaml_path",
        required=True,
        help="Path to the toc.yaml file containing the table of contents",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_path):
        print(f"Error: YAML file '{args.yaml_path}' does not exist.")
        sys.exit(1)

    with open(args.yaml_path, "r", encoding="utf-8") as f:
        toc_data = yaml.safe_load(f)

    toc_nodes = extract_toc_nodes(toc_data)

    if not toc_nodes:
        print(f"Error: No TOC nodes found under the 'sections' key in YAML file '{args.yaml_path}'.")
        sys.exit(1)

    print(
        f"Extracted {len(toc_nodes)} nodes from '{args.yaml_path}'. "
        f"Sending one request per node with up to {MAX_CONCURRENT_REQUESTS} concurrent workers..."
    )
    if LLM_PROVIDER == "copilot":
        from copilot import CopilotClient

        async with CopilotClient() as copilot_client:
            async with AsyncExitStack() as session_stack:
                session_count = min(MAX_CONCURRENT_REQUESTS, len(toc_nodes))
                copilot_sessions = [
                    await session_stack.enter_async_context(
                        await copilot_client.create_session(model=MODEL_NAME)
                    )
                    for _ in range(session_count)
                ]
                await analyze_peripheral_nodes(toc_nodes, copilot_sessions)
    else:
        await analyze_peripheral_nodes(toc_nodes)


if __name__ == "__main__":
    asyncio.run(main())
