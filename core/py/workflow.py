import argparse
import asyncio
import os
import re
import sys
import yaml

# --- 1. MODEL CONFIGURATION ---
LLM_PROVIDER = "copilot"  # Set to "qwen" to use the local Qwen model.

QWEN_MODEL_NAME = "qwen/qwen3.5-9b"
QWEN_BATCH_SIZE = 8

COPILOT_MODEL_NAME = "auto"
COPILOT_BATCH_SIZE = 8

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
    BATCH_SIZE = QWEN_BATCH_SIZE
elif LLM_PROVIDER == "copilot":
    MODEL_NAME = COPILOT_MODEL_NAME
    BATCH_SIZE = COPILOT_BATCH_SIZE
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# --- 2. EXTRACT ROOT NODES FROM THE YAML FILE ---
def extract_root_nodes(toc_data: dict) -> list:
    sections = toc_data.get("sections", [])
    root_nodes = []
    for node in sections:
        if isinstance(node, dict) and "id" in node and "title" in node:
            start_index = node.get("start_index")
            end_index = node.get("end_index")

            if start_index is not None and end_index is not None:
                pages_text = f"{start_index} - {end_index}" if start_index != end_index else f"{start_index}"
            elif start_index is not None:
                pages_text = str(start_index)
            else:
                pages_text = "N/A"

            root_nodes.append({
                "id": node["id"],
                "title": node["title"],
                "pages": pages_text
            })
    return root_nodes


# --- 3. LLM CALL AND RESPONSE PARSING ---
def call_qwen_llm(system_prompt: str, node_batch: list) -> str:
    llm_payload = [{"id": node["id"], "title": node["title"]} for node in node_batch]
    nodes_yaml = yaml.dump(llm_payload, allow_unicode=True, sort_keys=False)

    if client is None:
        raise RuntimeError("The Qwen client is not configured.")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please evaluate the following {len(llm_payload)} nodes:\n\n{nodes_yaml}"}
            ],
            temperature=0.1,
            max_tokens=4096,
            reasoning_effort="none",
            stream=False
        )
    except Exception:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please evaluate the following {len(llm_payload)} nodes:\n\n{nodes_yaml}"}
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=False
        )

    first_choice, *_ = response.choices
    content = first_choice.message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


async def call_llm(
    system_prompt: str,
    node_batch: list,
    copilot_session=None,
) -> str:
    if LLM_PROVIDER == "qwen":
        return call_qwen_llm(system_prompt, node_batch)

    if copilot_session is None:
        raise RuntimeError("The GitHub Copilot session has not been initialized.")

    llm_payload = [{"id": node["id"], "title": node["title"]} for node in node_batch]
    nodes_yaml = yaml.dump(llm_payload, allow_unicode=True, sort_keys=False)
    prompt = (
        f"{system_prompt}\n\n"
        f"Please evaluate the following {len(llm_payload)} nodes:\n\n{nodes_yaml}"
    )
    response = await copilot_session.send_and_wait(prompt)
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
    id_header, title_header, pages_header, confidence_header = (
        "Node ID",
        "Title",
        "Pages",
        "Confidence",
    )
    rows = []

    for node_id, score in sorted_scores:
        node_info = original_nodes[node_id]
        rows.append((
            str(node_id),
            node_info["title"],
            node_info["pages"],
            f"{score:.2f}"
        ))

    id_column, title_column, pages_column, confidence_column = zip(*rows)

    id_width = max(len(id_header), *(len(value) for value in id_column), 4)
    title_width = max(len(title_header), *(len(value) for value in title_column), 4)
    pages_width = max(len(pages_header), *(len(value) for value in pages_column), 4)
    confidence_width = max(
        len(confidence_header),
        *(len(value) for value in confidence_column),
        4,
    )

    print(f"| {id_header.ljust(id_width)} | {title_header.ljust(title_width)} | {pages_header.center(pages_width)} | {confidence_header.center(confidence_width)} |")
    print(f"| :{'-' * (id_width - 1)} | :{'-' * (title_width - 1)} | :{'-' * (pages_width - 2)}: | :{'-' * (confidence_width - 2)}: |")

    for row_id, row_title, row_pages, row_confidence in rows:
        print(f"| {row_id.ljust(id_width)} | {row_title.ljust(title_width)} | {row_pages.center(pages_width)} | {row_confidence.center(confidence_width)} |")


# --- 5. BATCH ANALYSIS WITH COMPLETE VALIDATION ---
async def analyze_peripheral_nodes(root_nodes: list, copilot_session=None):
    system_prompt = (
        "You are an expert in embedded architectures, ARM processors, and microcontrollers. "
        "You will be given a small batch of root nodes (id and title) from an ARM Technical Reference Manual.\n\n"
        "TASK:\n"
        "Assign each node an independent confidence score from 0.00 to 1.00 indicating how likely "
        "its title suggests information about ARM PERIPHERALS or peripheral/bus interfaces "
        "(e.g., GIC/interrupt controller, Timer, external buses/APB/Utility bus, LLPP/SPP peripheral ports, "
        "Debug/ETM/PMU, PPU, external registers).\n\n"
        "SCORING SCALE:\n"
        "- 0.85 - 1.00: Dedicated peripheral or peripheral bus/register chapter (e.g., GIC, Timer, External registers, ETM, PMU, Debug).\n"
        "- 0.40 - 0.84: System/bus/power/clock interfaces interacting with peripherals (e.g., PPUs, Memory system ports, Clocks/resets).\n"
        "- 0.00 - 0.15: CPU core internals, instructions, or document metadata (e.g., Contents, Introduction, Programmers' model, SIMD, Revisions).\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY a Markdown table with columns: `| Node ID | Title | Confidence |`\n"
        "- Include ALL provided nodes in the batch without skipping any.\n"
        "- No introductory or concluding text."
    )

    original_nodes = {
        str(node["id"]): {"title": node["title"], "pages": node["pages"]}
        for node in root_nodes
    }
    total_scores = {}

    total_batches = (len(root_nodes) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_start in range(0, len(root_nodes), BATCH_SIZE):
        batch = root_nodes[batch_start:batch_start + BATCH_SIZE]
        batch_number = (batch_start // BATCH_SIZE) + 1
        print(f"  Processing batch {batch_number}/{total_batches} ({len(batch)} nodes)...")

        llm_output = await call_llm(system_prompt, batch, copilot_session)
        batch_scores = extract_scores(llm_output, original_nodes)
        total_scores.update(batch_scores)

    recovery_attempts = 0
    while len(total_scores) < len(root_nodes) and recovery_attempts < 2:
        recovery_attempts += 1
        missing_nodes = [node for node in root_nodes if str(node["id"]) not in total_scores]
        print(f"  Retrying for {len(missing_nodes)} missing nodes...")
        llm_output = await call_llm(system_prompt, missing_nodes, copilot_session)
        total_scores.update(extract_scores(llm_output, original_nodes))

    # Sort by descending confidence, then by ascending node ID to break ties.
    sorted_scores = sorted(
        total_scores.items(),
        key=lambda item: (-item.pop() if False else 0)  # unused placeholder
    )
    sorted_scores = [
        (node_id, score)
        for score, node_id in sorted(
            ((score, int(node_id)) for node_id, score in total_scores.items()),
            key=lambda x: (-x.real if isinstance(x, float) else 0)
        )
    ]
    # Build the final ordering: descending score, then ascending node ID.
    sorted_pairs = sorted(
        ((score, int(node_id), node_id) for node_id, score in total_scores.items()),
        key=lambda pair: (-pair[0], pair[1])
    )
    sorted_scores = [(node_id, score) for score, _, node_id in sorted_pairs]

    print(f"\nAnalysis result ({len(sorted_scores)}/{len(root_nodes)} nodes evaluated):\n")
    print_aligned_table(sorted_scores, original_nodes)
    print()


# --- 6. PARAMETER HANDLING ---
async def main():
    parser = argparse.ArgumentParser(
        description="Classify root nodes from a YAML file by their relevance to ARM peripherals."
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

    root_nodes = extract_root_nodes(toc_data)

    if not root_nodes:
        print(f"Error: No root nodes found under the 'sections' key in YAML file '{args.yaml_path}'.")
        sys.exit(1)

    print(
        f"Extracted {len(root_nodes)} root nodes from '{args.yaml_path}'. "
        f"Analyzing in batches of {BATCH_SIZE}..."
    )
    if LLM_PROVIDER == "copilot":
        from copilot import CopilotClient

        async with CopilotClient() as copilot_client:
            async with await copilot_client.create_session(model=MODEL_NAME) as copilot_session:
                await analyze_peripheral_nodes(root_nodes, copilot_session)
    else:
        await analyze_peripheral_nodes(root_nodes)


if __name__ == "__main__":
    asyncio.run(main())
