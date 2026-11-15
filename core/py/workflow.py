import argparse
import os
import re
import sys
import yaml
import httpx2 as httpx
from openai import OpenAI

# --- 1. CONFIGURAZIONE CLIENT LM STUDIO ---
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    timeout=600.0,
    http_client=httpx.Client(trust_env=False, timeout=600.0)
)
MODEL_NAME = "qwen/qwen3.5-9b"
BATCH_SIZE = 8


# --- 2. ESTRAZIONE DEI NODI ROOT DAL FILE YAML ---
def estrai_nodi_root(toc_data: dict) -> list:
    sections = toc_data.get("sections", [])
    nodi_root = []
    for nodo in sections:
        if isinstance(nodo, dict) and "id" in nodo and "title" in nodo:
            start_idx = nodo.get("start_index")
            end_idx = nodo.get("end_index")

            if start_idx is not None and end_idx is not None:
                pagine_str = f"{start_idx} - {end_idx}" if start_idx != end_idx else f"{start_idx}"
            elif start_idx is not None:
                pagine_str = str(start_idx)
            else:
                pagine_str = "N/A"

            nodi_root.append({
                "id": nodo["id"],
                "title": nodo["title"],
                "pages": pagine_str
            })
    return nodi_root


# --- 3. CHIAMATA LLM E PARSING ---
def chiama_llm(system_prompt: str, nodi_batch: list) -> str:
    payload_llm = [{"id": n["id"], "title": n["title"]} for n in nodi_batch]
    nodi_yaml_str = yaml.dump(payload_llm, allow_unicode=True, sort_keys=False)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Evaluate these {len(payload_llm)} nodes:\n\n{nodi_yaml_str}"}
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
                {"role": "user", "content": f"Evaluate these {len(payload_llm)} nodes:\n\n{nodi_yaml_str}"}
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=False
        )

    first_choice, *_ = response.choices
    contenuto = first_choice.message.content or ""
    return re.sub(r"<think>.*?</think>", "", contenuto, flags=re.DOTALL).strip()


def estrai_punteggi(contenuto_llm: str, mappa_originali: dict) -> dict:
    punteggi = {}
    for riga in contenuto_llm.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|.*?\|\s*([0-9]*\.?[0-9]+)\s*\|$", riga.strip())
        if match:
            node_id, score = match.group(1), float(match.group(2))
            if node_id in mappa_originali:
                punteggi[node_id] = score
    return punteggi


# --- 4. FORMATTAZIONE TABELLA ALLINEATA VERTICALMENTE ---
def stampa_tabella_allineata(ordinati: list, mappa_originali: dict):
    h_id, h_title, h_pages, h_conf = "Node ID", "Title", "Pages", "Confidence"
    righe = []

    for node_id, score in ordinati:
        info = mappa_originali[node_id]
        righe.append((
            str(node_id),
            info["title"],
            info["pages"],
            f"{score:.2f}"
        ))

    col_id, col_title, col_pages, col_conf = zip(*righe)

    w_id = max(len(h_id), *(len(x) for x in col_id), 4)
    w_title = max(len(h_title), *(len(x) for x in col_title), 4)
    w_pages = max(len(h_pages), *(len(x) for x in col_pages), 4)
    w_conf = max(len(h_conf), *(len(x) for x in col_conf), 4)

    print(f"| {h_id.ljust(w_id)} | {h_title.ljust(w_title)} | {h_pages.center(w_pages)} | {h_conf.center(w_conf)} |")
    print(f"| :{'-' * (w_id - 1)} | :{'-' * (w_title - 1)} | :{'-' * (w_pages - 2)}: | :{'-' * (w_conf - 2)}: |")

    for r_id, r_title, r_pages, r_conf in righe:
        print(f"| {r_id.ljust(w_id)} | {r_title.ljust(w_title)} | {r_pages.center(w_pages)} | {r_conf.center(w_conf)} |")


# --- 5. ANALISI A BATCH CON VERIFICA COMPLETA ---
def analizza_nodi_periferiche(nodi_root: list):
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

    mappa_originali = {
        str(n["id"]): {"title": n["title"], "pages": n["pages"]}
        for n in nodi_root
    }
    punteggi_totali = {}

    totale_batch = (len(nodi_root) + BATCH_SIZE - 1) // BATCH_SIZE
    for idx in range(0, len(nodi_root), BATCH_SIZE):
        batch = nodi_root[idx:idx + BATCH_SIZE]
        num_batch = (idx // BATCH_SIZE) + 1
        print(f"  Elaborazione batch {num_batch}/{totale_batch} ({len(batch)} nodi)...")

        output_llm = chiama_llm(system_prompt, batch)
        punteggi_batch = estrai_punteggi(output_llm, mappa_originali)
        punteggi_totali.update(punteggi_batch)

    tentativi_recupero = 0
    while len(punteggi_totali) < len(nodi_root) and tentativi_recupero < 2:
        tentativi_recupero += 1
        mancanti = [n for n in nodi_root if str(n["id"]) not in punteggi_totali]
        print(f"  Recupero mirato per {len(mancanti)} nodi mancanti...")
        output_llm = chiama_llm(system_prompt, mancanti)
        punteggi_totali.update(estrai_punteggi(output_llm, mappa_originali))

    # Ordina prima per Confidence (decrescente) e a parità di punteggio per Node ID (crescente)
    ordinati = sorted(
        punteggi_totali.items(),
        key=lambda item: (-item.pop() if False else 0)  # placeholder ignorato
    )
    ordinati = [
        (nid, score)
        for score, nid in sorted(
            ((score, int(nid)) for nid, score in punteggi_totali.items()),
            key=lambda x: (-x.real if isinstance(x, float) else 0)
        )
    ]
    # Costruzione ordinamento pulito senza indici: (score decrescente, id crescente)
    coppie_ordinate = sorted(
        ((score, int(nid), nid) for nid, score in punteggi_totali.items()),
        key=lambda t: (-t[0], t[1])
    )
    ordinati = [(nid, score) for score, _, nid in coppie_ordinate]

    print(f"\nRisultato analisi ({len(ordinati)}/{len(nodi_root)} nodi valutati):\n")
    stampa_tabella_allineata(ordinati, mappa_originali)
    print()


# --- 6. GESTIONE PARAMETRI ---
def main():
    parser = argparse.ArgumentParser(
        description="Classifica i nodi root di un file YAML in base alla pertinenza con le periferiche ARM."
    )
    parser.add_argument("yaml", help="Percorso del file toc.yaml contenente l'indice")
    args = parser.parse_args()

    if not os.path.isfile(args.yaml):
        print(f"Errore: Il file YAML '{args.yaml}' non esiste.")
        sys.exit(1)

    with open(args.yaml, "r", encoding="utf-8") as f:
        toc_data = yaml.safe_load(f)

    nodi_root = estrai_nodi_root(toc_data)

    if not nodi_root:
        print("Errore: Nessun nodo root trovato sotto la chiave 'sections' nel file YAML.")
        sys.exit(1)

    print(f"Estratti {len(nodi_root)} nodi root da '{args.yaml}'. Analisi in corso a batch di {BATCH_SIZE}...")
    analizza_nodi_periferiche(nodi_root)


if __name__ == "__main__":
    main()
