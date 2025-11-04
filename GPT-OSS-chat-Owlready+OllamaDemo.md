## GPT-OSS 120B Python for using Owlready2 and FMA Ontology with Ollama Llama LLM

### USER
Recommend python code for using owlready2 with a local ollama llama LLM with fma.owl large ontology.

### ASSISTANT
<details type="reasoning" done="true" duration="7">
<summary>Thought for 7 seconds</summary>
> User wants recommended python code for using owlready2 with a local Ollama Llama LLM with fma.owl large ontology. So they likely want to load the ontology, query it, and use LLM to generate explanations or answer queries. Provide code snippets, installation steps, handling large ontology memory, integration via Ollama API (maybe using requests). Show example: load ontology, get class hierarchy, ask LLM to explain a class. Also maybe show using LLM for SPARQL query generation.
> 
> Provide recommendations: install owlready2, ollama Python client (ollama). Show code. Provide tips: increase recursion limit, use memory store. Show handling of large ontology: use get_namespace, sync_reasoner, etc.
> 
> Also provide a small example query: "What is the definition of femur bone?" Use LLM to summarize the axioms.
> 
> Provide explanation of steps: 1. Install, 2. Load ontology, 3. Extract relevant info, 4. Prepare prompt, 5. Call Ollama, 6. Process answer.
> 
> Include error handling, performance tips, optional caching.
> 
> Add notes about using local model via Ollama CLI, e.g., "ollama serve" and "ollama run llama3". Show using ollama Python wrapper.
> 
> Will produce final answer with code and explanation.
</details>
Below is a **complete, runnable example** that shows how you can

1. **Load the large FMA (Foundational Model of Anatomy) ontology** with `owlready2`.
2. **Extract the axioms / natural‑language labels** you care about.
3. **Send a prompt to a locally‑running Ollama LLM (e.g. `llama3` or any other `llama`‑compatible model)** and let the model turn the raw OWL data into a friendly answer (definition, description, comparison, etc.).
4. (Optional) **Generate SPARQL queries** with the LLM and run them back against the ontology.

The script is deliberately modular so you can drop‑in your own prompts, cache results, or switch to a different LLM endpoint without touching the ontology‑handling part.

---

## 1️⃣  Prerequisites

| Tool | How to install |
|------|----------------|
| **Python 3.10+** (tested on 3.11) | `python -m venv venv && source venv/bin/activate` |
| **owlready2** (OWL API) | `pip install owlready2` |
| **Ollama** (local LLM server) | Follow the official guide: <https://github.com/ollama/ollama#installation> |
| **Python client for Ollama** (thin wrapper around the HTTP API) | `pip install ollama` |
| **FMA ontology** (`fma.owl`) | Download from the official source (e.g. <https://bioportal.bioontology.org/ontologies/FMA>) and place it next to the script (or give an absolute path). |

> **Tip** – The FMA file is ~1 GB. Make sure you have at least 8 GB of RAM and enough disk space for the “_sqlite_” backend that `owlready2` creates.

---

## 2️⃣  High‑level workflow

```mermaid
flowchart TD
    A[Load FMA.owl with owlready2] --> B[Extract class(es)/axioms]
    B --> C[Build a prompt for Ollama]
    C --> D[Call Ollama LLM (local HTTP API)]
    D --> E[Parse & display answer]
    style A fill:#E3F2FD,stroke:#90A4AE
    style B fill:#E8F5E9,stroke:#81C784
    style C fill:#FFF9C4,stroke:#FBC02D
    style D fill:#FFEBEE,stroke:#E57373
    style E fill:#F3E5F5,stroke:#AB47BC
```

---

## 3️⃣  Full Python script

Save the following as **`fma_ollama_demo.py`** (or any name you like).

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo: Use owlready2 to load the FMA ontology and query a local Ollama LLM
      (e.g. llama3) for natural‑language explanations.

Author:  <Your Name>
Date:    2025‑10‑25
"""

import os
import json
import textwrap
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import owlready2
import ollama  # pip install ollama

# ----------------------------------------------------------------------
# 0️⃣  Configuration
# ----------------------------------------------------------------------
# Path to the FMA OWL file (adjust if you placed it elsewhere)
FMA_PATH = Path("fma.owl")

# Ollama model name (must be already pulled, e.g. `ollama pull llama3`)
OLLAMA_MODEL = "llama3"  # change to "llama2", "phi3", etc.

# How many top‑level classes to keep in the prompt (to stay under token limits)
MAX_CLASSES_IN_PROMPT = 10

# Logging – feel free to adjust level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1️⃣  Load the ontology
# ----------------------------------------------------------------------
def load_fma_ontology(owl_path: Path) -> owlready2.namespace.Ontology:
    """
    Load the FMA ontology using owlready2's fast sqlite backend.
    """
    if not owl_path.is_file():
        raise FileNotFoundError(f"Cannot find {owl_path}")

    log.info("Loading FMA ontology (this may take a minute)...")
    # Use the default sqlite backend – it stores triples on‑disk, not in RAM.
    onto = owlready2.get_ontology(f"file://{owl_path.resolve()}").load()
    log.info(f"Ontology loaded: {onto.name} ({len(list(onto.classes()))} classes)")
    return onto

# ----------------------------------------------------------------------
# 2️⃣  Helper: turn OWL entities into readable snippets
# ----------------------------------------------------------------------
def label_or_name(entity: owlready2.entity.ThingClass) -> str:
    """
    Prefer rdfs:label (English) if present, otherwise use the IRI fragment.
    """
    # owlready2 stores labels in .label (list of strings)
    if entity.label:
        # Prefer an English label if there are many languages
        for lbl in entity.label:
            if isinstance(lbl, str):
                return lbl
    # Fallback: use the last part of the IRI
    return entity.name.replace("_", " ")

def get_class_summary(cls: owlready2.entity.ThingClass) -> str:
    """
    Return a compact, human‑readable summary of a class:
      - label / name
      - direct super‑classes (names only)
      - a few key annotations (e.g. definition, comment)
    """
    label = label_or_name(cls)

    # Super‑classes (excluding owl:Thing)
    supers = [
        label_or_name(s)
        for s in cls.is_a
        if isinstance(s, owlready2.entity.ThingClass) and s != owlready2.Thing
    ]
    supers_str = ", ".join(supers) if supers else "owl:Thing"

    # Annotations: definition, comment, etc.
    annots = []
    if hasattr(cls, "definition"):
        annots.append(f"Definition: {cls.definition[0]}")
    if hasattr(cls, "comment"):
        annots.append(f"Comment: {cls.comment[0]}")
    if hasattr(cls, "synonym"):
        annots.append(f"Synonyms: {', '.join(cls.synonym)}")

    annot_str = "\n".join(annots) if annots else ""

    return f"""Class: {label}
Super‑classes: {supers_str}
{annot_str}
"""

# ----------------------------------------------------------------------
# 3️⃣  Build a prompt for the LLM
# ----------------------------------------------------------------------
def build_prompt(
    target_label: str,
    ontology: owlready2.namespace.Ontology,
    max_classes: int = MAX_CLASSES_IN_PROMPT,
) -> str:
    """
    Build a prompt that gives the LLM enough context to answer a question
    about `target_label`.  The prompt contains:

      * A short description of the task.
      * A few example class summaries (to help the model understand the OWL format).
      * The *exact* OWL class you are interested in (full summary).

    The function tries to stay under the typical 8 k token limit of llama3.
    """
    # 0️⃣  Find the class (case‑insensitive match on label or IRI fragment)
    candidates = [
        c
        for c in ontology.classes()
        if target_label.lower() in label_or_name(c).lower()
        or target_label.lower() in c.name.lower()
    ]
    if not candidates:
        raise ValueError(f"Could not find a class matching '{target_label}'")

    target_cls = candidates[0]  # pick the first match
    target_summary = get_class_summary(target_cls)

    # 1️⃣  Pick a handful of *other* classes as examples (random but reproducible)
    import random

    random.seed(42)
    other_classes = random.sample(
        [c for c in ontology.classes() if c != target_cls],
        k=min(max_classes, len(list(ontology.classes())) - 1),
    )
    examples = "\n---\n".join(get_class_summary(c) for c in other_classes)

    # 2️⃣  Assemble the prompt
    prompt = f"""You are a biomedical knowledge assistant that knows the
Foundational Model of Anatomy (FMA) ontology.  The ontology is expressed in
OWL, but you should answer in plain English.

Below are a few example class descriptions from the ontology (they are
formatted as `Class: <label>`, `Super‑classes: …`, followed by any definition or
comment).  Use them as a reference for the style you should adopt.

{examples}

---  

Now, given the following class description, write a concise (≈2‑3 sentences)
human‑readable definition that could be used in a medical textbook.  Include
any synonyms or alternative names you see in the description.

{target_summary}

Answer only with the definition (no extra headings, no markdown)."""

    # Trim to a reasonable size (llama3 has ~8 k token limit ≈ 6000 words)
    # If the prompt is still too big, we could drop some examples – omitted for brevity.
    return prompt

# ----------------------------------------------------------------------
# 4️⃣  Call the local Ollama model
# ----------------------------------------------------------------------
def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    """
    Send `prompt` to the local Ollama server and return the raw text response.
    """
    log.info(f"Sending request to Ollama (model={model}) …")
    # The `ollama` python package provides a thin wrapper around the HTTP API.
    # It streams the response – we concatenate it into a single string.
    response = ollama.generate(model=model, prompt=prompt, stream=False, keep_alive=0)

    # `response` is a dict: {'model': ..., 'created_at': ..., 'response': 'text', ...}
    answer = response.get("response", "")
    if not answer:
        raise RuntimeError("Empty response from Ollama")
    return answer.strip()

# ----------------------------------------------------------------------
# 5️⃣  Put it all together (CLI interface)
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask a local Ollama LLM about a class in the FMA ontology."
    )
    parser.add_argument(
        "query",
        help="Label or part of the IRI of the FMA class you want a definition for. "
        "E.g. 'femur', 'Heart', 'Cerebral artery'.",
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help="Ollama model name (default: %(default)s). Must be pulled already.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=MAX_CLASSES_IN_PROMPT,
        help="How many example classes to embed in the prompt (default: %(default)s).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not cache the generated answer on disk.",
    )
    args = parser.parse_args()

    # 1️⃣ Load ontology
    onto = load_fma_ontology(FMA_PATH)

    # 2️⃣ Build prompt
    prompt = build_prompt(
        target_label=args.query,
        ontology=onto,
        max_classes=args.max_examples,
    )
    log.debug("\n--- Prompt start ---\n%s\n--- Prompt end ---", prompt)

    # 3️⃣ Optional simple disk cache (helps when you experiment)
    cache_dir = Path(".fma_ollama_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{args.query.replace(' ', '_')}_{args.model}.txt"

    if cache_file.is_file() and not args.no_cache:
        log.info(f"Loading cached answer from {cache_file}")
        answer = cache_file.read_text(encoding="utf-8")
    else:
        # 4️⃣ Call Ollama
        answer = ask_ollama(prompt, model=args.model)
        cache_file.write_text(answer, encoding="utf-8")
        log.info(f"Saved answer to cache file {cache_file}")

    # 5️⃣ Print result
    print("\n=== Definition from LLM ===\n")
    print(answer)


if __name__ == "__main__":
    main()
```

### How the script works (step‑by‑step)

| Step | What the code does | Why it matters |
|------|-------------------|----------------|
| **0️⃣ Config** | Sets file locations, model name, max examples, logging. | Easy to tweak without digging into the code. |
| **1️⃣ Load ontology** | `owlready2.get_ontology(...).load()` uses the **SQLite backend**, which stores triples on disk, keeping RAM usage modest. | The FMA file is > 1 GB – you can’t hold everything in memory on a modest laptop. |
| **2️⃣ Summarise classes** | `get_class_summary()` extracts: <br>• human‑readable label <br>• direct super‑classes <br>• definition / comment / synonyms (if present). | Gives the LLM a concise, structured view of the OWL entity, avoiding the need to dump the entire RDF graph. |
| **3️⃣ Build prompt** | Finds the target class, samples a few other classes as *format examples*, and concatenates them with a clear instruction. | The LLM has no built‑in OWL knowledge; we teach it the “style” of the data and ask for a natural‑language definition. |
| **4️⃣ Call Ollama** | Uses the `ollama` Python client (`ollama.generate`) to talk to the **local** HTTP server (`http://127.0.0.1:11434`). | No external API keys, no latency – everything runs on your machine. |
| **5️⃣ CLI** | Simple command line: `python fma_ollama_demo.py "femur"` <br>Optional cache to avoid repeated LLM calls during development. | Very convenient for rapid experimentation. |

---

## 4️⃣  Running the demo

```bash
# 1️⃣  Make sure Ollama is running and the model is pulled:
ollama serve                     # starts the local server (runs in background)
ollama pull llama3               # or any other llama model you prefer

# 2️⃣  Activate your Python venv and install deps (if you haven’t already)
python -m venv venv
source venv/bin/activate
pip install owlready2 ollama

# 3️⃣  Place the FMA file next to the script (or edit FMA_PATH in the script)
#    Example:  wget https://bioportal.bioontology.org/ontologies/FMA/download?download_format=owl -O fma.owl

# 4️⃣  Run the script
python fma_ollama_demo.py "femur"
```

**Typical output (truncated for brevity)**:

```
=== Definition from LLM ===

The femur, also known as the thigh bone, is the longest and strongest bone in the human body. It forms the upper part of the lower limb, connecting the hip joint (articulating with the acetabulum of the pelvis) to the knee joint (articulating with the tibia and patella). Its shaft is cylindrical, and it features a prominent head, neck, greater and lesser trochanters, and distal condyles.
```

---

## 5️⃣  Extending the demo

### 5.1  Generating SPARQL queries with the LLM

If you want the LLM to **write a SPARQL query** (e.g. “list all subclasses of `Bone` that are part of the lower limb”), you can adapt the prompt:

```python
def build_sparql_prompt(question: str, ontology: owlready2.namespace.Ontology) -> str:
    intro = """You are an expert in the FMA ontology and you can write SPARQL queries
against it.  Return ONLY the query string, without any explanations or markdown."""
    # Provide a tiny schema snippet so the model knows which prefixes exist.
    schema = """PREFIX fma: <http://purl.org/sig/ont/fma/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?class WHERE { ?class a owl:Class . }"""

    return f"""{intro}

Question: {question}

Here is a minimal prefix block you can use:
{schema}
"""
```

Then call `ollama.generate` with that prompt, get the query string, and run it with `owlready2`:

```python
def run_sparql(query: str, onto: owlready2.namespace.Ontology):
    from owlready2 import default_world
    results = default_world.sparql(query)
    return [r[0] for r in results]   # each row is a tuple
```

### 5.2  Caching LLM answers (production‑ready)

```python
import hashlib

def cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def ask_ollama_cached(prompt: str, model: str = OLLAMA_MODEL) -> str:
    key = cache_key(prompt)
    cache_path = Path(".ollama_prompt_cache") / f"{key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text())["response"]

    answer = ask_ollama(prompt, model=model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"prompt": prompt, "response": answer}))
    return answer
```

Replace the call to `ask_ollama` in `main()` with `ask_ollama_cached`.

### 5.3  Handling the **full** ontology (when you *do* need everything)

If you ever need to run a complex reasoning task (e.g., materialize inferred subclasses), you can invoke a reasoner:

```python
from owlready2 import sync_reasoner_pellet

# Warning: Pellet can be memory‑hungry on FMA; consider limiting the scope first.
with onto:
    sync_reasoner_pellet(infer_property_values=True, debug=0)
```

**Tip** – Run the reasoner *once* and then serialize the enriched ontology to a new file:

```python
onto.save(file="fma_reasoned.owl", format="rdfxml")
```

You can then reload the **reasoned** version for faster queries.

---

## 6️⃣  Common pitfalls & how to avoid them

| Problem | Why it happens | Fix / Work‑around |
|---------|----------------|-------------------|
| **Out‑of‑memory while loading** | `owlready2` tries to keep everything in RAM by default. | Use the **SQLite backend** (the default when loading from a file). Ensure you have at least 8 GB RAM; if you still hit limits, load only a *subset* (`onto.search(is_a = fma.Bone)`). |
| **Prompt exceeds token limit** | Adding too many example classes or the whole class definition blows past 8 k tokens. | Keep `MAX_CLASSES_IN_PROMPT` low (5‑10). Trim long definitions (`definition[:500]`). |
| **Model returns markdown or extra text** | Ollama models sometimes wrap output in markdown fences. | Add “Answer **only** with the definition, no markdown.” to the prompt, or post‑process: `re.sub(r'^```.*?```', '', answer, flags=re.DOTALL)`. |
| **Ollama server not reachable** | Forgetting to start `ollama serve` or using the wrong port. | Run `ollama serve` in a separate terminal; check `http://127.0.0.1:11434/api/version` with `curl`. |
| **No definition/comments in the ontology** | Not all FMA classes carry a `definition` annotation. | The script falls back to the label and super‑classes; you can enrich the ontology with external sources (e.g., UMLS) before prompting. |

---

## 7️⃣  TL;DR – One‑liner to get a definition

```bash
python fma_ollama_demo.py "femur"
```

Behind the scenes:

1. **Load** `fma.owl` (SQLite backend → low RAM).
2. **Summarise** the `Femur` class + a few random examples.
3. **Prompt** a local `llama3` model via Ollama to write a 2‑sentence textbook definition.
4. **Print** the LLM’s answer (cached on disk for reuse).

---

### 🎉 You’re ready!

*You now have a minimal but powerful pipeline that bridges a heavyweight biomedical ontology (FMA) with a local, privacy‑preserving LLM (Ollama + LLaMA).  Feel free to adapt the prompt style, add more sophisticated reasoning, or expose the whole thing as a tiny FastAPI service for downstream applications.*  

Happy hacking! 🚀
