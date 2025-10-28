## GPT-OSS 120B phython Owlready2 Ollama Llama frontend for using neurofma-fma3.owl ontology

### USER
Make python coding recommendations with owlready2 for using an Ollama Server running Llama LLM as a front end for searching an OWL ontology named neurofma_fma3.owl

### ASSISTANT
<details type="reasoning" done="true" duration="3">
<summary>Thought for 3 seconds</summary>
> We need to respond with recommendations for using owlready2 with an Ollama server running Llama LLM as a front end for searching an OWL ontology named neurofma_fma3.owl. Provide coding recommendations, perhaps architecture, steps, code snippets. Should include installation, loading ontology, querying, using LLM for natural language to SPARQL conversion, interfacing with Ollama via API, handling responses, caching, error handling, performance tips, example queries. Provide best practices. Also maybe mention using LangChain or similar. Provide a sample Python script. Provide suggestions for modular design. Should be thorough.
</details>
Below is a **complete, production‑ready cheat‑sheet** that shows how you can glue together three moving parts:

| Piece | What it does | Recommended library / tool |
|-------|--------------|---------------------------|
| **OWL ontology** | `neurofma_fma3.owl` – the knowledge base you want to query | `owlready2` (pure‑Python, works with OWL 2) |
| **LLM front‑end** | Turn free‑form natural‑language questions into SPARQL (or OWL‑API) calls | **Ollama** (local Llama‑style model) – exposed via its HTTP API |
| **Orchestration / Prompt engineering** | Build a prompt that reliably steers the model to produce valid SPARQL, validate the query, run it, and format the answer | `requests` (to talk to Ollama) + optional **LangChain** wrapper for caching & tool use |

---

## 1️⃣  High‑level Architecture

```
+-------------------+        +------------------------+        +--------------------+
|  User / UI (CLI, |  -->   |  Python orchestrator   |  -->   |  Ollama LLM Server |
|  web, notebook)  |        |  (owlready2 + prompts) |        |  (Llama 3, Mistral…)|
+-------------------+        +------------------------+        +--------------------+
          ^                                 |                               |
          |                                 v                               v
          |                         +----------------+               +-----------------+
          |                         |  SPARQL engine |               |  Ontology file  |
          |                         |  (owlready2)   |               |  neurofma_fma3.owl |
          |                         +----------------+               +-----------------+
          |                                 ^                               ^
          |                                 |                               |
          +-------------------+   Results   +-------------------------------+
                              |
                    (Human‑readable answer)
```

* The **orchestrator** receives a natural‑language question, builds a prompt, calls Ollama, gets back a SPARQL string, validates it, runs it against the ontology via `owlready2`, and finally formats the result for the user.
* All heavy lifting (reasoning, inference) stays **local** – no data ever leaves your machine, which is important for medical ontologies.

---

## 2️⃣  Install the Dependencies

```bash
# 1️⃣  Ollama (install once, see https://ollama.com/download)
#    e.g. on Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh

# 2️⃣  Pull a Llama‑style model that is good at code & SPARQL.
#    For example, `llama3.1` or `mixtral`:
ollama pull llama3.1   # or `mixtral`

# 3️⃣  Python environment
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install owlready2 requests tqdm langchain # optional LangChain for cache
```

> **Tip** – If you use LangChain, you can also enable **`langchain_community`** for Ollama integration, but the pure‑`requests` approach below is lightweight and gives you full control over the prompt.

---

## 3️⃣  Load the Ontology with `owlready2`

```python
# --------------------------------------------------------------
# file: ontology_loader.py
# --------------------------------------------------------------
from pathlib import Path
from owlready2 import get_ontology, default_world, sync_reasoner_pellet

ONTOLOGY_PATH = Path("neurofma_fma3.owl")

def load_ontology():
    """
    Loads the OWL file, runs a Pellet reasoner (optional) and returns the
    ontology object.  The function is idempotent – you can call it many times
    because `owlready2` caches the world.
    """
    # Force a fresh world (helps when reloading after changes)
    default_world.set_backend(filename=":memory:")  # keep everything in RAM

    # Load
    onto = get_ontology(ONTOLOGY_PATH.as_uri()).load()
    
    # Optional: run a reasoner (Pellet ships with owlready2)
    # This will infer subclass relationships, disjointness, etc.
    # Takes a few seconds for a big ontology.
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    except Exception as exc:
        print("[⚠️] Reasoner failed (maybe Pellet not installed). Continuing without inference.")
    
    return onto

# Example usage:
# onto = load_ontology()
# print(list(onto.classes())[:5])
```

**Why `sync_reasoner_pellet`?**  
* For a neuro‑anatomy ontology you often need inferred `subClassOf` chains (e.g., *Cerebral cortex* ⊑ *Brain region*). Pellet is lightweight and works out‑of‑the‑box.

---

## 4️⃣  Prompt Engineering – Getting SPARQL from Llama

The LLM must be forced to output **only** a SPARQL query (no extra text). The classic “**few‑shot** + **output‑only**” pattern works best.

```python
# --------------------------------------------------------------
# file: prompt_builder.py
# --------------------------------------------------------------
def build_prompt(user_question: str) -> str:
    """
    Returns a prompt that asks the LLM to translate a natural language
    question into a SPARQL query that works against `neurofma_fma3.owl`.

    The prompt includes:
      * A short description of the ontology.
      * Example Q&A pairs (few‑shot) to teach the model the exact format.
      * A strict instruction to output ONLY the query (no markdown).
    """
    description = """
You are an expert in biomedical ontologies and SPARQL.
The target ontology is `neurofma_fma3.owl` which contains the
following relevant namespaces and prefixes:

PREFIX fma: <http://purl.org/sig/ont/fma/fma#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

All classes, object properties, and data properties use the `fma:` prefix.
"""

    # Few‑shot examples (feel free to extend)
    examples = """
Q: Which anatomical structures are part of the "cerebral cortex"?
A:
SELECT ?entity WHERE {
  ?entity fma:isPartOf fma:CerebralCortex .
}

Q: List all subclasses of "brain" with their preferred labels.
A:
SELECT ?subclass ?label WHERE {
  ?subclass rdfs:subClassOf fma:Brain .
  ?subclass rdfs:label ?label .
}
"""

    # The final instruction
    instruction = f"""
{description}
{examples}
Q: {user_question}
A: """   # The model should now write ONLY the query, no backticks.

    return instruction.strip()
```

**Key points in the prompt**

| Goal | How it’s achieved |
|------|-------------------|
| **Force output‑only** | End the prompt with `A:` and *do not* include triple back‑ticks. |
| **Provide namespace context** | List all prefixes up‑front so the model never guesses a wrong URI. |
| **Show concrete examples** | 2–3 “Q → A” pairs teach the model the exact formatting. |
| **Make the task explicit** | “Translate… into a SPARQL query that works against the ontology”. |

---

## 5️⃣  Call Ollama (HTTP API)

Ollama’s API is tiny – a `POST /api/generate` with a JSON body.

```python
# --------------------------------------------------------------
# file: ollama_client.py
# --------------------------------------------------------------
import json
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1"   # change if you pulled another model

def ask_ollama(prompt: str,
               model: str = DEFAULT_MODEL,
               temperature: float = 0.0,
               max_tokens: int = 512,
               stop: Optional[list] = None) -> str:
    """
    Sends the prompt to Ollama and returns the raw text response.
    Temperature 0.0 forces deterministic code generation.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,          # set True for streaming (optional)
        "stop": stop or []        # we can add a stop token like "\nQ:" if needed
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Ollama returns {"response": "...", "done": true, ...}
    return data.get("response", "").strip()
```

**Why `temperature=0.0`?**  
Deterministic output makes validation easier. If you need more creativity (e.g., paraphrasing), bump it to 0.2‑0.3 but keep an eye on malformed queries.

---

## 6️⃣  Validate & Clean the SPARQL

Never trust raw LLM output. A small validator catches syntax errors, stray characters, or missing prefixes.

```python
# --------------------------------------------------------------
# file: sparql_validator.py
# --------------------------------------------------------------
import re
from rdflib.plugins.sparql import parser

PREFIX_PATTERN = re.compile(r"PREFIX\s+\w+:\s+<[^>]+>", re.IGNORECASE)

def extract_query(raw: str) -> str:
    """
    Strips any surrounding markdown or explanatory text.
    Returns the *first* block that looks like a SPARQL query.
    """
    # Remove markdown fences if the model accidentally added them
    raw = re.sub(r"```(sparql)?\n?", "", raw)
    raw = raw.replace("```", "")

    # Keep only lines that start with SELECT / ASK / CONSTRUCT / DESCRIBE
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.upper().startswith(("SELECT", "ASK", "CONSTRUCT", "DESCRIBE")):
            return "\n".join(lines[i:])   # Return from first query line onward
    raise ValueError("No SPARQL query found in LLM output.")

def validate_syntax(sparql: str) -> bool:
    """
    Uses rdflib's internal parser to check syntax.
    Raises SyntaxError on failure.
    """
    parser.parseQuery(sparql)   # will raise if broken
    return True

def ensure_prefixes(sparql: str) -> str:
    """
    Guarantees that the standard prefixes are present.  If the LLM omitted
    them, we prepend them.
    """
    required = """\
PREFIX fma: <http://purl.org/sig/ont/fma/fma#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
    if not PREFIX_PATTERN.search(sparql):
        return required + "\n" + sparql
    return sparql
```

---

## 7️⃣  Run the Query on the Ontology

`owlready2` can execute SPARQL directly against its in‑memory world.

```python
# --------------------------------------------------------------
# file: sparql_executor.py
# --------------------------------------------------------------
from owlready2 import default_world
from typing import List, Tuple

def run_sparql(sparql: str) -> List[Tuple]:
    """
    Executes a SPARQL query on the current Owlready2 world.
    Returns a list of tuples (each tuple = a result row).
    """
    # owlready2 uses rdflib under the hood, so we can call .query()
    # but we need to convert the result to something JSON‑serialisable.
    results = list(default_world.sparql(sparql))
    return results

def format_results(results: List[Tuple]) -> str:
    """
    Turn the raw tuples into a nice human‑readable string.
    """
    if not results:
        return "🚫 No results found."

    # Determine column count from first row
    cols = len(results[0])
    lines = []
    for row in results:
        line = " | ".join(str(item) for item in row)
        lines.append(line)
    return "\n".join(lines)
```

---

## 8️⃣  Orchestrator – Putting It All Together

```python
# --------------------------------------------------------------
# file: neurofma_query_engine.py
# --------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

from ontology_loader import load_ontology
from prompt_builder import build_prompt
from ollama_client import ask_ollama
from sparql_validator import extract_query, validate_syntax, ensure_prefixes
from sparql_executor import run_sparql, format_results

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ONTOLOGY_PATH = Path("neurofma_fma3.owl")
# (You can also expose these via env vars or a small config file.)

# ------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------
def answer_question(question: str,
                    model: str = "llama3.1",
                    temperature: float = 0.0) -> str:
    """
    End‑to‑end pipeline:
        1️⃣ Build prompt
        2️⃣ Call Ollama → raw text
        3️⃣ Extract + validate SPARQL
        4️⃣ Run query on ontology
        5️⃣ Return a user‑friendly answer
    """
    # 1️⃣ Prompt
    prompt = build_prompt(question)
    logging.debug("Prompt sent to LLM:\n%s", prompt)

    # 2️⃣ LLM call
    raw_response = ask_ollama(prompt, model=model, temperature=temperature)
    logging.debug("Raw LLM response:\n%s", raw_response)

    # 3️⃣ Extract + clean SPARQL
    try:
        sparql_raw = extract_query(raw_response)
        sparql = ensure_prefixes(sparql_raw)
        validate_syntax(sparql)
        logging.info("Generated SPARQL:\n%s", sparql)
    except Exception as exc:
        logging.error("Failed to produce a valid SPARQL query: %s", exc)
        return f"❗️ Could not translate the question into a valid SPARQL query.\nDetails: {exc}"

    # 4️⃣ Run query
    try:
        results = run_sparql(sparql)
    except Exception as exc:
        logging.error("SPARQL execution error: %s", exc)
        return f"❗️ Query execution failed.\nDetails: {exc}"

    # 5️⃣ Format answer
    answer = format_results(results)
    return answer

# ------------------------------------------------------------------
# Simple CLI entry‑point (optional)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Ask natural‑language questions about the neuroFMA ontology.")
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("-m", "--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("-t", "--temp", type=float, default=0.0,
                        help="LLM temperature (0=deterministic)")
    args = parser.parse_args()

    if not args.question:
        print("💬 Enter a question (Ctrl‑D to quit):")
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    print("\n🔎 Question:", line)
                    print("💡 Answer:\n", answer_question(line,
                                                        model=args.model,
                                                        temperature=args.temp))
        except KeyboardInterrupt:
            pass
    else:
        q = " ".join(args.question)
        print("\n🔎 Question:", q)
        print("💡 Answer:\n", answer_question(q,
                                            model=args.model,
                                            temperature=args.temp))
```

### How it works

| Step | What the code does |
|------|--------------------|
| **Load ontology** (`ontology_loader.load_ontology`) | Called once at import time – the world stays in RAM for the whole process. |
| **Prompt** (`build_prompt`) | Adds prefixes, a short description, and few‑shot examples. |
| **Ollama** (`ask_ollama`) | Sends the prompt, receives a deterministic text block. |
| **Extract** (`extract_query`) | Strips markdown, returns the first `SELECT/ASK/...` block. |
| **Validate** (`ensure_prefixes` + `validate_syntax`) | Guarantees required prefixes, catches syntax errors early. |
| **Execute** (`run_sparql`) | Uses `default_world.sparql` to run the query against the in‑memory graph. |
| **Format** (`format_results`) | Turns the list of tuples into a pipe‑separated table (easy to read). |
| **CLI** | Simple REPL‑style command‑line tool, but you can wrap `answer_question` in a Flask API, a Jupyter widget, or a VS Code extension. |

---

## 9️⃣  Production‑Ready Tips & Gotchas

| Area | Recommendation |
|------|----------------|
| **Caching** | Frequently asked questions produce the same SPARQL. Use `functools.lru_cache` or a tiny SQLite cache keyed by the *question* → *SPARQL* → *results* tuple. LangChain’s `SQLCache` or `RedisCache` works too. |
| **Batching** | If you need to answer many questions at once, send them **one‑by‑one** to Ollama (its API isn’t multi‑question) but reuse the same ontology world – no reload overhead. |
| **Memory** | `owlready2` loads the whole graph in RAM. For `neurofma_fma3.owl` (~30 k triples) this is < 200 MiB. If you hit memory limits, consider `owlready2.world.set_backend(filename="mydb.sqlite")` to spill to disk. |
| **Reasoner choice** | Pellet is fine for class hierarchy inference. For property chains or SWRL rules you may need `HermiT` (requires Java) – switch `sync_reasoner_pellet` → `sync_reasoner_hermit`. |
| **Security** | Never expose the Ollama port (`11434`) publicly; bind it to `127.0.0.1` or use a reverse‑proxy with authentication. |
| **Prompt versioning** | Store the prompt text (or a Jinja template) in a separate file (`prompt.txt`). When you improve the prompt you only need to bump the version number in the cache key. |
| **Error handling** | Return **structured JSON** to a web front‑end: `{ "question": "...", "sparql": "...", "answer": "...", "error": null }`. The CLI above prints human text, but the same function can be wrapped to emit JSON. |
| **Logging** | Set `logging.basicConfig(level=logging.INFO)` for production; use `DEBUG` locally to see the raw LLM output. |
| **Testing** | Write unit tests that mock `ask_ollama` (return a known SPARQL string) and assert the full pipeline returns the expected rows. Use a small test ontology (e.g., `example.owl`) to keep CI fast. |
| **Model selection** | `llama3.1` works well for code, but if you see systematic prefix errors try `mixtral` (better at structured output) or fine‑tune a small instruction model on a few hundred **question → SPARQL** pairs. |
| **Streaming UI** | If you want a “typing” effect in a web UI, set `stream=True` in the Ollama request and forward each chunk to the frontend via Server‑Sent Events (SSE) or websockets. |

---

## 🔟  Example Session (CLI)

```bash
$ python neurofma_query_engine.py "Which structures are part of the cerebral cortex?"
🔎 Question: Which structures are part of the cerebral cortex?
💡 Answer:
http://purl.org/sig/ont/fma/fma#BrodmannArea4 | Brodmann area 4
http://purl.org/sig/ont/fma/fma#BrodmannArea6 | Brodmann area 6
...
```

If the LLM ever mis‑formats the query:

```
❗️ Could not translate the question into a valid SPARQL query.
Details: no prefix fma: defined
```

You can immediately fix the prompt (add a missing prefix) and re‑run – no code change required.

---

## 📦  Packaging (Optional)

Create a tiny pip‑installable package so your team can import `neurofma_query_engine` in notebooks:

```bash
# setup.cfg
[metadata]
name = neurofma-query
version = 0.1.0
description = Natural‑language query engine for the neuroFMA ontology using Ollama LLM
author = Your Name

[options]
packages = find:
install_requires =
    owlready2
    requests
    rdflib
    tqdm

[options.entry_points]
console_scripts =
    neurofma = neurofma_query_engine:main   # add a wrapper around the CLI
```

Install with `pip install -e .` and you get a clean command line plus an importable `answer_question` function.

---

## 📚  Further Reading & Resources

| Topic | Link |
|-------|------|
| **Ollama API Docs** | https://github.com/ollama/ollama/blob/main/docs/api.md |
| **owlready2 documentation** | https://owlready2.readthedocs.io |
| **SPARQL 1.1 Tutorial** | https://www.w3.org/TR/sparql11-query/ |
| **Prompt‑engineering checklist** | https://github.com/promptcraft/promptcraft |
| **LangChain Ollama integration** | https://python.langchain.com/docs/integrations/providers/ollama |
| **NeuroFMA (FMA 3) ontology** | https://bioportal.bioontology.org/ontologies/FMA |

---

### TL;DR

* Load the OWL file once with **owlready2** (run a Pellet reasoner if you need inference).  
* Build a **few‑shot, prefix‑rich prompt** that tells Ollama to emit *only* a SPARQL query.  
* Call the **Ollama HTTP API** with temperature 0 for deterministic code.  
* **Extract, prepend missing prefixes, validate** the query with `rdflib`.  
* Execute the query against the in‑memory ontology via `default_world.sparql`.  
* Return a clean, human‑readable answer (or JSON) and cache results for speed.  

Follow the code snippets above, adjust the prompt to your domain vocabulary, and you’ll have a **local, privacy‑preserving, LLM‑backed search UI** for the `neurofma_fma3.owl` knowledge base in minutes. Happy coding! 🚀

### USER
How can I add caching for repeated queries using Redis or SQLite?

### ASSISTANT
<details type="reasoning" done="true" duration="2">
<summary>Thought for 2 seconds</summary>
> We need to explain adding caching using Redis or SQLite for repeated queries. Provide code modifications, maybe a Cache class with decorator, integration with the pipeline. Show both options. Provide instructions for setup. Show usage example. Provide handling of cache keys (question, maybe also model and temperature). Explain expiration, size limits. Provide fallback if Redis not available. Provide example using sqlite3 with a simple table. Provide thread-safety. Also mention using functools.lru_cache for in-process. Provide code.
</details>
Below is a **step‑by‑step guide** (with ready‑to‑copy code) that shows you how to plug a **persistent cache** into the `answer_question` pipeline you already have.  
We’ll cover **two alternatives**:

| Cache backend | When to use it | Main pros / cons |
|---------------|----------------|------------------|
| **Redis** (networked, in‑memory) | You already run a Redis server (e.g., in a Docker compose stack) or you need a fast shared cache between multiple Python processes / machines. | • Sub‑millisecond latency <br>• Built‑in TTL & eviction policies <br>• Requires external service |
| **SQLite** (file‑based) | You want a **zero‑dependency** (apart from the std‑lib) cache that survives restarts and works even on a single‑user laptop. | • No external server <br>• Easy to ship with the project <br>• Slightly slower than Redis for very high QPS |

Both implementations expose the **same tiny API** (`Cache.get(key)`, `Cache.set(key, value, ttl=None)`) so you can swap them without touching the rest of the code.

---

## 1️⃣  Define a Minimal Cache Interface

Create a new module `cache_interface.py`. It contains an abstract base class and two concrete implementations.

```python
# --------------------------------------------------------------
# file: cache_interface.py
# --------------------------------------------------------------
import abc
import json
import hashlib
import time
from typing import Any, Optional

# ------------------------------------------------------------------
# Helper: deterministic cache key from a Python object (question, model…)
# ------------------------------------------------------------------
def make_key(*parts: Any) -> str:
    """
    Turns the supplied parts into a reproducible SHA‑256 hex string.
    The parts can be strings, numbers, or any JSON‑serialisable object.
    """
    # Normalise to JSON to guarantee stable ordering for dicts, lists, etc.
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Abstract base class – the only contract the rest of the code cares about
# ------------------------------------------------------------------
class Cache(abc.ABC):
    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return the cached value or None if missing/expired."""

    @abc.abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store `value` under `key`. `ttl` is seconds (None = no expiry)."""

    # Convenience wrapper used by the pipeline
    def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Try to fetch `key`; if missing, call `compute_fn()` (no args),
        store the result and return it.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(key, result, ttl=ttl)
        return result
```

---

## 2️⃣  Redis‑backed Cache

```python
# --------------------------------------------------------------
# file: redis_cache.py
# --------------------------------------------------------------
import json
import os
import redis
from typing import Any, Optional

from .cache_interface import Cache, make_key


class RedisCache(Cache):
    """
    Simple thin wrapper around `redis-py`.  Values are stored as JSON strings.
    The constructor reads connection details from environment variables,
    but you can also pass them directly.
    """

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", "6379")),
        db: int = int(os.getenv("REDIS_DB", "0")),
        password: Optional[str] = os.getenv("REDIS_PASSWORD"),
    ):
        self.client = redis.StrictRedis(host=host, port=port, db=db, password=password)

        # Quick sanity check – raise early if Redis is unreachable
        try:
            self.client.ping()
        except redis.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Redis at {host}:{port}") from exc

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        raw = self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Corrupt entry – delete it and treat as miss
            self.client.delete(key)
            return None

    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl is None:
            self.client.set(key, payload)
        else:
            self.client.setex(key, ttl, payload)
```

### How to spin up a Redis instance (Docker)

```bash
docker run -d --name redis-cache -p 6379:6379 redis:7-alpine
```

Add the following to your `.env` (or export in the shell) if you need custom credentials:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
# export REDIS_PASSWORD=your_secret   # optional
```

---

## 3️⃣  SQLite‑backed Cache

```python
# --------------------------------------------------------------
# file: sqlite_cache.py
# --------------------------------------------------------------
import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional

from .cache_interface import Cache, make_key


class SQLiteCache(Cache):
    """
    A tiny persistent cache that stores JSON‑serialised values in a single
    SQLite file.  It is safe for concurrent reads/writes from multiple threads
    (SQLite handles the locking), but **not** safe for concurrent writes from
    separate processes without `PRAGMA journal_mode=WAL` (which we enable).
    """

    _init_lock = threading.Lock()   # ensure table creation happens once

    def __init__(self, db_path: str = "query_cache.sqlite"):
        self.db_path = db_path
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _connect(self):
        # `detect_types` lets us store datetime as ISO strings if we ever need them.
        return sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=30,
            isolation_level=None,          # autocommit mode
        )

    # ------------------------------------------------------------------
    def _ensure_schema(self):
        with self._init_lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                # `expires_at` = NULL means never expires.
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        expires_at INTEGER   -- Unix epoch (seconds) or NULL
                    )
                    """
                )
                # Speed up look‑ups for expiration pruning
                cur.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache (expires_at)")
                # Enable WAL for better concurrency (multiple readers + 1 writer)
                cur.execute("PRAGMA journal_mode=WAL")
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            value_json, expires_at = row
            if expires_at is not None and expires_at < int(time.time()):
                # Expired – delete and treat as miss
                cur.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return json.loads(value_json)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = None if ttl is None else int(time.time()) + ttl
        payload = json.dumps(value, ensure_ascii=False)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (key, payload, expires_at),
            )
            conn.commit()
        finally:
            conn.close()
```

### Where to keep the DB file

* **Single‑user**: just leave it next to your script (`query_cache.sqlite`).  
* **Shared on a server**: put it under `/var/cache/your‑app/` and make sure the user running the Python process has write permission.

---

## 4️⃣  Wire the Cache into the Query Engine

We’ll add a small **factory** that picks a backend based on an environment variable, and we’ll modify `neurofma_query_engine.py` to use the cache for the **SPARQL string** *and* for the **final answer** (so you don’t even run the query again if the same question is asked).

### 4.1  Cache factory

```python
# --------------------------------------------------------------
# file: cache_factory.py
# --------------------------------------------------------------
import os

from .cache_interface import Cache
from .redis_cache import RedisCache
from .sqlite_cache import SQLiteCache


def get_cache() -> Cache:
    """
    Returns a Cache instance based on the `CACHE_BACKEND` env var.
    Supported values:
        * "redis"   – uses RedisCache (default host/port can be overridden with REDIS_*)
        * "sqlite"  – uses SQLiteCache (file path can be overridden with SQLITE_DB)
        * anything else – falls back to an in‑memory dict cache (good for unit‑tests)
    """
    backend = os.getenv("CACHE_BACKEND", "sqlite").lower()
    if backend == "redis":
        return RedisCache()
    elif backend == "sqlite":
        db_path = os.getenv("SQLITE_DB", "query_cache.sqlite")
        return SQLiteCache(db_path=db_path)
    else:
        # Very lightweight in‑process cache (no persistence)
        from .in_memory_cache import InMemoryCache  # defined below

        return InMemoryCache()
```

### 4.2  Simple in‑memory fallback (useful for CI)

```python
# --------------------------------------------------------------
# file: in_memory_cache.py
# --------------------------------------------------------------
import json
import time
from typing import Any, Optional

from .cache_interface import Cache


class InMemoryCache(Cache):
    """A dict‑based cache that lives only for the process lifetime."""

    def __init__(self):
        self.store: dict[str, tuple[Any, Optional[int]]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at < int(time.time()):
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = None if ttl is None else int(time.time()) + ttl
        self.store[key] = (value, expires_at)
```

### 4.3  Update `neurofma_query_engine.py`

Below is the **minimal change** you need to make. We’ll create a module‑level cache object and then use the `Cache.get_or_compute` helper to memoise two stages:

1. **`question → SPARQL`** (so you don’t ask the LLM again)
2. **`SPARQL → answer`** (so you don’t run the same query twice)

```python
# --------------------------------------------------------------
# file: neurofma_query_engine.py (modified)
# --------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

# Existing imports -------------------------------------------------
from ontology_loader import load_ontology
from prompt_builder import build_prompt
from ollama_client import ask_ollama
from sparql_validator import extract_query, validate_syntax, ensure_prefixes
from sparql_executor import run_sparql, format_results

# NEW imports -----------------------------------------------------
from cache_factory import get_cache
from cache_interface import make_key

# --------------------------------------------------------------
# Global objects
# --------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ONTOLOGY_PATH = Path("neurofma_fma3.owl")

# Load the ontology *once* at import time (keeps it in RAM)
_onto = load_ontology()

# Persistent cache (Redis, SQLite, or in‑memory)
cache = get_cache()

# --------------------------------------------------------------
# Helper: combine the three “parameters” that uniquely identify a request
# --------------------------------------------------------------
def _cache_key(question: str, model: str, temperature: float) -> str:
    """
    The key is a SHA‑256 hash of (question, model, temperature).
    If you later add more knobs (e.g., a different prompt version), just
    extend the tuple.
    """
    return make_key(question, model, temperature)


# --------------------------------------------------------------
# Core pipeline – now fully cached
# --------------------------------------------------------------
def answer_question(
    question: str,
    model: str = "llama3.1",
    temperature: float = 0.0,
    ttl_sparql: Optional[int] = 86400,   # 1 day by default – adjust to your needs
    ttl_answer: Optional[int] = 86400,
) -> dict:
    """
    Returns a dict with the full trace of what happened.
    The dict is JSON‑serialisable and works equally well for a CLI
    or a web endpoint.

    Example return:
    {
        "question": "...",
        "model": "...",
        "temperature": 0.0,
        "sparql": "...",
        "answer": "...",
        "error": null,
        "cached": {"question_to_sparql": true, "sparql_to_answer": false}
    }
    """
    # ------------------------------------------------------------------
    # 1️⃣  Build the *cache key* for the LLM step
    # ------------------------------------------------------------------
    key_llm = _cache_key(question, model, temperature)

    # ------------------------------------------------------------------
    # 2️⃣  Try to fetch a cached SPARQL string; if missing, compute it.
    # ------------------------------------------------------------------
    def _compute_sparql() -> str:
        # (a) Prompt construction
        prompt = build_prompt()
        # (b) Ask Ollama
        raw_output = ask_ollama(prompt, model=model, temperature=temperature)
        # (c) Pull out the SPARQL fragment
        try:
            raw_sparql = extract_query(raw_output)
        except Exception as exc:
            raise RuntimeError(
                f"LLM did not return a parsable SPARQL query. Raw output:\n{raw_output}"
            ) from exc

        # (d) Ensure the required prefixes are present
        full_sparql = ensure_prefixes(raw_sparql)

        # (e) Validate – will raise if broken
        validate_syntax(full_sparql)

        return full_sparql

    sparql = cache.get_or_compute(key_llm, _compute_sparql, ttl=ttl_sparql)

    # ------------------------------------------------------------------
    # 3️⃣  Cache the *answer* itself.  The key for this level is
    #     (question, model, temperature, sparql) – we already have a hash,
    #     so we just tack the SPARQL hash onto it.
    # ------------------------------------------------------------------
    key_answer = make_key(question, model, temperature, sparql)

    def _run_query() -> dict:
        # Run the SPARQL against the already‑loaded ontology
        rows = run_sparql(sparql)
        # Turn rows into a nice string for the caller
        answer_text = format_results(rows)
        return {
            "sparql": sparql,
            "answer": answer_text,
        }

    result = cache.get_or_compute(key_answer, _run_query, ttl=ttl_answer)

    # --------------------------------------------------------------
    # Build the final, user‑facing dict
    # --------------------------------------------------------------
    response = {
        "question": question,
        "model": model,
        "temperature": temperature,
        "sparql": result["sparql"],
        "answer": result["answer"],
        "error": None,
        # Flags that let you see *what* hit the cache
        "cached": {
            "question_to_sparql": cache.get(make_key(question, model, temperature)) is not None,
            "sparql_to_answer": cache.get(make_key(question, model, temperature, result["sparql"])) is not None,
        },
    }

    return response
```

### 4.4  Small CLI wrapper (optional)

If you still want the original “pretty‑print” CLI, add a tiny wrapper at the bottom of the same file:

```python
def main() -> None:   # pragma: no cover
    import argparse, json, sys

    parser = argparse.ArgumentParser(
        description="Ask a natural‑language question against the neuroFMA ontology."
    )
    parser.add_argument("question", help="Your free‑form query")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--ttl-sparql", type=int, default=86400, help="TTL for LLM cache (seconds)")
    parser.add_argument("--ttl-answer", type=int, default=86400, help="TTL for answer cache (seconds)")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = parser.parse_args()

    resp = answer_question(
        args.question,
        model=args.model,
        temperature=args.temp,
        ttl_sparql=args.ttl_sparql,
        ttl_answer=args.ttl_answer,
    )

    if args.json:
        json.dump(resp, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        # Human‑readable fallback (same as before)
        if resp["error"]:
            print(f"❗️ {resp['error']}")
        else:
            print(resp["answer"])
```

Now you can run the script exactly as before, but **repeated queries will be served from the cache**:

```bash
$ export CACHE_BACKEND=redis   # or =sqlite   (default)
$ python neurofma_query_engine.py "Which structures are part of the cerebral cortex?"
```

The first call will:
* hit the LLM,
* store the generated SPARQL in Redis/SQLite,
* run the query,
* store the answer.

The second call (same question, same model/temperature) will be **instantaneous** because both steps are retrieved from the cache.

---

## 5️⃣  Controlling Cache Lifetime

Both backends support a **TTL (time‑to‑live)** argument. In the examples above we used:

* `ttl_sparql = 86400` seconds (1 day) for the LLM‑generated query.
* `ttl_answer = 86400` seconds for the final answer.

You can change those defaults at call‑time, e.g.:

```python
answer = answer_question(
    "What is the volume of the left hippocampus?",
    ttl_sparql=12 * 3600,    # 12 h for the SPARQL string
    ttl_answer=24 * 3600,   # 24 h for the final answer
)
```

If you **don’t** want any expiry, simply pass `ttl=None` (the default).

---

## 6️⃣  TL;DR – Quick “copy‑paste” Summary

1. **Add the cache interface** (`cache_interface.py`) to your project.  
2. **Pick one implementation** (Redis or SQLite) and drop the corresponding file in the same package.  
3. **Create `cache_factory.py`** (and the tiny in‑memory fallback).  
4. **Instantiate a global `cache`** in `neurofma_query_engine.py` via `get_cache()`.  
5. **Wrap the two expensive steps** (`question → SPARQL` and `SPARQL → answer`) with `cache.get_or_compute`.  
6. **Set environment variables** to choose the backend:

```bash
# Choose Redis (requires a running Redis server)
export CACHE_BACKEND=redis
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379

# Choose SQLite (default)
export CACHE_BACKEND=sqlite
export SQLITE_DB=/path/to/query_cache.sqlite
```

That’s it – you now have a **persistent, cross‑process cache** for repeated natural‑language queries, with **zero code changes** required when you want to swap Redis ↔ SQLite.

Happy caching! 🎉

### USER
Is it possible to fine-tune the Ollama model on my own question-to-SPARQL examples?

### ASSISTANT
<details type="reasoning" done="true" duration="11">
<summary>Thought for 11 seconds</summary>
> User asks about fine-tuning Ollama model on own question-to-SPARQL examples. Need to explain possibilities: Ollama is a tool that runs local LLMs via Ollama server; fine-tuning is not universally supported; can use LoRA adapters, custom model building, or instruction fine-tuning via open-source models like Llama 3.1. Provide steps: data prep, format, use `ollama pull` for base model, use `ollama create` or `ollama push`? Actually Ollama supports "modelfile" to create custom models using `FROM` and `ADAPTER` instructions; can incorporate LoRA adapters. Provide example of training with `llama.cpp`? Provide suggestions: use `text-generation-webui` or `OpenChatKit`. Also mention that Ollama's current version (as of 2024) does not expose fine-tuning API; you need to fine-tune the base model outside Ollama then load as custom model.
> 
> So answer should cover:
> 
> - Whether fine-tuning is possible with Ollama directly.
> - Workarounds: use LoRA adapters, use other tools to fine-tune, then import into Ollama.
> - Steps to prepare dataset: JSONL with prompt and completion, include SPARQL as completion.
> - Use tools like `mlc-llm`, `loralib`, `peft`, `bitsandbytes`.
> - Then package as an Ollama model: `FROM` base, `ADAPTER` path, `SYSTEM` etc.
> - Provide sample `Modelfile`.
> - Show how to run fine-tuning with `accelerate` or `trl` (PEFT).
> - Provide evaluation suggestions.
> - Provide advice on compute resources.
> - Provide alternative: use few-shot prompting with examples or Retrieval Augmented Generation (RAG).
> 
> Need to be thorough, include code snippets, references.
</details>
## TL;DR  

**Ollama itself does not expose a “fine‑tune” command**, but you can still give the model your own *question → SPARQL* expertise:

| Option | How you do it | When it makes sense |
|--------|----------------|---------------------|
| **LoRA / adapter training** (outside Ollama) + load the adapter in Ollama | Train a lightweight LoRA on your Q‑→‑SPARQL pairs with 🤗 PEFT/TRL, then create an Ollama *Modelfile* that points to the base model **and** the LoRA file (`ADAPTER <path>`). | You have a modest amount of data (few hundreds to a few thousands of examples) and want to keep the original model unchanged. |
| **Full fine‑tuning** (outside Ollama) + import as a custom model | Use the same training code you would for any open‑source LLM (e.g., `accelerate launch scripts/train.py`). When the new checkpoint is ready, copy it into `~/.ollama/models/<my‑model>` and reference it with `FROM ./my‑model`. | You have a lot of data (tens of thousands+) and enough GPU memory to train the whole model. |
| **Prompt‑engineering / few‑shot prompting** (no training) | Put a few carefully‑chosen examples in the system prompt (or use a “retrieval‑augmented” prompt) and keep the base model unchanged. | You only have a handful of representative queries or you want a zero‑cost quick‑start. |

Below is a **complete, step‑by‑step guide** for the two realistic paths (LoRA adapter + Ollama, and full fine‑tuning + Ollama) plus a quick “few‑shot” fallback. Choose the one that matches your data size, hardware, and timeline.

---

## 1️⃣  Preparing the data

Ollama works with any model that follows the **ChatML** (or OpenAI‑compatible) format: a list of `role: user/assistant/system` messages. For a Q → SPARQL task we treat the **question** as the user message and the **SPARQL query** as the assistant message.

### 1.1  Example JSONL line

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that translates natural‑language questions about anatomy into valid SPARQL queries over the neuroFMA ontology. Return ONLY the SPARQL query, no extra text."},
    {"role": "user",   "content": "Which brain regions are part of the limbic system?"},
    {"role": "assistant","content": "PREFIX fma: <http://purl.org/sig/ont/fma/fmaOwl#>\nSELECT ?region WHERE {\n  ?region a fma:Region ;\n          fma:isPartOf fma:LimbicSystem .\n}"}
  ]
}
```

- **One line per training example** – keep the file in UTF‑8.
- Put **all examples** into `data/q_to_sparql_train.jsonl`.
- If you have a validation set, create `data/q_to_sparql_val.jsonl` with the same schema.

### 1.2  Minimal “system prompt”

The system prompt can be *static* and baked into the model later. Keep it short (the LoRA adapter will learn to apply it automatically).

```text
You are an expert that converts anatomy questions into SPARQL queries over the neuroFMA ontology.
Only output the SPARQL query; do NOT add explanations or surrounding text.
```

---

## 2️⃣  LoRA / Adapter training (lightweight, recommended for most users)

### 2.1  Why LoRA?

- **Parameter‑efficient** – you only train a few hundred thousand extra weights.
- **Fast** – a single RTX 4090 can finish a few‑thousand‑example dataset in < 30 min.
- **Non‑destructive** – the base model stays unchanged, you can drop the adapter any time.

### 2.2  Install the training stack

```bash
# Create a fresh venv (optional but recommended)
python -m venv venv && source venv/bin/activate

# Core libraries
pip install torch==2.4.* \
            transformers==4.44.* \
            accelerate==0.34.* \
            peft==0.11.* \
            trl==0.9.* \
            datasets==2.21.* \
            bitsandbytes==0.44.*   # for 4‑bit quantisation (optional)
```

> **Tip:** If you only have a single GPU, set `CUDA_VISIBLE_DEVICES=0`.  
> If you have multiple GPUs, `accelerate config` will let you launch a distributed job.

### 2.3  Load the base model

Ollama ships a lot of models (e.g., `llama3.1`). Those are the *same* weights you can pull from HuggingFace (`meta-llama/Meta-Llama-3.1-8B`).  
If you already have the model locally via Ollama, you can find it under `~/.ollama/models/<model‑name>/` – copy the `ggml-model-q4_0.bin` (or the full checkpoint) to a directory you’ll point to with `FROM` later.

```python
# train_adapter.py  (run with `accelerate launch train_adapter.py`)
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import Accelerator

# -------------------------------------------------
# 1️⃣  Settings (change as needed)
# -------------------------------------------------
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"   # or a local path
OUTPUT_DIR      = "./adapter_llama3_1_q2sparql"
TRAIN_FILE      = "data/q_to_sparql_train.jsonl"
VAL_FILE        = "data/q_to_sparql_val.jsonl"
MAX_STEPS       = 500                # adjust to your data size
BATCH_SIZE      = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE   = 2e-4
LORA_R          = 64
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
USE_4BIT        = True               # saves GPU memory
# -------------------------------------------------

accelerator = Accelerator()
device = accelerator.device

# -------------------------------------------------
# 2️⃣  Load tokenizer (must match the base model)
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # required for chat models

# -------------------------------------------------
# 3️⃣  Load model (with optional 4‑bit quantisation)
# -------------------------------------------------
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
    )

# -------------------------------------------------
# 4️⃣  Prepare model for LoRA
# -------------------------------------------------
model = prepare_model_for_kbit_training(model) if USE_4BIT else model

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)

# -------------------------------------------------
# 5️⃣  Load dataset (ChatML format)
# -------------------------------------------------
def format_chat(example):
    # `example["messages"]` is a list of dicts as we saved in JSONL
    # Convert to a single string that the model can ingest:
    #   <s>[INST] system_prompt [/INST] user_prompt [/INST] assistant_output </s>
    system = example["messages"][0]["content"]
    user   = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]
    # Ollama / Llama‑Chat style:
    formatted = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>"
    return {"text": formatted}

raw_train = load_dataset("json", data_files=TRAIN_FILE, split="train")
raw_val   = load_dataset("json", data_files=VAL_FILE, split="train")
train_ds = raw_train.map(format_chat, remove_columns=raw_train.column_names)
val_ds   = raw_val.map(format_chat, remove_columns=raw_val.column_names)

def tokenize_fn(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_dataset   = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# -------------------------------------------------
# 6️⃣  Trainer (TRL SFTTrainer)
# -------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=accelerator.state,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=1024,
    packing=False,
    dataset_text_field="text",
    args=accelerator.state,
    # Training hyper‑params
    optim="adamw_torch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=1,
    logging_steps=10,
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    output_dir=OUTPUT_DIR,
    fp16=True,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)   # saves both base checkpoint + LoRA weights

print("✅ LoRA adapter saved to", OUTPUT_DIR)
```

#### Running the training script

```bash
# 1️⃣  Let accelerate generate a config (only needed once)
accelerate config

# 2️⃣  Launch (will use the config you just made)
accelerate launch train_adapter.py
```

After training you will have a folder like:

```
adapter_llama3_1_q2sparql/
├─ adapter_config.json          # LoRA meta‑info
├─ adapter_model.bin           # LoRA weights
└─ tokenizer/                  # (copied from base)
```

### 2.4  Plug the adapter into Ollama

Create a **Modelfile** that points to the base model you already have in Ollama and adds the adapter:

```text
# File: MyQ2Sparql.Modelfile
FROM llama3.1          # <-- name of a model that already exists in Ollama
ADAPTER ./adapter_llama3_1_q2sparql   # relative path to the folder above
SYSTEM """
You are a specialist that translates natural‑language anatomy questions into SPARQL.
Only output the SPARQL query, nothing else.
"""
```

Place the file where you will run the `ollama` CLI, then:

```bash
# Build the custom model (Ollama will copy the adapter into its store)
ollama create my-q2sparql -f MyQ2Sparql.Modelfile
```

You now have a **new model** (`my-q2sparql`) that behaves like `llama3.1` but with your domain knowledge baked in.

```bash
ollama run my-q2sparql "Which brain regions are part of the limbic system?"
```

> **Result:** The model should output a clean SPARQL query (no extra commentary). If you still see stray text, add a few more examples or increase the LoRA rank (`LORA_R`) and re‑train.

---

## 3️⃣  Full fine‑tuning (when you have lots of data)

### 3.1  When to do it

- > 10 k high‑quality Q‑→‑SPARQL pairs.
- You have **≥ 1 GPU with ≥ 24 GB VRAM** (or multi‑GPU node) or you can afford cloud GPU time.
- You want the model to *internalise* the task completely (no adapter overhead).

### 3.2  Training script (using 🤗 Trainer)

Below is a stripped‑down script. The logic is the same as the LoRA script, but we **do not freeze any layers**.

```python
# train_full.py  (run with `accelerate launch train_full.py`)
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "./fine_tuned_llama3_1_q2sparql"
TRAIN_FILE = "data/q_to_sparql_train.jsonl"
VAL_FILE   = "data/q_to_sparql_val.jsonl"
MAX_EPOCHS = 2
BATCH_SIZE = 2   # 8‑B model needs tiny batch sizes unless you use gradient accumulation

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    # optionally enable 4‑bit to fit on a single 24 GB GPU
    # quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...)
)

def format_chat(example):
    system = example["messages"][0]["content"]
    user   = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]
    # Llama‑Chat style (same as LoRA)
    return {"text": f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>"}

def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_raw = load_dataset("json", data_files=TRAIN_FILE, split="train")
val_raw   = load_dataset("json", data_files=VAL_FILE,   split="train")

train_ds = train_raw.map(format_chat, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(format_chat,   remove_columns=val_raw.column_names)

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=MAX_EPOCHS,
    weight_decay=0.01,
    fp16=True,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print("✅ Full fine‑tuned checkpoint saved to", OUTPUT_DIR)
```

#### Build the custom model for Ollama

Ollama can **load any HuggingFace checkpoint** that lives on disk. After the script finishes you’ll have a folder:

```
fine_tuned_llama3_1_q2sparql/
├─ config.json
├─ generation_config.json
├─ pytorch_model.bin   (or many sharded *.bin files)
└─ tokenizer/
```

Create a Modelfile that simply *inherits* from the checkpoint:

```text
# MyFullQ2Sparql.Modelfile
FROM ./fine_tuned_llama3_1_q2sparql   # note the leading `./` – points to the folder you just created
SYSTEM """
You are an expert that translates anatomy questions into SPARQL.
Return ONLY the SPARQL query.
"""
```

Then:

```bash
ollama create my-q2sparql-full -f MyFullQ2Sparql.Modelfile
```

You now have a **stand‑alone model** (no adapters) that you can query exactly like any other Ollama model.

---

## 4️⃣  Prompt‑engineering (zero‑training fallback)

If you cannot afford any training, you can get surprisingly good results by **showing a few examples** in the prompt. This works because Llama‑style chat models are very good at in‑context learning.

```python
SYSTEM = """You are a specialist that translates anatomy questions into SPARQL queries.
Only output the SPARQL query, nothing else."""

EXAMPLES = [
    {
        "question": "Which brain regions are part of the limbic system?",
        "sparql": """PREFIX fma: <http://purl.org/sig/ont/fma/fmaOwl#>
SELECT ?region WHERE {
  ?region a fma:Region ;
          fma:isPartOf fma:LimbicSystem .
}"""
    },
    {
        "question": "What is the volume of the left hippocampus?",
        "sparql": """PREFIX fma: <http://purl.org/sig/ont/fma/fmaOwl#>
SELECT ?vol WHERE {
  fma:LeftHippocampus fma:hasVolume ?vol .
}"""
    },
]

def build_few_shot_prompt(user_question: str) -> str:
    prompt = f"{SYSTEM}\n\n"
    for ex in EXAMPLES:
        prompt += f"User: {ex['question']}\nAssistant: {ex['sparql']}\n\n"
    prompt += f"User: {user_question}\nAssistant:"
    return prompt

# Example usage
question = "Which structures are part of the basal ganglia?"
prompt   = build_few_shot_prompt(question)

# Send to Ollama (no extra training)
response = subprocess.check_output(
    ["ollama", "run", "llama3.1"], input=prompt.encode(), timeout=30
).decode()

print("🧩 Model output (should be a SPARQL query):")
print(response.strip())
```

**Tips to make it work:**

| Tip | How to apply |
|-----|--------------|
| **Add a `SYSTEM` message** that tells the model to *only emit the query* (as we did in the Modelfile). | `SYSTEM "You are a specialist …"` |
| **Include 3–5 high‑quality examples** in the prompt (as above). | In the code, just extend `EXAMPLES`. |
| **If the model still adds “Here is the query:”** – add a line in the system prompt that explicitly forbids it: `Never prepend any text before the query.` | Update `SYSTEM` and rebuild the custom model (if you used a Modelfile). |
| **If you need more coverage** – add a *retrieval* step: keep a vector store of all examples, fetch the *k* most similar examples at runtime, and inject them into the prompt. | Use `faiss` + `sentence‑transformers` to retrieve nearest neighbours. |

---

## 4️⃣  Evaluation & Debugging

### 4.1  Automatic validation set

When you train with `SFTTrainer` or `Trainer`, you already have an **eval step** that computes `eval_loss`. For a more semantic metric:

```python
from datasets import load_metric
import re

def sparql_exact_match(preds, refs):
    # Remove surrounding whitespace and compare line‑by‑line
    norm = lambda s: "\n".join([ln.strip() for ln in s.strip().splitlines() if ln.strip()])
    return sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(preds)

# In the `Trainer` you can pass a `compute_metrics` function:
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Decode
    preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    refs  = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Extract only the part after the last `</s>` (the SPARQL)
    preds = [re.search(r"(SELECT|ASK|CONSTRUCT|DESCRIBE).*", p, flags=re.DOTALL).group(0) if re.search(r"(SELECT|ASK|CONSTRUCT|DESCRIBE)", p) else "" for p in preds]
    refs  = [re.search(r"(SELECT|ASK|CONSTRUCT|DESCRIBE).*", r, flags=re.DOTALL).group(0) if re.search(r"(SELECT|ASK|CONSTRUCT|DESCRIBE)", r) else "" for r in refs]

    return {"exact_match": sparql_exact_match(preds, refs)}
```

Add `compute_metrics=compute_metrics` to the `Trainer`/`SFTTrainer` call.

### 4.2  Spot‑checking failures

1. **Extra text** – add a *post‑processing* step in your application that extracts the first block that looks like a SPARQL query (regex `(?s)PREFIX.*\nSELECT.*\}`).  
2. **Syntax errors** – run the generated query through an RDF validator (e.g., `rdflib` in Python) before sending it to the endpoint. If errors are common, increase the number of **few‑shot examples** in the system prompt or enlarge the LoRA rank.  
3. **Hallucinated predicates** – make sure your training data always uses the *exact* prefixes (`fma:`) you need; the model will copy whatever it sees most often.

---

## 5️⃣  Practical considerations

| Concern | Recommendation |
|---------|----------------|
| **Compute** | LoRA adapters can be trained on a **single RTX 4090** (8 B model) in < 30 min. Full fine‑tuning of an 8 B model needs ~ 24 GB VRAM; you can make it fit with 4‑bit quantisation (`bnb_4bit_compute_dtype="float16"`). |
| **Memory on the device** | When you ship the model to Ollama, the adapter is **merged on‑the‑fly**, so the final memory footprint is the same as the base model (the adapter adds only a few MB). |
| **Data quality** | The biggest win comes from *clean* examples that use **exactly the same prefix declarations** you need at inference time. If you have noisy data, filter it first (`jq` or a small Python script). |
| **Version compatibility** | The base model you pull from HuggingFace **must be the same version** that Ollama already has. If you use a checkpoint that Ollama does not recognise, you can add it to Ollama’s store with `ollama create my‑model -f Modelfile` where the `FROM` line points to a **local folder** containing the checkpoint. |
| **Licensing** | Llama 3.1 is under a *Meta* license that allows **research‑non‑commercial** fine‑tuning. If you need a commercial licence, use a model that explicitly permits fine‑tuning (e.g., Mistral‑7B‑Instruct, Gemma‑2B, or a model from **OpenAI‑compatible** providers). |
| **Iterative improvement** | Start with a **tiny LoRA** (`r=32`) and a handful of examples. If the output still contains prose, increase `r` to 64‑128, add a few more examples, and re‑train. You’ll see diminishing returns after `r≈128`. |

---

## 6️⃣  Full workflow summary (what you’ll actually type)

```bash
# -------------------------------------------------
# 0️⃣  Install everything (once)
# -------------------------------------------------
python -m venv venv && source venv/bin/activate
pip install torch transformers accelerate peft trl datasets bitsandbytes

# -------------------------------------------------
# 1️⃣  Create your JSONL training data (see section 1)
# -------------------------------------------------
# data/q_to_sparql_train.jsonl  (and optional val file)

# -------------------------------------------------
# 2️⃣  Train a LoRA adapter (fast)
# -------------------------------------------------
accelerate config          # answer the prompts once
accelerate launch train_adapter.py   # will produce ./adapter_llama3_1_q2sparql

# -------------------------------------------------
# 3️⃣  Build a custom Ollama model that uses the adapter
# -------------------------------------------------
cat > MyQ2Sparql.Modelfile <<'EOF'
FROM llama3.1
ADAPTER ./adapter_llama3_1_q2sparql
SYSTEM """
You are a specialist that translates anatomy questions into SPARQL.
Only output the SPARQL query, nothing else.
"""
EOF

ollama create my-q2sparql -f MyQ2Sparql.Modelfile

# -------------------------------------------------
# 4️⃣  Use the new model in your pipeline
# -------------------------------------------------
ollama run my-q2sparql "Which brain regions are part of the limbic system?"
```

If you later decide you need **more data**:

```bash
# Re‑run train_adapter.py with a larger LORA_R or with more examples.
# Or, replace the whole thing with train_full.py and rebuild a new Modelfile.
```

---

## 7️⃣  Alternative: Retrieval‑Augmented Generation (RAG)

If you have **hundreds of examples** but don’t want any model‑training at all, you can add a **vector‑store retriever** in front of Ollama:

1. **Index** each training example (question + SPARQL) with a sentence‑transformer (`all-MiniLM-L6-v2`).  
2. At inference time, **retrieve the top‑k nearest questions**, stitch them into the prompt (system + few‑shot examples).  
3. Send the assembled prompt to the *base* Ollama model.

This approach works on a CPU‑only machine (the heavy work is the embedding step) and often reaches > 90 % exact‑match on modest datasets.

```python
from sentence_transformers import SentenceTransformer
import faiss, json, pathlib

# ---------- Build the index (once) ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
train = load_dataset("json", data_files="data/q_to_sparql_train.jsonl", split="train")

questions = [ex["messages"][1]["content"] for ex in train]   # user messages
embs = embedder.encode(questions, normalize_embeddings=True)

dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)   # inner‑product (cosine) similarity
index.add(embs)

# Save the index + the original examples for later lookup
faiss.write_index(index, "sparql.index")
json.dump(train["messages"], open("sparql_examples.json", "w"))
```

```python
# ---------- Inference ----------
def rag_prompt(user_question: str, k: int = 5) -> str:
    # Load index & examples (once per process)
    index = faiss.read_index("sparql.index")
    examples = json.load(open("sparql_examples.json"))

    q_emb = embedder.encode([user_question], normalize_embeddings=True)
    D, I = index.search(q_emb, k)          # D = scores, I = indices

    # Build a few‑shot prompt
    prompt = """You are an expert that translates anatomy questions into SPARQL.
Only output the SPARQL query, nothing else.

"""
    for idx in I[0]:
        msgs = examples[idx]
        prompt += f"User: {msgs[1]['content']}\nAssistant: {msgs[2]['content']}\n\n"

    prompt += f"User: {user_question}\nAssistant:"
    return prompt

# Use the prompt with Ollama
system_prompt = rag_prompt("Which brain regions are part of the limbic system?")
response = subprocess.check_output(
    ["ollama", "run", "llama3.1"], input=system_prompt.encode()
).decode()
print(response.strip())
```

RAG is **zero‑cost** (no GPU training) and can be refined just by adding more examples to the index.

---

## 8️⃣  Resources & Further Reading

| Topic | Link |
|-------|------|
| **Ollama Modelfile reference** | <https://github.com/ollama/ollama/blob/main/docs/Modelfile.md> |
| **PEFT (LoRA) guide** | <https://huggingface.co/docs/peft/en/index> |
| **TRL (Supervised Fine‑Tuning)** | <https://github.com/huggingface/trl> |
| **Training Llama‑3.1 on a single GPU (4‑bit)** | <https://github.com/TimDettmers/bitsandbytes> |
| **ChatML / OpenAI chat format** | <https://github.com/openai/openai-python/blob/main/chatml.md> |
| **RAG with sentence‑transformers** | <https://www.sbert.net/> |
| **SPARQL validation with rdflib** | <https://rdflib.readthedocs.io/en/stable/> |

---

## 9️⃣  Checklist before you start

1. **Data** – at least 500 clean Q → SPARQL pairs for LoRA, > 10 k for full fine‑tuning.  
2. **GPU** – LoRA works on a single 24 GB GPU; full fine‑tuning needs 2 × 24 GB or a cloud instance.  
3. **Ollama version** – make sure you are on **Ollama ≥ 0.2.0** (the `ADAPTER` directive landed in that release).  
4. **License** – confirm that the base model’s licence permits fine‑tuning or adapter training for your intended use.  

Once those boxes are ticked, follow the steps above and you’ll have a **domain‑specialised LLM** that can reliably translate anatomy questions into SPARQL, ready to be used by your existing `neurofma_query_engine.py` pipeline. Happy training! 🚀
