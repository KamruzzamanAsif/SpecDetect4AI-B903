#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, time, random, re, sys
from typing import Dict, Any, List, Optional
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

# =========================
# ========= CONFIG ========
# =========================

# Vitesse: "quality" (le plus complet), "fast" (sélection par imports), "ultrafast" (sélection + pruning agressif)
SPEED_PROFILE = "fast"   # "quality" | "fast" | "ultrafast"

# Fenêtrage autour des lignes intéressantes (pour fast/ultrafast)
PRUNE_WINDOW = 5          # lignes de contexte de chaque côté
PRUNE_MAX_LINES = 800     # plafond de lignes envoyées en ultrafast
MAX_OUTPUT_TOKENS = 900   # borne maximale de tokens de sortie modèle

# Gestion des rate limits
USE_BEST_EFFORT = False               # False = backoff robuste; True = skippe poliment sur 429/erreurs transitoires
RETRY_ON_RATE_LIMIT_BEST_EFFORT = 1   # si USE_BEST_EFFORT=True, nb de retries légers (0 ou 1 conseillé)
PACE_BETWEEN_CALLS_S = 0.6            # délai entre fichiers (0.2–1.0 selon ton quota)

# Modèles à essayer en cascade si le premier sature
FALLBACK_MODELS: List[str] = []       # ex: ["gpt-4.1-mini"]

# Dossier contenant les prompts des smells
RESOURCES_DIR = "resources-VC"

# =========================
# ========= PROMPTS =======
# =========================

SYSTEM_PROMPT = (
    "You are a code analysis assistant. You receive a Python file with line numbers, "
    "plus 22 smell definitions/methodologies. Identify all instances of EACH smell and "
    "return JSON strictly matching the provided schema."
)

# Map nom du smell -> fichier prompt
CODESMELLS_PROMPTS = {
    "Broadcasting Feature Not Used": "prompt-Broadcasting-Feature-Not-Used.txt",
    "Chain Indexing": "prompt-chain-indexing.txt",
    "Columns and DataType Not Explicitly Set": "prompt-Columns-and-DataType-Not-Explicitly-Set.txt",
    "Data Leakage": "prompt-Data-Leakage.txt",
    "DataFrame Conversion API Misused": "prompt-DataFrame-Conversion-API-Misused.txt",
    "Deterministic Algorithm Option Not Used": "prompt-Deterministic-Algorithm-Option-Not-Used.txt",
    "Empty Column Misinitialization": "prompt-Empty-Column-Misinitialization.txt",
    "Gradients Not Cleared Before Backward Propagation": "prompt-Gradients-Not-Cleared-Before-Backward-Propagation.txt",
    "Hyperparameter Not Explicitly Set": "prompt-Hyperparameter-Not-Explicitly-Set.txt",
    "In-Place Apis Misused": "prompt-In-Place-Apis-Misused.txt",
    "Matrix Multiplication Api Misused": "prompt-Matrix-Multiplication-Api-Misused.txt",
    "Memory Not Freed": "prompt-Memory-Not-Freed.txt",
    "Merge Api Parameter Not Explicitly Set": "prompt-Merge-Api-Parameter-Not-Explicitly-Set.txt",
    "Missing The Mask Of Invalid Value": "prompt-Missing-The-Mask-Of-Invalid-Value.txt",
    "Nan Equivalence Comparison Misused": "prompt-Nan-Equivalence-Comparison-Misused.txt",
    "No Scaling Before Scaling Sensitive Operation": "prompt-No-Scaling-Before-Scaling-Sensitive-Operation.txt",
    "Pytorch Call Method Misused": "prompt-Pytorch-Call-Method-Misused.txt",
    "Randomness Uncontrolled": "prompt-Randomness-Uncontrolled.txt",
    "Tensorarray Not Used": "prompt-Tensorarray-Not-Used.txt",
    "Threshold Dependent Validation": "prompt-Threshold-Dependent-Validation.txt",
    "Training Evaluation Mode Improper Toggling": "prompt-Training-Evaluation-Mode-Improper-Toggling.txt",
    "Unnecessary Iteration": "prompt-Unnecessary-Iteration.txt",
}

# =========================
# ====== HELPER FUNCS =====
# =========================

def get_api_key() -> str:
    # Prefer environment variables to avoid hard-coding secrets in source control.
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAIKEY")
    if not key:
        raise ValueError("Clé API OpenAI manquante. Défini OPENAI_API_KEY (ou OPENAIKEY).")
    return key

client = OpenAI(api_key=get_api_key())

def build_schema_for_all_smells(smell_names) -> dict:
    # Unique schéma: { "<smell>": { "instances": [ {code, line}, ... ] }, ... }
    return {
        "type": "object",
        "properties": {
            name: {
                "type": "object",
                "properties": {
                    "instances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "line": {"type": "integer"},
                            },
                            "required": ["code", "line"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["instances"],
                "additionalProperties": False,
            }
            for name in smell_names
        },
        "required": list(smell_names),
        "additionalProperties": False,
    }

def load_prompts_block(resources_dir: str) -> str:
    parts = []
    for smell, fname in CODESMELLS_PROMPTS.items():
        path = os.path.join(resources_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt manquant: {path}")
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f"\n### Smell: {smell}\n{f.read().strip()}\n")
    return "\n".join(parts)

# --- Profilage par libs / pruning --------------------------------------------

SMELL_DEPS = {
    # pandas-centric
    "Chain Indexing": {"pandas"},
    "Merge Api Parameter Not Explicitly Set": {"pandas"},
    "Columns and DataType Not Explicitly Set": {"pandas"},
    "Empty Column Misinitialization": {"pandas"},
    "DataFrame Conversion API Misused": {"pandas"},
    "Nan Equivalence Comparison Misused": {"pandas", "numpy"},
    "Missing The Mask Of Invalid Value": {"pandas", "numpy"},
    # numpy / math
    "Broadcasting Feature Not Used": {"numpy", "torch", "tf"},
    "Matrix Multiplication Api Misused": {"numpy", "torch", "tf"},
    # sklearn / classic ML
    "Hyperparameter Not Explicitly Set": {"sklearn"},
    "Deterministic Algorithm Option Not Used": {"sklearn"},
    "No Scaling Before Scaling Sensitive Operation": {"sklearn"},
    "Threshold Dependent Validation": {"sklearn"},
    "Data Leakage": {"sklearn", "pandas"},
    # deep learning
    "Training Evaluation Mode Improper Toggling": {"torch", "tf"},
    "Gradients Not Cleared Before Backward Propagation": {"torch"},
    "Pytorch Call Method Misused": {"torch"},
    "Tensorarray Not Used": {"tf"},
    "Memory Not Freed": {"torch"},
    # generic
    "Randomness Uncontrolled": {"numpy", "torch", "tf", "sklearn", "random"},
    "Unnecessary Iteration": {"pandas", "numpy", "torch", "tf"},
}

def detect_libs(code_text: str) -> set:
    libs = set()
    if re.search(r'\b(import|from)\s+pandas\b', code_text): libs.add("pandas")
    if re.search(r'\b(import|from)\s+numpy\b', code_text) or re.search(r'\bas\s+np\b', code_text): libs.add("numpy")
    if re.search(r'\b(import|from)\s+sklearn\b', code_text): libs.add("sklearn")
    if re.search(r'\b(import|from)\s+torch\b', code_text): libs.add("torch")
    if re.search(r'\b(import|from)\s+tensorflow\b', code_text): libs.add("tf")
    if re.search(r'\b(import|from)\s+random\b', code_text): libs.add("random")
    if re.search(r'\b(import|from)\s+pyspark\b', code_text): libs.add("spark")
    return libs

def choose_smells_for_libs(libs: set) -> list:
    must_keep = {"Data Leakage", "Randomness Uncontrolled", "Matrix Multiplication Api Misused"}
    if SPEED_PROFILE == "quality":
        return list(CODESMELLS_PROMPTS.keys())

    selected = []
    for smell, deps in SMELL_DEPS.items():
        if deps & libs:
            selected.append(smell)
    selected = sorted(set(selected) | must_keep)

    if not selected:
        selected = list(must_keep)
    return selected

def interesting_line_indices(lines: List[str], libs: set) -> List[int]:
    patterns = [
        r'\.fit\(', r'\.predict\(', r'\.merge\(', r'\.astype\(', r'\.to\(',
        r'==\s*np\.nan|!=\s*np\.nan', r'\.isna\(|\.isnull\(',
        r'\bnp\.', r'\bpd\.', r'\btorch\.', r'\bsklearn\.', r'\btf\.',
        r'\.eval\(|\.train\(', r'\.backward\(', r'\.zero_grad\(',
    ]
    if "pandas" in libs:
        patterns += [r'\[[^\]]+\]\s*\[', r'\.loc\[.*\]\s*\[']
    if "sklearn" in libs:
        patterns += [r'StandardScaler\(|MinMaxScaler\(']
    regexes = [re.compile(p) for p in patterns]

    keep = set()
    for i, ln in enumerate(lines):
        if any(rx.search(ln) for rx in regexes):
            for j in range(max(0, i-PRUNE_WINDOW), min(len(lines), i+PRUNE_WINDOW+1)):
                keep.add(j)
    # Garder imports/def/class au début pour du contexte
    for i, ln in enumerate(lines[:200]):
        if re.search(r'^\s*(import|from|class |def )', ln):
            keep.add(i)
    return sorted(keep)

def code_with_line_numbers_pruned(path: str, code_text: str, libs: set) -> str:
    lines = code_text.splitlines(keepends=True)
    idxs = interesting_line_indices(lines, libs)
    if not idxs:
        idxs = list(range(min(len(lines), PRUNE_MAX_LINES)))
    else:
        if len(idxs) > PRUNE_MAX_LINES:
            idxs = idxs[:PRUNE_MAX_LINES]
    return "".join(f"Line {i+1} >>> {lines[i]}" for i in idxs)

# --- Backoff / Best-effort ----------------------------------------------------

def _sleep_from_exc(e, default_wait: float) -> float:
    msg = getattr(e, "message", str(e))
    m = re.search(r"try again in ([0-9.]+)s", msg)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    try:
        hdrs = getattr(e, "response", None) and getattr(e.response, "headers", {})
        if hdrs:
            for k in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
                if k in hdrs:
                    try:
                        return float(re.sub(r"[^0-9.]", "", hdrs[k]))
                    except Exception:
                        pass
    except Exception:
        pass
    return default_wait

def chat_with_backoff(messages, response_format, model: str, max_retries: int = 10):
    backoff = 2.0
    for _ in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
                response_format=response_format,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except RateLimitError as e:
            wait = _sleep_from_exc(e, backoff)
            time.sleep(wait + random.uniform(0, 0.4))
            backoff = min(backoff * 1.7, 60.0)
        except (APITimeoutError, APIError):
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff = min(backoff * 1.7, 60.0)
    raise RuntimeError("Max retries exceeded for OpenAI API")

def chat_best_effort(messages, response_format, model: str, retries: int = 1) -> Optional[Any]:
    def _once():
        return client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            response_format=response_format,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    for attempt in range(retries + 1):
        try:
            return _once()
        except RateLimitError as e:
            if attempt >= retries:
                return None
            time.sleep(_sleep_from_exc(e, 2.5))
        except (APITimeoutError, APIError):
            return None
    return None

def call_model(messages, response_format, models: List[str], use_best_effort: bool):
    last_err = None
    for model in models:
        if use_best_effort:
            resp = chat_best_effort(messages, response_format, model, RETRY_ON_RATE_LIMIT_BEST_EFFORT)
            if resp is not None:
                return resp
        else:
            try:
                return chat_with_backoff(messages, response_format, model)
            except Exception as e:
                last_err = e
        time.sleep(PACE_BETWEEN_CALLS_S)
    if use_best_effort:
        return None
    raise last_err if last_err else RuntimeError("All model attempts failed.")

# --- Analyse d'un fichier -----------------------------------------------------

def analyze_file_once(path: str, prompts_block_all: str, model_chain: List[str]) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    libs = detect_libs(raw)

    # 1) Choix des smells et prompts
    if SPEED_PROFILE == "quality":
        smell_names = list(CODESMELLS_PROMPTS.keys())
        prompts_block = prompts_block_all
    else:
        smell_names = choose_smells_for_libs(libs)
        parts = []
        for smell in smell_names:
            fname = CODESMELLS_PROMPTS[smell]
            with open(os.path.join(RESOURCES_DIR, fname), "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if SPEED_PROFILE == "ultrafast" and len(txt) > 800:
                txt = txt[:800] + "\n(…truncated for speed…)"
            parts.append(f"\n### Smell: {smell}\n{txt}\n")
        prompts_block = "\n".join(parts)

    # 2) Pruning du code selon profil
    if SPEED_PROFILE == "quality":
        lines = raw.splitlines(keepends=True)
        python_code = "".join(f"Line {i+1} >>> {lines[i]}" for i in range(len(lines)))
    elif SPEED_PROFILE == "fast":
        lines = raw.splitlines(keepends=True)
        cap = min(len(lines), max(PRUNE_MAX_LINES * 2, 1600))
        python_code = "".join(f"Line {i+1} >>> {lines[i]}" for i in range(cap))
    else:  # ultrafast
        python_code = code_with_line_numbers_pruned(path, raw, libs)

    # 3) Schéma pour le SOUS-ensemble envoyé
    schema_subset = build_schema_for_all_smells(smell_names)

    user_prompt = (
        f"### Python Code\n\n```python\n{python_code}\n```\n\n"
        "Analyze the code for EACH of the smells listed below (sections 'Smell: <name>'). "
        "Return ONE JSON object whose top-level keys are EXACTLY those smell names, "
        "each mapping to {'instances':[{'code':str,'line':int}, ...]}. "
        "If a smell has no instance, return an empty array for that smell. "
        "The 'line' must match the number before '>>>'.\n"
        f"{prompts_block}"
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": schema_subset, "strict": True},
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    resp = call_model(messages, response_format, model_chain, USE_BEST_EFFORT)
    if resp is None:
        return None

    content = resp.choices[0].message.content
    parsed = json.loads(content)

    # Normalisation post-traitement:
    # - s'assurer que toutes les clés demandées existent
    # - convertir line -> int si float
    for smell in smell_names:
        parsed.setdefault(smell, {"instances": []})
        parsed[smell].setdefault("instances", [])
        for inst in parsed[smell]["instances"]:
            if isinstance(inst.get("line"), float):
                inst["line"] = int(inst["line"])

    # Si tu préfères avoir TOUTES les 22 clés dans la sortie par fichier:
    for smell in CODESMELLS_PROMPTS.keys():
        if smell not in parsed:
            parsed[smell] = {"instances": []}

    return parsed

# =========================
# ========== MAIN =========
# =========================

def main():
    print("Welcome to the AI Code Smell detection tool powered by LLMs.")
    path_code_to_analyze = input("First enter the absolute path of the folder to check for AI code smells : ").strip()
    print("Now select the OpenAI model. Chose between gpt-5-mini, gpt-4o, gpt-4.1, o4-mini, o3-mini, gpt-4.1-mini, gpt-4.1-nano or any available OpenAI model.")
    the_model = input("Make sure you enter the exact name of the model (as written in the choices) : ").strip()
    output_file_name = input("Please enter the desired output file name (.json) : ").strip()
    if not output_file_name.endswith(".json"):
        output_file_name += ".json"

    if not os.path.isdir(path_code_to_analyze):
        print(f"[ERROR] Folder not found: {path_code_to_analyze}")
        sys.exit(1)

    try:
        prompts_block_all = load_prompts_block(RESOURCES_DIR)  # full prompts (pour quality)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    model_chain = [the_model] + list(FALLBACK_MODELS)

    print("Please wait. The code analysis is running...")
    py_files = glob.glob(os.path.join(path_code_to_analyze, "**", "*.py"), recursive=True)
    if not py_files:
        print("[WARN] No .py files found.")

    instances_per_file_dict: Dict[str, Dict[str, Any]] = {}

    for file_path in py_files:
        print("Analysing the file : " + file_path)
        try:
            results = analyze_file_once(file_path, prompts_block_all, model_chain)
            if results is None:
                instances_per_file_dict[file_path] = {"_status": "skipped_rate_limited"}
            else:
                instances_per_file_dict[file_path] = results
        except Exception as e:
            print(f"[WARN] Skipped {file_path}: {e}")
        time.sleep(PACE_BETWEEN_CALLS_S)

    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(instances_per_file_dict, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {output_file_name}")

if __name__ == "__main__":
    main()