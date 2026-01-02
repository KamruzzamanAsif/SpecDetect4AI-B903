import json
import os
import glob
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv()

OPENAIKEY = "your_openai_key_here"
if not OPENAIKEY:
    print("Warning! Missing OpenAI API key. Will fail if OpenAI model is used.")
    # raise ValueError("Clé API OpenAI manquante.")

OPENROUTERKEY = "your_openrouter_key_here"
if not OPENROUTERKEY:
    print("Warning! Missing OpenRouter API key. Will fail if the model used is not from OpenAI.")



"""
AI Code Smell detection tool (patched)
- Adds Qwen 3 Coder via OpenRouter
- Supports JSON structured outputs via Chat Completions with schema (fallback to json_object)
- Recursive .py discovery
- Robust paths to prompt templates
"""


# -------------------------
# Constants
# -------------------------
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

SYSTEM_PROMPT = (
    "You are a code analysis assistant. You receive a code file, a description of a code "
    "smell, and a methodology to detect it. Your task is to identify all instances of this "
    "code smell in the code and output them in the specified format.\n\n"
    "Important rules:\n"
    "- Reply with a SINGLE JSON object only; no Markdown, no explanations.\n"
    "- Strictly follow the provided JSON schema: {schema}.\n"
)

# -------------------------
# Helpers
# -------------------------

def build_client_and_model(model: str, openAI_model: bool):
    """Return (client, model_slug, structure_response_supported) according to provider/model."""
    if openAI_model:
        if not OPENAIKEY:
            raise RuntimeError("OPENAIKEY is missing but an OpenAI model was selected.")
        client = OpenAI(api_key=OPENAIKEY)
        # Use as-is for OpenAI models (e.g., gpt-4o, gpt-4.1, gpt-5, o4-mini)
        the_model = model
        structure_response_supported = True  # Chat Completions + json_schema
        return client, the_model, structure_response_supported

    # OpenRouter branch
    if not OPENROUTERKEY:
        raise RuntimeError("OPENROUTERKEY is missing but a non-OpenAI model was selected.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTERKEY)
    m = model.lower().strip()
    the_model = None
    structure_response_supported = True  # default True; some routes may need fallback

    if "deepseek" in m:
        # Free route; tends to not respect schemas -> mark unsupported
        the_model = "deepseek/deepseek-chat-v3.1:free"
        structure_response_supported = False
    elif "qwen" in m:
        # Qwen 3 Coder (choose free vs paid by user hint)
        if ":free" in m or "free" in m:
            the_model = "qwen/qwen3-coder:free"
        else:
            the_model = "qwen/qwen3-coder"
        structure_response_supported = True
    elif "llama" in m:
        the_model = "meta-llama/llama-3.1-405b-instruct:free"
        structure_response_supported = True

    if not the_model:
        raise SystemError("Model not supported! Try 'qwen3-coder', 'qwen/qwen3-coder', 'qwen/qwen3-coder:free', 'deepseek', or 'llama'.")

    return client, the_model, structure_response_supported


def call_model_json(client: OpenAI, the_model: str, system_prompt: str, user_prompt: str, schema: dict, structure_response_supported: bool):
    """Call Chat Completions API aiming for JSON output. First try JSON schema, then fallback."""
    if structure_response_supported:
        try:
            # First attempt: JSON Schema (strict)
            response = client.chat.completions.create(
                model=the_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            return response.choices[0].message.content
        except Exception:
            # Fallback to simple JSON object (no validation)
            response = client.chat.completions.create(
                model=the_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt + "\nAlways reply ONLY with a valid JSON object matching the schema."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
    else:
        # No schema support: force plain JSON with instructions
        response = client.chat.completions.create(
            model=the_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt + "\nReply ONLY with a valid JSON object."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def get_structured_response(system_prompt: str, user_prompt: str, schema: dict, model: str = "gpt-4.1", openAI_model: bool = True, max_retries: int = 3):
    client, the_model, schema_ok = build_client_and_model(model, openAI_model)

    tries = 0
    while True:
        tries += 1
        try:
            raw = call_model_json(client, the_model, system_prompt, user_prompt, schema, schema_ok)
            return json.loads(raw)
        except Exception as e:
            print(f"Model output parse error (try {tries}/{max_retries}): {e}")
            traceback.print_exc()
            if tries >= max_retries:
                raise
            # small backoff
            time.sleep(1.5 * tries)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Welcome to the AI Code Smell detection tool powered by LLMs.")

    path_code_to_analyze = input("First enter the absolute path of the folder to check for AI code smells : ").strip()
    print("Now select a model.")
    print("Supported Models include all OpenAI proprietary models (gpt-4o, gpt-4.1, gpt-5, o4-mini, etc), deepseek-chat-v3.1, llama-3.1-405b-instruct, qwen3-coder.")

    the_model = input("Make sure you enter the exact name of the model (as written in the choices) : ").strip()
    provider_is_openai = "true" == input("Is it an (proprietary) OpenAI Model ? (answer with true or false) : ").lower().strip()

    output_file_name = input("Please enter the desired output file name (.json) : ").strip()
    if not output_file_name.lower().endswith(".json"):
        output_file_name += ".json"

    # Resolve base paths
    BASE_DIR = Path(__file__).resolve().parent
    prompts_dir = BASE_DIR / "resources-VC"

    # Gather Python files (recursive)
    print("Please wait. The code analysis is running...")
    py_files = glob.glob(os.path.join(path_code_to_analyze, "**", "*.py"), recursive=True)

    if not py_files:
        print("No .py files found. Check the folder path.")

    instances_per_file_dict = {}

    for file_path in py_files:
        print("Analysing the file : " + file_path)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            python_code_lines = f.readlines()

        # Build numbered code block to help the model with line numbers
        python_code = ""
        for index, code_line in enumerate(python_code_lines):
            numbered_line = f"Line {index + 1} >>> {code_line}"
            python_code += numbered_line

        instances_dict = {}

        for code_smell, prompt_filename in CODESMELLS_PROMPTS.items():
            print(f"{file_path} - {code_smell}")

            schema = {
                "type": "object",
                "properties": {
                    "instances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": f"Code line of a specific instance of the {code_smell} code smell.",
                                },
                                "line": {
                                    "type": "integer",
                                    "description": (
                                        f"The line number of the detected {code_smell} code smell instance. "
                                        f"The line number is indicated at the beginning of each line (before the '>>>'), base yourself on it."
                                    ),
                                },
                            },
                            "required": ["code", "line"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["instances"],
                "additionalProperties": False,
            }

            prompt_path = prompts_dir / prompt_filename
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            with open(prompt_path, "r", encoding="utf-8") as pf:
                user_prompt_skel = pf.read()

            # Insert schema text into system prompt for extra guidance
            sys_prompt = SYSTEM_PROMPT.replace("{schema}", json.dumps(schema))

            user_prompt = (
                f"### Python Code\n\n```python\n{python_code}\n```\n" + user_prompt_skel
            )

            result = get_structured_response(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                schema=schema,
                model=the_model,
                openAI_model=provider_is_openai,
            )

            instances_dict[code_smell] = result

            # If using OpenRouter (often rate-limited), pause a bit
            if not provider_is_openai:
                print("Sleeps for 3 seconds to avoid rate limit. Increase if per minute rate limit error is encountered.")
                time.sleep(3)
                print("Resuming...")

        instances_per_file_dict[file_path] = instances_dict

    # Write outputs
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(instances_per_file_dict, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results written to: {output_file_name}")
