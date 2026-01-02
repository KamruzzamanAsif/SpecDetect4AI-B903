import json
import os
import glob
import time
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
#  CONFIGURATION
# ----------------------------


load_dotenv()

openai_key = os.getenv("OPENAIKEY")

client = OpenAI(api_key=openai_key)

# Taille du batch et temps de pause
BATCH_SIZE = 1       # nombre de fichiers à traiter avant de dormir
COOLDOWN = 170        # temps d’attente (en secondes) après chaque batch

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
    "You are a code analysis assistant. You receive a Python file, a description of a code smell, "
    "and a methodology to detect it. Your task is to identify all instances of this code smell in "
    "the code and output them in the specified format."
)

# ----------------------------
#  HELPER: appel LLM
# ----------------------------

def get_structured_response(system_prompt: str, user_prompt: str, schema: dict, model: str = "gpt-4.1"):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True
            }
        },
        max_tokens=700
    )
    return json.loads(response.choices[0].message.content)

# ----------------------------
#  MAIN SCRIPT
# ----------------------------

print("Welcome to the AI Code Smell detection tool powered by LLMs.")
path_code_to_analyze = input("First enter the absolute path of the folder to check for AI code smells : ").strip()
print("Now select the OpenAI model. (gpt-4o, gpt-4.1, gpt-4.1-mini, etc.)")
the_model = input("Enter the exact name of the model : ").strip()
output_file_name = input("Please enter the desired output file name (.json) : ").strip()
if not output_file_name.endswith(".json"):
    output_file_name += ".json"

print("Please wait. The code analysis is running...")
py_files = glob.glob(os.path.join(path_code_to_analyze, "*.py"))

instances_per_file_dict = {}

for i, file_path in enumerate(py_files, start=1):
    print("Analysing the file : " + file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        python_code_lines = f.readlines()

    # Ajoute les numéros de lignes dans le code (Line X >>> ...)
    numbered_code = ""
    for idx, code_line in enumerate(python_code_lines):
        numbered_code += f"Line {idx+1} >>> {code_line}"

    instances_dict = {}

    for code_smell, prompt_file in CODESMELLS_PROMPTS.items():
        schema = {
            "type": "object",
            "properties": {
                "instances": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {
                                "type": "integer",
                                "description": f"Line number of the {code_smell} smell. Use the prefix 'Line N >>>'."
                            },
                            "code": {
                                "type": "string",
                                "description": f"The exact code line at that line number."
                            }
                        },
                        "required": ["line", "code"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["instances"],
            "additionalProperties": False
        }

        with open("resources/" + prompt_file, "r", encoding="utf-8") as f:
            user_prompt_skel = f.read()

        user_prompt = (
            f"### Python Code\n\n```python\n{numbered_code}\n```\n\n"
            f"{user_prompt_skel}\n\n"
            "⚠️ IMPORTANT: You must output the exact line number (from the 'Line N >>>' prefix) and copy the full line as-is."
        )

        try:
            instances = get_structured_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=schema,
                model=the_model
            )["instances"]

            code_smell_lines = [inst["line"] for inst in instances]
            code_smell_snippets = [inst["code"] for inst in instances]

            instances_dict[code_smell] = {
                "instances": code_smell_snippets,
                "lines": code_smell_lines
            }

        except Exception as e:
            print(f"[WARN] Could not process {code_smell} in {file_path}: {e}")
            instances_dict[code_smell] = {"instances": [], "lines": []}

    instances_per_file_dict[file_path] = instances_dict

    # Pause après chaque batch
    if i % BATCH_SIZE == 0:
        print(f"[INFO] Processed {i} files, sleeping {COOLDOWN}s to respect rate limits...")
        time.sleep(COOLDOWN)

with open(output_file_name, "w", encoding="utf-8") as f:
    json.dump(instances_per_file_dict, f, indent=2, ensure_ascii=False)

print(f"Done. Results written to {output_file_name}")
