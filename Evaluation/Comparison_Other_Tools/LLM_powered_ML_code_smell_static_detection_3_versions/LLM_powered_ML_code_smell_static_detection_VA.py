import json

from openai import OpenAI
from dotenv import load_dotenv
import os
import glob

load_dotenv()

openai_key = os.getenv("OPENAIKEY")

if not openai_key:
    raise ValueError("Clé API OpenAI manquante.")

client = OpenAI(api_key=openai_key)

def get_structured_response(system_prompt: str, user_prompt: str, schema: dict, model: str = "gpt-4.1"):

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema
            }
        }
    )

    # Extract the JSON object from the model’s response
    # return response.choices[0].message.parsed
    return json.loads(response.choices[0].message.content)



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

SYSTEM_PROMPT = "You are a code analysis assistant. You receive a code file, a description of a code smell, and a methodology to detect it. Your task is to identify all instances of this code smell in the code and output them in the specified format."

print("Welcome to the AI Code Smell detection tool powered by LLMs.")
path_code_to_analyze = input("First enter the absolute path of the folder to check for AI code smells : ")
print("Now select the OpenAI model. Chose between gpt-4o, gpt-4.1, o4-mini, o3-mini, gpt-4.1-mini, gpt-4.1-nano or any available OpenAI model.")
the_model = input("Make sure you enter the exact name of the model (as written in the choices) : ")
output_file_name = input("Please enter the desired output file name (.json) : ")

print("Please wait. The code analysis is running...")
py_files = glob.glob(os.path.join(path_code_to_analyze, "*.py"))

instances_per_file_dict = {}

for file_path in py_files:

    print("Analysing the file : " + file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        python_code = f.read()
        python_code_lines = f.readlines()

    instances_dict = {}

    for code_smell, prompt_skeletonn_file_path in CODESMELLS_PROMPTS.items():

        # schema = {
        #     "type": "array",
        #     "items": {
        #         "type": "string",
        #         "description": f"Code line of a specific instance of the {code_smell} code smell."
        #     }
        # }

        schema = {
            "type": "object",
            "properties": {
                "instances": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": f"Code line of a specific instance of the {code_smell} code smell."
                    }
                }
            },
            "required": ["instances"],
            "additionalProperties": False
        }


        with open("resources/" + prompt_skeletonn_file_path, "r", encoding="utf-8") as f:
            user_prompt_skel = f.read()

        user_prompt = f"### Python Code\n\n```python\n{python_code}\n```{user_prompt_skel}"

        instances = get_structured_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=schema,
            model=the_model
        )["instances"]

        

        code_smell_lines = []

        python_code_lines_copy = python_code_lines.copy()

        for instance in instances:

            found = False
            
            for index, code_line in enumerate(python_code_lines_copy):
                
                if instance in code_line:
                    code_smell_lines.append(index + 1)
                    python_code_lines_copy[index] = ""
                    found = True
                    break

            if not found:
                print(f"Warning! The line of the instance '{instance}' of the {code_smell} code smell could not be found!")


        instances_dict[code_smell] = {
            "instances": instances,
            "lines": code_smell_lines
        }

    instances_per_file_dict[file_path] = instances_dict


with open(output_file_name, "w", encoding="utf-8") as f:
    json.dump(instances_per_file_dict, f, indent=2, ensure_ascii=False)

    

