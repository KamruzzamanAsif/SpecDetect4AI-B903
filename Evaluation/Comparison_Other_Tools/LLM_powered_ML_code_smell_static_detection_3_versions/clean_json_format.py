import json

def filter_non_empty(input_path, output_path):
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = {}

    
    for file_path, smells in data.items():
        non_empty_smells = {}
        for smell_name, smell_data in smells.items():
            
            if (smell_data.get("lines") and len(smell_data["lines"]) > 0):
                non_empty_smells[smell_name] = smell_data

        
        if non_empty_smells:
            filtered[file_path] = non_empty_smells

    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f" Résultat sauvegardé dans {output_path}")



filter_non_empty("gpt-4.1-nano.json", "filtered-gpt-4.1-nano.json")
