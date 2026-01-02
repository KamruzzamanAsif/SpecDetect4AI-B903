import json

# Charger le résultat Semgrep
with open("/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/SpecDetect4AI/DSL_Compare/Semgrep/resultats.json", "r") as f:
    data = json.load(f)

# Extraire la liste des alertes avec seulement le fichier et la ligne
output = []
for r in data.get("results", []):
    entry = {
        "file": r["path"],
        "line": r["start"]["line"]
    }
    output.append(entry)

# Sauver le résultat simplifié dans un nouveau fichier
with open("/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/SpecDetect4AI/DSL_Compare/Semgrep/resultatsComp.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"{len(output)} alertes extraites.")
