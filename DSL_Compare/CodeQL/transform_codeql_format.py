import json

# Lis le fichier JSON exporté par CodeQL
with open("/Users/bramss/Desktop/codeql_project/results_mlflow.json", "r") as f:
    data = json.load(f)

# Récupère les tuples du select
tuples = data["#select"]["tuples"]

# On va stocker le résultat ici
output = []

for t in tuples:
    # La colonne de localisation est toujours la 2e colonne ([1])
    location = t[1]["label"]  # ex: "examples/shap/binary_classification.py:19"
    if ":" in location:
        file_path, line = location.rsplit(":", 1)
        output.append({
            "file": file_path,
            "line": int(line)
        })

# Écrit le résultat dans un fichier ou affiche-le
with open("locations.json", "w") as f:
    json.dump(output, f, indent=2)

# Optionnel : affichage
print(json.dumps(output, indent=2))
