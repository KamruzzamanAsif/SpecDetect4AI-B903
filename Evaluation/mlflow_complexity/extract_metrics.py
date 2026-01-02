import ast
import os
import json

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            source_code = f.read()
            tree = ast.parse(source_code, filename=filepath)
        except Exception as e:
            print(f"Erreur de parsing pour {filepath}: {e}")
            return None

    stats = {
        "file": filepath,
        "num_functions": 0,
        "max_function_depth": 0,
        "num_classes": 0,
        "num_assignments": 0,
        "pipeline_patterns": 0,
        "custom_pipeline_classes": 0,
        "has_dynamic_construct": False,
        "loc": sum(1 for _ in source_code.splitlines())
    }

    # Ensemble des noms de fonctions/classes indiquant un pipeline
    pipeline_call_names = {"Pipeline", "pipeline", "make_pipeline"}
    pipeline_class_names = {"Pipeline"}  # noms exacts de classes Pipeline communes (utile pour héritage)

    # Analyser les importations pour repérer les alias de pipeline
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # ex: from lib import Pipeline as Alias
            for alias in node.names:
                orig_name = alias.name  # nom original importé
                alias_name = alias.asname  # alias utilisé dans le code, s'il y en a un
                # Si on importe une fonction ou classe dont le nom indique un pipeline
                if orig_name in {"Pipeline", "pipeline", "make_pipeline"} or "Pipeline" in orig_name:
                    # Ajout de l'alias ou du nom original aux noms à détecter
                    if alias_name:
                        pipeline_call_names.add(alias_name)
                        pipeline_class_names.add(alias_name)
                    else:
                        pipeline_call_names.add(orig_name)
                        pipeline_class_names.add(orig_name)
        elif isinstance(node, ast.Import):
            # ex: import some.module.pipeline as pipe
            for alias in node.names:
                module_name = alias.name  # ex: "sklearn.pipeline"
                alias_name = alias.asname
                # Si le module importé contient "pipeline" dans son nom, on peut garder l'alias 
                # car on s'attend à ce qu'il serve à appeler Pipeline via cet alias.
                if "pipeline" in module_name.lower() and alias_name:
                    # On n'ajoute pas directement l'alias dans pipeline_call_names, 
                    # car il s'agit d'un module. On comptera l'appel via attr (géré plus bas).
                    # Cependant, pour les classes importées via import (rare), on pourrait ajouter.
                    pipeline_class_names.add(alias_name)

    # Fonction utilitaire pour calculer la profondeur maximale d'imbrication des fonctions
    def compute_max_depth(node, depth=0):
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                max_d = max(max_d, compute_max_depth(child, depth + 1))
            else:
                max_d = max(max_d, compute_max_depth(child, depth))
        return max_d

    # Parcours de l'AST pour collecter les statistiques
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats["num_functions"] += 1
        if isinstance(node, ast.ClassDef):
            stats["num_classes"] += 1
            # Vérifier si la classe semble être un pipeline
            is_pipeline_class = False
            class_name_lower = node.name.lower()
            if "pipeline" in class_name_lower:
                is_pipeline_class = True
            # Vérifier les classes mères pour détecter un héritage d'une classe Pipeline
            for base in node.bases:
                # Obtenir le nom simple de la base (dernier identifiant du chemin d'attribut)
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr  # dernier élément après le point
                if base_name and ("pipeline" in base_name.lower() or base_name in pipeline_class_names):
                    is_pipeline_class = True
            if is_pipeline_class:
                stats["custom_pipeline_classes"] += 1
        if isinstance(node, ast.Assign):
            stats["num_assignments"] += 1
        if isinstance(node, ast.Call):
            # Nom de la fonction appelée
            func = node.func
            # Détection d'un appel de pipeline (différentes formes)
            if (isinstance(func, ast.Name) and func.id in pipeline_call_names) or \
               (isinstance(func, ast.Attribute) and func.attr in {"Pipeline", "pipeline", "make_pipeline"}):
                stats["pipeline_patterns"] += 1
            # Détection de constructions dynamiques
            if ((isinstance(func, ast.Name) and func.id in {"getattr", "setattr", "exec", "eval", "__import__"}) or \
               (isinstance(func, ast.Attribute) and func.attr in {"getattr", "setattr", "exec", "eval", "__import__"})):
                stats["has_dynamic_construct"] = True

    # Calcul de la profondeur maximale des fonctions imbriquées
    function_depths = [
        compute_max_depth(node, depth=1) 
        for node in ast.walk(tree) 
        if isinstance(node, ast.FunctionDef)
    ]
    stats["max_function_depth"] = max(function_depths) if function_depths else 0

    return stats

def analyze_folder(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                res = analyze_file(filepath)
                if res:
                    results.append(res)
    return results

if __name__ == "__main__":
    folder = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/mlflow-master"
    analysis = analyze_folder(folder)
    with open("mlflow.json", "w", encoding="utf-8") as out:
        json.dump(analysis, out, indent=2)
    print("Analyse terminée ! Résultats sauvegardés dans magenta_analysis.json")
