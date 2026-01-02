import json
import numpy as np
import os

ANALYSIS_DIR = "downloaded_repos"  # Dossier avec tous les *_analysis.json

def summarize(data, key):
    vals = [d[key] for d in data]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": int(np.min(vals)),
        "max": int(np.max(vals)),
        "median": float(np.median(vals)),
        "sum": int(np.sum(vals)),
    }

metrics = [
    "num_functions", "max_function_depth", "num_classes", "num_assignments",
    "pipeline_patterns", "custom_pipeline_classes", "loc"
]

project_summaries = {}

# 1. Calcul pour chaque projet
for fname in os.listdir(ANALYSIS_DIR):
    if not fname.endswith("_analysis.json"):
        continue
    project = fname.replace("_analysis.json", "")
    with open(os.path.join(ANALYSIS_DIR, fname), "r", encoding="utf-8") as f:
        data = json.load(f)
        if not data:  # Sauter si fichier vide
            continue
        summary = {k: summarize(data, k) for k in metrics}
        summary["has_dynamic_construct (%)"] = 100.0 * sum(1 for d in data if d["has_dynamic_construct"]) / len(data)
        project_summaries[project] = summary

# 2. Rassembler les moyennes/statistiques par projet
global_stats = {k: [] for k in metrics}
dynamic_percents = []

for proj, summ in project_summaries.items():
    for k in metrics:
        global_stats[k].append(summ[k]["mean"])  # On prend la "mean" de chaque projet pour la méta-stat
    dynamic_percents.append(summ["has_dynamic_construct (%)"])

# 3. Calculs globaux sur tous les projets (meta-stats)
meta_summary = {
    k: {
        "mean_of_means": float(np.mean(global_stats[k])),
        "std_of_means": float(np.std(global_stats[k])),
        "median_of_means": float(np.median(global_stats[k])),
        "min_of_means": float(np.min(global_stats[k])),
        "max_of_means": float(np.max(global_stats[k])),
         
    
    }
    for k in metrics
}
meta_summary["has_dynamic_construct (%)"] = {
    "mean": float(np.mean(dynamic_percents)),
    "median": float(np.median(dynamic_percents)),
    "min": float(np.min(dynamic_percents)),
    "max": float(np.max(dynamic_percents)),
    "std": float(np.std(dynamic_percents)),
}


with open("downloaded_repos/mlflow-mlflow_analysis.json", "r", encoding="utf-8") as f:
    mlflow_data = json.load(f)

mlflow_summary = {k: summarize(mlflow_data, k) for k in metrics}
mlflow_summary["has_dynamic_construct (%)"] = 100.0 * sum(1 for d in mlflow_data if d["has_dynamic_construct"]) / len(mlflow_data)


print("====== COMPARAISON MLFLOW vs GLOBAL ======\n")

for k in metrics:
    mlflow_val = mlflow_summary[k]["mean"]
    global_mean = meta_summary[k]["mean_of_means"]
    print(f"{k}:")
    print(f"  → mlflow        = {mlflow_val:.6f}")
    print(f"  → global_mean   = {global_mean:.6f}")
    print(f"  → ratio         = {mlflow_val / global_mean:.2f}x\n")

# Comparaison dynamique
mlflow_dyn = mlflow_summary["has_dynamic_construct (%)"]
global_dyn = meta_summary["has_dynamic_construct (%)"]["mean"]
print("has_dynamic_construct (%):")
print(f"  → mlflow        = {mlflow_dyn:.6f}%")
print(f"  → global_mean   = {global_dyn:.6f}%")
print(f"  → ratio         = {mlflow_dyn / global_dyn:.2f}x\n")
