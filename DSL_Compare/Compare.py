import json
from collections import Counter

# Charger les détections Semgrep
with open("/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/SpecDetect4AI/DSL_Compare/CodeQL/resultatsComp.json") as f:
    detections = json.load(f)

# Charger la vérité au sol
with open("/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/SpecDetect4AI/specDetect4ai_results.json") as f:
    gt = json.load(f)

# Construction des ensembles { (file, line) }
def parse_gt(gt):
    gt_pairs = set()
    for file, rules in gt.items():
        lines = []
        for msg in rules.get("R5", []):
            # Extraire le numéro de ligne de la string
            if "line" in msg:
                try:
                    l = int(msg.split("line")[-1].strip())
                    lines.append(l)
                except Exception:
                    continue
        for l in lines:
            gt_pairs.add( (file, l) )
    return gt_pairs

def parse_pred(preds):
    return set( (p["file"], p["line"]) for p in preds )

gt_pairs = parse_gt(gt)
pred_pairs = parse_pred(detections)

# Calcul TP, FP, FN
TP = len(gt_pairs & pred_pairs)
FP = len(pred_pairs - gt_pairs)
FN = len(gt_pairs - pred_pairs)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# Sauvegarder le résultat dans un fichier texte
with open("CodeQL/evaluation_results.txt", "w") as f:
    f.write(f"TP: {TP}\n")
    f.write(f"FP: {FP}\n")
    f.write(f"FN: {FN}\n")
    f.write(f"Précision: {precision:.3f}\n")
    f.write(f"Rappel: {recall:.3f}\n")
    f.write(f"F1-score: {f1:.3f}\n")

print("Les résultats ont été sauvegardés dans 'evaluation_results.txt'.")