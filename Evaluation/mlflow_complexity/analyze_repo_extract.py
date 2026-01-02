import csv, io, requests, zipfile, ast, json, os, time

CSV_FILE = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/github_repo_metadata.csv"  
OUTPUT_DIR = "downloaded_repos"       
GITHUB_TOKEN = "SECRET"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_max_depth(node, depth=0):
    m = depth
    for child in ast.iter_child_nodes(node):
        # Incrémente la profondeur pour les fonctions imbriquées
        if isinstance(child, ast.FunctionDef):
            m = max(m, compute_max_depth(child, depth+1))
        else:
            m = max(m, compute_max_depth(child, depth))
    return m

headers = {"Authorization": f"token {GITHUB_TOKEN}"}
with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader, start=1):
        owner, repo = row.get('repo_owner'), row.get('repo_name')
        if not owner or not repo:
            continue
        url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        print(f"Téléchargement de {owner}/{repo}... (répo #{i})")
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            print(f"Erreur réseau pour {owner}/{repo}: {e}")
            continue
        if response.status_code != 200:
            code = response.status_code
            msg = "introuvable (404)" if code==404 else "accès refusé (403)" if code==403 else f"échec (code {code})"
            print(f"✖ Dépôt {owner}/{repo} {msg}")
            continue

        results = []
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_name in z.namelist():
                    if file_name.endswith(".py"):
                        file_bytes = z.read(file_name)  # Lecture en mémoire
                        source = file_bytes.decode('utf-8', errors='ignore')
                        try:
                            tree = ast.parse(source, filename=file_name)
                        except Exception:
                            continue  # Fichier non analysable (syntax error)
                        stats = {
                            "file": f"{owner}/{repo}/{file_name}",
                            "num_functions": 0,
                            "max_function_depth": 0,
                            "num_classes": 0,
                            "num_assignments": 0,
                            "pipeline_patterns": 0,
                            "custom_pipeline_classes": 0,
                            "has_dynamic_construct": False,
                            "loc": len(source.splitlines())
                        }
                        # Parcours de l'AST pour extraire les métriques
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                stats["num_functions"] += 1
                            elif isinstance(node, ast.ClassDef):
                                stats["num_classes"] += 1
                                # Vérification des classes "Pipeline"
                                name_lower = node.name.lower()
                                is_pipeline = "pipeline" in name_lower
                                for base in node.bases:
                                    base_name = (base.id if isinstance(base, ast.Name) 
                                                 else base.attr if isinstance(base, ast.Attribute) 
                                                 else "")
                                    if "pipeline" in base_name.lower():
                                        is_pipeline = True
                                if is_pipeline:
                                    stats["custom_pipeline_classes"] += 1
                            elif isinstance(node, ast.Assign):
                                stats["num_assignments"] += 1
                            elif isinstance(node, ast.Call):
                                # Nom de la fonction appelée
                                func = node.func
                                func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                                if "pipeline" in func_name.lower():
                                    stats["pipeline_patterns"] += 1
                                if func_name in {"getattr", "setattr", "exec", "eval", "__import__"}:
                                    stats["has_dynamic_construct"] = True
                        # Profondeur maximale des fonctions imbriquées
                        depths = [compute_max_depth(n, depth=1) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                        stats["max_function_depth"] = max(depths) if depths else 0

                        results.append(stats)
        except zipfile.BadZipFile:
            print(f"✖ Archive ZIP corrompue pour {owner}/{repo}")
            continue

        # Écriture des résultats JSON pour le dépôt courant
        out_path = os.path.join(OUTPUT_DIR, f"{owner}-{repo}_analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✔ Analyse de {owner}/{repo} terminée : {len(results)} fichiers Python analysés.")
        # Pause toutes les 100 requêtes pour éviter de dépasser les limites de l'API
        if i % 100 == 0:
            print("⏸ Pause 60s après 100 dépôts traités...")
            time.sleep(60)
