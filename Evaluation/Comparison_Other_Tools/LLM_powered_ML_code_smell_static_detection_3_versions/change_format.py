import json
import re
import unicodedata
import pandas as pd
from pathlib import Path
from typing import List, Optional, Any, Dict

# --- Helpers de normalisation -------------------------------------------------

import re
import unicodedata

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _preclean_header(raw: str) -> str:
    """
    1) Supprime 'Line(s)/Ligne(s)' (et variantes) où qu'ils apparaissent
    2) Supprime tout ce qui suit 'Ex:' / 'Example(s):' / 'Exemple(s):'
    3) Compresse espaces/sauts de ligne
    """
    if raw is None:
        return ""
    s = str(raw).replace("\r", "\n")

    # 1) retirer toutes les variantes de Line(s)/Ligne(s)
    pattern_lines = re.compile(
        r"(?:\bLine\(s\)\b|\bLines?\b|\bLigne\(s\)\b|\bLignes?\b)"
        r"(?:\s*/\s*(?:\bLine\(s\)\b|\bLines?\b|\bLigne\(s\)\b|\bLignes?\b))?",
        flags=re.IGNORECASE
    )
    s = pattern_lines.sub("", s)

    # 2) couper tout ce qui suit les exemples
    # prend en charge: Ex:, Ex., Example:, Examples:, Exemple:, Exemples:
    s = re.split(r"\b(?:ex|ex\.|example|examples|exemple|exemples)\s*[:：]\s*", s, flags=re.IGNORECASE)[0]

    # 3) compresser espaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_smell_label(s: str) -> str:
    """
    - pre-clean
    - si 'R<digits> -' existe, garder ce qui SUIT
    - normaliser (accents, casse, ponctuation->espace)
    - petites harmonisations (e.g., apis->api)
    """
    if s is None:
        return ""
    s = _preclean_header(s)

    # garder ce qui suit 'Rxx -'
    m = re.search(r"R\s*\d+\s*[-–—:]\s*(.+)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)

    # harmonisations simples
    tokens = []
    for t in s.split():
        if t == "apis":
            t = "api"
        tokens.append(t)
    return " ".join(tokens)


# --- Lecture template & formatage ---------------------------------------------

def _load_template(template_excel: str):
    df_tmpl = pd.read_excel(template_excel, sheet_name=0)
    cols = [str(c) for c in df_tmpl.columns]
    if not cols:
        raise ValueError("Le template Excel ne contient aucune colonne.")
    first_col_name = cols[0]                       # on réutilise EXACTEMENT ce nom en sortie
    smell_cols = cols[1:] if len(cols) > 1 else [] # colonnes smells, dans l'ordre
    return first_col_name, smell_cols

def _format_lines(lines: Optional[List[Any]]) -> str:
    if not lines:
        return ""
    try:
        clean = sorted({int(x) for x in lines})
    except Exception:
        seen, clean = set(), []
        for x in lines:
            if x not in seen:
                seen.add(x)
                clean.append(x)
    return ";".join(str(n) for n in clean)

# --- Conversion JSON -> Excel -------------------------------------------------

def json_to_matrix_excel(
    json_path: str,
    template_excel: Optional[str],
    out_excel: str,
    strict: bool = True,
    sheet_name: str = "Sheet1",
    debug: bool = False,
) -> None:
    # 1) JSON brut
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

    # 2) Colonnes via template (ordre identique)
    tmpl_first_col, tmpl_smell_cols = _load_template(template_excel) if (template_excel and Path(template_excel).exists()) else ("file", [])

    # Fallback si pas de template
    if not tmpl_smell_cols:
        all_smells = set()
        for smells in data.values():
            all_smells.update(smells.keys())
        all_smells.discard("file")
        tmpl_smell_cols = sorted(all_smells)

    # Mapping normalisé: norm(smell) -> colonne template exacte
    norm_to_template_col: Dict[str, str] = {}
    for c in tmpl_smell_cols:
        norm = _normalize_smell_label(c)
        if norm:
            norm_to_template_col[norm] = c

    if debug:
        print("Exemples de colonnes template normalisées:")
        for c in tmpl_smell_cols[:8]:
            print(f"  '{c}' -> '{_normalize_smell_label(c)}'")

    # 3) DataFrame
    files = sorted(data.keys())
    df = pd.DataFrame(index=files, columns=tmpl_smell_cols, data="")
    df.index.name = "file"

    # 4) Remplissage
    unmatched_smells = set()
    for file_path, smells in data.items():
        for smell_name, payload in smells.items():
            norm_smell = _normalize_smell_label(smell_name)
            col = norm_to_template_col.get(norm_smell)
            if col is None:
                if not strict:
                    # ajouter la colonne absente si nécessaire
                    if smell_name not in df.columns:
                        df[smell_name] = ""
                        norm_to_template_col[norm_smell] = smell_name
                        col = smell_name
                else:
                    unmatched_smells.add(smell_name)
                    continue
            df.at[file_path, col] = _format_lines(payload.get("lines", []))

    # 5) Sortie (avec intitulé exact de la 1re colonne du template)
    out_df = df.reset_index().rename(columns={"file": tmpl_first_col})
    Path(out_excel).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name=sheet_name)

    if unmatched_smells:
        print("⚠️ Smells non mappés (absents du template, ignorés en strict=True) :")
        for s in sorted(unmatched_smells):
            print(f"  - {s}")
    print(f"✅ Excel généré : {out_excel}")

# --- Utilisation avec tes chemins --------------------------------------------

json_path = "gpt-4.1-nano.json"
template_excel = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/SpecDetect4AI/Evaluation/Comparison_Other_Tools/GroundTruth_Manual_Eval.xlsx"
out_excel = "gpt-4.1-nano.xlsx"

json_to_matrix_excel(
    json_path=json_path,
    template_excel=template_excel,
    out_excel=out_excel,
    strict=True,
    sheet_name="AutoEval",
    debug=True  # passe à False après vérification
)
