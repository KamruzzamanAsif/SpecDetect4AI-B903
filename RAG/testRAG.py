#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# --- Imports LangChain et autres ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------------------------
# Configuration utilisateur
# ------------------------------------------------------------------
DOCS_DIR = Path("docs")
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 4
MAX_TOKENS = 500
TEMPERATURE = 0.7
PROMPT_FILE = "prompt.txt" 
OUTPUT_FILE = "output.txt"

# Clé d'API OpenAI
openai_api_key = "TO_ADD"
if not openai_api_key:
    sys.exit(
        "ERREUR: aucune variable d'environnement OPENAI_API_KEY trouvée.\n"
        "  -> export OPENAI_API_KEY=votre_clé_secrète\n"
        "  -> puis relancez le script."
    )

# ------------------------------------------------------------------
# Chargement des documents
# ------------------------------------------------------------------
def load_documents(folder_path: Path):
    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier de documents introuvable: {folder_path.absolute()}")
    docs = []
    for file in folder_path.iterdir():
        if file.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
    return docs

print(f" Chargement des documents depuis: {DOCS_DIR}")
documents = load_documents(DOCS_DIR)
if not documents:
    print("  Aucun document chargé (le dossier docs/ est vide ?). Le RAG répondra surtout avec les connaissances du modèle.")

# ------------------------------------------------------------------
# Découpage des documents en chunks
# ------------------------------------------------------------------
print("  Découpage des documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# ------------------------------------------------------------------
# Embeddings OpenAI + FAISS
# ------------------------------------------------------------------
print(" Calcul des embeddings & construction de l'index FAISS...")
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-3-large"
)
vectorstore = FAISS.from_documents(split_docs, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)

# ------------------------------------------------------------------
# LLM OpenAI (modèle GPT)
# ------------------------------------------------------------------
print(f" Connexion au modèle OpenAI: {OPENAI_MODEL}")
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    openai_api_key=openai_api_key,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# ------------------------------------------------------------------
# Template de prompt personnalisé - AJOUT ICI
# ------------------------------------------------------------------
custom_prompt_template = """Tu es un assistant expert en programmation et expert de l'analyse de code.  
Ta tâche consiste à générer du code de qualité professionnelle basé uniquement sur le contexte fourni ci-dessous, qui provient de documents techniques internes et de ta base de connaissance.

### Consignes :
- Lis attentivement le contexte pour comprendre l'objectif et les contraintes.
- Génére uniquement du code (sans explication sauf si la question le demande explicitement).
- Si la question précise un langage, respecte-le (par défaut, suit le style de l'exemple qui est basé sur la grammaire dsl_grammar.txt).
- Si des morceaux de code existent dans le contexte, adapte-les intelligemment.
- Si tu n'es pas sûr d'une partie de la réponse, précise-le dans un commentaire dans le code.
- Si l'utilisateur demande une explication ou une documentation, ajoute des commentaires clairs.
- Utilises seulement la liste des prédicats disponibles dans predicats.txt. Ces prédicats sont utilisables tel quelle.


### Contexte :  
j'ai un parseur parse_dsl.txt qui me permet de parser des règles écrites dans ma dsl en suivant dsl_grammaire.txt, en code python. 
L'objectif pour toi est que je te donne une description textuelle de la règle, et que tu me donnes une règle écrit au bon format.

Par exemple, pour cette description : https://hynn01.github.io/ml-smells/posts/codesmells/1-unnecessary-iteration/

La règle que j'aimerais que tu me donnes pour cette description serait : "rule R17 "Unnecessary Iteration":
    condition:
        exists sub in AST: (
            isForLoop(sub)
            and (
                (usesIterrows(sub.iter) and isDataFrameVariable(get_base_name(sub.iter), sub.iter))
                or (usesItertuples(sub.iter) and isDataFrameVariable(get_base_name(sub.iter), sub.iter))
                or usesPythonLoopOnTensorFlow(sub)
            )
        )
    action: report "Unnecessary iteration detected; consider using vectorized operations (e.g., DataFrame.add, tf.reduce_sum) for efficiency at line '{'lineno'}'";


### Documents de référence :
{context}

### Question :  
{question}

### Réponse (génère uniquement du code adapté à la question, commente le code si besoin) :
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# ------------------------------------------------------------------
# Chaîne RetrievalQA (RAG) - MODIFICATION ICI
# ------------------------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  
)

# Le reste de votre code reste identique...
def main():
    # Vérifie si le fichier prompt.txt existe
    prompt_path = Path(PROMPT_FILE)
    if not prompt_path.exists():
        print(f" Le fichier {PROMPT_FILE} est introuvable.")
        sys.exit(1)
    
    # Lis le contenu de prompt.txt
    with open(prompt_path, "r", encoding="utf-8") as f:
        question = f.read().strip()
    
    print(f"\n Question lue depuis {PROMPT_FILE} :\n{question}\n")
    
    # Exécuter la chaîne QA avec la question du fichier
    result = qa.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    
    # Afficher la réponse
    print("\n Réponse :\n", answer)
    
   # Prépare le texte à écrire dans le fichier de sortie
    output_lines = []
    output_lines.append("Réponse :\n" + answer + "\n")
    
    if sources:
        output_lines.append("Sources utilisées (top-k = %d):" % len(sources))
        for i, doc in enumerate(sources, 1):
            src = doc.metadata.get("source", "inconnu")
            output_lines.append(f"  [{i}] {src}")
        output_lines.append("")  # Ligne vide pour la fin

    # Écrire dans le fichier de sortie
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(output_lines))
    
    print(f"\n Résultat écrit dans {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()
