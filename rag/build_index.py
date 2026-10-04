"""
Construit l'index vectoriel (RAG) à partir des documents de connaissance
dans rag/knowledge/ (glossaire des ratios + repères sectoriels).

Découpage : chaque section "## Titre" d'un fichier Markdown devient un
chunk indépendant (ces documents ont été écrits exprès pour que chaque
section soit une unité de sens autonome — pas besoin d'un découpeur plus
sophistiqué).

Les embeddings sont calculés par le modèle Ollama "nomic-embed-text" (déjà
présent si tu as suivi l'installation initiale — sinon : ollama pull
nomic-embed-text). L'index est persisté sur disque dans rag/chroma_db/,
donc ce script n'a besoin d'être relancé que si le contenu de
rag/knowledge/ change.

Usage :
    python rag/build_index.py
"""

from __future__ import annotations

import glob
import os
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"


def split_into_sections(text: str, source_name: str) -> list[Document]:
    """Découpe un fichier Markdown en une liste de Document, un par section '## '."""
    # On garde le titre "# ..." de premier niveau à part (pas indexé comme chunk).
    parts = re.split(r"\n(?=## )", text)
    docs = []
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue  # ignore le titre principal / l'avertissement hors-section
        title_line, _, body = part.partition("\n")
        title = title_line.removeprefix("## ").strip()
        docs.append(Document(
            page_content=f"{title}\n{body.strip()}",
            metadata={"source": source_name, "section": title},
        ))
    return docs


def build_index() -> int:
    all_docs: list[Document] = []
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source_name = os.path.basename(path)
        docs = split_into_sections(text, source_name)
        print(f"  {source_name} → {len(docs)} sections")
        all_docs.extend(docs)

    if not all_docs:
        raise SystemExit(f"Aucun document trouvé dans {KNOWLEDGE_DIR}")

    print(f"\nCalcul des embeddings avec '{EMBEDDING_MODEL}' (Ollama doit être lancé)...")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL") or None
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=ollama_base_url)

    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="financeagent_knowledge",
    )
    print(f"\nIndex construit : {len(all_docs)} sections indexées dans {PERSIST_DIR}")
    return len(all_docs)


if __name__ == "__main__":
    build_index()
