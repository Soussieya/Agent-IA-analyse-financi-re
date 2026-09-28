"""
API FastAPI — étape 1 de la feuille de route AutoDataAgent.

Une seule route : POST /agent/query
Elle envoie la question de l'utilisateur au graphe LangGraph (agent.py)
et retourne :
  - la réponse finale de l'agent
  - la trace des outils appelés (utile pour affichage "Agent Activity"
    côté frontend plus tard, cf. étape 4 de la roadmap)

Lancer le serveur :
    uvicorn main:app --reload --port 8000

Tester :
    curl -X POST http://localhost:8000/agent/query \
         -H "Content-Type: application/json" \
         -d '{"query": "Inspecte les états financiers et calcule les ratios de rentabilité", "dataset_path": "etats_financiers.csv"}'
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agent import DEFAULT_MODEL, autodata_agent
from tools import ALL_TOOLS


class UTF8JSONResponse(JSONResponse):
    """JSONResponse qui déclare explicitement charset=utf-8.

    Par défaut, FastAPI renvoie 'Content-Type: application/json' SANS
    charset. La plupart des clients supposent alors UTF-8, mais certains
    (dont Invoke-RestMethod sous PowerShell selon la version) retombent
    sur Latin-1, ce qui corrompt tous les caractères accentués
    (ex: 'rentabilité' devient 'rentabilitÃ©'). Le déclarer explicitement
    règle le problème pour tous les clients.
    """

    media_type = "application/json; charset=utf-8"


app = FastAPI(title="AutoDataAgent API", version="0.1.0", default_response_class=UTF8JSONResponse)

# CORS ouvert pour l'instant (dev only) — à restreindre avant tout déploiement.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    dataset_path: Optional[str] = None
    dataset_path_b: Optional[str] = None  # second fichier, pour compare_financial_statements


class ToolCallTrace(BaseModel):
    tool: str
    output: str


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallTrace]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": DEFAULT_MODEL, "tool_count": len(ALL_TOOLS)}


@app.post("/agent/query", response_model=QueryResponse)
def query_agent(request: QueryRequest) -> QueryResponse:
    # On injecte le chemin du dataset directement dans le message utilisateur :
    # Mistral doit le reprendre tel quel en argument des outils.
    user_content = request.query
    if request.dataset_path:
        user_content += f"\n\n(Chemin du dataset à utiliser : {request.dataset_path})"
    if request.dataset_path_b:
        user_content += f"\n(Chemin du second dataset à comparer : {request.dataset_path_b})"

    result = autodata_agent.invoke({"messages": [HumanMessage(content=user_content)]})

    messages = result["messages"]

    # Trace des outils appelés pendant l'exécution (pour l'UI "Agent Activity")
    tool_calls: list[ToolCallTrace] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_calls.append(ToolCallTrace(tool=msg.name or "unknown", output=str(msg.content)))

    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    return QueryResponse(answer=final_answer, tool_calls=tool_calls)