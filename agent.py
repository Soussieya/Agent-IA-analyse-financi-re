"""
Cœur agentique du projet : un graphe LangGraph minimal qui implémente la
boucle "décision -> outil -> observation -> nouvelle décision".

Architecture :

        START
          │
          ▼
      ┌───────┐   pas d'appel d'outil (réponse finale)
      │ agent │ ─────────────────────────────────► END
      └───┬───┘
          │ appel(s) d'outil demandé(s) par le LLM
          ▼
      ┌───────┐
      │ tools │  (inspect_financial_statement, compute_profitability_ratios,
      └───┬───┘   compute_liquidity_ratios, compute_leverage_ratios, analyze_trend, ...)
          │ résultat de l'outil ajouté aux messages
          └──────────────► retour vers "agent"

Domaine : analyse d'états financiers d'entreprise (bilan + compte de
résultat). Voir tools.py pour le format de données CSV attendu.

Le modèle est paramétrable (voir build_graph) pour permettre de tester
plusieurs LLM locaux différents sans éditer ce fichier — utilisé par
evals/run_eval.py pour comparer Mistral / Llama3.1 / Qwen2.5, etc.
"""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

# Modèle par défaut, surchargeable via la variable d'environnement
# FINANCEAGENT_MODEL (utilisé par evals/run_eval.py pour tester plusieurs
# modèles sans toucher au code). llama3.1:8b a été retenu par défaut après
# avoir constaté que Mistral 7B Instruct (quantifié Q4_K_M) hallucine des
# réponses complètes au lieu de déclencher de vrais appels d'outils,
# malgré la capacité "tools" déclarée par Ollama.
DEFAULT_MODEL = os.environ.get("FINANCEAGENT_MODEL", "llama3.1:8b")

# Adresse du serveur Ollama. None = comportement par défaut de ChatOllama
# (http://localhost:11434), correct en usage normal (hors conteneur).
# Dans Docker, Ollama tourne sur la machine hôte, pas dans le conteneur :
# docker-compose.yml fixe cette variable à http://host.docker.internal:11434.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL") or None

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Tu es FinanceAgent, un agent d'analyse d'états financiers d'entreprise. "
        "Tu disposes d'outils pour inspecter un bilan/compte de résultat et calculer "
        "des ratios de rentabilité, de liquidité et d'endettement. "

        "Tu disposes aussi d'un outil de recherche documentaire "
        "(search_financial_knowledge) pour les questions conceptuelles : définitions "
        "de ratios, repères sectoriels ou interprétation générale. Utilise CET outil "
        "pour ce type de question plutôt que de répondre depuis tes connaissances "
        "générales. "

        "Lorsqu'une question porte sur un fichier financier précis, utilise les "
        "outils disponibles avant de répondre. N'invente jamais de chiffres. "
        "Les résultats retournés par les outils constituent la source de vérité "
        "pour les données financières calculées. "

        "IMPORTANT : après l'utilisation d'un outil de calcul, base ton analyse "
        "uniquement sur les valeurs retournées par cet outil. Ne modifie, "
        "n'inverse et ne complète jamais ces valeurs avec des chiffres supposés. "

        "Pour analyser une tendance sur plusieurs périodes, compare explicitement "
        "les valeurs entre les périodes disponibles avant de conclure. "
        "Ne dis jamais qu'un indicateur augmente s'il diminue, ni qu'il diminue "
        "s'il augmente. Si l'évolution est mixte, indique qu'elle est mixte. "

        "Lorsque la question demande des valeurs précises, donne les valeurs "
        "exactes retournées par l'outil, avec leur période et leur unité. "
        "Lorsque la question demande une interprétation, explique ensuite "
        "simplement ce que ces valeurs signifient. "

        "Si les résultats d'un outil et une connaissance générale semblent "
        "contradictoires, fais confiance aux résultats de l'outil pour les données "
        "de l'entreprise analysée. "

        "Ta réponse finale doit être autosuffisante, concise et factuelle. "
        "Réponds toujours en français."
    )
)


def build_graph(model_name: str = DEFAULT_MODEL):
    """Construit et compile le graphe LangGraph pour un modèle Ollama donné.

    model_name : nom du modèle Ollama à utiliser (ex: "mistral:7b-instruct",
    "llama3.1:8b", "qwen2.5:7b"). Permet de reconstruire l'agent avec un
    modèle différent sans modifier ce fichier — c'est ce que fait
    evals/run_eval.py pour comparer plusieurs modèles.
    """
    llm = ChatOllama(model=model_name, temperature=0, base_url=OLLAMA_BASE_URL)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: MessagesState) -> dict:
        """Nœud de décision : le LLM lit l'historique et choisit sa prochaine action."""
        messages = [SYSTEM_PROMPT, *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    # tools_condition regarde si le dernier message contient un tool_call :
    # si oui -> "tools", sinon -> END (réponse finale de l'agent)
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# Graphe compilé avec le modèle par défaut, prêt à être invoqué depuis main.py
autodata_agent = build_graph()
