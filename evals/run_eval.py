"""
Mini-benchmark : compare la fiabilité du tool-calling de plusieurs modèles
Ollama sur le même jeu de questions financières.

Ce script répond à une question précise mise en évidence pendant le
développement de FinanceAgent : certains modèles locaux (ex. Mistral 7B
Instruct) déclarent supporter le tool-calling ("tools" dans `ollama show`)
mais, en pratique, hallucinent parfois une réponse complète au lieu de
déclencher un vrai appel d'outil. Ce script quantifie ce phénomène plutôt
que de se fier à des tests manuels ponctuels.

Métrique principale : "tool_triggered" — le modèle a-t-il déclenché au
moins un VRAI appel d'outil (exécuté par LangGraph, pas juste écrit en
texte) pour une question qui en nécessite un ? C'est la métrique la plus
fiable, car le calcul lui-même (fait par tools.py, du Python pur) est
toujours exact — ce qui varie d'un modèle à l'autre, c'est sa capacité à
déclencher l'outil plutôt qu'à halluciner.

Usage :
    python evals/run_eval.py
    python evals/run_eval.py --models mistral:7b-instruct llama3.1:8b qwen2.5:7b
    python evals/run_eval.py --test-cases evals/test_cases.json

Prérequis : Ollama doit tourner, avec chaque modèle testé déjà tiré
(ollama pull <modèle>). Le script doit être lancé depuis la racine du
projet (là où se trouvent les fichiers .csv utilisés par les tests).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Permet d'importer agent.py / tools.py depuis le dossier parent, que le
# script soit lancé depuis la racine du projet ou depuis evals/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_graph

DEFAULT_MODELS = ["mistral:7b-instruct", "llama3.1:8b", "qwen2.5:7b"]


def build_user_message(test_case: dict) -> str:
    content = test_case["query"]
    if test_case.get("dataset_path"):
        content += f"\n\n(Chemin du dataset à utiliser : {test_case['dataset_path']})"
    if test_case.get("dataset_path_b"):
        content += f"\n(Chemin du second dataset à comparer : {test_case['dataset_path_b']})"
    return content


def run_one(agent_graph, test_case: dict) -> dict:
    """Exécute un cas de test et retourne les métriques observées."""
    start = time.time()
    error = None
    tools_called: list[str] = []
    final_answer = ""

    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=build_user_message(test_case))]},
            config={"recursion_limit": 15},
        )
        messages = result["messages"]
        tools_called = [m.name for m in messages if isinstance(m, ToolMessage)]
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                final_answer = m.content
                break
    except Exception as exc:  # noqa: BLE001 — on veut capturer et logguer toute erreur pour le rapport
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.time() - start, 2)
    expected = set(test_case["expected_tools"])
    actual = set(tools_called)

    # Efficacité : proportion des outils réellement appelés qui faisaient partie
    # des outils attendus (1.0 = aucun appel inutile, moins = du gaspillage).
    # Non défini (None) si aucun outil n'a été appelé du tout.
    tool_efficiency = round(len(expected & actual) / len(actual), 2) if actual else None

    return {
        "id": test_case["id"],
        "expected_tools": sorted(expected),
        "tools_called": tools_called,
        "tool_triggered": bool(actual),  # au moins un vrai outil a été appelé
        "correct_tool_used": bool(expected & actual),  # au moins un des outils attendus a été appelé (souple)
        "task_success": expected.issubset(actual),  # TOUS les outils attendus ont été appelés (strict)
        "tool_efficiency": tool_efficiency,
        "extra_tools_count": len(actual - expected),
        "elapsed_seconds": elapsed,
        "final_answer_preview": (final_answer[:160] + "…") if len(final_answer) > 160 else final_answer,
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    n_triggered = sum(r["tool_triggered"] for r in results)
    n_correct = sum(r["correct_tool_used"] for r in results)
    n_task_success = sum(r["task_success"] for r in results)
    n_errors = sum(r["error"] is not None for r in results)
    efficiencies = [r["tool_efficiency"] for r in results if r["tool_efficiency"] is not None]
    avg_efficiency = round(sum(efficiencies) / len(efficiencies), 2) if efficiencies else None
    avg_time = round(sum(r["elapsed_seconds"] for r in results) / n, 2) if n else 0
    return {
        "n_cases": n,
        "tool_triggered_rate": round(100 * n_triggered / n, 1) if n else 0,
        "correct_tool_rate": round(100 * n_correct / n, 1) if n else 0,
        "task_success_rate": round(100 * n_task_success / n, 1) if n else 0,
        "avg_tool_efficiency": avg_efficiency,
        "error_rate": round(100 * n_errors / n, 1) if n else 0,
        "avg_seconds_per_query": avg_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare la fiabilité du tool-calling entre modèles Ollama.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Modèles Ollama à tester")
    parser.add_argument("--test-cases", default=os.path.join(os.path.dirname(__file__), "test_cases.json"))
    parser.add_argument("--output", default=None, help="Chemin du fichier de résultats JSON (par défaut horodaté)")
    args = parser.parse_args()

    with open(args.test_cases, encoding="utf-8") as f:
        test_cases = json.load(f)

    all_results = {}
    for model_name in args.models:
        print(f"\n=== Modèle : {model_name} ===")
        try:
            agent_graph = build_graph(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  Impossible de construire l'agent pour {model_name}: {exc}")
            print(f"  (as-tu bien fait 'ollama pull {model_name}' ?)")
            continue

        results = []
        for tc in test_cases:
            r = run_one(agent_graph, tc)
            status = "✓" if r["correct_tool_used"] else ("✗ AUCUN OUTIL" if not r["tool_triggered"] else "✗ mauvais outil")
            print(f"  [{status}] {tc['id']} — outils appelés: {r['tools_called']} ({r['elapsed_seconds']}s)")
            results.append(r)

        summary = summarize(results)
        print(f"  → Taux de vrai tool-calling : {summary['tool_triggered_rate']}%")
        print(f"  → Taux de bon outil choisi  : {summary['correct_tool_rate']}% (souple — au moins 1 outil attendu)")
        print(f"  → Taux de succès de tâche   : {summary['task_success_rate']}% (strict — TOUS les outils attendus)")
        print(f"  → Efficacité outils (moy.)  : {summary['avg_tool_efficiency']}")
        print(f"  → Temps moyen / question    : {summary['avg_seconds_per_query']}s")

        all_results[model_name] = {"summary": summary, "details": results}

    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats détaillés sauvegardés dans : {output_path}")
    print("\n=== Résumé comparatif ===")
    print(f"{'Modèle':<22} {'Tool-calling':>13} {'Bon outil':>11} {'Succès tâche':>13} {'Efficacité':>11} {'Erreurs':>9} {'Temps moy.':>11}")
    for model_name, data in all_results.items():
        s = data["summary"]
        eff = s["avg_tool_efficiency"] if s["avg_tool_efficiency"] is not None else "N/A"
        print(f"{model_name:<22} {s['tool_triggered_rate']:>12}% {s['correct_tool_rate']:>10}% {s['task_success_rate']:>12}% {eff!s:>11} {s['error_rate']:>8}% {s['avg_seconds_per_query']:>10}s")


if __name__ == "__main__":
    main()
