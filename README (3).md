# FinanceAgent

**Agent IA local qui analyse des états financiers d'entreprise en langage naturel** — 100% sur ta machine, aucune donnée envoyée à un cloud, zéro coût d'API.

Pose une question comme *« calcule le score de santé financière de cette entreprise »* ou *« qu'est-ce que le BFR ? »* → l'agent choisit lui-même le bon outil parmi 10 (9 calculs financiers + 1 recherche documentaire), l'exécute, et répond en français.

Dépôt : https://github.com/Soussieya/Agent-IA-analyse-financi-re

<!-- 📸 Ajoute ici une capture d'écran ou un GIF de dashboard.html en action -->

## Pourquoi ce projet

- **100% local** : LLM servi par [Ollama](https://ollama.com), aucune donnée financière ne quitte la machine — argument pertinent pour des données sensibles.
- **Les calculs ne sont jamais faits par le LLM** : tout ratio financier est du code Python déterministe (`tools.py`). Le LLM ne fait qu'une chose — décider quel outil appeler — ce qui rend le système auditable et fiable.
- **Fiabilité mesurée, pas supposée** : voir [Résultats du benchmark](#résultats-du-benchmark-de-fiabilité) ci-dessous.
- **Agentique ET RAG** : deux briques IA distinctes dans un seul projet — appel d'outils pour le calcul, recherche documentaire (Chroma + embeddings locaux) pour l'explication conceptuelle et le contexte sectoriel.

## Architecture

```
Utilisateur (dashboard.html ou requête HTTP)
          │
          ▼
   FastAPI (main.py) ──── POST /agent/query
          │
          ▼
   Agent LangGraph (agent.py)
          │
   ┌──────┴──────┐
   │  boucle:    │   décision → outil → observation → nouvelle décision
   │  LLM ⇄ tools│
   └──────┬──────┘
          │
    ┌─────┴──────────────────┐
    ▼                         ▼
9 outils financiers      search_financial_knowledge
(tools.py, calculs        │
Python purs sur CSV)      ▼
                     Chroma + embeddings Ollama
                     (glossaire + repères sectoriels)
```

## Résultats du benchmark de fiabilité

Un problème découvert en développant ce projet : certains modèles locaux *déclarent* supporter le tool-calling (`"tools"` dans `ollama show`) mais, en pratique, **hallucinent une réponse complète au lieu d'exécuter le calcul réel**. Plutôt que de le constater une fois et de l'ignorer, ce comportement a été quantifié avec un mini-benchmark reproductible (`evals/`) : 15 questions financières, chacune avec l'outil attendu connu à l'avance, testées sur 3 modèles locaux différents.

| Modèle                  | Vrai tool-calling | Bon outil choisi | Temps moyen/question | Constat |
|---------------------------|:------------------:|:------------------:|:-----------------------:|---------|
| `mistral:7b-instruct`     | 0%                  | 0%                  | 33.2s                    | Déclare "tools" dans Ollama mais hallucine systématiquement une réponse au lieu d'appeler un outil |
| `llama3.1:8b`              | **100%**            | **100%**            | 19.8s                    | Fiable sur les 15 cas, y compris les questions formulées indirectement et les cas multi-outils |
| `deepseek-coder:6.7b`      | 0%                  | 0%                  | 0.02s                    | Rejeté immédiatement par Ollama (`does not support tools`, HTTP 400) — modèle de génération de code, pas de function-calling |

**Reproductible** : `python evals/run_eval.py --models <modèle1> <modèle2> ...` (jeu de test dans `evals/test_cases.json`). Le harnais mesure aussi le taux de succès de tâche strict (tous les outils attendus appelés, pas juste un) et l'efficacité (proportion d'appels d'outils réellement utiles, sans gaspillage).

Conclusion pratique : *déclarer* la capacité "tools" ne garantit pas de *savoir* l'utiliser. `llama3.1:8b` est le modèle retenu par défaut pour cette raison.

## Fonctionnalités

9 outils de calcul + 1 outil de recherche documentaire, tous accessibles en langage naturel via l'agent :

| Outil                                | Ce qu'il fait |
|----------------------------------------|-------------------|
| `inspect_financial_statement`           | Structure du fichier, colonnes reconnues |
| `compute_profitability_ratios`          | Marge brute, marge nette, ROA, ROE |
| `compute_liquidity_ratios`              | Liquidité générale et réduite |
| `compute_leverage_ratios`               | Ratio d'endettement, dette/capitaux propres |
| `analyze_trend`                          | Croissance CA et résultat net (année sur année) |
| `compute_turnover_ratios`               | Rotation stocks, délai clients (DSO), délai fournisseurs (DPO) |
| `compute_working_capital_requirement`   | Besoin en fonds de roulement (BFR), précis ou approximatif |
| `compute_financial_health_score`        | Score global 0-100, transparent (détail par composante) |
| `compare_financial_statements`          | Comparaison chiffrée entre deux entreprises/fichiers |
| `search_financial_knowledge`            | Recherche RAG : définitions de ratios, repères sectoriels |

Testé avec des données réelles (comptes Apple Inc. FY2022-2024, `apple_reelles.csv`) en plus de CSV factices.

## Base de connaissances (RAG)

`search_financial_knowledge` interroge une base vectorielle locale (Chroma) construite à partir de deux documents :
- `rag/knowledge/glossaire.md` — définition de chaque ratio en langage clair (12 sections).
- `rag/knowledge/reperes_sectoriels.md` — ordres de grandeur typiques par secteur (grande distribution, industrie, tech/SaaS, services), pour contextualiser un ratio ("0.9 de liquidité est-il normal ?").

⚠️ Les repères sectoriels sont des ordres de grandeur **pédagogiques et illustratifs**, pas des statistiques officielles sourcées — précisé explicitement dans le document lui-même. Pour un usage professionnel réel, les remplacer par des données sectorielles officielles (Damodaran, INSEE, Banque de France).

Les embeddings sont calculés localement avec `nomic-embed-text` via Ollama (aucun appel externe). Construction de l'index : `python rag/build_index.py` (une seule fois, à relancer seulement si les `.md` changent).

## Démarrage rapide

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
python rag/build_index.py
uvicorn main:app --reload --port 8000
```

Puis ouvre `dashboard.html` dans un navigateur (aucun serveur web requis pour ce fichier), ou envoie une requête directement :

```powershell
$body = @{
  query = "Calcule le score de santé financière de l'entreprise"
  dataset_path = "etats_financiers.csv"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/agent/query" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
$response | ConvertTo-Json -Depth 6
```

Comparer deux entreprises : ajoute `dataset_path_b = "entreprise_b.csv"` au corps de la requête.

## Format de données attendu

Un CSV avec **une ligne par période** (année ou trimestre), postes comptables en colonnes, noms reconnus de façon flexible (français ou anglais) :

```csv
period,revenue,cogs,net_income,total_assets,total_liabilities,total_equity,current_assets,current_liabilities,inventory,accounts_receivable,accounts_payable,cash
2022,500000,300000,40000,800000,500000,300000,250000,150000,60000,80000,70000,40000
2023,560000,330000,52000,860000,510000,350000,270000,160000,65000,90000,75000,45000
2024,610000,350000,58000,900000,500000,400000,290000,155000,70000,95000,80000,50000
```

Les 3 dernières colonnes (créances clients, dettes fournisseurs, trésorerie) sont optionnelles — elles activent les ratios de rotation précis et le BFR précis ; sans elles, ces outils basculent automatiquement sur des formules approximatives plutôt que de planter.

| Poste standard        | Synonymes reconnus (extrait)              |
|--------------------------|----------------------------------------------|
| `revenue`                 | chiffre d'affaires, ca, sales, turnover        |
| `cogs`                    | coût des ventes, cost of goods sold            |
| `net_income`               | résultat net, bénéfice net, net profit         |
| `total_assets`             | total actif, actif total                       |
| `total_liabilities`        | total passif, dettes totales                   |
| `total_equity`              | capitaux propres, fonds propres                |
| `current_assets`            | actif circulant, actifs courants               |
| `current_liabilities`       | passif circulant, dettes court terme           |
| `inventory`                 | stocks                                        |
| `accounts_receivable`       | créances clients, receivables                  |
| `accounts_payable`          | dettes fournisseurs, payables                   |
| `cash`                      | trésorerie, disponibilités                     |

## Structure du projet

```
.
├── main.py                  # API FastAPI — route POST /agent/query
├── agent.py                  # Graphe LangGraph (décision ⇄ outils), modèle paramétrable
├── tools.py                   # 9 outils financiers + outil RAG — calculs Python purs
├── dashboard.html              # Interface web autonome (aucun serveur requis)
├── requirements.txt
├── rag/
│   ├── build_index.py           # Construction de l'index vectoriel Chroma
│   ├── knowledge/
│   │   ├── glossaire.md           # Définitions des ratios
│   │   └── reperes_sectoriels.md  # Ordres de grandeur par secteur
│   └── chroma_db/                 # Index persisté (généré, pas versionné)
├── evals/
│   ├── test_cases.json          # 15 questions avec outil attendu connu
│   └── run_eval.py               # Harnais de comparaison entre modèles
└── *.csv                        # Jeux de données d'exemple (factices + réelles)
```

## Limites connues

- Le LLM reformule parfois mal les chiffres exacts en résumant (prompt ajusté pour limiter ça, voir `agent.py`) — le détail brut de chaque outil reste la source de vérité, toujours affiché.
- Les repères sectoriels du RAG sont illustratifs, pas une base de données officielle sourcée (voir avertissement dans `reperes_sectoriels.md`).
- Le score de santé financière utilise des seuils génériques par défaut ; le RAG permet de les contextualiser en conversation, mais ne les ajuste pas automatiquement dans le calcul lui-même.

## Pistes d'évolution

- Faire lire le score par le secteur détecté automatiquement (croiser `compute_financial_health_score` et `search_financial_knowledge`).
- Étendre le benchmark à plus de modèles (Qwen2.5, Gemma3) et mesurer aussi la qualité du RAG (precision/recall de la recherche).
- Dockerisation pour un déploiement reproductible.
