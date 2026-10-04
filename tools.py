"""
Outils exposés à l'agent LangGraph — domaine : analyse d'états financiers
d'entreprise (bilan + compte de résultat).

Format de données attendu (CSV) : UNE LIGNE PAR PÉRIODE (année ou
trimestre), avec les postes comptables en colonnes. Exemple minimal :

    period,revenue,cogs,net_income,total_assets,total_liabilities,total_equity,current_assets,current_liabilities,inventory
    2022,500000,300000,40000,800000,500000,300000,250000,150000,60000
    2023,560000,330000,52000,860000,510000,350000,270000,160000,65000

Les noms de colonnes sont reconnus de façon flexible (français ou anglais,
avec ou sans underscore/espace) via un dictionnaire de synonymes — voir
STANDARD_FIELDS ci-dessous. Si une colonne nécessaire à un calcul est
introuvable, l'outil renvoie un message explicite plutôt que de planter,
pour que l'agent puisse le relayer à l'utilisateur.

5 outils de base :
  - inspect_financial_statement : structure des données, colonnes reconnues
  - compute_profitability_ratios : marge brute, marge nette, ROA, ROE
  - compute_liquidity_ratios     : ratio de liquidité générale et réduite
  - compute_leverage_ratios      : ratio d'endettement, dette/capitaux propres
  - analyze_trend                : croissance du CA et du résultat net (YoY)

2 outils étape 2 (rotation + BFR) :
  - compute_turnover_ratios              : rotation des stocks, délai clients, délai fournisseurs
  - compute_working_capital_requirement  : besoin en fonds de roulement (BFR)

1 outil étape 3 (synthèse) :
  - compute_financial_health_score : score global 0-100, agrégation pondérée et transparente
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

# --------------------------------------------------------------------------
# Sécurité : répertoire de données autorisé (anti path traversal)
# --------------------------------------------------------------------------
# Tous les chemins de dataset fournis par l'utilisateur (ou halluciné/choisi
# par le LLM) sont résolus PAR RAPPORT à ce répertoire, et on vérifie que le
# résultat reste bien à l'intérieur. Sans ça, un chemin comme
# "../../../Users/quelquun/document_prive.csv" serait lu sans broncher.
# Configurable via la variable d'environnement FINANCEAGENT_DATASET_DIR ;
# par défaut, le dossier du projet (où se trouve ce fichier).
DATASET_DIR = Path(
    os.environ.get("FINANCEAGENT_DATASET_DIR", os.path.dirname(os.path.abspath(__file__)))
).resolve()


class UnsafeDatasetPathError(PermissionError):
    """Levée quand un chemin de dataset sort du répertoire autorisé (DATASET_DIR)."""


def _resolve_dataset_path(dataset_path: str) -> Path:
    """Résout un chemin de dataset dans DATASET_DIR de manière sécurisée.

    Accepte par exemple :
    - "apple_reelles.csv"
    - "data/apple_reelles.csv"
    - "entreprise_b.csv"

    Les chemins absolus et les chemins sortant de DATASET_DIR sont refusés.
    """
    raw = Path(dataset_path)

    # Les chemins absolus sont interdits pour éviter l'accès à des fichiers
    # arbitraires sur la machine.
    if raw.is_absolute():
        raise UnsafeDatasetPathError(
            f"Chemin non autorisé : '{dataset_path}' est un chemin absolu. "
            f"Donne un chemin relatif au répertoire de données ({DATASET_DIR})."
        )

    # Si l'utilisateur fournit "data/fichier.csv" alors que DATASET_DIR
    # pointe déjà vers ".../data", on retire le préfixe "data/".
    parts = raw.parts
    if parts and parts[0].lower() == DATASET_DIR.name.lower():
        raw = Path(*parts[1:])

    candidate = (DATASET_DIR / raw).resolve()

    # Protection contre ../ et toute sortie du répertoire autorisé.
    try:
        candidate.relative_to(DATASET_DIR)
    except ValueError:
        raise UnsafeDatasetPathError(
            f"Chemin non autorisé : '{dataset_path}' sort du répertoire de données "
            f"autorisé ({DATASET_DIR})."
        )

    return candidate

# --------------------------------------------------------------------------
# Reconnaissance flexible des colonnes (français / anglais)
# --------------------------------------------------------------------------

STANDARD_FIELDS: dict[str, list[str]] = {
    "period": ["period", "periode", "année", "annee", "year", "exercice", "date"],
    "revenue": ["revenue", "chiffre affaires", "chiffre d affaires", "ca", "sales", "turnover", "produits"],
    "cogs": ["cogs", "cost of goods sold", "cout des ventes", "cout des biens vendus", "charges variables"],
    "net_income": ["net income", "resultat net", "benefice net", "profit net", "net profit"],
    "total_assets": ["total assets", "total actif", "actif total", "actifs totaux"],
    "total_liabilities": ["total liabilities", "total passif", "passif total", "dettes totales", "total dettes"],
    "total_equity": ["total equity", "capitaux propres", "equity", "fonds propres"],
    "current_assets": ["current assets", "actif circulant", "actifs courants"],
    "current_liabilities": ["current liabilities", "passif circulant", "dettes court terme", "passifs courants"],
    "inventory": ["inventory", "stocks", "stock"],
    "accounts_receivable": ["accounts receivable", "creances clients", "clients", "receivables"],
    "accounts_payable": ["accounts payable", "dettes fournisseurs", "fournisseurs", "payables"],
    "cash": ["cash", "tresorerie", "disponibilites", "cash and equivalents"],
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Retrouve la vraie colonne du DataFrame correspondant à un poste comptable standard."""
    synonyms = [_normalize(s) for s in STANDARD_FIELDS[field]]
    normalized_columns = {col: _normalize(col) for col in df.columns}
    for col, norm_col in normalized_columns.items():
        if norm_col in synonyms or any(s in norm_col for s in synonyms):
            return col
    return None


def _detect_fields(df: pd.DataFrame) -> dict[str, Optional[str]]:
    return {field: _find_column(df, field) for field in STANDARD_FIELDS}


def _load_dataframe(dataset_path: str) -> pd.DataFrame:
    resolved = _resolve_dataset_path(dataset_path)  # lève UnsafeDatasetPathError si hors DATASET_DIR
    if not resolved.exists():
        raise FileNotFoundError(
            f"Le fichier '{dataset_path}' est introuvable dans {DATASET_DIR}. "
            "Vérifie le chemin transmis par l'utilisateur."
        )
    if resolved.suffix.lower() != ".csv":
        raise ValueError("Seuls les fichiers .csv sont supportés dans ce squelette.")
    return pd.read_csv(resolved)


def _missing_fields_message(required: list[str], detected: dict[str, Optional[str]]) -> Optional[str]:
    missing = [f for f in required if detected.get(f) is None]
    if missing:
        return (
            "Impossible de calculer ce ratio : colonnes manquantes ou non reconnues pour "
            f"{', '.join(missing)}. Colonnes disponibles dans le fichier : "
            f"{', '.join(detected.get('_all_columns', []))}"
        )
    return None


# --------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------


@tool
def inspect_financial_statement(dataset_path: str) -> str:
    """Inspecte la structure d'un fichier d'états financiers CSV.

    Retourne le nombre de périodes couvertes et la correspondance entre
    les postes comptables standards (chiffre d'affaires, résultat net,
    total actif, etc.) et les colonnes réellement détectées dans le
    fichier. À utiliser en premier pour vérifier que les données sont
    exploitables avant de calculer des ratios.

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)

    lines = [
        f"Fichier: {os.path.basename(dataset_path)}",
        f"Nombre de périodes (lignes): {len(df)}",
        "",
        "Postes comptables détectés:",
    ]
    for field, col in detected.items():
        status = f"colonne '{col}'" if col else "NON DÉTECTÉ"
        lines.append(f"  - {field} : {status}")

    lines.append("")
    lines.append(f"Colonnes brutes du fichier: {', '.join(df.columns)}")

    return "\n".join(lines)


@tool
def compute_profitability_ratios(dataset_path: str) -> str:
    """Calcule les ratios de rentabilité pour chaque période.

    Marge brute = (Chiffre d'affaires - Coût des ventes) / Chiffre d'affaires
    Marge nette = Résultat net / Chiffre d'affaires
    ROA (Return on Assets) = Résultat net / Total actif
    ROE (Return on Equity) = Résultat net / Capitaux propres

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    period_col = detected["period"]
    lines = ["Ratios de rentabilité par période:"]

    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        parts = []

        revenue = row[detected["revenue"]] if detected["revenue"] else None
        cogs = row[detected["cogs"]] if detected["cogs"] else None
        net_income = row[detected["net_income"]] if detected["net_income"] else None
        total_assets = row[detected["total_assets"]] if detected["total_assets"] else None
        total_equity = row[detected["total_equity"]] if detected["total_equity"] else None

        if revenue and cogs is not None:
            parts.append(f"marge brute={((revenue - cogs) / revenue * 100):.1f}%")
        if revenue and net_income is not None:
            parts.append(f"marge nette={(net_income / revenue * 100):.1f}%")
        if total_assets and net_income is not None:
            parts.append(f"ROA={(net_income / total_assets * 100):.1f}%")
        if total_equity and net_income is not None:
            parts.append(f"ROE={(net_income / total_equity * 100):.1f}%")

        if not parts:
            return _missing_fields_message(
                ["revenue", "cogs", "net_income", "total_assets", "total_equity"], detected
            ) or "Aucune donnée exploitable trouvée."

        lines.append(f"  {label} : " + ", ".join(parts))

    return "\n".join(lines)


@tool
def compute_liquidity_ratios(dataset_path: str) -> str:
    """Calcule les ratios de liquidité pour chaque période.

    Ratio de liquidité générale = Actif circulant / Passif circulant
    Ratio de liquidité réduite = (Actif circulant - Stocks) / Passif circulant

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    missing_msg = _missing_fields_message(["current_assets", "current_liabilities"], detected)
    if missing_msg:
        return missing_msg

    period_col = detected["period"]
    lines = ["Ratios de liquidité par période:"]

    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        current_assets = row[detected["current_assets"]]
        current_liabilities = row[detected["current_liabilities"]]
        inventory = row[detected["inventory"]] if detected["inventory"] else 0

        if current_liabilities == 0:
            lines.append(f"  {label} : passif circulant nul — ratios de liquidité non calculables (division par zéro)")
            continue

        current_ratio = current_assets / current_liabilities
        quick_ratio = (current_assets - inventory) / current_liabilities

        lines.append(
            f"  {label} : liquidité générale={current_ratio:.2f}, liquidité réduite={quick_ratio:.2f}"
        )

    return "\n".join(lines)


@tool
def compute_leverage_ratios(dataset_path: str) -> str:
    """Calcule les ratios d'endettement pour chaque période.

    Ratio d'endettement = Total passif / Total actif
    Ratio dette / capitaux propres = Total passif / Capitaux propres

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    missing_msg = _missing_fields_message(["total_liabilities", "total_assets", "total_equity"], detected)
    if missing_msg:
        return missing_msg

    period_col = detected["period"]
    lines = ["Ratios d'endettement par période:"]

    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        total_liabilities = row[detected["total_liabilities"]]
        total_assets = row[detected["total_assets"]]
        total_equity = row[detected["total_equity"]]

        parts = []
        if total_assets == 0:
            parts.append("ratio d'endettement=N/A (actif total nul)")
        else:
            parts.append(f"ratio d'endettement={total_liabilities / total_assets:.2f}")

        if total_equity == 0:
            parts.append("dette/capitaux propres=N/A (capitaux propres nuls)")
        else:
            parts.append(f"dette/capitaux propres={total_liabilities / total_equity:.2f}")

        lines.append(f"  {label} : " + ", ".join(parts))

    return "\n".join(lines)


@tool
def analyze_trend(dataset_path: str) -> str:
    """Analyse la croissance du chiffre d'affaires et du résultat net entre périodes.

    Calcule la variation en pourcentage d'une période à l'autre (croissance
    année sur année) pour repérer une tendance d'amélioration ou de
    dégradation. Nécessite au moins 2 périodes dans le fichier.

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    if len(df) < 2:
        return "Au moins 2 périodes sont nécessaires pour analyser une tendance."

    missing_msg = _missing_fields_message(["revenue", "net_income"], detected)
    if missing_msg:
        return missing_msg

    period_col = detected["period"]
    revenue_col = detected["revenue"]
    net_income_col = detected["net_income"]

    lines = ["Évolution période sur période:"]
    prev_row = None
    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        if prev_row is not None:
            prev_revenue = prev_row[revenue_col]
            prev_net_income = prev_row[net_income_col]

            rev_part = "CA N/A (valeur nulle l'année précédente)" if prev_revenue == 0 else (
                f"CA {(row[revenue_col] - prev_revenue) / prev_revenue * 100:+.1f}%"
            )
            ni_part = "résultat net N/A (valeur nulle l'année précédente)" if prev_net_income == 0 else (
                f"résultat net {(row[net_income_col] - prev_net_income) / prev_net_income * 100:+.1f}%"
            )
            lines.append(f"  {label} : {rev_part}, {ni_part}")
        prev_row = row

    return "\n".join(lines)


@tool
def compute_turnover_ratios(dataset_path: str) -> str:
    """Calcule les ratios de rotation pour chaque période.

    Rotation des stocks (jours) = Stocks / Coût des ventes * 365
    Délai de paiement clients (DSO, jours) = Créances clients / Chiffre d'affaires * 365
    Délai de paiement fournisseurs (DPO, jours) = Dettes fournisseurs / Coût des ventes * 365

    Ces ratios mesurent la vitesse à laquelle l'entreprise transforme ses
    stocks en ventes, encaisse ses créances clients, et paie ses
    fournisseurs. Nécessite les colonnes stocks, créances clients et
    dettes fournisseurs (accounts_receivable / accounts_payable).

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    period_col = detected["period"]
    lines = ["Ratios de rotation par période (en jours):"]

    any_computed = False
    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        parts = []

        cogs = row[detected["cogs"]] if detected["cogs"] else None
        revenue = row[detected["revenue"]] if detected["revenue"] else None
        inventory = row[detected["inventory"]] if detected["inventory"] else None
        receivables = row[detected["accounts_receivable"]] if detected["accounts_receivable"] else None
        payables = row[detected["accounts_payable"]] if detected["accounts_payable"] else None

        if inventory is not None and cogs:
            parts.append(f"rotation stocks={(inventory / cogs * 365):.0f}j")
        if receivables is not None and revenue:
            parts.append(f"délai clients (DSO)={(receivables / revenue * 365):.0f}j")
        if payables is not None and cogs:
            parts.append(f"délai fournisseurs (DPO)={(payables / cogs * 365):.0f}j")

        if not parts:
            continue
        any_computed = True
        lines.append(f"  {label} : " + ", ".join(parts))

    if not any_computed:
        return _missing_fields_message(
            ["cogs", "revenue", "inventory", "accounts_receivable", "accounts_payable"], detected
        ) or "Aucune donnée exploitable trouvée."

    return "\n".join(lines)


@tool
def compute_working_capital_requirement(dataset_path: str) -> str:
    """Calcule le besoin en fonds de roulement (BFR) pour chaque période.

    Formule privilégiée (BFR d'exploitation) :
        BFR = Stocks + Créances clients - Dettes fournisseurs

    Si les créances clients / dettes fournisseurs ne sont pas disponibles,
    utilise une formule approximative à partir du bilan :
        BFR ≈ (Actif circulant - Trésorerie) - Passif circulant

    Un BFR positif signifie que l'entreprise doit financer un décalage de
    trésorerie lié à son cycle d'exploitation (stocks + créances non
    encore encaissées, moins ce qu'elle doit à ses fournisseurs).

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    detected["_all_columns"] = list(df.columns)

    period_col = detected["period"]
    lines = ["Besoin en fonds de roulement (BFR) par période:"]

    has_precise = detected["inventory"] and detected["accounts_receivable"] and detected["accounts_payable"]
    has_approx = detected["current_assets"] and detected["current_liabilities"]

    if not has_precise and not has_approx:
        return _missing_fields_message(
            ["inventory", "accounts_receivable", "accounts_payable"], detected
        )

    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"

        if has_precise:
            inventory = row[detected["inventory"]]
            receivables = row[detected["accounts_receivable"]]
            payables = row[detected["accounts_payable"]]
            bfr = inventory + receivables - payables
            lines.append(f"  {label} : BFR (précis) = {bfr:,.0f}")
        else:
            current_assets = row[detected["current_assets"]]
            current_liabilities = row[detected["current_liabilities"]]
            cash = row[detected["cash"]] if detected["cash"] else 0
            bfr_approx = (current_assets - cash) - current_liabilities
            lines.append(f"  {label} : BFR (approximatif) = {bfr_approx:,.0f}")

    return "\n".join(lines)


def _score_bucket(value: float, thresholds: list[tuple[float, int]], higher_is_better: bool) -> int:
    """thresholds: liste de (seuil, score) triée du meilleur au pire seuil."""
    for threshold, score in thresholds:
        if (higher_is_better and value >= threshold) or (not higher_is_better and value <= threshold):
            return score
    return 0


def _health_components_for_row(row: pd.Series, detected: dict) -> list[tuple[str, int]]:
    """Calcule les composantes du score de santé (rentabilité/liquidité/endettement/rotation) pour une ligne."""
    components: list[tuple[str, int]] = []

    revenue = row[detected["revenue"]] if detected["revenue"] else None
    net_income = row[detected["net_income"]] if detected["net_income"] else None
    current_assets = row[detected["current_assets"]] if detected["current_assets"] else None
    current_liabilities = row[detected["current_liabilities"]] if detected["current_liabilities"] else None
    total_assets = row[detected["total_assets"]] if detected["total_assets"] else None
    total_liabilities = row[detected["total_liabilities"]] if detected["total_liabilities"] else None
    inventory = row[detected["inventory"]] if detected["inventory"] else None
    cogs = row[detected["cogs"]] if detected["cogs"] else None

    if revenue and net_income is not None:
        net_margin = net_income / revenue * 100
        components.append(("rentabilité", _score_bucket(
            net_margin, [(15, 100), (10, 80), (5, 60), (0, 40), (-1e9, 0)], higher_is_better=True
        )))

    if current_liabilities:
        current_ratio = current_assets / current_liabilities if current_assets else 0
        components.append(("liquidité", _score_bucket(
            current_ratio, [(2, 100), (1.5, 80), (1, 60), (0.5, 40), (0, 0)], higher_is_better=True
        )))

    if total_assets:
        debt_ratio = (total_liabilities / total_assets * 100) if total_liabilities is not None else 0
        components.append(("endettement", _score_bucket(
            debt_ratio, [(30, 100), (50, 80), (70, 60), (90, 40), (1e9, 0)], higher_is_better=False
        )))

    if inventory is not None and cogs:
        days_inventory = inventory / cogs * 365
        components.append(("rotation stocks", _score_bucket(
            days_inventory, [(30, 100), (60, 80), (90, 60), (120, 40), (1e9, 0)], higher_is_better=False
        )))

    return components


def _verdict_for_score(score: int) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Bon"
    elif score >= 40:
        return "Moyen"
    elif score >= 20:
        return "Fragile"
    return "Critique"


@tool
def compute_financial_health_score(dataset_path: str) -> str:
    """Calcule un score de santé financière global (0-100) pour chaque période.

    Agrège 4 composantes, chacune notée sur 100 selon des seuils usuels,
    puis fait la moyenne des composantes disponibles :
      - Rentabilité (marge nette) : >15%=100, 10-15%=80, 5-10%=60, 0-5%=40, <0%=0
      - Liquidité (ratio de liquidité générale) : >2=100, 1.5-2=80, 1-1.5=60, 0.5-1=40, <0.5=0
      - Endettement (dettes/actif, plus bas = mieux) : <30%=100, 30-50%=80, 50-70%=60, 70-90%=40, >90%=0
      - Rotation des stocks (jours, plus bas = mieux) : <30j=100, 30-60j=80, 60-90j=60, 90-120j=40, >120j=0

    Le calcul est transparent (pas de boîte noire) : la sortie détaille
    chaque composante utilisée. Si une composante manque (colonnes
    absentes), elle est simplement exclue de la moyenne, sans pénaliser
    le score.

    Args:
        dataset_path: chemin vers le fichier CSV des états financiers.
    """
    df = _load_dataframe(dataset_path)
    detected = _detect_fields(df)
    period_col = detected["period"]

    lines = ["Score de santé financière par période (0-100):"]

    for idx, row in df.iterrows():
        label = row[period_col] if period_col else f"ligne {idx}"
        components = _health_components_for_row(row, detected)

        if not components:
            lines.append(f"  {label} : impossible à calculer (aucun poste comptable exploitable)")
            continue

        global_score = round(sum(s for _, s in components) / len(components))
        verdict = _verdict_for_score(global_score)

        detail = ", ".join(f"{name}={score}/100" for name, score in components)
        lines.append(f"  {label} : score global = {global_score}/100 ({verdict}) — détail: {detail}")

    return "\n".join(lines)


@tool
def compare_financial_statements(dataset_path_a: str, dataset_path_b: str) -> str:
    """Compare deux entreprises (ou deux exercices dans des fichiers séparés).

    Prend la période la plus récente (dernière ligne) de chaque fichier et
    affiche côte à côte : chiffre d'affaires, marge nette, ratio de
    liquidité générale, ratio d'endettement, et score de santé financière
    global — avec l'écart entre les deux.

    Utilise cet outil quand l'utilisateur demande de comparer deux
    entreprises ou deux fichiers financiers différents (pas deux périodes
    du même fichier — pour ça, utilise analyze_trend).

    Args:
        dataset_path_a: chemin vers le premier fichier CSV.
        dataset_path_b: chemin vers le second fichier CSV.
    """
    def summarize(path: str) -> dict:
        df = _load_dataframe(path)
        detected = _detect_fields(df)
        row = df.iloc[-1]  # période la plus récente = dernière ligne
        label = row[detected["period"]] if detected["period"] else "dernière période"

        revenue = row[detected["revenue"]] if detected["revenue"] else None
        net_income = row[detected["net_income"]] if detected["net_income"] else None
        current_assets = row[detected["current_assets"]] if detected["current_assets"] else None
        current_liabilities = row[detected["current_liabilities"]] if detected["current_liabilities"] else None
        total_assets = row[detected["total_assets"]] if detected["total_assets"] else None
        total_liabilities = row[detected["total_liabilities"]] if detected["total_liabilities"] else None

        net_margin = (net_income / revenue * 100) if revenue and net_income is not None else None
        current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities else None
        debt_ratio = (total_liabilities / total_assets * 100) if total_assets and total_liabilities is not None else None

        components = _health_components_for_row(row, detected)
        health_score = round(sum(s for _, s in components) / len(components)) if components else None

        return {
            "file": os.path.basename(path),
            "period": label,
            "revenue": revenue,
            "net_margin": net_margin,
            "current_ratio": current_ratio,
            "debt_ratio": debt_ratio,
            "health_score": health_score,
        }

    a = summarize(dataset_path_a)
    b = summarize(dataset_path_b)

    def fmt(value, suffix=""):
        return f"{value:.1f}{suffix}" if value is not None else "N/A"

    def delta(value_a, value_b, suffix=""):
        if value_a is None or value_b is None:
            return "N/A"
        d = value_a - value_b
        return f"{d:+.1f}{suffix}"

    lines = [
        f"Comparaison : {a['file']} (période {a['period']}) vs {b['file']} (période {b['period']})",
        "",
        f"{'Indicateur':<22} {'Fichier A':>15} {'Fichier B':>15} {'Écart (A-B)':>15}",
        f"{'Chiffre affaires':<22} {fmt(a['revenue']):>15} {fmt(b['revenue']):>15} {delta(a['revenue'], b['revenue']):>15}",
        f"{'Marge nette':<22} {fmt(a['net_margin'], '%'):>15} {fmt(b['net_margin'], '%'):>15} {delta(a['net_margin'], b['net_margin'], 'pts'):>15}",
        f"{'Liquidité générale':<22} {fmt(a['current_ratio']):>15} {fmt(b['current_ratio']):>15} {delta(a['current_ratio'], b['current_ratio']):>15}",
        f"{'Endettement':<22} {fmt(a['debt_ratio'], '%'):>15} {fmt(b['debt_ratio'], '%'):>15} {delta(a['debt_ratio'], b['debt_ratio'], 'pts'):>15}",
        f"{'Score santé (0-100)':<22} {fmt(a['health_score']):>15} {fmt(b['health_score']):>15} {delta(a['health_score'], b['health_score']):>15}",
    ]

    if a["health_score"] is not None and b["health_score"] is not None:
        winner = a["file"] if a["health_score"] > b["health_score"] else (
            b["file"] if b["health_score"] > a["health_score"] else "égalité"
        )
        lines.append("")
        lines.append(f"Conclusion : {winner} a le meilleur score de santé financière globale.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Outil RAG : recherche dans la base de connaissances (glossaire + repères
# sectoriels), pour les questions conceptuelles plutôt que calculatoires.
# --------------------------------------------------------------------------

_RAG_VECTORSTORE = None  # chargé une seule fois, en différé (lazy)
_RAG_UNAVAILABLE_REASON: Optional[str] = None


def _get_rag_vectorstore():
    """Charge l'index vectoriel Chroma en différé, une seule fois par processus."""
    global _RAG_VECTORSTORE, _RAG_UNAVAILABLE_REASON
    if _RAG_VECTORSTORE is not None or _RAG_UNAVAILABLE_REASON is not None:
        return _RAG_VECTORSTORE

    persist_dir = os.path.join(os.path.dirname(__file__), "rag", "chroma_db")
    if not os.path.isdir(persist_dir):
        _RAG_UNAVAILABLE_REASON = (
            "Index de connaissances introuvable. Lance d'abord "
            "'python rag/build_index.py' (nécessite Ollama démarré)."
        )
        return None

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        ollama_base_url = os.environ.get("OLLAMA_BASE_URL") or None
        _RAG_VECTORSTORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url),
            collection_name="financeagent_knowledge",
        )
    except Exception as exc:  # noqa: BLE001 — on veut un message utile pour l'agent, pas un crash
        _RAG_UNAVAILABLE_REASON = f"Base de connaissances indisponible ({type(exc).__name__}: {exc})."
        return None

    return _RAG_VECTORSTORE


@tool
def search_financial_knowledge(query: str) -> str:
    """Recherche dans une base de connaissances financières (glossaire + repères sectoriels).

    À utiliser pour les questions CONCEPTUELLES — définitions de ratios,
    ce qu'un chiffre signifie, ou si un ratio est normal pour un secteur
    d'activité donné (ex: "qu'est-ce que le BFR ?", "un ratio de liquidité
    de 0.8 est-il normal dans la grande distribution ?"). Ne pas utiliser
    pour calculer un ratio à partir d'un fichier — pour ça, utilise les
    autres outils (compute_profitability_ratios, etc.).

    Args:
        query: la question ou le terme à rechercher (en français).
    """
    vectorstore = _get_rag_vectorstore()
    if vectorstore is None:
        return _RAG_UNAVAILABLE_REASON

    try:
        results = vectorstore.similarity_search(query, k=3)
    except Exception as exc:  # noqa: BLE001 — ex: Ollama non démarré au moment de la requête
        return f"Base de connaissances momentanément indisponible ({type(exc).__name__}: {exc})."

    if not results:
        return "Aucune information pertinente trouvée dans la base de connaissances."

    lines = ["Extraits pertinents de la base de connaissances :"]
    for doc in results:
        lines.append(f"\n[{doc.metadata.get('source', '?')} — {doc.metadata.get('section', '?')}]")
        lines.append(doc.page_content)

    return "\n".join(lines)


# Liste utilisée par agent.py pour binder les outils au LLM et construire le ToolNode
ALL_TOOLS = [
    inspect_financial_statement,
    compute_profitability_ratios,
    compute_liquidity_ratios,
    compute_leverage_ratios,
    analyze_trend,
    compute_turnover_ratios,
    compute_working_capital_requirement,
    compute_financial_health_score,
    compare_financial_statements,
    search_financial_knowledge,
]
