"""
Tests unitaires des 9 outils de calcul financier (tools.py).

Couvre : le calcul correct en cas normal, ET les cas limites qui ont
provoqué de vrais bugs (division par zéro non protégée) dans
compute_liquidity_ratios, compute_leverage_ratios et analyze_trend.

Toutes les données sont des CSV temporaires (pytest tmp_path), jamais les
vrais fichiers du projet.

Lancer : pytest tests/test_tools.py -v
"""

import pandas as pd
import pytest

import tools
from tools import (
    analyze_trend,
    compare_financial_statements,
    compute_financial_health_score,
    compute_leverage_ratios,
    compute_liquidity_ratios,
    compute_profitability_ratios,
    compute_turnover_ratios,
    compute_working_capital_requirement,
)


@pytest.fixture(autouse=True)
def isolated_dataset_dir(tmp_path, monkeypatch):
    """Redirige DATASET_DIR vers un dossier temporaire pour CHAQUE test de ce fichier.

    Garantit qu'aucun test ne touche aux vrais CSV du projet (etats_financiers.csv,
    apple_reelles.csv, etc.) — chaque test crée ses propres données jetables.
    """
    monkeypatch.setattr(tools, "DATASET_DIR", tmp_path)
    return tmp_path


def write_csv(tmp_path, filename: str, rows: dict) -> str:
    """Écrit un CSV dans le dossier temporaire et retourne son nom (chemin relatif attendu par les outils)."""
    pd.DataFrame(rows).to_csv(tmp_path / filename, index=False)
    return filename


# --------------------------------------------------------------------------
# Données de référence (un cas "normal" réutilisé par plusieurs tests)
# --------------------------------------------------------------------------

NORMAL_ROWS = {
    "period": [2022, 2023],
    "revenue": [500000, 560000],
    "cogs": [300000, 330000],
    "net_income": [40000, 52000],
    "total_assets": [800000, 860000],
    "total_liabilities": [500000, 510000],
    "total_equity": [300000, 350000],
    "current_assets": [250000, 270000],
    "current_liabilities": [150000, 160000],
    "inventory": [60000, 65000],
    "accounts_receivable": [80000, 90000],
    "accounts_payable": [70000, 75000],
    "cash": [40000, 45000],
}


class TestProfitabilityRatios:
    def test_calcul_normal(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_profitability_ratios.invoke({"dataset_path": name})
        # Marge brute 2022 = (500000-300000)/500000 = 40.0%
        assert "marge brute=40.0%" in result
        # Marge nette 2022 = 40000/500000 = 8.0%
        assert "marge nette=8.0%" in result
        # ROE 2022 = 40000/300000 = 13.3%
        assert "ROE=13.3%" in result

    def test_revenue_zero_ne_plante_pas(self, isolated_dataset_dir):
        rows = {**NORMAL_ROWS, "revenue": [0, 560000]}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_profitability_ratios.invoke({"dataset_path": name})  # ne doit pas lever d'exception
        assert "2023" in result


class TestLiquidityRatios:
    def test_calcul_normal(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_liquidity_ratios.invoke({"dataset_path": name})
        # 2022 : 250000/150000 = 1.67
        assert "liquidité générale=1.67" in result

    def test_passif_circulant_zero_ne_plante_pas(self, isolated_dataset_dir):
        """Bug corrigé : current_liabilities=0 faisait planter (ZeroDivisionError)."""
        rows = {**NORMAL_ROWS, "current_liabilities": [0, 160000]}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_liquidity_ratios.invoke({"dataset_path": name})
        assert "non calculables" in result
        assert "2023 : liquidité générale=1.69" in result  # la 2e ligne reste calculée normalement


class TestLeverageRatios:
    def test_calcul_normal(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_leverage_ratios.invoke({"dataset_path": name})
        # 2022 : 500000/800000 = 0.62
        assert "ratio d'endettement=0.62" in result

    def test_actif_total_zero_ne_plante_pas(self, isolated_dataset_dir):
        """Bug corrigé : total_assets=0 faisait planter (ZeroDivisionError)."""
        rows = {**NORMAL_ROWS, "total_assets": [0, 860000]}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_leverage_ratios.invoke({"dataset_path": name})
        assert "ratio d'endettement=N/A" in result

    def test_capitaux_propres_zero_ne_plante_pas(self, isolated_dataset_dir):
        """Bug corrigé : total_equity=0 faisait planter (ZeroDivisionError)."""
        rows = {**NORMAL_ROWS, "total_equity": [0, 350000]}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_leverage_ratios.invoke({"dataset_path": name})
        assert "dette/capitaux propres=N/A" in result


class TestAnalyzeTrend:
    def test_calcul_normal(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = analyze_trend.invoke({"dataset_path": name})
        # CA : (560000-500000)/500000 = +12.0%
        assert "CA +12.0%" in result
        # résultat net : (52000-40000)/40000 = +30.0%
        assert "résultat net +30.0%" in result

    def test_une_seule_periode_message_clair(self, isolated_dataset_dir):
        rows = {k: v[:1] for k, v in NORMAL_ROWS.items()}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = analyze_trend.invoke({"dataset_path": name})
        assert "Au moins 2 périodes" in result

    def test_valeur_precedente_zero_ne_plante_pas(self, isolated_dataset_dir):
        """Bug corrigé : revenue ou net_income = 0 l'année précédente faisait planter."""
        rows = {**NORMAL_ROWS, "revenue": [0, 560000], "net_income": [0, 52000]}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = analyze_trend.invoke({"dataset_path": name})
        assert "CA N/A" in result
        assert "résultat net N/A" in result


class TestTurnoverRatios:
    def test_calcul_normal(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_turnover_ratios.invoke({"dataset_path": name})
        # rotation stocks 2022 = 60000/300000*365 = 73j
        assert "rotation stocks=73j" in result

    def test_sans_creances_fournisseurs_degrade_proprement(self, isolated_dataset_dir):
        """Sans accounts_receivable/payable, seule la rotation des stocks doit rester calculable."""
        rows = {k: v for k, v in NORMAL_ROWS.items() if k not in ("accounts_receivable", "accounts_payable")}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_turnover_ratios.invoke({"dataset_path": name})
        assert "rotation stocks=73j" in result
        assert "DSO" not in result
        assert "DPO" not in result


class TestWorkingCapitalRequirement:
    def test_formule_precise_avec_toutes_les_colonnes(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_working_capital_requirement.invoke({"dataset_path": name})
        # BFR 2022 = stocks + créances - dettes fournisseurs = 60000+80000-70000 = 70000
        assert "BFR (précis) = 70,000" in result

    def test_formule_approximative_sans_creances_fournisseurs(self, isolated_dataset_dir):
        rows = {k: v for k, v in NORMAL_ROWS.items() if k not in ("accounts_receivable", "accounts_payable")}
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_working_capital_requirement.invoke({"dataset_path": name})
        assert "BFR (approximatif)" in result


class TestFinancialHealthScore:
    def test_cas_sain_score_eleve(self, isolated_dataset_dir):
        name = write_csv(isolated_dataset_dir, "data.csv", NORMAL_ROWS)
        result = compute_financial_health_score.invoke({"dataset_path": name})
        assert "score global = 65/100 (Bon)" in result

    def test_cas_degrade_score_faible(self, isolated_dataset_dir):
        """Résultat net négatif + endettement extrême → score bas, verdict Critique."""
        rows = {
            "period": [2024], "revenue": [500000], "cogs": [470000], "net_income": [-20000],
            "total_assets": [800000], "total_liabilities": [750000], "total_equity": [50000],
            "current_assets": [100000], "current_liabilities": [180000], "inventory": [200000],
        }
        name = write_csv(isolated_dataset_dir, "data.csv", rows)
        result = compute_financial_health_score.invoke({"dataset_path": name})
        assert "Critique" in result


class TestCompareFinancialStatements:
    def test_comparaison_identifie_le_meilleur_score(self, isolated_dataset_dir):
        name_a = write_csv(isolated_dataset_dir, "entreprise_a.csv", NORMAL_ROWS)
        rows_b = {
            "period": [2024], "revenue": [500000], "cogs": [470000], "net_income": [-20000],
            "total_assets": [800000], "total_liabilities": [750000], "total_equity": [50000],
            "current_assets": [100000], "current_liabilities": [180000], "inventory": [200000],
        }
        name_b = write_csv(isolated_dataset_dir, "entreprise_b.csv", rows_b)

        result = compare_financial_statements.invoke({
            "dataset_path_a": name_a, "dataset_path_b": name_b,
        })
        assert "entreprise_a.csv a le meilleur score" in result
