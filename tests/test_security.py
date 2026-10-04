"""
Tests de sécurité pour tools.py — en particulier la protection contre le
path traversal sur dataset_path (voir _resolve_dataset_path dans tools.py).

Lancer : pytest tests/test_security.py -v
"""

import pandas as pd
import pytest

import tools
from tools import UnsafeDatasetPathError, _load_dataframe, _resolve_dataset_path


@pytest.fixture
def sandbox_dataset_dir(tmp_path, monkeypatch):
    """Isole DATASET_DIR dans un dossier temporaire pour ne pas dépendre de l'état réel du projet."""
    monkeypatch.setattr(tools, "DATASET_DIR", tmp_path)

    # Un CSV valide à l'intérieur du répertoire autorisé
    valid_csv = tmp_path / "etats_financiers.csv"
    pd.DataFrame({"period": [2024], "revenue": [100000], "net_income": [5000]}).to_csv(valid_csv, index=False)

    # Un fichier "secret" HORS du répertoire autorisé, pour vérifier qu'on ne peut pas l'atteindre
    secret_dir = tmp_path.parent / "secret_outside"
    secret_dir.mkdir(exist_ok=True)
    secret_csv = secret_dir / "donnees_privees.csv"
    pd.DataFrame({"secret": [1]}).to_csv(secret_csv, index=False)

    return tmp_path


class TestPathTraversalProtection:
    def test_chemin_relatif_valide_fonctionne(self, sandbox_dataset_dir):
        """Un chemin relatif normal, à l'intérieur du répertoire autorisé, doit fonctionner."""
        df = _load_dataframe("etats_financiers.csv")
        assert len(df) == 1
        assert "revenue" in df.columns

    def test_remontee_de_repertoire_bloquee(self, sandbox_dataset_dir):
        """'../secret_outside/donnees_privees.csv' doit être refusé, pas lu silencieusement."""
        with pytest.raises(UnsafeDatasetPathError):
            _load_dataframe("../secret_outside/donnees_privees.csv")

    def test_remontee_profonde_bloquee(self, sandbox_dataset_dir):
        """Plusieurs niveaux de '../../..' doivent aussi être bloqués."""
        with pytest.raises(UnsafeDatasetPathError):
            _load_dataframe("../../../../../../etc/passwd")

    def test_chemin_absolu_bloque(self, sandbox_dataset_dir, tmp_path):
        """Un chemin absolu fourni directement doit être refusé, même s'il pointe ailleurs."""
        absolute_outside = str(tmp_path.parent / "secret_outside" / "donnees_privees.csv")
        with pytest.raises(UnsafeDatasetPathError):
            _load_dataframe(absolute_outside)

    def test_sous_dossier_du_repertoire_autorise_fonctionne(self, sandbox_dataset_dir):
        """Un sous-dossier À L'INTÉRIEUR de DATASET_DIR doit rester autorisé (pas de faux positif)."""
        subdir = sandbox_dataset_dir / "clients"
        subdir.mkdir()
        csv_path = subdir / "client_x.csv"
        pd.DataFrame({"period": [2024], "revenue": [1]}).to_csv(csv_path, index=False)

        df = _load_dataframe("clients/client_x.csv")
        assert len(df) == 1

    def test_fichier_non_csv_refuse(self, sandbox_dataset_dir):
        """Une extension différente de .csv doit être refusée (indépendamment de la sécurité du chemin)."""
        txt_file = sandbox_dataset_dir / "notes.txt"
        txt_file.write_text("pas un csv")
        with pytest.raises(ValueError):
            _load_dataframe("notes.txt")

    def test_fichier_inexistant_message_clair(self, sandbox_dataset_dir):
        """Un fichier inexistant (mais chemin sûr) doit lever FileNotFoundError, pas planter ailleurs."""
        with pytest.raises(FileNotFoundError):
            _load_dataframe("ce_fichier_n_existe_pas.csv")
