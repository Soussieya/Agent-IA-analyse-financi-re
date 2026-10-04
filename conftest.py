"""
Permet à pytest de trouver tools.py / agent.py / main.py quand on lance
simplement "pytest" depuis la racine du projet (sans "python -m pytest").

Sans ce fichier, pytest n'ajoute que le dossier tests/ au chemin de
recherche Python, pas la racine du projet — d'où "ModuleNotFoundError:
No module named 'tools'". La présence de ce conftest.py à la racine fait
que pytest ajoute automatiquement la racine du projet (là où se trouve ce
fichier) au chemin de recherche, pour toute la session de test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
