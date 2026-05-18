"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MAIN — Pipeline Complet SMA Revue de Littérature              ║
║                                                                              ║
║  Orchestre les 6 agents dans l'ordre :                                       ║
║    1. Agent Curateur     → parsing PDF, extraction métadonnées               ║
║    2. Agent Indexeur     → chunking + embeddings + ChromaDB                  ║
║    3. Agent Cartographe  → clustering thématique + carte du corpus           ║
║    4. Agent Narrateur    → rédaction de la revue de littérature              ║
║    5. Agent Détecteur    → identification des research gaps                  ║
║    6. Agent Citateur     → vérification des citations + bibliographie        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage :
    python main.py                          # pipeline complet
    python main.py --etape curateur         # une seule étape
    python main.py --etape indexeur
    python main.py --etape cartographe
    python main.py --etape narrateur
    python main.py --etape detecteur
    python main.py --etape citateur
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path

# ── Ajout du dossier agents au path ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = BASE_DIR / "agents"

# Chemins absolus vers chaque dossier d'agent
paths_to_add = [
    str(AGENTS_DIR / "curateur"),
    str(AGENTS_DIR / "indexeur"),
    str(AGENTS_DIR / "cartographe"),
    str(AGENTS_DIR / "narrateur"),
    str(AGENTS_DIR / "detecteur"),
    str(AGENTS_DIR / "citateur"),
]

for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Chemins par défaut ───────────────────────────────────────────────────────
DOSSIER_PDFS    = str(BASE_DIR / "data" / "articles")
DOSSIER_CORPUS  = str(BASE_DIR / "data" / "corpus")
DOSSIER_CHROMA  = str(BASE_DIR / "data" / "chroma_db")

CORPUS_JSON     = str(BASE_DIR / "data" / "corpus" / "corpus_complet.json")
CARTE_JSON      = str(BASE_DIR / "data" / "corpus" / "carte_corpus.json")
REVUE_JSON      = str(BASE_DIR / "data" / "corpus" / "revue_litterature.json")
REVUE_MD        = str(BASE_DIR / "data" / "corpus" / "revue_litterature.md")
GAPS_JSON       = str(BASE_DIR / "data" / "corpus" / "gaps_detectes.json")


def banner():
    print("\n" + "=" * 70)
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║     SMA — Système Multi-Agents Revue de Littérature        ║")
    print("  ║                   Pipeline Complet                          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print("=" * 70)

    # Initialisation des dossiers
    for d in [DOSSIER_PDFS, DOSSIER_CORPUS, DOSSIER_CHROMA]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 : AGENT CURATEUR
# ═════════════════════════════════════════════════════════════════════════════

def etape_curateur():
    import os
    path_curateur = Path(AGENTS_DIR / "curateur" / "agent_curateur.py")
    if not path_curateur.exists():
        print(f"  ✗ ERREUR : {path_curateur} est introuvable !")
    
    try:
        from agent_curateur import AgentCurateur
    except ImportError as e:
        print(f"  ✗ ERREUR d'importation : {e}")
        print(f"  sys.path actuel : {sys.path[:3]}")
        raise e

    curateur = AgentCurateur()
    corpus = curateur.run(
        dossier_pdfs=DOSSIER_PDFS,
        dossier_output=DOSSIER_CORPUS
    )
    return corpus


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 : AGENT INDEXEUR
# ═════════════════════════════════════════════════════════════════════════════

def etape_indexeur():
    from agent_indexeur import AgentIndexeur

    indexeur = AgentIndexeur(
        dossier_chroma=DOSSIER_CHROMA,
        taille_chunk=400,
        overlap=80,
        batch_size=32
    )

    rapport = indexeur.run(
        chemin_corpus=CORPUS_JSON,
        dossier_output=DOSSIER_CORPUS
    )
    return indexeur, rapport


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 : AGENT CARTOGRAPHE
# ═════════════════════════════════════════════════════════════════════════════

def etape_cartographe(indexeur=None):
    from agent_cartographe import AgentCartographe

    cartographe = AgentCartographe()

    if indexeur:
        carte = cartographe.run(
            dossier_output=DOSSIER_CORPUS,
            indexeur=indexeur
        )
    else:
        carte = cartographe.run(
            dossier_output=DOSSIER_CORPUS,
            chemin_corpus=CORPUS_JSON
        )
    return carte


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 4 : AGENT NARRATEUR
# ═════════════════════════════════════════════════════════════════════════════

def etape_narrateur(indexeur=None):
    from agent_narrateur import AgentNarrateur

    narrateur = AgentNarrateur()
    revue = narrateur.run(
        chemin_carte=CARTE_JSON,
        chemin_corpus=CORPUS_JSON,
        dossier_output=DOSSIER_CORPUS,
        indexeur=indexeur
    )
    return revue


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 5 : AGENT DÉTECTEUR DE GAPS
# ═════════════════════════════════════════════════════════════════════════════

def etape_detecteur(indexeur=None):
    from agent_detecteur import AgentDetecteurGaps

    detecteur = AgentDetecteurGaps()
    rapport = detecteur.run(
        chemin_carte=CARTE_JSON,
        chemin_corpus=CORPUS_JSON,
        dossier_output=DOSSIER_CORPUS,
        chemin_revue=REVUE_JSON,
        indexeur=indexeur
    )
    return rapport


# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 6 : AGENT CITATEUR
# ═════════════════════════════════════════════════════════════════════════════

def etape_citateur(indexeur=None):
    from agent_citateur import AgentCitateur

    citateur = AgentCitateur()
    rapport = citateur.run(
        chemin_revue_json=REVUE_JSON,
        chemin_revue_md=REVUE_MD,
        chemin_corpus=CORPUS_JSON,
        dossier_output=DOSSIER_CORPUS,
        indexeur=indexeur
    )
    return rapport


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLET
# ═════════════════════════════════════════════════════════════════════════════

def pipeline_complet():
    banner()
    debut = time.time()

    # ── Étape 1 : Curateur ────────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 1/6 — AGENT CURATEUR")
    print("▓" * 70)
    corpus = etape_curateur()

    # ── Étape 2 : Indexeur ────────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 2/6 — AGENT INDEXEUR")
    print("▓" * 70)
    indexeur, rapport_index = etape_indexeur()

    # ── Étape 3 : Cartographe ─────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 3/6 — AGENT CARTOGRAPHE")
    print("▓" * 70)
    carte = etape_cartographe(indexeur=indexeur)

    # ── Étape 4 : Narrateur ───────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 4/6 — AGENT NARRATEUR")
    print("▓" * 70)
    revue = etape_narrateur(indexeur=indexeur)

    # ── Étape 5 : Détecteur ───────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 5/6 — AGENT DÉTECTEUR DE GAPS")
    print("▓" * 70)
    gaps = etape_detecteur(indexeur=indexeur)

    # ── Étape 6 : Citateur ────────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("  ÉTAPE 6/6 — AGENT CITATEUR")
    print("▓" * 70)
    rapport_cit = etape_citateur(indexeur=indexeur)

    # ── Résumé final ──────────────────────────────────────────────────────
    duree = round(time.time() - debut, 1)
    print("\n" + "=" * 70)
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              PIPELINE TERMINÉ AVEC SUCCÈS                   ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"  Durée totale : {duree}s")
    print(f"  Fichiers produits :")
    print(f"    → {CORPUS_JSON}")
    print(f"    → {CARTE_JSON}")
    print(f"    → {REVUE_MD}")
    print(f"    → {GAPS_JSON}")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMA Revue de Littérature — Pipeline")
    parser.add_argument(
        "--etape",
        choices=["curateur", "indexeur", "cartographe", "narrateur", "detecteur", "citateur"],
        help="Exécuter une seule étape du pipeline"
    )
    args = parser.parse_args()

    if args.etape:
        banner()
        print(f"\n  Exécution de l'étape : {args.etape.upper()}\n")

        if args.etape == "curateur":
            etape_curateur()
        elif args.etape == "indexeur":
            etape_indexeur()
        elif args.etape == "cartographe":
            etape_cartographe()
        elif args.etape == "narrateur":
            etape_narrateur()
        elif args.etape == "detecteur":
            etape_detecteur()
        elif args.etape == "citateur":
            etape_citateur()
    else:
        pipeline_complet()
