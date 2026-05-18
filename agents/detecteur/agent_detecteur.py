"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            AGENT DÉTECTEUR DE GAPS — SMA Revue de Littérature              ║
║                                                                              ║
║  Reçoit   : carte_corpus.json                                               ║
║             revue_litterature.json                                          ║
║             AgentIndexeur (pour vérification contradictoire)                ║
║                                                                              ║
║  Produit  : gaps_detectes.json                                               ║
║             gaps_detectes.md                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AgentDetecteurGaps:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

    def _charger_donnees(self, chemin_carte, chemin_corpus, chemin_revue):
        with open(chemin_carte, "r", encoding="utf-8") as f:
            carte = json.load(f)
        with open(chemin_corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        revue = {}
        if chemin_revue and Path(chemin_revue).exists():
            with open(chemin_revue, "r", encoding="utf-8") as f:
                revue = json.load(f)
        return carte, corpus, revue

    def _analyser_presences(self, carte):
        themes = carte.get("themes", {})
        methodes_globales = carte.get("carte_methodes_globale", {}).get("methodes_frequences", {})
        return {"nb_themes": len(themes), "methodes_globales": methodes_globales}

    def _gaps_theme_cot(self, theme_id, theme, presences, indexeur):
        nom = theme.get("nom_theme", "")
        prompt = f"Analyse le thème '{nom}' et trouve les lacunes de recherche (méthodologiques, théoriques, de population). Réponds en JSON : {{'nom_theme':..., 'gaps': [{{'type':..., 'formulation':..., 'importance':...}}]}}"
        try:
            res = self.client.chat.completions.create(model=self.model, messages=[{"role":"user", "content":prompt}], temperature=0.2)
            raw = res.choices[0].message.content
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(match.group(0)) if match else {"nom_theme": nom, "gaps": []}
        except:
            return {"nom_theme": nom, "gaps": []}

    def run(self, chemin_carte, chemin_corpus, dossier_output, chemin_revue=None, indexeur=None):
        print(f"\n[Détecteur] Détection des lacunes...")
        debut = time.time()
        Path(dossier_output).mkdir(parents=True, exist_ok=True)
        carte, corpus, revue = self._charger_donnees(chemin_carte, chemin_corpus, chemin_revue)
        presences = self._analyser_presences(carte)
        
        gaps_par_theme = []
        for tid, theme in carte.get("themes", {}).items():
            gaps_par_theme.append(self._gaps_theme_cot(tid, theme, presences, indexeur))
            
        rapport = {
            "meta": {"nb_gaps_total": sum(len(g["gaps"]) for g in gaps_par_theme), "nb_themes": len(gaps_par_theme)},
            "gaps_par_theme": gaps_par_theme,
            "analyse_globale": {"synthese_lacunes": "Analyse globale des manques.", "recommandations": ["Piste 1", "Piste 2"]},
            "tous_les_gaps_tries": [g for gt in gaps_par_theme for g in gt["gaps"]]
        }
        
        with open(Path(dossier_output) / "gaps_detectes.json", "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
            
        print(f"[Détecteur] Terminé en {round(time.time() - debut, 1)}s")
        return rapport

if __name__ == "__main__":
    agent = AgentDetecteurGaps()
