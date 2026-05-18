"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AGENT DÉTECTEUR DE GAPS — SMA Revue de Littérature                ║
║                                                                              ║
║  Reçoit   : carte_corpus.json       (Agent Cartographe)                     ║
║             corpus_complet.json     (Agent Curateur)                         ║
║             revue_litterature.json  (Agent Narrateur)                        ║
║             AgentIndexeur           (recherche sémantique)                   ║
║                                                                              ║
║  Techniques :                                                                ║
║    → Prompts contradictoires : "voici ce QUI existe, dis ce qui MANQUE"     ║
║    → Chain of Thought (CoT)  : raisonnement pas-à-pas avant conclusion      ║
║    → Multi-angles            : populations, méthodes, contextes,             ║
║                                croisements, temporalité, preuves             ║
║                                                                              ║
║  Produit  : gaps_detectes.json + gaps_detectes.md                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

CORRECTIONS APPLIQUÉES :
  Fix #1  — Retry robuste dans _appel_cot (backoff exponentiel, 5 tentatives,
             60s max, logging des erreurs non-rate-limit)
  Fix #2  — Vérification GROQ_API_KEY au __init__ (fail-fast avec message clair)
  Fix #3  — Pause inter-thèmes portée à 5s (intégrée dans le retry)
  Fix #4  — _rechercher_avec_fallback : discussion → conclusion → sans filtre
  Fix #5  — Logging DEBUG du raw LLM en cas d'échec JSON
  Fix #6  — Fallback "gap" si categorie ET type sont absents dans le rapport MD
  Fix #7  — Trous temporels : affichage liste réelle si non-consécutive
  Fix #8  — Logging structuré avec fichier .log (remplacement des print)
  Fix #9  — Import agent_indexeur protégé par try/except avec message clair
  Fix #10 — Sauvegarde incrémentale des gaps par thème (JSON partiel)
"""

import os
import json
import time
import re
import logging
import sys
from pathlib import Path
from collections import defaultdict, Counter

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING STRUCTURÉ — Fix #8
# ═════════════════════════════════════════════════════════════════════════════

def _configurer_logging(dossier_output: str = "data/corpus") -> logging.Logger:
    """
    Configure un logger avec sortie console ET fichier .log.
    Remplace tous les print() du module.
    """
    Path(dossier_output).mkdir(parents=True, exist_ok=True)
    log_path = Path(dossier_output) / "detecteur_gaps.log"

    logger = logging.getLogger("DetecteurGaps")
    logger.setLevel(logging.DEBUG)

    # Évite les handlers dupliqués si re-instancié
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler fichier (DEBUG — capture tout)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialisé → {log_path}")
    return logger


# Logger global (recréé avec le bon dossier dans run())
logger = logging.getLogger("DetecteurGaps")


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTES — RÉFÉRENTIELS DE COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════

POPULATIONS_REFERENTIEL = [
    "enfants", "adolescents", "personnes âgées", "femmes", "hommes",
    "étudiants", "enseignants", "professionnels", "patients", "migrants",
    "personnes en situation de handicap", "communautés rurales",
    "communautés urbaines", "pays en développement", "PME", "startups",
    "gouvernements", "ONG", "minorités ethniques", "personnes à faible revenu"
]

METHODES_REFERENTIEL = [
    "randomized controlled trial", "meta-analysis", "systematic review",
    "longitudinal study", "case study", "ethnography", "grounded theory",
    "mixed methods", "survey", "interview", "focus group", "experiment",
    "simulation", "machine learning", "deep learning", "NLP",
    "reinforcement learning", "graph neural network", "transformer",
    "bayesian", "fuzzy logic", "genetic algorithm", "agent-based modeling"
]

CONTEXTES_REFERENTIEL = [
    "Afrique", "Afrique subsaharienne", "Afrique du Nord", "Moyen-Orient",
    "Asie du Sud", "Asie du Sud-Est", "Amérique latine", "Caraïbes",
    "Europe de l'Est", "Océanie", "pays arabes", "BRICS",
    "pays francophones", "pays anglophones"
]

EVIDENCE_REFERENTIEL = [
    "strong", "moderate", "weak", "anecdotal"
]


# ═════════════════════════════════════════════════════════════════════════════
#  1. EXTRACTEUR DE PRÉSENCES
# ═════════════════════════════════════════════════════════════════════════════

class ExtracteurPresences:
    """
    Extrait de manière exhaustive tout ce qui EST présent dans le corpus.
    C'est la base du raisonnement contradictoire :
    on ne peut identifier les absences qu'en connaissant précisément les présences.
    """

    def __init__(self, corpus: list, carte: dict):
        self.corpus = corpus
        self.carte  = carte

    def extraire_tout(self) -> dict:
        return {
            "populations":          self._extraire_populations(),
            "methodes":             self._extraire_methodes(),
            "contextes_geo":        self._extraire_contextes(),
            "domaines":             self._extraire_domaines(),
            "periodes":             self._extraire_periodes(),
            "claim_types":          self._extraire_claim_types(),
            "evidence_levels":      self._extraire_evidence(),
            "themes":               self._extraire_themes(),
            "croisements_presents": self._extraire_croisements(),
            "langues":              self._extraire_langues(),
            "types_documents":      self._extraire_types_docs()
        }

    def _extraire_populations(self) -> dict:
        toutes = []
        for doc in self.corpus:
            pops = doc.get("populations_etudiees") or []
            for p in pops:
                if p:
                    toutes.append(str(p).strip().lower())
        return {
            "liste":     list(set(toutes)),
            "frequences": dict(Counter(toutes).most_common(20))
        }

    def _extraire_methodes(self) -> dict:
        toutes = []
        for doc in self.corpus:
            meths = doc.get("methodes") or []
            for m in meths:
                if m:
                    toutes.append(str(m).strip().lower())
        return {
            "liste":      list(set(toutes)),
            "frequences": dict(Counter(toutes).most_common(20))
        }

    def _extraire_contextes(self) -> dict:
        tous = []
        for doc in self.corpus:
            ctx = doc.get("contexte_geographique") or []
            for c in ctx:
                if c:
                    tous.append(str(c).strip())
        return {
            "liste":      list(set(tous)),
            "frequences": dict(Counter(tous).most_common(20))
        }

    def _extraire_domaines(self) -> list:
        return list(set(
            str(d.get("domaine","")).strip()
            for d in self.corpus if d.get("domaine")
        ))

    def _extraire_periodes(self) -> dict:
        annees = [
            int(d.get("annee") or 0)
            for d in self.corpus
            if d.get("annee") and int(d.get("annee") or 0) > 1990
        ]
        if not annees:
            return {}
        return {
            "annee_min":    min(annees),
            "annee_max":    max(annees),
            "distribution": dict(Counter(annees)),
            "periodes_etude": list(set(
                str(d.get("periode_etude","")).strip()
                for d in self.corpus if d.get("periode_etude")
            ))
        }

    def _extraire_claim_types(self) -> dict:
        types = [
            str(d.get("claim_type","")).strip()
            for d in self.corpus if d.get("claim_type")
        ]
        return dict(Counter(types))

    def _extraire_evidence(self) -> dict:
        levels = [
            str(d.get("evidence_level","")).strip()
            for d in self.corpus if d.get("evidence_level")
        ]
        return dict(Counter(levels))

    def _extraire_themes(self) -> list:
        themes = self.carte.get("themes", {})
        return [t.get("nom_theme","") for t in themes.values()]

    def _extraire_croisements(self) -> list:
        croisements = []
        themes = self.carte.get("themes", {})
        for theme in themes.values():
            nom = theme.get("nom_theme","")
            methodes = theme.get("methodes_dominantes", [])
            for m in methodes:
                croisements.append(f"{nom} — {m}")
        return croisements

    def _extraire_langues(self) -> dict:
        langues = [
            str(d.get("langue","")).strip()
            for d in self.corpus if d.get("langue")
        ]
        return dict(Counter(langues))

    def _extraire_types_docs(self) -> dict:
        types = [
            str(d.get("type_document","")).strip()
            for d in self.corpus if d.get("type_document")
        ]
        return dict(Counter(types))


# ═════════════════════════════════════════════════════════════════════════════
#  2. AGENT DÉTECTEUR DE GAPS
# ═════════════════════════════════════════════════════════════════════════════

class AgentDetecteurGaps:
    """
    Identifie systématiquement ce qui N'A PAS été étudié.

    Stratégie en 3 niveaux :

    Niveau 1 — Gaps directs (référentiel)
        Comparaison mécanique entre ce qui existe et des référentiels connus.

    Niveau 2 — Gaps thématiques (CoT par thème)
        Pour chaque thème, CoT : "Voici ce qui existe. Que manque-t-il ?"

    Niveau 3 — Gaps globaux (prompt contradictoire global)
        Un seul appel LLM avec tout le portrait du corpus.
        "Voici tout ce que contient ce corpus. Qu'est-ce qui est ABSENT ?"
    """

    def __init__(self):
        # ── Fix #2 : Vérification GROQ_API_KEY au démarrage ──────────────
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[DetecteurGaps] GROQ_API_KEY est absent ou vide.\n"
                "Définissez-la dans votre fichier .env ou en variable d'environnement.\n"
                "Exemple : GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
            )
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"

    # ─────────────────────────────────────────────────────────────────────────
    #  CHARGEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    def _charger(self, chemin_carte: str, chemin_corpus: str,
                  chemin_revue: str = None) -> tuple:
        with open(chemin_carte, "r", encoding="utf-8") as f:
            carte = json.load(f)
        with open(chemin_corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        revue = {}
        if chemin_revue and Path(chemin_revue).exists():
            with open(chemin_revue, "r", encoding="utf-8") as f:
                revue = json.load(f)
        return carte, corpus, revue

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #1 — APPEL LLM CoT AVEC RETRY ROBUSTE
    # ─────────────────────────────────────────────────────────────────────────

    def _appel_cot(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Appel LLM avec retry robuste sur rate limit.

        Fix #1 :
        - 5 tentatives (au lieu de 3)
        - Backoff exponentiel : 30s, 45s, 60s, 90s (au lieu de 12s fixe)
        - Les erreurs non-rate-limit sont loggées en WARNING (au lieu d'être avalées)
        - Retourne "" uniquement après épuisement de toutes les tentatives
        """
        delais = [30, 45, 60, 90]  # secondes entre chaque retry

        for tentative in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e).lower()

                if "rate_limit" in err_str:
                    if tentative < 4:
                        delai = delais[min(tentative, len(delais) - 1)]
                        logger.warning(
                            f"Rate limit Groq (tentative {tentative+1}/5) "
                            f"— attente {delai}s..."
                        )
                        time.sleep(delai)
                    else:
                        logger.error(
                            "Rate limit Groq persistant après 5 tentatives. "
                            "Abandon de cet appel."
                        )
                        return ""

                elif "timeout" in err_str or "connection" in err_str:
                    if tentative < 4:
                        delai = delais[min(tentative, len(delais) - 1)]
                        logger.warning(
                            f"Erreur réseau (tentative {tentative+1}/5) "
                            f"— attente {delai}s... | Détail : {e}"
                        )
                        time.sleep(delai)
                    else:
                        logger.error(f"Erreur réseau persistante : {e}")
                        return ""

                else:
                    # Fix #1 : erreurs non-rate-limit loggées (plus avalées silencieusement)
                    logger.error(
                        f"Erreur LLM inattendue (tentative {tentative+1}/5) : {e}"
                    )
                    if tentative < 4:
                        time.sleep(15)
                    else:
                        return ""

        return ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #5 — PARSEURS JSON AVEC LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def _parser_json_liste(self, raw: str, contexte: str = "") -> list:
        """
        Extrait une liste JSON depuis la réponse LLM.
        Fix #5 : logue le raw en DEBUG si le parsing échoue.
        """
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.warning(
                    f"Échec parsing JSON liste{' (' + contexte + ')' if contexte else ''} : {e}"
                )
                logger.debug(f"Raw LLM (liste) :\n{raw[:500]}")
        else:
            if raw.strip():
                logger.warning(
                    f"Aucun pattern JSON liste trouvé"
                    f"{' (' + contexte + ')' if contexte else ''}"
                )
                logger.debug(f"Raw LLM (no match) :\n{raw[:500]}")
        return []

    def _parser_json_objet(self, raw: str, contexte: str = "") -> dict:
        """
        Extrait un objet JSON depuis la réponse LLM.
        Fix #5 : logue le raw en DEBUG si le parsing échoue.
        """
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.warning(
                    f"Échec parsing JSON objet{' (' + contexte + ')' if contexte else ''} : {e}"
                )
                logger.debug(f"Raw LLM (objet) :\n{raw[:500]}")
        else:
            if raw.strip():
                logger.warning(
                    f"Aucun pattern JSON objet trouvé"
                    f"{' (' + contexte + ')' if contexte else ''}"
                )
                logger.debug(f"Raw LLM (no match) :\n{raw[:500]}")
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #4 — RECHERCHE RAG AVEC FALLBACK MULTI-SECTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _rechercher_avec_fallback(self, indexeur, requete: str,
                                   sections_cibles: list,
                                   n_resultats: int = 5) -> list:
        """
        Recherche RAG avec fallback progressif :
        1. Essaie chaque section dans l'ordre (discussion → conclusion → …)
        2. Si toutes les sections échouent, recherche sans filtre

        Fix #4 : évite les listes vides quand section_nom n'existe pas.
        """
        for section in sections_cibles:
            try:
                res = indexeur.rechercher(
                    requete,
                    n_resultats=n_resultats,
                    filtre={"section_nom": section}
                )
                if res:
                    logger.debug(
                        f"RAG fallback : {len(res)} résultats "
                        f"(section='{section}') pour '{requete[:50]}'"
                    )
                    return res
            except Exception as e:
                logger.debug(f"RAG erreur section '{section}' : {e}")

        # Fallback sans filtre
        try:
            res = indexeur.rechercher(requete, n_resultats=n_resultats)
            logger.debug(
                f"RAG fallback sans filtre : {len(res)} résultats "
                f"pour '{requete[:50]}'"
            )
            return res
        except Exception as e:
            logger.warning(f"RAG sans filtre échoué : {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  NIVEAU 1 — GAPS DIRECTS PAR RÉFÉRENTIEL
    # ─────────────────────────────────────────────────────────────────────────

    def _gaps_par_referentiel(self, presences: dict) -> dict:
        """
        Comparaison mécanique entre le corpus et les référentiels.
        Rapide, objectif, pas de LLM nécessaire.
        """
        logger.info("[Niveau 1] Gaps par référentiel...")

        # Populations absentes
        pops_presentes = set(p.lower() for p in presences["populations"]["liste"])
        pops_absentes  = [
            p for p in POPULATIONS_REFERENTIEL
            if not any(p.lower() in pp or pp in p.lower() for pp in pops_presentes)
        ]

        # Méthodes absentes
        meths_presentes = set(m.lower() for m in presences["methodes"]["liste"])
        meths_absentes  = [
            m for m in METHODES_REFERENTIEL
            if not any(m.lower() in mm or mm in m.lower() for mm in meths_presentes)
        ]

        # Contextes géographiques absents
        ctx_presentes = set(c.lower() for c in presences["contextes_geo"]["liste"])
        ctx_absents   = [
            c for c in CONTEXTES_REFERENTIEL
            if not any(c.lower() in cc or cc in c.lower() for cc in ctx_presentes)
        ]

        # Evidence levels absents ou sous-représentés
        ev_presents  = presences["evidence_levels"]
        ev_absents   = [e for e in EVIDENCE_REFERENTIEL if e not in ev_presents]
        ev_faibles   = [e for e, nb in ev_presents.items() if nb <= 1]

        # Langues — dominance d'une langue ?
        langues = presences["langues"]
        total_docs = sum(langues.values()) if langues else 0
        langues_dominantes = {
            l: round(n/total_docs*100, 1)
            for l, n in langues.items()
        } if total_docs > 0 else {}
        biais_linguistique = (
            max(langues.values()) / total_docs > 0.8
            if langues and total_docs > 0 else False
        )

        # Trous temporels
        periodes = presences["periodes"]
        trous_temporels = []
        if periodes.get("distribution"):
            dist = periodes["distribution"]
            annee_min = periodes["annee_min"]
            annee_max = periodes["annee_max"]
            for annee in range(annee_min, annee_max + 1):
                if dist.get(annee, 0) == 0:
                    trous_temporels.append(annee)

        # Types de documents sous-représentés
        types_docs = presences["types_documents"]
        total_types = sum(types_docs.values()) if types_docs else 0
        types_rares = [t for t, n in types_docs.items()
                       if total_types > 0 and n / total_types < 0.05]

        return {
            "populations_absentes":    pops_absentes,
            "methodes_absentes":       meths_absentes,
            "contextes_geo_absents":   ctx_absents,
            "evidence_levels_absents": ev_absents,
            "evidence_levels_faibles": ev_faibles,
            "biais_linguistique":      biais_linguistique,
            "langues_distribution":    langues_dominantes,
            "trous_temporels":         trous_temporels,
            "types_documents_rares":   types_rares
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  NIVEAU 2 — GAPS PAR THÈME (CoT)
    # ─────────────────────────────────────────────────────────────────────────

    def _gaps_theme_cot(self, theme_id, theme: dict,
                         presences: dict,
                         indexeur=None) -> dict:
        """
        Prompt CoT par thème.
        Technique : on donne au LLM ce qui EXISTE,
        on lui demande de raisonner étape par étape pour trouver
        ce qui MANQUE.

        Fix #4 : utilise _rechercher_avec_fallback pour le RAG.
        Fix #3 : la pause inter-thèmes est gérée dans run() (5s min).
        """
        nom_theme    = theme.get("nom_theme", f"Thème {theme_id}")
        methodes_dom = theme.get("methodes_dominantes", [])
        populations  = theme.get("populations_etudiees", [])
        sous_themes  = theme.get("sous_themes", [])
        nb_articles  = theme.get("nb_articles", 0)
        tendance     = theme.get("evolution_temporelle", {}).get("tendance", "")

        # Fix #4 — Recherche RAG avec fallback multi-sections
        chunks_limites = []
        if indexeur:
            chunks_limites = self._rechercher_avec_fallback(
                indexeur,
                requete=f"{nom_theme} limitation future work gap",
                sections_cibles=["discussion", "conclusion", "limitations"],
                n_resultats=5
            )

        passages_limites = "\n".join(
            f"- {c.get('texte','')[:200]}" for c in chunks_limites
        )

        articles_info = "\n".join(
            f"  • {art.get('titre','')[:60]} | "
            f"annee={art.get('annee','?')} | "
            f"claim={art.get('claim_type','?')} | "
            f"evidence={art.get('evidence_level','?')}"
            for art in theme.get("articles", [])[:10]
        )

        prompt = f"""Tu es un chercheur expert en identification de lacunes dans la littérature scientifique.

═══════════════════════════════════════════
CE QUI EXISTE DÉJÀ sur le thème "{nom_theme}"
═══════════════════════════════════════════

Nombre d'articles : {nb_articles}
Tendance : {tendance}
Méthodes utilisées : {', '.join(methodes_dom)}
Populations étudiées : {', '.join(str(p) for p in populations)}
Sous-thèmes couverts : {', '.join(sous_themes)}

Articles présents :
{articles_info}

Limites déclarées par les auteurs eux-mêmes :
{passages_limites or "Aucun passage disponible."}

═══════════════════════════════════════════
MAINTENANT : RAISONNE POUR TROUVER CE QUI MANQUE
═══════════════════════════════════════════

Utilise le raisonnement suivant (Chain of Thought) :

ÉTAPE 1 — POPULATIONS ABSENTES
Réfléchis : quelles populations existent dans ce domaine mais ne sont PAS
étudiées dans les articles listés ci-dessus ? Pense aux groupes d'âge,
aux genres, aux contextes socio-économiques, aux types d'organisations.

ÉTAPE 2 — MÉTHODES INEXPLOITÉES
Réfléchis : quelles méthodes ou approches de recherche seraient pertinentes
pour ce thème mais n'ont PAS encore été utilisées ?

ÉTAPE 3 — CONTEXTES GÉOGRAPHIQUES MANQUANTS
Réfléchis : dans quels pays, régions ou contextes culturels ce thème
n'a PAS encore été étudié alors qu'il serait pertinent de le faire ?

ÉTAPE 4 — CROISEMENTS THÉMATIQUES ABSENTS
Réfléchis : avec quels autres thèmes ou disciplines ce thème devrait
être croisé mais ne l'a PAS encore été ?

ÉTAPE 5 — TYPES DE PREUVES MANQUANTS
Réfléchis : quels types d'études manquent (longitudinales, expérimentales,
meta-analyses, études comparatives, réplication) ?

ÉTAPE 6 — FORMULATION FINALE DES GAPS
Pour chaque gap identifié, formule-le de façon précise :
"Aucune étude ne traite de [QUOI] dans le contexte [OÙ]
 avec la méthode [COMMENT] pour la population [QUI]."

Réponds UNIQUEMENT avec un JSON valide :

{{
  "raisonnement_cot": {{
    "etape_1_populations": "ton raisonnement ici...",
    "etape_2_methodes":    "ton raisonnement ici...",
    "etape_3_contextes":   "ton raisonnement ici...",
    "etape_4_croisements": "ton raisonnement ici...",
    "etape_5_preuves":     "ton raisonnement ici..."
  }},
  "gaps_identifies": [
    {{
      "type":        "population | methode | contexte | croisement | preuve | temporel",
      "formulation": "Aucune étude ne traite de X dans le contexte Y avec la méthode Z pour W.",
      "importance":  "haute | moyenne | faible",
      "justification": "Pourquoi ce gap est important pour la progression du domaine.",
      "piste_recherche": "Question de recherche concrète qui comblerait ce gap."
    }}
  ],
  "gap_prioritaire": "Le gap le plus critique de ce thème en une phrase."
}}"""

        logger.info(f"  → CoT gap analyse : '{nom_theme}'")
        raw = self._appel_cot(prompt, max_tokens=1800)

        if not raw:
            logger.warning(f"  ⚠ Réponse LLM vide pour le thème '{nom_theme}'")
            return {
                "theme_id":        theme_id,
                "nom_theme":       nom_theme,
                "gaps":            [],
                "gap_prioritaire": "",
                "raisonnement":    {}
            }

        # Fix #5 : contexte de parsing pour le logging
        result = self._parser_json_objet(raw, contexte=f"theme '{nom_theme}'")

        if not result:
            return {
                "theme_id":        theme_id,
                "nom_theme":       nom_theme,
                "gaps":            [],
                "gap_prioritaire": "",
                "raisonnement":    {}
            }

        return {
            "theme_id":        theme_id,
            "nom_theme":       nom_theme,
            "gaps":            result.get("gaps_identifies", []),
            "gap_prioritaire": result.get("gap_prioritaire", ""),
            "raisonnement":    result.get("raisonnement_cot", {})
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  NIVEAU 3 — GAPS GLOBAUX (prompt contradictoire global)
    # ─────────────────────────────────────────────────────────────────────────

    def _gaps_globaux_contradictoire(self, presences: dict,
                                      gaps_par_theme: list,
                                      carte: dict) -> dict:
        """
        Prompt contradictoire global sur tout le corpus.
        """
        logger.info("[Niveau 3] Gaps globaux (prompt contradictoire)...")

        themes = carte.get("themes", {})
        resume_themes = "\n".join(
            f"- {t.get('nom_theme','')} : "
            f"{t.get('nb_articles',0)} articles, "
            f"méthodes={','.join(t.get('methodes_dominantes',[])[:3])}, "
            f"tendance={t.get('evolution_temporelle',{}).get('tendance','?')}"
            for t in themes.values()
        )

        resume_gaps_themes = "\n".join(
            f"- {g['nom_theme']} : {g.get('gap_prioritaire','')}"
            for g in gaps_par_theme if g.get("gap_prioritaire")
        )

        croisements_presents = presences["croisements_presents"][:10]

        prompt = f"""Tu es un expert en épistémologie et en analyse critique de la littérature scientifique.

╔═══════════════════════════════════════════════════════════╗
║        PORTRAIT COMPLET DU CORPUS — CE QUI EXISTE         ║
╚═══════════════════════════════════════════════════════════╝

THÈMES COUVERTS :
{resume_themes}

POPULATIONS ÉTUDIÉES :
{json.dumps(presences['populations']['frequences'], ensure_ascii=False)}

MÉTHODES UTILISÉES :
{json.dumps(presences['methodes']['frequences'], ensure_ascii=False)}

CONTEXTES GÉOGRAPHIQUES :
{json.dumps(presences['contextes_geo']['frequences'], ensure_ascii=False)}

TYPES DE PREUVES (evidence levels) :
{json.dumps(presences['evidence_levels'], ensure_ascii=False)}

TYPES DE CONTRIBUTIONS (claim types) :
{json.dumps(presences['claim_types'], ensure_ascii=False)}

LANGUES DES ÉTUDES :
{json.dumps(presences['langues'], ensure_ascii=False)}

CROISEMENTS THÈME × MÉTHODE DÉJÀ PRÉSENTS :
{chr(10).join(f'- {c}' for c in croisements_presents)}

GAPS PRIORITAIRES PAR THÈME (déjà identifiés) :
{resume_gaps_themes}

╔═══════════════════════════════════════════════════════════╗
║   MAINTENANT : ANALYSE CRITIQUE DE CE QUI EST ABSENT      ║
╚═══════════════════════════════════════════════════════════╝

En te basant UNIQUEMENT sur ce que tu vois ci-dessus,
identifie les lacunes GLOBALES du corpus qui transcendent
les thèmes individuels.

Raisonne en Chain of Thought :

ÉTAPE A : Quels CROISEMENTS entre thèmes sont absents ?
ÉTAPE B : Quels BIAIS SYSTÉMIQUES traversent tout le corpus ?
ÉTAPE C : Quelles QUESTIONS FONDAMENTALES le corpus ne pose-t-il jamais ?
ÉTAPE D : Que manque-t-il en termes de TYPES D'ÉTUDES ?
ÉTAPE E : Quels CONTEXTES APPLICATIFS sont absents ?

Réponds UNIQUEMENT avec un JSON valide :

{{
  "raisonnement_global": {{
    "etape_A_croisements_absents":  "...",
    "etape_B_biais_systemiques":    "...",
    "etape_C_angles_morts":         "...",
    "etape_D_types_etudes":         "...",
    "etape_E_contextes_applicatifs":"..."
  }},
  "gaps_globaux": [
    {{
      "categorie":    "croisement | biais | angle_mort | type_etude | contexte_applicatif",
      "titre":        "Titre court du gap (10 mots max)",
      "description":  "Description précise du gap en 2-3 phrases.",
      "formulation_gap": "Aucune étude ne traite de X dans Y avec Z pour W.",
      "importance":   "critique | haute | moyenne",
      "themes_concernes": ["nom_theme_1", "nom_theme_2"],
      "piste_recherche": "Question de recherche concrète qui comblerait ce gap."
    }}
  ],
  "synthese_lacunes": "Paragraphe de synthèse (5-8 phrases) sur les grandes absences du corpus.",
  "recommandations_recherche": [
    "Recommandation concrète 1 pour la recherche future",
    "Recommandation concrète 2",
    "Recommandation concrète 3"
  ]
}}"""

        raw    = self._appel_cot(prompt, max_tokens=2000)

        if not raw:
            logger.warning("Réponse LLM vide pour l'analyse globale")
            return {
                "gaps_globaux":        [],
                "synthese_lacunes":    "",
                "recommandations":     [],
                "raisonnement_global": {}
            }

        # Fix #5 : contexte de parsing
        result = self._parser_json_objet(raw, contexte="analyse globale")

        if not result:
            return {
                "gaps_globaux":        [],
                "synthese_lacunes":    "",
                "recommandations":     [],
                "raisonnement_global": {}
            }

        return {
            "gaps_globaux":        result.get("gaps_globaux", []),
            "synthese_lacunes":    result.get("synthese_lacunes", ""),
            "recommandations":     result.get("recommandations_recherche", []),
            "raisonnement_global": result.get("raisonnement_global", {})
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  SCORING DES GAPS
    # ─────────────────────────────────────────────────────────────────────────

    def _scorer_gaps(self, tous_les_gaps: list) -> list:
        """
        Calcule un score de priorité pour chaque gap.

        Fix #6 : fallback "gap" si categorie ET type sont absents.
        """
        TYPE_BONUS = {
            "angle_mort":            3,
            "croisement":            2,
            "biais":                 2,
            "methode":               1,
            "population":            1,
            "contexte":              1,
            "preuve":                1,
            "temporel":              0,
            "type_etude":            1,
            "contexte_applicatif":   1
        }
        IMPORTANCE_SCORE = {"critique": 3, "haute": 2, "moyenne": 1, "faible": 0}

        for gap in tous_les_gaps:
            score = 0
            # Fix #6 : fallback "gap" si les deux clés sont absentes
            type_gap = gap.get("categorie") or gap.get("type") or "gap"
            score += IMPORTANCE_SCORE.get(gap.get("importance", "moyenne"), 1)
            score += TYPE_BONUS.get(type_gap, 1)
            score += min(len(gap.get("themes_concernes", [])), 3)
            if gap.get("piste_recherche"):
                score += 1
            gap["score_priorite"] = score

        return sorted(tous_les_gaps, key=lambda g: g.get("score_priorite", 0), reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  GÉNÉRATION DU RAPPORT MARKDOWN
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_rapport_md(self, rapport: dict, dossier_output: str):
        """
        Fix #6 : fallback "gap" si categorie et type absents dans les titres.
        Fix #7 : affichage correct des trous temporels non-consécutifs.
        """
        lignes = []
        lignes.append("# Rapport de Détection des Research Gaps\n")
        lignes.append(f"*{rapport['meta']['nb_gaps_total']} gaps identifiés "
                      f"sur {rapport['meta']['nb_articles_analyses']} articles*\n")
        lignes.append("---\n")

        # Synthèse globale
        lignes.append("## Synthèse des lacunes\n")
        lignes.append(rapport["analyse_globale"].get("synthese_lacunes","") + "\n")

        # Gaps par niveau d'importance
        lignes.append("## Gaps critiques et prioritaires\n")
        tous = rapport.get("tous_les_gaps_tries", [])
        critiques = [g for g in tous if g.get("importance") in ("critique","haute")]

        for g in critiques[:10]:
            titre = g.get("titre") or g.get("formulation","")[:60]
            # Fix #6 : fallback "gap" si les deux absents
            cat   = g.get("categorie") or g.get("type") or "gap"
            lignes.append(f"### [{cat.upper()}] {titre}")
            lignes.append(f"**Formulation** : {g.get('formulation_gap') or g.get('formulation','')}\n")
            lignes.append(f"**Justification** : {g.get('justification') or g.get('description','')}\n")
            if g.get("piste_recherche"):
                lignes.append(f"**Piste de recherche** : {g['piste_recherche']}\n")
            lignes.append("")

        # Gaps par thème
        lignes.append("## Gaps par thème\n")
        for g_theme in rapport.get("gaps_par_theme", []):
            lignes.append(f"### {g_theme['nom_theme']}")
            if g_theme.get("gap_prioritaire"):
                lignes.append(f"**Gap prioritaire** : {g_theme['gap_prioritaire']}\n")
            for gap in g_theme.get("gaps", [])[:4]:
                # Fix #6 : fallback "gap"
                type_gap = gap.get("type") or gap.get("categorie") or "gap"
                lignes.append(f"- **[{type_gap}]** {gap.get('formulation','')}")
                if gap.get("piste_recherche"):
                    lignes.append(f"  *→ {gap['piste_recherche']}*")
            lignes.append("")

        # Gaps par référentiel
        ref = rapport.get("gaps_referentiel", {})
        lignes.append("## Absences identifiées par référentiel\n")

        if ref.get("populations_absentes"):
            lignes.append(f"**Populations non étudiées** : "
                          f"{', '.join(ref['populations_absentes'][:8])}\n")
        if ref.get("methodes_absentes"):
            lignes.append(f"**Méthodes inexploitées** : "
                          f"{', '.join(ref['methodes_absentes'][:8])}\n")
        if ref.get("contextes_geo_absents"):
            lignes.append(f"**Contextes géographiques absents** : "
                          f"{', '.join(ref['contextes_geo_absents'][:8])}\n")
        if ref.get("biais_linguistique"):
            lignes.append(f"**Biais linguistique détecté** : "
                          f"{json.dumps(ref.get('langues_distribution',''), ensure_ascii=False)}\n")

        # Fix #7 — Trous temporels : affichage adapté selon consécutivité
        if ref.get("trous_temporels"):
            trous = ref["trous_temporels"]
            if len(trous) == 1:
                desc_trous = f"année {trous[0]}"
            else:
                # Vérifie si les années sont consécutives
                est_consecutif = all(
                    trous[i+1] - trous[i] == 1 for i in range(len(trous)-1)
                )
                if est_consecutif:
                    desc_trous = f"{trous[0]}–{trous[-1]} (plage continue)"
                else:
                    # Liste les années réelles pour éviter l'ambiguïté
                    desc_trous = (
                        f"{trous[0]}–{trous[-1]} "
                        f"(années manquantes : {', '.join(str(a) for a in trous)})"
                    )
            lignes.append(
                f"**Trous temporels** : {desc_trous} "
                f"({len(trous)} année(s) sans publication)\n"
            )

        # Recommandations
        lignes.append("## Recommandations pour la recherche future\n")
        for i, rec in enumerate(rapport["analyse_globale"].get("recommandations", []), 1):
            lignes.append(f"{i}. {rec}")

        texte = "\n".join(lignes)
        path  = Path(dossier_output) / "gaps_detectes.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(texte)
        logger.info(f"Rapport MD → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #10 — SAUVEGARDE INCRÉMENTALE DES GAPS PAR THÈME
    # ─────────────────────────────────────────────────────────────────────────

    def _sauvegarder_theme_incremental(self, gaps_theme: dict,
                                        dossier_output: str,
                                        theme_index: int):
        """
        Sauvegarde le résultat d'un thème immédiatement après son calcul.
        En cas de crash au thème N, les thèmes 1..N-1 sont préservés.
        """
        path = Path(dossier_output) / f"gaps_theme_{theme_index:02d}_partiel.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gaps_theme, f, ensure_ascii=False, indent=2)
            logger.debug(f"Sauvegarde incrémentale thème {theme_index} → {path}")
        except Exception as e:
            logger.warning(f"Échec sauvegarde incrémentale thème {theme_index} : {e}")

    def _charger_themes_partiels(self, dossier_output: str) -> dict:
        """
        Charge les gaps déjà calculés depuis les fichiers partiels.
        Retourne un dict {theme_id_index: gaps_theme}.
        """
        partiels = {}
        dossier = Path(dossier_output)
        for f in sorted(dossier.glob("gaps_theme_*_partiel.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                # Extrait l'index du nom de fichier
                idx = int(f.stem.split("_")[2])
                partiels[idx] = data
                logger.info(f"Reprise : thème partiel chargé → {f.name}")
            except Exception as e:
                logger.warning(f"Impossible de charger {f.name} : {e}")
        return partiels

    def _nettoyer_partiels(self, dossier_output: str):
        """Supprime les fichiers partiels après une exécution réussie."""
        for f in Path(dossier_output).glob("gaps_theme_*_partiel.json"):
            try:
                f.unlink()
                logger.debug(f"Partiel supprimé : {f.name}")
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, chemin_carte: str,
            chemin_corpus: str,
            dossier_output: str,
            chemin_revue: str = None,
            indexeur=None,
            reprendre_partiels: bool = True) -> dict:
        """
        Exécution complète de l'agent.

        Paramètres
        ----------
        reprendre_partiels : bool
            Si True, recharge les gaps de thèmes déjà calculés lors d'un
            crash précédent (Fix #10).
        """
        # Fix #8 : initialise le logger avec le bon dossier de sortie
        global logger
        logger = _configurer_logging(dossier_output)

        logger.info("=" * 60)
        logger.info("  AGENT DÉTECTEUR DE GAPS — DÉMARRAGE")
        logger.info("=" * 60)
        debut = time.time()
        Path(dossier_output).mkdir(parents=True, exist_ok=True)

        # ── Chargement ────────────────────────────────────────────────────
        carte, corpus, revue = self._charger(chemin_carte, chemin_corpus, chemin_revue)
        logger.info(f"{len(corpus)} articles | {len(carte.get('themes',{}))} thèmes")

        # ── Niveau 1 : Portrait des présences ─────────────────────────────
        logger.info("[Niveau 1] Extraction des présences...")
        extracteur = ExtracteurPresences(corpus, carte)
        presences  = extracteur.extraire_tout()

        # ── Niveau 1 : Gaps référentiel ───────────────────────────────────
        gaps_ref = self._gaps_par_referentiel(presences)

        # ── Niveau 2 : Gaps CoT par thème ────────────────────────────────
        logger.info("[Niveau 2] Analyse CoT par thème...")
        gaps_par_theme = []
        themes = carte.get("themes", {})

        # Fix #10 : reprendre les partiels si disponibles
        partiels_existants = {}
        if reprendre_partiels:
            partiels_existants = self._charger_themes_partiels(dossier_output)

        for idx, (theme_id, theme) in enumerate(themes.items(), start=1):
            nom_theme = theme.get("nom_theme", f"Thème {theme_id}")

            # Fix #10 : réutilise le partiel si déjà calculé
            if idx in partiels_existants:
                logger.info(
                    f"  [Thème {idx}/{len(themes)}] '{nom_theme}' "
                    f"— chargé depuis partiel (skip LLM)"
                )
                gaps_par_theme.append(partiels_existants[idx])
                continue

            logger.info(f"  [Thème {idx}/{len(themes)}] '{nom_theme}'")
            gaps_theme = self._gaps_theme_cot(
                theme_id, theme, presences, indexeur
            )
            gaps_par_theme.append(gaps_theme)

            # Fix #10 : sauvegarde incrémentale immédiate
            self._sauvegarder_theme_incremental(gaps_theme, dossier_output, idx)

            # Fix #3 : pause de 5s minimum entre les appels (au lieu de 1.5s)
            if idx < len(themes):
                logger.debug("Pause inter-thèmes : 5s")
                time.sleep(5)

        # ── Niveau 3 : Gaps globaux contradictoires ───────────────────────
        logger.info("[Niveau 3] Analyse contradictoire globale...")
        analyse_globale = self._gaps_globaux_contradictoire(
            presences, gaps_par_theme, carte
        )
        time.sleep(5)  # Fix #3 : cohérence avec la pause inter-appels

        # ── Consolidation et scoring ──────────────────────────────────────
        logger.info("Scoring et tri des gaps...")
        tous_les_gaps = []

        for g_theme in gaps_par_theme:
            for gap in g_theme.get("gaps", []):
                gap["source"]       = "theme"
                gap["theme_source"] = g_theme["nom_theme"]
                tous_les_gaps.append(gap)

        for gap in analyse_globale.get("gaps_globaux", []):
            gap["source"] = "global"
            tous_les_gaps.append(gap)

        tous_tries = self._scorer_gaps(tous_les_gaps)

        # ── Rapport final ─────────────────────────────────────────────────
        rapport = {
            "meta": {
                "nb_articles_analyses": len(corpus),
                "nb_themes":            len(themes),
                "nb_gaps_total":        len(tous_tries),
                "nb_gaps_par_theme":    sum(len(g.get("gaps",[])) for g in gaps_par_theme),
                "nb_gaps_globaux":      len(analyse_globale.get("gaps_globaux",[])),
                "duree_secondes":       round(time.time() - debut, 1)
            },
            "presences":           presences,
            "gaps_referentiel":    gaps_ref,
            "gaps_par_theme":      gaps_par_theme,
            "analyse_globale":     analyse_globale,
            "tous_les_gaps_tries": tous_tries
        }

        # Sauvegarde JSON finale
        json_path = Path(dossier_output) / "gaps_detectes.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
        logger.info(f"Rapport JSON → {json_path}")

        # Rapport Markdown
        self._generer_rapport_md(rapport, dossier_output)

        # Fix #10 : nettoyage des partiels après succès
        self._nettoyer_partiels(dossier_output)

        # Résumé console
        logger.info("=" * 60)
        logger.info("  RÉSUMÉ AGENT DÉTECTEUR DE GAPS")
        logger.info("=" * 60)
        logger.info(f"Gaps totaux identifiés : {len(tous_tries)}")
        logger.info(
            f"Gaps critiques/hauts   : "
            f"{sum(1 for g in tous_tries if g.get('importance') in ('critique','haute'))}"
        )
        logger.info(
            f"Pistes de recherche    : "
            f"{sum(1 for g in tous_tries if g.get('piste_recherche'))}"
        )
        logger.info(f"Durée totale           : {rapport['meta']['duree_secondes']}s")
        logger.info("Top 5 gaps prioritaires :")
        for g in tous_tries[:5]:
            formulation = g.get("formulation_gap") or g.get("formulation","")
            logger.info(f"  [{g.get('score_priorite',0)}] {formulation[:80]}...")

        return rapport


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Fix #9 — Import protégé avec message d'erreur explicite
    try:
        from agent_indexeur import AgentIndexeur
    except ModuleNotFoundError as e:
        print(
            f"[ERREUR] Impossible d'importer AgentIndexeur : {e}\n"
            "Vérifiez que le fichier agent_indexeur.py (ou agent_indexeur_v2.py) "
            "est dans le même répertoire et que son nom correspond à l'import.\n"
            "Si votre fichier s'appelle 'agent_indexeur_v2.py', renommez l'import "
            "en : from agent_indexeur_v2 import AgentIndexeur"
        )
        sys.exit(1)
    except ImportError as e:
        print(
            f"[ERREUR] Erreur d'import AgentIndexeur : {e}\n"
            "Vérifiez les dépendances du module (chromadb, sentence-transformers, etc.)"
        )
        sys.exit(1)

    indexeur = AgentIndexeur(dossier_chroma="data/chroma_db")

    detecteur = AgentDetecteurGaps()
    rapport = detecteur.run(
        chemin_carte      = "data/corpus/carte_corpus.json",
        chemin_corpus     = "data/corpus/corpus_complet.json",
        dossier_output    = "data/corpus",
        chemin_revue      = "data/corpus/revue_litterature.json",
        indexeur          = indexeur,
        reprendre_partiels= True   # Fix #10 : reprend automatiquement après un crash
    )