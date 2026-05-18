"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            AGENT CITATEUR — SMA Revue de Littérature                       ║
║                                                                              ║
║  Reçoit   : revue_litterature.md    (Agent Narrateur)                       ║
║             revue_litterature.json  (Agent Narrateur)                       ║
║             corpus_complet.json     (Agent Curateur)                        ║
║             AgentIndexeur           (RAG de vérification)                   ║
║                                                                              ║
║  Techniques :                                                                ║
║    → Vérification croisée : chaque claim de la revue vs corpus              ║
║    → RAG sémantique       : retrouve les sources originales par similarité  ║
║    → Détection d'hallucinations : auteurs/années/titres inventés            ║
║    → Audit bibliographie  : cohérence corpus ↔ références citées            ║
║                                                                              ║
║  Produit  : rapport_citations.json                                           ║
║             revue_corrigee.md (version annotée avec statuts)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

CORRECTIONS APPLIQUÉES :
  Fix #1  — GROQ_API_KEY vérifiée au __init__ (fail-fast)
  Fix #2  — Retry robuste dans _appel_llm (backoff exponentiel, 5 tentatives)
  Fix #3  — Logging structuré fichier + console (plus de print())
  Fix #4  — Parseurs JSON avec logging du raw en cas d'échec
  Fix #5  — Sauvegarde incrémentale des claims par section
  Fix #6  — Fallback RAG multi-sections pour la vérification sémantique
  Fix #7  — Score fiabilité défensif (division par zéro impossible)
  Fix #8  — Import agent_indexeur protégé par try/except
  Fix #9  — Segmentation de la revue en chunks pour éviter les context overflow
  Fix #10 — Rapport MD avec fallback sur tous les champs optionnels
"""

import os
import json
import time
import re
import sys
import logging
from pathlib import Path
from collections import Counter

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING STRUCTURÉ — Fix #3
# ═════════════════════════════════════════════════════════════════════════════

def _configurer_logging(dossier_output: str = "data/corpus") -> logging.Logger:
    Path(dossier_output).mkdir(parents=True, exist_ok=True)
    log_path = Path(dossier_output) / "citateur.log"

    logger = logging.getLogger("AgentCitateur")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialisé → {log_path}")
    return logger


logger = logging.getLogger("AgentCitateur")


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT CITATEUR
# ═════════════════════════════════════════════════════════════════════════════

class AgentCitateur:
    """
    Audit complet des citations d'une revue de littérature.

    Fonctionnement :
    ─────────────────
    1. Extraction des claims : segmente la revue en affirmations vérifiables
    2. Vérification par RAG  : retrouve les sources dans l'indexeur sémantique
    3. Vérification croisée  : compare auteurs/années/titres cités vs corpus réel
    4. Détection hallucinat. : identifie les références inventées
    5. Audit bibliographie   : cohérence entre corpus reçu et références citées
    6. Rapport annoté        : génère une version MD annotée avec statuts
    """

    # Statuts possibles pour un claim
    STATUT_VALIDE      = "valide"
    STATUT_APPROX      = "approximatif"
    STATUT_HALLU       = "hallucination"
    STATUT_NON_VERIF   = "non_verifiable"
    STATUT_MANQUE_REF  = "manque_reference"

    def __init__(self):
        # Fix #1 — Vérification GROQ_API_KEY au démarrage
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[AgentCitateur] GROQ_API_KEY est absent ou vide.\n"
                "Définissez-la dans votre fichier .env ou en variable d'environnement.\n"
                "Exemple : GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
            )
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #2 — APPEL LLM AVEC RETRY ROBUSTE
    # ─────────────────────────────────────────────────────────────────────────

    def _appel_llm(self, prompt: str, max_tokens: int = 1200) -> str:
        """
        Appel LLM avec backoff exponentiel sur rate limit / erreurs réseau.
        5 tentatives, délais : 30s → 45s → 60s → 90s.
        Toutes les erreurs sont loggées (plus d'avalage silencieux).
        """
        delais = [30, 45, 60, 90]

        for tentative in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,   # très bas pour la vérification factuelle
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
                        logger.error("Rate limit persistant après 5 tentatives.")
                        return ""

                elif "timeout" in err_str or "connection" in err_str:
                    if tentative < 4:
                        delai = delais[min(tentative, len(delais) - 1)]
                        logger.warning(
                            f"Erreur réseau (tentative {tentative+1}/5) "
                            f"— attente {delai}s... | {e}"
                        )
                        time.sleep(delai)
                    else:
                        logger.error(f"Erreur réseau persistante : {e}")
                        return ""

                else:
                    logger.error(f"Erreur LLM inattendue (tentative {tentative+1}/5) : {e}")
                    if tentative < 4:
                        time.sleep(15)
                    else:
                        return ""
        return ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #4 — PARSEURS JSON AVEC LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def _parser_json_liste(self, raw: str, contexte: str = "") -> list:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.warning(f"Échec parsing JSON liste ({contexte}) : {e}")
                logger.debug(f"Raw LLM :\n{raw[:600]}")
        elif raw.strip():
            logger.warning(f"Aucun pattern JSON liste ({contexte})")
            logger.debug(f"Raw LLM :\n{raw[:600]}")
        return []

    def _parser_json_objet(self, raw: str, contexte: str = "") -> dict:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.warning(f"Échec parsing JSON objet ({contexte}) : {e}")
                logger.debug(f"Raw LLM :\n{raw[:600]}")
        elif raw.strip():
            logger.warning(f"Aucun pattern JSON objet ({contexte})")
            logger.debug(f"Raw LLM :\n{raw[:600]}")
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    #  CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _charger(self, chemin_revue_json: str, chemin_revue_md: str,
                  chemin_corpus: str) -> tuple:
        with open(chemin_corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        revue_json = {}
        if Path(chemin_revue_json).exists():
            with open(chemin_revue_json, "r", encoding="utf-8") as f:
                revue_json = json.load(f)
        else:
            logger.warning(f"revue_litterature.json introuvable : {chemin_revue_json}")

        texte_revue = ""
        if Path(chemin_revue_md).exists():
            with open(chemin_revue_md, "r", encoding="utf-8") as f:
                texte_revue = f.read()
        else:
            logger.warning(f"revue_litterature.md introuvable : {chemin_revue_md}")

        return corpus, revue_json, texte_revue

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #9 — SEGMENTATION DE LA REVUE EN SECTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _segmenter_revue(self, texte_revue: str,
                          taille_max_chars: int = 3000) -> list[dict]:
        """
        Découpe la revue en sections pour éviter les context overflow LLM.
        Découpe d'abord sur les titres Markdown (## / ###),
        puis subdivise les sections trop longues.

        Retourne une liste de {'titre': str, 'contenu': str}.
        """
        sections = []

        # Split sur les titres Markdown
        blocs = re.split(r'\n(?=#{1,3} )', texte_revue)

        for bloc in blocs:
            if not bloc.strip():
                continue

            # Extraire le titre du bloc
            lignes = bloc.strip().split('\n')
            titre = lignes[0].strip('#').strip() if lignes else "Sans titre"
            contenu = '\n'.join(lignes[1:]).strip()

            if not contenu:
                contenu = bloc.strip()

            # Subdiviser si trop long
            if len(contenu) > taille_max_chars:
                # Découpe par paragraphes
                paragraphes = contenu.split('\n\n')
                chunk_courant = ""
                sous_idx = 1
                for para in paragraphes:
                    if len(chunk_courant) + len(para) > taille_max_chars and chunk_courant:
                        sections.append({
                            "titre":   f"{titre} (partie {sous_idx})",
                            "contenu": chunk_courant.strip()
                        })
                        chunk_courant = para + "\n\n"
                        sous_idx += 1
                    else:
                        chunk_courant += para + "\n\n"
                if chunk_courant.strip():
                    sections.append({
                        "titre":   f"{titre} (partie {sous_idx})",
                        "contenu": chunk_courant.strip()
                    })
            else:
                sections.append({"titre": titre, "contenu": contenu})

        if not sections:
            # Fallback : toute la revue en un seul bloc tronqué
            sections = [{"titre": "Revue complète", "contenu": texte_revue[:taille_max_chars]}]

        logger.info(f"Revue segmentée en {len(sections)} section(s)")
        return sections

    # ─────────────────────────────────────────────────────────────────────────
    #  CONSTRUCTION DE L'INDEX CORPUS (pour vérification croisée)
    # ─────────────────────────────────────────────────────────────────────────

    def _construire_index_corpus(self, corpus: list) -> dict:
        """
        Construit un index rapide du corpus pour la vérification croisée.
        Clés : auteur normalisé, année, titre normalisé.
        """
        index = {
            "par_auteur":  {},   # auteur_lower -> [doc, ...]
            "par_annee":   {},   # annee_str    -> [doc, ...]
            "par_titre":   {},   # mots_cles    -> [doc, ...]
            "ids_connus":  set() # set des identifiants uniques
        }

        for doc in corpus:
            # Index auteurs
            auteurs = doc.get("auteurs") or []
            if isinstance(auteurs, str):
                auteurs = [auteurs]
            for aut in auteurs:
                cle = str(aut).strip().lower()
                index["par_auteur"].setdefault(cle, []).append(doc)

            # Index années
            annee = str(doc.get("annee", "")).strip()
            if annee:
                index["par_annee"].setdefault(annee, []).append(doc)

            # Index titre (mots significatifs)
            titre = str(doc.get("titre", "")).lower()
            mots = [m for m in re.findall(r'\b\w{4,}\b', titre)]
            for mot in mots:
                index["par_titre"].setdefault(mot, []).append(doc)

            # IDs
            doc_id = doc.get("id") or doc.get("doi") or doc.get("titre","")
            if doc_id:
                index["ids_connus"].add(str(doc_id).strip())

        return index

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #6 — RECHERCHE RAG AVEC FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _rechercher_rag(self, indexeur, requete: str,
                         n_resultats: int = 3) -> list:
        """
        Recherche RAG sans filtre de section (le citateur veut tout le texte).
        Fix #6 : gestion d'exception pour ne pas crasher si l'indexeur est None.
        """
        if indexeur is None:
            return []
        try:
            return indexeur.rechercher(requete, n_resultats=n_resultats)
        except Exception as e:
            logger.warning(f"RAG échoué pour '{requete[:50]}' : {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  ÉTAPE 1 — EXTRACTION DES CLAIMS D'UNE SECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _extraire_claims_section(self, section: dict) -> list:
        """
        Extrait les affirmations vérifiables d'une section de la revue.
        Un claim = une phrase factuelle qui cite (ou devrait citer) une source.
        """
        prompt = f"""Tu es un expert en audit de citations académiques.

Voici une section d'une revue de littérature :

TITRE DE SECTION : {section['titre']}
---
{section['contenu'][:2500]}
---

Extrait TOUTES les affirmations factuelles vérifiables de ce texte.
Une affirmation vérifiable est une phrase qui :
- Énonce un résultat, une statistique, une conclusion de recherche
- Attribue une idée à un auteur (avec ou sans citation explicite)
- Fait une comparaison ou un contraste entre études
- Décrit une tendance ou un consensus dans le domaine

Pour chaque affirmation, identifie aussi la citation mentionnée si elle est présente
(auteur, année entre parenthèses comme "Smith, 2020" ou "Smith et al., 2019").

Réponds UNIQUEMENT avec un JSON valide :
[
  {{
    "claim": "Texte exact de l'affirmation (phrase complète)",
    "citation_mentionnee": "Smith, 2020" ou null si aucune citation,
    "auteurs_cites": ["Smith"] ou [],
    "annee_citee": "2020" ou null,
    "type_claim": "resultat | statistique | consensus | comparaison | methodologie | autre"
  }}
]

Si aucune affirmation vérifiable, retourne [].
"""
        raw = self._appel_llm(prompt, max_tokens=1200)
        return self._parser_json_liste(raw, contexte=f"claims section '{section['titre']}'")

    # ─────────────────────────────────────────────────────────────────────────
    #  ÉTAPE 2 — VÉRIFICATION D'UN CLAIM
    # ─────────────────────────────────────────────────────────────────────────

    def _verifier_claim(self, claim: dict, corpus: list,
                         index_corpus: dict, indexeur=None) -> dict:
        """
        Vérifie un claim par trois méthodes complémentaires :
        A) Vérification structurelle (auteur/année dans corpus)
        B) Vérification RAG sémantique
        C) Jugement LLM final

        Retourne le claim enrichi avec statut + justification.
        """
        texte_claim      = claim.get("claim", "")
        citation_mentee  = claim.get("citation_mentionnee")
        auteurs_cites    = claim.get("auteurs_cites", [])
        annee_citee      = claim.get("annee_citee")

        resultat = {
            **claim,
            "statut":          self.STATUT_NON_VERIF,
            "justification":   "",
            "source_trouvee":  None,
            "score_confiance": 0.0,
            "methode_verif":   []
        }

        # ── A) Vérification structurelle ─────────────────────────────────
        docs_candidats = []

        if auteurs_cites:
            for aut in auteurs_cites:
                cle = aut.strip().lower()
                # Recherche exacte puis partielle
                if cle in index_corpus["par_auteur"]:
                    docs_candidats.extend(index_corpus["par_auteur"][cle])
                else:
                    # Recherche partielle (nom de famille seul)
                    for cle_idx, docs in index_corpus["par_auteur"].items():
                        if cle in cle_idx or cle_idx in cle:
                            docs_candidats.extend(docs)

        if annee_citee and annee_citee in index_corpus["par_annee"]:
            docs_annee = index_corpus["par_annee"][annee_citee]
            # Intersection : docs qui matchent auteur ET année
            if docs_candidats:
                ids_candidats = {id(d) for d in docs_candidats}
                docs_candidats = [d for d in docs_annee if id(d) in ids_candidats]
                if not docs_candidats:
                    # Si intersection vide, garder les deux ensembles
                    docs_candidats.extend(docs_annee)
            else:
                docs_candidats.extend(docs_annee)

        if docs_candidats:
            resultat["methode_verif"].append("structurelle")
            resultat["source_trouvee"] = {
                "titre": docs_candidats[0].get("titre", ""),
                "annee": docs_candidats[0].get("annee", ""),
                "auteurs": docs_candidats[0].get("auteurs", [])
            }

        # ── B) Vérification RAG sémantique ───────────────────────────────
        chunks_rag = self._rechercher_rag(indexeur, texte_claim[:200], n_resultats=3)
        passages_rag = ""
        if chunks_rag:
            resultat["methode_verif"].append("RAG")
            passages_rag = "\n".join(
                f"- [{c.get('meta',{}).get('titre','?')}, "
                f"{c.get('meta',{}).get('annee','?')}] "
                f"{c.get('texte','')[:150]}"
                for c in chunks_rag
            )

        # ── C) Jugement LLM ──────────────────────────────────────────────
        # Prépare le contexte corpus pour le LLM
        docs_contexte = docs_candidats[:3] if docs_candidats else []
        if not docs_contexte and corpus:
            # Fallback : quelques docs du corpus pour ancrage
            docs_contexte = corpus[:3]

        contexte_corpus = "\n".join(
            f"• {d.get('titre','')[:80]} | "
            f"Auteurs: {', '.join(str(a) for a in (d.get('auteurs') or [])[:3])} | "
            f"Année: {d.get('annee','?')} | "
            f"Résumé: {str(d.get('resume','') or d.get('abstract',''))[:120]}"
            for d in docs_contexte
        )

        prompt_verif = f"""Tu es un expert en vérification de citations académiques.

AFFIRMATION À VÉRIFIER :
"{texte_claim}"

CITATION MENTIONNÉE DANS LE TEXTE : {citation_mentee or "Aucune"}

SOURCES DISPONIBLES DANS LE CORPUS :
{contexte_corpus or "Aucune source identifiée par l'auteur/année."}

PASSAGES SÉMANTIQUEMENT PROCHES (RAG) :
{passages_rag or "Aucun passage RAG disponible."}

CONSIGNE :
Détermine le statut de cette affirmation parmi :
- "valide"          : l'affirmation est confirmée par une source du corpus
- "approximatif"    : l'affirmation est proche mais avec inexactitude mineure (année, formulation)
- "hallucination"   : la source citée n'existe pas dans le corpus ou est inventée
- "non_verifiable"  : impossible de confirmer ou infirmer avec les sources disponibles
- "manque_reference": l'affirmation est plausible mais aucune citation n'est fournie

Réponds UNIQUEMENT avec un JSON valide :
{{
  "statut": "valide | approximatif | hallucination | non_verifiable | manque_reference",
  "justification": "Explication courte (1-2 phrases) du verdict.",
  "score_confiance": 0.0 à 1.0,
  "suggestion": "Correction proposée ou source à citer (si applicable)" ou null
}}"""

        raw_verif = self._appel_llm(prompt_verif, max_tokens=400)
        verif = self._parser_json_objet(raw_verif, contexte=f"vérif claim '{texte_claim[:40]}'")

        if verif:
            resultat["methode_verif"].append("LLM")
            resultat["statut"]          = verif.get("statut", self.STATUT_NON_VERIF)
            resultat["justification"]   = verif.get("justification", "")
            resultat["score_confiance"] = float(verif.get("score_confiance", 0.0))
            resultat["suggestion"]      = verif.get("suggestion")
        else:
            # Fix #7 — fallback si LLM muet
            if docs_candidats:
                resultat["statut"]        = self.STATUT_APPROX
                resultat["justification"] = "Source trouvée structurellement mais non confirmée par LLM."
                resultat["score_confiance"] = 0.5
            elif not citation_mentee:
                resultat["statut"]        = self.STATUT_MANQUE_REF
                resultat["justification"] = "Aucune citation fournie et aucune source trouvée."
                resultat["score_confiance"] = 0.3

        return resultat

    # ─────────────────────────────────────────────────────────────────────────
    #  ÉTAPE 3 — AUDIT DE LA BIBLIOGRAPHIE
    # ─────────────────────────────────────────────────────────────────────────

    def _auditer_bibliographie(self, texte_revue: str, corpus: list) -> dict:
        """
        Vérifie la cohérence entre les références citées dans la revue
        et les articles du corpus reçu.

        Détecte :
        - Citations fantômes (citées dans la revue mais absentes du corpus)
        - Articles non cités (dans le corpus mais jamais mentionnés)
        - Taux de couverture
        """
        logger.info("Audit bibliographie...")

        # Extraction de toutes les citations (pattern "Auteur, AAAA" ou "Auteur et al., AAAA")
        pattern_citations = re.findall(
            r'([A-Z][a-zÀ-ÿ]+(?:\s+et\s+al\.)?),?\s*\(?(\d{4})\)?',
            texte_revue
        )

        citations_dans_revue = {}
        for auteur, annee in pattern_citations:
            cle = f"{auteur.strip()}, {annee}"
            citations_dans_revue[cle] = citations_dans_revue.get(cle, 0) + 1

        # Auteurs et années du corpus réel
        corpus_refs = set()
        for doc in corpus:
            auteurs = doc.get("auteurs") or []
            if isinstance(auteurs, str):
                auteurs = [auteurs]
            annee = str(doc.get("annee", "")).strip()
            for aut in auteurs:
                nom = str(aut).strip().split()[-1] if str(aut).strip() else ""
                if nom and annee:
                    corpus_refs.add(f"{nom}, {annee}")

        # Détection des citations fantômes
        citations_fantomes = []
        citations_matchees = []
        for cit, freq in citations_dans_revue.items():
            auteur_cit = cit.split(",")[0].strip().lower()
            annee_cit  = cit.split(",")[1].strip() if "," in cit else ""
            trouve = any(
                auteur_cit in ref.lower() and annee_cit in ref
                for ref in corpus_refs
            )
            if trouve:
                citations_matchees.append(cit)
            else:
                citations_fantomes.append({"citation": cit, "occurrences": freq})

        # Articles du corpus jamais cités
        articles_non_cites = []
        for doc in corpus:
            auteurs = doc.get("auteurs") or []
            if isinstance(auteurs, str):
                auteurs = [auteurs]
            annee = str(doc.get("annee", "")).strip()
            est_cite = False
            for aut in auteurs:
                nom = str(aut).strip().split()[-1].lower() if str(aut).strip() else ""
                if any(nom in cit.lower() and annee in cit
                       for cit in citations_dans_revue):
                    est_cite = True
                    break
            if not est_cite:
                articles_non_cites.append({
                    "titre":   doc.get("titre","")[:80],
                    "auteurs": auteurs[:3],
                    "annee":   annee
                })

        # Fix #7 — calcul sécurisé du taux de couverture
        nb_cites_uniques = len(citations_matchees)
        nb_corpus        = len(corpus)
        taux_couverture  = round(nb_cites_uniques / nb_corpus * 100, 1) if nb_corpus > 0 else 0.0

        # Verdict global
        if len(citations_fantomes) == 0 and taux_couverture >= 80:
            verdict = "coherente"
        elif len(citations_fantomes) <= 2 and taux_couverture >= 60:
            verdict = "acceptable"
        elif len(citations_fantomes) > 5 or taux_couverture < 40:
            verdict = "problematique"
        else:
            verdict = "a_verifier"

        return {
            "verdict_bibliographie":   verdict,
            "taux_couverture_corpus":  taux_couverture,
            "nb_articles_cites":       nb_cites_uniques,
            "nb_articles_corpus":      nb_corpus,
            "nb_citations_fantomes":   len(citations_fantomes),
            "citations_fantomes":      citations_fantomes[:20],
            "articles_non_cites":      articles_non_cites[:20],
            "toutes_citations_revue":  list(citations_dans_revue.keys())[:50]
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #5 — SAUVEGARDE INCRÉMENTALE
    # ─────────────────────────────────────────────────────────────────────────

    def _sauvegarder_section_incrementale(self, claims: list,
                                           dossier_output: str,
                                           section_index: int,
                                           titre_section: str):
        path = Path(dossier_output) / f"claims_section_{section_index:02d}_partiel.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "section_index": section_index,
                    "titre":         titre_section,
                    "claims":        claims
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"Partiel sauvegardé → {path.name}")
        except Exception as e:
            logger.warning(f"Échec sauvegarde partielle section {section_index} : {e}")

    def _charger_sections_partielles(self, dossier_output: str) -> dict:
        partiels = {}
        for f in sorted(Path(dossier_output).glob("claims_section_*_partiel.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                idx = int(f.stem.split("_")[2])
                partiels[idx] = data.get("claims", [])
                logger.info(f"Reprise : section partielle chargée → {f.name}")
            except Exception as e:
                logger.warning(f"Impossible de charger {f.name} : {e}")
        return partiels

    def _nettoyer_partiels(self, dossier_output: str):
        for f in Path(dossier_output).glob("claims_section_*_partiel.json"):
            try:
                f.unlink()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Fix #10 — GÉNÉRATION DU RAPPORT MARKDOWN ANNOTÉ
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_rapport_md(self, rapport: dict, dossier_output: str):
        """
        Génère deux fichiers :
        1. rapport_citations.md — audit complet avec stats
        2. revue_corrigee.md   — revue annotée claim par claim

        Fix #10 : tous les champs ont des fallbacks, pas de KeyError possible.
        """
        meta   = rapport.get("meta", {})
        stats  = rapport.get("stats_claims", {})
        biblio = rapport.get("audit_bibliographie", {})
        claims = rapport.get("claims_classifies", [])

        # ── Rapport d'audit ───────────────────────────────────────────────
        lignes = []
        lignes.append("# Rapport d'Audit des Citations\n")
        lignes.append(f"*{meta.get('nb_claims_total', 0)} affirmations vérifiées "
                      f"| Score fiabilité : {meta.get('score_fiabilite', 0):.1f}%*\n")
        lignes.append("---\n")

        # Stats globales
        lignes.append("## Résumé statistique\n")
        lignes.append(f"| Statut | Nombre |")
        lignes.append(f"|--------|--------|")
        for statut, nb in stats.items():
            lignes.append(f"| {statut} | {nb} |")
        lignes.append("")

        # Audit bibliographie
        lignes.append("## Audit bibliographie\n")
        verdict = biblio.get("verdict_bibliographie", "inconnu")
        emoji_verdict = {
            "coherente":     "✅",
            "acceptable":    "🟡",
            "a_verifier":    "🟠",
            "problematique": "❌"
        }.get(verdict, "❓")
        lignes.append(f"**Verdict** : {emoji_verdict} {verdict.upper()}\n")
        lignes.append(f"- Taux de couverture du corpus : {biblio.get('taux_couverture_corpus', 0)}%")
        lignes.append(f"- Articles cités / corpus total : "
                      f"{biblio.get('nb_articles_cites', 0)} / {biblio.get('nb_articles_corpus', 0)}")
        lignes.append(f"- Citations fantômes détectées : {biblio.get('nb_citations_fantomes', 0)}\n")

        # Citations fantômes
        fantomes = biblio.get("citations_fantomes", [])
        if fantomes:
            lignes.append("### Citations fantômes (absentes du corpus)\n")
            for c in fantomes[:10]:
                cit  = c.get("citation", "?") if isinstance(c, dict) else str(c)
                freq = c.get("occurrences", 1) if isinstance(c, dict) else 1
                lignes.append(f"- `{cit}` (mentionnée {freq}×)")
            lignes.append("")

        # Articles non cités
        non_cites = biblio.get("articles_non_cites", [])
        if non_cites:
            lignes.append("### Articles du corpus jamais cités\n")
            for a in non_cites[:10]:
                titre   = a.get("titre", "?") if isinstance(a, dict) else str(a)
                auteurs = a.get("auteurs", []) if isinstance(a, dict) else []
                annee   = a.get("annee", "?") if isinstance(a, dict) else "?"
                lignes.append(f"- {titre} ({', '.join(str(x) for x in auteurs[:2])}, {annee})")
            lignes.append("")

        # Hallucinations détectées
        hallus = [c for c in claims if c.get("statut") == self.STATUT_HALLU]
        if hallus:
            lignes.append("## ⚠ Hallucinations détectées\n")
            for h in hallus[:10]:
                lignes.append(f"- **Claim** : {h.get('claim','')[:100]}")
                lignes.append(f"  **Citation** : {h.get('citation_mentionnee','N/A')}")
                lignes.append(f"  **Justification** : {h.get('justification','')}")
                if h.get("suggestion"):
                    lignes.append(f"  **Suggestion** : {h['suggestion']}")
                lignes.append("")

        # Claims sans référence
        sans_ref = [c for c in claims if c.get("statut") == self.STATUT_MANQUE_REF]
        if sans_ref:
            lignes.append("## 📌 Affirmations sans référence\n")
            for s in sans_ref[:10]:
                lignes.append(f"- {s.get('claim','')[:120]}")
            lignes.append("")

        rapport_path = Path(dossier_output) / "rapport_citations.md"
        with open(rapport_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes))
        logger.info(f"Rapport citations MD → {rapport_path}")

        # ── Revue annotée ─────────────────────────────────────────────────
        EMOJI_STATUT = {
            self.STATUT_VALIDE:     "✅",
            self.STATUT_APPROX:     "🟡",
            self.STATUT_HALLU:      "❌",
            self.STATUT_NON_VERIF:  "❓",
            self.STATUT_MANQUE_REF: "📌"
        }

        lignes_corrigee = []
        lignes_corrigee.append("# Revue de Littérature — Version Annotée\n")
        lignes_corrigee.append(
            "*Les annotations [✅❌🟡❓📌] indiquent le statut de chaque affirmation vérifiée.*\n"
        )
        lignes_corrigee.append("---\n")

        for claim in claims:
            texte   = claim.get("claim", "")
            statut  = claim.get("statut", self.STATUT_NON_VERIF)
            emoji   = EMOJI_STATUT.get(statut, "❓")
            justif  = claim.get("justification", "")
            suggest = claim.get("suggestion", "")

            lignes_corrigee.append(f"{emoji} *{texte}*")
            if justif:
                lignes_corrigee.append(f"   > {justif}")
            if suggest and statut in (self.STATUT_HALLU, self.STATUT_MANQUE_REF):
                lignes_corrigee.append(f"   > 💡 {suggest}")
            lignes_corrigee.append("")

        revue_path = Path(dossier_output) / "revue_corrigee.md"
        with open(revue_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes_corrigee))
        logger.info(f"Revue corrigée MD → {revue_path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, chemin_revue_json: str,
            chemin_revue_md: str,
            chemin_corpus: str,
            dossier_output: str,
            indexeur=None,
            reprendre_partiels: bool = True) -> dict:
        """
        Exécution complète de l'audit de citations.

        Paramètres
        ----------
        reprendre_partiels : bool
            Si True, recharge les sections déjà vérifiées après un crash.
        """
        global logger
        logger = _configurer_logging(dossier_output)

        logger.info("=" * 60)
        logger.info("  AGENT CITATEUR — DÉMARRAGE")
        logger.info("=" * 60)
        debut = time.time()
        Path(dossier_output).mkdir(parents=True, exist_ok=True)

        # ── Chargement ────────────────────────────────────────────────────
        corpus, revue_json, texte_revue = self._charger(
            chemin_revue_json, chemin_revue_md, chemin_corpus
        )
        logger.info(
            f"{len(corpus)} articles corpus | "
            f"Revue : {len(texte_revue)} caractères"
        )

        if not texte_revue.strip():
            logger.error("La revue est vide. Abandon.")
            return {}

        # ── Index corpus ──────────────────────────────────────────────────
        index_corpus = self._construire_index_corpus(corpus)
        logger.info(
            f"Index corpus : {len(index_corpus['par_auteur'])} auteurs, "
            f"{len(index_corpus['par_annee'])} années"
        )

        # ── Segmentation de la revue ──────────────────────────────────────
        sections = self._segmenter_revue(texte_revue)

        # ── Fix #5 : chargement des partiels ─────────────────────────────
        partiels = {}
        if reprendre_partiels:
            partiels = self._charger_sections_partielles(dossier_output)

        # ── Étape 1+2 : Extraction et vérification des claims ────────────
        logger.info("[Étape 1+2] Extraction et vérification des claims par section...")
        tous_claims_verifies = []

        for idx, section in enumerate(sections, start=1):
            logger.info(
                f"  Section {idx}/{len(sections)} : '{section['titre']}'"
            )

            # Fix #5 : utilise le partiel si disponible
            if idx in partiels:
                logger.info(f"  → chargée depuis partiel (skip LLM)")
                tous_claims_verifies.extend(partiels[idx])
                continue

            # Extraction des claims
            claims_bruts = self._extraire_claims_section(section)
            logger.info(f"    {len(claims_bruts)} claims extraits")

            if not claims_bruts:
                self._sauvegarder_section_incrementale(
                    [], dossier_output, idx, section["titre"]
                )
                time.sleep(5)
                continue

            # Vérification de chaque claim
            claims_verifies = []
            for i, claim in enumerate(claims_bruts):
                logger.debug(
                    f"    Vérif claim {i+1}/{len(claims_bruts)} : "
                    f"{claim.get('claim','')[:50]}..."
                )
                claim_verifie = self._verifier_claim(
                    claim, corpus, index_corpus, indexeur
                )
                claims_verifies.append(claim_verifie)
                time.sleep(5)  # pause entre les appels LLM

            tous_claims_verifies.extend(claims_verifies)

            # Fix #5 : sauvegarde incrémentale de la section
            self._sauvegarder_section_incrementale(
                claims_verifies, dossier_output, idx, section["titre"]
            )

            # Pause inter-sections
            if idx < len(sections):
                logger.debug("Pause inter-sections : 5s")
                time.sleep(5)

        # ── Étape 3 : Audit bibliographie ─────────────────────────────────
        logger.info("[Étape 3] Audit bibliographie...")
        audit_biblio = self._auditer_bibliographie(texte_revue, corpus)

        # ── Calcul des statistiques ───────────────────────────────────────
        stats = Counter(c.get("statut", self.STATUT_NON_VERIF)
                        for c in tous_claims_verifies)

        nb_total = len(tous_claims_verifies)

        # Fix #7 — score fiabilité sans division par zéro
        nb_valides = stats.get(self.STATUT_VALIDE, 0) + stats.get(self.STATUT_APPROX, 0)
        score_fiabilite = round(nb_valides / nb_total * 100, 1) if nb_total > 0 else 0.0

        # ── Rapport final ─────────────────────────────────────────────────
        rapport = {
            "meta": {
                "nb_claims_total":    nb_total,
                "score_fiabilite":    score_fiabilite,
                "nb_articles_corpus": len(corpus),
                "nb_sections":        len(sections),
                "duree_secondes":     round(time.time() - debut, 1)
            },
            "stats_claims":       dict(stats),
            "claims_classifies":  tous_claims_verifies,
            "audit_bibliographie": audit_biblio
        }

        # Sauvegarde JSON
        json_path = Path(dossier_output) / "rapport_citations.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
        logger.info(f"Rapport JSON → {json_path}")

        # Rapport Markdown + revue annotée
        self._generer_rapport_md(rapport, dossier_output)

        # Fix #5 : nettoyage des partiels après succès
        self._nettoyer_partiels(dossier_output)

        # Résumé console
        logger.info("=" * 60)
        logger.info("  RÉSUMÉ AGENT CITATEUR")
        logger.info("=" * 60)
        logger.info(f"Claims vérifiés    : {nb_total}")
        logger.info(f"Score fiabilité    : {score_fiabilite}%")
        logger.info(f"Hallucinations     : {stats.get(self.STATUT_HALLU, 0)}")
        logger.info(f"Sans référence     : {stats.get(self.STATUT_MANQUE_REF, 0)}")
        logger.info(f"Citations fantômes : {audit_biblio.get('nb_citations_fantomes', 0)}")
        logger.info(f"Durée totale       : {rapport['meta']['duree_secondes']}s")

        return rapport


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Fix #8 — Import protégé avec message d'erreur explicite
    indexeur = None
    try:
        from agent_indexeur import AgentIndexeur
        indexeur = AgentIndexeur(dossier_chroma="data/chroma_db")
        print("[Citateur] AgentIndexeur chargé — vérification RAG activée.")
    except ModuleNotFoundError as e:
        print(
            f"[AVERTISSEMENT] Impossible d'importer AgentIndexeur : {e}\n"
            "La vérification RAG sera désactivée (uniquement structurelle + LLM).\n"
            "Si votre fichier s'appelle 'agent_indexeur_v2.py', adaptez l'import."
        )
    except ImportError as e:
        print(
            f"[AVERTISSEMENT] Erreur d'import AgentIndexeur : {e}\n"
            "Vérifiez les dépendances (chromadb, sentence-transformers…).\n"
            "La vérification RAG sera désactivée."
        )

    agent = AgentCitateur()
    rapport = agent.run(
        chemin_revue_json  = "data/corpus/revue_litterature.json",
        chemin_revue_md    = "data/corpus/revue_litterature.md",
        chemin_corpus      = "data/corpus/corpus_complet.json",
        dossier_output     = "data/corpus",
        indexeur           = indexeur,
        reprendre_partiels = True
    )