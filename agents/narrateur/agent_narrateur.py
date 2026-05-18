"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            AGENT NARRATEUR — SMA Revue de Littérature v2                   ║
║                                                                              ║
║  Corrections v2 :                                                            ║
║  ✓ Retry + backoff exponentiel sur TOUS les appels LLM                      ║
║  ✓ except granulaire : rate limit ≠ erreur fatale                           ║
║  ✓ _generer_conclusion reçoit un résumé réel des sections                   ║
║  ✓ _generer_introduction reçoit les métadonnées complètes de la carte       ║
║  ✓ sleep(3) entre chaque appel LLM (rate limit Groq)                        ║
║  ✓ Filtre citations [source inconnue] dans _construire_contexte_theme       ║
║  ✓ GROQ_API_KEY vérifiée au démarrage (fail-fast)                           ║
║  ✓ IndexError si corpus vide géré proprement                                ║
║  ✓ logging structuré avec fichier narrateur.log                             ║
║  ✓ Limite de taille du contexte (MAX_CONTEXTE_CHARS)                        ║
║  ✓ Titre de la revue basé sur le sujet global, pas le premier thème         ║
║  ✓ Sauvegarde incrémentale des sections (protection contre crash)           ║
║                                                                              ║
║  Reçoit   : carte_corpus.json  (Agent Cartographe)                          ║
║             AgentIndexeur      (recherche sémantique dans les chunks)        ║
║             corpus_complet.json (métadonnées complètes)                      ║
║                                                                              ║
║  Produit  : revue_litterature.md  (Markdown académique avec citations)       ║
║             revue_litterature.json (structure machine-readable)              ║
║             sections/             (sections sauvegardées individuellement)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation :
    pip install groq python-dotenv

Usage :
    python agent_narrateur_v2.py
"""

import os
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("narrateur.log", encoding="utf-8")
    ]
)
log = logging.getLogger("AgentNarrateur")

# ── Vérification clé API au démarrage (fail-fast) ────────────────────────────
_API_KEY = os.getenv("GROQ_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "\n[ERREUR] GROQ_API_KEY manquante.\n"
        "Crée un fichier .env à la racine du projet avec :\n"
        "    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx\n"
        "Obtiens ta clé sur : https://console.groq.com"
    )

# ── Constantes ────────────────────────────────────────────────────────────────
MAX_CONTEXTE_CHARS  = 6000   # Limite du contexte injecté dans le prompt
SLEEP_ENTRE_APPELS  = 3      # Secondes entre chaque appel LLM (rate limit)
N_CHUNKS_PAR_THEME  = 6      # Chunks RAG récupérés par thème


# ═════════════════════════════════════════════════════════════════════════════
#  1. GESTIONNAIRE DE CITATIONS
# ═════════════════════════════════════════════════════════════════════════════

class GestionnaireCitations:
    """
    Maintient un registre de toutes les citations utilisées dans la revue.
    Produit la bibliographie finale en format académique (APA simplifié).
    """

    def __init__(self, corpus: list):
        self.index               = {doc["doc_id"]: doc for doc in corpus if doc.get("doc_id")}
        self.citations_utilisees = {}   # doc_id → numéro d'ordre
        self.compteur            = 0

    def doc_connu(self, doc_id: str) -> bool:
        """✅ NOUVEAU : vérifie qu'un doc_id existe dans le corpus avant de le citer."""
        return doc_id in self.index

    def citer(self, doc_id: str) -> str:
        """Retourne la clé de citation [Auteur, Année] et enregistre l'usage."""
        if not doc_id or doc_id not in self.index:
            return None  # ✅ CORRIGÉ : None au lieu de "[source inconnue]"

        if doc_id not in self.citations_utilisees:
            self.compteur += 1
            self.citations_utilisees[doc_id] = self.compteur

        doc     = self.index[doc_id]
        auteurs = doc.get("auteurs", [])
        annee   = doc.get("annee", "s.d.")

        if auteurs:
            premier = str(auteurs[0]).split()[-1]
            if len(auteurs) > 2:
                cle = f"[{premier} et al., {annee}]"
            elif len(auteurs) == 2:
                second = str(auteurs[1]).split()[-1]
                cle = f"[{premier} & {second}, {annee}]"
            else:
                cle = f"[{premier}, {annee}]"
        else:
            cle = f"[Inconnu, {annee}]"

        return cle

    def get_doc(self, doc_id: str) -> dict:
        return self.index.get(doc_id, {})

    def generer_bibliographie(self) -> str:
        if not self.citations_utilisees:
            return "## Références\n\n_Aucune citation enregistrée._"

        lignes = ["## Références\n"]
        triees = sorted(self.citations_utilisees.items(), key=lambda x: x[1])

        for doc_id, _ in triees:
            doc     = self.index.get(doc_id, {})
            auteurs = doc.get("auteurs", ["Auteur inconnu"])
            annee   = doc.get("annee", "s.d.")
            titre   = doc.get("titre", "Titre inconnu")
            journal = doc.get("journal_ou_conference", "")
            doi     = doc.get("doi", "")

            auteurs_str = ", ".join(str(a) for a in auteurs[:3])
            if len(auteurs) > 3:
                auteurs_str += " et al."

            ref = f"- {auteurs_str} ({annee}). *{titre}*."
            if journal:
                ref += f" _{journal}_."
            if doi:
                ref += f" https://doi.org/{doi}"

            lignes.append(ref)

        return "\n".join(lignes)

    def get_registre(self) -> list:
        registre = []
        for doc_id, ordre in self.citations_utilisees.items():
            doc = self.index.get(doc_id, {})
            registre.append({
                "ordre_apparition": ordre,
                "doc_id":           doc_id,
                "titre":            doc.get("titre", ""),
                "auteurs":          doc.get("auteurs", []),
                "annee":            doc.get("annee"),
                "doi":              doc.get("doi"),
                "fichier_source":   doc.get("fichier_source", "")
            })
        return sorted(registre, key=lambda x: x["ordre_apparition"])


# ═════════════════════════════════════════════════════════════════════════════
#  2. AGENT NARRATEUR v2
# ═════════════════════════════════════════════════════════════════════════════

class AgentNarrateur:
    """
    Génère la revue de littérature section par section avec :
    - Retry automatique sur rate limit Groq
    - Contexte riche injecté dans chaque prompt
    - Sauvegarde incrémentale des sections
    - Citations filtrées (plus de [source inconnue])
    """

    def __init__(self):
        self.client = Groq(api_key=_API_KEY)
        self.model  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        log.info(f"Agent Narrateur initialisé | modèle : {self.model}")

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITAIRE LLM — retry + backoff exponentiel
    # ─────────────────────────────────────────────────────────────────────────

    def _appel_llm(self, prompt: str, max_tokens: int = 1200,
                   temperature: float = 0.4) -> str:
        """
        ✅ CORRIGÉ : retry avec backoff sur rate limit.
        ✅ CORRIGÉ : except granulaire — rate limit ≠ erreur fatale.
        Retourne le texte généré, ou lève une exception après 4 tentatives.
        """
        delais = [15, 30, 60, 120]
        for tentative in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err = str(e).lower()
                est_retryable = any(k in err for k in
                                    ["429", "rate", "timeout", "overloaded", "503"])

                if tentative < 3 and est_retryable:
                    delai = delais[tentative]
                    log.warning(f"Groq rate limit — retry {tentative+1}/3 dans {delai}s")
                    time.sleep(delai)
                else:
                    # Erreur non-récupérable (clé invalide, JSON cassé, etc.)
                    log.error(f"Groq échec définitif ({type(e).__name__}) : {e}")
                    raise

        return ""  # jamais atteint

    # ─────────────────────────────────────────────────────────────────────────
    #  CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _charger_carte(self, chemin: str) -> dict:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)

    def _charger_corpus(self, chemin: str) -> list:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ordonner_themes(self, carte: dict) -> list:
        """Trie les thèmes par année de première publication (chronologique)."""
        themes = carte.get("themes", {})
        return sorted(
            themes.items(),
            key=lambda x: (
                x[1].get("evolution_temporelle", {})
                    .get("annee_premiere_publication") or 9999
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  CONSTRUCTION DU CONTEXTE THÉMATIQUE
    # ─────────────────────────────────────────────────────────────────────────

    def _construire_contexte_theme(self, theme: dict, indexeur,
                                    citations: GestionnaireCitations,
                                    requete_supplement: str = "") -> str:
        """
        ✅ CORRIGÉ : filtre les citations [source inconnue].
        ✅ CORRIGÉ : limite MAX_CONTEXTE_CHARS pour ne pas dépasser le contexte LLM.
        """
        nom_theme  = theme.get("nom_theme", "")
        evolution  = theme.get("evolution_temporelle", {})
        methodes   = theme.get("methodes_dominantes", [])

        contexte = (
            f"THÈME : {nom_theme}\n"
            f"Tendance : {evolution.get('tendance', '?')} | "
            f"Depuis : {evolution.get('annee_premiere_publication', '?')} | "
            f"Articles récents (5 ans) : {evolution.get('nb_articles_5_dernieres_annees', 0)}\n"
            f"Méthodes dominantes : {', '.join(methodes) if methodes else 'non renseignées'}\n\n"
            f"ARTICLES DE CE THÈME :\n"
        )

        # Articles du thème (avec filtre doc_id connu)
        for art in theme.get("articles", []):
            doc_id = art.get("doc_id", "")
            if not citations.doc_connu(doc_id):
                continue
            doc          = citations.get_doc(doc_id)
            cle_citation = citations.citer(doc_id)
            contrib      = str(doc.get("contribution_principale", "") or "")[:150]
            contexte += (
                f"- {cle_citation} : {str(doc.get('titre',''))[:80]}\n"
                f"  Contribution : {contrib}\n\n"
            )

        # Chunks RAG — recherche sémantique dans l'Indexeur
        if indexeur:
            requete    = f"{nom_theme} {requete_supplement}".strip()
            chunks_rag = indexeur.rechercher(requete, n_resultats=N_CHUNKS_PAR_THEME)

            if chunks_rag:
                contexte += "\nPASSAGES PERTINENTS (extraits des articles) :\n"
                for chunk in chunks_rag:
                    doc_id = chunk.get("doc_id", "")

                    # ✅ CORRIGÉ : ne citer que les doc_id connus du corpus
                    if not citations.doc_connu(doc_id):
                        continue

                    cle     = citations.citer(doc_id)
                    extrait = str(chunk.get("texte", ""))[:300]
                    contexte += f"{cle} : {extrait}\n\n"

        # ✅ CORRIGÉ : limite de taille pour éviter le dépassement de tokens
        if len(contexte) > MAX_CONTEXTE_CHARS:
            contexte = contexte[:MAX_CONTEXTE_CHARS] + "\n[...contexte tronqué...]"

        return contexte

    # ─────────────────────────────────────────────────────────────────────────
    #  GÉNÉRATION DE L'INTRODUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_introduction(self, carte: dict,
                               themes_ordonnes: list,
                               citations: GestionnaireCitations) -> str:
        """
        ✅ CORRIGÉ : le LLM reçoit les métadonnées complètes de la carte,
        pas juste une liste de noms de thèmes.
        """
        log.info("Génération : Introduction...")

        meta       = carte.get("meta", {})
        nb_articles = meta.get("nb_articles_total", "?")
        narrative   = meta.get("analyse_narrative_globale", "")

        # Évolution globale
        ev_globale  = carte.get("evolution_globale", {})
        annees      = ev_globale.get("annees_couvertes", [])
        periode     = (f"{min(annees)}–{max(annees)}"
                       if len(annees) >= 2 else str(annees[0]) if annees else "inconnue")

        # Résumé des thèmes
        resume_themes = ""
        for tid, theme in themes_ordonnes:
            nom      = theme.get("nom_theme", "?")
            nb_art   = theme.get("nb_articles", 0)
            tendance = theme.get("evolution_temporelle", {}).get("tendance", "?")
            resume_themes += f"- {nom} ({nb_art} articles, tendance : {tendance})\n"

        # Méthodes globales
        carte_methodes = carte.get("carte_methodes_globale", {})
        methodes_top   = list(carte_methodes.get("methodes_frequences", {}).keys())[:5]

        prompt = f"""Tu es un expert en rédaction académique. Rédige l'introduction d'une revue de littérature scientifique.

CONTEXTE DU CORPUS :
- Période couverte : {periode}
- Nombre total d'articles analysés : {nb_articles}
- Thèmes identifiés :
{resume_themes}
- Méthodes les plus utilisées dans le corpus : {', '.join(methodes_top) if methodes_top else 'variées'}
- Analyse narrative globale : {narrative[:500] if narrative else 'non disponible'}

INSTRUCTIONS :
- Rédige une introduction académique de 4 à 6 paragraphes
- Présente la problématique générale du domaine
- Justifie l'intérêt d'une revue de littérature sur ce sujet
- Annonce le plan de la revue (les thèmes qui seront traités)
- Utilise un ton scientifique, précis et structuré
- Ne cite pas directement les articles dans l'introduction
- Écris directement en français académique, sans titres ni puces"""

        try:
            texte = self._appel_llm(prompt, max_tokens=1000, temperature=0.35)
            log.info(f"Introduction générée ({len(texte.split())} mots)")
            return texte
        except Exception as e:
            log.error(f"Introduction échouée : {e}")
            return "_Introduction non générée suite à une erreur LLM._"

    # ─────────────────────────────────────────────────────────────────────────
    #  GÉNÉRATION D'UNE SECTION THÉMATIQUE
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_section_theme(self, theme_id, theme: dict, carte: dict,
                                indexeur, citations: GestionnaireCitations,
                                themes_precedents: list) -> str:
        """
        Génère une section académique complète pour un thème.
        Inclut le contexte des thèmes déjà traités pour assurer la cohérence narrative.
        """
        nom_theme = theme.get("nom_theme", f"Thème {theme_id}")
        log.info(f"Génération : Section '{nom_theme}'...")

        contexte = self._construire_contexte_theme(theme, indexeur, citations)

        # Contexte de transition narrative
        transition = ""
        if themes_precedents:
            noms_precedents = ", ".join(themes_precedents[-2:])
            transition = (
                f"\nNOTE DE TRANSITION : Cette section fait suite aux thèmes '{noms_precedents}'. "
                f"Assure une transition narrative cohérente depuis ces thèmes précédents.\n"
            )

        # Mots-clés représentatifs pour orienter la rédaction
        mots_cles = theme.get("mots_cles_representatifs", [])
        sous_themes = theme.get("sous_themes", [])

        prompt = f"""Tu es un expert en rédaction académique. Rédige une section complète d'une revue de littérature.

{contexte}
{transition}
MOTS-CLÉS DU THÈME : {', '.join(mots_cles) if mots_cles else 'non renseignés'}
SOUS-THÈMES : {', '.join(sous_themes) if sous_themes else 'non renseignés'}
RÉSUMÉ NARRATIF DU THÈME : {theme.get('resume_narratif', '')}

INSTRUCTIONS :
- Rédige une section académique de 5 à 8 paragraphes sur le thème "{nom_theme}"
- Synthétise les contributions des articles listés (ne résume pas article par article)
- Cite les auteurs au format [Nom, Année] ou [Nom et al., Année] dans le texte
- Présente les débats, convergences et divergences entre auteurs
- Mentionne les méthodes utilisées et leurs apports
- Conclus la section en ouvrant sur les limites et les manques identifiés
- Écris directement en français académique, sans sous-titres ni puces"""

        try:
            texte = self._appel_llm(prompt, max_tokens=1500, temperature=0.4)
            log.info(f"Section '{nom_theme}' générée ({len(texte.split())} mots)")
            return texte
        except Exception as e:
            log.error(f"Section '{nom_theme}' échouée : {e}")
            return f"_Section '{nom_theme}' non générée suite à une erreur LLM._"

    # ─────────────────────────────────────────────────────────────────────────
    #  GÉNÉRATION DE LA CONCLUSION
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_conclusion(self, carte: dict, sections: dict,
                             themes_ordonnes: list,
                             citations: GestionnaireCitations) -> str:
        """
        ✅ CORRIGÉ : la conclusion reçoit un résumé réel des sections générées.
        Le LLM sait de quoi parle la revue.
        """
        log.info("Génération : Conclusion...")

        # Résumé des sections thématiques (extraits des 300 premiers chars)
        resume_sections = ""
        for tid, theme in themes_ordonnes:
            cle_section = f"theme_{tid}"
            nom         = theme.get("nom_theme", f"Thème {tid}")
            texte_sect  = sections.get(cle_section, "")
            extrait     = texte_sect[:300].replace("\n", " ") if texte_sect else "non généré"
            resume_sections += f"- **{nom}** : {extrait}...\n\n"

        # Méthodes globales et tendances
        carte_methodes  = carte.get("carte_methodes_globale", {})
        methodes_top    = list(carte_methodes.get("methodes_frequences", {}).keys())[:5]
        ev_globale      = carte.get("evolution_globale", {})
        annees          = ev_globale.get("annees_couvertes", [])
        periode         = (f"{min(annees)}–{max(annees)}"
                           if len(annees) >= 2 else "période inconnue")

        # Thèmes avec tendance croissante / émergente
        themes_emergents = [
            theme.get("nom_theme", "?")
            for _, theme in themes_ordonnes
            if theme.get("evolution_temporelle", {}).get("tendance")
               in ("croissante", "emergente")
        ]

        prompt = f"""Tu es un expert en rédaction académique. Rédige la conclusion d'une revue de littérature scientifique.

RÉSUMÉ DES SECTIONS RÉDIGÉES :
{resume_sections}

DONNÉES GLOBALES DU CORPUS :
- Période couverte : {periode}
- Thèmes traités : {', '.join([t.get('nom_theme','') for _, t in themes_ordonnes])}
- Méthodes dominantes dans le corpus : {', '.join(methodes_top) if methodes_top else 'variées'}
- Thèmes en croissance ou émergents : {', '.join(themes_emergents) if themes_emergents else 'aucun identifié'}
- Nombre de références citées : {len(citations.citations_utilisees)}

INSTRUCTIONS :
- Rédige une conclusion académique de 4 à 6 paragraphes
- Synthétise les principales contributions identifiées dans chaque thème
- Mets en évidence les convergences et les tensions entre thèmes
- Identifie au moins 3 pistes de recherche futures concrètes et justifiées
- Souligne les limites méthodologiques transversales du corpus
- Conclue par une perspective d'ensemble sur l'évolution du domaine
- Écris directement en français académique, sans titres ni puces"""

        try:
            texte = self._appel_llm(prompt, max_tokens=900, temperature=0.35)
            log.info(f"Conclusion générée ({len(texte.split())} mots)")
            return texte
        except Exception as e:
            log.error(f"Conclusion échouée : {e}")
            return "_Conclusion non générée suite à une erreur LLM._"

    # ─────────────────────────────────────────────────────────────────────────
    #  SAUVEGARDE INCRÉMENTALE
    # ─────────────────────────────────────────────────────────────────────────

    def _sauvegarder_section(self, nom: str, texte: str, dossier: Path):
        """
        ✅ NOUVEAU : sauvegarde chaque section au fur et à mesure.
        Permet de récupérer le travail en cas de crash en milieu de génération.
        """
        dossier_sections = dossier / "sections"
        dossier_sections.mkdir(exist_ok=True)
        chemin = dossier_sections / f"{nom}.txt"
        chemin.write_text(texte, encoding="utf-8")
        log.info(f"Section sauvegardée → {chemin}")

    def _charger_section_cache(self, nom: str, dossier: Path) -> str:
        """Charge une section déjà générée si elle existe (reprise après crash)."""
        chemin = dossier / "sections" / f"{nom}.txt"
        if chemin.exists():
            log.info(f"Section '{nom}' rechargée depuis le cache")
            return chemin.read_text(encoding="utf-8")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  ASSEMBLAGE FINAL EN MARKDOWN
    # ─────────────────────────────────────────────────────────────────────────

    def _assembler_markdown(self, sections: dict, themes_ordonnes: list,
                             carte: dict, citations: GestionnaireCitations,
                             sujet: str) -> str:
        """
        ✅ CORRIGÉ : titre basé sur le sujet global, pas le premier thème.
        """
        meta  = carte.get("meta", {})
        ev    = carte.get("evolution_globale", {})
        annees = ev.get("annees_couvertes", [])
        periode = (f"{min(annees)}–{max(annees)}"
                   if len(annees) >= 2 else "")

        sous_titre = f"_{periode} · {meta.get('nb_articles_total','?')} articles · "                    f"{meta.get('nb_themes','?')} thèmes identifiés_"

        lignes = [
            f"# Revue de Littérature : {sujet}",
            sous_titre,
            "",
            "---",
            "",
            "## 1. Introduction",
            "",
            sections.get("introduction", "_Introduction non disponible._"),
            "",
        ]

        for i, (tid, theme) in enumerate(themes_ordonnes):
            nom_theme   = theme.get("nom_theme", f"Thème {tid}")
            cle_section = f"theme_{tid}"
            texte       = sections.get(cle_section, f"_Section '{nom_theme}' non disponible._")
            lignes += [
                f"## {i + 2}. {nom_theme}",
                "",
                texte,
                "",
            ]

        nb_section_conclusion = len(themes_ordonnes) + 2
        lignes += [
            f"## {nb_section_conclusion}. Conclusion",
            "",
            sections.get("conclusion", "_Conclusion non disponible._"),
            "",
            "---",
            "",
            citations.generer_bibliographie(),
        ]

        return "\n".join(lignes)

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, chemin_carte: str, chemin_corpus: str,
            dossier_output: str, indexeur=None,
            sujet: str = "Revue de Littérature",
            forcer_regeneration: bool = False) -> dict:
        """
        Pipeline complet de génération de la revue.

        chemin_carte         : carte_corpus.json (Agent Cartographe)
        chemin_corpus        : corpus_complet.json (Agent Curateur)
        dossier_output       : dossier de sortie
        indexeur             : AgentIndexeur en mémoire (optionnel, active le RAG)
        sujet                : titre du domaine de recherche
        forcer_regeneration  : True = ignore le cache des sections
        """
        log.info(f"{'='*60}")
        log.info(f"  AGENT NARRATEUR v2")
        log.info(f"{'='*60}")
        debut = time.time()

        dossier = Path(dossier_output)
        dossier.mkdir(parents=True, exist_ok=True)

        # ── Chargement ────────────────────────────────────────────────────
        log.info("Chargement de la carte et du corpus...")
        carte  = self._charger_carte(chemin_carte)
        corpus = self._charger_corpus(chemin_corpus)

        citations       = GestionnaireCitations(corpus)
        themes_ordonnes = self._ordonner_themes(carte)

        # ✅ CORRIGÉ : validation avant de commencer
        if not themes_ordonnes:
            raise ValueError(
                "La carte ne contient aucun thème. "
                "Vérifiez que l'Agent Cartographe a bien tourné."
            )

        log.info(f"{len(themes_ordonnes)} thème(s) à narrer | "
                 f"{len(corpus)} articles dans le corpus")

        # ── Si pas de sujet fourni, dériver de la carte ───────────────────
        if sujet == "Revue de Littérature":
            noms_themes = [t.get("nom_theme", "") for _, t in themes_ordonnes]
            if noms_themes:
                # Prendre les 2-3 premiers mots du premier thème comme sujet
                sujet = " & ".join(noms_themes[:2]) if len(noms_themes) >= 2 else noms_themes[0]

        log.info(f"Sujet de la revue : {sujet}")

        sections        = {}
        themes_deja_vus = []

        # ── Introduction ──────────────────────────────────────────────────
        cache_intro = self._charger_section_cache("introduction", dossier) if not forcer_regeneration else None
        if cache_intro:
            sections["introduction"] = cache_intro
        else:
            sections["introduction"] = self._generer_introduction(
                carte, themes_ordonnes, citations
            )
            self._sauvegarder_section("introduction", sections["introduction"], dossier)
            time.sleep(SLEEP_ENTRE_APPELS)  # ✅ Pause rate limit

        # ── Sections thématiques ──────────────────────────────────────────
        for tid, theme in themes_ordonnes:
            nom_theme   = theme.get("nom_theme", f"Thème {tid}")
            cle_section = f"theme_{tid}"

            cache_theme = (
                self._charger_section_cache(cle_section, dossier)
                if not forcer_regeneration else None
            )

            if cache_theme:
                sections[cle_section] = cache_theme
            else:
                sections[cle_section] = self._generer_section_theme(
                    tid, theme, carte, indexeur, citations, themes_deja_vus
                )
                self._sauvegarder_section(cle_section, sections[cle_section], dossier)
                time.sleep(SLEEP_ENTRE_APPELS)  # ✅ Pause rate limit

            themes_deja_vus.append(nom_theme)

        # ── Conclusion ────────────────────────────────────────────────────
        cache_conclu = (
            self._charger_section_cache("conclusion", dossier)
            if not forcer_regeneration else None
        )
        if cache_conclu:
            sections["conclusion"] = cache_conclu
        else:
            sections["conclusion"] = self._generer_conclusion(
                carte, sections, themes_ordonnes, citations
            )
            self._sauvegarder_section("conclusion", sections["conclusion"], dossier)

        # ── Assemblage Markdown ───────────────────────────────────────────
        log.info("Assemblage du document Markdown...")
        texte_revue = self._assembler_markdown(
            sections, themes_ordonnes, carte, citations, sujet
        )

        chemin_md = dossier / "revue_litterature.md"
        chemin_md.write_text(texte_revue, encoding="utf-8")
        log.info(f"Revue Markdown → {chemin_md}")

        # ── Sauvegarde JSON machine-readable ─────────────────────────────
        nb_mots = len(texte_revue.split())
        revue_json = {
            "sujet":          sujet,
            "nb_themes":      len(themes_ordonnes),
            "nb_articles":    len(corpus),
            "nb_mots_total":  nb_mots,
            "nb_citations":   len(citations.citations_utilisees),
            "duree_secondes": round(time.time() - debut, 1),
            "sections":       {k: v[:500] + "..." if len(v) > 500 else v
                               for k, v in sections.items()},
            "sections_completes": sections,
            "citations":      citations.get_registre(),
            "themes": [
                {
                    "theme_id":  tid,
                    "nom":       theme.get("nom_theme", ""),
                    "nb_articles": theme.get("nb_articles", 0),
                    "tendance":  theme.get("evolution_temporelle", {}).get("tendance", "?")
                }
                for tid, theme in themes_ordonnes
            ]
        }

        chemin_json = dossier / "revue_litterature.json"
        with open(chemin_json, "w", encoding="utf-8") as f:
            json.dump(revue_json, f, ensure_ascii=False, indent=2)
        log.info(f"Revue JSON → {chemin_json}")

        # ── Résumé console ────────────────────────────────────────────────
        log.info(f"{'='*60}")
        log.info(f"  RÉSUMÉ AGENT NARRATEUR v2")
        log.info(f"{'='*60}")
        log.info(f"  Sujet          : {sujet}")
        log.info(f"  Sections       : {len(sections)}")
        log.info(f"  Mots total     : {nb_mots}")
        log.info(f"  Citations      : {len(citations.citations_utilisees)}")
        log.info(f"  Durée totale   : {revue_json['duree_secondes']}s")
        log.info(f"  Revue MD   → {chemin_md}")
        log.info(f"  Revue JSON → {chemin_json}")

        return revue_json


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agent = AgentNarrateur()
    revue = agent.run(
        chemin_carte   = "data/corpus/carte_corpus.json",
        chemin_corpus  = "data/corpus/corpus_complet.json",
        dossier_output = "data/revue",
        indexeur       = None,   # Remplacer par l'instance AgentIndexeur si disponible
        sujet          = "Intelligence Artificielle en Éducation",
        forcer_regeneration = False  # True pour régénérer même si cache présent
    )

    print(f"\n  Revue générée : {revue['nb_mots_total']} mots | "
          f"{revue['nb_citations']} citations | "
          f"{revue['duree_secondes']}s")
