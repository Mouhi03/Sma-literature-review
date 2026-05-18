"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AGENT CARTOGRAPHE THÉMATIQUE — SMA Revue de Littérature            ║
║                                                                              ║
║  Reçoit   : AgentIndexeur (abstracts + embeddings déjà calculés)            ║
║             corpus_complet.json (métadonnées complètes)                      ║
║                                                                              ║
║  Produit  : carte_corpus.json                                                ║
║    → 4-6 thèmes nommés et décrits par le LLM                                ║
║    → Approches méthodologiques par thème                                     ║
║    → Évolutions temporelles (articles par année par thème)                   ║
║    → Carte auteurs principaux par thème                                      ║
║                                                                              ║
║  Ce que les agents suivants attendent :                                      ║
║  → Narrateur  : thèmes ordonnés + articles par thème + évolution             ║
║  → Détecteur  : thèmes + méthodes dominantes (pour trouver les manques)      ║
║  → Citateur   : doc_id → thème (pour organiser les citations)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation :
    pip install scikit-learn groq python-dotenv numpy

Optionnel (meilleur clustering sur petits corpus) :
    pip install hdbscan
"""

import os
import json
import time
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from groq import Groq
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

try:
    import hdbscan
    HDBSCAN_OK = True
except ImportError:
    HDBSCAN_OK = False

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
#  1. SÉLECTEUR DU NOMBRE OPTIMAL DE CLUSTERS
# ═════════════════════════════════════════════════════════════════════════════

class SelecteurK:
    """
    Trouve automatiquement le meilleur nombre de clusters k
    en utilisant le score de silhouette.

    Le projet demande 4 à 6 thèmes. On teste k ∈ [3, 7] et on
    garde celui avec le meilleur silhouette score.
    """

    @staticmethod
    def trouver_optimal(embeddings: np.ndarray,
                        k_min: int = 3,
                        k_max: int = 7) -> tuple:
        """
        Retourne (k_optimal, scores_silhouette).
        Si le corpus est trop petit pour certains k, ils sont ignorés.
        """
        n = len(embeddings)
        scores = {}

        for k in range(k_min, min(k_max + 1, n)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)

            # Silhouette nécessite au moins 2 clusters non vides
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels, metric="cosine")
            scores[k] = round(score, 4)

        if not scores:
            return k_min, scores

        k_optimal = max(scores, key=scores.get)
        return k_optimal, scores


# ═════════════════════════════════════════════════════════════════════════════
#  2. MOTEUR DE CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

class MoteurClustering:
    """
    Deux algorithmes disponibles :

    KMeans   : rapide, k fixé, bon pour corpus homogènes
    HDBSCAN  : détecte le nombre de clusters automatiquement,
               gère les outliers, meilleur sur petits corpus irréguliers

    On utilise KMeans par défaut (plus stable, plus prévisible).
    HDBSCAN est activé si installé et si le corpus < 30 articles.
    """

    def __init__(self, algo: str = "auto"):
        """
        algo : "kmeans" | "hdbscan" | "auto"
               "auto" → HDBSCAN si disponible et n < 30, sinon KMeans
        """
        self.algo = algo

    def clusteriser(self, embeddings: np.ndarray,
                    k: int = None) -> tuple:
        """
        Retourne (labels, k_utilise, algo_utilise).
        labels : array d'entiers, un par article (indice de cluster).
        """
        n = len(embeddings)

        # Choix de l'algo
        utiliser_hdbscan = (
            HDBSCAN_OK and
            (self.algo == "hdbscan" or (self.algo == "auto" and n < 30))
        )

        if utiliser_hdbscan:
            return self._hdbscan(embeddings)
        else:
            k_final = k or 5
            return self._kmeans(embeddings, k_final)

    def _kmeans(self, embeddings: np.ndarray, k: int) -> tuple:
        km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
        labels = km.fit_predict(embeddings)
        return labels, k, "kmeans"

    def _hdbscan(self, embeddings: np.ndarray) -> tuple:
        """
        min_cluster_size=2 pour accepter des petits clusters.
        Les outliers (label=-1) sont assignés au cluster le plus proche.
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(embeddings)

        # Réassigner les outliers (-1) au cluster le plus proche
        if -1 in labels:
            labels = self._reassigner_outliers(embeddings, labels)

        k = len(set(labels))
        return labels, k, "hdbscan"

    @staticmethod
    def _reassigner_outliers(embeddings: np.ndarray,
                              labels: np.ndarray) -> np.ndarray:
        """Assigne chaque outlier au centroïde de cluster le plus proche."""
        labels = labels.copy()
        clusters_valides = [l for l in set(labels) if l != -1]

        if not clusters_valides:
            labels[:] = 0
            return labels

        centroides = {
            c: embeddings[labels == c].mean(axis=0)
            for c in clusters_valides
        }

        for i, label in enumerate(labels):
            if label == -1:
                distances = {
                    c: np.linalg.norm(embeddings[i] - centroide)
                    for c, centroide in centroides.items()
                }
                labels[i] = min(distances, key=distances.get)

        return labels


# ═════════════════════════════════════════════════════════════════════════════
#  3. AGENT CARTOGRAPHE THÉMATIQUE
# ═════════════════════════════════════════════════════════════════════════════

class AgentCartographe:
    """
    Pipeline complet :
      1. Charger les abstracts + embeddings depuis l'Indexeur
      2. Sélectionner le k optimal (silhouette)
      3. Clusteriser (KMeans ou HDBSCAN)
      4. Pour chaque cluster → prompt LLM pour nommer le thème,
         décrire les méthodes, identifier les populations
      5. Analyser les évolutions temporelles
      6. Analyser les approches méthodologiques globales
      7. Identifier les auteurs principaux par thème
      8. Produire carte_corpus.json
    """

    def __init__(self):
        self.client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model   = "llama-3.3-70b-versatile"
        self.selecteur  = SelecteurK()
        self.moteur_cl  = MoteurClustering(algo="auto")

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 1 : Chargement des abstracts depuis l'Indexeur
    # ─────────────────────────────────────────────────────────────────────────

    def _charger_depuis_indexeur(self, indexeur) -> tuple:
        """
        Récupère les abstracts et leurs embeddings directement
        depuis l'AgentIndexeur (déjà calculés, pas de recalcul).

        Retourne (abstracts, embeddings_matrix).
        """
        print("[Cartographe] Chargement des abstracts depuis l'Indexeur...")
        abstracts = indexeur.get_tous_abstracts()

        if not abstracts:
            raise ValueError("Aucun abstract trouvé dans ChromaDB. "
                             "Vérifiez que l'Agent Indexeur a bien tourné.")

        # Matrice numpy : shape (n_articles, dim_embedding)
        embeddings = np.array([a["embedding"] for a in abstracts])
        print(f"[Cartographe] {len(abstracts)} abstracts | "
              f"dim embedding : {embeddings.shape[1]}")

        return abstracts, embeddings

    def _charger_depuis_corpus(self, chemin_corpus: str) -> tuple:
        """
        Alternative : charge depuis corpus_complet.json si l'Indexeur
        n'est pas disponible en mémoire. Recalcule les embeddings.
        """
        from sentence_transformers import SentenceTransformer
        print("[Cartographe] Chargement depuis corpus_complet.json...")

        with open(chemin_corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        # Ne garder que les articles avec abstract
        corpus = [d for d in corpus if d.get("abstract") and len(str(d["abstract"])) > 50]
        print(f"[Cartographe] {len(corpus)} articles avec abstract")

        # Recalcul des embeddings
        modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        textes = [str(d.get("resume_court") or d.get("abstract", "")) for d in corpus]
        embeddings = modele.encode(textes, normalize_embeddings=True, show_progress_bar=True)

        # Formater comme des "abstracts" pour compatibilité avec le reste
        abstracts = []
        for i, doc in enumerate(corpus):
            abstracts.append({
                "doc_id":               doc.get("doc_id", ""),
                "texte":                textes[i],
                "titre":                doc.get("titre", ""),
                "auteurs":              json.dumps(doc.get("auteurs", []), ensure_ascii=False),
                "annee":                int(doc.get("annee") or 0),
                "domaine":              doc.get("domaine", ""),
                "mots_cles":            json.dumps(doc.get("mots_cles", []), ensure_ascii=False),
                "methodes":             json.dumps(doc.get("methodes", []), ensure_ascii=False),
                "populations_etudiees": json.dumps(doc.get("populations_etudiees", []), ensure_ascii=False),
                "contexte_geographique":json.dumps(doc.get("contexte_geographique", []), ensure_ascii=False),
                "claim_type":           doc.get("claim_type", ""),
                "evidence_level":       doc.get("evidence_level", ""),
                "langue":               doc.get("langue", ""),
                "fichier_source":       doc.get("fichier_source", ""),
                "contribution_principale": doc.get("contribution_principale", ""),
                "embedding":            embeddings[i].tolist()
            })

        return abstracts, embeddings

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 2 : Réduction dimensionnelle pour visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def _reduire_dimensions(self, embeddings: np.ndarray, n: int = 2) -> np.ndarray:
        """
        PCA → 2D pour visualisation (coordonnées stockées dans la carte).
        Permet de générer un scatter plot des thèmes si besoin.
        """
        if embeddings.shape[0] < 3:
            return embeddings[:, :n]
        pca = PCA(n_components=min(n, embeddings.shape[1], embeddings.shape[0]))
        return pca.fit_transform(embeddings)

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 3 : Nommage et description d'un cluster par LLM
    # ─────────────────────────────────────────────────────────────────────────

    def _analyser_cluster_llm(self, abstracts_cluster: list,
                               index_cluster: int) -> dict:
        """
        Envoie jusqu'à 8 abstracts du cluster au LLM.
        Demande : nom du thème, description, méthodes dominantes,
                  populations, mots-clés représentatifs.

        Retourne un dict JSON structuré.
        """
        # On prend les 8 premiers abstracts (évite dépassement contexte)
        echantillon = abstracts_cluster[:8]

        # Construction du prompt
        textes_abstracts = ""
        for i, a in enumerate(echantillon):
            titre    = a.get("titre", "Inconnu")[:80]
            annee    = a.get("annee", "?")
            texte    = a.get("texte", "")[:400]
            contrib  = a.get("contribution_principale", "")
            textes_abstracts += (
                f"\n--- Article {i+1} ({annee}) ---\n"
                f"Titre : {titre}\n"
                f"Abstract : {texte}\n"
                f"Contribution : {contrib}\n"
            )

        prompt = f"""Tu es un expert en analyse de littérature scientifique.

Voici {len(echantillon)} articles qui ont été regroupés ensemble par similarité sémantique :

{textes_abstracts}

Analyse ces articles et réponds UNIQUEMENT avec un JSON valide, sans texte avant ni après :

{{
  "nom_theme": "Nom court et précis du thème commun (5-8 mots max)",
  "description_theme": "Description du thème en 3-4 phrases : de quoi parle-t-on, quelle problématique est adressée, pourquoi c'est important.",
  "sous_themes": ["sous-thème 1", "sous-thème 2", "sous-thème 3"],
  "methodes_dominantes": ["méthode ou technique la plus utilisée dans ce groupe"],
  "approche_dominante": "empirical ou theoretical ou methodological ou review ou mixed",
  "populations_etudiees": ["type de population ou objet principal étudié"],
  "mots_cles_representatifs": ["5 à 8 mots-clés qui résument le mieux ce thème"],
  "niveau_maturite": "emergent ou en_developpement ou mature",
  "resume_narratif": "En une phrase : ce que ce groupe d'articles apporte collectivement au domaine."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            raw = response.choices[0].message.content.strip()

            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)

            return json.loads(raw)

        except Exception as e:
            print(f"  ⚠  LLM échoué pour cluster {index_cluster} : {e}")
            return {
                "nom_theme":               f"Thème {index_cluster + 1}",
                "description_theme":       "Description non disponible.",
                "sous_themes":             [],
                "methodes_dominantes":     [],
                "approche_dominante":      "unknown",
                "populations_etudiees":    [],
                "mots_cles_representatifs":[],
                "niveau_maturite":         "unknown",
                "resume_narratif":         ""
            }

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 4 : Analyse des évolutions temporelles
    # ─────────────────────────────────────────────────────────────────────────

    def _analyser_evolutions_temporelles(self, abstracts: list,
                                          labels: np.ndarray,
                                          themes: dict) -> dict:
        """
        Pour chaque thème, calcule :
        - Nombre d'articles par année
        - Tendance (croissante / décroissante / stable / émergente)
        - Année de première apparition
        - Année de pic d'activité

        Utilisé par l'Agent Narrateur pour raconter l'évolution du domaine.
        """
        # Distribution articles par thème et par année
        par_theme_annee = defaultdict(lambda: defaultdict(int))

        for i, a in enumerate(abstracts):
            theme_id = int(labels[i])
            annee    = int(a.get("annee") or 0)
            if annee > 1990:  # filtre les années invalides
                par_theme_annee[theme_id][annee] += 1

        evolution_globale = defaultdict(int)
        for i, a in enumerate(abstracts):
            annee = int(a.get("annee") or 0)
            if annee > 1990:
                evolution_globale[annee] += 1

        evolution = {}

        for theme_id, theme_info in themes.items():
            dist_annee = dict(sorted(par_theme_annee[theme_id].items()))

            if not dist_annee:
                evolution[theme_id] = {
                    "distribution_annuelle": {},
                    "annee_premiere_publication": None,
                    "annee_pic": None,
                    "tendance": "inconnue",
                    "nb_articles_5_dernieres_annees": 0
                }
                continue

            annees      = sorted(dist_annee.keys())
            annee_debut = annees[0]
            annee_pic   = max(dist_annee, key=dist_annee.get)

            # Tendance : compare première moitié vs deuxième moitié de la période
            milieu = len(annees) // 2
            nb_debut = sum(dist_annee[a] for a in annees[:milieu]) if milieu > 0 else 0
            nb_fin   = sum(dist_annee[a] for a in annees[milieu:])

            if len(annees) < 3:
                tendance = "insuffisant"
            elif annee_debut >= 2020:
                tendance = "emergente"
            elif nb_fin > nb_debut * 1.5:
                tendance = "croissante"
            elif nb_fin < nb_debut * 0.7:
                tendance = "decroissante"
            else:
                tendance = "stable"

            # Articles des 5 dernières années
            annee_max = max(annees)
            nb_recents = sum(
                v for a, v in dist_annee.items()
                if a >= annee_max - 4
            )

            evolution[theme_id] = {
                "distribution_annuelle":        dist_annee,
                "annee_premiere_publication":   annee_debut,
                "annee_pic":                    annee_pic,
                "tendance":                     tendance,
                "nb_articles_5_dernieres_annees": nb_recents
            }

        # Évolution globale du corpus (tous thèmes confondus)
        evolution["_global"] = {
            "distribution_annuelle": dict(sorted(evolution_globale.items())),
            "annees_couvertes":      sorted(evolution_globale.keys()),
            "pic_global":            max(evolution_globale, key=evolution_globale.get)
                                     if evolution_globale else None
        }

        return evolution

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 5 : Cartographie des approches méthodologiques
    # ─────────────────────────────────────────────────────────────────────────

    def _cartographier_methodes(self, abstracts: list,
                                 labels: np.ndarray) -> dict:
        """
        Pour chaque thème, liste les méthodes utilisées et leur fréquence.
        Calcule aussi la carte globale des méthodes du corpus.

        Utilisé par l'Agent Détecteur de Gaps pour trouver les méthodes
        peu ou pas utilisées dans certains thèmes.
        """
        methodes_par_theme  = defaultdict(list)
        methodes_globales   = []
        claim_types_theme   = defaultdict(list)
        evidence_par_theme  = defaultdict(list)

        for i, a in enumerate(abstracts):
            theme_id = int(labels[i])

            # Méthodes (stockées en JSON string dans ChromaDB)
            methodes_raw = a.get("methodes", "[]")
            try:
                methodes = json.loads(methodes_raw) if isinstance(methodes_raw, str) else methodes_raw
            except Exception:
                methodes = []

            for m in (methodes or []):
                if m and len(str(m)) > 2:
                    methodes_par_theme[theme_id].append(str(m).strip().lower())
                    methodes_globales.append(str(m).strip().lower())

            # Claim types et evidence levels
            ct = a.get("claim_type", "")
            el = a.get("evidence_level", "")
            if ct:
                claim_types_theme[theme_id].append(ct)
            if el:
                evidence_par_theme[theme_id].append(el)

        # Comptage par thème
        carte_methodes = {}
        for theme_id, methodes in methodes_par_theme.items():
            compteur = Counter(methodes)
            carte_methodes[theme_id] = {
                "methodes_frequences":    dict(compteur.most_common(10)),
                "methode_dominante":      compteur.most_common(1)[0][0] if compteur else None,
                "nb_methodes_distinctes": len(set(methodes)),
                "claim_types":            dict(Counter(claim_types_theme[theme_id])),
                "evidence_levels":        dict(Counter(evidence_par_theme[theme_id]))
            }

        # Carte globale
        compteur_global = Counter(methodes_globales)
        carte_methodes["_global"] = {
            "methodes_frequences":    dict(compteur_global.most_common(15)),
            "methode_dominante":      compteur_global.most_common(1)[0][0] if compteur_global else None,
            "nb_methodes_distinctes": len(set(methodes_globales)),
            "claim_types":            dict(Counter([
                a.get("claim_type", "") for a in abstracts if a.get("claim_type")
            ])),
            "evidence_levels":        dict(Counter([
                a.get("evidence_level", "") for a in abstracts if a.get("evidence_level")
            ]))
        }

        return carte_methodes

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 6 : Auteurs principaux par thème
    # ─────────────────────────────────────────────────────────────────────────

    def _identifier_auteurs_principaux(self, abstracts: list,
                                        labels: np.ndarray) -> dict:
        """
        Pour chaque thème, identifie les auteurs les plus productifs.
        Utile pour le Narrateur (citer les chercheurs clés de chaque thème).
        """
        auteurs_par_theme = defaultdict(list)

        for i, a in enumerate(abstracts):
            theme_id  = int(labels[i])
            auteurs_raw = a.get("auteurs", "[]")
            try:
                auteurs = json.loads(auteurs_raw) if isinstance(auteurs_raw, str) else auteurs_raw
            except Exception:
                auteurs = []

            for auteur in (auteurs or []):
                if auteur and len(str(auteur)) > 2:
                    auteurs_par_theme[theme_id].append(str(auteur).strip())

        result = {}
        for theme_id, auteurs in auteurs_par_theme.items():
            compteur = Counter(auteurs)
            result[theme_id] = {
                "auteurs_frequences":   dict(compteur.most_common(8)),
                "auteurs_principaux":   [a for a, _ in compteur.most_common(5)],
                "nb_auteurs_distincts": len(set(auteurs))
            }

        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 7 : Analyse LLM des évolutions temporelles (narrative)
    # ─────────────────────────────────────────────────────────────────────────

    def _analyser_evolution_narrative(self, themes: dict,
                                       evolutions: dict) -> str:
        """
        Demande au LLM de produire une analyse narrative des évolutions
        temporelles de l'ensemble du corpus.

        Utilisé directement par l'Agent Narrateur.
        """
        # Résumé compact des thèmes et leur tendance
        resume_themes = ""
        for theme_id, theme_info in themes.items():
            nom      = theme_info.get("nom_theme", f"Thème {theme_id}")
            tendance = evolutions.get(theme_id, {}).get("tendance", "inconnue")
            annee_d  = evolutions.get(theme_id, {}).get("annee_premiere_publication", "?")
            nb_recents = evolutions.get(theme_id, {}).get("nb_articles_5_dernieres_annees", 0)
            resume_themes += (
                f"- {nom} : tendance={tendance}, "
                f"depuis {annee_d}, {nb_recents} articles récents\n"
            )

        global_ev  = evolutions.get("_global", {})
        annees     = global_ev.get("annees_couvertes", [])
        periode    = f"{min(annees)}-{max(annees)}" if annees else "inconnue"

        prompt = f"""Tu es un expert en analyse de littérature scientifique.

Voici la distribution temporelle d'un corpus de recherche couvrant {periode} :

{resume_themes}

Rédige en 4-6 phrases une analyse narrative de l'évolution de ce domaine de recherche :
- Comment le domaine a-t-il évolué dans le temps ?
- Quels thèmes ont émergé récemment ?
- Y a-t-il des thèmes en déclin ?
- Quelle est la dynamique globale actuelle ?

Réponds directement en texte, sans JSON, sans titres, sans puces."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠  Analyse narrative échouée : {e}")
            return "Analyse narrative non disponible."

    # ─────────────────────────────────────────────────────────────────────────
    #  ASSEMBLAGE DE LA CARTE DU CORPUS
    # ─────────────────────────────────────────────────────────────────────────

    def _assembler_carte(self, abstracts: list, labels: np.ndarray,
                          k: int, algo: str, scores_silhouette: dict,
                          coords_2d: np.ndarray) -> dict:
        """
        Assemble la carte complète du corpus :
        - Thèmes nommés par LLM
        - Articles par thème
        - Évolutions temporelles
        - Méthodes par thème
        - Auteurs principaux
        - Analyse narrative
        """
        print(f"\n[Cartographe] Assemblage de la carte ({k} thèmes)...")

        # ── Regroupement des articles par cluster ─────────────────────────
        clusters = defaultdict(list)
        for i, a in enumerate(abstracts):
            clusters[int(labels[i])].append(a)

        # ── Nommage de chaque thème par LLM ──────────────────────────────
        themes = {}
        for cluster_id in sorted(clusters.keys()):
            articles_cluster = clusters[cluster_id]
            print(f"  → LLM analyse le cluster {cluster_id + 1}/{k} "
                  f"({len(articles_cluster)} articles)...")

            description_llm = self._analyser_cluster_llm(articles_cluster, cluster_id)

            # Articles du cluster avec leurs métadonnées
            articles_info = []
            for a in articles_cluster:
                idx = list(abstracts).index(a)
                articles_info.append({
                    "doc_id":         a.get("doc_id", ""),
                    "titre":          a.get("titre", ""),
                    "annee":          int(a.get("annee") or 0),
                    "fichier_source": a.get("fichier_source", ""),
                    "langue":         a.get("langue", ""),
                    "claim_type":     a.get("claim_type", ""),
                    "evidence_level": a.get("evidence_level", ""),
                    "coord_x":        float(coords_2d[idx][0]) if coords_2d is not None else 0.0,
                    "coord_y":        float(coords_2d[idx][1]) if coords_2d is not None else 0.0
                })

            themes[cluster_id] = {
                **description_llm,
                "cluster_id":     cluster_id,
                "nb_articles":    len(articles_cluster),
                "articles":       articles_info
            }

            # Pause pour respecter la rate limit de Groq
            time.sleep(0.5)

        # ── Analyses transversales ────────────────────────────────────────
        print("  → Analyse des évolutions temporelles...")
        evolutions = self._analyser_evolutions_temporelles(abstracts, labels, themes)

        print("  → Cartographie des méthodes...")
        carte_methodes = self._cartographier_methodes(abstracts, labels)

        print("  → Identification des auteurs principaux...")
        auteurs_principaux = self._identifier_auteurs_principaux(abstracts, labels)

        print("  → Analyse narrative globale...")
        analyse_narrative = self._analyser_evolution_narrative(themes, evolutions)

        # ── Enrichissement des thèmes avec les analyses ───────────────────
        for cluster_id in themes:
            themes[cluster_id]["evolution_temporelle"]  = evolutions.get(cluster_id, {})
            themes[cluster_id]["carte_methodes"]        = carte_methodes.get(cluster_id, {})
            themes[cluster_id]["auteurs_principaux"]    = auteurs_principaux.get(cluster_id, {})

        # ── Statistiques du clustering ────────────────────────────────────
        stats_clustering = {
            "algorithme":         algo,
            "k_optimal":          k,
            "scores_silhouette":  scores_silhouette,
            "meilleur_silhouette":max(scores_silhouette.values()) if scores_silhouette else None,
            "nb_articles_total":  len(abstracts),
            "distribution_themes":{
                cluster_id: len(articles_cluster)
                for cluster_id, articles_cluster in clusters.items()
            }
        }

        return {
            "meta": {
                "nb_themes":          k,
                "nb_articles_total":  len(abstracts),
                "stats_clustering":   stats_clustering,
                "analyse_narrative_globale": analyse_narrative
            },
            "themes":                 themes,
            "evolution_globale":      evolutions.get("_global", {}),
            "carte_methodes_globale": carte_methodes.get("_global", {})
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, dossier_output: str,
            indexeur=None,
            chemin_corpus: str = None,
            k_min: int = 3, k_max: int = 7) -> dict:
        """
        Point d'entrée.

        Usage avec l'Indexeur en mémoire (recommandé) :
            cartographe.run("data/corpus", indexeur=indexeur)

        Usage depuis le fichier JSON (si Indexeur non disponible) :
            cartographe.run("data/corpus",
                            chemin_corpus="data/corpus/corpus_complet.json")
        """
        print(f"\n{'='*60}")
        print(f"  AGENT CARTOGRAPHE THÉMATIQUE")
        print(f"{'='*60}")
        debut = time.time()

        Path(dossier_output).mkdir(parents=True, exist_ok=True)

        # ── Chargement ────────────────────────────────────────────────────
        if indexeur is not None:
            abstracts, embeddings = self._charger_depuis_indexeur(indexeur)
        elif chemin_corpus:
            abstracts, embeddings = self._charger_depuis_corpus(chemin_corpus)
        else:
            raise ValueError("Fournir 'indexeur' ou 'chemin_corpus'.")

        if len(abstracts) < 3:
            raise ValueError(f"Corpus trop petit ({len(abstracts)} articles). "
                             "Il faut au minimum 3 articles.")

        # ── Sélection du k optimal ────────────────────────────────────────
        print(f"\n[Cartographe] Recherche du nombre optimal de thèmes "
              f"(k ∈ [{k_min}, {k_max}])...")
        k_optimal, scores_silhouette = self.selecteur.trouver_optimal(
            embeddings, k_min, k_max
        )
        print(f"[Cartographe] k optimal = {k_optimal} "
              f"(silhouette = {scores_silhouette.get(k_optimal, '?')})")
        print(f"[Cartographe] Scores par k : {scores_silhouette}")

        # ── Clustering ────────────────────────────────────────────────────
        print(f"\n[Cartographe] Clustering en cours...")
        labels, k_utilise, algo = self.moteur_cl.clusteriser(embeddings, k_optimal)
        print(f"[Cartographe] Algorithme : {algo} | k utilisé : {k_utilise}")

        # Distribution par cluster
        dist = Counter(labels.tolist())
        for cid, nb in sorted(dist.items()):
            print(f"  Cluster {cid} : {nb} articles")

        # ── Réduction 2D pour visualisation ──────────────────────────────
        coords_2d = self._reduire_dimensions(embeddings, n=2)

        # ── Assemblage de la carte ────────────────────────────────────────
        carte = self._assembler_carte(
            abstracts, labels, k_utilise, algo,
            scores_silhouette, coords_2d
        )

        # ── Sauvegarde ────────────────────────────────────────────────────
        carte_path = Path(dossier_output) / "carte_corpus.json"
        with open(carte_path, "w", encoding="utf-8") as f:
            json.dump(carte, f, ensure_ascii=False, indent=2)

        # ── Rapport lisible ───────────────────────────────────────────────
        rapport = self._generer_rapport_lisible(carte, dossier_output)

        duree = round(time.time() - debut, 1)

        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ AGENT CARTOGRAPHE")
        print(f"{'='*60}")
        print(f"  Thèmes identifiés  : {k_utilise}")
        print(f"  Articles couverts  : {len(abstracts)}")
        print(f"  Algorithme         : {algo}")
        print(f"  Durée              : {duree}s")
        print(f"  Carte sauvegardée  → {carte_path}")
        print(f"\n  Thèmes identifiés :")
        for cid, theme in carte["themes"].items():
            print(f"    [{cid}] {theme['nom_theme']} "
                  f"({theme['nb_articles']} articles | "
                  f"tendance : {theme['evolution_temporelle'].get('tendance', '?')})")

        print(f"\n  Analyse narrative :")
        print(f"  {carte['meta']['analyse_narrative_globale'][:200]}...")

        return carte

    # ─────────────────────────────────────────────────────────────────────────
    #  RAPPORT LISIBLE (Markdown)
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_rapport_lisible(self, carte: dict,
                                  dossier_output: str) -> str:
        """Génère un fichier Markdown lisible de la carte du corpus."""
        lignes = []
        lignes.append("# Carte du Corpus — Agent Cartographe Thématique\n")
        lignes.append(f"**Nombre de thèmes** : {carte['meta']['nb_themes']}")
        lignes.append(f"**Articles analysés** : {carte['meta']['nb_articles_total']}\n")
        lignes.append("## Analyse narrative globale\n")
        lignes.append(carte["meta"]["analyse_narrative_globale"] + "\n")
        lignes.append("---\n")

        for cid, theme in carte["themes"].items():
            lignes.append(f"## Thème {int(cid)+1} : {theme['nom_theme']}")
            lignes.append(f"**Articles** : {theme['nb_articles']}  ")
            lignes.append(f"**Tendance** : {theme['evolution_temporelle'].get('tendance', '?')}  ")
            lignes.append(f"**Maturité** : {theme.get('niveau_maturite', '?')}\n")
            lignes.append(f"### Description")
            lignes.append(theme.get("description_theme", "") + "\n")
            lignes.append(f"### Méthodes dominantes")
            lignes.append(", ".join(theme.get("methodes_dominantes", [])) + "\n")
            lignes.append(f"### Mots-clés représentatifs")
            lignes.append(", ".join(theme.get("mots_cles_representatifs", [])) + "\n")
            lignes.append(f"### Résumé narratif")
            lignes.append(theme.get("resume_narratif", "") + "\n")
            lignes.append("---\n")

        rapport_md = "\n".join(lignes)
        rapport_path = Path(dossier_output) / "carte_corpus.md"
        with open(rapport_path, "w", encoding="utf-8") as f:
            f.write(rapport_md)

        return rapport_md


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Option 1 : pipeline complet depuis zéro ───────────────────────────
    from agent_indexeur import AgentIndexeur

    indexeur = AgentIndexeur(
        dossier_chroma = "data/chroma_db",
        taille_chunk   = 400,
        overlap        = 80
    )
    # Si déjà indexé, on ne relance pas run()
    # indexeur.run("data/corpus/corpus_complet.json", "data/corpus")

    cartographe = AgentCartographe()
    carte = cartographe.run(
        dossier_output = "data/corpus",
        indexeur       = indexeur,
        k_min          = 3,
        k_max          = 7
    )

    # ── Option 2 : sans Indexeur (corpus JSON seulement) ─────────────────
    # cartographe = AgentCartographe()
    # carte = cartographe.run(
    #     dossier_output = "data/corpus",
    #     chemin_corpus  = "data/corpus/corpus_complet.json"
    # )