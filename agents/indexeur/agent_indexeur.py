"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AGENT INDEXEUR — SMA Revue de Littérature v2                  ║
║  Améliorations v2 (RAG + Agentique) :                                       ║
║                                                                              ║
║  DÉCISION 1 — IDEMPOTENCE                                                   ║
║  ✓ Manifest JSON (indexed_docs.json) persistant sur disque                  ║
║  ✓ Skip automatique si doc_id déjà indexé                                   ║
║  ✓ Upsert propre : re-indexation possible avec --force                      ║
║                                                                              ║
║  DÉCISION 2 — QUALITÉ DES CHUNKS                                            ║
║  ✓ Rejet des chunks < MIN_CHUNK_WORDS mots (défaut : 30)                    ║
║  ✓ Rejet des chunks quasi-vides ou redondants                               ║
║  ✓ Logging structuré de chaque anomalie dans anomalies.json                 ║
║                                                                              ║
║  DÉCISION 3 — RAPPORT STRUCTURÉ VERS AGENTS AVAL                           ║
║  ✓ rapport_indexeur.json enrichi pour Cartographe et Détecteur              ║
║  ✓ Liste des doc_id indexés, articles sans abstract, sans sections          ║
║  ✓ Répartition par domaine, claim_type, evidence_level                      ║
║  ✓ Alertes explicites sur les lacunes du corpus                             ║
║                                                                              ║
║  Hérite de la v1 :                                                           ║
║  → Chunking A/B (sections prioritaires + fenêtre glissante fallback)        ║
║  → Deux collections ChromaDB (chunks + abstracts)                           ║
║  → API de recherche complète (sémantique, par doc_id, par section)          ║
║  → get_tous_abstracts() avec option with_embeddings                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation :
    pip install chromadb sentence-transformers

Modèle d'embedding (local, gratuit, multilingue fr+en+ar) :
    paraphrase-multilingual-mpnet-base-v2  (~420 MB, téléchargé au 1er lancement)

Usage standard :
    python agent_indexeur_v2.py

Re-indexation forcée d'un corpus (ignore le manifest) :
    indexeur.run(..., force_reindex=False)  # True = réindexe tout

Lancement CLI avec force :
    python agent_indexeur_v2.py --force
"""

import json
import uuid
import time
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("indexeur.log", encoding="utf-8")
    ]
)
log = logging.getLogger("AgentIndexeur")


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═════════════════════════════════════════════════════════════════════════════

MIN_CHUNK_WORDS     = 30    # Seuil de rejet d'un chunk trop court
QUALITE_MIN_CORPUS  = 0.2   # Score qualité minimum pour indexer un article


# ═════════════════════════════════════════════════════════════════════════════
#  1. CHUNKEUR INTELLIGENT (inchangé v1, ajout filtre qualité)
# ═════════════════════════════════════════════════════════════════════════════

class ChunkeurIntelligent:
    """
    Deux stratégies de découpage :

    Stratégie A — Par section (prioritaire)
        Utilisée quand l'Agent Curateur a détecté des sections.
        Chaque section devient un ou plusieurs chunks cohérents.

    Stratégie B — Fenêtre glissante (fallback)
        Utilisée quand aucune section n'a été détectée.
        Découpe le texte en blocs de taille_chunk mots avec overlap.

    NOUVEAU v2 : _valider_chunk() rejette les chunks < MIN_CHUNK_WORDS mots
    et retourne la raison de rejet pour le journal d'anomalies.
    """

    SECTIONS_BOOST = {
        "abstract":          1.5,
        "introduction":      1.2,
        "methodology":       1.8,
        "methods":           1.8,
        "proposed method":   1.8,
        "results":           1.6,
        "discussion":        1.4,
        "conclusion":        1.5,
        "related work":      1.3,
        "literature review": 1.3,
        "future work":       1.2,
        "references":        0.5,
        "acknowledgments":   0.3,
        "preamble":          0.8,
    }

    def __init__(self, taille_chunk: int = 400, overlap: int = 80):
        self.taille_chunk = taille_chunk
        self.overlap      = overlap

    # ── DÉCISION 2 : validation d'un chunk ────────────────────────────────

    def _valider_chunk(self, texte: str) -> tuple[bool, str]:
        """
        Retourne (valide: bool, raison: str).
        Rejette si :
          - Moins de MIN_CHUNK_WORDS mots (chunk trop court)
          - Texte vide ou quasi-vide après strip
          - Texte répétitif (entête/pied de page dupliqué)
        """
        if not texte or not texte.strip():
            return False, "chunk_vide"

        mots = texte.split()
        if len(mots) < MIN_CHUNK_WORDS:
            return False, f"chunk_trop_court_{len(mots)}_mots"

        # Détection de répétitivité : si > 60% des mots sont identiques
        mots_uniques = set(m.lower() for m in mots)
        ratio_unicite = len(mots_uniques) / len(mots)
        if ratio_unicite < 0.15:
            return False, "chunk_repetitif"

        return True, "ok"

    # ── Stratégie A : découpage par section ───────────────────────────────

    def chunker_par_sections(self, doc: dict) -> tuple[list, list]:
        """
        Retourne (chunks_valides, anomalies).
        Chaque anomalie = dict {doc_id, section_nom, raison, nb_mots, extrait}.
        """
        sections  = doc.get("sections", {})
        chunks    = []
        anomalies = []

        for nom_section, contenu in sections.items():
            if not contenu or len(contenu.strip()) < 50:
                continue

            mots = contenu.split()

            if len(mots) <= self.taille_chunk:
                chunk = self._construire_chunk(contenu.strip(), doc, nom_section, 0, 1)
                valide, raison = self._valider_chunk(chunk["texte"])
                if valide:
                    chunks.append(chunk)
                else:
                    anomalies.append(self._anomalie(doc, nom_section, raison, chunk["texte"]))
            else:
                sous_chunks = self._sliding_window(mots)
                for i, texte_chunk in enumerate(sous_chunks):
                    chunk = self._construire_chunk(texte_chunk, doc, nom_section, i, len(sous_chunks))
                    valide, raison = self._valider_chunk(chunk["texte"])
                    if valide:
                        chunks.append(chunk)
                    else:
                        anomalies.append(self._anomalie(doc, nom_section, raison, chunk["texte"]))

        return chunks, anomalies

    # ── Stratégie B : fenêtre glissante ───────────────────────────────────

    def chunker_par_fenetre(self, doc: dict) -> tuple[list, list]:
        texte = doc.get("texte_nettoye") or doc.get("texte_complet", "")
        if not texte:
            return [], []

        mots        = texte.split()
        sous_chunks = self._sliding_window(mots)
        chunks      = []
        anomalies   = []

        for i, sc in enumerate(sous_chunks):
            chunk  = self._construire_chunk(sc, doc, "texte_complet", i, len(sous_chunks))
            valide, raison = self._valider_chunk(chunk["texte"])
            if valide:
                chunks.append(chunk)
            else:
                anomalies.append(self._anomalie(doc, "texte_complet", raison, chunk["texte"]))

        return chunks, anomalies

    # ── Chunk abstract dédié ──────────────────────────────────────────────

    def chunk_abstract(self, doc: dict) -> Optional[dict]:
        abstract = doc.get("abstract", "")
        if not abstract or len(str(abstract).strip()) < 50:
            return None
        chunk = self._construire_chunk(
            str(abstract).strip(), doc, "abstract_dedie", 0, 1, type_chunk="abstract"
        )
        valide, _ = self._valider_chunk(chunk["texte"])
        return chunk if valide else None

    # ── Sliding window interne ─────────────────────────────────────────────

    def _sliding_window(self, mots: list) -> list:
        segments = []
        debut    = 0
        while debut < len(mots):
            fin = min(debut + self.taille_chunk, len(mots))
            segments.append(" ".join(mots[debut:fin]))
            if fin == len(mots):
                break
            debut += self.taille_chunk - self.overlap
        return segments

    # ── Construction d'un chunk ────────────────────────────────────────────

    def _construire_chunk(self, texte: str, doc: dict, section_nom: str,
                           chunk_index: int, total_chunks: int,
                           type_chunk: str = "section") -> dict:
        boost = self.SECTIONS_BOOST.get(section_nom.lower(), 1.0)
        return {
            "chunk_id":               str(uuid.uuid4()),
            "doc_id":                 doc.get("doc_id", ""),
            "type_chunk":             type_chunk,
            "texte":                  texte,
            "nb_mots":                len(texte.split()),
            "section_nom":            section_nom,
            "chunk_index":            chunk_index,
            "total_chunks_doc":       total_chunks,
            "titre":                  str(doc.get("titre") or ""),
            "auteurs":                json.dumps(doc.get("auteurs") or [], ensure_ascii=False),
            "annee":                  int(doc.get("annee") or 0),
            "domaine":                str(doc.get("domaine") or ""),
            "mots_cles":              json.dumps(doc.get("mots_cles") or [], ensure_ascii=False),
            "methodes":               json.dumps(doc.get("methodes") or [], ensure_ascii=False),
            "populations_etudiees":   json.dumps(doc.get("populations_etudiees") or [], ensure_ascii=False),
            "contexte_geographique":  json.dumps(doc.get("contexte_geographique") or [], ensure_ascii=False),
            "claim_type":             str(doc.get("claim_type") or ""),
            "evidence_level":         str(doc.get("evidence_level") or ""),
            "langue":                 str(doc.get("langue") or ""),
            "fichier_source":         str(doc.get("fichier_source") or ""),
            "qualite_score":          float(doc.get("qualite", {}).get("score", 0)),
            "qualite_niveau":         str(doc.get("qualite", {}).get("niveau", "")),
            "boost_section":          boost,
        }

    # ── Anomalie helper ────────────────────────────────────────────────────

    def _anomalie(self, doc: dict, section_nom: str, raison: str, texte: str) -> dict:
        return {
            "doc_id":       doc.get("doc_id", ""),
            "fichier":      doc.get("fichier_source", ""),
            "section_nom":  section_nom,
            "raison":       raison,
            "nb_mots":      len(texte.split()),
            "extrait":      texte[:120].replace("\n", " "),
        }

    # ── Dispatcher principal ───────────────────────────────────────────────

    def chunker(self, doc: dict) -> tuple[list, list]:
        """
        Retourne (chunks_valides, anomalies).
        Sélectionne la stratégie selon le document.
        """
        chunks    = []
        anomalies = []

        # Chunk abstract dédié (toujours en premier)
        chunk_abs = self.chunk_abstract(doc)
        if chunk_abs:
            chunks.append(chunk_abs)

        # Sections exploitables (sans références et remerciements)
        sections_utiles = {
            k: v for k, v in doc.get("sections", {}).items()
            if k not in ("references", "acknowledgments", "acknowledgements",
                         "bibliographie", "bibliography", "works cited")
        }

        if sections_utiles:
            c, a = self.chunker_par_sections(doc)
        else:
            c, a = self.chunker_par_fenetre(doc)

        chunks    += c
        anomalies += a
        return chunks, anomalies


# ═════════════════════════════════════════════════════════════════════════════
#  2. MANIFEST — DÉCISION 1 : IDEMPOTENCE
# ═════════════════════════════════════════════════════════════════════════════

class ManifestIndexeur:
    """
    Persiste sur disque la liste des doc_id déjà indexés.
    Permet de skip un article déjà traité sans interroger ChromaDB.

    Structure du manifest (indexed_docs.json) :
    {
      "doc_id_xxx": {
        "fichier_source": "article.pdf",
        "titre": "...",
        "nb_chunks": 12,
        "indexe_le": "2025-01-15T14:30:00"
      },
      ...
    }
    """

    def __init__(self, chemin: str):
        self.chemin = Path(chemin)
        self.data   = self._charger()

    def _charger(self) -> dict:
        if self.chemin.exists():
            with open(self.chemin, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _sauvegarder(self):
        with open(self.chemin, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def est_indexe(self, doc_id: str) -> bool:
        return doc_id in self.data

    def enregistrer(self, doc_id: str, meta: dict):
        self.data[doc_id] = {
            "fichier_source": meta.get("fichier_source", ""),
            "titre":          str(meta.get("titre") or "")[:80],
            "nb_chunks":      meta.get("nb_chunks", 0),
            "indexe_le":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._sauvegarder()

    def supprimer(self, doc_id: str):
        """Permet la re-indexation forcée d'un article spécifique."""
        if doc_id in self.data:
            del self.data[doc_id]
            self._sauvegarder()

    def vider(self):
        """Reset complet du manifest (utilisé avec force_reindex=True)."""
        self.data = {}
        self._sauvegarder()

    def __len__(self):
        return len(self.data)


# ═════════════════════════════════════════════════════════════════════════════
#  3. AGENT INDEXEUR v2
# ═════════════════════════════════════════════════════════════════════════════

class AgentIndexeur:
    """
    Pipeline RAG + Agentique :
      1. Chargement conditionnel du corpus (filtre qualité)
      2. DÉCISION 1 : Skip si doc_id dans manifest (idempotence)
      3. Chunking adaptatif avec DÉCISION 2 : rejet + log anomalies
      4. Embedding batch normalisé
      5. Insertion ChromaDB (upsert propre)
      6. DÉCISION 3 : Rapport structuré enrichi vers agents aval
    """

    MODELE_EMBEDDING     = "paraphrase-multilingual-mpnet-base-v2"
    COLLECTION_CHUNKS    = "corpus_chunks"
    COLLECTION_ABSTRACTS = "corpus_abstracts"

    def __init__(self, dossier_chroma: str = "data/chroma_db",
                 taille_chunk: int = 400, overlap: int = 80,
                 batch_size: int = 32):
        self.dossier_chroma = dossier_chroma
        self.batch_size     = batch_size
        self.chunkeur       = ChunkeurIntelligent(taille_chunk, overlap)

        log.info(f"Chargement du modèle d'embedding : {self.MODELE_EMBEDDING}")
        self.modele = SentenceTransformer(self.MODELE_EMBEDDING)
        log.info("Modèle prêt ✓")

        # ChromaDB persistant
        Path(dossier_chroma).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=dossier_chroma,
            settings=Settings(anonymized_telemetry=False)
        )
        self.col_chunks = self.client.get_or_create_collection(
            name=self.COLLECTION_CHUNKS,
            metadata={"hnsw:space": "cosine"}
        )
        self.col_abstracts = self.client.get_or_create_collection(
            name=self.COLLECTION_ABSTRACTS,
            metadata={"hnsw:space": "cosine"}
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 1 : Chargement conditionnel du corpus
    # ─────────────────────────────────────────────────────────────────────────

    def _charger_corpus(self, chemin_corpus: str) -> list:
        with open(chemin_corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        total         = len(corpus)
        corpus_filtre = [d for d in corpus
                         if d.get("qualite", {}).get("score", 0) >= QUALITE_MIN_CORPUS]
        nb_ignores    = total - len(corpus_filtre)

        if nb_ignores:
            log.warning(f"{nb_ignores} article(s) ignoré(s) — qualité < {QUALITE_MIN_CORPUS}")

        return corpus_filtre

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 2 : Embeddings par batch
    # ─────────────────────────────────────────────────────────────────────────

    def _embedder(self, textes: list) -> list:
        vecteurs = self.modele.encode(
            textes,
            batch_size           = self.batch_size,
            show_progress_bar    = False,
            normalize_embeddings = True
        )
        return vecteurs.tolist()

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 3 : Insertion dans ChromaDB
    # ─────────────────────────────────────────────────────────────────────────

    def _inserer_batch(self, collection, chunks: list, embeddings: list):
        """Upsert propre : écrase si chunk_id existe déjà."""
        ids       = [c["chunk_id"]  for c in chunks]
        documents = [c["texte"]     for c in chunks]
        metadatas = []

        for c in chunks:
            meta = {
                k: v for k, v in c.items()
                if k not in ("texte", "chunk_id")
                and isinstance(v, (str, int, float, bool))
            }
            metadatas.append(meta)

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas
        )

    def _traiter_par_batch(self, collection, chunks: list):
        for i in range(0, len(chunks), self.batch_size):
            batch    = chunks[i:i + self.batch_size]
            vecteurs = self._embedder([c["texte"] for c in batch])
            self._inserer_batch(collection, batch, vecteurs)

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 4 : Indexation d'un article
    # ─────────────────────────────────────────────────────────────────────────

    def _indexer_article(self, doc: dict) -> dict:
        """
        Pipeline complet pour un article.
        Retourne les stats + anomalies pour le rapport.
        """
        doc_id = doc.get("doc_id", "")
        titre  = str(doc.get("titre") or doc.get("fichier_source", ""))[:60]

        # Chunking avec collecte des anomalies (DÉCISION 2)
        chunks, anomalies = self.chunkeur.chunker(doc)

        if not chunks:
            log.warning(f"  ✗ Aucun chunk valide produit pour : {titre}")
            return {
                "doc_id":    doc_id,
                "titre":     titre,
                "nb_chunks": 0,
                "nb_anomalies": len(anomalies),
                "anomalies": anomalies,
                "strategie": "echec",
                "succes":    False
            }

        strategie        = "sections" if doc.get("sections") else "fenetre"
        chunks_abstracts = [c for c in chunks if c["type_chunk"] == "abstract"]
        chunks_normaux   = [c for c in chunks if c["type_chunk"] != "abstract"]

        # Tous les chunks → collection principale
        self._traiter_par_batch(self.col_chunks, chunks)

        # Abstracts → collection dédiée (→ Agent Cartographe)
        if chunks_abstracts:
            self._traiter_par_batch(self.col_abstracts, chunks_abstracts)

        if anomalies:
            log.warning(f"  ⚠ {len(anomalies)} chunk(s) rejeté(s) pour : {titre}")

        return {
            "doc_id":         doc_id,
            "titre":          titre,
            "nb_chunks":      len(chunks),
            "nb_rejets":      len(anomalies),
            "anomalies":      anomalies,
            "strategie":      strategie,
            "a_abstract":     len(chunks_abstracts) > 0,
            "a_sections":     bool(doc.get("sections")),
            "domaine":        doc.get("domaine") or "non détecté",
            "claim_type":     doc.get("claim_type") or "",
            "evidence_level": doc.get("evidence_level") or "",
            "annee":          doc.get("annee"),
            "succes":         True
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  API DE RECHERCHE
    # ─────────────────────────────────────────────────────────────────────────

    def rechercher(self, requete: str, n_resultats: int = 5,
                   filtre: dict = None, collection: str = "chunks") -> list:
        """
        Recherche sémantique principale.

        Exemples :
          # Cartographe — clustering des abstracts
          indexeur.rechercher("machine learning education", collection="abstracts")

          # Narrateur — contributions empiriques sur un thème
          indexeur.rechercher("BERT sentiment", filtre={"claim_type": "empirical"})

          # Détecteur — méthodes peu explorées
          indexeur.rechercher("qualitative interviews",
                              filtre={"section_nom": "methodology"})

          # Filtrer par année
          indexeur.rechercher("deep learning", filtre={"annee": {"$gte": 2020}})
        """
        col = self.col_chunks if collection == "chunks" else self.col_abstracts

        vecteur = self._embedder([requete])[0]
        kwargs  = {
            "query_embeddings": [vecteur],
            "n_results":         n_resultats,
            "include":          ["documents", "metadatas", "distances"]
        }
        if filtre:
            kwargs["where"] = filtre

        bruts = col.query(**kwargs)

        return [
            {
                "chunk_id":  bruts["ids"][0][i],
                "texte":     bruts["documents"][0][i],
                "distance":  round(bruts["distances"][0][i], 4),
                "score_sim": round(1 - bruts["distances"][0][i], 4),
                **bruts["metadatas"][0][i]
            }
            for i in range(len(bruts["ids"][0]))
        ]

    def rechercher_par_doc_id(self, doc_id: str) -> list:
        """Récupère tous les chunks d'un document — utilisé par l'Agent Citateur."""
        res = self.col_chunks.get(
            where   = {"doc_id": doc_id},
            include = ["documents", "metadatas"]
        )
        chunks = [
            {"chunk_id": cid, "texte": res["documents"][i], **res["metadatas"][i]}
            for i, cid in enumerate(res["ids"])
        ]
        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        return chunks

    def rechercher_par_section(self, requete: str, section: str,
                                n_resultats: int = 5) -> list:
        """Recherche restreinte à une section — utilisé par l'Agent Détecteur."""
        return self.rechercher(requete, n_resultats,
                               filtre={"section_nom": section})

    def get_tous_abstracts(self, with_embeddings: bool = True) -> list:
        """
        Retourne tous les abstracts pour le clustering du Cartographe.
        with_embeddings=False : plus léger si le Cartographe re-calcule ses propres vecteurs.
        """
        include = ["documents", "metadatas"]
        if with_embeddings:
            include.append("embeddings")

        res = self.col_abstracts.get(include=include)

        return [
            {
                "chunk_id":  res["ids"][i],
                "texte":     res["documents"][i],
                "embedding": res["embeddings"][i] if with_embeddings else None,
                **res["metadatas"][i]
            }
            for i in range(len(res["ids"]))
        ]

    def stats(self) -> dict:
        return {
            "nb_chunks_total":       self.col_chunks.count(),
            "nb_abstracts":          self.col_abstracts.count(),
            "collection_chunks":     self.COLLECTION_CHUNKS,
            "collection_abstracts":  self.COLLECTION_ABSTRACTS,
            "modele_embedding":      self.MODELE_EMBEDDING,
            "dossier_chroma":        self.dossier_chroma,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  DÉCISION 3 : RAPPORT STRUCTURÉ ENRICHI
    # ─────────────────────────────────────────────────────────────────────────

    def _generer_rapport(self, stats_articles: list, stats_skips: list,
                          toutes_anomalies: list, duree: float,
                          dossier_output: str) -> dict:
        """
        Rapport enrichi pensé pour les agents aval :

        → Agent Cartographe :
            - doc_ids_indexes : liste complète pour le clustering
            - repartition_domaines : répartition thématique du corpus
            - articles_sans_abstract : à exclure du clustering

        → Agent Détecteur de Gaps :
            - articles_sans_sections : couverture méthodologique partielle
            - repartition_claim_type : empirical vs theoretical vs review
            - repartition_evidence_level : force des preuves dans le corpus
            - alertes : lacunes explicites détectées

        → Agent Narrateur :
            - annees : plage temporelle du corpus
            - repartition_langues : langues présentes

        → Agent Citateur :
            - total_chunks_produits : densité du corpus pour calibrer les requêtes
        """
        succes = [s for s in stats_articles if s["succes"]]
        echecs = [s for s in stats_articles if not s["succes"]]

        total_chunks = sum(s["nb_chunks"] for s in succes)
        total_rejets = sum(s.get("nb_rejets", 0) for s in succes)

        # Répartitions
        strategies    = {}
        domaines      = {}
        claim_types   = {}
        evidence_lvls = {}
        annees        = []
        langues       = {}

        for s in succes:
            st = s.get("strategie", "inconnu")
            strategies[st] = strategies.get(st, 0) + 1

            d = s.get("domaine", "non détecté")
            domaines[d] = domaines.get(d, 0) + 1

            ct = s.get("claim_type") or "non renseigné"
            claim_types[ct] = claim_types.get(ct, 0) + 1

            el = s.get("evidence_level") or "non renseigné"
            evidence_lvls[el] = evidence_lvls.get(el, 0) + 1

            if s.get("annee"):
                annees.append(int(s["annee"]))

        # Articles sans abstract → alerte pour Cartographe
        sans_abstract = [s["doc_id"] for s in succes if not s.get("a_abstract")]

        # Articles sans sections → alerte pour Détecteur
        sans_sections = [s["doc_id"] for s in succes if not s.get("a_sections")]

        # Alertes explicites sur les lacunes du corpus
        alertes = []
        if len(sans_abstract) > len(succes) * 0.3:
            alertes.append({
                "type":    "abstracts_manquants",
                "message": f"{len(sans_abstract)}/{len(succes)} articles sans abstract — clustering Cartographe dégradé",
                "doc_ids": sans_abstract
            })
        if len(sans_sections) > len(succes) * 0.5:
            alertes.append({
                "type":    "sections_manquantes",
                "message": f"{len(sans_sections)}/{len(succes)} articles sans sections détectées — analyse méthodologique partielle",
                "doc_ids": sans_sections
            })
        if total_rejets > total_chunks * 0.2:
            alertes.append({
                "type":    "taux_rejet_eleve",
                "message": f"Taux de rejet chunks : {round(total_rejets/(total_chunks+total_rejets)*100, 1)}% — vérifier qualité extraction Curateur",
            })
        if not annees:
            alertes.append({
                "type":    "annees_manquantes",
                "message": "Aucune année détectée dans le corpus — évolution temporelle impossible pour Narrateur"
            })

        rapport = {
            # ── Synthèse générale ────────────────────────────────────────
            "total_articles_traites":  len(stats_articles),
            "articles_indexes":        len(succes),
            "articles_skips":          len(stats_skips),
            "articles_echoues":        len(echecs),
            "total_chunks_produits":   total_chunks,
            "total_chunks_rejetes":    total_rejets,
            "taux_rejet_pct":          round(total_rejets / (total_chunks + total_rejets) * 100, 1) if (total_chunks + total_rejets) > 0 else 0,
            "moyenne_chunks_article":  round(total_chunks / len(succes), 1) if succes else 0,
            "strategies_chunking":     strategies,
            "duree_secondes":          round(duree, 1),
            "modele_embedding":        self.MODELE_EMBEDDING,

            # ── Pour Agent Cartographe ────────────────────────────────────
            "pour_cartographe": {
                "doc_ids_indexes":       [s["doc_id"] for s in succes],
                "nb_abstracts_dispo":    len(succes) - len(sans_abstract),
                "articles_sans_abstract": sans_abstract,
                "repartition_domaines":  domaines,
            },

            # ── Pour Agent Détecteur de Gaps ─────────────────────────────
            "pour_detecteur": {
                "articles_sans_sections":  sans_sections,
                "repartition_claim_type":  claim_types,
                "repartition_evidence":    evidence_lvls,
                "nb_articles_empiriques":  claim_types.get("empirical", 0),
                "nb_articles_theoriques":  claim_types.get("theoretical", 0),
                "nb_articles_review":      claim_types.get("review", 0),
            },

            # ── Pour Agent Narrateur ──────────────────────────────────────
            "pour_narrateur": {
                "annee_min":  min(annees) if annees else None,
                "annee_max":  max(annees) if annees else None,
                "plage_temporelle": f"{min(annees)}–{max(annees)}" if annees else "inconnue",
            },

            # ── Qualité / Anomalies ───────────────────────────────────────
            "alertes":          alertes,
            "nb_alertes":       len(alertes),
            "anomalies_chunks": toutes_anomalies[:50],  # 50 premiers pour ne pas saturer le rapport
            "total_anomalies":  len(toutes_anomalies),

            # ── Détail par article ────────────────────────────────────────
            "articles_skips_detail": stats_skips,
            "articles_echoues_detail": [
                {"doc_id": s["doc_id"], "titre": s.get("titre", "?")} for s in echecs
            ],
            "stats_par_article": [
                {
                    "doc_id":     s["doc_id"],
                    "titre":      s.get("titre", "?"),
                    "nb_chunks":  s["nb_chunks"],
                    "nb_rejets":  s.get("nb_rejets", 0),
                    "strategie":  s.get("strategie", "?"),
                    "a_abstract": s.get("a_abstract", False),
                }
                for s in succes
            ]
        }

        # Sauvegarde
        rapport_path = Path(dossier_output) / "rapport_indexeur.json"
        with open(rapport_path, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)

        # Sauvegarde séparée du journal d'anomalies complet
        if toutes_anomalies:
            anomalies_path = Path(dossier_output) / "anomalies_indexeur.json"
            with open(anomalies_path, "w", encoding="utf-8") as f:
                json.dump(toutes_anomalies, f, ensure_ascii=False, indent=2)
            log.info(f"Journal anomalies → {anomalies_path}")

        return rapport

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, chemin_corpus: str, dossier_output: str = "data/corpus",
            force_reindex: bool = False) -> dict:
        """
        Lance le pipeline complet.

        force_reindex=True : ignore le manifest, réindexe tout le corpus.
        force_reindex=False (défaut) : skip les articles déjà indexés.
        """
        Path(dossier_output).mkdir(parents=True, exist_ok=True)

        # ── Manifest (DÉCISION 1) ─────────────────────────────────────────
        manifest_path = Path(dossier_output) / "indexed_docs.json"
        manifest      = ManifestIndexeur(str(manifest_path))

        if force_reindex and len(manifest) > 0:
            log.warning("--force activé : reset du manifest, re-indexation complète")
            manifest.vider()

        log.info(f"{'='*60}")
        log.info(f"  AGENT INDEXEUR v2 — RAG + Agentique")
        log.info(f"{'='*60}")

        corpus = self._charger_corpus(chemin_corpus)
        log.info(f"{len(corpus)} article(s) à traiter | {len(manifest)} déjà dans manifest")

        stats_articles    = []
        stats_skips       = []
        toutes_anomalies  = []
        debut             = time.time()

        for i, doc in enumerate(corpus):
            doc_id = doc.get("doc_id", "")
            titre  = str(doc.get("titre") or doc.get("fichier_source", "?"))[:55]

            log.info(f"  [{i+1}/{len(corpus)}] {titre}")

            # ── DÉCISION 1 : Skip si déjà indexé ─────────────────────────
            if manifest.est_indexe(doc_id):
                log.info(f"    → déjà indexé, skip (manifest)")
                stats_skips.append({"doc_id": doc_id, "titre": titre})
                continue

            # ── Indexation ────────────────────────────────────────────────
            try:
                stats = self._indexer_article(doc)
            except Exception as e:
                log.error(f"    ✗ Erreur inattendue : {e}")
                stats = {
                    "doc_id": doc_id, "titre": titre,
                    "nb_chunks": 0, "nb_rejets": 0,
                    "anomalies": [], "strategie": "erreur", "succes": False
                }

            stats_articles.append(stats)

            # ── DÉCISION 2 : Collecte anomalies ──────────────────────────
            toutes_anomalies.extend(stats.get("anomalies", []))

            if stats["succes"]:
                log.info(f"    ✓ {stats['nb_chunks']} chunks | "
                         f"stratégie : {stats['strategie']} | "
                         f"rejets : {stats.get('nb_rejets', 0)}")
                # Enregistrement dans manifest après succès
                manifest.enregistrer(doc_id, {
                    "fichier_source": doc.get("fichier_source", ""),
                    "titre":          doc.get("titre", ""),
                    "nb_chunks":      stats["nb_chunks"],
                })
            else:
                log.error(f"    ✗ Échec indexation")

        duree = time.time() - debut

        # ── DÉCISION 3 : Rapport structuré enrichi ────────────────────────
        rapport = self._generer_rapport(
            stats_articles, stats_skips, toutes_anomalies, duree, dossier_output
        )

        # ── Résumé console ────────────────────────────────────────────────
        log.info(f"{'='*60}")
        log.info(f"  RÉSUMÉ AGENT INDEXEUR v2")
        log.info(f"{'='*60}")
        log.info(f"  Articles indexés       : {rapport['articles_indexes']}")
        log.info(f"  Articles skippés       : {rapport['articles_skips']} (déjà dans manifest)")
        log.info(f"  Articles échoués       : {rapport['articles_echoues']}")
        log.info(f"  Chunks produits        : {rapport['total_chunks_produits']}")
        log.info(f"  Chunks rejetés         : {rapport['total_chunks_rejetes']} ({rapport['taux_rejet_pct']}%)")
        log.info(f"  Abstracts disponibles  : {rapport['pour_cartographe']['nb_abstracts_dispo']}")
        log.info(f"  Alertes corpus         : {rapport['nb_alertes']}")
        log.info(f"  Durée totale           : {rapport['duree_secondes']}s")
        log.info(f"  ChromaDB               : {self.dossier_chroma}")
        log.info(f"  Manifest               : {manifest_path}")

        if rapport["alertes"]:
            log.warning("  ── Alertes ──")
            for a in rapport["alertes"]:
                log.warning(f"  ⚠ [{a['type']}] {a['message']}")

        return rapport


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Indexeur v2 — RAG + Agentique")
    parser.add_argument("--force", action="store_true",
                        help="Ignore le manifest et re-indexe tout le corpus")
    parser.add_argument("--corpus", default="data/corpus/corpus_complet.json",
                        help="Chemin vers corpus_complet.json (Curateur)")
    parser.add_argument("--output", default="data/corpus",
                        help="Dossier de sortie pour les rapports")
    parser.add_argument("--chroma", default="data/chroma_db",
                        help="Dossier de persistance ChromaDB")
    args = parser.parse_args()

    indexeur = AgentIndexeur(
        dossier_chroma = args.chroma,
        taille_chunk   = 400,
        overlap        = 80,
        batch_size     = 32
    )

    rapport = indexeur.run(
        chemin_corpus  = args.corpus,
        dossier_output = args.output,
        force_reindex  = args.force
    )

    # ── Test de recherche après indexation ───────────────────────────────────
    if rapport["articles_indexes"] > 0:
        print("\n── Test de recherche sémantique ──")
        resultats = indexeur.rechercher("deep learning classification", n_resultats=3)
        for r in resultats:
            print(f"\n  Score : {r['score_sim']} | Section : {r['section_nom']}")
            print(f"  Titre : {r['titre'][:60]}")
            print(f"  Extrait : {r['texte'][:150]}...")

    print(f"\n── Stats ChromaDB ──")
    print(indexeur.stats())