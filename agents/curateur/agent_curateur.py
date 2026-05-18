"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AGENT CURATEUR — SMA Revue de Littérature v3                  ║
║  Point d'entrée du pipeline. Produit des documents complets pour :          ║
║  → Agent Indexeur     : texte_nettoye, sections, doc_id                     ║
║  → Agent Cartographe  : abstract, resume_court, mots_cles, methodes, annee  ║
║  → Agent Narrateur    : contribution_principale, claim_type, evidence_level  ║
║  → Agent Detecteur    : populations, contexte_geo, periode, limites          ║
║  → Agent Citateur     : doi, url, references_extraites                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation :
    pip install groq pdfplumber pymupdf pytesseract pillow python-dotenv

Tesseract (OCR) :
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Linux   : sudo apt install tesseract-ocr tesseract-ocr-fra
    macOS   : brew install tesseract tesseract-lang
"""

import os
import re
import json
import uuid
import io
import time
from pathlib import Path

import fitz           # PyMuPDF
if not hasattr(fitz, "open"):
    raise ImportError("Conflit de module : Le package 'fitz' installé n'est pas PyMuPDF. "
                      "Action requise : pip uninstall fitz && pip install --force-reinstall pymupdf")

import pdfplumber
from groq import Groq
from dotenv import load_dotenv

# ── Imports conditionnels ─────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    print("  pytesseract non installé → pip install pytesseract pillow")

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_OK = True
except ImportError:
    UNSTRUCTURED_OK = False

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
#  1. DÉTECTEUR DE TYPE PDF
# ═════════════════════════════════════════════════════════════════════════════

class DetecteurPDF:
    """
    Analyse un PDF avant extraction pour choisir la bonne stratégie.
    Retourne : type ("natif"|"scanne"|"mixte"), ratio texte, pages scannées,
               présence de tableaux, nombre de pages.
    """

    @staticmethod
    def analyser(pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        nb_pages = len(doc)
        pages_avec_texte = 0
        pages_sans_texte = []

        for i, page in enumerate(doc):
            texte = page.get_text().strip()
            if len(texte) > 50:
                pages_avec_texte += 1
            else:
                pages_sans_texte.append(i)

        doc.close()

        # Détection tableaux sur les 5 premières pages
        a_tableaux = False
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:5]:
                    tables = page.extract_tables()
                    if tables and any(len(t) > 1 for t in tables):
                        a_tableaux = True
                        break
        except Exception:
            pass

        ratio = pages_avec_texte / nb_pages if nb_pages > 0 else 0

        if ratio >= 0.8:
            type_pdf = "natif"
        elif ratio <= 0.2:
            type_pdf = "scanne"
        else:
            type_pdf = "mixte"

        return {
            "type":            type_pdf,
            "a_tableaux":      a_tableaux,
            "nb_pages":        nb_pages,
            "ratio_texte":     round(ratio, 2),
            "pages_sans_texte": pages_sans_texte
        }


# ═════════════════════════════════════════════════════════════════════════════
#  2. EXTRACTEUR NATIF (texte + tableaux structurés)
# ═════════════════════════════════════════════════════════════════════════════

class ExtracteurNatif:
    """
    3 stratégies en cascade :
      1. pdfplumber  → texte + tableaux JSON
      2. PyMuPDF texte simple → fallback
      3. PyMuPDF blocs triés → fallback multi-colonnes
    """

    def extraire(self, pdf_path: str, **kwargs) -> dict:
        texte_total = ""
        tableaux    = []
        methode     = ""

        # ── Stratégie 1 : pdfplumber ──────────────────────────────────────
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text()
                    if t and t.strip():
                        texte_total += f"\n--- Page {i+1} ---\n{t.strip()}"

                    # Tableaux
                    for j, table in enumerate(page.extract_tables() or []):
                        tab = self._nettoyer_tableau(table, i + 1, j + 1)
                        if tab:
                            tableaux.append(tab)

            if len(texte_total.strip()) > 200:
                methode = "pdfplumber"
                return {"texte": texte_total.strip(), "tableaux": tableaux, "methode_extraction": methode}

        except Exception as e:
            print(f"    pdfplumber échoué : {e}")

        # ── Stratégie 2 : PyMuPDF texte ───────────────────────────────────
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                t = page.get_text("text").strip()
                if t:
                    texte_total += f"\n--- Page {i+1} ---\n{t}"
            doc.close()

            if len(texte_total.strip()) > 200:
                methode = "pymupdf_text"
                return {"texte": texte_total.strip(), "tableaux": tableaux, "methode_extraction": methode}

        except Exception as e:
            print(f"    PyMuPDF texte échoué : {e}")

        # ── Stratégie 3 : PyMuPDF blocs (multi-colonnes) ─────────────────
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                blocs = page.get_text("blocks")
                # Tri par bande verticale (20px) puis gauche→droite
                blocs_tries = sorted(blocs, key=lambda b: (round(b[1] / 20) * 20, b[0]))
                lignes = [b[4].strip() for b in blocs_tries if b[6] == 0 and b[4].strip()]
                texte_total += f"\n--- Page {i+1} ---\n" + "\n".join(lignes)
            doc.close()
            methode = "pymupdf_blocs"

        except Exception as e:
            print(f"    PyMuPDF blocs échoué : {e}")
            methode = "echec"

        return {"texte": texte_total.strip(), "tableaux": tableaux, "methode_extraction": methode}

    # ── Nettoyage tableau ──────────────────────────────────────────────────

    def _nettoyer_tableau(self, table: list, num_page: int, num_tableau: int) -> dict:
        """
        Convertit un tableau brut pdfplumber en dict propre.
        Gère : cellules None, headers vides, headers dupliqués, lignes vides.
        """
        if not table or len(table) < 2:
            return None

        # Nettoyage des None
        table_propre = [
            [str(c).strip() if c is not None else "" for c in row]
            for row in table
        ]
        # Suppression des lignes entièrement vides
        table_propre = [r for r in table_propre if any(c != "" for c in r)]

        if len(table_propre) < 2:
            return None

        # Headers : nommer les vides col_N, dédoublonner les doublons
        bruts = table_propre[0]
        headers = []
        compteur = {}
        for k, h in enumerate(bruts):
            nom = h if h else f"col_{k+1}"
            if bruts.count(h) > 1 and h:
                compteur[nom] = compteur.get(nom, 0) + 1
                nom = f"{nom}_{compteur[nom]}"
            headers.append(nom)

        # Lignes → liste de dicts
        rows = []
        for row in table_propre[1:]:
            row_dict = {headers[k]: v for k, v in enumerate(row) if k < len(headers)}
            rows.append(row_dict)

        return {
            "page":          num_page,
            "tableau_index": num_tableau,
            "headers":       headers,
            "nb_lignes":     len(rows),
            "rows":          rows
        }


# ═════════════════════════════════════════════════════════════════════════════
#  3. EXTRACTEUR OCR (PDFs scannés)
# ═════════════════════════════════════════════════════════════════════════════

class ExtracteurOCR:
    """
    OCR via Tesseract.
    pages_cibles = None → toutes les pages
    pages_cibles = [2, 5, 7] → seulement ces pages (index 0-based)
    """

    def __init__(self, langue: str = "fra+eng"):
        self.langue = langue

    def extraire(self, pdf_path: str, pages_cibles: list = None, **kwargs) -> dict:
        if not TESSERACT_OK:
            print("  ⚠  Tesseract non disponible — texte vide.")
            return {"texte": "", "tableaux": [], "methode_extraction": "ocr_indisponible"}

        doc = fitz.open(pdf_path)
        pages_a_traiter = pages_cibles if pages_cibles is not None else range(len(doc))
        texte_total = ""

        for i in pages_a_traiter:
            page = doc[i]
            # Rendu 300 DPI pour bonne qualité OCR
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            # Conversion correcte bytes → PIL Image
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # Prétraitement léger : passage en niveaux de gris
            img = img.convert("L")

            config = "--oem 3 --psm 6"
            texte_page = pytesseract.image_to_string(img, lang=self.langue, config=config)

            if texte_page.strip():
                texte_total += f"\n--- Page {i+1} (OCR) ---\n{texte_page.strip()}"

        doc.close()
        return {
            "texte":             texte_total.strip(),
            "tableaux":          [],
            "methode_extraction": "tesseract_ocr"
        }


# ═════════════════════════════════════════════════════════════════════════════
#  4. EXTRACTEUR MIXTE (pages natives + pages scannées)
# ═════════════════════════════════════════════════════════════════════════════

class ExtracteurMixte:
    """
    Traite chaque page selon son type :
    - Pages avec texte natif  → ExtracteurNatif
    - Pages sans texte (image) → ExtracteurOCR ciblé
    Fusionne les deux résultats en ordre de pages.
    """

    def __init__(self, langue_ocr: str = "fra+eng"):
        self.natif = ExtracteurNatif()
        self.ocr   = ExtracteurOCR(langue=langue_ocr)

    def extraire(self, pdf_path: str, pages_sans_texte: list = None, **kwargs) -> dict:
        pages_sans_texte = pages_sans_texte or []

        # Extraction natif sur tout le document
        res_natif = self.natif.extraire(pdf_path)

        # OCR ciblé uniquement sur les pages scannées
        res_ocr = {"texte": "", "tableaux": []}
        if pages_sans_texte and TESSERACT_OK:
            print(f"    OCR ciblé : {len(pages_sans_texte)} page(s) scannée(s)")
            res_ocr = self.ocr.extraire(pdf_path, pages_cibles=pages_sans_texte)

        texte_fusionne = res_natif["texte"]
        if res_ocr["texte"]:
            texte_fusionne += "\n\n" + res_ocr["texte"]

        return {
            "texte":             texte_fusionne.strip(),
            "tableaux":          res_natif["tableaux"] + res_ocr.get("tableaux", []),
            "methode_extraction": "mixte_natif+ocr"
        }


# ═════════════════════════════════════════════════════════════════════════════
#  5. AGENT CURATEUR — Orchestrateur principal
# ═════════════════════════════════════════════════════════════════════════════

class AgentCurateur:
    """
    Pipeline complet :
      Détection → Extraction adaptée → Nettoyage → Sections
      → Métadonnées LLM → Références LLM → Qualité → Document final
    """

    # Sections connues d'un article scientifique
    SECTIONS_CIBLES = [
        "abstract", "introduction", "related work", "literature review",
        "background", "methodology", "methods", "proposed method",
        "approach", "framework", "model", "architecture",
        "experiments", "experimental setup", "evaluation",
        "results", "discussion", "conclusion", "future work",
        "acknowledgments", "acknowledgements", "references"
    ]

    def __init__(self):
        self.client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model   = "llama-3.3-70b-versatile"
        self.detecteur         = DetecteurPDF()
        self.extracteur_natif  = ExtracteurNatif()
        self.extracteur_ocr    = ExtracteurOCR(langue="fra+eng")
        self.extracteur_mixte  = ExtracteurMixte(langue_ocr="fra+eng")

    # ─────────────────────────────────────────────────────────────────────────
    #  ROUTAGE : choisit l'extracteur selon le type de PDF
    # ─────────────────────────────────────────────────────────────────────────

    def _choisir_extracteur(self, info: dict):
        """Retourne (extracteur, kwargs) selon le type détecté."""
        if info["type"] == "scanne":
            return self.extracteur_ocr, {"pages_cibles": None}
        if info["type"] == "mixte":
            return self.extracteur_mixte, {"pages_sans_texte": info["pages_sans_texte"]}
        return self.extracteur_natif, {}

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 1 : Nettoyage du texte extrait
    # ─────────────────────────────────────────────────────────────────────────

    def _nettoyer_texte(self, texte: str) -> str:
        """
        Nettoie le texte brut pour améliorer :
        - la qualité des embeddings (Agent Indexeur)
        - la similarité sémantique
        - la cohérence du chunking
        """
        # Normalisation des fins de ligne
        texte = texte.replace("\r\n", "\n").replace("\r", "\n")

        # Fusion des mots coupés en fin de ligne (ex: "algo-\nrithme" → "algorithme")
        texte = re.sub(r'-\n(\w)', r'\1', texte)

        # Fusion des lignes brisées au milieu d'une phrase
        texte = re.sub(r'(?<=[a-z,;:])\n(?=[a-z])', ' ', texte)

        # Suppression des caractères parasites (OCR, encodage cassé)
        texte = re.sub(
            r'[^\x00-\x7Féàèùâêîôûäëïöüçœæ\s\-\'\"\(\)\[\]\{\}\.,;:!?\d°%€$#@&*+=<>/\\]',
            '', texte
        )

        # Suppression des en-têtes/pieds de page (lignes très courtes < 4 chars)
        lignes = [l for l in texte.split("\n") if len(l.strip()) > 3 or l.strip() == ""]

        texte = "\n".join(lignes)
        texte = re.sub(r'[ \t]+', ' ', texte)       # espaces multiples
        texte = re.sub(r'\n{3,}', '\n\n', texte)    # sauts de ligne excessifs

        return texte.strip()

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 2 : Détection des sections
    # ─────────────────────────────────────────────────────────────────────────

    def _detecter_sections(self, texte: str) -> dict:
        """
        Identifie et extrait le contenu de chaque section.
        Utilisé par l'Agent Indexeur pour un chunking intelligent par section.
        """
        sections = {}
        lignes = texte.split("\n")
        section_courante = "preamble"
        contenu_courant  = []

        for ligne in lignes:
            ligne_lower = ligne.strip().lower()
            section_trouvee = None

            for section in self.SECTIONS_CIBLES:
                # Ligne courte contenant le nom de la section = titre de section
                if (section in ligne_lower
                        and 2 < len(ligne.strip()) < 70
                        and not ligne_lower.startswith("http")):
                    section_trouvee = section
                    break

            if section_trouvee:
                if contenu_courant:
                    sections[section_courante] = "\n".join(contenu_courant).strip()
                section_courante = section_trouvee
                contenu_courant  = []
            else:
                contenu_courant.append(ligne)

        if contenu_courant:
            sections[section_courante] = "\n".join(contenu_courant).strip()

        # Ne garder que les sections avec assez de contenu
        return {k: v for k, v in sections.items() if len(v) > 50}

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 3 : Extraction des métadonnées via LLM
    # ─────────────────────────────────────────────────────────────────────────

    def _extraire_metadonnees(self, texte: str, nom_fichier: str) -> dict:
        """
        Appel LLM pour extraire toutes les métadonnées nécessaires aux agents.

        Champs pour chaque agent :
        - Indexeur    : abstract, mots_cles, methodes
        - Cartographe : abstract, resume_court, mots_cles, methodes, annee, domaine
        - Narrateur   : contribution_principale, claim_type, evidence_level
        - Detecteur   : populations_etudiees, contexte_geographique, periode_etude,
                        limites_declarees
        - Citateur    : doi, url, langue
        """
        # On utilise les 4000 premiers caractères : plus fiable pour le LLM
        extrait = texte[:4000]

        prompt = f"""Tu es un expert en analyse d'articles scientifiques.

Voici le début d'un article scientifique :

---
{extrait}
---

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, sans balises markdown.

{{
  "titre": "titre complet de l'article",
  "auteurs": ["Nom Prénom"],
  "annee": 2024,
  "doi": "10.xxxx/xxxxx ou null",
  "url": "lien si visible dans le document ou null",
  "langue": "fr ou en ou ar ou autre",
  "journal_ou_conference": "nom complet du journal ou de la conférence",
  "abstract": "résumé complet tel qu'il apparaît dans l'article",
  "resume_court": "synthèse de la contribution en 3 à 5 phrases simples et accessibles",
  "mots_cles": ["mot1", "mot2"],
  "methodes": ["méthode ou technique utilisée"],
  "domaine": "domaine de recherche principal (ex: NLP, Computer Vision, Education)",
  "populations_etudiees": ["type de population ou d'objet étudié (ex: étudiants, tweets, images médicales)"],
  "contexte_geographique": ["pays ou région concerné par l'étude ou null"],
  "periode_etude": "période couverte par l'étude (ex: 2018-2022) ou null",
  "contribution_principale": "la contribution centrale de cet article en une phrase",
  "claim_type": "empirical ou theoretical ou methodological ou review",
  "evidence_level": "strong ou moderate ou weak ou anecdotal",
  "limites_declarees": ["limite explicitement mentionnée par les auteurs"],
  "type_document": "article ou thèse ou rapport ou survey ou autre"
}}

Si une information est absente, mets null. Ne génère rien d'autre que le JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )

        raw = response.choices[0].message.content.strip()

        # Extraction robuste du JSON même si le LLM ajoute du texte autour
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ⚠  JSON invalide — fallback minimal")
            meta = {
                "titre":                   nom_fichier,
                "auteurs":                 [],
                "annee":                   None,
                "doi":                     None,
                "url":                     None,
                "langue":                  None,
                "journal_ou_conference":   None,
                "abstract":                extrait[:500],
                "resume_court":            None,
                "mots_cles":               [],
                "methodes":                [],
                "domaine":                 None,
                "populations_etudiees":    [],
                "contexte_geographique":   [],
                "periode_etude":           None,
                "contribution_principale": None,
                "claim_type":              None,
                "evidence_level":          None,
                "limites_declarees":       [],
                "type_document":           None,
                "erreur_parsing":          True
            }

        return meta

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 4 : Extraction des références bibliographiques
    # ─────────────────────────────────────────────────────────────────────────

    def _extraire_references(self, section_references: str) -> list:
        """
        Parse la section References en liste structurée.
        Indispensable pour l'Agent Citateur (vérification croisée des claims).
        """
        if not section_references or len(section_references.strip()) < 50:
            return []

        # Limite à 4000 chars pour ne pas dépasser le contexte
        extrait = section_references[:4000]

        prompt = f"""Voici la section références d'un article scientifique :

---
{extrait}
---

Réponds UNIQUEMENT avec une liste JSON valide, sans texte avant ni après :

[
  {{
    "auteurs": ["Nom Prénom"],
    "titre": "titre de l'article cité",
    "annee": 2020,
    "journal_ou_conference": "nom ou null",
    "doi": "10.xxx/xxx ou null",
    "url": "lien ou null"
  }}
]

Si une info est absente, mets null. Ne génère rien d'autre que la liste JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            raw = response.choices[0].message.content.strip()

            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                raw = match.group(0)

            return json.loads(raw)

        except Exception as e:
            print(f"  ⚠  Extraction références échouée : {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  OUTIL 5 : Score de qualité du document
    # ─────────────────────────────────────────────────────────────────────────

    def _calculer_qualite(self, texte: str, meta: dict, sections: dict,
                          references: list, tableaux: list) -> dict:
        """
        Score de 0 à 1. Permet aux agents suivants de prioriser
        les articles les plus exploitables.

        Critères :
          - Longueur du texte        (0.20)
          - Présence abstract        (0.15)
          - Richesse métadonnées     (0.20)
          - Sections scientifiques   (0.15)
          - Références extraites     (0.10)
          - Champs agents spécifiques(0.15)
          - Pas d'erreur parsing     (0.05)
        """
        score = 0.0
        details = {}

        # Longueur (0.20)
        nb_mots = len(texte.split())
        if nb_mots > 4000:
            score += 0.20; details["longueur"] = f"{nb_mots} mots (excellente)"
        elif nb_mots > 1500:
            score += 0.12; details["longueur"] = f"{nb_mots} mots (acceptable)"
        else:
            score += 0.04; details["longueur"] = f"{nb_mots} mots (faible)"

        # Abstract (0.15)
        if meta.get("abstract") and len(str(meta["abstract"])) > 100:
            score += 0.15; details["abstract"] = "présent"
        else:
            details["abstract"] = "absent"

        # Métadonnées de base (0.20) — 4 champs × 0.05
        champs_base = {
            "titre":   bool(meta.get("titre")),
            "auteurs": bool(meta.get("auteurs")),
            "annee":   bool(meta.get("annee")),
            "mots_cles": bool(meta.get("mots_cles"))
        }
        score += sum(champs_base.values()) * 0.05
        details["metadonnees_base"] = f"{sum(champs_base.values())}/4"

        # Sections scientifiques (0.15)
        sections_sci = ["methodology", "methods", "results", "discussion", "conclusion"]
        nb_sections = sum(1 for s in sections_sci if s in sections)
        score += min(nb_sections * 0.03, 0.15)
        details["sections_scientifiques"] = f"{nb_sections}/5"

        # Références extraites (0.10)
        if len(references) > 5:
            score += 0.10; details["references"] = f"{len(references)} références"
        elif len(references) > 0:
            score += 0.05; details["references"] = f"{len(references)} références"
        else:
            details["references"] = "aucune"

        # Champs utiles aux agents (0.15) — 5 champs × 0.03
        champs_agents = {
            "contribution_principale": bool(meta.get("contribution_principale")),
            "populations_etudiees":    bool(meta.get("populations_etudiees")),
            "limites_declarees":       bool(meta.get("limites_declarees")),
            "contexte_geographique":   bool(meta.get("contexte_geographique")),
            "resume_court":            bool(meta.get("resume_court"))
        }
        score += sum(champs_agents.values()) * 0.03
        details["champs_agents"] = f"{sum(champs_agents.values())}/5"

        # Pas d'erreur parsing (0.05)
        if not meta.get("erreur_parsing"):
            score += 0.05; details["parsing"] = "ok"
        else:
            details["parsing"] = "erreur"

        score_final = round(min(score, 1.0), 2)
        return {
            "score":    score_final,
            "niveau":   "élevé" if score_final >= 0.7 else "moyen" if score_final >= 0.4 else "faible",
            "nb_mots":  nb_mots,
            "nb_tableaux": len(tableaux),
            "details":  details
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  PIPELINE PRINCIPAL — un PDF → un document complet
    # ─────────────────────────────────────────────────────────────────────────

    def traiter_pdf(self, pdf_path: str) -> dict:
        nom = Path(pdf_path).name
        print(f"\n  ┌─ {nom}")

        # ── Étape 1 : Détection du type ───────────────────────────────────
        info = self.detecteur.analyser(pdf_path)
        print(f"  │  Type : {info['type']} | ratio texte : {info['ratio_texte']} "
              f"| pages : {info['nb_pages']} | tableaux : {info['a_tableaux']}")

        # ── Étape 2 : Extraction adaptée ─────────────────────────────────
        extracteur, kwargs = self._choisir_extracteur(info)
        resultat = extracteur.extraire(pdf_path, **kwargs)

        texte_brut = resultat["texte"]
        tableaux   = resultat.get("tableaux", [])
        methode    = resultat["methode_extraction"]

        if not texte_brut or len(texte_brut.strip()) < 200:
            print(f"  └─ ✗ Ignoré : texte insuffisant après extraction")
            return None

        print(f"  │  Extraction : {methode} | {len(texte_brut)} chars | {len(tableaux)} tableau(x)")

        # ── Étape 3 : Nettoyage ───────────────────────────────────────────
        texte_nettoye = self._nettoyer_texte(texte_brut)

        # ── Étape 4 : Détection des sections ─────────────────────────────
        sections = self._detecter_sections(texte_nettoye)
        print(f"  │  Sections : {list(sections.keys())}")

        # ── Étape 5 : Métadonnées LLM ────────────────────────────────────
        meta = self._extraire_metadonnees(texte_nettoye, nom)
        titre_court = str(meta.get("titre") or nom)[:65]
        print(f"  │  Titre : {titre_court}")

        # ── Étape 6 : Références LLM ─────────────────────────────────────
        # (seulement si la section references a été détectée)
        refs_brutes  = sections.get("references", "")
        references   = self._extraire_references(refs_brutes)
        print(f"  │  Références extraites : {len(references)}")

        # ── Étape 7 : Score de qualité ────────────────────────────────────
        qualite = self._calculer_qualite(texte_nettoye, meta, sections, references, tableaux)
        print(f"  └─ ✓ Qualité : {qualite['score']} ({qualite['niveau']})")

        # ── Document final structuré ───────────────────────────────────────
        document = {

            # ── Identification ─────────────────────────────────────────────
            "doc_id":         str(uuid.uuid4()),
            "fichier_source": nom,

            # ── Métadonnées de base ────────────────────────────────────────
            "titre":                  meta.get("titre"),
            "auteurs":                meta.get("auteurs", []),
            "annee":                  meta.get("annee"),
            "doi":                    meta.get("doi"),
            "url":                    meta.get("url"),
            "langue":                 meta.get("langue"),
            "journal_ou_conference":  meta.get("journal_ou_conference"),
            "type_document":          meta.get("type_document"),

            # ── Contenu sémantique (→ Indexeur, Cartographe) ───────────────
            "abstract":               meta.get("abstract"),
            "resume_court":           meta.get("resume_court"),
            "mots_cles":              meta.get("mots_cles", []),
            "methodes":               meta.get("methodes", []),
            "domaine":                meta.get("domaine"),

            # ── Analyse de la contribution (→ Narrateur) ───────────────────
            "contribution_principale": meta.get("contribution_principale"),
            "claim_type":              meta.get("claim_type"),
            "evidence_level":          meta.get("evidence_level"),

            # ── Contexte de l'étude (→ Détecteur de Gaps) ─────────────────
            "populations_etudiees":   meta.get("populations_etudiees", []),
            "contexte_geographique":  meta.get("contexte_geographique", []),
            "periode_etude":          meta.get("periode_etude"),
            "limites_declarees":      meta.get("limites_declarees", []),

            # ── Bibliographie (→ Citateur) ─────────────────────────────────
            "references_extraites":   references,

            # ── Contenu textuel complet ────────────────────────────────────
            "texte_brut":     texte_brut,
            "texte_nettoye":  texte_nettoye,
            "sections":       sections,
            "tableaux":       tableaux,

            # ── Informations techniques ────────────────────────────────────
            "extraction": {
                "methode":    methode,
                "type_pdf":   info["type"],
                "ratio_texte":info["ratio_texte"],
                "nb_pages":   info["nb_pages"],
                "a_tableaux": info["a_tableaux"]
            },

            # ── Score de qualité ───────────────────────────────────────────
            "qualite": qualite
        }

        return document

    # ─────────────────────────────────────────────────────────────────────────
    #  RUNNER — traite un dossier complet
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, dossier_pdfs: str, dossier_output: str) -> list:
        Path(dossier_output).mkdir(parents=True, exist_ok=True)
        corpus = []
        rejetes = []

        pdfs = list(Path(dossier_pdfs).glob("*.pdf"))
        print(f"\n{'='*60}")
        print(f"  AGENT CURATEUR — {len(pdfs)} PDF(s) trouvé(s)")
        print(f"{'='*60}")

        debut = time.time()

        for pdf_path in pdfs:
            chemin_json = Path(dossier_output) / (pdf_path.stem + ".json")
            
            # ── DÉCISION : Idempotence (Skip si déjà traité) ──────────────
            if chemin_json.exists():
                print(f"  │  [SKIP] {pdf_path.name} (déjà présent dans le corpus)")
                try:
                    with open(chemin_json, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                        corpus.append(doc)
                        continue
                except Exception:
                    print(f"  │  [RE-TRY] Erreur lecture JSON, retraitement de {pdf_path.name}")

            doc = self.traiter_pdf(str(pdf_path))

            if doc:
                corpus.append(doc)
                chemin = Path(dossier_output) / (pdf_path.stem + ".json")
                with open(chemin, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
            else:
                rejetes.append(pdf_path.name)

        # Sauvegarde corpus complet
        corpus_path = Path(dossier_output) / "corpus_complet.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        # Rapport de synthèse
        rapport = self._generer_rapport(corpus, rejetes, time.time() - debut)
        rapport_path = Path(dossier_output) / "rapport_curateur.json"
        with open(rapport_path, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)

        # Affichage résumé
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ AGENT CURATEUR")
        print(f"{'='*60}")
        print(f"  Articles traités : {len(corpus)}")
        print(f"  Articles rejetés : {len(rejetes)}")
        print(f"  Qualité élevée   : {sum(1 for d in corpus if d['qualite']['niveau']=='élevé')}")
        print(f"  Qualité moyenne  : {sum(1 for d in corpus if d['qualite']['niveau']=='moyen')}")
        print(f"  Qualité faible   : {sum(1 for d in corpus if d['qualite']['niveau']=='faible')}")
        print(f"  Durée totale     : {rapport['duree_secondes']}s")
        print(f"  Corpus → {corpus_path}")
        print(f"  Rapport → {rapport_path}")

        if rejetes:
            print(f"\n  PDFs rejetés :")
            for r in rejetes:
                print(f"    - {r}")

        return corpus

    def _generer_rapport(self, corpus: list, rejetes: list, duree: float) -> dict:
        """Rapport de synthèse pour audit de l'extraction."""
        types_pdf  = {}
        methodes   = {}
        domaines   = {}

        for doc in corpus:
            info = doc.get("extraction", {})
            t = info.get("type_pdf", "inconnu")
            m = info.get("methode", "inconnu")
            d = doc.get("domaine") or "non détecté"

            types_pdf[t]  = types_pdf.get(t, 0) + 1
            methodes[m]   = methodes.get(m, 0) + 1
            domaines[d]   = domaines.get(d, 0) + 1

        return {
            "total_traites":          len(corpus),
            "total_rejetes":          len(rejetes),
            "pdfs_rejetes":           rejetes,
            "duree_secondes":         round(duree, 1),
            "repartition_types_pdf":  types_pdf,
            "methodes_extraction":    methodes,
            "domaines_detectes":      domaines,
            "avec_references":        sum(1 for d in corpus if d.get("references_extraites")),
            "avec_tableaux":          sum(1 for d in corpus if d.get("tableaux")),
            "avec_doi":               sum(1 for d in corpus if d.get("doi")),
            "scores_qualite": {
                "élevé": sum(1 for d in corpus if d["qualite"]["niveau"] == "élevé"),
                "moyen": sum(1 for d in corpus if d["qualite"]["niveau"] == "moyen"),
                "faible":sum(1 for d in corpus if d["qualite"]["niveau"] == "faible"),
            },
            "score_moyen": round(
                sum(d["qualite"]["score"] for d in corpus) / len(corpus), 2
            ) if corpus else 0
        }


# ═════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    curateur = AgentCurateur()
    corpus = curateur.run(
        dossier_pdfs="data/articles",
        dossier_output="data/corpus"
    )
