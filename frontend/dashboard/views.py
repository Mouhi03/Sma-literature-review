"""
Dashboard views — reads JSON outputs from the SMA agents
and presents them in a premium web interface.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

import markdown


# ── Helper : load JSON safely ────────────────────────────────────────────────

def load_json(filename):
    """Load a JSON file from the corpus directory, return {} or [] on error."""
    filepath = settings.CORPUS_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_text(filename):
    """Load a text/markdown file from the corpus directory."""
    filepath = settings.CORPUS_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""
    return ""


def count_pdfs():
    """Count PDFs in the articles directory."""
    articles_dir = settings.ARTICLES_DIR
    if articles_dir.exists():
        return len(list(articles_dir.glob('*.pdf')))
    return 0


def get_pipeline_status():
    """Check which pipeline outputs exist."""
    checks = {
        'curateur': (settings.CORPUS_DIR / 'corpus_complet.json').exists(),
        'indexeur': (settings.CORPUS_DIR / 'rapport_indexeur.json').exists(),
        'cartographe': (settings.CORPUS_DIR / 'carte_corpus.json').exists(),
        'narrateur': (settings.CORPUS_DIR / 'revue_litterature.md').exists(),
        'detecteur': (settings.CORPUS_DIR / 'gaps_detectes.json').exists(),
        'citateur': (settings.CORPUS_DIR / 'rapport_citations.json').exists(),
    }
    return checks


# ═════════════════════════════════════════════════════════════════════════════
#  HOME — Dashboard principal
# ═════════════════════════════════════════════════════════════════════════════

def home(request):
    corpus = load_json('corpus_complet.json')
    if not isinstance(corpus, list):
        corpus = []

    carte = load_json('carte_corpus.json')
    gaps = load_json('gaps_detectes.json')
    citations_rapport = load_json('rapport_citations.json')
    status = get_pipeline_status()

    # Stats
    nb_articles = len(corpus)
    nb_pdfs = count_pdfs()
    nb_themes = len(carte.get('themes', {})) if isinstance(carte, dict) else 0
    nb_gaps = gaps.get('meta', {}).get('nb_gaps_total', 0) if isinstance(gaps, dict) else 0
    score_fiabilite = citations_rapport.get('meta', {}).get('score_fiabilite', 0) if isinstance(citations_rapport, dict) else 0

    # Quality distribution
    qualite_dist = {'élevé': 0, 'moyen': 0, 'faible': 0}
    for doc in corpus:
        niveau = doc.get('qualite', {}).get('niveau', 'faible')
        qualite_dist[niveau] = qualite_dist.get(niveau, 0) + 1

    # Year distribution for chart
    year_dist = {}
    for doc in corpus:
        annee = doc.get('annee')
        if annee and int(annee) > 1990:
            year_dist[int(annee)] = year_dist.get(int(annee), 0) + 1
    year_labels = sorted(year_dist.keys())
    year_values = [year_dist[y] for y in year_labels]

    # Pipeline completion
    completed_steps = sum(1 for v in status.values() if v)
    total_steps = len(status)

    progress_pct = round(completed_steps / total_steps * 100) if total_steps > 0 else 0
    
    # Calculate dashoffset for the circular progress (circumference = 364.4)
    # offset = circumference - (percent / 100) * circumference
    dashoffset = 364.4 - (progress_pct / 100) * 364.4

    context = {
        'nb_articles': nb_articles,
        'nb_pdfs': nb_pdfs,
        'nb_themes': nb_themes,
        'nb_gaps': nb_gaps,
        'score_fiabilite': score_fiabilite,
        'qualite_dist': qualite_dist,
        'year_labels': json.dumps(year_labels),
        'year_values': json.dumps(year_values),
        'status': status,
        'completed_steps': completed_steps,
        'total_steps': total_steps,
        'progress_pct': progress_pct,
        'dashoffset': dashoffset,
    }
    return render(request, 'dashboard/home.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  CORPUS — Liste des articles
# ═════════════════════════════════════════════════════════════════════════════

def corpus(request):
    data = load_json('corpus_complet.json')
    if not isinstance(data, list):
        data = []

    # Enrichir chaque article
    articles = []
    for doc in data:
        articles.append({
            'doc_id': doc.get('doc_id', ''),
            'titre': doc.get('titre', 'Sans titre'),
            'auteurs': doc.get('auteurs', []),
            'annee': doc.get('annee', '?'),
            'domaine': doc.get('domaine', ''),
            'langue': doc.get('langue', ''),
            'methodes': doc.get('methodes', []),
            'mots_cles': doc.get('mots_cles', []),
            'qualite_score': doc.get('qualite', {}).get('score', 0),
            'qualite_niveau': doc.get('qualite', {}).get('niveau', 'faible'),
            'abstract': (doc.get('abstract') or '')[:300],
            'contribution': doc.get('contribution_principale', ''),
            'claim_type': doc.get('claim_type', ''),
            'evidence_level': doc.get('evidence_level', ''),
            'nb_references': len(doc.get('references_extraites', [])),
        })

    # Sort by year descending
    articles.sort(key=lambda x: x.get('annee') or 0, reverse=True)

    context = {
        'articles': articles,
        'nb_articles': len(articles),
    }
    return render(request, 'dashboard/corpus.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  ARTICLE DETAIL
# ═════════════════════════════════════════════════════════════════════════════

def article_detail(request, doc_id):
    data = load_json('corpus_complet.json')
    if not isinstance(data, list):
        data = []

    article = None
    for doc in data:
        if doc.get('doc_id') == doc_id:
            article = doc
            break

    if not article:
        messages.error(request, "Article non trouvé.")
        return redirect('dashboard:corpus')

    context = {'article': article}
    return render(request, 'dashboard/article_detail.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  THEMES — Cartographie thématique
# ═════════════════════════════════════════════════════════════════════════════

def themes(request):
    carte = load_json('carte_corpus.json')
    if not isinstance(carte, dict):
        carte = {}

    themes_data = carte.get('themes', {})
    meta = carte.get('meta', {})
    evolution_globale = carte.get('evolution_globale', {})
    carte_methodes_globale = carte.get('carte_methodes_globale', {})

    # Prepare themes list
    themes_list = []
    for cid, theme in themes_data.items():
        themes_list.append({
            'id': cid,
            'nom': theme.get('nom_theme', f'Thème {cid}'),
            'description': theme.get('description_theme', ''),
            'nb_articles': theme.get('nb_articles', 0),
            'sous_themes': theme.get('sous_themes', []),
            'methodes_dominantes': theme.get('methodes_dominantes', []),
            'mots_cles': theme.get('mots_cles_representatifs', []),
            'niveau_maturite': theme.get('niveau_maturite', '?'),
            'resume_narratif': theme.get('resume_narratif', ''),
            'approche_dominante': theme.get('approche_dominante', ''),
            'tendance': theme.get('evolution_temporelle', {}).get('tendance', '?'),
            'annee_debut': theme.get('evolution_temporelle', {}).get('annee_premiere_publication', '?'),
            'articles': theme.get('articles', []),
            'populations': theme.get('populations_etudiees', []),
            'auteurs_principaux': theme.get('auteurs_principaux', {}).get('auteurs_principaux', []),
        })

    # Scatter plot data
    scatter_data = []
    colors = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4']
    for theme in themes_list:
        color = colors[int(theme['id']) % len(colors)]
        for art in theme.get('articles', []):
            scatter_data.append({
                'x': art.get('coord_x', 0),
                'y': art.get('coord_y', 0),
                'label': (art.get('titre', '') or '')[:40],
                'theme': theme['nom'],
                'color': color,
            })

    context = {
        'themes': themes_list,
        'meta': meta,
        'narrative': meta.get('analyse_narrative_globale', ''),
        'scatter_data': json.dumps(scatter_data),
        'methodes_globales': carte_methodes_globale.get('methodes_frequences', {}),
    }
    return render(request, 'dashboard/themes.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  REVUE — Literature review
# ═════════════════════════════════════════════════════════════════════════════

def revue(request):
    revue_md = load_text('revue_litterature.md')
    revue_json = load_json('revue_litterature.json')

    # Convert markdown to HTML
    revue_html = ''
    if revue_md:
        revue_html = markdown.markdown(revue_md, extensions=['tables', 'fenced_code'])

    nb_mots = revue_json.get('nb_mots_total', len(revue_md.split()) if revue_md else 0)
    nb_citations = revue_json.get('nb_citations', 0)
    themes_ordre = revue_json.get('themes_ordre', [])

    # Load citations or fallback to first 10 articles of corpus
    citations = revue_json.get('citations', [])
    if not citations:
        corpus = load_json('corpus_complet.json')
        if isinstance(corpus, list):
            citations = corpus[:10]

    context = {
        'revue_html': revue_html,
        'revue_md': revue_md,
        'texte_md': revue_md,  # Matches the 'texte_md' variable used in the template
        'nb_mots': nb_mots,
        'nb_citations': nb_citations,
        'themes_ordre': themes_ordre,
        'citations': citations,  # Matches the 'citations' loop used in the template
        'has_revue': bool(revue_md),
    }
    return render(request, 'dashboard/revue.html', context)



# ═════════════════════════════════════════════════════════════════════════════
#  GAPS — Research gaps
# ═════════════════════════════════════════════════════════════════════════════

def gaps(request):
    data = load_json('gaps_detectes.json')
    if not isinstance(data, dict):
        data = {}

    meta = data.get('meta', {}) or {}
    gaps_ref = data.get('gaps_referentiel', {}) or {}
    gaps_par_theme = data.get('gaps_par_theme', []) or []
    analyse_globale = data.get('analyse_globale', {}) or {}
    tous_gaps = data.get('tous_les_gaps_tries', []) or []

    if not isinstance(tous_gaps, list):
        tous_gaps = []

    # Separate by importance
    critiques = []
    moyens = []
    for g in tous_gaps:
        if isinstance(g, dict):
            imp = g.get('importance', '')
            if imp in ('critique', 'haute'):
                critiques.append(g)
            elif imp == 'moyenne':
                moyens.append(g)

    context = {
        'meta': meta,
        'gaps_ref': gaps_ref,
        'gaps_par_theme': gaps_par_theme,
        'analyse_globale': analyse_globale,
        'tous_gaps': tous_gaps,
        'gaps': tous_gaps,  # Ajout pour la compatibilité avec le template
        'critiques': critiques,
        'moyens': moyens,
        'synthese': analyse_globale.get('synthese_lacunes', '') if isinstance(analyse_globale, dict) else '',
        'recommandations': analyse_globale.get('recommandations', []) if isinstance(analyse_globale, dict) else [],
        'has_gaps': bool(tous_gaps),
    }
    return render(request, 'dashboard/gaps.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  CITATIONS — Verification report
# ═════════════════════════════════════════════════════════════════════════════

def citations(request):
    data = load_json('rapport_citations.json')
    if not isinstance(data, dict):
        data = {}

    meta = data.get('meta', {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    stats_claims = data.get('stats_claims', {}) or {}
    if not isinstance(stats_claims, dict):
        stats_claims = {}

    audit = data.get('audit_bibliographie', {}) or {}
    if not isinstance(audit, dict):
        audit = {}

    claims = data.get('claims_classifies', []) or []
    if not isinstance(claims, list):
        claims = []

    # Enrich claims with percentage safely
    clean_claims = []
    for c in claims:
        if isinstance(c, dict):
            rag = c.get('verification_rag', {}) or {}
            if not isinstance(rag, dict):
                rag = {}
            score = rag.get('score_max', 0)
            if score is None:
                score = 0
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            c['confidence_pct'] = round(score * 100)
            if 'statut_final' not in c and 'statut' in c:
                c['statut_final'] = c['statut']
            clean_claims.append(c)
    claims = clean_claims

    # Stats for charts
    hallucinations = []
    valides = []
    incertains = []
    for c in claims:
        statut = c.get('statut_final', '')
        if statut == 'hallucination':
            hallucinations.append(c)
        elif statut in ('valide', 'valide_llm'):
            valides.append(c)
        elif statut == 'incertain':
            incertains.append(c)

    # Calculate dashoffset for the circular progress (circumference = 175.9) safely
    score_val = meta.get('score_fiabilite', 0)
    if score_val is None:
        score_val = 0
    try:
        score_val = float(score_val)
    except (ValueError, TypeError):
        score_val = 0.0
        
    score_dashoffset = f"{175.9 - (score_val / 100.0) * 175.9:.2f}"

    context = {
        'meta': meta,
        'stats_claims': stats_claims,
        'stats': stats_claims,  # Ajout pour le template
        'audit': audit,
        'claims': claims,       # Ajout crucial pour la liste
        'hallucinations': hallucinations[:15],
        'valides_count': len(valides),
        'incertains_count': len(incertains),
        'hallucinations_count': len(hallucinations),
        'total_claims': len(claims),
        'score': score_val,
        'score_dashoffset': score_dashoffset,
        'has_report': bool(claims),
    }
    return render(request, 'dashboard/citations.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE — Run pipeline
# ═════════════════════════════════════════════════════════════════════════════

def pipeline(request):
    status = get_pipeline_status()
    nb_pdfs = count_pdfs()
    context = {
        'status': status,
        'nb_pdfs': nb_pdfs,
    }
    return render(request, 'dashboard/pipeline.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD PDFs
# ═════════════════════════════════════════════════════════════════════════════

def upload_pdfs(request):
    if request.method == 'POST':
        files = request.FILES.getlist('pdfs')
        settings.ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

        uploaded = 0
        for f in files:
            if f.name.lower().endswith('.pdf'):
                dest = settings.ARTICLES_DIR / f.name
                with open(dest, 'wb') as out:
                    for chunk in f.chunks():
                        out.write(chunk)
                uploaded += 1

        messages.success(request, f'{uploaded} PDF(s) uploadé(s) avec succès.')
        return redirect('dashboard:pipeline')

    nb_pdfs = count_pdfs()
    existing = []
    if settings.ARTICLES_DIR.exists():
        existing = [p.name for p in settings.ARTICLES_DIR.glob('*.pdf')]

    context = {
        'nb_pdfs': nb_pdfs,
        'existing': existing,
    }
    return render(request, 'dashboard/upload.html', context)


# ═════════════════════════════════════════════════════════════════════════════
#  API — Run pipeline step
# ═════════════════════════════════════════════════════════════════════════════

@csrf_exempt
def api_run_pipeline(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        body = json.loads(request.body)
        step = body.get('step', 'all')
    except Exception:
        step = 'all'

    main_py = settings.PROJECT_ROOT / 'main.py'

    import sys
    if step == 'all':
        cmd = [sys.executable, str(main_py)]
    else:
        cmd = [sys.executable, str(main_py), '--etape', step]

    def run_bg():
        try:
            subprocess.run(cmd, cwd=str(settings.PROJECT_ROOT), timeout=1800)
        except Exception as e:
            print(f"Pipeline error: {e}")

    thread = threading.Thread(target=run_bg)
    thread.start()

    return JsonResponse({'status': 'started', 'step': step})


# ═════════════════════════════════════════════════════════════════════════════
#  EXPORT — LaTeX Export view
# ═════════════════════════════════════════════════════════════════════════════

def export_latex(request):
    """Generate and export a publication-grade LaTeX .tex document from revue_litterature.md."""
    from django.http import HttpResponse
    import re

    md_content = load_text('revue_litterature.md')
    if not md_content:
        md_content = "# Revue de Littérature\nRevue non disponible."

    # Parse Title
    title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Revue de Littérature"
    
    # Remove the first title line from content so we don't repeat it
    content_lines = md_content.splitlines()
    if content_lines and content_lines[0].startswith('# '):
        content_lines.pop(0)
    # Remove metadata line if exists (starts with *)
    if content_lines and content_lines[0].strip().startswith('*') and content_lines[0].strip().endswith('*'):
        content_lines.pop(0)
    
    clean_md = "\n".join(content_lines)

    # LaTeX Conversion
    latex = []
    
    # Standard Header
    latex.append(r"\documentclass[11pt,a4paper]{article}")
    latex.append(r"\usepackage[utf8]{inputenc}")
    latex.append(r"\usepackage[T1]{fontenc}")
    latex.append(r"\usepackage[french]{babel}")
    latex.append(r"\usepackage{amsmath}")
    latex.append(r"\usepackage{amsfonts}")
    latex.append(r"\usepackage{amssymb}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{hyperref}")
    latex.append(r"\usepackage{geometry}")
    latex.append(r"\geometry{margin=1in}")
    latex.append(r"")
    latex.append(r"\title{" + title + "}")
    latex.append(r"\author{Système d'Agents Multi-Agents (SMA) OS}")
    latex.append(r"\date{\today}")
    latex.append(r"")
    latex.append(r"\begin{document}")
    latex.append(r"\maketitle")
    latex.append(r"")
    
    in_list = False
    in_references = False
    
    lines = clean_md.splitlines()
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if in_list:
                latex.append(r"\end{itemize}")
                in_list = False
            latex.append("")
            continue
        
        # Horizontal rules
        if line_strip == "---":
            if in_list:
                latex.append(r"\end{itemize}")
                in_list = False
            latex.append(r"\noindent\makebox[\linewidth]{\rule{\textwidth}{0.4pt}}")
            continue
            
        # Headers
        if line_strip.startswith("## "):
            if in_list:
                latex.append(r"\end{itemize}")
                in_list = False
            header_text = line_strip[3:].strip()
            if "Références" in header_text:
                in_references = True
                latex.append(r"\section*{" + header_text + "}")
                latex.append(r"\begin{enumerate}")
            else:
                latex.append(r"\section{" + header_text + "}")
            continue
            
        if line_strip.startswith("### "):
            if in_list:
                latex.append(r"\end{itemize}")
                in_list = False
            header_text = line_strip[4:].strip()
            latex.append(r"\subsection{" + header_text + "}")
            continue
            
        # Lists
        if line_strip.startswith("- ") or line_strip.startswith("* ") or (line_strip and line_strip[0].isdigit() and line_strip.split('.', 1)[0].isdigit() and line_strip.startswith(tuple(str(i) for i in range(10)))):
            if not in_list and not in_references:
                latex.append(r"\begin{itemize}")
                in_list = True
            
            item_text = line_strip
            if line_strip.startswith("- "):
                item_text = line_strip[2:]
            elif line_strip.startswith("* "):
                item_text = line_strip[2:]
            elif "." in line_strip[:4]:
                parts = line_strip.split(".", 1)
                item_text = parts[1].strip()
            
            item_text = escape_latex_formatting(item_text)
            if in_references:
                latex.append(r"\item " + item_text)
            else:
                latex.append(r"\item " + item_text)
            continue
            
        # Standard line
        if in_list:
            latex.append(r"\end{itemize}")
            in_list = False
            
        line_escaped = escape_latex_formatting(line_strip)
        
        if in_references:
            if line_escaped.startswith(("[", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                latex.append(r"\item " + line_escaped)
            else:
                latex.append(line_escaped)
        else:
            latex.append(line_escaped)
            
    if in_list:
        latex.append(r"\end{itemize}")
    if in_references:
        latex.append(r"\end{enumerate}")
        
    latex.append(r"\end{document}")
    
    latex_document = "\n".join(latex)
    
    response = HttpResponse(latex_document, content_type='application/x-tex')
    response['Content-Disposition'] = 'attachment; filename="revue_litterature.tex"'
    return response


def escape_latex_formatting(text):
    """Helper to convert Markdown formatting to LaTeX equivalent."""
    import re
    # Escape special characters
    text = text.replace('%', r'\%')
    text = text.replace('&', r'\&')
    text = text.replace('$', r'\$')
    text = text.replace('_', r'\_')
    text = text.replace('#', r'\#')
    
    # Bold **text** or __text__
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'__(.*?)__', r'\\textbf{\1}', text)
    
    # Italics *text* or _text_
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    
    return text
