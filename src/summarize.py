"""
Résumé intelligent du texte avec DistilBART.

Charge un modèle local de résumé (cache + LRU), nettoie les phrases
non cliniques (stats, software, etc.), puis crée un résumé concentré.
Pour les textes courts (< 60 mots), retourne le texte original.

Corrections appliquées (clean++):
- Nettoyage renforcé des artefacts bibliographiques (CrossRef, DOI, Open Access, etc.)
- Troncature sûre avant résumé pour éviter les warnings de longueur (tokens > max)
- Filtre de sécurité: si le résumé généré est trop court / non pertinent -> retourne "" (ou texte court)
- Suppression du warning "clean_up_tokenization_spaces" via paramètre explicite
"""
from __future__ import annotations
from functools import lru_cache
import re
from transformers import pipeline


@lru_cache(maxsize=1)
def _summarizer():
    """
    Modèle de résumé local (téléchargé une fois et en cache).
    """
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        clean_up_tokenization_spaces=True,  # évite FutureWarning
    )


def _clean_for_summary(text: str) -> str:
    """
    Supprime les phrases non cliniques ou méthodologiques
    (logiciels, statistiques, outils, bibliographie, etc.).
    """
    blacklist = [
        # Méta / outils
        r"endnote",
        r"software",
        r"statistical analysis",
        r"ibm",
        r"spss",
        r"database",
        r"search strategy",
        r"screened",
        r"eligibility criteria",
        r"data extraction",

        # Stats fréquentes
        r"p\s*<\s*0\.\d+",
        r"p\s*=\s*0\.\d+",

        # Artefacts bibliographiques / édition
        r"crossref",
        r"doi\s*:",
        r"https?://doi\.org/",
        r"open access",
        r"creativecommons",
        r"license",
        r"references",
        r"publisher",
        r"issn",
        r"pmcid",
        r"pmid",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_low = s_clean.lower()
        if any(re.search(b, s_low) for b in blacklist):
            continue
        kept.append(s_clean)

    return " ".join(kept).strip()


def _truncate_for_model(text: str, tokenizer) -> str:
    """
    Tronque le texte pour ne pas dépasser la longueur maximale du modèle.
    DistilBART supporte généralement 1024 tokens max (selon tokenizer.model_max_length).
    On garde une marge de sécurité pour éviter warnings/erreurs.
    """
    if not text:
        return ""

    max_len = tokenizer.model_max_length or 1024

    # marge de sécurité (réserve) pour éviter les warnings/erreurs
    reserve = 64
    target_len = max(128, min(max_len - reserve, max_len))

    # Important: truncation=True garantit qu'on ne dépasse pas
    encoded = tokenizer.encode(text, truncation=True, max_length=target_len)
    return tokenizer.decode(encoded, skip_special_tokens=True)


def clean_extracted_text(text: str) -> str:
    """
    Nettoie le texte brut extrait des sections Abstract et Conclusion.

    - Normalise les espaces et sauts de ligne
    - Supprime les URLs / DOI
    - Supprime des lignes typiques d'édition / revue
    - Supprime citations [1], [2], etc.
    - Réduit le bruit de ponctuation / symboles
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Normalise les fins de ligne et les sauts multiples
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Supprime les URLs
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"www\.[^\s]+", "", text)

    # Supprime DOI (plusieurs formes)
    text = re.sub(r"https?://doi\.org/[^\s]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoi:\s*[^\s]+", "", text, flags=re.IGNORECASE)

    # Supprime quelques marqueurs très fréquents en bas de page / édition
    footer_noise = [
        r"(?i)crossref",
        r"(?i)open\s+access",
        r"(?i)creative\s+commons",
        r"(?i)license",
        r"(?i)pmcid",
        r"(?i)pmid",
        r"(?i)issn",
    ]
    lines = []
    for line in text.split("\n"):
        if any(re.search(pat, line) for pat in footer_noise):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Supprime les lignes de métadonnées de revue (ex: EFORT Open Reviews (2025) 10 316–326)
    journal_patterns = [
        r"(?i)efort\s+open\s+reviews",
        r"(?i)trauma\s+efort",
        r"(?i)open\s+reviews",
    ]
    page_year_pattern = r"\(\d{4}\).*\d"
    cleaned_lines = []
    for line in text.split("\n"):
        if any(re.search(pat, line) for pat in journal_patterns):
            continue
        if re.search(page_year_pattern, line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Supprime citations numérotées en brackets [1], [2], etc.
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # Nettoie espaces multiples et tabulations
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # Supprime espaces superflus autour de la ponctuation
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)

    # Assure un espacement correct après la ponctuation
    text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

    # Supprime symboles orphelins au début des lignes
    text = re.sub(r"^[\W\d_]+\s*", "", text, flags=re.MULTILINE)

    # Nettoie les lignes vides
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = "\n\n".join(block for block in text.split("\n\n") if block.strip())

    return text.strip()


def summarize(text: str, max_length: int = 200, min_length: int = 150) -> str:
    """
    Génère un résumé propre et cliniquement pertinent.

    Stratégie:
    1) Nettoyage (sections, liens, citations)
    2) Nettoyage "clinique" (supprimer phrases méta/biblio)
    3) Troncature sûre (évite tokens > max)
    4) Résumé DistilBART
    5) Garde-fou: si sortie trop courte / non pertinente -> "" (évite "CrossRef]")
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Nettoyage fort en amont (utile si tu passes raw text)
    text = clean_extracted_text(text)

    # Nettoyage clinique/méthodologique + bibliographie
    text = _clean_for_summary(text)

    # Si texte trop court, inutile de résumer (et évite hallucinations)
    if len(text.split()) < 60:
        return text

    summarizer = _summarizer()

    # Troncature sûre pour éviter warnings / erreurs de longueur
    text = _truncate_for_model(text, summarizer.tokenizer)

    out = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,  # sécurité supplémentaire côté pipeline
    )
    summary = (out[0].get("summary_text") or "").strip()

    # Garde-fou: éviter résumés absurdes type "CrossRef]"
    # - trop court
    # - ou contient uniquement des tokens de bibliographie
    if len(summary.split()) < 5:
        return ""

    if re.search(r"(?i)\bcrossref\b|\bdoi\b|creativecommons|open\s+access", summary):
        # Si le résumé n'est que du bruit bibliographique, on le rejette
        # (tu peux aussi choisir de renvoyer un texte court alternatif)
        return ""

    return summary

















# """Résumé intelligent du texte avec DistilBART.

# Charge un modèle local de résumé (cache + LRU), nettoie les phrases
# non cliniques (stats, software, etc.), puis crée un résumé concentré.
# Pour les textes courts (< 60 mots), retourne le texte original.
# """
# from __future__ import annotations
# from functools import lru_cache
# import re
# from transformers import pipeline


# @lru_cache(maxsize=1)
# def _summarizer():
#     """
#     Modèle de résumé local (téléchargé une fois et en cache).
#     """
#     return pipeline(
#         "summarization",
#         model="sshleifer/distilbart-cnn-12-6"
#     )


# def _clean_for_summary(text: str) -> str:
#     """
#     Supprime les phrases non cliniques ou méthodologiques
#     (logiciels, statistiques, outils, etc.).
#     """
#     blacklist = [
#         r"endnote",
#         r"software",
#         r"statistical analysis",
#         r"p\s*<\s*0\.\d+",
#         r"ibm",
#         r"spss",
#     ]

#     sentences = re.split(r"(?<=[.!?])\s+", text)
#     kept = []

#     for s in sentences:
#         s_low = s.lower()
#         if any(re.search(b, s_low) for b in blacklist):
#             continue
#         kept.append(s)

#     return " ".join(kept).strip()


# def _truncate_for_model(text: str, tokenizer) -> str:
#     """
#     Tronque le texte pour ne pas dépasser la longueur maximale du modèle (BART ~1024 tokens).
#     """
#     if not text:
#         return ""
#     max_len = tokenizer.model_max_length or 1024
#     # On garde une marge de sécurité de 16 tokens
#     target_len = max(64, min(max_len - 16, 1024))
#     encoded = tokenizer.encode(text, truncation=True, max_length=target_len)
#     return tokenizer.decode(encoded, skip_special_tokens=True)


# def clean_extracted_text(text: str) -> str:
#     """
#     Nettoie le texte brut extrait des sections Abstract et Conclusion.
    
#     - Normalise les espaces et sauts de ligne
#     - Supprime les espaces supplémentaires et tabulations
#     - Supprime les URLs et références en brackets
#     - Élimine les nombres orphelins et symboles
#     - Assure un espacement correct entre les phrases
#     """
#     text = (text or "").strip()
#     if not text:
#         return ""
    
#     # Normalise les fins de ligne et les sauts multiples
#     text = text.replace("\r\n", "\n")
#     text = text.replace("\r", "\n")
#     text = re.sub(r"\n{3,}", "\n\n", text)
    
#     # Supprime les URLs
#     text = re.sub(r"https?://[^\s]+", "", text)
#     text = re.sub(r"www\.[^\s]+", "", text)
    
#     # Supprime les références DOI
#     text = re.sub(r"https://doi\.org/[^\s]+", "", text)
#     text = re.sub(r"doi:\s*[^\s]+", "", text)

#     # Supprime les lignes de métadonnées de revue (ex: EFORT Open Reviews (2025) 10 316–326)
#     journal_patterns = [
#         r"(?i)efort\s+open\s+reviews",
#         r"(?i)open\s+reviews",
#         r"(?i)trauma\s+efort",
#     ]
#     page_year_pattern = r"\(\d{4}\).*\d"
#     cleaned_lines = []
#     for line in text.split("\n"):
#         if any(re.search(pat, line) for pat in journal_patterns):
#             cleaned_lines.append("")
#             continue
#         if re.search(page_year_pattern, line):
#             # lignes typiques de citations avec année + pagination
#             cleaned_lines.append("")
#             continue
#         cleaned_lines.append(line)
#     text = "\n".join(l for l in cleaned_lines if l.strip())
    
#     # Supprime les citations numérotées en brackets [1], [2], etc.
#     text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    
#     # Nettoie les espaces multiples et tabulations
#     text = re.sub(r"\t+", " ", text)
#     text = re.sub(r" {2,}", " ", text)
    
#     # Supprime les espaces superflus autour de la ponctuation
#     text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    
#     # Assure un espacement correct après la ponctuation
#     text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)
    
#     # Supprime les symboles orphelins au début des lignes
#     text = re.sub(r"^[\W\d_]+\s*", "", text, flags=re.MULTILINE)
    
#     # Nettoie les espaces en début/fin de ligne
#     text = "\n".join(line.strip() for line in text.split("\n"))
#     text = "\n\n".join(line for line in text.split("\n\n") if line.strip())
    
#     return text.strip()


# def summarize(text: str, max_length: int = 200, min_length: int = 150) -> str:
#     """
#     Génère un résumé propre et cliniquement pertinent.
#     """
#     text = (text or "").strip()
#     if not text:
#         return ""

#     text = _clean_for_summary(text)

#     if len(text.split()) < 60:
#         return text

#     summarizer = _summarizer()
    
#     text = _truncate_for_model(text, summarizer.tokenizer)
#     out = summarizer(
#         text,
#         max_length=max_length,
#         min_length=min_length,
#         do_sample=False
#     )
#     return out[0]["summary_text"].strip()
