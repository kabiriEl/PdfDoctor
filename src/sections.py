"""Extraction des sections Abstract et Conclusion d'un texte.

Repère les titres de section (en français et anglais) et découpe
le texte pour isoler l'abstract et la conclusion. Utile pour
focaliser l'analyse sur les parties clés du document.
"""
from __future__ import annotations
import re
from typing import Tuple


def _normalize(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _find_section(text: str, start_titles: list[str], stop_titles: list[str]) -> str:
    t = _normalize(text)

    # Construit un pattern "début de section"
    start_pat = r"(?im)^(?:%s)\s*$" % "|".join(map(re.escape, start_titles))
    m = re.search(start_pat, t)
    if not m:
        return ""

    start_idx = m.end()

    # Coupe jusqu'à la prochaine section connue
    stop_pat = r"(?im)^(?:%s)\s*$" % "|".join(map(re.escape, stop_titles))
    m2 = re.search(stop_pat, t[start_idx:])
    end_idx = start_idx + (m2.start() if m2 else len(t) - start_idx)

    return t[start_idx:end_idx].strip()


def extract_abstract_and_conclusion(text: str) -> Tuple[str, str]:
    """
    Tente d'abord des patterns directs (abstract:/conclusion:) puis fallback sur les titres classiques.
    """
    t = _normalize(text)

    abstract_pat = r"(?im)^\s*abstract\s*:\s*(.+?)(?=^\s*\d+\.\s+|\n\s*keywords\s*:|\n\s*introduction\s*|\Z)"
    conclusion_pat = r"(?im)^\s*(\d+\.\s*)?conclusions?\s*:\s*(.+?)(?=^\s*author contributions|^\s*funding|^\s*references|\Z)"

    abstract = ""
    conclusion = ""

    m_abs = re.search(abstract_pat, t, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if m_abs:
        abstract = m_abs.group(1).strip()

    m_conc = re.search(conclusion_pat, t, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if m_conc:
        conclusion = m_conc.group(2).strip()

    # Fallback sur les titres si non trouvé
    if not abstract or not conclusion:
        abstract_titles = ["Abstract", "ABSTRACT", "Résumé", "RESUME"]
        conclusion_titles = ["Conclusion", "Conclusions", "CONCLUSION", "CONCLUSIONS", "Concluding remarks"]

        common_stops = [
            "Introduction", "INTRODUCTION",
            "Methods", "METHODS", "Materials and Methods", "MATERIALS AND METHODS",
            "Results", "RESULTS",
            "Discussion", "DISCUSSION", "Keywords", "KEYWORDS",
            "References", "REFERENCES", "Acknowledgements", "ACKNOWLEDGEMENTS", "Supplementary materials", "SUPPLEMENTARY MATERIALS",
        ]

        if not abstract:
            abstract = _find_section(text, abstract_titles, stop_titles=common_stops + conclusion_titles)
        if not conclusion:
            conclusion = _find_section(text, conclusion_titles, stop_titles=common_stops + ["References", "REFERENCES"])

    return abstract.strip(), conclusion.strip()
