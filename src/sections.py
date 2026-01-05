"""Extraction des sections Abstract et Conclusion."""
from __future__ import annotations
import re
from typing import Tuple


def _normalize(text: str) -> str:
    """Normalise les sauts de ligne et espaces."""
    text = text.replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)


def _find_section(text: str, start_titles: list[str], stop_titles: list[str]) -> str:
    """Trouve une section entre ses titres de début et fin."""
    text = _normalize(text)
    
    start_pattern = r"(?im)^(?:%s)\s*$" % "|".join(map(re.escape, start_titles))
    match = re.search(start_pattern, text)
    if not match:
        return ""

    start_idx = match.end()
    stop_pattern = r"(?im)^(?:%s)\s*$" % "|".join(map(re.escape, stop_titles))
    stop_match = re.search(stop_pattern, text[start_idx:])
    
    end_idx = start_idx + (stop_match.start() if stop_match else len(text) - start_idx)
    return text[start_idx:end_idx].strip()


def extract_abstract_and_conclusion(text: str) -> Tuple[str, str]:
    """Extrait les sections Abstract et Conclusion du texte."""
    text = _normalize(text)

    abstract_pattern = r"(?im)^\s*abstract\s*:\s*(.+?)(?=^\s*\d+\.\s+|\n\s*keywords\s*:|\n\s*introduction\s*|\Z)"
    conclusion_pattern = r"(?im)^\s*(\d+\.\s*)?conclusions?\s*:\s*(.+?)(?=^\s*author contributions|^\s*funding|^\s*references|\Z)"

    abstract = ""
    conclusion = ""

    m = re.search(abstract_pattern, text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if m:
        abstract = m.group(1).strip()

    m = re.search(conclusion_pattern, text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if m:
        conclusion = m.group(2).strip()

    # Fallback sur les titres de section
    if not abstract or not conclusion:
        abstract_titles = ["Abstract", "ABSTRACT", "Résumé", "RESUME"]
        conclusion_titles = ["Conclusion", "Conclusions", "CONCLUSION", "CONCLUSIONS"]
        stop_titles = [
            "Introduction", "Methods", "Materials and Methods", "Results",
            "Discussion", "Keywords", "References", "Acknowledgements",
            "Supplementary materials", "Author Contribution"
        ]

        if not abstract:
            abstract = _find_section(text, abstract_titles, stop_titles + conclusion_titles)
        if not conclusion:
            conclusion = _find_section(text, conclusion_titles, stop_titles + ["References"])

    return abstract.strip(), conclusion.strip()
