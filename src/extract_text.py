"""Extraction du texte brut depuis un fichier PDF.

Utilise PyMuPDF (fitz) pour parcourir toutes les pages du PDF
et récupérer le texte en format simple, page par page.
"""
from __future__ import annotations
from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text("text")
            if t:
                text_parts.append(t)

    return "\n".join(text_parts).strip()
