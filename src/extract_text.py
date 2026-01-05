"""Extraction du texte brut d'un fichier PDF."""
from __future__ import annotations
from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extrait le texte de toutes les pages d'un PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF non trouvé : {pdf_path}")

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

    return "\n".join(pages).strip()
