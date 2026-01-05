"""Analyse complète d'un PDF scientifique sur les fractures.

Pipeline principal :
1. Extraction du texte brut du PDF
2. Extraction des sections (abstract, conclusion)
3. Classification par mots-clés médicaux
4. Génération des résumés
5. Sauvegarde en base de données SQLite
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.extract_text import extract_text_from_pdf
from src.sections import extract_abstract_and_conclusion
from src.llm_extract import llm_extract_abstract_conclusion
from src.keywords import classify_by_keywords
from src.summarize import summarize, clean_extracted_text
from src.db import get_conn, insert_paper


def _extract_sections(text: str) -> tuple[str, str]:
    """Extrait abstract et conclusion avec fallback LLM si nécessaire."""
    abstract, conclusion = extract_abstract_and_conclusion(text)
    
    if (not abstract or len(abstract.split()) < 50) or \
       (not conclusion or len(conclusion.split()) < 30):
        abs_llm, conc_llm = llm_extract_abstract_conclusion(text)
        if not abstract or len(abstract.split()) < 50:
            abstract = abs_llm or abstract
        if not conclusion or len(conclusion.split()) < 30:
            conclusion = conc_llm or conclusion
    
    return abstract, conclusion


def _get_text_tail(text: str, max_words: int = 200) -> str:
    """Extrait la fin du texte (utilisée comme fallback)."""
    parts = [p.strip() for p in text.replace("\r", "\n").split("\n\n") if p.strip()]
    words = []
    for p in reversed(parts):
        words.extend(p.split())
        if len(words) >= max_words:
            break
    return " ".join(reversed(words[-max_words:])).strip()


def _generate_conclusion_summary(conclusion_clean: str, conclusion_raw: str, raw_text: str) -> str:
    """Génère un résumé de conclusion avec fallback sur contenu général."""
    summary = summarize(conclusion_clean or conclusion_raw) if (conclusion_clean or conclusion_raw) else ""
    
    if not summary:
        fallback = _get_text_tail(raw_text, 200) or clean_extracted_text(raw_text) or raw_text
        summary = summarize(fallback)
    
    return summary or "Conclusion non disponible."


def main():
    parser = argparse.ArgumentParser(description="Analyse PDF scientifique")
    parser.add_argument("--pdf", required=True, help="Chemin vers le PDF")
    parser.add_argument("--db", default="db/papers.sqlite", help="Chemin vers la BD SQLite")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    raw_text = extract_text_from_pdf(pdf_path)

    # Extraction et nettoyage des sections
    abstract_raw, conclusion_raw = _extract_sections(raw_text)
    abstract_clean = clean_extracted_text(abstract_raw)
    conclusion_clean = clean_extracted_text(conclusion_raw)
    raw_text_clean = clean_extracted_text(raw_text)

    # Classification par mots-clés
    classification = classify_by_keywords(raw_text_clean)

    # Génération des résumés
    abstract_summary = summarize(abstract_clean or abstract_raw) if (abstract_clean or abstract_raw) else ""
    conclusion_summary = _generate_conclusion_summary(conclusion_clean, conclusion_raw, raw_text)

    # Résultat JSON
    result = {
        "pdf": str(pdf_path),
        "region": classification.region,
        "region_score": classification.region_score,
        "fracture_types": classification.fracture_types,
        "locations": classification.locations,
        "abstract_summary": abstract_summary,
        "conclusion_summary": conclusion_summary,
    }

    # Sauvegarde en BD
    conn = get_conn(args.db)
    paper_id = insert_paper(conn, {
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "raw_text": raw_text_clean,
        "abstract_text": abstract_clean,
        "conclusion_text": conclusion_clean,
        "region": classification.region,
        "region_score": classification.region_score,
        "fracture_types": ",".join(classification.fracture_types),
        "locations": ",".join(classification.locations),
        "abstract_summary": abstract_summary,
        "conclusion_summary": conclusion_summary,
    })
    result["db_id"] = paper_id

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
