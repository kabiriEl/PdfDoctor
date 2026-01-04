"""Script principal pour analyser un PDF scientifique sur les fractures.

Lit un PDF, extrait le texte, récupère l'abstract et la conclusion,
classifie par mots-clés médicaux (région, type, localisation),
résume les sections pertinentes, puis enregistre tout dans SQLite.
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


def _fallback_tail(text: str, max_words: int = 200) -> str:
    """Prend la fin du texte brut pour fabriquer une pseudo-conclusion si rien n'est extrait."""
    t = (text or "").replace("\r", "\n")
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]
    tail = list(reversed(parts))
    words: list[str] = []
    for p in tail:
        words.extend(p.split())
        if len(words) >= max_words:
            break
    words = list(reversed(words))  # on reprend dans l'ordre naturel pour la fin
    return " ".join(words[-max_words:]).strip()


def main():
    ap = argparse.ArgumentParser(description="Pipeline PDF fracture (mots-clés + résumé local)")
    ap.add_argument("--pdf", required=True, help="Chemin vers un PDF scientifique")
    ap.add_argument("--db", default="db/papers.sqlite", help="Chemin SQLite")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    raw_text = extract_text_from_pdf(pdf_path)

    # ÉTAPE 1 : Extraction basée sur règles regex (rapide)
    abstract_text, conclusion_text = extract_abstract_and_conclusion(raw_text)
    
    # ÉTAPE 2 : Fallback LLM si extraction insuffisante
    # Seuils : abstract < 50 mots OU conclusion < 30 mots → fallback
    MIN_ABSTRACT_WORDS = 50
    MIN_CONCLUSION_WORDS = 30
    
    abstract_ok = abstract_text and len(abstract_text.split()) >= MIN_ABSTRACT_WORDS
    conclusion_ok = conclusion_text and len(conclusion_text.split()) >= MIN_CONCLUSION_WORDS
    
    if not abstract_ok or not conclusion_ok:
        print(f"⚠️  Extraction regex insuffisante (abstract: {len(abstract_text.split()) if abstract_text else 0} mots, conclusion: {len(conclusion_text.split()) if conclusion_text else 0} mots)")
        print("🔄 Activation du fallback LLM (FLAN-T5)...")
        
        abs_llm, conc_llm = llm_extract_abstract_conclusion(raw_text)
        
        if not abstract_ok and abs_llm:
            abstract_text = abs_llm
            print(f"✓ Abstract extrait par LLM ({len(abs_llm.split())} mots)")
        
        if not conclusion_ok and conc_llm:
            conclusion_text = conc_llm
            print(f"✓ Conclusion extraite par LLM ({len(conc_llm.split())} mots)")
    
    # ÉTAPE 3 : Nettoyage des sections extraites
    abstract_text_raw = abstract_text
    conclusion_text_raw = conclusion_text

    abstract_text = clean_extracted_text(abstract_text)
    conclusion_text = clean_extracted_text(conclusion_text)
    raw_text_clean = clean_extracted_text(raw_text)
    
    # ÉTAPE 4 : Classification par mots-clés médicaux
    cls = classify_by_keywords(raw_text_clean)

    # ÉTAPE 5 : Génération des résumés
    # Si le nettoyage vide une section mais que la version brute existe, on résume la version brute
    abstract_source = abstract_text if abstract_text else abstract_text_raw
    fallback_conclusion = _fallback_tail(raw_text, max_words=200)
    conclusion_source = (
        conclusion_text
        or conclusion_text_raw
        or fallback_conclusion
        or raw_text_clean
        or raw_text
    )

    abstract_summary = summarize(abstract_source) if abstract_source else ""
    conclusion_summary = summarize(conclusion_source) if conclusion_source else ""
    if not conclusion_summary and conclusion_source:
        conclusion_summary = conclusion_source
    
    # ÉTAPE 6 : Construction du résultat JSON
    result = {
        "pdf": str(pdf_path),
        "region": cls.region,
        "region_score": cls.region_score,
        "fracture_types": cls.fracture_types,
        "locations": cls.locations,
        "abstract_summary": abstract_summary,
        "conclusion_summary": conclusion_summary,
    }

    # ÉTAPE 7 : Insertion en base de données
    conn = get_conn(args.db)
    paper_id = insert_paper(conn, {
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "raw_text": raw_text_clean,
        "abstract_text": abstract_text,
        "conclusion_text": conclusion_text,
        "region": cls.region,
        "region_score": cls.region_score,
        "fracture_types": ",".join(cls.fracture_types),
        "locations": ",".join(cls.locations),
        "abstract_summary": abstract_summary,
        "conclusion_summary": conclusion_summary,
    })
    result["db_id"] = paper_id

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
