"""Résumé intelligent du texte avec DistilBART.

Nettoie le texte et génère des résumés via DistilBART,
avec support pour textes longs via division en chunks.
"""
from __future__ import annotations
from functools import lru_cache
import re
from transformers import pipeline


@lru_cache(maxsize=1)
def _summarizer():
    """Charge le modèle DistilBART en cache."""
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        clean_up_tokenization_spaces=True,
    )


def _is_reference_section(text: str) -> bool:
    """Détecte si le texte est une section de références."""
    if not text:
        return False

    lines = text.strip().split("\n")
    
    ref_patterns = [
        r"\[\d+\]", r"\d{1,2}\.", r"\[PubMed\]",
        r"\[CrossRef\]", r"doi:", r"https://doi\.org/",
    ]

    ref_count = sum(
        1 for line in lines[:min(20, len(lines))]
        if any(re.search(p, line, re.IGNORECASE) for p in ref_patterns)
    )

    if len(lines) > 10 and ref_count > len(lines) * 0.6:
        return True

    author_count = len(re.findall(r"[A-Z]\.;\s*[A-Z]\.;", text))
    return author_count > 5 and len(text.split()) > 100


def clean_extracted_text(text: str) -> str:
    """Nettoie le texte extrait (supprime références, URLs, métadonnées)."""
    text = (text or "").strip()
    if not text or _is_reference_section(text):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Supprime URLs et DOI
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"www\.[^\s]+", "", text)
    text = re.sub(r"\bdoi:\s*[^\s]+", "", text, flags=re.IGNORECASE)

    # Supprime marqueurs de bruit
    footer_noise = [
        r"(?i)crossref", r"(?i)open\s+access", r"(?i)license",
        r"(?i)pmcid", r"(?i)pmid", r"(?i)issn",
    ]
    text = "\n".join(
        line for line in text.split("\n")
        if not any(re.search(pat, line) for pat in footer_noise)
    )

    # Supprime références et métadonnées
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[(?:PubMed|CrossRef|PMC|Medline)\]", "", text, flags=re.IGNORECASE)
    
    ref_pattern = r"^[A-Z]\.[A-Z]?;.*(?:\d{4}|doi|Surg|Rev|J\.|Lancet)"
    text = "\n".join(
        line for line in text.split("\n")
        if not re.match(ref_pattern, line, flags=re.IGNORECASE)
    )

    # Nettoie espaces
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)
    text = re.sub(r"^[\W\d_]+\s*", "", text, flags=re.MULTILINE)

    text = "\n\n".join(
        block for block in (line.strip() for line in text.split("\n"))
        if block
    )
    return text.strip()


def _clean_for_summary(text: str) -> str:
    """Supprime les phrases non cliniques."""
    if _is_reference_section(text):
        return ""

    blacklist = [
        r"endnote", r"software", r"statistical analysis", r"ibm", r"spss",
        r"database", r"search strategy", r"screened", r"p\s*[<|=]\s*0\.\d+",
        r"crossref", r"doi\s*:", r"https?://doi\.org/", r"open access",
        r"references", r"pmcid", r"pmid", r"issn",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        s.strip() for s in sentences
        if s.strip() and not any(re.search(b, s.lower()) for b in blacklist)
    ]
    return " ".join(kept).strip()


def _split_into_chunks(text: str, tokenizer, max_tokens: int = 512) -> list[str]:
    """Divise le texte en chunks respectant la limite de tokens."""
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _summarize_chunk(text: str, summarizer, max_length: int = 150, min_length: int = 100) -> str:
    """Résume un chunk individuel."""
    if not text or len(text.split()) < 20 or _is_reference_section(text):
        return ""

    try:
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        summary = (out[0].get("summary_text") or "").strip()

        if len(summary.split()) < 5:
            return ""
        if re.search(r"(?i)\bcrossref\b|\bdoi\b|open\s+access|\[pubmed\]|\[crossref\]", summary):
            return ""
        if _is_reference_section(summary):
            return ""
        if len(re.findall(r"\d{4}", summary)) > 3:
            return ""

        return summary
    except Exception:
        return ""


def summarize(text: str, max_length: int = 200, min_length: int = 150) -> str:
    """Génère un résumé propre et pertinent."""
    text = (text or "").strip()
    if not text:
        return ""

    text = clean_extracted_text(text)
    text = _clean_for_summary(text)

    if len(text.split()) < 60:
        return text

    summarizer = _summarizer()
    max_tokens = max(300, (summarizer.tokenizer.model_max_length or 1024) - 64)

    chunks = _split_into_chunks(text, summarizer.tokenizer, max_tokens=max_tokens)

    if len(chunks) == 1:
        summary = _summarize_chunk(chunks[0], summarizer, max_length, min_length)
        return summary if summary else text

    chunk_summaries = [
        s for chunk in chunks
        if (s := _summarize_chunk(chunk, summarizer, max_length=120, min_length=80))
    ]

    if not chunk_summaries:
        return text

    combined = " ".join(chunk_summaries)
    if len(combined.split()) > max_length:
        final = _summarize_chunk(combined, summarizer, max_length, min_length)
        return final if final else combined

    return combined













