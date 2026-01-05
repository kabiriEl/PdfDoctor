"""Extraction par LLM (fallback si extraction regex échoue)."""
from __future__ import annotations
import re
from typing import Tuple
from transformers import pipeline


def _normalize(text: str) -> str:
    """Normalise le texte."""
    text = text.replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_output(text: str) -> str:
    """Nettoie la sortie du modèle LLM."""
    text = text.strip()
    text = re.sub(r"(?i)^abstract\s*:\s*", "", text).strip()
    text = re.sub(r"(?i)^conclusion(s)?\s*:\s*", "", text).strip()
    return text.strip()


def _chunk_text(text: str, tokenizer, max_tokens: int = 512) -> list[str]:
    """Divise le texte en chunks selon la limite du modèle."""
    text = _normalize(text)
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


def _fallback_first_paragraphs(text: str, max_words: int = 180) -> str:
    """Extrait les premiers paragraphes comme dernier recours."""
    paras = [p.strip() for p in _normalize(text).split("\n\n") if len(p.strip()) > 40]
    if not paras:
        return ""

    words = []
    for p in paras[:3]:
        words.extend(p.split())
        if len(words) >= max_words:
            break
    return " ".join(words[:max_words]).strip()


def llm_extract_abstract_conclusion(
    raw_text: str,
    model_name: str = "google/flan-t5-base",
) -> Tuple[str, str]:
    """Extrait abstract et conclusion avec un modèle LLM."""
    raw_text = _normalize(raw_text)

    gen = pipeline("text2text-generation", model=model_name)
    tokenizer = gen.tokenizer

    # Calcule les tokens disponibles pour le texte
    prompt_abs = "Extract ONLY the Abstract text from the scientific article. If no abstract exists, output EMPTY.\n\nTEXT:\n"
    prompt_conc = "Extract ONLY the Conclusion(s) text from the scientific article. If no conclusion exists, output EMPTY.\n\nTEXT:\n"

    prompt_tokens = max(
        len(tokenizer.encode(prompt_abs, add_special_tokens=False)),
        len(tokenizer.encode(prompt_conc, add_special_tokens=False)),
    )
    max_tokens = max(256, (tokenizer.model_max_length or 512) - prompt_tokens - 32)

    chunks = _chunk_text(raw_text, tokenizer, max_tokens=max_tokens)

    # Extrait l'abstract du premier chunk
    abstract = ""
    try:
        out = gen(prompt_abs + chunks[0], max_new_tokens=220, do_sample=False)[0]["generated_text"]
        abstract = _clean_output(out)
        if abstract.upper() == "EMPTY":
            abstract = ""
    except Exception:
        pass

    # Extrait la conclusion du dernier chunk
    conclusion = ""
    try:
        out = gen(prompt_conc + chunks[-1], max_new_tokens=220, do_sample=False)[0]["generated_text"]
        conclusion = _clean_output(out)
        if conclusion.upper() == "EMPTY":
            conclusion = ""
    except Exception:
        pass

    # Fallback : si abstract absent, prend les premiers paragraphes
    if not abstract:
        abstract = _fallback_first_paragraphs(raw_text)

    return abstract.strip(), conclusion.strip()
