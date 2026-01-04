from __future__ import annotations
import re
from typing import Tuple, Optional

from transformers import pipeline


def _normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(
    text: str,
    tokenizer,
    reserve_tokens: int = 200,
    min_chunk_tokens: int = 128,
) -> list[str]:
    """
    Token-level chunking to respect the model's max input length.
    We keep an early chunk (for Abstract) and a late chunk (for Conclusion).
    """
    text = _normalize(text)

    max_model_tokens = tokenizer.model_max_length or 512
    chunk_tokens = max(min_chunk_tokens, max_model_tokens - reserve_tokens)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_tokens:
        return [text]

    head_ids = token_ids[:chunk_tokens]
    tail_ids = token_ids[-chunk_tokens:]
    head = tokenizer.decode(head_ids, skip_special_tokens=True)
    tail = tokenizer.decode(tail_ids, skip_special_tokens=True)

    return [head, tail]


def _clean_generated(s: str) -> str:
    s = s.strip()
    s = re.sub(r"(?i)^abstract\s*:\s*", "", s).strip()
    s = re.sub(r"(?i)^conclusion(s)?\s*:\s*", "", s).strip()
    return s.strip()


def _fallback_first_paragraphs(text: str, max_words: int = 180) -> str:
    """
    Last-resort fallback if no Abstract title exists:
    take the first non-empty paragraphs.
    """
    t = _normalize(text)
    paras = [p.strip() for p in t.split("\n\n") if len(p.strip()) > 40]
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
    """
    LLM fallback: extract Abstract + Conclusion from raw text.
    Uses local transformers pipeline (no API).
    """
    raw_text = _normalize(raw_text)

    prompt_abs = (
        "Extract ONLY the Abstract text from the following scientific article text. "
        "If no abstract exists, output EMPTY.\n\nTEXT:\n"
    )
    prompt_conc = (
        "Extract ONLY the Conclusion(s) text from the following scientific article text. "
        "If no conclusion exists, output EMPTY.\n\nTEXT:\n"
    )

    gen = pipeline("text2text-generation", model=model_name)
    tokenizer = gen.tokenizer

    prompt_tokens = max(
        len(tokenizer.encode(prompt_abs, add_special_tokens=False)),
        len(tokenizer.encode(prompt_conc, add_special_tokens=False)),
    )
    chunks = _chunk_text(raw_text, tokenizer, reserve_tokens=prompt_tokens + 32)

    # Try extract abstract from the first chunk
    abstract = ""
    out_abs = gen(prompt_abs + chunks[0], max_new_tokens=220, do_sample=False)[0]["generated_text"]
    abstract = _clean_generated(out_abs)
    if abstract.upper() == "EMPTY":
        abstract = ""

    # Try extract conclusion from the last chunk
    conclusion = ""
    out_conc = gen(prompt_conc + chunks[-1], max_new_tokens=220, do_sample=False)[0]["generated_text"]
    conclusion = _clean_generated(out_conc)
    if conclusion.upper() == "EMPTY":
        conclusion = ""

    # Ultimate fallback: if abstract missing, take first paragraphs
    if not abstract:
        abstract = _fallback_first_paragraphs(raw_text)

    return abstract.strip(), conclusion.strip()
