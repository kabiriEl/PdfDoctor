"""Classification par mots-clés médicaux.

Définit des dictionnaires de termes cliniques pour identifier
la région de fracture (tibial plateau, pelvis, etc.), le type
(intra-articulaire, comminutée, etc.) et la localisation
(proximal, distal, gauche, droite...). Compte les occurrences
et retourne une classification structurée.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple


def _count_hits(text: str, keywords: List[str]) -> int:
    """
    Compte les occurrences de mots-clés (insensible à la casse).
    """
    t = text.lower()
    hits = 0
    for kw in keywords:
        pattern = re.escape(kw.lower())
        hits += len(re.findall(pattern, t))
    return hits


# =========================
# DÉFINITIONS DES MOTS-CLÉS
# =========================

REGION_KEYWORDS: Dict[str, List[str]] = {
    "tibial_plateau": [
        "tibial plateau",
        "schatzker",
        "proximal tibia",
        "lateral tibial plateau",
        "medial tibial plateau",
    ],
    "pelvis": [
        "pelvic fracture",
        "pelvis",
        "pelvic ring",
        "acetabulum",
        "acetabular",
        "pubic ramus",
        "iliac",
    ],
    "distal_radius": [
        "distal radius",
        "radius fracture",
        "colles",
        "smith fracture",
        "wrist fracture",
    ],
}

TYPE_KEYWORDS: Dict[str, List[str]] = {
    "intra_articular": ["intra-articular", "intra articular"],
    "extra_articular": ["extra-articular", "extra articular"],
    "comminuted": ["comminuted"],
    "displaced": ["displaced", "displacement"],
    "open": ["open fracture"],
    "closed": ["closed fracture"],
}

LOCATION_KEYWORDS: Dict[str, List[str]] = {
    "proximal": ["proximal"],
    "distal": ["distal"],
    "left": [" left ", " lt ", " left-sided"],
    "right": [" right ", " rt ", " right-sided"],
    "medial": ["medial"],
    "lateral": ["lateral"],
    "posterior": ["posterior"],
    "anterior": ["anterior"],
}


# =========================
# DATA STRUCTURE
# =========================

@dataclass
class Classification:
    region: str
    region_score: int
    fracture_types: List[str]
    locations: List[str]


# =========================
# LOCATION POST-PROCESSING
# =========================

def _top_locations(text: str, max_items: int = 3) -> List[str]:
    """
    Sélectionne les localisations anatomiques les plus pertinentes
    tout en évitant les incohérences (gauche+droite, proximal+distal).
    """
    counts = {}
    for loc, kws in LOCATION_KEYWORDS.items():
        c = _count_hits(text, kws)
        if c > 0:
            counts[loc] = c

    if not counts:
        return []

    # Trie par fréquence
    ordered = [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]

    side = [x for x in ordered if x in ("left", "right")]
    zone = [x for x in ordered if x in ("proximal", "distal")]
    others = [x for x in ordered if x not in side + zone]

    final = []
    if side:
        final.append(side[0])      # garder un seul côté
    if zone:
        final.append(zone[0])      # garder une seule zone
    final.extend(others)

    return final[:max_items]


# =========================
# CLASSIFICATION PRINCIPALE
# =========================

def classify_by_keywords(text: str) -> Classification:
    """
    Classification basée sur des règles utilisant des mots-clés médicaux.
    """
    # --- RÉGION ---
    scores: List[Tuple[str, int]] = []
    for region, kws in REGION_KEYWORDS.items():
        scores.append((region, _count_hits(text, kws)))
    scores.sort(key=lambda x: x[1], reverse=True)

    best_region, best_score = scores[0]
    if best_score == 0:
        best_region = "inconnu"

    # --- TYPE (multi-label) ---
    fracture_types = [
        k for k, kws in TYPE_KEYWORDS.items()
        if _count_hits(text, kws) > 0
    ]

    # --- LOCALISATION (post-traitée) ---
    locations = _top_locations(text)

    return Classification(
        region=best_region,
        region_score=best_score,
        fracture_types=sorted(set(fracture_types)),
        locations=locations,
    )
