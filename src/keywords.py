"""Classification par mots-clés médicaux."""
from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass
class Classification:
    """Résultat de la classification."""
    region: str
    region_score: int
    fracture_types: list[str]
    locations: list[str]


REGION_KEYWORDS = {
    "tibial_plateau": [
        "tibial plateau", "schatzker", "proximal tibia",
        "lateral tibial plateau", "medial tibial plateau"
    ],
    "pelvis": [
        "pelvic fracture", "pelvis", "pelvic ring",
        "acetabulum", "acetabular", "pubic ramus", "iliac"
    ],
    "distal_radius": [
        "distal radius", "radius fracture",
        "colles", "smith fracture", "wrist fracture"
    ],
}

TYPE_KEYWORDS = {
    "intra_articular": ["intra-articular", "intra articular"],
    "extra_articular": ["extra-articular", "extra articular"],
    "comminuted": ["comminuted"],
    "displaced": ["displaced", "displacement"],
    "open": ["open fracture"],
    "closed": ["closed fracture"],
}

LOCATION_KEYWORDS = {
    "proximal": ["proximal"],
    "distal": ["distal"],
    "left": [" left ", " lt ", " left-sided"],
    "right": [" right ", " rt ", " right-sided"],
    "medial": ["medial"],
    "lateral": ["lateral"],
    "posterior": ["posterior"],
    "anterior": ["anterior"],
}


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Compte les occurrences de mots-clés."""
    text = text.lower()
    count = 0
    for kw in keywords:
        count += len(re.findall(re.escape(kw.lower()), text))
    return count


def _select_best_locations(text: str, max_count: int = 3) -> list[str]:
    """Sélectionne les meilleures localisations (évite incohérences)."""
    scores = {
        loc: _count_keyword_hits(text, kws)
        for loc, kws in LOCATION_KEYWORDS.items()
        if _count_keyword_hits(text, kws) > 0
    }
    
    if not scores:
        return []

    sorted_locs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    sides = [l for l, _ in sorted_locs if l in ("left", "right")]
    zones = [l for l, _ in sorted_locs if l in ("proximal", "distal")]
    others = [l for l, _ in sorted_locs if l not in sides + zones]
    
    result = []
    if sides:
        result.append(sides[0])
    if zones:
        result.append(zones[0])
    result.extend(others)
    
    return result[:max_count]


def classify_by_keywords(text: str) -> Classification:
    """Classifie le texte par région, type et localisation."""
    region_scores = [
        (region, _count_keyword_hits(text, kws))
        for region, kws in REGION_KEYWORDS.items()
    ]
    best_region, best_score = max(region_scores, key=lambda x: x[1])
    if best_score == 0:
        best_region = "inconnu"

    types = [
        k for k, kws in TYPE_KEYWORDS.items()
        if _count_keyword_hits(text, kws) > 0
    ]

    locations = _select_best_locations(text)

    return Classification(
        region=best_region,
        region_score=best_score,
        fracture_types=sorted(set(types)),
        locations=locations,
    )
