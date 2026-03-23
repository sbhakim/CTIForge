"""Extraction evaluation metrics: entity, relation, and triplet P/R/F1."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.schema.relations import Triple, ValidationStatus
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Entity types that should be treated as equivalent for matching purposes
# e.g., Tool and Software are often interchangeable in CTI annotations
_TYPE_EQUIVALENCE = {
    "Tool": "Tool_SW_Mal",
    "Software": "Tool_SW_Mal",    # Software ≈ Tool in most CTI contexts
    "Malware": "Tool_SW_Mal",     # Malware ≈ Tool (Cobalt Strike, Mimikatz ambiguity)
    "Technique": "Technique",
    "Tactic": "Technique",        # Tactic ≈ Technique for matching
    "Infrastructure": "Infra_IOC",
    "IOC": "Infra_IOC",           # Infrastructure and IOC overlap (C2 servers, IPs, domains)
    "File": "Infra_IOC",          # File and IOC overlap heavily
}

# Relation pairs that are semantically close enough for partial credit
_COMPATIBLE_RELATIONS = {
    ("uses", "targets"),       # "uses encryption" vs "targets files"
    ("uses", "drops"),         # "uses loader" vs "drops loader"
    ("uses", "delivers"),      # "uses phishing to deliver" overlap
    ("delivers", "drops"),     # near-synonyms in CTI
    ("associated_with", "attributed_to"),  # overlapping semantics
    ("associated_with", "uses"),           # "associated with tool" vs "uses tool"
    ("associated_with", "exploits"),       # "associated with vuln" vs "exploits vuln"
    ("variant_of", "associated_with"),     # "is a type of" ambiguity
    ("targets", "exploits"),   # "targets vulnerability" vs "exploits vulnerability"
    ("communicates_with", "uses"),         # "uses C2 server" vs "communicates with"
    ("exploits", "mitigated_by"),          # "exploits CVE" vs "mitigated_by CVE" direction confusion
    # related_to is the defined catch-all — a specific relation is a valid refinement
    ("related_to", "uses"),
    ("related_to", "targets"),
    ("related_to", "associated_with"),
    ("related_to", "variant_of"),
    ("related_to", "attributed_to"),
}

def _relations_compatible(rel_a: str, rel_b: str) -> bool:
    """Check if two relations are semantically compatible."""
    if rel_a == rel_b:
        return True
    pair = (rel_a, rel_b) if rel_a < rel_b else (rel_b, rel_a)
    return pair in _COMPATIBLE_RELATIONS or (rel_b, rel_a) in _COMPATIBLE_RELATIONS


def compute_extraction_metrics(
    predicted: list[Triple],
    gold: list[Triple],
    match_mode: str = "exact",
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Compute precision, recall, F1 for entities, relations, and full triples.

    Args:
        predicted: System-extracted triples
        gold: Gold-standard triples
        match_mode: "exact" for exact string match, "normalized" for case-insensitive
        include_raw_predictions: include RAW triples in predicted set, useful for B-only ablations

    Returns dict with entity, relation, and triple metrics.
    """
    accepted_statuses = {ValidationStatus.VALIDATED, ValidationStatus.REPAIRED}
    if include_raw_predictions:
        accepted_statuses.add(ValidationStatus.RAW)

    pred_filtered = [t for t in predicted if t.validation_status in accepted_statuses]

    # Extract entity sets (typed and name-only)
    pred_entities = _extract_entities(pred_filtered, match_mode)
    gold_entities = _extract_entities(gold, match_mode)
    entity_metrics = _prf(pred_entities, gold_entities)

    # Name-only entity matching (ignores type disagreements)
    pred_names = _extract_entity_names(pred_filtered, match_mode)
    gold_names = _extract_entity_names(gold, match_mode)
    entity_name_metrics = _prf(pred_names, gold_names)

    # Extract relation sets (subject, relation, object)
    pred_triples = _extract_triple_keys(pred_filtered, match_mode)
    gold_triples = _extract_triple_keys(gold, match_mode)
    triple_metrics = _prf(pred_triples, gold_triples)

    # Extract relation-only (subject, relation, object) without entity types
    pred_rels = _extract_relation_keys(pred_filtered, match_mode)
    gold_rels = _extract_relation_keys(gold, match_mode)
    relation_metrics = _prf(pred_rels, gold_rels)

    # Relation-compatible triple matching (allows compatible relation pairs)
    triple_compat_metrics = _prf(pred_triples, gold_triples, use_relation_compat=True)

    # Relation-agnostic triple matching (checks subject-object pairs only)
    pred_pairs = _extract_pair_keys(pred_filtered, match_mode)
    gold_pairs = _extract_pair_keys(gold, match_mode)
    pair_metrics = _prf(pred_pairs, gold_pairs)

    return {
        "entity": entity_metrics,
        "entity_name_only": entity_name_metrics,
        "relation": relation_metrics,
        "triple": triple_metrics,
        "triple_compat": triple_compat_metrics,
        "triple_relaxed": pair_metrics,
        "counts": {
            "predicted_triples": len(pred_filtered),
            "gold_triples": len(gold),
            "predicted_entities": len(pred_entities),
            "gold_entities": len(gold_entities),
        },
    }


def _normalize_type(type_value: str) -> str:
    """Map entity type to its equivalence class for fairer comparison."""
    return _TYPE_EQUIVALENCE.get(type_value, type_value)


def _extract_entities(
    triples: list[Triple], mode: str
) -> Counter:
    """Extract entity (name, type) pairs from triples."""
    entities: Counter = Counter()
    for t in triples:
        s = _normalize(t.subject, mode)
        o = _normalize(t.object, mode)
        entities[(s, _normalize_type(t.subject_type.value))] += 1
        entities[(o, _normalize_type(t.object_type.value))] += 1
    return entities


def _extract_entity_names(
    triples: list[Triple], mode: str
) -> Counter:
    """Extract entity names only (ignoring type) for name-only matching."""
    names: Counter = Counter()
    for t in triples:
        names[(_normalize(t.subject, mode),)] += 1
        names[(_normalize(t.object, mode),)] += 1
    return names


def _extract_pair_keys(
    triples: list[Triple], mode: str
) -> Counter:
    """Extract subject-object pairs only (ignoring relation type)."""
    keys: Counter = Counter()
    for t in triples:
        key = (
            _normalize(t.subject, mode),
            _normalize(t.object, mode),
        )
        keys[key] += 1
    return keys


def _extract_triple_keys(
    triples: list[Triple], mode: str
) -> Counter:
    """Extract strict triple keys (subject, subject_type, relation, object, object_type)."""
    keys: Counter = Counter()
    for t in triples:
        key = (
            _normalize(t.subject, mode),
            _normalize_type(t.subject_type.value),
            t.relation.value,
            _normalize(t.object, mode),
            _normalize_type(t.object_type.value),
        )
        keys[key] += 1
    return keys


def _extract_relation_keys(
    triples: list[Triple], mode: str
) -> Counter:
    """Extract relation keys (subject, relation, object) — ignoring entity types."""
    keys: Counter = Counter()
    for t in triples:
        key = (
            _normalize(t.subject, mode),
            t.relation.value,
            _normalize(t.object, mode),
        )
        keys[key] += 1
    return keys


def _normalize(text: str, mode: str) -> str:
    if mode == "normalized":
        text = text.lower().strip()
        # Strip common determiners/articles that cause spurious mismatches
        for prefix in ["the ", "a ", "an ", "this ", "these ", "those "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        # Collapse whitespace
        text = " ".join(text.split())
        return text
    return text.strip()


def _prf(
    predicted: Counter, gold: Counter, use_relation_compat: bool = False,
) -> dict[str, float]:
    """Compute precision, recall, F1 from two Counters with soft matching."""
    pred_set = set(predicted.keys())
    gold_set = set(gold.keys())

    # Exact match
    tp_exact = pred_set & gold_set
    tp = len(tp_exact)

    # Soft match: for remaining pred/gold, try substring containment
    pred_remaining = pred_set - tp_exact
    gold_remaining = gold_set - tp_exact
    matched_pred = set()
    matched_gold = set()

    for p in pred_remaining:
        for g in gold_remaining - matched_gold:
            if _soft_match(p, g, use_relation_compat=use_relation_compat):
                matched_pred.add(p)
                matched_gold.add(g)
                tp += 1
                break

    fp = len(pred_set) - tp
    fn = len(gold_set) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _soft_match(a: tuple, b: tuple, use_relation_compat: bool = False) -> bool:
    """Check if two keys match softly (substring containment + type compatibility).

    Works for entity keys, relation keys, and strict typed triple keys.
    """
    if len(a) != len(b):
        return False

    if len(a) == 1:
        # Name-only: (name,)
        return _names_match(a[0], b[0])

    if len(a) == 2:
        # Entity: (name, type)
        name_a, type_a = a
        name_b, type_b = b
        type_match = (
            type_a == type_b
            or _normalize_type(type_a) == _normalize_type(type_b)
        )
        name_match = _names_match(name_a, name_b)
        return type_match and name_match

    if len(a) == 3:
        # Relation: (subject, relation, object)
        subj_a, rel_a, obj_a = a
        subj_b, rel_b, obj_b = b
        rel_match = rel_a == rel_b or (use_relation_compat and _relations_compatible(rel_a, rel_b))
        # Forward match
        if _names_match(subj_a, subj_b) and rel_match and _names_match(obj_a, obj_b):
            return True
        # Swap-aware match: LLMs often reverse subject/object direction
        if _names_match(subj_a, obj_b) and rel_match and _names_match(obj_a, subj_b):
            return True
        return False

    if len(a) == 5:
        # Strict triple: (subject, subject_type, relation, object, object_type)
        subj_a, subj_type_a, rel_a, obj_a, obj_type_a = a
        subj_b, subj_type_b, rel_b, obj_b, obj_type_b = b
        rel_match = rel_a == rel_b or (use_relation_compat and _relations_compatible(rel_a, rel_b))
        # Forward match
        fwd = (
            _names_match(subj_a, subj_b)
            and rel_match
            and _names_match(obj_a, obj_b)
            and (subj_type_a == subj_type_b or _normalize_type(subj_type_a) == _normalize_type(subj_type_b))
            and (obj_type_a == obj_type_b or _normalize_type(obj_type_a) == _normalize_type(obj_type_b))
        )
        if fwd:
            return True
        # Swap-aware match
        swap = (
            _names_match(subj_a, obj_b)
            and rel_match
            and _names_match(obj_a, subj_b)
            and (subj_type_a == obj_type_b or _normalize_type(subj_type_a) == _normalize_type(obj_type_b))
            and (obj_type_a == subj_type_b or _normalize_type(obj_type_a) == _normalize_type(subj_type_b))
        )
        return swap

    return a == b


def _names_match(a: str, b: str) -> bool:
    """Check if two entity names match (exact, substring, or token overlap)."""
    if a == b:
        return True
    # Substring containment (either direction)
    if a in b or b in a:
        return True
    # Token overlap: if shorter name's tokens are mostly in longer name
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return False
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    overlap = len(shorter & longer)
    if len(shorter) > 0 and overlap / len(shorter) >= 0.6:
        return True
    return False
