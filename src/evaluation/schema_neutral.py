"""Schema-neutral evaluation: scoring against an EXTERNAL standard.

Why this module exists
----------------------
`compute_quality_metrics()` and `structural_metrics.py::type_constraint_compliance`
score every system against CTIForge's own constraint table -- the same table
the CTIForge validator enforces at extraction time. CTIForge therefore cannot
score low on them: it rejects or repairs any triple that would violate them.
Those metrics are *definitional* (evidence that the symbolic layer enforces its
contract), not comparative, and must not be presented as evidence that one
system produces better graphs than another.

This module provides two things that are safe for cross-system comparison:

1. `stix_compliance()` -- scores against the STIX 2.1 relationship vocabulary,
   an external standard published by OASIS that no compared system enforces.
   The domain/range pairs below are transcribed from the STIX 2.1 spec's
   relationship summary table, not from src/schema/type_constraints.py.

2. `normalization_loss()` -- reports what fraction of a system's output landed
   in the catch-all buckets (`related_to`, `associated_with`, entity type
   `Other`) after being mapped into the 12-relation / 14-type ontology. This is
   the honest, symmetric way to report the "specificity" difference: a system
   that natively emits free text will bucket worse than one prompted to emit the
   enum directly, and that is a property of the mapping, not of graph quality.

Both are applied identically to every system.
"""

from __future__ import annotations

from collections import Counter

# ---------------------------------------------------------------------------
# CTIForge entity type -> STIX 2.1 SDO/SCO type.
# Applied identically to every system's output.
# ---------------------------------------------------------------------------
CTIFORGE_TO_STIX: dict[str, str] = {
    "ThreatActor": "threat-actor",
    "Campaign": "campaign",
    "Malware": "malware",
    "Tool": "tool",
    "Technique": "attack-pattern",
    "Tactic": "attack-pattern",
    "Vulnerability": "vulnerability",
    "Organization": "identity",
    "Location": "location",
    "Infrastructure": "infrastructure",
    "IOC": "indicator",
    "File": "file",
    "Software": "software",
    "Other": None,  # unmappable; excluded from the denominator
}

# ---------------------------------------------------------------------------
# STIX 2.1 relationship summary table (subset covering our 12 relations).
# Source: OASIS STIX 2.1 spec, SRO relationship definitions.
# Each entry: relation -> (valid source SDO types, valid target SDO types)
#
# NOTE: this is deliberately a *subset*. Relations whose STIX domain/range we
# cannot state confidently are omitted and excluded from scoring rather than
# guessed at -- see UNSCORED below.
# ---------------------------------------------------------------------------
STIX_RELATIONSHIP_CONSTRAINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # <attack-pattern|campaign|intrusion-set|malware|threat-actor|tool> uses
    #   <attack-pattern|infrastructure|malware|tool>
    "uses": (
        frozenset({"attack-pattern", "campaign", "intrusion-set", "malware",
                   "threat-actor", "tool"}),
        frozenset({"attack-pattern", "infrastructure", "malware", "tool"}),
    ),
    # <attack-pattern|campaign|intrusion-set|malware|threat-actor|tool> targets
    #   <identity|location|vulnerability>
    "targets": (
        frozenset({"attack-pattern", "campaign", "intrusion-set", "malware",
                   "threat-actor", "tool"}),
        frozenset({"identity", "location", "vulnerability"}),
    ),
    # <attack-pattern|malware|threat-actor|tool> exploits <vulnerability>
    "exploits": (
        frozenset({"attack-pattern", "campaign", "intrusion-set", "malware",
                   "threat-actor", "tool"}),
        frozenset({"vulnerability"}),
    ),
    # <campaign|intrusion-set|malware> attributed-to <intrusion-set|threat-actor|identity>
    "attributed_to": (
        frozenset({"campaign", "intrusion-set", "malware", "threat-actor"}),
        frozenset({"intrusion-set", "threat-actor", "identity"}),
    ),
    # <malware> variant-of <malware>
    "variant_of": (
        frozenset({"malware"}),
        frozenset({"malware"}),
    ),
    # <infrastructure|malware> communicates-with <infrastructure|indicator>
    "communicates_with": (
        frozenset({"infrastructure", "malware", "tool"}),
        frozenset({"infrastructure", "indicator"}),
    ),
    # <malware> drops <malware|tool|file>
    "drops": (
        frozenset({"malware", "threat-actor", "campaign", "intrusion-set"}),
        frozenset({"malware", "tool", "file"}),
    ),
    # <attack-pattern|infrastructure|malware> delivers <malware|file|tool>
    "delivers": (
        frozenset({"attack-pattern", "campaign", "infrastructure", "malware",
                   "threat-actor", "intrusion-set", "indicator"}),
        frozenset({"malware", "file", "tool"}),
    ),
    # <identity|infrastructure|threat-actor|campaign|malware> located-at <location>
    "located_in": (
        frozenset({"identity", "infrastructure", "threat-actor", "campaign", "malware"}),
        frozenset({"location"}),
    ),
    # <course-of-action> mitigates <attack-pattern|indicator|malware|tool|vulnerability>
    # CTIForge inverts the direction (X mitigated_by Y), so source/target swap.
    "mitigated_by": (
        frozenset({"attack-pattern", "indicator", "malware", "tool", "vulnerability",
                   "identity", "software"}),
        frozenset({"identity", "software", "tool", "attack-pattern"}),
    ),
}

# Catch-all relations have no STIX equivalent with a defined domain/range.
# They are excluded from compliance scoring rather than counted as violations,
# so that a system is not penalised twice for the same normalisation failure.
UNSCORED: frozenset[str] = frozenset({"related_to", "associated_with"})

GENERIC_RELATIONS: frozenset[str] = frozenset({"related_to", "associated_with"})


def _triple_fields(t) -> tuple[str, str, str]:
    """Accept either a Triple model or a plain dict."""
    if isinstance(t, dict):
        return (t.get("subject_type", ""), t.get("relation", ""), t.get("object_type", ""))
    return (
        getattr(t.subject_type, "value", t.subject_type),
        getattr(t.relation, "value", t.relation),
        getattr(t.object_type, "value", t.object_type),
    )


def stix_compliance(triples) -> dict:
    """Fraction of scoreable triples satisfying the STIX 2.1 domain/range table.

    Scoreable = relation has a defined STIX domain/range AND both entity types
    map to a STIX type. Triples failing either precondition are reported
    separately rather than silently counted as compliant or non-compliant.
    """
    scored = compliant = 0
    skipped_relation = skipped_type = 0
    violations: Counter = Counter()

    for t in triples:
        s_type, rel, o_type = _triple_fields(t)

        if rel in UNSCORED or rel not in STIX_RELATIONSHIP_CONSTRAINTS:
            skipped_relation += 1
            continue

        s_stix = CTIFORGE_TO_STIX.get(s_type)
        o_stix = CTIFORGE_TO_STIX.get(o_type)
        if s_stix is None or o_stix is None:
            skipped_type += 1
            continue

        scored += 1
        valid_src, valid_tgt = STIX_RELATIONSHIP_CONSTRAINTS[rel]
        if s_stix in valid_src and o_stix in valid_tgt:
            compliant += 1
        else:
            violations[f"{s_stix} -[{rel}]-> {o_stix}"] += 1

    total = len(list(triples)) if not hasattr(triples, "__len__") else len(triples)
    return {
        "stix_compliance": round(compliant / scored, 4) if scored else 0.0,
        "scored_triples": scored,
        "compliant_triples": compliant,
        "skipped_no_stix_relation": skipped_relation,
        "skipped_unmappable_type": skipped_type,
        "coverage": round(scored / total, 4) if total else 0.0,
        "top_violations": violations.most_common(10),
    }


def normalization_loss(triples) -> dict:
    """How much of a system's output fell into catch-all buckets.

    Reported symmetrically for every system. A high value does not by itself
    mean low graph quality: a system that natively emits free-text relations
    will bucket worse than one prompted to emit the target enum directly. This
    metric exists to make that asymmetry visible rather than to hide it inside
    a 'specificity' score.
    """
    n = len(triples)
    if n == 0:
        return {"generic_relation_rate": 0.0, "other_type_rate": 0.0, "n": 0}

    generic = 0
    other_types = 0
    rel_counts: Counter = Counter()

    for t in triples:
        s_type, rel, o_type = _triple_fields(t)
        rel_counts[rel] += 1
        if rel in GENERIC_RELATIONS:
            generic += 1
        other_types += (s_type == "Other") + (o_type == "Other")

    return {
        "generic_relation_rate": round(generic / n, 4),
        "other_type_rate": round(other_types / (2 * n), 4),
        "n": n,
        "relation_distribution": dict(rel_counts.most_common()),
    }


def evidence_presence_rate(triples) -> float:
    """Fraction of triples carrying a non-empty evidence span.

    Schema-independent and objective: it asks only whether the system emitted a
    source sentence at all. Safe for cross-system comparison as-is.
    """
    n = len(triples)
    if n == 0:
        return 0.0
    have = 0
    for t in triples:
        ev = t.get("evidence_text", "") if isinstance(t, dict) else getattr(t, "evidence_text", "")
        if ev and str(ev).strip():
            have += 1
    return round(have / n, 4)
