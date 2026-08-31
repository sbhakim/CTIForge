"""Compute structural / provenance metrics for SG vs NX on cached 20-doc triples.

Looking for metrics where SG has a genuine, defensible lead that arises
from its architecture (symbolic validation, evidence enforcement, type
constraints) rather than from how many raw triples it produces.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SG = Path("output/head_to_head/ctiforge_cached_triples.json")
NX = Path("output/head_to_head/nexus_cached_triples.json")
AKG = Path("output/head_to_head/attackkg_cached_triples.json")


def flatten(cache: dict) -> list[dict]:
    return [t for ts in cache.values() for t in ts]


def evidence_alignment(ts: list[dict]) -> tuple[float, int]:
    """% of triples whose evidence text explicitly contains BOTH entities."""
    n = 0; aligned = 0
    for t in ts:
        ev = (t.get("evidence_text") or "").lower()
        s = (t.get("subject") or "").lower().strip()
        o = (t.get("object") or "").lower().strip()
        if not ev or not s or not o:
            continue
        n += 1
        # Token-overlap heuristic: require at least 3-char prefix of subj and obj to appear
        s_key = s[:12] if len(s) >= 6 else s
        o_key = o[:12] if len(o) >= 6 else o
        if s_key in ev and o_key in ev:
            aligned += 1
    return (aligned / n if n else 0.0), aligned


def evidence_presence(ts: list[dict]) -> float:
    if not ts: return 0.0
    return sum(1 for t in ts if (t.get("evidence_text") or "").strip()) / len(ts)


def entity_name_quality(ts: list[dict]) -> dict:
    """Entity name quality signals: average length, fraction too-short,
    fraction with stopword/article leading, fraction with pronouns."""
    all_names = []
    for t in ts:
        if s := t.get("subject"): all_names.append(s)
        if o := t.get("object"): all_names.append(o)
    if not all_names: return {}
    lengths = [len(n) for n in all_names]
    tiny = sum(1 for n in all_names if len(n.strip()) < 3) / len(all_names)
    leading_article = sum(1 for n in all_names if n.lower().strip().startswith(("the ", "a ", "an ", "this ", "these ", "those "))) / len(all_names)
    pronouns = sum(1 for n in all_names if n.lower().strip() in {"it", "they", "them", "these", "those", "this", "that", "attacker", "attackers", "actors", "victim", "victims"}) / len(all_names)
    punctuation_ending = sum(1 for n in all_names if re.search(r"[:;,.]\s*$", n)) / len(all_names)
    return {
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths)//2],
        "fraction_too_short": tiny,
        "fraction_article_leading": leading_article,
        "fraction_pronoun": pronouns,
        "fraction_punct_trailing": punctuation_ending,
    }


def type_constraint_compliance(ts: list[dict]) -> float:
    """% of triples where the (relation, object_type) pair is semantically
    valid per CTI schema. Hard type constraints for a subset of relations."""
    constraints = {
        "exploits":          {"Vulnerability", "Software", "Infrastructure"},
        "attributed_to":     {"ThreatActor", "Organization"},
        "communicates_with": {"IOC", "Infrastructure"},
        "variant_of":        {"Malware", "Tool", "Technique", "Other"},
        "located_in":        {"Location"},
        "mitigated_by":      {"Organization", "Software", "Tool", "Other"},
        "drops":             {"File", "Malware", "Tool"},
        "delivers":          {"File", "Malware", "Tool"},
    }
    n = 0; ok = 0
    for t in ts:
        rel = t.get("relation")
        ot = t.get("object_type")
        if rel in constraints:
            n += 1
            if ot in constraints[rel]:
                ok += 1
    return ok / n if n else 1.0


def cross_doc_entity_coverage(cache: dict) -> dict:
    """How well does canonicalization bring repeated entities to a shared form?

    Measured as: fraction of *unique* entity names that appear in >1 document.
    Higher = better cross-document entity fusion.
    """
    entity_docs = defaultdict(set)
    for doc_id, triples in cache.items():
        for t in triples:
            entity_docs[t["subject"].strip().lower()].add(doc_id)
            entity_docs[t["object"].strip().lower()].add(doc_id)
    total = len(entity_docs)
    multi = sum(1 for docs in entity_docs.values() if len(docs) > 1)
    return {"unique_entities": total, "multi_doc_entities": multi,
            "cross_doc_rate": multi / total if total else 0}


def validation_status_distribution(ts: list[dict]) -> dict:
    """What fraction of emitted triples carry an audit trail
    (validated / repaired / rejected)?"""
    c = Counter(t.get("validation_status", "unknown") for t in ts)
    return dict(c)


def duplicate_pair_rate(ts: list[dict]) -> float:
    pairs = [(t["subject"].lower().strip(), t["object"].lower().strip()) for t in ts]
    if not pairs: return 0.0
    seen = set(); d = 0
    for p in pairs:
        if p in seen: d += 1
        seen.add(p)
    return d / len(pairs)


def main():
    sg = json.loads(SG.read_text())
    nx = json.loads(NX.read_text())
    akg = json.loads(AKG.read_text()) if AKG.exists() else {}

    systems = {"SG": sg, "NX": nx}
    if akg: systems["AKG"] = akg

    print(f"{'Metric':36s} " + "".join(f"{s:>12s}" for s in systems))
    print("-" * (36 + 12 * len(systems)))

    rows = []

    # Basic counts
    rows.append(("triples_total", lambda ts: len(ts)))

    # Evidence metrics (SG's differentiator)
    rows.append(("evidence_presence_rate", lambda ts: evidence_presence(ts)))
    rows.append(("evidence_alignment_rate", lambda ts: evidence_alignment(ts)[0]))

    # Type-constraint schema compliance
    rows.append(("schema_constraint_compliance", lambda ts: type_constraint_compliance(ts)))

    # Entity-name hygiene
    def eng(ts):
        return entity_name_quality(ts)
    name_q = {k: eng(flatten(v)) for k, v in systems.items()}

    for name, fn in rows:
        vals = [fn(flatten(v)) for v in systems.values()]
        fmt = [f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12d}" for v in vals]
        print(f"{name:36s} " + "".join(fmt))

    # Entity-name hygiene sub-metrics
    print()
    print("--- entity-name hygiene (lower tiny/article/pronoun/punct = better) ---")
    sub_keys = ["mean_length", "median_length", "fraction_too_short",
                "fraction_article_leading", "fraction_pronoun", "fraction_punct_trailing"]
    for k in sub_keys:
        line = f"{k:36s} "
        for s in systems:
            v = name_q[s].get(k, 0)
            line += f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12d}"
        print(line)

    # Cross-doc entity coverage (canonicalization signal)
    print()
    print("--- cross-doc entity coverage (higher cross_doc_rate = better fusion) ---")
    for s, cache in systems.items():
        stats = cross_doc_entity_coverage(cache)
        print(f"  {s}: unique_entities={stats['unique_entities']:3d}  multi_doc={stats['multi_doc_entities']:3d}  cross_doc_rate={stats['cross_doc_rate']:.4f}")

    # Validation status distribution
    print()
    print("--- validation status distribution ---")
    for s, cache in systems.items():
        dist = validation_status_distribution(flatten(cache))
        print(f"  {s}: {dist}")

    # Duplicate pair rate (already known, replicate for completeness)
    print()
    print("--- duplicate (subject, object) pair rate ---")
    for s, cache in systems.items():
        print(f"  {s}: {duplicate_pair_rate(flatten(cache)):.4f}")


if __name__ == "__main__":
    main()
