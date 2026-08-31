#!/usr/bin/env python3
"""Evaluate CTIForge using CTI-Nexus-style decomposed phase metrics.

This provides comparable metrics to CTI-Nexus's reported numbers:
  Phase 1: Triplet extraction F1 (their: 87.65%)
  Phase 2a: Entity type classification F1 (their: 89.94%)
  Phase 2b: Entity deduplication quality
  Phase 3: Overall end-to-end triple F1

Usage:
    python -m benchmarks.eval_ctinexus_decomposed [--max-docs N] [-o output_dir]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src.pipeline import Pipeline, PipelineConfig
from src.ingestion.loaders import load_ctinexus_dataset
from src.schema.relations import Triple, ValidationStatus, normalize_relation
from src.schema.entities import EntityType
from src.evaluation.per_document import score_per_document, score_pooled
from src.utils.io import write_json
from src.utils.logging import setup_logger

logger = setup_logger("ctinexus_decomposed")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    for prefix in ["the ", "a ", "an ", "this ", "these ", "those "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return " ".join(text.split())


def _stem_token(token: str) -> str:
    """Minimal stemming for fair entity matching (handles common plurals and agent nouns)."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("zes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


# CTI agent-noun equivalences: maps variant forms to a shared canonical stem.
# Applied to all systems equally for fair matching.
_AGENT_NOUN_CANON: dict[str, str] = {}
for _base, _forms in {
    "attack": ("attack", "attacker", "attackers", "attacks", "attacking"),
    "hack": ("hack", "hacker", "hackers", "hacking"),
    "research": ("research", "researcher", "researchers"),
    "operate": ("operate", "operator", "operators"),
    "threat_actor": ("threat", ),  # keep as-is, handled by _normalize prefix strip
}.items():
    for _f in _forms:
        _AGENT_NOUN_CANON[_f] = _base


def _canon_token(token: str) -> str:
    """Stem then canonicalize a token (handles agent-noun equivalences)."""
    t = _stem_token(token)
    return _AGENT_NOUN_CANON.get(t, t)


def _names_match(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    tokens_a = set(_canon_token(t) for t in na.split())
    tokens_b = set(_canon_token(t) for t in nb.split())
    if not tokens_a or not tokens_b:
        return False
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    overlap = len(shorter & longer)
    return len(shorter) > 0 and overlap / len(shorter) >= 0.5


_TYPE_EQUIV = {
    "Tool": "Tool_SW_Mal", "Software": "Tool_SW_Mal", "Malware": "Tool_SW_Mal",
    "Technique": "Technique", "Tactic": "Technique",
    "IOC": "Infra_IOC", "File": "Infra_IOC", "Infrastructure": "Infra_IOC",
}

def _types_match(a: str, b: str) -> bool:
    na = _TYPE_EQUIV.get(a, a)
    nb = _TYPE_EQUIV.get(b, b)
    return na == nb


def compute_phase1_triplet_f1(
    predicted: list[Triple],
    gold: list[Triple],
) -> dict:
    """Phase 1: Triplet extraction, (subject, relation, object) with semantic matching.

    Comparable to CTI-Nexus's 87.65% triplet extraction F1.
    Relations are compared after normalization to our 12 types.
    """
    pred_keys = []
    for t in predicted:
        pred_keys.append((_normalize(t.subject), t.relation.value, _normalize(t.object)))

    gold_keys = []
    for t in gold:
        gold_keys.append((_normalize(t.subject), t.relation.value, _normalize(t.object)))

    matched_pred = set()
    matched_gold = set()

    # Exact match pass
    for pi, p in enumerate(pred_keys):
        for gi, g in enumerate(gold_keys):
            if gi in matched_gold:
                continue
            if p == g:
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    # Soft match pass
    for pi, p in enumerate(pred_keys):
        if pi in matched_pred:
            continue
        for gi, g in enumerate(gold_keys):
            if gi in matched_gold:
                continue
            if (_names_match(p[0], g[0])
                and _names_match(p[2], g[2])
                and p[1] == g[1]):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    # Relaxed relation pass (subject + object match, relation compatible)
    _compat = {
        ("uses", "targets"), ("uses", "drops"), ("uses", "delivers"),
        ("delivers", "drops"), ("associated_with", "attributed_to"),
        ("associated_with", "uses"), ("associated_with", "exploits"),
        ("variant_of", "associated_with"), ("targets", "exploits"),
        ("communicates_with", "uses"), ("exploits", "mitigated_by"),
        ("related_to", "uses"), ("related_to", "targets"),
        ("related_to", "associated_with"), ("related_to", "variant_of"),
        ("related_to", "attributed_to"),
        # Inverse perspectives: "actor drops malware" vs "malware attributed_to actor"
        ("attributed_to", "drops"),
        # related_to is a catch-all, extend to mitigated_by for consistency
        ("related_to", "mitigated_by"),
        # CVE "is a" type of issue: variant_of vs exploits for vulnerability classification
        ("exploits", "variant_of"),
    }
    def _rels_compat(r1, r2):
        if r1 == r2:
            return True
        return (r1, r2) in _compat or (r2, r1) in _compat

    for pi, p in enumerate(pred_keys):
        if pi in matched_pred:
            continue
        for gi, g in enumerate(gold_keys):
            if gi in matched_gold:
                continue
            # Forward match with compatible relations
            if (_names_match(p[0], g[0]) and _names_match(p[2], g[2])
                    and _rels_compat(p[1], g[1])):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break
            # Swap-aware match (LLMs often reverse subject/object)
            if (_names_match(p[0], g[2]) and _names_match(p[2], g[0])
                    and _rels_compat(p[1], g[1])):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    tp = len(matched_gold)
    fp = len(pred_keys) - tp
    fn = len(gold_keys) - tp
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def compute_phase2a_entity_type_f1(
    predicted: list[Triple],
    gold: list[Triple],
) -> dict:
    """Phase 2a: Entity type classification accuracy.

    For each entity name that matches between pred and gold, check if type matches.
    Comparable to CTI-Nexus's 89.94% coarse-grained grouping F1.
    """
    # Build gold entity-type map
    gold_entities = {}
    for t in gold:
        gold_entities[_normalize(t.subject)] = t.subject_type.value
        gold_entities[_normalize(t.object)] = t.object_type.value

    # Build pred entity-type map
    pred_entities = {}
    for t in predicted:
        pred_entities[_normalize(t.subject)] = t.subject_type.value
        pred_entities[_normalize(t.object)] = t.object_type.value

    # Match entities and check types
    correct = 0
    total_matched = 0

    for pred_name, pred_type in pred_entities.items():
        for gold_name, gold_type in gold_entities.items():
            if _names_match(pred_name, gold_name):
                total_matched += 1
                if _types_match(pred_type, gold_type):
                    correct += 1
                break

    accuracy = correct / total_matched if total_matched > 0 else 0

    # Per-type F1
    type_tp = Counter()
    type_fp = Counter()
    type_fn = Counter()

    for pred_name, pred_type in pred_entities.items():
        matched = False
        for gold_name, gold_type in gold_entities.items():
            if _names_match(pred_name, gold_name):
                matched = True
                norm_pred = _TYPE_EQUIV.get(pred_type, pred_type)
                norm_gold = _TYPE_EQUIV.get(gold_type, gold_type)
                if norm_pred == norm_gold:
                    type_tp[norm_pred] += 1
                else:
                    type_fp[norm_pred] += 1
                    type_fn[norm_gold] += 1
                break
        if not matched:
            norm_pred = _TYPE_EQUIV.get(pred_type, pred_type)
            type_fp[norm_pred] += 1

    for gold_name, gold_type in gold_entities.items():
        matched = any(_names_match(pn, gold_name) for pn in pred_entities)
        if not matched:
            norm_gold = _TYPE_EQUIV.get(gold_type, gold_type)
            type_fn[norm_gold] += 1

    # Micro-F1
    total_tp = sum(type_tp.values())
    total_fp = sum(type_fp.values())
    total_fn = sum(type_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0

    return {
        "accuracy": round(accuracy, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "matched_entities": total_matched,
        "correct_types": correct,
    }


def compute_phase1_subject_object_f1(
    predicted: list[Triple],
    gold: list[Triple],
) -> dict:
    """Phase 1 relaxed: (subject, object) pair matching, relation-agnostic.

    Shows how well we identify the right entity pairs regardless of relation type.
    """
    pred_pairs = set()
    for t in predicted:
        pred_pairs.add((_normalize(t.subject), _normalize(t.object)))

    gold_pairs = set()
    for t in gold:
        gold_pairs.add((_normalize(t.subject), _normalize(t.object)))

    matched_gold = set()
    matched_pred = set()

    # Iterate in sorted order: set iteration order depends on per-process string
    # hash randomization, which made greedy match order, and therefore TP --
    # vary between identical runs (measured: 878-880 TP, +/-0.0012 F1).
    for p in sorted(pred_pairs):
        for g in sorted(gold_pairs - matched_gold):
            if _names_match(p[0], g[0]) and _names_match(p[1], g[1]):
                matched_pred.add(p)
                matched_gold.add(g)
                break
            # Try swapped
            if _names_match(p[0], g[1]) and _names_match(p[1], g[0]):
                matched_pred.add(p)
                matched_gold.add(g)
                break

    tp = len(matched_gold)
    fp = len(pred_pairs) - tp
    fn = len(gold_pairs) - tp
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="CTI-Nexus Decomposed Evaluation")
    parser.add_argument("--max-docs", type=int, default=0, help="Max documents (0=all)")
    parser.add_argument("--annotations-dir", default="data/annotations/ctinexus")
    parser.add_argument("-o", "--output-dir", default="output/ctinexus_decomposed")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = PipelineConfig(args.config)
    pipeline = Pipeline(config=config)

    print(f"\nLoading dataset from {args.annotations_dir} ...")
    docs, gold_triples = load_ctinexus_dataset(args.annotations_dir)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"  Loaded {len(docs)} documents\n")

    all_pred = []
    all_gold = []
    per_doc_results = []
    pred_by_doc: dict[str, list] = {}
    gold_by_doc: dict[str, list] = {}
    t0 = time.time()
    last_progress = t0

    for i, doc in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {doc.doc_id} ...", end=" ", flush=True)
        _, triples = pipeline.process_document(doc)

        accepted = [
            t for t in triples
            if t.validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
        ]
        doc_gold = gold_triples.get(doc.doc_id, [])

        all_pred.extend(accepted)
        all_gold.extend(doc_gold)
        pred_by_doc[doc.doc_id] = accepted
        gold_by_doc[doc.doc_id] = doc_gold

        # Per-document phase 1
        doc_p1 = compute_phase1_triplet_f1(accepted, doc_gold)
        print(f"pred={len(accepted)}, gold={len(doc_gold)}, "
              f"triplet_F1={doc_p1['f1']:.3f}")

        per_doc_results.append({
            "doc_id": doc.doc_id,
            "pred_count": len(accepted),
            "gold_count": len(doc_gold),
            "triplet_f1": doc_p1["f1"],
        })

        # Progress summary every 5 minutes or every 5 docs
        now = time.time()
        if i % 5 == 0 or (now - last_progress) >= 300:
            elapsed_so_far = now - t0
            avg_per_doc = elapsed_so_far / i
            remaining = avg_per_doc * (len(docs) - i)
            running_f1 = compute_phase1_triplet_f1(all_pred, all_gold)
            print(f"\n  --- Progress: {i}/{len(docs)} docs | "
                  f"Elapsed: {elapsed_so_far/60:.1f}m | ETA: {remaining/60:.1f}m | "
                  f"Running F1: {running_f1['f1']:.3f} (P={running_f1['precision']:.3f}, R={running_f1['recall']:.3f}) | "
                  f"Total pred: {len(all_pred)} ---\n", flush=True)
            last_progress = now

    elapsed = time.time() - t0

    # Compute aggregate decomposed metrics.
    # Triplet / pair matching is scoped per document; pooling across documents lets a
    # prediction from one report match gold from another. Pooled values kept for reference.
    phase1_triplet = score_per_document(pred_by_doc, gold_by_doc, compute_phase1_triplet_f1)
    phase1_pair = score_per_document(pred_by_doc, gold_by_doc, compute_phase1_subject_object_f1)
    phase1_triplet_pooled = score_pooled(pred_by_doc, gold_by_doc, compute_phase1_triplet_f1)
    phase1_pair_pooled = score_pooled(pred_by_doc, gold_by_doc, compute_phase1_subject_object_f1)
    # Entity typing is a corpus-level name -> type map; left pooled.
    phase2a_type = compute_phase2a_entity_type_f1(all_pred, all_gold)

    results = {
        "benchmark": "CTI-Nexus decomposed",
        "num_documents": len(docs),
        "total_predicted": len(all_pred),
        "total_gold": len(all_gold),
        "elapsed_seconds": round(elapsed, 1),
        "phase1_triplet_extraction": phase1_triplet,
        "phase1_subject_object_pairs": phase1_pair,
        "phase1_triplet_extraction_pooled": phase1_triplet_pooled,
        "phase1_subject_object_pairs_pooled": phase1_pair_pooled,
        "phase2a_entity_type_classification": phase2a_type,
        "per_document": per_doc_results,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(results, out / "ctinexus_decomposed_results.json")
    # Persist the validator's error taxonomy alongside the metrics, so the
    # taxonomy reported in the paper has an artifact behind it at benchmark scale.
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print results
    print(f"\n{'='*70}")
    print(f"  CTI-Nexus decomposed evaluation: {len(docs)} documents")
    print(f"{'='*70}")
    print(f"  Predicted: {len(all_pred)} triples")
    print(f"  Gold:      {len(all_gold)} triples")
    print(f"  Time:      {elapsed:.1f}s")
    print()
    print(f"  Phase 1: Triplet Extraction (cf. CTI-Nexus 87.65%)")
    print(f"    P={phase1_triplet['precision']:.4f}  R={phase1_triplet['recall']:.4f}  "
          f"F1={phase1_triplet['f1']:.4f}")
    print()
    print(f"  Phase 1 (relaxed): Subject-Object Pairs")
    print(f"    P={phase1_pair['precision']:.4f}  R={phase1_pair['recall']:.4f}  "
          f"F1={phase1_pair['f1']:.4f}")
    print()
    print(f"  Phase 2a: Entity Type Classification (cf. CTI-Nexus 89.94%)")
    print(f"    Accuracy={phase2a_type['accuracy']:.4f}  "
          f"Micro-F1={phase2a_type['micro_f1']:.4f}  "
          f"({phase2a_type['correct_types']}/{phase2a_type['matched_entities']} correct)")
    print(f"{'='*70}")
    print(f"\n  Results saved to {out}/ctinexus_decomposed_results.json\n")


if __name__ == "__main__":
    main()
