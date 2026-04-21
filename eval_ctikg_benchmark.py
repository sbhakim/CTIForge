#!/usr/bin/env python3
"""Evaluate CTIForge against the CTIKG sentence-level benchmark.

This script runs our pipeline on the 255 CTI sentences from the CTIKG benchmark
and compares our results against both the gold annotations and CTIKG's outputs.

Usage:
    python eval_ctikg_benchmark.py [--max-sentences N] [-o output_dir]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from src.pipeline import Pipeline, PipelineConfig
from src.schema.graph_schema import CTIDocument
from src.schema.relations import Triple, ValidationStatus
from src.evaluation.per_document import score_per_document, score_pooled
from src.utils.io import write_json
from src.utils.logging import setup_logger

logger = setup_logger("ctikg_benchmark")

CSV_PATH = "data/external/triple extraction benchmark and ctikg results.csv"


def parse_triples_from_text(text: str) -> list[tuple[str, str, str]]:
    """Parse [subject, relation, object] triples from CTIKG's bracket format."""
    text = text.strip()
    raw_triples = re.findall(r"\[([^\[\]]+)\]", text)
    parsed = []
    for t in raw_triples:
        parts = [p.strip() for p in t.split(",")]
        if len(parts) >= 3:
            subj = parts[0].strip()
            rel = parts[1].strip()
            obj = ", ".join(parts[2:]).strip()
            if subj and rel and obj:
                parsed.append((subj, rel, obj))
        elif len(parts) == 2 and parts[0].strip() and parts[1].strip():
            parsed.append((parts[0].strip(), parts[1].strip(), ""))
    return parsed


def normalize_for_match(text: str) -> str:
    """Normalize text for matching: lowercase, strip articles, collapse whitespace."""
    text = text.lower().strip()
    for prefix in ["the ", "a ", "an ", "this ", "these ", "those "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = " ".join(text.split())
    return text


def names_match(a: str, b: str) -> bool:
    """Check if two entity names match (exact, substring, or token overlap)."""
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    if not tokens_a or not tokens_b:
        return False
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    overlap = len(shorter & longer)
    return len(shorter) > 0 and overlap / len(shorter) >= 0.5


def compute_triple_prf(
    predicted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> dict[str, float]:
    """Compute P/R/F1 between predicted and gold triples using soft matching.

    Matching is done on (subject, object) pairs — relation matching is relaxed
    because gold and predicted use different free-form relation strings.
    """
    matched_pred = set()
    matched_gold = set()

    # Pass 1: strict match on subject + object
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            if names_match(p[0], g[0]) and names_match(p[2], g[2]):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    # Pass 2: try swapped subject/object
    for pi, p in enumerate(predicted):
        if pi in matched_pred:
            continue
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            if names_match(p[0], g[2]) and names_match(p[2], g[0]):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    tp = len(matched_gold)
    fp = len(predicted) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_strict_triple_prf(
    predicted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> dict[str, float]:
    """Compute P/R/F1 requiring subject + relation + object all match."""
    matched_pred = set()
    matched_gold = set()

    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            if (
                names_match(p[0], g[0])
                and names_match(p[1], g[1])
                and names_match(p[2], g[2])
            ):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    tp = len(matched_gold)
    fp = len(predicted) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="CTIKG Benchmark Evaluation")
    parser.add_argument("--max-sentences", type=int, default=0, help="Max sentences (0=all)")
    parser.add_argument("-o", "--output-dir", default="output/ctikg_benchmark", help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline config")
    args = parser.parse_args()

    config = PipelineConfig(args.config)
    pipeline = Pipeline(config=config)

    # Load benchmark
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append(row)

    if args.max_sentences > 0:
        rows = rows[:args.max_sentences]

    print(f"\n{'='*60}")
    print(f"  CTIKG Benchmark Evaluation")
    print(f"  Sentences: {len(rows)}")
    print(f"{'='*60}\n")

    # Collect all results
    all_our_triples: list[tuple[str, str, str]] = []
    all_ctikg_triples: list[tuple[str, str, str]] = []
    all_gold_triples: list[tuple[str, str, str]] = []
    per_sentence_results = []
    # Matching must be scoped to a sentence: pooling across all 255 sentences lets a
    # prediction from one sentence be credited against gold from another.
    our_by_sent: dict[str, list] = {}
    ctikg_by_sent: dict[str, list] = {}
    gold_by_sent: dict[str, list] = {}

    t0 = time.time()

    for i, row in enumerate(rows):
        sentence = row[0]
        tactic = row[3] if len(row) > 3 else ""
        gold_text = row[4] if len(row) > 4 else ""
        ctikg_text = row[5] if len(row) > 5 else ""

        gold_triples = parse_triples_from_text(gold_text)
        ctikg_triples = parse_triples_from_text(ctikg_text)

        # Create a minimal document for our pipeline
        doc = CTIDocument(
            doc_id=f"ctikg_s{i:03d}",
            text=sentence,
            source="ctikg_benchmark",
        )

        # Run our pipeline
        try:
            _, triples = pipeline.process_document(doc)
            accepted = [
                t for t in triples
                if t.validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
            ]
            our_triples = [(t.subject, t.relation.value, t.object) for t in accepted]
        except Exception as e:
            logger.warning(f"Error on sentence {i}: {e}")
            our_triples = []

        all_our_triples.extend(our_triples)
        all_ctikg_triples.extend(ctikg_triples)
        all_gold_triples.extend(gold_triples)

        sent_key = f"s{i:03d}"
        our_by_sent[sent_key] = our_triples
        ctikg_by_sent[sent_key] = ctikg_triples
        gold_by_sent[sent_key] = gold_triples

        per_sentence_results.append({
            "sentence_id": i,
            "sentence": sentence[:100],
            "tactic": tactic,
            "gold_count": len(gold_triples),
            "our_count": len(our_triples),
            "ctikg_count": len(ctikg_triples),
        })

        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] Processed, our={len(our_triples)}, "
                  f"gold={len(gold_triples)}, ctikg={len(ctikg_triples)}")

    elapsed = time.time() - t0

    # Compute aggregate metrics, scoped per sentence (see note above).
    # Relaxed: subject + object match (relation-agnostic)
    our_relaxed = score_per_document(our_by_sent, gold_by_sent, compute_triple_prf)
    ctikg_relaxed = score_per_document(ctikg_by_sent, gold_by_sent, compute_triple_prf)

    # Strict: subject + relation + object must all match
    our_strict = score_per_document(our_by_sent, gold_by_sent, compute_strict_triple_prf)
    ctikg_strict = score_per_document(ctikg_by_sent, gold_by_sent, compute_strict_triple_prf)

    # Legacy pooled values, retained so the difference stays visible.
    our_relaxed_pooled = score_pooled(our_by_sent, gold_by_sent, compute_triple_prf)
    ctikg_relaxed_pooled = score_pooled(ctikg_by_sent, gold_by_sent, compute_triple_prf)
    our_strict_pooled = score_pooled(our_by_sent, gold_by_sent, compute_strict_triple_prf)
    ctikg_strict_pooled = score_pooled(ctikg_by_sent, gold_by_sent, compute_strict_triple_prf)

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "benchmark": "CTIKG 255-sentence",
        "sentences": len(rows),
        "gold_triples": len(all_gold_triples),
        "our_triples": len(all_our_triples),
        "ctikg_triples": len(all_ctikg_triples),
        "elapsed_seconds": round(elapsed, 1),
        "relaxed_match": {
            "description": "subject + object match (relation-agnostic)",
            "CTIForge": our_relaxed,
            "CTIKG": ctikg_relaxed,
        },
        "strict_match": {
            "description": "subject + relation + object all match",
            "CTIForge": our_strict,
            "CTIKG": ctikg_strict,
        },
        "relaxed_match_pooled": {
            "description": "legacy: matched across all sentences pooled",
            "CTIForge": our_relaxed_pooled,
            "CTIKG": ctikg_relaxed_pooled,
        },
        "strict_match_pooled": {
            "description": "legacy: matched across all sentences pooled",
            "CTIForge": our_strict_pooled,
            "CTIKG": ctikg_strict_pooled,
        },
        "per_sentence": per_sentence_results,
    }
    write_json(results, out / "ctikg_benchmark_results.json")
    # Persist the validator's error taxonomy alongside the metrics (see
    # improvement_trace.md Section 17 -- benchmark runs previously wrote none).
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  CTIKG Benchmark — Comparison Results")
    print(f"{'='*70}")
    print(f"  Sentences:      {len(rows)}")
    print(f"  Gold triples:   {len(all_gold_triples)}")
    print(f"  Our triples:    {len(all_our_triples)}")
    print(f"  CTIKG triples:  {len(all_ctikg_triples)}")
    print(f"  Time:           {elapsed:.1f}s")
    print()
    print(f"  --- Relaxed Match (subject + object) ---")
    print(f"  {'System':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*46}")
    print(f"  {'CTIForge':<16} {our_relaxed['precision']:>10.4f} {our_relaxed['recall']:>10.4f} {our_relaxed['f1']:>10.4f}")
    print(f"  {'CTIKG':<16} {ctikg_relaxed['precision']:>10.4f} {ctikg_relaxed['recall']:>10.4f} {ctikg_relaxed['f1']:>10.4f}")
    print()
    print(f"  --- Strict Match (subject + relation + object) ---")
    print(f"  {'System':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*46}")
    print(f"  {'CTIForge':<16} {our_strict['precision']:>10.4f} {our_strict['recall']:>10.4f} {our_strict['f1']:>10.4f}")
    print(f"  {'CTIKG':<16} {ctikg_strict['precision']:>10.4f} {ctikg_strict['recall']:>10.4f} {ctikg_strict['f1']:>10.4f}")
    print(f"{'='*70}")
    print(f"\n  Results saved to {out}/ctikg_benchmark_results.json\n")


if __name__ == "__main__":
    main()
