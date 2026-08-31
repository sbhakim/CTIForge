#!/usr/bin/env python3
"""Module ablation (B / B+C / B+C+D) over one shared extraction pass.

The appendix ablation answers RQ1 -- "does the symbolic layer improve extraction
quality?" -- on 10 documents, where B+C beats B-only by one true positive. That
is not an answer. Two things were wrong with it: the sample was too small, and
each arm re-ran extraction, so the arms differed by both the module under test
and a fresh sampling of the LLM. Resampling noise at temperature 0.1 is about
the size of the effect being measured.

This runs extraction once and derives every arm from those same raw triples.
Everything downstream of extraction is deterministic, so the only thing that
varies between arms is the module. Scoring uses the same matcher and the same
per-document scoping as eval_ctinexus_decomposed.py, so the numbers are directly
comparable with the headline tables.

Predicted triples are persisted per arm, which also unblocks the flag
calibration study and post-hoc error attribution without another run.

Usage:
    conda run -n cti python eval_module_ablation.py --config configs/openrouter-minimax.yaml \
        --max-docs 149 -o output/paper/ablation_149_m3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_ctinexus_decomposed import (
    compute_phase1_triplet_f1,
    compute_phase1_subject_object_f1,
)
from src.evaluation.per_document import score_per_document, score_pooled
from src.ingestion.loaders import load_ctinexus_dataset
from src.pipeline import Pipeline, PipelineConfig
from src.schema.relations import ValidationStatus

# (label, enable_validation, enable_canonicalization)
ARMS = [
    ("B-only", False, False),
    ("B+C", True, False),
    ("B+C+D", True, True),
]


def accepted(triples):
    """Triples that reach the graph.

    Not `status in (VALIDATED, REPAIRED)`: with validation disabled every triple
    keeps status RAW, and that filter would score B-only as zero.
    """
    return [t for t in triples if t.validation_status != ValidationStatus.REJECTED]


def triple_record(t):
    return {
        "subject": t.subject,
        "subject_type": t.subject_type.value if t.subject_type else None,
        "relation": t.relation.value,
        "object": t.object,
        "object_type": t.object_type.value if t.object_type else None,
        "evidence_text": t.evidence_text,
        "confidence": t.confidence,
        "validation_status": t.validation_status.value,
        "triple_id": t.event_id,
        "source_doc_id": t.source_doc_id,
        "source_chunk_id": t.source_chunk_id,
    }


def main():
    ap = argparse.ArgumentParser(description="Shared-extraction module ablation")
    ap.add_argument("--config", default="configs/openrouter-minimax.yaml")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--annotations-dir", default="data/annotations/ctinexus")
    ap.add_argument("-o", "--output-dir", default="output/module_ablation")
    ap.add_argument("--max-empty-rate", type=float, default=0.05,
                    help="abort if more than this fraction of documents "
                         "extract zero triples (default 0.05)")
    args = ap.parse_args()
    max_empty_rate = args.max_empty_rate

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(args.config)

    docs, gold_triples = load_ctinexus_dataset(args.annotations_dir)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"\n  {len(docs)} documents | config {args.config}")

    # ---- Phase 1: one extraction pass -----------------------------------
    extractor_pipe = Pipeline(config=config)
    raw_by_doc: dict[str, list] = {}
    seg_by_doc: dict[str, object] = {}

    print("\n  Extraction (once, shared by every arm)")
    t0 = time.time()
    for i, doc in enumerate(docs, 1):
        segmented, raw = extractor_pipe.extract_document(doc)
        raw_by_doc[doc.doc_id] = raw
        seg_by_doc[doc.doc_id] = segmented
        print(f"    [{i}/{len(docs)}] {doc.doc_id}: {len(raw)} raw", flush=True)
        if i % 10 == 0:
            el = time.time() - t0
            print(f"    --- {i}/{len(docs)} | {el/60:.1f}m | "
                  f"ETA {el/i*(len(docs)-i)/60:.1f}m ---", flush=True)
    extract_seconds = time.time() - t0
    total_raw = sum(len(v) for v in raw_by_doc.values())
    print(f"  Extraction done: {total_raw} raw triples in {extract_seconds/60:.1f}m")

    # A document that yields nothing is usually a swallowed provider error, not
    # a document with no facts in it. Extraction failures degrade silently --
    # the arms still score, and the run still writes a complete-looking results
    # file -- so refuse to report metrics computed over a contaminated pass.
    empty = [d for d, ts in raw_by_doc.items() if not ts]
    empty_rate = len(empty) / max(len(docs), 1)
    if empty:
        print(f"  WARNING: {len(empty)}/{len(docs)} documents extracted 0 triples "
              f"({empty_rate:.1%})")
        for d in empty[:10]:
            print(f"           empty: {d}")
    if empty_rate > max_empty_rate:
        raise SystemExit(
            f"\n  ABORT: empty-document rate {empty_rate:.1%} exceeds the "
            f"--max-empty-rate ceiling of {max_empty_rate:.1%}.\n"
            f"  This usually means the provider was rate-limiting or timing out.\n"
            f"  Metrics over this pass would understate the backbone, so no\n"
            f"  results file was written. Re-run once the provider is healthy.\n"
        )

    json.dump(
        {d: [triple_record(t) for t in ts] for d, ts in raw_by_doc.items()},
        open(out / "extracted_triples.json", "w"), indent=1,
    )

    gold_by_doc = {d.doc_id: gold_triples.get(d.doc_id, []) for d in docs}

    # ---- Phase 2: arms over the same raw triples ------------------------
    results = {}
    for label, use_val, use_canon in ARMS:
        pipe = Pipeline(
            config=config,
            enable_validation=use_val,
            enable_canonicalization=use_canon,
        )
        pred_by_doc = {}
        for doc in docs:
            # Validation mutates status and confidence in place, so each arm
            # needs its own copies or arm N sees arm N-1's edits.
            raw_copy = [t.model_copy(deep=True) for t in raw_by_doc[doc.doc_id]]
            _, triples = pipe.postprocess(raw_copy, seg_by_doc[doc.doc_id])
            pred_by_doc[doc.doc_id] = accepted(triples)

        trip = score_per_document(pred_by_doc, gold_by_doc, compute_phase1_triplet_f1)
        pair = score_per_document(pred_by_doc, gold_by_doc, compute_phase1_subject_object_f1)
        trip_pooled = score_pooled(pred_by_doc, gold_by_doc, compute_phase1_triplet_f1)

        n_pred = sum(len(v) for v in pred_by_doc.values())
        results[label] = {
            "triplet": trip, "subject_object": pair, "triplet_pooled": trip_pooled,
            "predicted": n_pred,
            "error_summary": pipe.error_logger.get_summary(),
        }
        json.dump(
            {d: [triple_record(t) for t in ts] for d, ts in pred_by_doc.items()},
            open(out / f"predictions_{label.replace('+', '_')}.json", "w"), indent=1,
        )
        pipe.error_logger.save(out / f"error_taxonomy_{label.replace('+', '_')}")
        print(f"  {label:8s} triplet F1={trip['f1']:.4f} "
              f"P={trip['precision']:.4f} R={trip['recall']:.4f} pred={n_pred}")

    # ---- Phase 3: report ------------------------------------------------
    gold_n = sum(len(v) for v in gold_by_doc.values())
    print("\n" + "=" * 70)
    print(f"  Module ablation, {len(docs)} docs, {gold_n} gold triples")
    print(f"  One extraction pass ({total_raw} raw triples), per-document scoping")
    print("=" * 70)
    print(f"\n  {'arm':10s} {'F1':>8s} {'P':>8s} {'R':>8s} {'S-O F1':>8s} {'pred':>7s}")
    print("  " + "-" * 54)
    base = results["B-only"]["triplet"]["f1"]
    for label, _, _ in ARMS:
        r = results[label]
        delta = "(base)" if label == "B-only" else f"{r['triplet']['f1'] - base:+.4f}"
        print(f"  {label:10s} {r['triplet']['f1']:8.4f} {r['triplet']['precision']:8.4f} "
              f"{r['triplet']['recall']:8.4f} {r['subject_object']['f1']:8.4f} "
              f"{r['predicted']:7d}   {delta}")

    payload = {
        "benchmark": "CTI-Nexus module ablation (shared extraction)",
        "config": args.config,
        "num_documents": len(docs),
        "gold_triples": gold_n,
        "raw_triples": total_raw,
        "extract_seconds": round(extract_seconds, 1),
        "arms": results,
    }
    json.dump(payload, open(out / "module_ablation_results.json", "w"), indent=2)
    print(f"\n  Written to {out}/module_ablation_results.json\n")


if __name__ == "__main__":
    main()
