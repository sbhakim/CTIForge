#!/usr/bin/env python3
"""
CTIForge: Neuro-Symbolic CTI Knowledge Graph Construction

Main entry point for the CTIForge pipeline. This script provides
a standard interface for running extraction, evaluation, and ablation
studies on Cyber Threat Intelligence reports.

Usage:
    python main.py extract <input_file> [options]
    python main.py evaluate [options]
    python main.py ablation [options]
    python main.py info
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger("main")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract a knowledge graph from a single CTI report."""
    from src.pipeline import Pipeline, PipelineConfig
    from src.ingestion.loaders import load_plain_text, load_ctinexus_annotation
    from src.graph.export import export_json, export_edge_csv, export_node_csv

    config = PipelineConfig(args.config)
    pipeline = Pipeline(
        config=config,
        enable_validation=not args.no_validation,
        enable_canonicalization=not args.no_canonicalization,
    )

    # Load document
    path = Path(args.input)
    if path.suffix == ".json":
        doc, gold = load_ctinexus_annotation(path)
    else:
        doc = load_plain_text(path)
        gold = []

    t0 = time.time()
    graph, triples = pipeline.process_document(doc)
    elapsed = time.time() - t0

    # Export outputs
    out = Path(args.output_dir)
    export_json(graph, out / f"{doc.doc_id}_graph.json")
    export_edge_csv(graph, out / f"{doc.doc_id}_edges.csv")
    export_node_csv(graph, out / f"{doc.doc_id}_nodes.csv")
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print summary
    summary = graph.summary()
    valid = sum(1 for t in triples if t.validation_status.value in ("validated", "repaired"))
    rejected = sum(1 for t in triples if t.validation_status.value == "rejected")

    print(f"\n{'='*60}")
    print(f"  CTIForge, Extraction Complete")
    print(f"{'='*60}")
    print(f"  Document:   {doc.doc_id}")
    print(f"  Chunks:     {len(doc.chunks)}")
    print(f"  Raw triples: {len(triples)}")
    print(f"  Valid:       {valid}")
    print(f"  Rejected:    {rejected}")
    print(f"  Nodes:       {summary['total_nodes']}")
    print(f"  Edges:       {summary['total_edges']}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Output:      {out.resolve()}/")
    if gold:
        print(f"  Gold triples: {len(gold)}")
    print(f"{'='*60}\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate pipeline against gold-annotated CTI-Nexus dataset."""
    from src.pipeline import Pipeline, PipelineConfig
    from src.ingestion.loaders import load_ctinexus_dataset
    from src.evaluation.extraction_metrics import compute_extraction_metrics
    from src.evaluation.graph_metrics import compute_graph_metrics
    from src.utils.io import write_json

    config = PipelineConfig(args.config)
    pipeline = Pipeline(
        config=config,
        enable_validation=not args.no_validation,
        enable_canonicalization=not args.no_canonicalization,
    )

    print(f"\nLoading dataset from {args.annotations_dir} ...")
    docs, gold_triples = load_ctinexus_dataset(args.annotations_dir)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"  Loaded {len(docs)} documents, {sum(len(v) for v in gold_triples.values())} gold triples\n")

    all_pred = []
    all_gold = []
    t0 = time.time()

    for i, doc in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] Processing {doc.doc_id} ...", end=" ", flush=True)
        graph, triples = pipeline.process_document(doc)
        doc_gold = gold_triples.get(doc.doc_id, [])
        all_pred.extend(triples)
        all_gold.extend(doc_gold)
        valid = sum(1 for t in triples if t.validation_status.value in ("validated", "repaired"))
        print(f"→ {valid} valid triples (gold: {len(doc_gold)})")

    elapsed = time.time() - t0

    # Compute metrics
    ext_metrics = compute_extraction_metrics(all_pred, all_gold, match_mode="normalized")
    graph_metrics = compute_graph_metrics(graph)

    results = {
        "pipeline_config": pipeline.get_ablation_config_name(),
        "num_documents": len(docs),
        "extraction_metrics": ext_metrics,
        "graph_metrics": graph_metrics,
        "error_summary": pipeline.error_logger.get_summary(),
        "elapsed_seconds": round(elapsed, 1),
    }

    out = Path(args.output_dir)
    write_json(results, out / "evaluation_results.json")
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print results
    print(f"\n{'='*60}")
    print(f"  Evaluation Results, Config: {pipeline.get_ablation_config_name()}")
    print(f"{'='*60}")
    print(f"  Documents:  {len(docs)}")
    print(f"  Predicted:  {len(all_pred)} triples")
    print(f"  Gold:       {len(all_gold)} triples")
    print(f"  Time:       {elapsed:.1f}s")
    print()

    for level in ["entity", "entity_name_only", "relation", "triple", "triple_compat", "triple_relaxed"]:
        m = ext_metrics.get(level, {})
        labels = {
            "entity_name_only": "ENTITY(name)",
            "triple_compat": "TRIPLE(compat)",
            "triple_relaxed": "TRIPLE(relax)",
        }
        label = labels.get(level, level.upper())
        print(f"  {label:14s}  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  F1={m.get('f1',0):.4f}")

    print(f"\n  Error taxonomy: {out}/error_taxonomy/")
    print(f"  Full results:   {out}/evaluation_results.json")
    print(f"{'='*60}\n")


def cmd_ablation(args: argparse.Namespace) -> None:
    """Run ablation study comparing pipeline configurations."""
    from src.pipeline import Pipeline, PipelineConfig
    from src.ingestion.loaders import load_ctinexus_dataset
    from src.evaluation.extraction_metrics import compute_extraction_metrics
    from src.utils.io import write_json

    config = PipelineConfig(args.config)

    print(f"\nLoading dataset from {args.annotations_dir} ...")
    docs, gold_triples = load_ctinexus_dataset(args.annotations_dir)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"  Loaded {len(docs)} documents\n")

    # Ablation configurations: (name, validation, canonicalization)
    configs = [
        ("B-only",   False, False),
        ("B+C",      True,  False),
        ("B+C+D",    True,  True),
    ]

    all_results = {}

    for cfg_name, use_val, use_canon in configs:
        print(f"{'='*50}")
        print(f"  Running config: {cfg_name}")
        print(f"{'='*50}")

        pipeline = Pipeline(
            config=config,
            enable_validation=use_val,
            enable_canonicalization=use_canon,
        )

        all_pred = []
        all_gold = []
        t0 = time.time()

        for i, doc in enumerate(docs, 1):
            print(f"    [{i}/{len(docs)}] {doc.doc_id} ...", flush=True)
            _, triples = pipeline.process_document(doc)
            doc_gold = gold_triples.get(doc.doc_id, [])
            all_pred.extend(triples)
            all_gold.extend(doc_gold)

        elapsed = time.time() - t0
        ext_metrics = compute_extraction_metrics(
            all_pred,
            all_gold,
            match_mode="normalized",
            include_raw_predictions=not use_val,
        )

        all_results[cfg_name] = {
            "extraction_metrics": ext_metrics,
            "error_summary": pipeline.error_logger.get_summary(),
            "elapsed_seconds": round(elapsed, 1),
            "num_predicted": len(all_pred),
            "num_gold": len(all_gold),
        }

        for level in ["entity", "relation", "triple"]:
            m = ext_metrics.get(level, {})
            print(f"    {level.upper():10s}  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  F1={m.get('f1',0):.4f}")
        print(f"    Time: {elapsed:.1f}s\n")

    # Save comparison
    out = Path(args.output_dir)
    write_json(all_results, out / "ablation_results.json")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Ablation Comparison, Triple-Level F1")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for cfg_name in ["B-only", "B+C", "B+C+D"]:
        m = all_results[cfg_name]["extraction_metrics"].get("triple", {})
        print(f"  {cfg_name:<12} {m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")
    print(f"{'='*70}")
    print(f"\n  Results saved to {out}/ablation_results.json\n")


def cmd_info(args: argparse.Namespace) -> None:
    """Print system info, available providers, and dataset statistics."""
    from src.pipeline import PipelineConfig
    from src.utils.config import get_settings, load_yaml_config

    settings = get_settings()
    yaml_cfg = load_yaml_config(args.config)
    pipe_cfg = PipelineConfig(args.config)

    print(f"\n{'='*60}")
    print(f"  CTIForge, System Information")
    print(f"{'='*60}")
    print(f"  Provider:    {pipe_cfg.model_provider}")
    print(f"  Model:       {pipe_cfg.model}")
    print(f"  Embedding:   {pipe_cfg.embedding_model}")
    print(f"  Providers:   {', '.join(settings.available_providers())}")
    print(f"  Config:      {args.config}")

    # Dataset stats
    annotations = Path(yaml_cfg.get("paths", {}).get("annotations", "data/annotations/ctinexus"))
    if annotations.exists():
        count = len(list(annotations.glob("*.json")))
        print(f"  Annotations: {count} files in {annotations}")

    external = Path(yaml_cfg.get("paths", {}).get("external", "data/external"))
    if external.exists():
        files = list(external.iterdir())
        print(f"  External:    {len(files)} files in {external}")

    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="CTIForge",
        description="Neuro-Symbolic CTI Knowledge Graph Construction",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- extract ---
    p_ext = subparsers.add_parser("extract", help="Extract KG from a single CTI report")
    p_ext.add_argument("input", help="Path to CTI report (text or JSON)")
    p_ext.add_argument("-o", "--output-dir", default="output", help="Output directory")
    p_ext.add_argument("--no-validation", action="store_true", help="Disable symbolic validation")
    p_ext.add_argument("--no-canonicalization", action="store_true", help="Disable canonicalization")

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate against gold annotations")
    p_eval.add_argument("--annotations-dir", default="data/annotations/ctinexus", help="Gold annotations directory")
    p_eval.add_argument("-o", "--output-dir", default="output/eval", help="Evaluation output directory")
    p_eval.add_argument("--max-docs", type=int, default=0, help="Max documents (0=all)")
    p_eval.add_argument("--no-validation", action="store_true", help="Disable symbolic validation")
    p_eval.add_argument("--no-canonicalization", action="store_true", help="Disable canonicalization")

    # --- ablation ---
    p_abl = subparsers.add_parser("ablation", help="Run ablation study")
    p_abl.add_argument("--annotations-dir", default="data/annotations/ctinexus", help="Gold annotations directory")
    p_abl.add_argument("-o", "--output-dir", default="output/ablation", help="Ablation output directory")
    p_abl.add_argument("--max-docs", type=int, default=5, help="Max documents per config")

    # --- info ---
    subparsers.add_parser("info", help="Show system and dataset info")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "extract": cmd_extract,
        "evaluate": cmd_evaluate,
        "ablation": cmd_ablation,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
