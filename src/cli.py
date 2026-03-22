"""Command-line interface for CTIForge."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="ctiforge", help="CTIForge: Neuro-symbolic CTI KG construction")
console = Console()


@app.command()
def extract(
    input_path: str = typer.Argument(help="Path to CTI report (text or JSON)"),
    output_dir: str = typer.Option("output", help="Output directory"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
    no_validation: bool = typer.Option(False, help="Disable symbolic validation (ablation)"),
    no_canonicalization: bool = typer.Option(False, help="Disable canonicalization (ablation)"),
):
    """Extract a CTI knowledge graph from a report."""
    from src.pipeline import Pipeline, PipelineConfig
    from src.ingestion.loaders import load_plain_text, load_ctinexus_annotation
    from src.graph.export import export_json, export_edge_csv, export_node_csv

    pipe_config = PipelineConfig(config)
    pipeline = Pipeline(
        config=pipe_config,
        enable_validation=not no_validation,
        enable_canonicalization=not no_canonicalization,
    )

    # Load document
    path = Path(input_path)
    if path.suffix == ".json":
        doc, _ = load_ctinexus_annotation(path)
    else:
        doc = load_plain_text(path)

    # Process
    graph, triples = pipeline.process_document(doc)

    # Export
    out = Path(output_dir)
    export_json(graph, out / f"{doc.doc_id}_graph.json")
    export_edge_csv(graph, out / f"{doc.doc_id}_edges.csv")
    export_node_csv(graph, out / f"{doc.doc_id}_nodes.csv")

    # Save error taxonomy
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print summary
    summary = graph.summary()
    console.print(f"\n[bold green]Graph Summary[/bold green]")
    console.print(f"  Nodes: {summary['total_nodes']}")
    console.print(f"  Edges: {summary['total_edges']}")
    console.print(f"  Edge status: {summary['edge_status']}")
    console.print(f"\n  Output: {out}/")


@app.command()
def evaluate(
    annotations_dir: str = typer.Argument(help="Path to gold annotations directory"),
    output_dir: str = typer.Option("output/eval", help="Evaluation output directory"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
    max_docs: int = typer.Option(0, help="Max documents to evaluate (0 = all)"),
):
    """Run evaluation against gold annotations."""
    from src.pipeline import Pipeline, PipelineConfig
    from src.ingestion.loaders import load_ctinexus_dataset
    from src.evaluation.extraction_metrics import compute_extraction_metrics
    from src.evaluation.graph_metrics import compute_graph_metrics
    from src.utils.io import write_json

    pipe_config = PipelineConfig(config)
    pipeline = Pipeline(config=pipe_config)

    docs, gold_triples = load_ctinexus_dataset(annotations_dir)
    if max_docs > 0:
        docs = docs[:max_docs]

    all_pred = []
    all_gold = []

    for doc in docs:
        graph, triples = pipeline.process_document(doc)
        doc_gold = gold_triples.get(doc.doc_id, [])
        all_pred.extend(triples)
        all_gold.extend(doc_gold)

    # Compute metrics
    ext_metrics = compute_extraction_metrics(all_pred, all_gold, match_mode="normalized")
    graph_metrics = compute_graph_metrics(graph)

    results = {
        "extraction_metrics": ext_metrics,
        "graph_metrics": graph_metrics,
        "error_summary": pipeline.error_logger.get_summary(),
        "num_documents": len(docs),
    }

    out = Path(output_dir)
    write_json(results, out / "evaluation_results.json")
    pipeline.error_logger.save(out / "error_taxonomy")

    # Print results
    _print_metrics_table(ext_metrics)
    console.print(f"\nResults saved to {out}/")


@app.command()
def load_data(
    annotations_dir: str = typer.Argument(
        default="data/annotations/ctinexus",
        help="Path to annotations",
    ),
):
    """Test data loading and show dataset statistics."""
    from src.ingestion.loaders import load_ctinexus_dataset
    from src.schema.entities import EntityType

    docs, gold_triples = load_ctinexus_dataset(annotations_dir)

    console.print(f"\n[bold]Dataset Statistics[/bold]")
    console.print(f"  Documents: {len(docs)}")
    console.print(f"  Total gold triples: {sum(len(t) for t in gold_triples.values())}")

    # Entity type distribution
    type_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}
    for triples in gold_triples.values():
        for t in triples:
            type_counts[t.subject_type.value] = type_counts.get(t.subject_type.value, 0) + 1
            type_counts[t.object_type.value] = type_counts.get(t.object_type.value, 0) + 1
            rel_counts[t.relation.value] = rel_counts.get(t.relation.value, 0) + 1

    console.print(f"\n  [bold]Entity Types:[/bold]")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        console.print(f"    {etype}: {count}")

    console.print(f"\n  [bold]Relation Types:[/bold]")
    for rtype, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        console.print(f"    {rtype}: {count}")


def _print_metrics_table(metrics: dict) -> None:
    """Print extraction metrics as a rich table."""
    table = Table(title="Extraction Metrics")
    table.add_column("Level", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="green")

    for level in ["entity", "relation", "triple"]:
        m = metrics.get(level, {})
        table.add_row(
            level.title(),
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
