"""Ablation study runner for comparing pipeline configurations.

Runs the system in different configurations to measure the
contribution of each module:
- B-only: extraction only
- B+C: extraction + symbolic validation
- B+C+D: extraction + validation + canonicalization
- B+C+D+E: full pipeline with fusion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.extraction_metrics import compute_extraction_metrics
from src.evaluation.graph_metrics import compute_graph_metrics
from src.schema.relations import Triple
from src.schema.graph_schema import CTIGraph
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class AblationResult:
    """Stores results for one ablation configuration."""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.extraction_metrics: dict[str, Any] = {}
        self.graph_metrics: dict[str, Any] = {}
        self.error_summary: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_name,
            "extraction": self.extraction_metrics,
            "graph": self.graph_metrics,
            "errors": self.error_summary,
        }


def run_ablation_comparison(
    results: list[AblationResult],
    output_path: str | Path,
) -> None:
    """Compare ablation results and save comparison table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "ablation_results": [r.to_dict() for r in results],
        "comparison_table": _build_comparison_table(results),
    }

    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Ablation comparison saved to {path}")

    # Print summary to console
    _print_comparison(results)


def _build_comparison_table(results: list[AblationResult]) -> list[dict]:
    """Build a flat comparison table for easy reading."""
    rows = []
    for r in results:
        row = {"config": r.config_name}
        # Extraction metrics
        for level in ["entity", "relation", "triple"]:
            metrics = r.extraction_metrics.get(level, {})
            for metric in ["precision", "recall", "f1"]:
                row[f"{level}_{metric}"] = metrics.get(metric, 0.0)
        # Graph metrics
        row["num_nodes"] = r.graph_metrics.get("num_nodes", 0)
        row["num_edges"] = r.graph_metrics.get("num_validated_edges", 0)
        row["valid_edge_ratio"] = r.graph_metrics.get("valid_edge_ratio", 0.0)
        row["isolated_node_ratio"] = r.graph_metrics.get("isolated_node_ratio", 0.0)
        rows.append(row)
    return rows


def _print_comparison(results: list[AblationResult]) -> None:
    """Print a readable comparison to the console."""
    header = f"{'Config':<15} {'Triple P':>9} {'Triple R':>9} {'Triple F1':>10} {'Nodes':>6} {'Edges':>6} {'Valid%':>7}"
    logger.info("=" * len(header))
    logger.info("ABLATION COMPARISON")
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        triple = r.extraction_metrics.get("triple", {})
        gm = r.graph_metrics
        logger.info(
            f"{r.config_name:<15} "
            f"{triple.get('precision', 0):.4f}    "
            f"{triple.get('recall', 0):.4f}    "
            f"{triple.get('f1', 0):.4f}     "
            f"{gm.get('num_nodes', 0):>5} "
            f"{gm.get('num_validated_edges', 0):>5} "
            f"{gm.get('valid_edge_ratio', 0):.3f}"
        )
    logger.info("=" * len(header))
