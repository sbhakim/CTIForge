"""Graph quality metrics for CTI knowledge graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.schema.graph_schema import CTIGraph
from src.schema.relations import ValidationStatus
from src.graph.export import to_networkx
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def compute_graph_metrics(graph: CTIGraph) -> dict[str, Any]:
    """Compute structural quality metrics for a CTI graph."""
    G = to_networkx(graph)

    total_edges = graph.num_edges
    validated = sum(
        1 for e in graph.edges
        if e.validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
    )
    rejected = sum(
        1 for e in graph.edges if e.validation_status == ValidationStatus.REJECTED
    )

    # Connected components (treating as undirected for connectivity)
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    num_components = len(components)
    isolated_nodes = sum(1 for c in components if len(c) == 1)

    # Density
    density = nx.density(G) if G.number_of_nodes() > 0 else 0.0

    # Average evidence per edge
    evidence_counts = [
        len(e.evidence_text.split(" | ")) if e.evidence_text else 0
        for e in graph.get_validated_edges()
    ]
    avg_evidence = (
        sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0
    )

    # Entity type distribution
    type_dist: dict[str, int] = {}
    for entity in graph.nodes.values():
        type_dist[entity.entity_type.value] = (
            type_dist.get(entity.entity_type.value, 0) + 1
        )

    # Relation type distribution
    rel_dist: dict[str, int] = {}
    for edge in graph.get_validated_edges():
        rel_dist[edge.relation.value] = rel_dist.get(edge.relation.value, 0) + 1

    metrics = {
        "num_nodes": graph.num_nodes,
        "num_edges": total_edges,
        "num_validated_edges": validated,
        "num_rejected_edges": rejected,
        "valid_edge_ratio": round(validated / max(total_edges, 1), 4),
        "num_connected_components": num_components,
        "num_isolated_nodes": isolated_nodes,
        "isolated_node_ratio": round(isolated_nodes / max(graph.num_nodes, 1), 4),
        "graph_density": round(density, 6),
        "avg_evidence_per_edge": round(avg_evidence, 2),
        "entity_type_distribution": type_dist,
        "relation_type_distribution": rel_dist,
    }

    logger.info(
        f"Graph metrics: {metrics['num_nodes']} nodes, "
        f"{metrics['num_validated_edges']}/{metrics['num_edges']} valid edges, "
        f"{metrics['num_connected_components']} components"
    )
    return metrics
