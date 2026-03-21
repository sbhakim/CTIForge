"""Export CTI graphs to various formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx

from src.schema.graph_schema import CTIGraph
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def export_json(graph: CTIGraph, path: str | Path) -> None:
    """Export graph to JSON (nodes + edges)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "nodes": [entity.model_dump() for entity in graph.nodes.values()],
        "edges": [triple.model_dump() for triple in graph.get_validated_edges()],
        "summary": graph.summary(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Exported graph JSON to {path}")


def export_edge_csv(graph: CTIGraph, path: str | Path) -> None:
    """Export edges to CSV with provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id", "subject", "subject_type", "relation",
        "object", "object_type", "evidence_text",
        "source_doc_id", "source_chunk_id", "attack_ids",
        "confidence", "validation_status",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for edge in graph.get_validated_edges():
            writer.writerow({
                "event_id": edge.event_id,
                "subject": edge.subject,
                "subject_type": edge.subject_type.value,
                "relation": edge.relation.value,
                "object": edge.object,
                "object_type": edge.object_type.value,
                "evidence_text": edge.evidence_text[:200],
                "source_doc_id": edge.source_doc_id,
                "source_chunk_id": edge.source_chunk_id,
                "attack_ids": "|".join(edge.attack_ids),
                "confidence": f"{edge.confidence:.3f}",
                "validation_status": edge.validation_status.value,
            })

    logger.info(f"Exported {len(graph.get_validated_edges())} edges to {path}")


def export_node_csv(graph: CTIGraph, path: str | Path) -> None:
    """Export canonical node table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "node_id", "canonical_name", "entity_type",
        "aliases", "source_mentions", "external_ids",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entity in graph.nodes.values():
            writer.writerow({
                "node_id": entity.node_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type.value,
                "aliases": "|".join(entity.aliases),
                "source_mentions": "|".join(entity.source_mentions[:5]),
                "external_ids": json.dumps(entity.external_ids),
            })

    logger.info(f"Exported {len(graph.nodes)} nodes to {path}")


def to_networkx(graph: CTIGraph) -> nx.DiGraph:
    """Convert CTIGraph to a NetworkX directed graph for analysis."""
    G = nx.DiGraph()

    for entity in graph.nodes.values():
        G.add_node(
            entity.node_id,
            label=entity.canonical_name,
            entity_type=entity.entity_type.value,
        )

    for edge in graph.get_validated_edges():
        subj_id = f"{edge.subject_type.value.lower()}:{edge.subject.lower().strip().replace(' ', '_')}"
        obj_id = f"{edge.object_type.value.lower()}:{edge.object.lower().strip().replace(' ', '_')}"
        G.add_edge(
            subj_id, obj_id,
            relation=edge.relation.value,
            confidence=edge.confidence,
            event_id=edge.event_id,
        )

    return G
