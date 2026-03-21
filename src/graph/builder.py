"""Graph construction from validated triples and entities."""

from __future__ import annotations

from src.schema.entities import Entity
from src.schema.relations import Triple, ValidationStatus
from src.schema.graph_schema import CTIGraph
from src.fusion.provenance import ProvenanceTracker
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def build_graph(
    entities: dict[str, Entity],
    triples: list[Triple],
    doc_id: str = "",
    provenance: ProvenanceTracker | None = None,
) -> CTIGraph:
    """Build a CTIGraph from entities and validated triples.

    Only includes triples with VALIDATED or REPAIRED status.
    """
    graph = CTIGraph()
    if doc_id:
        graph.source_documents.append(doc_id)

    # Add entities
    for entity in entities.values():
        graph.add_entity(entity)
        if provenance:
            for mention in entity.source_mentions:
                parts = mention.split(":")
                provenance.track(
                    element_id=entity.node_id,
                    element_type="node",
                    source_doc_id=parts[0] if parts else doc_id,
                    source_chunk_id=mention,
                )

    # Add valid triples
    for triple in triples:
        if triple.validation_status in (
            ValidationStatus.VALIDATED,
            ValidationStatus.REPAIRED,
        ):
            graph.add_triple(triple)
            if provenance:
                provenance.track(
                    element_id=triple.event_id,
                    element_type="edge",
                    source_doc_id=triple.source_doc_id,
                    source_chunk_id=triple.source_chunk_id,
                    evidence_text=triple.evidence_text,
                )

    logger.info(
        f"Built graph: {graph.num_nodes} nodes, "
        f"{graph.num_edges} edges (from {len(triples)} total triples)"
    )
    return graph
