"""Cross-document graph fusion with provenance (Module E).

Merges per-document graph fragments into a unified CTI knowledge graph.
Focus: merge precision over merge scale.
"""

from __future__ import annotations

from src.schema.entities import Entity
from src.schema.relations import Triple, ValidationStatus
from src.schema.graph_schema import CTIGraph
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class GraphMerger:
    """Merges multiple per-document graphs into a single CTI graph."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def merge(self, doc_graphs: list[CTIGraph]) -> CTIGraph:
        """Merge multiple document-level graphs into one.

        Steps:
        1. Merge entities by node_id (already canonicalized)
        2. Merge edges, deduplicating same (subject, relation, object)
        3. Aggregate confidence for edges confirmed by multiple documents
        4. Preserve provenance for all merged edges
        """
        merged = CTIGraph()

        for graph in doc_graphs:
            merged.source_documents.extend(graph.source_documents)

            # Merge entities
            for entity in graph.nodes.values():
                merged.add_entity(entity)

            # Merge edges
            for edge in graph.edges:
                if edge.validation_status == ValidationStatus.REJECTED:
                    continue
                if edge.confidence < self.min_confidence:
                    continue
                merged.add_triple(edge)

        # Deduplicate edges
        merged.edges = self._deduplicate_edges(merged.edges)

        logger.info(
            f"Merged {len(doc_graphs)} graphs -> "
            f"{merged.num_nodes} nodes, {merged.num_edges} edges"
        )
        return merged

    def _deduplicate_edges(self, edges: list[Triple]) -> list[Triple]:
        """Deduplicate edges with same (subject, relation, object).

        When duplicates exist, keep the one with highest confidence
        and merge provenance from all sources.
        """
        # Group by (subject_lower, relation, object_lower)
        groups: dict[tuple, list[Triple]] = {}
        for edge in edges:
            key = (
                edge.subject.lower().strip(),
                edge.relation.value,
                edge.object.lower().strip(),
            )
            groups.setdefault(key, []).append(edge)

        deduplicated = []
        duplicates_removed = 0

        for key, group in groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
                continue

            # Pick the edge with highest confidence as the canonical one
            canonical = max(group, key=lambda e: e.confidence)
            canonical = canonical.model_copy()

            # Boost confidence for edges confirmed by multiple docs
            unique_docs = set(e.source_doc_id for e in group)
            if len(unique_docs) > 1:
                canonical.confidence = min(
                    canonical.confidence + 0.05 * (len(unique_docs) - 1), 1.0
                )

            # Merge evidence from all sources
            all_evidence = [e.evidence_text for e in group if e.evidence_text]
            if all_evidence:
                canonical.evidence_text = " | ".join(dict.fromkeys(all_evidence))

            deduplicated.append(canonical)
            duplicates_removed += len(group) - 1

        if duplicates_removed:
            logger.info(f"Deduplicated {duplicates_removed} duplicate edges")

        return deduplicated
