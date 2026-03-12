"""Document, chunk, and graph-level data models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.schema.entities import Entity
from src.schema.relations import Triple


class Chunk(BaseModel):
    """A paragraph-level segment of a CTI document."""

    chunk_id: str = Field(description="Stable identifier: '{doc_id}_chunk{N}'")
    doc_id: str
    text: str
    chunk_index: int = Field(description="0-based position within the document")
    char_start: int = Field(default=0)
    char_end: int = Field(default=0)


class CTIDocument(BaseModel):
    """A normalized CTI report with metadata."""

    doc_id: str
    title: str = ""
    source: str = Field(default="", description="E.g. vendor name, blog, NVD")
    text: str = Field(description="Full cleaned text")
    chunks: list[Chunk] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class CTIGraph(BaseModel):
    """The full knowledge graph output of CTIForge."""

    nodes: dict[str, Entity] = Field(
        default_factory=dict, description="node_id -> Entity"
    )
    edges: list[Triple] = Field(default_factory=list)
    source_documents: list[str] = Field(
        default_factory=list, description="doc_ids that contributed to this graph"
    )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_validated_edges(self) -> int:
        from src.schema.relations import ValidationStatus
        return sum(1 for e in self.edges if e.validation_status == ValidationStatus.VALIDATED)

    @property
    def num_rejected_edges(self) -> int:
        from src.schema.relations import ValidationStatus
        return sum(1 for e in self.edges if e.validation_status == ValidationStatus.REJECTED)

    def add_entity(self, entity: Entity) -> None:
        """Add or merge an entity into the graph."""
        if entity.node_id in self.nodes:
            existing = self.nodes[entity.node_id]
            # Merge aliases and source mentions
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            for mention in entity.source_mentions:
                if mention not in existing.source_mentions:
                    existing.source_mentions.append(mention)
        else:
            self.nodes[entity.node_id] = entity

    def add_triple(self, triple: Triple) -> None:
        """Add a triple (edge) to the graph."""
        self.edges.append(triple)

    def get_validated_edges(self) -> list[Triple]:
        from src.schema.relations import ValidationStatus
        return [
            e for e in self.edges
            if e.validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
        ]

    def summary(self) -> dict:
        from src.schema.relations import ValidationStatus
        status_counts = {}
        for e in self.edges:
            status_counts[e.validation_status.value] = (
                status_counts.get(e.validation_status.value, 0) + 1
            )
        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "edge_status": status_counts,
            "source_documents": len(self.source_documents),
        }
