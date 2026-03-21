"""Provenance tracking for merged graph elements."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Tracks the origin of a graph element across documents."""

    element_id: str = Field(description="node_id or edge event_id")
    element_type: str = Field(description="'node' or 'edge'")
    source_doc_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)
    merge_history: list[str] = Field(
        default_factory=list,
        description="Log of merge actions, e.g. 'merged from doc1 and doc3'",
    )


class ProvenanceTracker:
    """Maintains provenance records for all graph elements."""

    def __init__(self):
        self.records: dict[str, ProvenanceRecord] = {}

    def track(
        self,
        element_id: str,
        element_type: str,
        source_doc_id: str,
        source_chunk_id: str = "",
        evidence_text: str = "",
    ) -> None:
        if element_id not in self.records:
            self.records[element_id] = ProvenanceRecord(
                element_id=element_id,
                element_type=element_type,
            )
        rec = self.records[element_id]
        if source_doc_id and source_doc_id not in rec.source_doc_ids:
            rec.source_doc_ids.append(source_doc_id)
        if source_chunk_id and source_chunk_id not in rec.source_chunk_ids:
            rec.source_chunk_ids.append(source_chunk_id)
        if evidence_text and evidence_text not in rec.evidence_texts:
            rec.evidence_texts.append(evidence_text)

    def log_merge(self, element_id: str, message: str) -> None:
        if element_id in self.records:
            self.records[element_id].merge_history.append(message)

    def get_provenance(self, element_id: str) -> ProvenanceRecord | None:
        return self.records.get(element_id)

    def retention_rate(self) -> float:
        """Fraction of elements that have at least one source document."""
        if not self.records:
            return 0.0
        with_source = sum(1 for r in self.records.values() if r.source_doc_ids)
        return with_source / len(self.records)
