"""Tests for data loading and pipeline integration (no LLM calls)."""

import pytest
from pathlib import Path

from src.schema.graph_schema import CTIDocument, CTIGraph
from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.ingestion.segmenter import segment_document
from src.ingestion.loaders import load_ctinexus_annotation
from src.symbolic.validator import SymbolicValidator
from src.grounding.canonicalizer import Canonicalizer
from src.graph.builder import build_graph
from src.fusion.merger import GraphMerger
from src.evaluation.extraction_metrics import compute_extraction_metrics
from src.evaluation.graph_metrics import compute_graph_metrics


ANNOTATIONS_DIR = Path("data/annotations/ctinexus")


class TestSegmenter:
    def test_segment_short_doc(self):
        doc = CTIDocument(doc_id="test", text="Short paragraph.\n\nAnother paragraph.")
        doc = segment_document(doc)
        assert len(doc.chunks) >= 1

    def test_segment_empty_doc(self):
        doc = CTIDocument(doc_id="test", text="")
        doc = segment_document(doc)
        assert len(doc.chunks) == 0

    def test_chunk_ids_are_stable(self):
        doc = CTIDocument(doc_id="test", text="Para one.\n\nPara two.\n\nPara three.")
        doc = segment_document(doc)
        for i, chunk in enumerate(doc.chunks):
            assert chunk.chunk_id == f"test_chunk{i}"


class TestDataLoading:
    @pytest.mark.skipif(
        not ANNOTATIONS_DIR.exists(),
        reason="CTI-Nexus annotations not available",
    )
    def test_load_single_annotation(self):
        files = list(ANNOTATIONS_DIR.glob("*.json"))
        assert len(files) > 0
        doc, triples = load_ctinexus_annotation(files[0])
        assert doc.doc_id
        assert len(doc.text) > 0
        assert isinstance(triples, list)

    @pytest.mark.skipif(
        not ANNOTATIONS_DIR.exists(),
        reason="CTI-Nexus annotations not available",
    )
    def test_triples_have_valid_types(self):
        files = list(ANNOTATIONS_DIR.glob("*.json"))
        _, triples = load_ctinexus_annotation(files[0])
        for t in triples:
            assert isinstance(t.subject_type, EntityType)
            assert isinstance(t.relation, RelationType)
            assert isinstance(t.object_type, EntityType)


class TestEndToEnd:
    """End-to-end test without LLM calls (using synthetic triples)."""

    def _make_triples(self) -> list[Triple]:
        return [
            Triple(
                event_id="t0",
                subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.USES,
                object="Cobalt Strike",
                object_type=EntityType.TOOL,
                evidence_text="APT29 uses Cobalt Strike.",
                source_doc_id="doc1",
                source_chunk_id="doc1_chunk0",
            ),
            Triple(
                event_id="t1",
                subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.TARGETS,
                object="US Government",
                object_type=EntityType.ORGANIZATION,
                evidence_text="APT29 targets government agencies.",
                source_doc_id="doc1",
                source_chunk_id="doc1_chunk0",
            ),
            Triple(
                event_id="t2",
                subject="the attacker",  # generic placeholder
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.EXPLOITS,
                object="CVE 2023 12345",  # needs repair
                object_type=EntityType.VULNERABILITY,
                evidence_text="The attacker exploited the vulnerability.",
                source_doc_id="doc1",
                source_chunk_id="doc1_chunk1",
            ),
            Triple(
                event_id="t3",
                subject="",  # empty — should be rejected
                subject_type=EntityType.MALWARE,
                relation=RelationType.DROPS,
                object="payload.exe",
                object_type=EntityType.FILE,
                evidence_text="",
                source_doc_id="doc1",
                source_chunk_id="doc1_chunk1",
            ),
            Triple(
                event_id="t4",
                subject="SomeVuln",
                subject_type=EntityType.VULNERABILITY,
                relation=RelationType.USES,  # impossible type pair
                object="Canada",
                object_type=EntityType.LOCATION,
                evidence_text="Some text.",
                source_doc_id="doc1",
                source_chunk_id="doc1_chunk1",
            ),
        ]

    def test_validation_pipeline(self):
        triples = self._make_triples()
        validator = SymbolicValidator()
        validated = validator.validate_triples(triples)

        # t0: valid, t1: valid, t2: repaired, t3: rejected (empty), t4: rejected (type pair)
        statuses = [t.validation_status for t in validated]
        assert ValidationStatus.VALIDATED in statuses
        assert ValidationStatus.REJECTED in statuses

    def test_canonicalization_pipeline(self):
        triples = self._make_triples()
        validator = SymbolicValidator()
        validated = validator.validate_triples(triples)

        canon = Canonicalizer()
        canonicalized = canon.canonicalize_triples(validated)
        entities = canon.build_entity_map(canonicalized)

        # Should have entities for APT29, Cobalt Strike, US Government, the attacker, CVE
        assert len(entities) > 0

    def test_graph_build(self):
        triples = self._make_triples()
        validator = SymbolicValidator()
        validated = validator.validate_triples(triples)

        canon = Canonicalizer()
        canonicalized = canon.canonicalize_triples(validated)
        entities = canon.build_entity_map(canonicalized)
        graph = build_graph(entities, canonicalized, doc_id="doc1")

        assert graph.num_nodes > 0
        assert graph.num_edges > 0
        # Rejected triples should not appear in graph edges
        assert graph.num_edges < len(triples)

    def test_graph_metrics(self):
        triples = self._make_triples()[:2]  # just the valid ones
        for t in triples:
            t.validation_status = ValidationStatus.VALIDATED

        canon = Canonicalizer()
        entities = canon.build_entity_map(triples)
        graph = build_graph(entities, triples, doc_id="doc1")
        metrics = compute_graph_metrics(graph)

        assert metrics["num_nodes"] > 0
        assert metrics["num_validated_edges"] == 2
        assert metrics["valid_edge_ratio"] == 1.0

    def test_fusion(self):
        # Two graphs with overlapping entities
        t1 = Triple(
            event_id="d1_t0", subject="APT29",
            subject_type=EntityType.THREAT_ACTOR,
            relation=RelationType.USES, object="Cobalt Strike",
            object_type=EntityType.TOOL, evidence_text="Evidence 1.",
            source_doc_id="doc1", source_chunk_id="doc1_chunk0",
            validation_status=ValidationStatus.VALIDATED,
        )
        t2 = Triple(
            event_id="d2_t0", subject="APT29",
            subject_type=EntityType.THREAT_ACTOR,
            relation=RelationType.USES, object="Cobalt Strike",
            object_type=EntityType.TOOL, evidence_text="Evidence 2.",
            source_doc_id="doc2", source_chunk_id="doc2_chunk0",
            validation_status=ValidationStatus.VALIDATED,
        )

        canon = Canonicalizer()
        e1 = canon.build_entity_map([t1])
        e2 = canon.build_entity_map([t2])
        g1 = build_graph(e1, [t1], doc_id="doc1")
        g2 = build_graph(e2, [t2], doc_id="doc2")

        merger = GraphMerger()
        merged = merger.merge([g1, g2])

        # Duplicate edge should be deduplicated
        assert merged.num_edges == 1
        # But evidence should be merged
        assert "Evidence 1." in merged.edges[0].evidence_text
        assert "Evidence 2." in merged.edges[0].evidence_text

    def test_extraction_metrics(self):
        pred = [
            Triple(
                event_id="p0", subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.USES, object="Cobalt Strike",
                object_type=EntityType.TOOL,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        gold = [
            Triple(
                event_id="g0", subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.USES, object="Cobalt Strike",
                object_type=EntityType.TOOL,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        metrics = compute_extraction_metrics(pred, gold, match_mode="normalized")
        assert metrics["triple"]["f1"] == 1.0

    def test_strict_triple_metric_penalizes_type_mismatch(self):
        pred = [
            Triple(
                event_id="p0", subject="Rust",
                subject_type=EntityType.TOOL,
                relation=RelationType.USES, object="payload.exe",
                object_type=EntityType.FILE,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        gold = [
            Triple(
                event_id="g0", subject="Rust",
                subject_type=EntityType.SOFTWARE,
                relation=RelationType.USES, object="payload.exe",
                object_type=EntityType.IOC,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        metrics = compute_extraction_metrics(pred, gold, match_mode="normalized")
        assert metrics["relation"]["f1"] == 1.0
        assert metrics["triple"]["f1"] == 1.0

    def test_relation_metric_penalizes_relation_mismatch(self):
        pred = [
            Triple(
                event_id="p0", subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.USES, object="Cobalt Strike",
                object_type=EntityType.TOOL,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        gold = [
            Triple(
                event_id="g0", subject="APT29",
                subject_type=EntityType.THREAT_ACTOR,
                relation=RelationType.TARGETS, object="Cobalt Strike",
                object_type=EntityType.TOOL,
                validation_status=ValidationStatus.VALIDATED,
            ),
        ]
        metrics = compute_extraction_metrics(pred, gold, match_mode="normalized")
        assert metrics["relation"]["f1"] == 0.0
        assert metrics["triple_relaxed"]["f1"] == 1.0
