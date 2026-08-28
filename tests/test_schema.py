"""Tests for schema definitions."""

import pytest
from src.schema.entities import Entity, EntityType, IOCSubtype
from src.schema.relations import Triple, RelationType, ValidationStatus, normalize_relation
from src.schema.type_constraints import TypeConstraints
from src.schema.graph_schema import CTIGraph


class TestEntityType:
    def test_all_types_exist(self):
        assert len(EntityType) == 14

    def test_make_node_id(self):
        nid = Entity.make_node_id(EntityType.MALWARE, "PlugX")
        assert nid == "malware:plugx"

    def test_make_node_id_with_spaces(self):
        nid = Entity.make_node_id(EntityType.THREAT_ACTOR, "Mustang Panda")
        assert nid == "threatactor:mustang_panda"

    def test_make_node_id_strips_punctuation(self):
        nid = Entity.make_node_id(EntityType.MALWARE, "PlugX (variant)")
        assert "(" not in nid
        assert ")" not in nid


class TestRelationType:
    def test_all_types_exist(self):
        assert len(RelationType) == 12

    def test_normalize_exact(self):
        assert normalize_relation("uses") == RelationType.USES
        assert normalize_relation("targets") == RelationType.TARGETS

    def test_normalize_variant(self):
        assert normalize_relation("exploited") == RelationType.EXPLOITS
        assert normalize_relation("is targeting") == RelationType.TARGETS

    def test_normalize_fallback(self):
        assert normalize_relation("some random relation") == RelationType.RELATED_TO


class TestTypeConstraints:
    def test_valid_uses(self):
        assert TypeConstraints.is_valid_type_pair(
            EntityType.THREAT_ACTOR, RelationType.USES, EntityType.MALWARE
        )

    def test_invalid_uses(self):
        assert not TypeConstraints.is_valid_type_pair(
            EntityType.VULNERABILITY, RelationType.USES, EntityType.LOCATION
        )

    def test_exploits_requires_vulnerability(self):
        assert TypeConstraints.is_valid_type_pair(
            EntityType.MALWARE, RelationType.EXPLOITS, EntityType.VULNERABILITY
        )
        assert not TypeConstraints.is_valid_type_pair(
            EntityType.MALWARE, RelationType.EXPLOITS, EntityType.ORGANIZATION
        )

    def test_associated_with_unconstrained(self):
        assert TypeConstraints.is_valid_type_pair(
            EntityType.OTHER, RelationType.ASSOCIATED_WITH, EntityType.OTHER
        )

    def test_self_loop_forbidden(self):
        assert not TypeConstraints.is_self_loop_allowed(RelationType.USES)
        assert not TypeConstraints.is_self_loop_allowed(RelationType.TARGETS)

    def test_violation_reason(self):
        reason = TypeConstraints.get_violation_reason(
            EntityType.VULNERABILITY, RelationType.USES, EntityType.LOCATION
        )
        assert reason is not None
        assert "not valid" in reason


class TestCTIGraph:
    def test_add_entity(self):
        graph = CTIGraph()
        entity = Entity(
            node_id="malware:plugx",
            canonical_name="PlugX",
            entity_type=EntityType.MALWARE,
        )
        graph.add_entity(entity)
        assert graph.num_nodes == 1

    def test_add_duplicate_entity_merges(self):
        graph = CTIGraph()
        e1 = Entity(
            node_id="malware:plugx",
            canonical_name="PlugX",
            entity_type=EntityType.MALWARE,
            aliases=["Korplug"],
            source_mentions=["doc1:chunk0"],
        )
        e2 = Entity(
            node_id="malware:plugx",
            canonical_name="PlugX",
            entity_type=EntityType.MALWARE,
            aliases=["PKPLUG"],
            source_mentions=["doc2:chunk1"],
        )
        graph.add_entity(e1)
        graph.add_entity(e2)
        assert graph.num_nodes == 1
        assert "Korplug" in graph.nodes["malware:plugx"].aliases
        assert "PKPLUG" in graph.nodes["malware:plugx"].aliases

    def test_summary(self):
        graph = CTIGraph()
        triple = Triple(
            subject="APT29",
            subject_type=EntityType.THREAT_ACTOR,
            relation=RelationType.USES,
            object="Cobalt Strike",
            object_type=EntityType.TOOL,
            validation_status=ValidationStatus.VALIDATED,
        )
        graph.add_triple(triple)
        summary = graph.summary()
        assert summary["total_edges"] == 1
        assert "validated" in summary["edge_status"]
