"""Tests for relation repair and reranking."""

from src.schema.entities import EntityType
from src.schema.relations import RelationType, Triple, ValidationStatus
from src.symbolic.relation_repair import RelationRepairer


def _make_triple(
    relation: RelationType = RelationType.RELATED_TO,
    evidence: str = "APT29 used Cobalt Strike during the intrusion.",
    object_type: EntityType = EntityType.TOOL,
) -> Triple:
    return Triple(
        event_id="t0",
        subject="APT29",
        subject_type=EntityType.THREAT_ACTOR,
        relation=relation,
        object="Cobalt Strike",
        object_type=object_type,
        evidence_text=evidence,
        source_doc_id="doc1",
        source_chunk_id="doc1_chunk0",
        validation_status=ValidationStatus.VALIDATED,
    )


def test_related_to_is_promoted_to_uses():
    repairer = RelationRepairer()
    repaired = repairer.repair_triples([_make_triple()])
    assert repaired[0].relation == RelationType.USES


def test_associated_with_is_promoted_to_exploits():
    repairer = RelationRepairer()
    triple = _make_triple(
        relation=RelationType.ASSOCIATED_WITH,
        evidence="APT29 exploited CVE-2023-12345 in the attack.",
        object_type=EntityType.VULNERABILITY,
    )
    triple.object = "CVE-2023-12345"
    repaired = repairer.repair_triples([triple])
    assert repaired[0].relation == RelationType.ASSOCIATED_WITH


def test_associated_with_can_be_promoted_when_enabled():
    repairer = RelationRepairer(promote_associated_with=True)
    triple = _make_triple(
        relation=RelationType.ASSOCIATED_WITH,
        evidence="APT29 exploited CVE-2023-12345 in the attack.",
        object_type=EntityType.VULNERABILITY,
    )
    triple.object = "CVE-2023-12345"
    repaired = repairer.repair_triples([triple])
    assert repaired[0].relation == RelationType.EXPLOITS


def test_related_to_is_not_promoted_without_entity_mentions_in_evidence():
    repairer = RelationRepairer(require_entity_mentions=True)
    triple = _make_triple(
        evidence="The actor used the tool during the intrusion.",
    )
    repaired = repairer.repair_triples([triple])
    assert repaired[0].relation == RelationType.RELATED_TO
