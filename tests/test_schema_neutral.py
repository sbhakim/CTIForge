"""Tests for schema-neutral (external-standard) evaluation.

Guards the property that makes these metrics usable for cross-system comparison:
scoring must depend only on the STIX 2.1 table, never on CTIForge's own
type_constraints.py. A triple that CTIForge's validator accepts must still be
able to fail STIX scoring, otherwise the metric is circular.
"""

import pytest

from src.evaluation.schema_neutral import (
    STIX_RELATIONSHIP_CONSTRAINTS,
    CTIFORGE_TO_STIX,
    evidence_presence_rate,
    normalization_loss,
    stix_compliance,
)
from src.schema.type_constraints import TypeConstraints
from src.schema.entities import EntityType
from src.schema.relations import RelationType


def _d(stype, rel, otype, evidence="ev"):
    return {"subject_type": stype, "relation": rel, "object_type": otype,
            "evidence_text": evidence}


class TestNotCircular:
    """The whole point: STIX scoring must be independent of CTIForge's table."""

    def test_ctiforge_valid_triple_can_fail_stix(self):
        # CTIForge's table permits Vulnerability -[exploits]-> Vulnerability.
        # STIX does not (exploits targets vulnerability, but a vulnerability is
        # not a valid source). Observed 25x in the real 149-doc cache.
        assert TypeConstraints.is_valid_type_pair(
            EntityType.VULNERABILITY, RelationType.EXPLOITS, EntityType.VULNERABILITY
        )
        result = stix_compliance([_d("Vulnerability", "exploits", "Vulnerability")])
        assert result["scored_triples"] == 1
        assert result["compliant_triples"] == 0
        assert result["stix_compliance"] == 0.0

    def test_stix_table_is_not_the_ctiforge_table(self):
        """Sanity: the two tables must disagree somewhere, or this is theatre."""
        disagreements = 0
        for rel_name in STIX_RELATIONSHIP_CONSTRAINTS:
            rel = RelationType(rel_name)
            for st in EntityType:
                for ot in EntityType:
                    s_stix = CTIFORGE_TO_STIX.get(st.value)
                    o_stix = CTIFORGE_TO_STIX.get(ot.value)
                    if s_stix is None or o_stix is None:
                        continue
                    src, tgt = STIX_RELATIONSHIP_CONSTRAINTS[rel_name]
                    stix_ok = s_stix in src and o_stix in tgt
                    sg_ok = TypeConstraints.is_valid_type_pair(st, rel, ot)
                    if stix_ok != sg_ok:
                        disagreements += 1
        assert disagreements > 0, "STIX table mirrors CTIForge's -- metric is circular"


class TestStixCompliance:
    def test_compliant_triple_scores_one(self):
        r = stix_compliance([_d("ThreatActor", "uses", "Malware")])
        assert r["stix_compliance"] == 1.0
        assert r["scored_triples"] == 1

    def test_generic_relations_are_skipped_not_penalised(self):
        """related_to has no STIX domain/range; counting it as a violation would
        penalise a system twice for one normalisation failure."""
        r = stix_compliance([_d("ThreatActor", "related_to", "Malware")])
        assert r["scored_triples"] == 0
        assert r["skipped_no_stix_relation"] == 1

    def test_unmappable_type_is_skipped(self):
        r = stix_compliance([_d("Other", "uses", "Malware")])
        assert r["scored_triples"] == 0
        assert r["skipped_unmappable_type"] == 1

    def test_coverage_reports_scored_fraction(self):
        triples = [
            _d("ThreatActor", "uses", "Malware"),        # scoreable
            _d("ThreatActor", "related_to", "Malware"),  # skipped
        ]
        r = stix_compliance(triples)
        assert r["coverage"] == pytest.approx(0.5)

    def test_violations_are_reported_for_diagnosis(self):
        r = stix_compliance([_d("Malware", "targets", "Malware")])
        assert r["top_violations"]
        assert "malware" in r["top_violations"][0][0]

    def test_empty_input(self):
        r = stix_compliance([])
        assert r["stix_compliance"] == 0.0
        assert r["scored_triples"] == 0


class TestNormalizationLoss:
    def test_generic_rate(self):
        triples = [
            _d("ThreatActor", "related_to", "Malware"),
            _d("ThreatActor", "associated_with", "Malware"),
            _d("ThreatActor", "uses", "Malware"),
            _d("ThreatActor", "targets", "Organization"),
        ]
        r = normalization_loss(triples)
        assert r["generic_relation_rate"] == pytest.approx(0.5)

    def test_other_type_rate_counts_both_endpoints(self):
        r = normalization_loss([_d("Other", "uses", "Other")])
        assert r["other_type_rate"] == pytest.approx(1.0)
        r2 = normalization_loss([_d("Other", "uses", "Malware")])
        assert r2["other_type_rate"] == pytest.approx(0.5)

    def test_empty_input(self):
        assert normalization_loss([])["n"] == 0


class TestEvidencePresence:
    def test_objective_and_schema_free(self):
        assert evidence_presence_rate([_d("ThreatActor", "uses", "Malware", "sentence")]) == 1.0
        assert evidence_presence_rate([_d("ThreatActor", "uses", "Malware", "")]) == 0.0
        assert evidence_presence_rate([_d("ThreatActor", "uses", "Malware", "   ")]) == 0.0

    def test_empty_input(self):
        assert evidence_presence_rate([]) == 0.0
