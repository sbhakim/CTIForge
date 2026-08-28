"""Tests for symbolic validator (Module C)."""

import pytest
from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.symbolic.validator import SymbolicValidator
from src.symbolic.error_logger import ErrorTaxonomyLogger, ErrorCategory
from src.symbolic.ioc_rules import (
    validate_cve,
    validate_attack_id,
    validate_ipv4,
    validate_domain,
    validate_hash,
    detect_ioc_subtype,
)
from src.schema.entities import IOCSubtype


class TestIOCRules:
    def test_valid_cve(self):
        valid, repaired = validate_cve("CVE-2023-12345")
        assert valid
        assert repaired == "CVE-2023-12345"

    def test_repair_cve_lowercase(self):
        valid, repaired = validate_cve("cve-2023-12345")
        assert valid
        assert repaired == "CVE-2023-12345"

    def test_repair_cve_spacing(self):
        valid, repaired = validate_cve("CVE 2023 12345")
        assert valid
        assert repaired == "CVE-2023-12345"

    def test_invalid_cve(self):
        valid, repaired = validate_cve("not-a-cve")
        assert not valid

    def test_valid_attack_id(self):
        assert validate_attack_id("T1059")
        assert validate_attack_id("T1059.001")
        assert validate_attack_id("TA0001")
        assert validate_attack_id("S0013")

    def test_invalid_attack_id(self):
        assert not validate_attack_id("T10")
        assert not validate_attack_id("TXYZ")
        assert not validate_attack_id("random")

    def test_valid_ipv4(self):
        valid, normalized = validate_ipv4("192.168.1.1")
        assert valid
        assert normalized == "192.168.1.1"

    def test_ipv4_strip_leading_zeros(self):
        valid, normalized = validate_ipv4("192.168.001.001")
        assert valid
        assert normalized == "192.168.1.1"

    def test_invalid_ipv4(self):
        valid, _ = validate_ipv4("999.999.999.999")
        assert not valid

    def test_valid_domain(self):
        valid, normalized = validate_domain("evil.example.com")
        assert valid
        assert normalized == "evil.example.com"

    def test_defanged_domain(self):
        valid, normalized = validate_domain("evil[.]example[.]com")
        assert valid

    def test_valid_hash_md5(self):
        valid, _ = validate_hash("d41d8cd98f00b204e9800998ecf8427e")
        assert valid

    def test_valid_hash_sha256(self):
        valid, _ = validate_hash(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert valid

    def test_detect_ioc_subtype_ip(self):
        assert detect_ioc_subtype("192.168.1.1") == IOCSubtype.IP

    def test_detect_ioc_subtype_domain(self):
        assert detect_ioc_subtype("evil.example.com") == IOCSubtype.DOMAIN

    def test_detect_ioc_subtype_url(self):
        assert detect_ioc_subtype("https://evil.com/payload") == IOCSubtype.URL

    def test_detect_ioc_subtype_email(self):
        assert detect_ioc_subtype("attacker@evil.com") == IOCSubtype.EMAIL

    def test_detect_ioc_subtype_hash(self):
        assert detect_ioc_subtype("d41d8cd98f00b204e9800998ecf8427e") == IOCSubtype.HASH


class TestSymbolicValidator:
    def _make_triple(self, **kwargs) -> Triple:
        defaults = {
            "event_id": "test_t0",
            "subject": "APT29",
            "subject_type": EntityType.THREAT_ACTOR,
            "relation": RelationType.USES,
            "object": "Cobalt Strike",
            "object_type": EntityType.TOOL,
            "evidence_text": "APT29 uses Cobalt Strike for C2.",
            "source_doc_id": "doc1",
            "source_chunk_id": "doc1_chunk0",
        }
        defaults.update(kwargs)
        return Triple(**defaults)

    def test_valid_triple_passes(self):
        validator = SymbolicValidator()
        triple = self._make_triple()
        results = validator.validate_triples([triple])
        assert results[0].validation_status == ValidationStatus.VALIDATED

    def test_empty_subject_rejected(self):
        validator = SymbolicValidator()
        triple = self._make_triple(subject="")
        results = validator.validate_triples([triple])
        assert results[0].validation_status == ValidationStatus.REJECTED

    def test_impossible_type_pair_rejected(self):
        validator = SymbolicValidator()
        triple = self._make_triple(
            subject_type=EntityType.VULNERABILITY,
            relation=RelationType.USES,
            object_type=EntityType.LOCATION,
        )
        results = validator.validate_triples([triple])
        assert results[0].validation_status == ValidationStatus.REJECTED
        assert "type pair" in results[0].rejection_reason.lower()

    def test_self_loop_rejected(self):
        validator = SymbolicValidator()
        triple = self._make_triple(subject="APT29", object="APT29")
        results = validator.validate_triples([triple])
        assert results[0].validation_status == ValidationStatus.REJECTED

    def test_generic_placeholder_lowers_confidence(self):
        validator = SymbolicValidator()
        triple = self._make_triple(subject="the attacker")
        results = validator.validate_triples([triple])
        assert results[0].confidence < 1.0

    def test_missing_evidence_lowers_confidence(self):
        validator = SymbolicValidator()
        triple = self._make_triple(evidence_text="")
        results = validator.validate_triples([triple])
        assert results[0].confidence < 1.0

    def test_cve_repair(self):
        validator = SymbolicValidator()
        triple = self._make_triple(
            subject="cve-2023-12345",
            subject_type=EntityType.VULNERABILITY,
            relation=RelationType.EXPLOITS,
            object="APT29",
            object_type=EntityType.THREAT_ACTOR,
        )
        # This will be rejected for type pair (vuln exploits actor is invalid)
        # Let's fix the type pair to test CVE repair
        triple2 = self._make_triple(
            subject="APT29",
            subject_type=EntityType.THREAT_ACTOR,
            relation=RelationType.EXPLOITS,
            object="cve 2023 12345",
            object_type=EntityType.VULNERABILITY,
        )
        results = validator.validate_triples([triple2])
        assert results[0].object == "CVE-2023-12345"
        assert results[0].validation_status == ValidationStatus.REPAIRED

    def test_error_logger_populated(self):
        error_logger = ErrorTaxonomyLogger()
        validator = SymbolicValidator(error_logger=error_logger)
        triples = [
            self._make_triple(subject=""),
            self._make_triple(
                subject_type=EntityType.VULNERABILITY,
                relation=RelationType.USES,
                object_type=EntityType.LOCATION,
            ),
            self._make_triple(subject="the attacker"),
        ]
        validator.validate_triples(triples)
        summary = error_logger.get_summary()
        assert summary["total_entries"] > 0
        assert "by_category" in summary

    def test_evidence_mismatch_penalized(self):
        """When evidence mentions neither entity, confidence is penalized but triple is kept."""
        validator = SymbolicValidator(reject_unsupported_by_evidence=True)
        triple = self._make_triple(
            subject="APT29",
            object="Cobalt Strike",
            evidence_text="The campaign hit multiple victims without naming the tool.",
        )
        results = validator.validate_triples([triple])
        assert results[0].validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
        assert results[0].confidence < 1.0

    def test_partial_evidence_match_lowers_confidence(self):
        validator = SymbolicValidator(reject_unsupported_by_evidence=True)
        triple = self._make_triple(
            subject="APT29",
            object="Cobalt Strike",
            evidence_text="APT29 was observed during the intrusion.",
        )
        results = validator.validate_triples([triple])
        assert results[0].validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)
        assert results[0].confidence < 1.0
