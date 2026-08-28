"""Symbolic validator and repairer for CTI triples (Module C).

This is the core neuro-symbolic layer. It applies hard rules, soft rules,
and repair rules to raw LLM extraction outputs. Every action is logged
to the error taxonomy for analysis.
"""

from __future__ import annotations

import re

from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.schema.type_constraints import TypeConstraints, SELF_LOOP_FORBIDDEN
from src.symbolic.ioc_rules import (
    validate_cve,
    validate_attack_id,
    validate_ipv4,
    validate_domain,
    validate_hash,
    detect_ioc_subtype,
)
from src.symbolic.error_logger import ErrorTaxonomyLogger, ErrorCategory
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Generic placeholder names that indicate low-quality extraction
GENERIC_PLACEHOLDERS = frozenset({
    "attacker", "the attacker", "attackers", "the attackers",
    "threat actor", "the threat actor", "actor",
    "victim", "the victim", "target", "the target",
    "server", "the server", "malware", "the malware",
    "tool", "the tool", "file", "the file",
    "unknown", "n/a", "none",
    # Pronouns, LLMs sometimes extract pronouns as entity names
    "they", "them", "their", "it", "its", "he", "she", "him", "her",
    "this", "these", "those", "we", "our", "the group", "the actor",
    "the campaign", "the malware family", "the vulnerability",
})
LOW_VALUE_OBJECT_PATTERNS = (
    "known cybercrime organizations",
    "cybercrime organizations",
    "various sectors",
    "multiple services",
    "single attack",
    "malicious activity",
)


class SymbolicValidator:
    """Validates and repairs raw LLM-extracted triples using symbolic rules."""

    def __init__(
        self,
        error_logger: ErrorTaxonomyLogger | None = None,
        reject_missing_evidence: bool = False,
        min_entity_name_length: int = 1,
        reject_unsupported_by_evidence: bool = True,
        min_confidence_threshold: float = 0.0,
    ):
        self.error_logger = error_logger or ErrorTaxonomyLogger()
        self.reject_missing_evidence = reject_missing_evidence
        self.min_entity_name_length = max(min_entity_name_length, 1)
        self.reject_unsupported_by_evidence = reject_unsupported_by_evidence
        self.min_confidence_threshold = min_confidence_threshold

    def validate_triples(self, triples: list[Triple]) -> list[Triple]:
        """Apply all validation rules to a list of triples.

        Returns the triples with updated validation_status fields.
        Rejected triples are kept in the list but marked as REJECTED.
        """
        results = []
        for triple in triples:
            validated = self._validate_single(triple)
            results.append(validated)

        # Deduplicate: overlapping chunks produce identical triples.
        # Keep highest-confidence version of each (subject, relation, object) tuple.
        results = self._deduplicate(results)

        # Log summary
        n_raw = sum(1 for t in results if t.validation_status == ValidationStatus.RAW)
        n_valid = sum(1 for t in results if t.validation_status == ValidationStatus.VALIDATED)
        n_repaired = sum(1 for t in results if t.validation_status == ValidationStatus.REPAIRED)
        n_rejected = sum(1 for t in results if t.validation_status == ValidationStatus.REJECTED)
        logger.info(
            f"Validation: {len(results)} triples -> "
            f"{n_valid} validated, {n_repaired} repaired, {n_rejected} rejected"
        )
        return results

    @staticmethod
    def _deduplicate(triples: list[Triple]) -> list[Triple]:
        """Remove duplicate triples from overlapping chunks, keeping highest confidence."""
        seen: dict[tuple, int] = {}  # dedup_key -> index in result list
        result: list[Triple] = []
        deduped = 0

        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                result.append(t)
                continue
            key = (
                t.subject.lower().strip(),
                t.relation.value,
                t.object.lower().strip(),
            )
            if key in seen:
                idx = seen[key]
                if t.confidence > result[idx].confidence:
                    result[idx] = t
                deduped += 1
            else:
                seen[key] = len(result)
                result.append(t)

        if deduped:
            logger.info(f"Deduplication removed {deduped} duplicate triples")
        return result

    def _validate_single(self, triple: Triple) -> Triple:
        """Apply all rules to a single triple, in execution order."""
        t = triple.model_copy()

        # 1. Empty field rejection
        if not t.subject.strip() or not t.object.strip():
            return self._reject(t, ErrorCategory.EMPTY_FIELD, "Empty subject or object")

        if not t.relation:
            return self._reject(t, ErrorCategory.EMPTY_FIELD, "Empty relation")

        if len(t.subject.strip()) < self.min_entity_name_length:
            return self._reject(t, ErrorCategory.EMPTY_FIELD, "Subject too short")
        if len(t.object.strip()) < self.min_entity_name_length:
            return self._reject(t, ErrorCategory.EMPTY_FIELD, "Object too short")

        # 2. Identifier format normalization and repair
        t = self._repair_identifiers(t)
        if t.validation_status == ValidationStatus.REJECTED:
            return t

        # 3. Entity type validation
        t = self._validate_entity_types(t)
        if t.validation_status == ValidationStatus.REJECTED:
            return t

        # 4. Semantic near-miss repair before hard type-pair rejection
        t = self._repair_semantic_near_miss(t)
        if t.validation_status == ValidationStatus.REJECTED:
            return t

        # 4. Relation type validation (already enforced by enum, but check type pairs)

        # 5. Type-pair constraint checking (with auto-swap repair)
        if not TypeConstraints.is_valid_type_pair(t.subject_type, t.relation, t.object_type):
            # Try swapping subject and object, LLMs often reverse the direction
            if TypeConstraints.is_valid_type_pair(t.object_type, t.relation, t.subject_type):
                old_subj, old_obj = t.subject, t.object
                old_stype, old_otype = t.subject_type, t.object_type
                t.subject, t.object = old_obj, old_subj
                t.subject_type, t.object_type = old_otype, old_stype
                t.validation_status = ValidationStatus.REPAIRED
                self.error_logger.log(
                    category=ErrorCategory.REPAIRED_IDENTIFIER,
                    action="repaired",
                    triple_id=t.event_id,
                    details=f"Swapped subject/object: ({old_subj}, {old_stype.value}) <-> ({old_obj}, {old_otype.value})",
                    source_doc_id=t.source_doc_id,
                    source_chunk_id=t.source_chunk_id,
                )
            else:
                reason = TypeConstraints.get_violation_reason(
                    t.subject_type, t.relation, t.object_type
                )
                return self._reject(
                    t, ErrorCategory.IMPOSSIBLE_TYPE_PAIR,
                    f"Invalid type pair: {reason}"
                )

        # 6. Self-loop check
        if (
            t.subject.lower().strip() == t.object.lower().strip()
            and t.relation in SELF_LOOP_FORBIDDEN
        ):
            return self._reject(
                t, ErrorCategory.SELF_LOOP,
                f"Self-loop on '{t.relation.value}'"
            )

        # 7. Missing evidence check
        if not t.evidence_text.strip():
            if self.reject_missing_evidence:
                return self._reject(
                    t, ErrorCategory.MISSING_EVIDENCE, "No evidence text provided"
                )
            self.error_logger.log(
                category=ErrorCategory.MISSING_EVIDENCE,
                action="flagged",
                triple_id=t.event_id,
                details="No evidence text provided",
                source_doc_id=t.source_doc_id,
                source_chunk_id=t.source_chunk_id,
            )
            t.confidence = max(t.confidence - 0.2, 0.1)

        # 8. Soft rules: generic placeholder detection
        t = self._check_generic_placeholders(t)
        if t.validation_status == ValidationStatus.REJECTED:
            return t

        # 9. Check evidence alignment against subject/object mentions
        t = self._check_evidence_alignment(t)
        if t.validation_status == ValidationStatus.REJECTED:
            return t

        # 10. Validate ATT&CK IDs if present
        t = self._validate_attack_ids(t)

        # 11. Low-confidence rejection, triples penalized below threshold are dropped
        if self.min_confidence_threshold > 0 and t.confidence <= self.min_confidence_threshold:
            return self._reject(
                t, ErrorCategory.CONFIDENCE_LOWERED,
                f"Confidence {t.confidence:.2f} below threshold {self.min_confidence_threshold}"
            )

        # If we got here without rejection or prior repair, mark as validated
        if t.validation_status == ValidationStatus.RAW:
            t.validation_status = ValidationStatus.VALIDATED

        return t

    def _repair_identifiers(self, triple: Triple) -> Triple:
        """Attempt to repair malformed identifiers (CVEs, IOCs)."""
        t = triple

        # Repair CVE format in subject or object
        for field in ["subject", "object"]:
            value = getattr(t, field)
            if "cve" in value.lower():
                is_valid, repaired = validate_cve(value)
                if is_valid and repaired and repaired != value:
                    setattr(t, field, repaired)
                    t.validation_status = ValidationStatus.REPAIRED
                    self.error_logger.log(
                        category=ErrorCategory.REPAIRED_IDENTIFIER,
                        action="repaired",
                        triple_id=t.event_id,
                        details=f"CVE repaired: '{value}' -> '{repaired}'",
                        source_doc_id=t.source_doc_id,
                        source_chunk_id=t.source_chunk_id,
                    )
                elif not is_valid:
                    self.error_logger.log(
                        category=ErrorCategory.MALFORMED_IDENTIFIER,
                        action="flagged",
                        triple_id=t.event_id,
                        details=f"Malformed CVE: '{value}'",
                        source_doc_id=t.source_doc_id,
                        source_chunk_id=t.source_chunk_id,
                    )

        # Validate IOC entities
        for field, type_field in [("subject", "subject_type"), ("object", "object_type")]:
            if getattr(t, type_field) == EntityType.IOC:
                value = getattr(t, field)
                subtype = detect_ioc_subtype(value)
                # Try to validate the specific IOC format
                if subtype.value == "ip":
                    valid, normalized = validate_ipv4(value)
                    if valid and normalized:
                        setattr(t, field, normalized)
                elif subtype.value == "domain":
                    valid, normalized = validate_domain(value)
                    if valid and normalized:
                        setattr(t, field, normalized)
                elif subtype.value == "hash":
                    valid, normalized = validate_hash(value)
                    if valid and normalized:
                        setattr(t, field, normalized)

        return t

    def _validate_entity_types(self, triple: Triple) -> Triple:
        """Check entity types and attempt to repair OTHER assignments."""
        for field, type_field in [("subject", "subject_type"), ("object", "object_type")]:
            if getattr(triple, type_field) == EntityType.OTHER:
                name = getattr(triple, field)
                repaired_type = _repair_other_type(name)
                if repaired_type is not None:
                    setattr(triple, type_field, repaired_type)
                    triple.validation_status = ValidationStatus.REPAIRED
                    self.error_logger.log(
                        category=ErrorCategory.REPAIRED_IDENTIFIER,
                        action="repaired",
                        triple_id=triple.event_id,
                        details=f"Type repair: '{name}' Other → {repaired_type.value}",
                        source_doc_id=triple.source_doc_id,
                        source_chunk_id=triple.source_chunk_id,
                    )
                else:
                    self.error_logger.log(
                        category=ErrorCategory.TYPE_MISASSIGNMENT,
                        action="flagged",
                        triple_id=triple.event_id,
                        details=f"{field} '{name}' typed as OTHER",
                        source_doc_id=triple.source_doc_id,
                        source_chunk_id=triple.source_chunk_id,
                    )
        return triple

    def _check_evidence_alignment(self, triple: Triple) -> Triple:
        """Reject or downweight triples whose evidence does not mention the entities."""
        evidence = triple.evidence_text.strip()
        if not evidence:
            return triple

        # Skip evidence check for very short entity names (≤3 chars), they match as substrings too often
        subj_skip = len(triple.subject.strip()) <= 3
        obj_skip = len(triple.object.strip()) <= 3

        subject_present = subj_skip or _text_supports_entity(evidence, triple.subject)
        object_present = obj_skip or _text_supports_entity(evidence, triple.object)

        if subject_present and object_present:
            return triple

        if not subject_present and not object_present:
            # Heavy penalty but don't reject, LLMs often paraphrase entity names
            # in evidence spans, so absence doesn't necessarily mean hallucination.
            triple.confidence = max(triple.confidence - 0.3, 0.1)
            self.error_logger.log(
                category=ErrorCategory.HALLUCINATED_ENTITY,
                action="flagged",
                triple_id=triple.event_id,
                details="Evidence does not mention subject or object (confidence penalized)",
                source_doc_id=triple.source_doc_id,
                source_chunk_id=triple.source_chunk_id,
            )
            return triple

        # Reduced penalty: -0.1 for partial matches (one entity present)
        triple.confidence = max(triple.confidence - 0.1, 0.1)
        self.error_logger.log(
            category=ErrorCategory.CONFIDENCE_LOWERED,
            action="flagged",
            triple_id=triple.event_id,
            details=(
                "Evidence weakly aligned: "
                f"subject_present={subject_present}, object_present={object_present}"
            ),
            source_doc_id=triple.source_doc_id,
            source_chunk_id=triple.source_chunk_id,
        )
        return triple

    def _check_generic_placeholders(self, triple: Triple) -> Triple:
        """Lower confidence for generic placeholder entity names."""
        for field in ["subject", "object"]:
            name = getattr(triple, field).lower().strip()
            if name in GENERIC_PLACEHOLDERS:
                triple.confidence = max(triple.confidence - 0.3, 0.1)
                self.error_logger.log(
                    category=ErrorCategory.GENERIC_PLACEHOLDER,
                    action="flagged",
                    triple_id=triple.event_id,
                    details=f"Generic placeholder: '{getattr(triple, field)}'",
                    source_doc_id=triple.source_doc_id,
                    source_chunk_id=triple.source_chunk_id,
                )
        if triple.object.lower().strip() in LOW_VALUE_OBJECT_PATTERNS:
            return self._reject(
                triple,
                ErrorCategory.GENERIC_PLACEHOLDER,
                f"Low-value generic object: '{triple.object}'",
            )
        return triple

    def _repair_semantic_near_miss(self, triple: Triple) -> Triple:
        """Repair common local-Mistral near-miss triples before type-pair rejection."""
        obj = triple.object.lower().strip()
        evidence = triple.evidence_text.lower()

        if triple.object_type == EntityType.OTHER:
            repaired = _repair_other_type(triple.object)
            if repaired is not None and repaired != triple.object_type:
                old_type = triple.object_type
                triple.object_type = repaired
                triple.validation_status = ValidationStatus.REPAIRED
                self.error_logger.log(
                    category=ErrorCategory.REPAIRED_IDENTIFIER,
                    action="repaired",
                    triple_id=triple.event_id,
                    details=f"Semantic type repair: '{triple.object}' {old_type.value} → {repaired.value}",
                    source_doc_id=triple.source_doc_id,
                    source_chunk_id=triple.source_chunk_id,
                )

        # Malware often targets/deletes VSS copies rather than communicates with them.
        if (
            triple.relation == RelationType.COMMUNICATES_WITH
            and ("shadow" in obj or "vss" in obj)
        ):
            triple.relation = RelationType.TARGETS
            triple.validation_status = ValidationStatus.REPAIRED
            self.error_logger.log(
                category=ErrorCategory.REPAIRED_IDENTIFIER,
                action="repaired",
                triple_id=triple.event_id,
                details="Repaired communicates_with on VSS/shadow-copy object to targets",
                source_doc_id=triple.source_doc_id,
                source_chunk_id=triple.source_chunk_id,
            )

        # Disclosure/reporting orgs are often mis-labeled as exploiting a vulnerability.
        if (
            triple.subject_type == EntityType.ORGANIZATION
            and triple.relation == RelationType.EXPLOITS
            and triple.object_type == EntityType.VULNERABILITY
        ):
            if any(hint in evidence for hint in ("disclos", "reported", "tracked as", "research", "researchers", "discovered", "identified", "observed")):
                triple.relation = RelationType.ASSOCIATED_WITH
                triple.validation_status = ValidationStatus.REPAIRED
                self.error_logger.log(
                    category=ErrorCategory.REPAIRED_IDENTIFIER,
                    action="repaired",
                    triple_id=triple.event_id,
                    details="Repaired Organization exploits Vulnerability to associated_with based on evidence",
                    source_doc_id=triple.source_doc_id,
                    source_chunk_id=triple.source_chunk_id,
                )

        # Generic target-like objects with weak evidence should be dropped early.
        if triple.relation in (RelationType.TARGETS, RelationType.COMMUNICATES_WITH) and obj in LOW_VALUE_OBJECT_PATTERNS:
            return self._reject(
                triple,
                ErrorCategory.GENERIC_PLACEHOLDER,
                f"Low-value object for relation '{triple.relation.value}': '{triple.object}'",
            )

        return triple

    def _validate_attack_ids(self, triple: Triple) -> Triple:
        """Validate ATT&CK technique IDs if present."""
        valid_ids = []
        for aid in triple.attack_ids:
            if validate_attack_id(aid):
                valid_ids.append(aid)
            else:
                self.error_logger.log(
                    category=ErrorCategory.MALFORMED_IDENTIFIER,
                    action="flagged",
                    triple_id=triple.event_id,
                    details=f"Invalid ATT&CK ID removed: '{aid}'",
                    source_doc_id=triple.source_doc_id,
                    source_chunk_id=triple.source_chunk_id,
                )
        triple.attack_ids = valid_ids
        return triple

    def _reject(self, triple: Triple, category: str, reason: str) -> Triple:
        """Mark a triple as rejected and log the reason."""
        triple.validation_status = ValidationStatus.REJECTED
        triple.rejection_reason = reason
        self.error_logger.log(
            category=category,
            action="rejected",
            triple_id=triple.event_id,
            details=reason,
            source_doc_id=triple.source_doc_id,
            source_chunk_id=triple.source_chunk_id,
        )
        return triple


_CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
_IP_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_HASH_RE = re.compile(r"^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$")
_DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?\.[a-z]{2,}$", re.IGNORECASE)
_FILE_EXT_RE = re.compile(r"\.(exe|dll|ps1|bat|doc|docx|xls|xlsx|pdf|js|vbs|hta|lnk|scr|msi|iso|zip|rar|7z)$", re.IGNORECASE)

_SOFTWARE_NAMES = frozenset({
    "windows", "linux", "macos", "mac os", "ios", "android",
    "chrome", "firefox", "safari", "edge", "opera",
    "microsoft office", "outlook", "exchange", "sharepoint",
    "python", "java", "rust", "c++", "c#", ".net",
    "vmware", "citrix", "oracle", "sap", "cisco",
    "adobe", "wordpress", "apache", "nginx", "iis",
})

_THREAT_ACTOR_HINTS = frozenset({
    "apt", "group", "gang", "hacker", "team", "bear", "panda",
    "kitten", "spider", "typhoon", "sandworm", "lazarus",
})

_CAMPAIGN_HINTS = frozenset({
    "campaign", "operation", "project",
})

_TECHNIQUE_RE = re.compile(r"^T\d{4}(\.\d{3})?$")

_TECHNIQUE_NAMES = frozenset({
    "credential dumping", "spear phishing", "spearphishing", "lateral movement",
    "privilege escalation", "dll sideloading", "dll side-loading", "dll injection",
    "process injection", "command and control", "data exfiltration", "keylogging",
    "screen capture", "brute force", "pass the hash", "pass the ticket",
    "phishing", "watering hole", "supply chain", "code signing",
    "obfuscation", "rootkit", "living off the land", "lolbin",
    "registry modification", "scheduled task", "persistence",
    "defense evasion", "initial access", "execution", "discovery",
    "collection", "impact", "resource development",
    "social engineering", "drive-by compromise", "exploit public-facing",
})
_CRYPTO_TECHNIQUE_HINTS = (
    "aes",
    "rsa",
    "ctr",
    "cbc",
    "gcm",
    "encryption",
    "cipher",
    "key exchange",
)


def _repair_other_type(name: str) -> EntityType | None:
    """Try to infer a better entity type for an Other-typed entity."""
    if not name:
        return None
    lower = name.lower().strip()

    # CVE pattern → Vulnerability
    if _CVE_RE.search(name):
        return EntityType.VULNERABILITY

    # IP, hash, domain → IOC
    if _IP_RE.fullmatch(name) or _HASH_RE.fullmatch(name) or _DOMAIN_RE.fullmatch(lower):
        return EntityType.IOC

    # File extension → File (which is IOC-adjacent)
    if _FILE_EXT_RE.search(lower):
        return EntityType.IOC

    # ATT&CK technique ID pattern (T1234 or T1234.567)
    if _TECHNIQUE_RE.fullmatch(name.strip()):
        return EntityType.TECHNIQUE

    # Known technique names
    if lower in _TECHNIQUE_NAMES or any(tn in lower for tn in _TECHNIQUE_NAMES if len(tn) > 8):
        return EntityType.TECHNIQUE

    if any(hint in lower for hint in _CRYPTO_TECHNIQUE_HINTS):
        return EntityType.TECHNIQUE

    if "extension" in lower or lower.startswith("."):
        return EntityType.FILE

    # Campaign hints (check before software to avoid "opera" matching "operation")
    for hint in _CAMPAIGN_HINTS:
        if hint in lower:
            return EntityType.CAMPAIGN

    # Threat actor hints
    for hint in _THREAT_ACTOR_HINTS:
        if hint in lower:
            return EntityType.THREAT_ACTOR

    # Known software/platform names (word-boundary match)
    lower_tokens = set(lower.split())
    for sw in _SOFTWARE_NAMES:
        sw_tokens = set(sw.split())
        if sw_tokens & lower_tokens:
            return EntityType.SOFTWARE

    return None


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _text_supports_entity(text: str, entity: str) -> bool:
    evidence = _normalize_text(text)
    candidate = _normalize_text(entity)
    if not evidence or not candidate:
        return False
    if candidate in evidence:
        return True

    entity_tokens = candidate.split()
    evidence_tokens = set(evidence.split())
    if len(entity_tokens) == 1:
        return entity_tokens[0] in evidence_tokens

    overlap = sum(1 for tok in entity_tokens if tok in evidence_tokens)
    return overlap / len(entity_tokens) >= 0.6
