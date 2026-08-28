"""Lightweight relation reranking and repair for weak extracted edges."""

from __future__ import annotations

from src.schema.entities import EntityType
from src.schema.relations import RelationType, Triple, ValidationStatus
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_USES_HINTS = ("used", "using", "leverag", "employ", "written in", "built with", "utiliz", "execut", "ran ", "runs ", "running", "psexec", "powershell", "cobalt strike")
_EXPLOITS_HINTS = ("exploit", "abuse", "vulnerability", "cve-", "take advantage", "zero-day", "0-day")
_TARGETS_HINTS = ("target", "victim", "against", "encrypt", "steal", "exfiltrat", "compromis", "affect", "attack", "infect", "breach", "intrusion into", "directed at", "stop", "shut", "disabl", "wipe", "destroy", "delet", "disrupt", "lock", "render", "hit ", "victimized", "victims in", "targets in")
_DELIVERS_HINTS = ("deliver", "distribut", "payload", "dropper", "sent", "propagat", "spread", "phish")
_DROPS_HINTS = ("drop", "download", "deploy", "install", "load", "write to disk", "side-load", "sideload")
_COMM_HINTS = ("communicat", "beacon", "contact", "connect", "c2", "command and control", "callback", "phone home", "resolv")
_ATTR_HINTS = ("attribut", "linked to", "behind", "operated by", "created by", "developed by", "responsible for", "traced to", "associated with the group")
_LOC_HINTS = ("located in", "based in", "hosted in", "operates in", "originat")
_MITIGATE_HINTS = ("patch", "fix", "mitigat", "address", "remediat", "update")
_ASSOC_HINTS = ("linked to", "associated with", "connected to", "related to", "observed", "seen", "discovered", "confirmed", "classified", "described as")
_VARIANT_HINTS = ("variant", "family", "type of", "rebrand", "successor", "evolved from", "fork of", "based on",
                   "is a ", "classified as", "categorized as", "form of", "known as", "also called",
                   "renamed", "rebranded", "previously known", "formerly")


class RelationRepairer:
    """Repairs weak relations using evidence text and entity-type constraints."""

    def __init__(
        self,
        promote_associated_with: bool = False,
        require_entity_mentions: bool = True,
    ):
        self.promote_associated_with = promote_associated_with
        self.require_entity_mentions = require_entity_mentions

    def repair_triples(self, triples: list[Triple]) -> list[Triple]:
        repaired: list[Triple] = []
        changes = 0
        for triple in triples:
            if triple.validation_status == ValidationStatus.REJECTED:
                repaired.append(triple)
                continue

            updated = triple.model_copy()
            new_relation = self._suggest_relation(updated)
            if new_relation and new_relation != updated.relation:
                updated.relation = new_relation
                if updated.validation_status == ValidationStatus.VALIDATED:
                    updated.validation_status = ValidationStatus.REPAIRED
                changes += 1
            repaired.append(updated)

        if changes:
            logger.info("Relation repair updated %s triples", changes)
        return repaired

    def _suggest_relation(self, triple: Triple) -> RelationType | None:
        evidence = (triple.evidence_text or "").lower()
        relation = triple.relation

        if self.require_entity_mentions and evidence and not self._evidence_mentions_entities(triple, evidence):
            return relation

        # Fix semantically wrong relations based on entity type constraints
        corrected = self._fix_wrong_relation(triple, evidence)
        if corrected and corrected != relation:
            return corrected

        if relation == RelationType.ASSOCIATED_WITH:
            if self.promote_associated_with:
                promoted = self._relation_from_evidence(triple, evidence, strong_only=True)
                return promoted or relation
            return relation

        if relation != RelationType.RELATED_TO:
            fallback = self._fix_weak_relation(triple, evidence)
            return fallback or relation

        promoted = self._relation_from_evidence(triple, evidence, strong_only=False)
        if promoted:
            return promoted

        # Last resort: type-driven inference when evidence gives no signal
        type_inferred = self._infer_from_types(triple)
        return type_inferred or relation

    def _fix_weak_relation(self, triple: Triple, evidence: str) -> RelationType | None:
        """Promote weak free-form local relations into the schema even when not related_to."""
        relation_text = triple.relation.value.lower()

        if self._contains_any(relation_text, ("victim_of", "impacted_by")):
            return RelationType.TARGETS
        if self._contains_any(relation_text, ("targeted_by", "attacked_by", "infected_by", "encrypted_by", "compromised_by")):
            return RelationType.TARGETS
        if self._contains_any(relation_text, ("linked", "connected", "associated", "confirmed", "classified", "described")):
            if triple.subject_type in (EntityType.MALWARE, EntityType.CAMPAIGN) and triple.object_type in (EntityType.MALWARE, EntityType.CAMPAIGN):
                return RelationType.VARIANT_OF
            return RelationType.ASSOCIATED_WITH
        if self._contains_any(relation_text, ("employ", "switch", "written_in", "built_with", "uses")):
            return RelationType.USES
        if self._contains_any(relation_text, ("used_by", "deployed_by")):
            return RelationType.USES
        if self._contains_any(relation_text, ("operated_by", "run_by")):
            return RelationType.ATTRIBUTED_TO
        if self._contains_any(relation_text, ("deployed", "deploy", "drops", "drop")):
            if triple.object_type in (EntityType.FILE, EntityType.MALWARE, EntityType.TOOL):
                return RelationType.DROPS
            return RelationType.USES
        if self._contains_any(relation_text, ("first_appeared", "appeared", "emerged", "came_to_public_attention", "remained")):
            return RelationType.ASSOCIATED_WITH
        if self._contains_any(relation_text, ("targeted", "attacked", "impacted", "gains_access", "gain_access")):
            return RelationType.TARGETS

        if evidence and self._contains_any(evidence, _ASSOC_HINTS):
            return RelationType.ASSOCIATED_WITH
        return None

    def _fix_wrong_relation(self, triple: Triple, evidence: str) -> RelationType | None:
        """Fix semantically wrong relations based on entity type patterns.

        Catches common LLM errors where the relation doesn't match the
        entity types involved.
        """
        subj_type = triple.subject_type
        obj_type = triple.object_type
        relation = triple.relation

        # Organization + exploits + Vulnerability → likely mitigated_by (org patches vuln)
        # unless evidence clearly says they exploited it
        if (relation == RelationType.EXPLOITS
                and subj_type == EntityType.ORGANIZATION
                and obj_type == EntityType.VULNERABILITY):
            if self._contains_any(evidence, ("disclos", "reported", "tracked as", "research", "researchers", "discovered", "identified", "observed")):
                return RelationType.ASSOCIATED_WITH
            if self._contains_any(evidence, _MITIGATE_HINTS + ("patch", "fix", "address", "update", "release")):
                return RelationType.MITIGATED_BY
            # If no exploit evidence either, likely associated_with (org discovered/reported it)
            if not self._contains_any(evidence, ("exploit", "abuse", "attack", "compromise")):
                return RelationType.ASSOCIATED_WITH

        # Organization + exploits + Vulnerability where org is a research/security group
        # → likely associated_with (they discovered/analyzed it)
        if (relation == RelationType.EXPLOITS
                and subj_type == EntityType.ORGANIZATION
                and obj_type == EntityType.VULNERABILITY
                and self._contains_any(evidence, ("discover", "found", "report", "analyz", "identif", "observ", "track"))):
            return RelationType.ASSOCIATED_WITH

        # exploits used too broadly, if object is NOT a Vulnerability/Software/Infrastructure,
        # downgrade to uses or targets based on evidence
        if (relation == RelationType.EXPLOITS
                and obj_type not in (EntityType.VULNERABILITY, EntityType.SOFTWARE, EntityType.INFRASTRUCTURE)):
            if self._contains_any(evidence, _TARGETS_HINTS):
                return RelationType.TARGETS
            return RelationType.USES

        # Vulnerability + exploits + non-Vulnerability → likely targets (vuln affects software)
        if (relation == RelationType.EXPLOITS
                and subj_type == EntityType.VULNERABILITY
                and obj_type not in (EntityType.VULNERABILITY,)):
            if self._contains_any(evidence, _TARGETS_HINTS + ("affect", "impact")):
                return RelationType.TARGETS
            return RelationType.ASSOCIATED_WITH

        # associated_with / related_to where subject is malware/campaign/tool
        # and object is threat_actor → attributed_to
        if (relation in (RelationType.ASSOCIATED_WITH, RelationType.RELATED_TO)
                and subj_type in (EntityType.MALWARE, EntityType.CAMPAIGN, EntityType.TOOL, EntityType.THREAT_ACTOR)
                and obj_type == EntityType.THREAT_ACTOR):
            if self._contains_any(evidence, _ATTR_HINTS + ("linked", "tied", "connected", "operated", "backed")):
                return RelationType.ATTRIBUTED_TO
            # Even without evidence keywords, if subject is malware/campaign
            # and object is threat_actor, attributed_to is the right relation
            if subj_type in (EntityType.MALWARE, EntityType.CAMPAIGN):
                return RelationType.ATTRIBUTED_TO

        # Reverse: threat_actor + associated_with/related_to + malware/tool → uses
        if (relation in (RelationType.ASSOCIATED_WITH, RelationType.RELATED_TO)
                and subj_type == EntityType.THREAT_ACTOR
                and obj_type in (EntityType.MALWARE, EntityType.TOOL, EntityType.TECHNIQUE)):
            if self._contains_any(evidence, _USES_HINTS + ("deploy", "wield", "operate")):
                return RelationType.USES
            # Default: threat actors use their tools/malware
            if obj_type in (EntityType.MALWARE, EntityType.TOOL):
                return RelationType.USES

        # Vulnerability/Malware + related_to/exploits + Technique/Other → variant_of
        # Gold pattern: "CVE-X is a WebKit confusion issue", "Agent Tesla is a RAT"
        if (relation in (RelationType.RELATED_TO, RelationType.EXPLOITS, RelationType.ASSOCIATED_WITH)
                and subj_type in (EntityType.VULNERABILITY, EntityType.MALWARE)
                and obj_type in (EntityType.TECHNIQUE, EntityType.OTHER, EntityType.MALWARE)):
            if self._contains_any(evidence, _VARIANT_HINTS):
                return RelationType.VARIANT_OF

        # Campaign/Infrastructure + delivers + Malware → delivers (keep, but also
        # fix related_to → delivers when evidence has delivery hints)
        if (relation == RelationType.RELATED_TO
                and subj_type in (EntityType.CAMPAIGN, EntityType.INFRASTRUCTURE, EntityType.IOC)
                and obj_type in (EntityType.MALWARE, EntityType.TOOL)):
            if self._contains_any(evidence, _DELIVERS_HINTS):
                return RelationType.DELIVERS

        # Malware/ThreatActor + related_to + Organization/Software often indicates a victim/target.
        if (relation in (RelationType.RELATED_TO, RelationType.ASSOCIATED_WITH)
                and subj_type in (EntityType.MALWARE, EntityType.THREAT_ACTOR, EntityType.CAMPAIGN)
                and obj_type in (EntityType.ORGANIZATION, EntityType.SOFTWARE, EntityType.LOCATION)):
            if self._contains_any(evidence, _TARGETS_HINTS):
                return RelationType.TARGETS

        # Malware + related_to + Tool/Software strongly trends to uses for local outputs.
        if (relation in (RelationType.RELATED_TO, RelationType.ASSOCIATED_WITH)
                and subj_type in (EntityType.MALWARE, EntityType.CAMPAIGN)
                and obj_type in (EntityType.TOOL, EntityType.SOFTWARE, EntityType.TECHNIQUE, EntityType.FILE)):
            if self._contains_any(evidence, _USES_HINTS + _DROPS_HINTS + _DELIVERS_HINTS):
                return RelationType.USES

        return None

    def _relation_from_evidence(
        self,
        triple: Triple,
        evidence: str,
        strong_only: bool,
    ) -> RelationType | None:
        subj_type = triple.subject_type
        obj_type = triple.object_type

        if self._contains_any(evidence, _EXPLOITS_HINTS) and obj_type in (
            EntityType.VULNERABILITY,
            EntityType.SOFTWARE,
            EntityType.INFRASTRUCTURE,
        ):
            return RelationType.EXPLOITS

        if self._contains_any(evidence, _COMM_HINTS) and obj_type in (
            EntityType.IOC,
            EntityType.INFRASTRUCTURE,
        ):
            return RelationType.COMMUNICATES_WITH

        if self._contains_any(evidence, _DROPS_HINTS) and obj_type in (
            EntityType.FILE,
            EntityType.MALWARE,
            EntityType.TOOL,
        ):
            return RelationType.DROPS

        if self._contains_any(evidence, _DELIVERS_HINTS) and obj_type in (
            EntityType.FILE,
            EntityType.MALWARE,
            EntityType.TOOL,
        ):
            return RelationType.DELIVERS

        if self._contains_any(evidence, _USES_HINTS) and obj_type in (
            EntityType.TOOL,
            EntityType.SOFTWARE,
            EntityType.TECHNIQUE,
            EntityType.INFRASTRUCTURE,
            EntityType.IOC,
            EntityType.FILE,
            EntityType.MALWARE,
        ):
            return RelationType.USES

        if self._contains_any(evidence, _ATTR_HINTS) and obj_type in (
            EntityType.THREAT_ACTOR,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
        ):
            return RelationType.ATTRIBUTED_TO

        if self._contains_any(evidence, _LOC_HINTS) and obj_type == EntityType.LOCATION:
            return RelationType.LOCATED_IN

        if self._contains_any(evidence, _MITIGATE_HINTS) and subj_type in (
            EntityType.VULNERABILITY,
            EntityType.SOFTWARE,
            EntityType.ORGANIZATION,
        ):
            return RelationType.MITIGATED_BY

        if subj_type == EntityType.MALWARE and obj_type == EntityType.MALWARE:
            if self._contains_any(evidence, _VARIANT_HINTS):
                return RelationType.VARIANT_OF
            # Name-based variant detection: shared suffix patterns (e.g., XingLocker/MountLocker)
            subj_lower = triple.subject.lower()
            obj_lower = triple.object.lower()
            for suffix in ("locker", "ransomware", "rat", "loader", "stealer", "wiper", "bot"):
                if subj_lower.endswith(suffix) and obj_lower.endswith(suffix) and subj_lower != obj_lower:
                    return RelationType.VARIANT_OF

        if self._contains_any(evidence, _TARGETS_HINTS) and obj_type in (
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.SOFTWARE,
            EntityType.INFRASTRUCTURE,
            EntityType.IOC,
            EntityType.FILE,
            EntityType.VULNERABILITY,
            EntityType.OTHER,
        ):
            return RelationType.TARGETS

        if strong_only and self._contains_any(evidence, ("uses",)) and obj_type in (
            EntityType.TOOL,
            EntityType.SOFTWARE,
            EntityType.TECHNIQUE,
            EntityType.INFRASTRUCTURE,
            EntityType.IOC,
            EntityType.FILE,
            EntityType.MALWARE,
        ):
            return RelationType.USES

        return None

    @staticmethod
    def _infer_from_types(triple: Triple) -> RelationType | None:
        """Infer relation from entity type pairs when evidence gives no signal.

        Only fires for related_to triples as a last-resort upgrade.
        """
        st, ot = triple.subject_type, triple.object_type

        # Actor/Malware/Campaign + Location → located_in
        if st in (EntityType.THREAT_ACTOR, EntityType.ORGANIZATION, EntityType.INFRASTRUCTURE) \
                and ot == EntityType.LOCATION:
            return RelationType.LOCATED_IN

        # Actor/Malware + Technique → uses
        if st in (EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN) \
                and ot == EntityType.TECHNIQUE:
            return RelationType.USES

        # Actor/Malware + Organization/Location (as target) → targets
        if st in (EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN) \
                and ot in (EntityType.ORGANIZATION, EntityType.SOFTWARE):
            return RelationType.TARGETS

        # Malware + Malware → variant_of (two malware in related_to is usually a family relation)
        if st == EntityType.MALWARE and ot == EntityType.MALWARE:
            return RelationType.VARIANT_OF

        # Malware + IOC/Infrastructure → communicates_with
        if st in (EntityType.MALWARE, EntityType.TOOL) \
                and ot in (EntityType.IOC, EntityType.INFRASTRUCTURE):
            return RelationType.COMMUNICATES_WITH

        return None

    @staticmethod
    def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return " ".join(text.lower().split())

    def _evidence_mentions_entities(self, triple: Triple, evidence: str) -> bool:
        subject = self._normalize_for_match(triple.subject)
        object_ = self._normalize_for_match(triple.object)
        if not subject or not object_:
            return False
        return subject in evidence and object_ in evidence
