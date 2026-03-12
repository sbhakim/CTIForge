"""Relation type definitions and triple data model."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.schema.entities import EntityType


class RelationType(str, Enum):
    """12 core relation types for CTI knowledge graphs.

    Kept intentionally small to reduce extraction confusion. Overlapping
    types (downloads/executes/installs) are collapsed into uses/delivers/drops.
    """

    USES = "uses"
    TARGETS = "targets"
    EXPLOITS = "exploits"
    DELIVERS = "delivers"
    COMMUNICATES_WITH = "communicates_with"
    DROPS = "drops"
    ATTRIBUTED_TO = "attributed_to"
    ASSOCIATED_WITH = "associated_with"
    VARIANT_OF = "variant_of"
    LOCATED_IN = "located_in"
    MITIGATED_BY = "mitigated_by"
    RELATED_TO = "related_to"


# Mapping from free-form CTI-Nexus relation strings to our normalized types
# This is a best-effort mapping; unmatched relations fall to RELATED_TO
CTINEXUS_RELATION_MAP: dict[str, RelationType] = {
    "uses": RelationType.USES,
    "use": RelationType.USES,
    "used": RelationType.USES,
    "is used by": RelationType.USES,
    "has started using": RelationType.USES,
    "leverages": RelationType.USES,
    "leveraged": RelationType.USES,
    "are using": RelationType.USES,
    "began using": RelationType.USES,
    "executes": RelationType.USES,
    "installs": RelationType.USES,
    "installed": RelationType.USES,
    "runs on": RelationType.USES,
    "targets": RelationType.TARGETS,
    "target": RelationType.TARGETS,
    "targeted": RelationType.TARGETS,
    "has targeted": RelationType.TARGETS,
    "are targeting": RelationType.TARGETS,
    "is targeting": RelationType.TARGETS,
    "began targeting": RelationType.TARGETS,
    "is attacking": RelationType.TARGETS,
    "attacked": RelationType.TARGETS,
    "exploits": RelationType.EXPLOITS,
    "exploit": RelationType.EXPLOITS,
    "exploited": RelationType.EXPLOITS,
    "exploiting": RelationType.EXPLOITS,
    "are exploiting": RelationType.EXPLOITS,
    "is exploiting": RelationType.EXPLOITS,
    "have been exploiting": RelationType.EXPLOITS,
    "attempt to exploit": RelationType.EXPLOITS,
    "delivers": RelationType.DELIVERS,
    "delivered": RelationType.DELIVERS,
    "deliver": RelationType.DELIVERS,
    "distribute": RelationType.DELIVERS,
    "distributed via": RelationType.DELIVERS,
    "distributed through": RelationType.DELIVERS,
    "communicates_with": RelationType.COMMUNICATES_WITH,
    "communicates with": RelationType.COMMUNICATES_WITH,
    "connects to": RelationType.COMMUNICATES_WITH,
    "drops": RelationType.DROPS,
    "dropped": RelationType.DROPS,
    "deploys": RelationType.DROPS,
    "deployed": RelationType.DROPS,
    "attributed to": RelationType.ATTRIBUTED_TO,
    "attributed_to": RelationType.ATTRIBUTED_TO,
    "is attributed to": RelationType.ATTRIBUTED_TO,
    "is behind": RelationType.ATTRIBUTED_TO,
    "is linked to": RelationType.ATTRIBUTED_TO,
    "linked to": RelationType.ATTRIBUTED_TO,
    "associated with": RelationType.ASSOCIATED_WITH,
    "associated_with": RelationType.ASSOCIATED_WITH,
    "is associated with": RelationType.ASSOCIATED_WITH,
    "is connected to": RelationType.ASSOCIATED_WITH,
    "overlaps with": RelationType.ASSOCIATED_WITH,
    "shares similarities with": RelationType.ASSOCIATED_WITH,
    "variant of": RelationType.VARIANT_OF,
    "variant_of": RelationType.VARIANT_OF,
    "is a type of": RelationType.VARIANT_OF,
    "is successor of": RelationType.VARIANT_OF,
    "rebranded as": RelationType.VARIANT_OF,
    "located in": RelationType.LOCATED_IN,
    "located_in": RelationType.LOCATED_IN,
    "is based in": RelationType.LOCATED_IN,
    "are located in": RelationType.LOCATED_IN,
    "resided in": RelationType.LOCATED_IN,
    "mitigated by": RelationType.MITIGATED_BY,
    "mitigated_by": RelationType.MITIGATED_BY,
    "patched": RelationType.MITIGATED_BY,
    "fixes": RelationType.MITIGATED_BY,
    "addresses": RelationType.MITIGATED_BY,
    # Additional CTI-Nexus free-form relation coverage
    "is": RelationType.VARIANT_OF,
    "is a": RelationType.VARIANT_OF,
    "is a type of": RelationType.VARIANT_OF,
    "is a variant of": RelationType.VARIANT_OF,
    "is also known as": RelationType.ASSOCIATED_WITH,
    "also known as": RelationType.ASSOCIATED_WITH,
    "observed": RelationType.ASSOCIATED_WITH,
    "discovered": RelationType.ASSOCIATED_WITH,
    "identified": RelationType.ASSOCIATED_WITH,
    "found": RelationType.ASSOCIATED_WITH,
    "detected": RelationType.ASSOCIATED_WITH,
    "reported": RelationType.ASSOCIATED_WITH,
    "analyzed": RelationType.ASSOCIATED_WITH,
    "tracked": RelationType.ASSOCIATED_WITH,
    "monitors": RelationType.ASSOCIATED_WITH,
    "investigated": RelationType.ASSOCIATED_WITH,
    "switched to": RelationType.USES,
    "attempted to deploy": RelationType.DELIVERS,
    "attempted to use": RelationType.USES,
    "attempted to exploit": RelationType.EXPLOITS,
    "is written in": RelationType.USES,
    "written in": RelationType.USES,
    "developed in": RelationType.USES,
    "built with": RelationType.USES,
    "coded in": RelationType.USES,
    "attempts to stop": RelationType.TARGETS,
    "attempts to delete": RelationType.TARGETS,
    "stops": RelationType.TARGETS,
    "deletes": RelationType.TARGETS,
    "encrypts": RelationType.TARGETS,
    "encrypted": RelationType.TARGETS,
    "steals": RelationType.TARGETS,
    "exfiltrates": RelationType.TARGETS,
    "scans": RelationType.TARGETS,
    "infects": RelationType.TARGETS,
    "compromised": RelationType.TARGETS,
    "compromises": RelationType.TARGETS,
    "abuses": RelationType.EXPLOITS,
    "abused": RelationType.EXPLOITS,
    "takes advantage of": RelationType.EXPLOITS,
    "affects": RelationType.TARGETS,
    "affected": RelationType.TARGETS,
    "impacts": RelationType.TARGETS,
    "impacted": RelationType.TARGETS,
    "downloads": RelationType.DROPS,
    "downloaded": RelationType.DROPS,
    "downloads and installs": RelationType.DROPS,
    "loads": RelationType.DROPS,
    "loaded": RelationType.DROPS,
    "launches": RelationType.USES,
    "launched": RelationType.USES,
    "utilizes": RelationType.USES,
    "employs": RelationType.USES,
    "employed": RelationType.USES,
    "created": RelationType.ATTRIBUTED_TO,
    "created by": RelationType.ATTRIBUTED_TO,
    "developed by": RelationType.ATTRIBUTED_TO,
    "operated by": RelationType.ATTRIBUTED_TO,
    "maintained by": RelationType.ATTRIBUTED_TO,
    "used by": RelationType.USES,
    "hosted on": RelationType.LOCATED_IN,
    "hosted in": RelationType.LOCATED_IN,
    "operates in": RelationType.LOCATED_IN,
    "based in": RelationType.LOCATED_IN,
    "sends": RelationType.COMMUNICATES_WITH,
    "receives": RelationType.COMMUNICATES_WITH,
    "contacts": RelationType.COMMUNICATES_WITH,
    "beacons to": RelationType.COMMUNICATES_WITH,
    "calls back to": RelationType.COMMUNICATES_WITH,
    "resolves to": RelationType.COMMUNICATES_WITH,
    # More CTI-Nexus patterns from gold data
    "affects": RelationType.TARGETS,
    "affected by": RelationType.EXPLOITS,
    "impersonated": RelationType.TARGETS,
    "impersonates": RelationType.TARGETS,
    "is tracked as": RelationType.ASSOCIATED_WITH,
    "tracked as": RelationType.ASSOCIATED_WITH,
    "stole": RelationType.TARGETS,
    "steals": RelationType.TARGETS,
    "stolen": RelationType.TARGETS,
    "patched in": RelationType.MITIGATED_BY,
    "included sectors": RelationType.TARGETS,
    "hired people specialized in": RelationType.USES,
    "enables": RelationType.USES,
    "allows": RelationType.USES,
    "allows attacker to": RelationType.USES,
    "had rebrands": RelationType.VARIANT_OF,
    "disclosed": RelationType.ASSOCIATED_WITH,
    "was released by": RelationType.ATTRIBUTED_TO,
    "released by": RelationType.ATTRIBUTED_TO,
    "could propagate": RelationType.DELIVERS,
    "propagates": RelationType.DELIVERS,
    "includes": RelationType.ASSOCIATED_WITH,
    "intended to achieve": RelationType.USES,
    "potentially connected to": RelationType.ASSOCIATED_WITH,
    "occurred in": RelationType.LOCATED_IN,
    "modifies": RelationType.TARGETS,
    "modified": RelationType.TARGETS,
    "from": RelationType.ATTRIBUTED_TO,
    "is located in": RelationType.LOCATED_IN,
    "deploys": RelationType.DROPS,
    # Step 1: expanded gold relation coverage (top unmatched from CTI-Nexus)
    "include": RelationType.ASSOCIATED_WITH,
    "contain": RelationType.ASSOCIATED_WITH,
    "contains": RelationType.ASSOCIATED_WITH,
    "has": RelationType.ASSOCIATED_WITH,
    "is classified as": RelationType.VARIANT_OF,
    "described as": RelationType.VARIANT_OF,
    "has been active since": RelationType.ASSOCIATED_WITH,
    "can perform": RelationType.USES,
    "can execute": RelationType.USES,
    "performs": RelationType.USES,
    "conducts": RelationType.TARGETS,
    "conducted": RelationType.TARGETS,
    "provides": RelationType.USES,
    "breached": RelationType.TARGETS,
    "has vulnerability": RelationType.EXPLOITS,
    "is a vulnerability in": RelationType.EXPLOITS,
    "uncovered": RelationType.ASSOCIATED_WITH,
    "released": RelationType.ASSOCIATED_WITH,
    "released joint statement regarding": RelationType.ASSOCIATED_WITH,
    "include those manufactured by": RelationType.ASSOCIATED_WITH,
    "has been observed": RelationType.ASSOCIATED_WITH,
    "was observed": RelationType.ASSOCIATED_WITH,
    "is known for": RelationType.ASSOCIATED_WITH,
    "is responsible for": RelationType.ATTRIBUTED_TO,
    "responsible for": RelationType.ATTRIBUTED_TO,
    "focuses on": RelationType.TARGETS,
    "focused on": RelationType.TARGETS,
    "collects": RelationType.TARGETS,
    "harvests": RelationType.TARGETS,
    "captures": RelationType.TARGETS,
    "exploited vulnerability in": RelationType.EXPLOITS,
    "has exploited": RelationType.EXPLOITS,
    "has been exploited": RelationType.EXPLOITS,
    "is used for": RelationType.USES,
    "used for": RelationType.USES,
    "was used by": RelationType.USES,
    "is deployed by": RelationType.DROPS,
    "was deployed": RelationType.DROPS,
    "operates": RelationType.USES,
    "developed": RelationType.ATTRIBUTED_TO,
    "authored": RelationType.ATTRIBUTED_TO,
    "published": RelationType.ASSOCIATED_WITH,
    "issued": RelationType.ASSOCIATED_WITH,
    "warned about": RelationType.ASSOCIATED_WITH,
    "recommended": RelationType.MITIGATED_BY,
    "advises": RelationType.MITIGATED_BY,
}


def normalize_relation(raw_relation: str) -> RelationType:
    """Map a free-form relation string to a normalized RelationType."""
    key = raw_relation.lower().strip()
    if key in CTINEXUS_RELATION_MAP:
        return CTINEXUS_RELATION_MAP[key]

    # Substring matching for common patterns
    if "exploit" in key or "abus" in key or "takes advantage" in key:
        return RelationType.EXPLOITS
    if "target" in key or "attack" in key or "infect" in key or "compromis" in key:
        return RelationType.TARGETS
    if "use" in key or "using" in key or "leverag" in key or "employ" in key or "utiliz" in key or "switch" in key:
        return RelationType.USES
    if "deliver" in key or "distribut" in key or "spread" in key:
        return RelationType.DELIVERS
    if "drop" in key or "deploy" in key or "download" in key or "install" in key or "load" in key:
        return RelationType.DROPS
    if "attribut" in key or "linked" in key or "created by" in key or "developed by" in key or "operated by" in key:
        return RelationType.ATTRIBUTED_TO
    if "associat" in key or "observ" in key or "discover" in key or "identif" in key or "detect" in key or "report" in key or "track" in key or "known as" in key:
        return RelationType.ASSOCIATED_WITH
    if "locat" in key or "based in" in key or "hosted" in key:
        return RelationType.LOCATED_IN
    if "mitigat" in key or "patch" in key or "fix" in key or "address" in key or "remediat" in key:
        return RelationType.MITIGATED_BY
    if "variant" in key or "successor" in key or "rebrand" in key or "is a type" in key:
        return RelationType.VARIANT_OF
    if "communicat" in key or "connect" in key or "beacon" in key or "contact" in key or "resolv" in key:
        return RelationType.COMMUNICATES_WITH
    if "encrypt" in key or "steal" in key or "exfiltrat" in key or "scan" in key or "delet" in key or "stop" in key or "affect" in key or "impact" in key:
        return RelationType.TARGETS
    if "written in" in key or "coded in" in key or "built with" in key:
        return RelationType.USES
    if "launch" in key:
        return RelationType.USES
    if "include" in key or "contain" in key:
        return RelationType.ASSOCIATED_WITH
    if "classif" in key or "describ" in key:
        return RelationType.VARIANT_OF
    if "perform" in key or "conduct" in key or "provid" in key or "operat" in key:
        return RelationType.USES
    if "breach" in key:
        return RelationType.TARGETS
    if "vulnerab" in key:
        return RelationType.EXPLOITS
    if "collect" in key or "harvest" in key or "captur" in key or "focus" in key:
        return RelationType.TARGETS
    if "responsib" in key or "author" in key:
        return RelationType.ATTRIBUTED_TO
    if "recommend" in key or "advis" in key:
        return RelationType.MITIGATED_BY
    if "publish" in key or "issu" in key or "warn" in key or "uncov" in key or "releas" in key:
        return RelationType.ASSOCIATED_WITH

    return RelationType.RELATED_TO


class ValidationStatus(str, Enum):
    """Tracks the symbolic validation state of a triple."""

    RAW = "raw"
    VALIDATED = "validated"
    REPAIRED = "repaired"
    REJECTED = "rejected"


class Triple(BaseModel):
    """An extracted or validated CTI triple (edge)."""

    event_id: str = Field(default="", description="Unique triple identifier")
    subject: str = Field(description="Subject entity name")
    subject_type: EntityType
    relation: RelationType
    object: str = Field(description="Object entity name")
    object_type: EntityType
    evidence_text: str = Field(default="", description="Source text supporting this triple")
    source_doc_id: str = Field(default="")
    source_chunk_id: str = Field(default="")
    attack_ids: list[str] = Field(
        default_factory=list, description="ATT&CK technique IDs if grounded"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    validation_status: ValidationStatus = Field(default=ValidationStatus.RAW)
    rejection_reason: Optional[str] = Field(
        default=None, description="Why this triple was rejected, if applicable"
    )
