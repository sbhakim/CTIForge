"""Entity type definitions and entity data model."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """14 core entity types for CTI knowledge graphs.

    IOC subtypes (IP, domain, URL, hash, email) are collapsed into a single IOC
    type with a subtype field. This reduces type confusion in LLM outputs while
    the symbolic validator can verify subtype formats via regex.
    """

    THREAT_ACTOR = "ThreatActor"
    MALWARE = "Malware"
    TOOL = "Tool"
    CAMPAIGN = "Campaign"
    VULNERABILITY = "Vulnerability"
    TECHNIQUE = "Technique"
    TACTIC = "Tactic"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    IOC = "IOC"
    FILE = "File"
    INFRASTRUCTURE = "Infrastructure"
    SOFTWARE = "Software"
    OTHER = "Other"


class IOCSubtype(str, Enum):
    """Subtypes for IOC entities, validated by regex in the symbolic layer."""

    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    HASH = "hash"
    EMAIL = "email"
    UNKNOWN = "unknown"


# Mapping from CTI-Nexus entity types to our schema for data loading
CTINEXUS_TYPE_MAP: dict[str, EntityType] = {
    "Attacker": EntityType.THREAT_ACTOR,
    "Malware": EntityType.MALWARE,
    "Tool": EntityType.TOOL,
    "Vulnerability": EntityType.VULNERABILITY,
    "Organization": EntityType.ORGANIZATION,
    "Location": EntityType.LOCATION,
    "Infrastructure": EntityType.INFRASTRUCTURE,
    "Indicator:Domain": EntityType.IOC,
    "Indicator:Email": EntityType.IOC,
    "Indicator:File": EntityType.IOC,
    "Indicator:URL": EntityType.IOC,
    "Event": EntityType.CAMPAIGN,
    "Information": EntityType.OTHER,
    "Account": EntityType.OTHER,
    "Credential": EntityType.OTHER,
    "Time": EntityType.OTHER,
    "Exploit Target": EntityType.VULNERABILITY,
    "Malware Characteristic:Behavior": EntityType.TECHNIQUE,
    "Malware Characteristic:Capability": EntityType.TECHNIQUE,
    "Malware Characteristic:Feature": EntityType.TECHNIQUE,
    "Malware Characteristic:Payload": EntityType.MALWARE,
    "Malware Characteristic:Variants": EntityType.MALWARE,
    "This entity cannot be classified into any of the existing types": EntityType.OTHER,
}


class Entity(BaseModel):
    """A canonicalized CTI entity node."""

    node_id: str = Field(description="Unique identifier: '{type_lower}:{normalized_name}'")
    canonical_name: str = Field(description="Display name after normalization")
    entity_type: EntityType
    ioc_subtype: Optional[IOCSubtype] = Field(
        default=None, description="Only set when entity_type is IOC"
    )
    aliases: list[str] = Field(default_factory=list)
    source_mentions: list[str] = Field(
        default_factory=list, description="List of 'doc_id:chunk_id' references"
    )
    external_ids: dict[str, list[str]] = Field(
        default_factory=dict, description="E.g. {'attack_software': ['S0013']}"
    )

    @staticmethod
    def make_node_id(entity_type: EntityType, name: str) -> str:
        """Generate a deterministic node_id from type and name."""
        normalized = name.lower().strip().replace(" ", "_")
        # Remove common punctuation that causes duplicate IDs
        for ch in ["(", ")", "'", '"', ",", ";"]:
            normalized = normalized.replace(ch, "")
        return f"{entity_type.value.lower()}:{normalized}"
