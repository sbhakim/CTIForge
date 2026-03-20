"""Entity canonicalization and merge logic (Module D).

Normalizes entity names, resolves aliases, and merges duplicate entities
using exact match, alias lookup, and embedding similarity.
"""

from __future__ import annotations

from src.schema.entities import Entity, EntityType
from src.schema.relations import Triple, ValidationStatus
from src.grounding.alias_tables import AliasTable
from src.symbolic.error_logger import ErrorTaxonomyLogger, ErrorCategory
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Well-known CTI entity names with canonical casing
_KNOWN_CASING: dict[str, str] = {
    "cobalt strike": "Cobalt Strike", "mimikatz": "Mimikatz",
    "metasploit": "Metasploit", "emotet": "Emotet", "trickbot": "TrickBot",
    "ryuk": "Ryuk", "conti": "Conti", "lockbit": "LockBit",
    "revil": "REvil", "darkside": "DarkSide", "blackcat": "BlackCat",
    "alphv": "ALPHV", "blackmatter": "BlackMatter", "ragnarlocker": "RagnarLocker",
    "wannacry": "WannaCry", "notpetya": "NotPetya", "stuxnet": "Stuxnet",
    "lazarus": "Lazarus Group", "lazarus group": "Lazarus Group",
    "apt28": "APT28", "apt29": "APT29", "apt41": "APT41",
    "fin7": "FIN7", "fin11": "FIN11", "fin12": "FIN12",
    "sandworm": "Sandworm", "turla": "Turla", "kimsuky": "Kimsuky",
    "powershell": "PowerShell", "psexec": "PsExec",
    "bloodhound": "BloodHound", "nmap": "Nmap",
    "windows": "Windows", "linux": "Linux", "macos": "macOS",
    "exchange": "Microsoft Exchange", "microsoft exchange": "Microsoft Exchange",
}


class Canonicalizer:
    """Canonicalizes entity names and merges duplicates."""

    def __init__(
        self,
        group_aliases: AliasTable | None = None,
        software_aliases: AliasTable | None = None,
        similarity_threshold: float = 0.85,
        error_logger: ErrorTaxonomyLogger | None = None,
    ):
        self.group_aliases = group_aliases or AliasTable()
        self.software_aliases = software_aliases or AliasTable()
        self.similarity_threshold = similarity_threshold
        self.error_logger = error_logger or ErrorTaxonomyLogger()

        # Canonical name cache: (entity_type, raw_name_lower) -> canonical_name
        self._cache: dict[tuple[EntityType, str], str] = {}

    def canonicalize_name(self, name: str, entity_type: EntityType) -> str:
        """Resolve a raw entity name to its canonical form.

        Priority:
        1. Exact alias table match (for actors, malware, tools)
        2. String normalization (strip, case-normalize)
        """
        key = (entity_type, name.lower().strip())
        if key in self._cache:
            return self._cache[key]

        canonical = " ".join(name.strip().split())

        # Apply known casing normalization
        lower_name = canonical.lower()
        if lower_name in _KNOWN_CASING:
            canonical = _KNOWN_CASING[lower_name]

        # Try alias resolution based on entity type
        if entity_type in (EntityType.THREAT_ACTOR, EntityType.CAMPAIGN):
            resolved = self.group_aliases.resolve(name)
            if resolved:
                canonical = resolved
                self.error_logger.log(
                    category=ErrorCategory.REPAIRED_ALIAS,
                    action="repaired",
                    details=f"Alias resolved: '{name}' -> '{resolved}'",
                )
        elif entity_type in (EntityType.MALWARE, EntityType.TOOL, EntityType.SOFTWARE):
            resolved = self.software_aliases.resolve(name)
            if resolved:
                canonical = resolved
                self.error_logger.log(
                    category=ErrorCategory.REPAIRED_ALIAS,
                    action="repaired",
                    details=f"Alias resolved: '{name}' -> '{resolved}'",
                )

        self._cache[key] = canonical
        return canonical

    def canonicalize_triples(self, triples: list[Triple]) -> list[Triple]:
        """Apply canonicalization to all entity names in triples."""
        result = []
        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                result.append(t)
                continue

            t_copy = t.model_copy()
            t_copy.subject = self.canonicalize_name(t.subject, t.subject_type)
            t_copy.object = self.canonicalize_name(t.object, t.object_type)
            result.append(t_copy)

        return result

    def build_entity_map(self, triples: list[Triple]) -> dict[str, Entity]:
        """Build a deduplicated entity map from canonicalized triples.

        Returns dict of node_id -> Entity.
        """
        entities: dict[str, Entity] = {}

        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                continue

            # Process subject
            subj_id = Entity.make_node_id(t.subject_type, t.subject)
            if subj_id not in entities:
                entities[subj_id] = Entity(
                    node_id=subj_id,
                    canonical_name=t.subject,
                    entity_type=t.subject_type,
                    source_mentions=[f"{t.source_doc_id}:{t.source_chunk_id}"],
                )
            else:
                mention = f"{t.source_doc_id}:{t.source_chunk_id}"
                if mention not in entities[subj_id].source_mentions:
                    entities[subj_id].source_mentions.append(mention)
                if t.subject != entities[subj_id].canonical_name and t.subject not in entities[subj_id].aliases:
                    entities[subj_id].aliases.append(t.subject)

            # Process object
            obj_id = Entity.make_node_id(t.object_type, t.object)
            if obj_id not in entities:
                entities[obj_id] = Entity(
                    node_id=obj_id,
                    canonical_name=t.object,
                    entity_type=t.object_type,
                    source_mentions=[f"{t.source_doc_id}:{t.source_chunk_id}"],
                )
            else:
                mention = f"{t.source_doc_id}:{t.source_chunk_id}"
                if mention not in entities[obj_id].source_mentions:
                    entities[obj_id].source_mentions.append(mention)
                if t.object != entities[obj_id].canonical_name and t.object not in entities[obj_id].aliases:
                    entities[obj_id].aliases.append(t.object)

        logger.info(f"Built entity map: {len(entities)} unique entities")
        return entities
