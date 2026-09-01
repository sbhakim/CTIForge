"""Parse LLM extraction outputs into Triple objects."""

from __future__ import annotations

import json
import re

from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Valid enum values for fast lookup
_VALID_ENTITY_TYPES = {e.value.lower(): e for e in EntityType}
_VALID_RELATION_TYPES = {r.value.lower(): r for r in RelationType}
_PASSIVE_TARGET_PATTERNS = (
    "victim_of",
    "victim of",
    "impacted_by",
    "impacted by",
    "targeted_by",
    "targeted by",
    "attacked_by",
    "attacked by",
    "infected_by",
    "infected by",
    "encrypted_by",
    "encrypted by",
    "compromised_by",
    "compromised by",
)
_PASSIVE_USES_PATTERNS = ("used_by", "used by", "deployed_by", "deployed by")
_PASSIVE_ATTRIBUTION_PATTERNS = ("operated_by", "operated by", "run_by", "run by")
_TEMPORAL_OR_NUMERIC_PATTERNS = (
    r"^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}$",
    r"^\d{4}$",
    r"^\d{1,2}/\d{1,2}/\d{2,4}$",
    r"^\$[\d,.]+(?:\s*(?:million|billion|m|bn))?$",
    r"^\d+\s+(?:organizations?|victims?|companies|entities|sectors|services|systems?)$",
)
_GENERIC_OBJECTS = frozenset({
    "ransomware",
    "malware",
    "campaign",
    "threat actor",
    "threat actors",
    "actor",
    "actors",
    "tool",
    "tools",
    "technique",
    "techniques",
    "services",
    "multiple services",
    "various sectors",
    "sectors",
    "organizations",
    "rudimentary",
})


def parse_extraction_response(
    raw_response: str,
    source_doc_id: str,
    source_chunk_id: str,
    source_text: str = "",
) -> list[Triple]:
    """Parse a raw LLM JSON response into a list of Triple objects.

    Handles common LLM output issues:
    - markdown code blocks wrapping the JSON
    - trailing commas
    - partial/malformed JSON
    """
    cleaned = _strip_json_wrappers(raw_response)
    data: dict | None = None

    for candidate in [cleaned, *_extract_balanced_json_candidates(cleaned)]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        data = _coerce_extraction_payload(parsed)
        if data is not None:
            break

    if data is None:
        repaired = _repair_json_like_text(cleaned)
        data = _coerce_extraction_payload(repaired)

    if data is None:
        logger.warning(f"Failed to parse extraction response for {source_chunk_id}")
        return []

    raw_triples = data.get("triples", [])
    if not isinstance(raw_triples, list):
        logger.warning(f"'triples' is not a list in response for {source_chunk_id}")
        return []

    triples = []
    for i, rt in enumerate(raw_triples):
        if not isinstance(rt, dict):
            continue

        # `dict.get(k, default)` returns the default only when the key is
        # ABSENT. A key present with an explicit JSON null returns None, and
        # None.strip() raises -- which killed a full 149-document run when
        # Gemma-2-9B emitted {"object": null}. `or` covers both cases.
        subject = _clean_entity_name((rt.get("subject") or "").strip())
        object_ = _clean_entity_name((rt.get("object") or "").strip())
        if not subject or not object_:
            continue

        subject_type = _resolve_entity_type(rt.get("subject_type") or "Other")
        object_type = _resolve_entity_type(rt.get("object_type") or "Other")
        raw_relation = (rt.get("relation") or "related_to").strip()
        relation = _resolve_relation_type(raw_relation)
        evidence = (rt.get("evidence") or rt.get("evidence_text") or "").strip()
        normalized_items = _normalize_extracted_items(
            subject=subject,
            subject_type=subject_type,
            relation=relation,
            raw_relation=raw_relation,
            object_=object_,
            object_type=object_type,
        )
        for item in normalized_items:
            norm_subject, norm_subject_type, norm_relation, norm_object, norm_object_type = item
            if _is_low_value_fact(norm_relation, norm_object, norm_object_type):
                continue

            norm_evidence = evidence
            if not norm_evidence and source_text:
                norm_evidence = _infer_evidence_text(source_text, norm_subject, norm_object)

            triple = Triple(
                event_id=f"{source_chunk_id}_t{i}",
                subject=norm_subject,
                subject_type=norm_subject_type,
                relation=norm_relation,
                object=norm_object,
                object_type=norm_object_type,
                evidence_text=norm_evidence,
                source_doc_id=source_doc_id,
                source_chunk_id=source_chunk_id,
                validation_status=ValidationStatus.RAW,
            )
            triples.append(triple)

    # Per-chunk pronoun resolution
    triples = resolve_pronouns(triples)

    return triples


def _normalize_extracted_items(
    subject: str,
    subject_type: EntityType,
    relation: RelationType,
    raw_relation: str,
    object_: str,
    object_type: EntityType,
) -> list[tuple[str, EntityType, RelationType, str, EntityType]]:
    """Normalize passive phrasing and split bundled entity objects."""
    items = [(subject, subject_type, relation, object_, object_type)]
    raw_relation_l = raw_relation.lower().replace("-", "_").strip()

    if any(pattern in raw_relation_l for pattern in _PASSIVE_TARGET_PATTERNS):
        if subject_type in (
            EntityType.ORGANIZATION,
            EntityType.SOFTWARE,
            EntityType.INFRASTRUCTURE,
            EntityType.LOCATION,
            EntityType.FILE,
            EntityType.IOC,
            EntityType.OTHER,
        ) and object_type in (
            EntityType.THREAT_ACTOR,
            EntityType.MALWARE,
            EntityType.CAMPAIGN,
            EntityType.TOOL,
        ):
            items = [(object_, object_type, RelationType.TARGETS, subject, subject_type)]

    elif any(pattern in raw_relation_l for pattern in _PASSIVE_USES_PATTERNS):
        if subject_type in (
            EntityType.MALWARE,
            EntityType.TOOL,
            EntityType.TECHNIQUE,
            EntityType.SOFTWARE,
            EntityType.FILE,
        ) and object_type in (
            EntityType.THREAT_ACTOR,
            EntityType.CAMPAIGN,
            EntityType.ORGANIZATION,
        ):
            items = [(object_, object_type, RelationType.USES, subject, subject_type)]

    elif any(pattern in raw_relation_l for pattern in _PASSIVE_ATTRIBUTION_PATTERNS):
        if subject_type in (
            EntityType.MALWARE,
            EntityType.CAMPAIGN,
            EntityType.TOOL,
        ) and object_type in (
            EntityType.THREAT_ACTOR,
            EntityType.ORGANIZATION,
        ):
            items = [(subject, subject_type, RelationType.ATTRIBUTED_TO, object_, object_type)]

    expanded: list[tuple[str, EntityType, RelationType, str, EntityType]] = []
    for norm_subject, norm_subject_type, norm_relation, norm_object, norm_object_type in items:
        object_parts = _split_compound_entities(norm_object, norm_object_type)
        if len(object_parts) == 1:
            expanded.append((
                norm_subject,
                norm_subject_type,
                norm_relation,
                norm_object,
                norm_object_type,
            ))
            continue
        for part in object_parts:
            expanded.append((
                norm_subject,
                norm_subject_type,
                norm_relation,
                part,
                norm_object_type,
            ))
    return expanded


def _split_compound_entities(text: str, entity_type: EntityType) -> list[str]:
    """Split bundled entity mentions like 'A, B, and C' into separate objects."""
    if entity_type not in (
        EntityType.THREAT_ACTOR,
        EntityType.MALWARE,
        EntityType.TOOL,
        EntityType.CAMPAIGN,
        EntityType.ORGANIZATION,
        EntityType.LOCATION,
        EntityType.SOFTWARE,
        EntityType.TECHNIQUE,
    ):
        return [text]

    normalized = text.replace(", and ", ", ").replace(" and ", ", ")
    if "," not in normalized and ";" not in normalized:
        return [text]
    if normalized.lower() in {"command and control", "research and development"}:
        return [text]

    parts = [part.strip() for part in re.split(r"\s*[,;]\s*", normalized) if part.strip()]
    if len(parts) < 2:
        return [text]
    if len(parts) >= 3:
        return parts
    if any(len(part.split()) > 6 for part in parts):
        return [text]
    return parts


def _is_low_value_fact(relation: RelationType, object_: str, object_type: EntityType) -> bool:
    """Drop low-value temporal, numeric, and generic summary objects."""
    object_l = object_.strip().lower()
    if not object_l:
        return True

    if object_l in _GENERIC_OBJECTS and relation in (
        RelationType.RELATED_TO,
        RelationType.ASSOCIATED_WITH,
        RelationType.VARIANT_OF,
        RelationType.TARGETS,
    ):
        return True

    if object_l.startswith(("to ", "on ", "as ")) and relation in (
        RelationType.RELATED_TO,
        RelationType.ASSOCIATED_WITH,
        RelationType.TARGETS,
        RelationType.DROPS,
        RelationType.VARIANT_OF,
    ):
        return True

    if relation in (
        RelationType.RELATED_TO,
        RelationType.ASSOCIATED_WITH,
        RelationType.TARGETS,
    ):
        if any(re.fullmatch(pattern, object_l) for pattern in _TEMPORAL_OR_NUMERIC_PATTERNS):
            return True

    if relation == RelationType.VARIANT_OF and object_type == EntityType.OTHER and object_l in {
        "ransomware",
        "malware",
        "trojan",
        "rat",
        "loader",
    }:
        return True

    return False


def _infer_evidence_text(source_text: str, subject: str, object_: str) -> str:
    """Backfill evidence from the source chunk when the model omits it."""
    sentences = re.split(r"(?<=[.!?])\s+", source_text.strip())
    subject_l = subject.lower()
    object_l = object_.lower()

    for sent in sentences:
        sl = sent.lower()
        if subject_l in sl and object_l in sl:
            return sent.strip()

    for sent in sentences:
        sl = sent.lower()
        if subject_l in sl or object_l in sl:
            return sent.strip()

    return ""


def _strip_json_wrappers(text: str) -> str:
    """Remove common wrappers around the JSON payload."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^\s*JSON\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _coerce_extraction_payload(data: object) -> dict | None:
    """Normalize equivalent extraction payload shapes to {'triples': [...]}."""
    if isinstance(data, list):
        return {"triples": data}
    if not isinstance(data, dict):
        return None

    for key in ("triples", "triplets", "predicted_triples"):
        value = data.get(key)
        if isinstance(value, list):
            return {"triples": value}

    for key in ("triple", "predicted_triple"):
        value = data.get(key)
        if isinstance(value, dict):
            return {"triples": [value]}

    if {"subject", "relation", "object"}.issubset(data.keys()):
        return {"triples": [data]}

    return None


def _extract_balanced_json_candidates(text: str) -> list[str]:
    """Extract balanced JSON object substrings from mixed model output."""
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _repair_json_like_text(text: str) -> dict | None:
    """Best-effort repair for near-JSON local model output."""
    candidate = text.strip()
    candidate = re.sub(r"```(?:json)?", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Replace single quotes with double quotes only when the structure appears JSON-like.
    if "'" in candidate and '"' not in candidate[: min(len(candidate), 200)]:
        candidate2 = candidate.replace("'", '"')
        candidate2 = re.sub(r",(\s*[}\]])", r"\1", candidate2)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            candidate = candidate2

    salvaged = _salvage_triples_array(candidate)
    if salvaged is not None:
        return salvaged
    return None


def _salvage_triples_array(text: str) -> dict | None:
    """Recover complete triple objects from truncated local JSON output."""
    key_idx = text.find('"triples"')
    if key_idx == -1:
        return None

    list_start = text.find("[", key_idx)
    if list_start == -1:
        return None

    objects: list[dict] = []
    depth = 0
    obj_start: int | None = None
    in_string = False
    escape = False

    for i in range(list_start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    raw_obj = text[obj_start:i + 1]
                    raw_obj = re.sub(r",(\s*[}\]])", r"\1", raw_obj)
                    try:
                        parsed = json.loads(raw_obj)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    obj_start = None

    if objects:
        return {"triples": objects}
    return None


_PRONOUNS = frozenset({
    "they", "them", "it", "its", "he", "she",
    "the group", "the actor", "the actors",
    "the threat actor", "the threat actors",
    "the attacker", "the attackers",
    "the malware", "the ransomware", "the campaign", "the tool",
    "the operator", "the operators",
    "the hackers", "the hacker",
    "the adversary", "the adversaries",
    "attackers", "the gang",
    "government backed actors", "government-backed actors",
    "state-sponsored actors", "state sponsored actors",
    "threat actors", "threat actor",
})


def resolve_pronouns(triples: list[Triple]) -> list[Triple]:
    """Replace pronoun subjects/objects with the most recent named entity."""
    if not triples:
        return triples

    # Track last seen named entity per type category
    last_actor: str = ""
    last_malware: str = ""

    result = []
    for t in triples:
        # Update tracking from non-pronoun entities
        subj_lower = t.subject.lower().strip()
        obj_lower = t.object.lower().strip()

        if subj_lower not in _PRONOUNS:
            if t.subject_type in (EntityType.THREAT_ACTOR, EntityType.CAMPAIGN):
                last_actor = t.subject
            elif t.subject_type in (EntityType.MALWARE, EntityType.TOOL):
                last_malware = t.subject

        if obj_lower not in _PRONOUNS:
            if t.object_type in (EntityType.THREAT_ACTOR, EntityType.CAMPAIGN):
                last_actor = t.object
            elif t.object_type in (EntityType.MALWARE, EntityType.TOOL):
                last_malware = t.object

        # Resolve pronouns
        updated = t
        needs_update = False
        new_subject = t.subject
        new_object = t.object

        if subj_lower in _PRONOUNS:
            replacement = ""
            if t.subject_type in (EntityType.THREAT_ACTOR,) and last_actor:
                replacement = last_actor
            elif t.subject_type in (EntityType.MALWARE, EntityType.TOOL) and last_malware:
                replacement = last_malware
            elif last_actor:
                replacement = last_actor
            elif last_malware:
                replacement = last_malware
            if replacement:
                new_subject = replacement
                needs_update = True

        if obj_lower in _PRONOUNS:
            replacement = ""
            if t.object_type in (EntityType.THREAT_ACTOR,) and last_actor:
                replacement = last_actor
            elif t.object_type in (EntityType.MALWARE, EntityType.TOOL) and last_malware:
                replacement = last_malware
            elif last_actor:
                replacement = last_actor
            elif last_malware:
                replacement = last_malware
            if replacement:
                new_object = replacement
                needs_update = True

        if needs_update:
            updated = t.model_copy(update={"subject": new_subject, "object": new_object})

        result.append(updated)

    return result


def _clean_entity_name(name: str) -> str:
    """Clean LLM-extracted entity names by stripping trailing qualifiers.

    LLMs often append extra context like 'exploit chain for iPhones' when
    'exploit chain' is the actual entity. This strips common patterns while
    preserving the core entity name.
    """
    if not name:
        return name
    # Strip leading articles
    for prefix in ("the ", "a ", "an "):
        if name.lower().startswith(prefix) and len(name) > len(prefix) + 2:
            name = name[len(prefix):]
    # Strip trailing prepositional qualifiers that describe separate entities
    # e.g., "0-day exploit chain for iPhones" → "0-day exploit chain"
    # But preserve legitimate compound names like "Command and Control"
    name = re.sub(
        r"\s+(?:for|from|against|on|in|of|by|via|through|within|across)\s+(?:the\s+)?(?:[A-Z]|\w{2,}).*$",
        "",
        name,
    )
    # Strip verbose qualifiers that gold annotations do not use
    # e.g., "threat actors to execute arbitrary malicious code" → "arbitrary malicious code"
    name = re.sub(r"^(?:threat actors? to execute|attackers? to execute|allows? (?:attackers?|threat actors?) to (?:execute|run|deploy))\s+", "", name, flags=re.IGNORECASE)
    return name.strip()


def _resolve_entity_type(raw: str) -> EntityType:
    """Resolve a raw entity type string to an EntityType enum."""
    key = raw.strip().lower().replace(" ", "").replace("_", "")
    # Direct match
    for name, etype in _VALID_ENTITY_TYPES.items():
        if key == name.replace("_", "").lower():
            return etype
    # Fuzzy fallbacks
    if "threat" in key or "actor" in key or "apt" in key or "attacker" in key:
        return EntityType.THREAT_ACTOR
    if "malware" in key or "ransomware" in key or "trojan" in key:
        return EntityType.MALWARE
    if "vuln" in key or "cve" in key:
        return EntityType.VULNERABILITY
    if "technique" in key or "ttp" in key:
        return EntityType.TECHNIQUE
    if "tactic" in key:
        return EntityType.TACTIC
    if "campaign" in key or "operation" in key:
        return EntityType.CAMPAIGN
    if "ioc" in key or "indicator" in key:
        return EntityType.IOC
    if "infra" in key:
        return EntityType.INFRASTRUCTURE
    if "org" in key:
        return EntityType.ORGANIZATION
    if "location" in key or "country" in key or "region" in key:
        return EntityType.LOCATION
    if "tool" in key:
        return EntityType.TOOL
    if "file" in key:
        return EntityType.FILE
    if "software" in key:
        return EntityType.SOFTWARE
    return EntityType.OTHER


def _resolve_relation_type(raw: str) -> RelationType:
    """Resolve a raw relation string to a RelationType enum."""
    key = raw.strip().lower().replace(" ", "_")
    if key in _VALID_RELATION_TYPES:
        return _VALID_RELATION_TYPES[key]
    # Try without underscores
    key_no_us = key.replace("_", "")
    for name, rtype in _VALID_RELATION_TYPES.items():
        if key_no_us == name.replace("_", ""):
            return rtype
    # Substring fallbacks
    if "exploit" in key:
        return RelationType.EXPLOITS
    if "target" in key:
        return RelationType.TARGETS
    if "use" in key:
        return RelationType.USES
    if "employ" in key or "switch" in key:
        return RelationType.USES
    if "deliver" in key:
        return RelationType.DELIVERS
    if "drop" in key or "deploy" in key:
        return RelationType.DROPS
    if "attribut" in key:
        return RelationType.ATTRIBUTED_TO
    if "communicat" in key:
        return RelationType.COMMUNICATES_WITH
    if "variant" in key:
        return RelationType.VARIANT_OF
    if "version_of" in key or "known_as" in key or "rebrand" in key:
        return RelationType.VARIANT_OF
    if "locat" in key:
        return RelationType.LOCATED_IN
    if "mitigat" in key:
        return RelationType.MITIGATED_BY
    if "associat" in key:
        return RelationType.ASSOCIATED_WITH
    if "linked" in key or "connected" in key or "tied" in key:
        return RelationType.ASSOCIATED_WITH
    if "victim_of" in key or "impacted_by" in key:
        return RelationType.TARGETS
    if "classif" in key or "described_as" in key or "is_a_" in key:
        return RelationType.VARIANT_OF
    if "appear" in key or "emerge" in key or "first_appeared" in key:
        return RelationType.ASSOCIATED_WITH
    if "infect" in key or "compromis" in key or "attack" in key:
        return RelationType.TARGETS
    if "creat" in key or "develop" in key or "author" in key:
        return RelationType.ATTRIBUTED_TO
    if "evolv" in key or "rebrand" in key or "fork" in key:
        return RelationType.VARIANT_OF
    return RelationType.RELATED_TO
