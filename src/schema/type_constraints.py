"""Type-pair constraint table for symbolic validation.

This is the core neuro-symbolic contract: it defines which
(subject_type, relation, object_type) combinations are valid.
The symbolic validator uses this to reject impossible triples.
"""

from src.schema.entities import EntityType
from src.schema.relations import RelationType

# Each relation maps to a set of (valid_subject_types, valid_object_types)
# 'any' means no constraint on that side
_ANY = frozenset(EntityType)

_ACTOR_LIKE = frozenset({
    EntityType.THREAT_ACTOR,
    EntityType.MALWARE,
    EntityType.CAMPAIGN,
})

_TOOL_LIKE = frozenset({
    EntityType.MALWARE,
    EntityType.TOOL,
    EntityType.TECHNIQUE,
    EntityType.INFRASTRUCTURE,
    EntityType.IOC,
})

TYPE_PAIR_CONSTRAINTS: dict[RelationType, tuple[frozenset[EntityType], frozenset[EntityType]]] = {
    RelationType.USES: (
        frozenset({
            EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN,
            EntityType.TOOL, EntityType.VULNERABILITY, EntityType.ORGANIZATION,
        }),
        frozenset({
            EntityType.MALWARE, EntityType.TOOL, EntityType.TECHNIQUE,
            EntityType.INFRASTRUCTURE, EntityType.IOC, EntityType.SOFTWARE,
            EntityType.FILE, EntityType.VULNERABILITY,
        }),
    ),
    RelationType.TARGETS: (
        frozenset({
            EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN,
            EntityType.TOOL, EntityType.VULNERABILITY,
        }),
        frozenset({
            EntityType.ORGANIZATION, EntityType.LOCATION, EntityType.SOFTWARE,
            EntityType.INFRASTRUCTURE, EntityType.IOC, EntityType.FILE,
            EntityType.VULNERABILITY, EntityType.OTHER, EntityType.TECHNIQUE,
            EntityType.CAMPAIGN,
        }),
    ),
    RelationType.EXPLOITS: (
        frozenset({
            EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN,
            EntityType.TOOL, EntityType.VULNERABILITY,
        }),
        frozenset({EntityType.VULNERABILITY, EntityType.SOFTWARE, EntityType.INFRASTRUCTURE}),
    ),
    RelationType.DELIVERS: (
        frozenset({
            EntityType.THREAT_ACTOR, EntityType.MALWARE, EntityType.CAMPAIGN,
            EntityType.INFRASTRUCTURE, EntityType.IOC, EntityType.VULNERABILITY,
        }),
        frozenset({EntityType.MALWARE, EntityType.FILE, EntityType.TOOL}),
    ),
    RelationType.COMMUNICATES_WITH: (
        frozenset({EntityType.MALWARE, EntityType.TOOL}),
        frozenset({EntityType.IOC, EntityType.INFRASTRUCTURE}),
    ),
    RelationType.DROPS: (
        frozenset({EntityType.MALWARE, EntityType.THREAT_ACTOR, EntityType.CAMPAIGN}),
        frozenset({EntityType.MALWARE, EntityType.FILE, EntityType.TOOL, EntityType.IOC}),
    ),
    RelationType.ATTRIBUTED_TO: (
        frozenset({EntityType.MALWARE, EntityType.CAMPAIGN, EntityType.TOOL, EntityType.THREAT_ACTOR}),
        frozenset({EntityType.THREAT_ACTOR, EntityType.ORGANIZATION, EntityType.LOCATION}),
    ),
    RelationType.ASSOCIATED_WITH: (_ANY, _ANY),  # intentionally unconstrained
    RelationType.VARIANT_OF: (
        frozenset({
            EntityType.MALWARE, EntityType.TOOL, EntityType.VULNERABILITY,
            EntityType.THREAT_ACTOR, EntityType.CAMPAIGN, EntityType.TECHNIQUE,
        }),
        frozenset({
            EntityType.MALWARE, EntityType.TOOL, EntityType.TECHNIQUE,
            EntityType.OTHER, EntityType.THREAT_ACTOR, EntityType.CAMPAIGN,
        }),
    ),
    RelationType.LOCATED_IN: (
        frozenset({
            EntityType.ORGANIZATION, EntityType.THREAT_ACTOR, EntityType.INFRASTRUCTURE,
            EntityType.CAMPAIGN, EntityType.MALWARE,
        }),
        frozenset({EntityType.LOCATION}),
    ),
    RelationType.MITIGATED_BY: (
        frozenset({EntityType.VULNERABILITY, EntityType.ORGANIZATION, EntityType.SOFTWARE}),
        frozenset({EntityType.TOOL, EntityType.TECHNIQUE, EntityType.SOFTWARE, EntityType.ORGANIZATION}),
    ),
    RelationType.RELATED_TO: (_ANY, _ANY),  # catch-all, intentionally unconstrained
}

# Relations where self-loops (subject == object) are forbidden
SELF_LOOP_FORBIDDEN: frozenset[RelationType] = frozenset({
    RelationType.USES,
    RelationType.TARGETS,
    RelationType.EXPLOITS,
    RelationType.DELIVERS,
    RelationType.COMMUNICATES_WITH,
    RelationType.DROPS,
    RelationType.ATTRIBUTED_TO,
    RelationType.VARIANT_OF,
    RelationType.LOCATED_IN,
    RelationType.MITIGATED_BY,
})


class TypeConstraints:
    """Validates (subject_type, relation, object_type) triples against the constraint table."""

    @staticmethod
    def is_valid_type_pair(
        subject_type: EntityType,
        relation: RelationType,
        object_type: EntityType,
    ) -> bool:
        """Check whether the type pair is allowed for this relation."""
        if relation not in TYPE_PAIR_CONSTRAINTS:
            return True
        valid_subjects, valid_objects = TYPE_PAIR_CONSTRAINTS[relation]
        return subject_type in valid_subjects and object_type in valid_objects

    @staticmethod
    def is_self_loop_allowed(relation: RelationType) -> bool:
        """Check whether self-loops are permitted for this relation."""
        return relation not in SELF_LOOP_FORBIDDEN

    @staticmethod
    def get_violation_reason(
        subject_type: EntityType,
        relation: RelationType,
        object_type: EntityType,
    ) -> str | None:
        """Return a human-readable violation reason, or None if valid."""
        if relation not in TYPE_PAIR_CONSTRAINTS:
            return None
        valid_subjects, valid_objects = TYPE_PAIR_CONSTRAINTS[relation]
        violations = []
        if subject_type not in valid_subjects:
            violations.append(
                f"subject type '{subject_type.value}' not valid for relation '{relation.value}'"
            )
        if object_type not in valid_objects:
            violations.append(
                f"object type '{object_type.value}' not valid for relation '{relation.value}'"
            )
        return "; ".join(violations) if violations else None
