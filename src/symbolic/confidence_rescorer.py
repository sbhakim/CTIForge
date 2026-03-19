"""Symbolic confidence re-scorer for CTI triples.

After extraction and validation, this module re-scores each triple using
multiple symbolic signals and prunes the lowest-confidence triples.
This is a neuro-symbolic precision filter: the neural layer proposes,
the symbolic layer scores and selects.

Signals used:
  1. Evidence alignment: does evidence mention both subject and object?
  2. Entity name specificity: longer, more specific names score higher
  3. Relation specificity: non-generic relations score higher
  4. Type-pair strength: constrained type pairs score higher than catch-alls
  5. Entity type specificity: non-Other types score higher
"""

from __future__ import annotations

from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.schema.type_constraints import TYPE_PAIR_CONSTRAINTS
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_GENERIC_RELATIONS = frozenset({RelationType.RELATED_TO, RelationType.ASSOCIATED_WITH})

# Minimum word count for a "specific" entity name
_MIN_SPECIFIC_NAME_WORDS = 2


class ConfidenceRescorer:
    """Re-score triples using symbolic signals and prune low-confidence ones."""

    def __init__(self, prune_threshold: float = 0.35):
        self.prune_threshold = prune_threshold

    def _evidence_score(self, triple: Triple) -> float:
        """Score based on how well evidence supports the triple."""
        evidence = triple.evidence_text.lower().strip()
        if not evidence:
            return 0.0

        subj_lower = triple.subject.lower().strip()
        obj_lower = triple.object.lower().strip()

        # Check if subject and object appear in evidence
        subj_in = subj_lower in evidence or any(
            tok in evidence for tok in subj_lower.split() if len(tok) > 3
        )
        obj_in = obj_lower in evidence or any(
            tok in evidence for tok in obj_lower.split() if len(tok) > 3
        )

        if subj_in and obj_in:
            return 1.0
        elif subj_in or obj_in:
            return 0.5
        return 0.1

    def _entity_name_specificity(self, name: str) -> float:
        """Score entity name specificity (longer, more specific = higher)."""
        name = name.strip()
        if len(name) <= 2:
            return 0.1
        words = name.split()
        if len(words) >= 3:
            return 1.0
        elif len(words) == 2:
            return 0.8
        elif len(name) >= 5:
            return 0.7
        return 0.4

    def _relation_specificity_score(self, relation: RelationType) -> float:
        """Score relation specificity."""
        if relation == RelationType.RELATED_TO:
            return 0.2
        if relation == RelationType.ASSOCIATED_WITH:
            return 0.4
        return 1.0

    def _type_pair_strength(self, triple: Triple) -> float:
        """Score based on how constrained the type-pair is."""
        if triple.relation in (RelationType.RELATED_TO, RelationType.ASSOCIATED_WITH):
            return 0.5  # Unconstrained catch-alls
        constraints = TYPE_PAIR_CONSTRAINTS.get(triple.relation)
        if constraints is None:
            return 0.5
        valid_subjects, valid_objects = constraints
        if triple.subject_type in valid_subjects and triple.object_type in valid_objects:
            return 1.0
        return 0.2  # Invalid type pair that somehow passed validation

    def _entity_type_score(self, triple: Triple) -> float:
        """Score based on entity type specificity."""
        score = 0.0
        score += 0.0 if triple.subject_type == EntityType.OTHER else 0.5
        score += 0.0 if triple.object_type == EntityType.OTHER else 0.5
        return score

    def rescore(self, triple: Triple) -> float:
        """Compute composite confidence score for a triple."""
        evidence = self._evidence_score(triple)
        name_spec = (
            self._entity_name_specificity(triple.subject) +
            self._entity_name_specificity(triple.object)
        ) / 2.0
        rel_spec = self._relation_specificity_score(triple.relation)
        type_strength = self._type_pair_strength(triple)
        type_spec = self._entity_type_score(triple)

        # Weighted combination
        score = (
            0.35 * evidence +
            0.15 * name_spec +
            0.15 * rel_spec +
            0.20 * type_strength +
            0.15 * type_spec
        )
        return round(score, 4)

    def rescore_and_prune(self, triples: list[Triple]) -> list[Triple]:
        """Re-score all triples and prune those below threshold."""
        result = []
        pruned = 0

        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                result.append(t)
                continue

            new_confidence = self.rescore(t)
            if new_confidence < self.prune_threshold:
                pruned += 1
                t_copy = t.model_copy()
                t_copy.validation_status = ValidationStatus.REJECTED
                t_copy.rejection_reason = f"confidence_rescore={new_confidence:.3f} < {self.prune_threshold}"
                result.append(t_copy)
            else:
                t_copy = t.model_copy()
                t_copy.confidence = new_confidence
                result.append(t_copy)

        if pruned:
            logger.info(
                "Confidence re-scorer: pruned %d low-confidence triples (threshold=%.2f)",
                pruned, self.prune_threshold,
            )

        return result
