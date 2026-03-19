"""Saliency-based triple filter (neuro-symbolic precision booster).

Ranks triples by how central/important they are to the CTI report
and keeps only the top-K, reducing over-generation of technically
valid but non-salient triples.

Scoring signals (all symbolic, no LLM call):
  1. Relation specificity: specific relations (exploits, delivers) score
     higher than generic ones (associated_with, related_to)
  2. Entity centrality: entities that appear in many triples are likely
     more important to the report
  3. Evidence strength: triples with longer, more specific evidence text
     are more likely to be genuinely important
"""

from __future__ import annotations

from collections import Counter

from src.schema.relations import Triple, RelationType, ValidationStatus
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Relation importance weights — specific CTI relations score higher
_RELATION_WEIGHT: dict[RelationType, float] = {
    RelationType.EXPLOITS: 1.0,
    RelationType.DELIVERS: 0.95,
    RelationType.DROPS: 0.95,
    RelationType.TARGETS: 0.9,
    RelationType.USES: 0.85,
    RelationType.ATTRIBUTED_TO: 0.9,
    RelationType.COMMUNICATES_WITH: 0.85,
    RelationType.MITIGATED_BY: 0.8,
    RelationType.VARIANT_OF: 0.75,
    RelationType.LOCATED_IN: 0.7,
    RelationType.ASSOCIATED_WITH: 0.5,
    RelationType.RELATED_TO: 0.3,
}


class SaliencyFilter:
    """Rank triples by saliency and keep only the most important ones."""

    def __init__(self, max_triples_per_doc: int = 20):
        self.max_triples_per_doc = max_triples_per_doc

    def _score_triple(
        self,
        triple: Triple,
        entity_freq: Counter,
        max_freq: int,
    ) -> float:
        """Compute a saliency score for a single triple."""
        # Signal 1: Relation specificity (0.0 - 1.0)
        rel_score = _RELATION_WEIGHT.get(triple.relation, 0.3)

        # Signal 2: Entity centrality — average frequency of subject/object
        subj_freq = entity_freq.get(triple.subject.lower(), 1)
        obj_freq = entity_freq.get(triple.object.lower(), 1)
        centrality = ((subj_freq + obj_freq) / 2) / max(max_freq, 1)

        # Signal 3: Evidence strength — longer evidence = more grounded
        evidence_len = len(triple.evidence_text)
        # Normalize: 50+ chars is good, cap at 200
        evidence_score = min(evidence_len / 200, 1.0) if evidence_len > 0 else 0.3

        # Signal 4: Model confidence
        confidence = triple.confidence

        # Weighted combination
        score = (
            0.35 * rel_score
            + 0.30 * centrality
            + 0.15 * evidence_score
            + 0.20 * confidence
        )
        return score

    def filter_triples(self, triples: list[Triple]) -> list[Triple]:
        """Rank and filter triples, keeping only top-K most salient ones."""
        # Separate rejected triples (pass them through unchanged)
        active = [t for t in triples if t.validation_status != ValidationStatus.REJECTED]
        rejected = [t for t in triples if t.validation_status == ValidationStatus.REJECTED]

        if len(active) <= self.max_triples_per_doc:
            return triples  # No filtering needed

        # Build entity frequency map
        entity_freq: Counter = Counter()
        for t in active:
            entity_freq[t.subject.lower()] += 1
            entity_freq[t.object.lower()] += 1
        max_freq = max(entity_freq.values()) if entity_freq else 1

        # Score each triple
        scored = [
            (self._score_triple(t, entity_freq, max_freq), i, t)
            for i, t in enumerate(active)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))  # Highest score first, stable

        kept = [t for _, _, t in scored[:self.max_triples_per_doc]]
        pruned_count = len(active) - len(kept)

        if pruned_count > 0:
            logger.info(
                "Saliency filter: kept %d / %d triples (pruned %d)",
                len(kept), len(active), pruned_count,
            )

        return kept + rejected
