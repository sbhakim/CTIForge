"""Entity deduplication utilities for cross-document fusion."""

from __future__ import annotations

from src.schema.entities import Entity
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def find_duplicate_candidates(
    entities: dict[str, Entity],
) -> list[tuple[str, str, str]]:
    """Find potential duplicate entity pairs based on string similarity.

    Returns list of (node_id_1, node_id_2, reason) tuples.
    This is a lightweight pass using normalized string matching only.
    Embedding-based similarity is a future extension.
    """
    candidates = []

    # Group entities by type for faster comparison
    by_type: dict[str, list[Entity]] = {}
    for entity in entities.values():
        by_type.setdefault(entity.entity_type.value, []).append(entity)

    for etype, ents in by_type.items():
        for i, e1 in enumerate(ents):
            for e2 in ents[i + 1 :]:
                # Check if canonical names match after normalization
                n1 = _normalize_for_comparison(e1.canonical_name)
                n2 = _normalize_for_comparison(e2.canonical_name)

                if n1 == n2 and e1.node_id != e2.node_id:
                    candidates.append(
                        (e1.node_id, e2.node_id, f"normalized match: '{n1}'")
                    )
                elif _is_substring_match(n1, n2):
                    candidates.append(
                        (e1.node_id, e2.node_id, f"substring match: '{n1}' / '{n2}'")
                    )
                # Check alias overlap
                elif e1.aliases and e2.aliases:
                    overlap = set(a.lower() for a in e1.aliases) & set(
                        a.lower() for a in e2.aliases
                    )
                    if overlap:
                        candidates.append(
                            (e1.node_id, e2.node_id, f"shared alias: {overlap}")
                        )

    logger.info(f"Found {len(candidates)} duplicate candidates across {len(entities)} entities")
    return candidates


def _normalize_for_comparison(name: str) -> str:
    """Normalize a name for duplicate detection."""
    n = name.lower().strip()
    # Remove common prefixes/suffixes
    for prefix in ["the ", "a "]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    # Remove punctuation
    for ch in ["(", ")", "'", '"', ",", ";", ".", "-", "_"]:
        n = n.replace(ch, "")
    return n.strip()


def _is_substring_match(a: str, b: str) -> bool:
    """Check if one name is a meaningful substring of the other."""
    if len(a) < 4 or len(b) < 4:
        return False
    return a in b or b in a
