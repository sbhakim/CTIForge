"""Canonicalization and grounding evaluation metrics."""

from __future__ import annotations

from typing import Any

from src.schema.entities import Entity
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def compute_canonicalization_metrics(
    predicted_entities: dict[str, Entity],
    gold_entities: dict[str, Entity],
) -> dict[str, Any]:
    """Compute merge precision, recall, and false merge rate.

    Compares predicted canonical entities against gold entities.
    """
    pred_ids = set(predicted_entities.keys())
    gold_ids = set(gold_entities.keys())

    tp = len(pred_ids & gold_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "merge_precision": round(precision, 4),
        "merge_recall": round(recall, 4),
        "merge_f1": round(f1, 4),
        "predicted_canonical_nodes": len(pred_ids),
        "gold_canonical_nodes": len(gold_ids),
        "node_reduction": len(pred_ids) - len(gold_ids),
    }


def compute_duplicate_reduction(
    raw_entity_count: int,
    canonical_entity_count: int,
) -> dict[str, Any]:
    """Measure how much canonicalization reduced duplicates."""
    reduction = raw_entity_count - canonical_entity_count
    reduction_pct = reduction / max(raw_entity_count, 1) * 100

    return {
        "raw_entities": raw_entity_count,
        "canonical_entities": canonical_entity_count,
        "duplicates_removed": reduction,
        "reduction_percentage": round(reduction_pct, 2),
    }
