"""Error taxonomy logger for tracking symbolic validation actions.

This module logs every rejection, repair, and flag applied by the
symbolic validator. The resulting log is the basis for the LLM
extraction error taxonomy, a standalone research contribution.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ErrorCategory:
    """Standard error categories for the CTI extraction error taxonomy."""

    HALLUCINATED_ENTITY = "hallucinated_entity"
    HALLUCINATED_RELATION = "hallucinated_relation"
    TYPE_MISASSIGNMENT = "type_misassignment"
    IMPOSSIBLE_TYPE_PAIR = "impossible_type_pair"
    MALFORMED_IDENTIFIER = "malformed_identifier"
    GENERIC_PLACEHOLDER = "generic_placeholder"
    DUPLICATE_ENTITY = "duplicate_entity"
    MISSING_EVIDENCE = "missing_evidence"
    SELF_LOOP = "self_loop"
    EMPTY_FIELD = "empty_field"
    OVER_EXTRACTION = "over_extraction"
    REPAIRED_IDENTIFIER = "repaired_identifier"
    REPAIRED_ALIAS = "repaired_alias"
    CONFIDENCE_LOWERED = "confidence_lowered"


class ErrorLogEntry:
    """A single error log entry."""

    def __init__(
        self,
        category: str,
        action: str,
        triple_id: str = "",
        details: str = "",
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ):
        self.category = category
        self.action = action  # "rejected", "repaired", "flagged"
        self.triple_id = triple_id
        self.details = details
        self.source_doc_id = source_doc_id
        self.source_chunk_id = source_chunk_id

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "action": self.action,
            "triple_id": self.triple_id,
            "details": self.details,
            "source_doc_id": self.source_doc_id,
            "source_chunk_id": self.source_chunk_id,
        }


class ErrorTaxonomyLogger:
    """Accumulates and reports error taxonomy statistics."""

    def __init__(self):
        self.entries: list[ErrorLogEntry] = []

    def log(
        self,
        category: str,
        action: str,
        triple_id: str = "",
        details: str = "",
        source_doc_id: str = "",
        source_chunk_id: str = "",
    ) -> None:
        entry = ErrorLogEntry(
            category=category,
            action=action,
            triple_id=triple_id,
            details=details,
            source_doc_id=source_doc_id,
            source_chunk_id=source_chunk_id,
        )
        self.entries.append(entry)

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics for the error taxonomy."""
        category_counts = Counter(e.category for e in self.entries)
        action_counts = Counter(e.action for e in self.entries)

        return {
            "total_entries": len(self.entries),
            "by_category": dict(category_counts.most_common()),
            "by_action": dict(action_counts.most_common()),
            "rejection_rate": (
                action_counts.get("rejected", 0) / max(len(self.entries), 1)
            ),
            "repair_rate": (
                action_counts.get("repaired", 0) / max(len(self.entries), 1)
            ),
        }

    def save(self, path: str | Path) -> None:
        """Save the full error log and summary to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Full log
        log_path = path / "error_log.jsonl"
        with open(log_path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

        # Summary
        summary_path = path / "error_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(
            f"Saved {len(self.entries)} error log entries to {path}"
        )

    def reset(self) -> None:
        self.entries = []
