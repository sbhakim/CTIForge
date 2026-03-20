"""Alias tables for CTI entity canonicalization.

Loads alias mappings from MITRE ATT&CK data (groups, software) and
provides lookup for canonical name resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _load_nested_contents(entry: dict[str, Any]) -> dict[str, Any]:
    """Expand nested JSON stored in the CTIArena-style `contents` field."""
    contents = entry.get("contents")
    if isinstance(contents, str):
        try:
            nested = json.loads(contents)
        except json.JSONDecodeError:
            return {}
        if isinstance(nested, dict):
            return nested
    elif isinstance(contents, dict):
        return contents
    return {}


class AliasTable:
    """Bidirectional alias lookup table."""

    def __init__(self):
        # alias (lowercase) -> canonical name
        self._alias_to_canonical: dict[str, str] = {}
        # canonical name (lowercase) -> set of aliases
        self._canonical_to_aliases: dict[str, set[str]] = {}

    def add(self, canonical: str, aliases: list[str]) -> None:
        """Register a canonical name and its aliases."""
        canon_key = canonical.lower().strip()
        if canon_key not in self._canonical_to_aliases:
            self._canonical_to_aliases[canon_key] = set()

        self._alias_to_canonical[canon_key] = canonical
        for alias in aliases:
            alias_key = alias.lower().strip()
            if alias_key:
                self._alias_to_canonical[alias_key] = canonical
                self._canonical_to_aliases[canon_key].add(alias)

    def resolve(self, name: str) -> str | None:
        """Look up the canonical name for a given name or alias.

        Returns the canonical name if found, None otherwise.
        """
        key = name.lower().strip()
        return self._alias_to_canonical.get(key)

    def get_aliases(self, canonical: str) -> set[str]:
        """Get all known aliases for a canonical name."""
        key = canonical.lower().strip()
        return self._canonical_to_aliases.get(key, set())

    @property
    def size(self) -> int:
        return len(self._alias_to_canonical)


def load_mitre_aliases(mitre_jsonl_path: str | Path) -> tuple[AliasTable, AliasTable]:
    """Load MITRE ATT&CK group and software aliases from CTIArena mitre.jsonl.

    Returns (group_aliases, software_aliases)
    """
    group_table = AliasTable()
    software_table = AliasTable()

    path = Path(mitre_jsonl_path)
    if not path.exists():
        logger.warning(f"MITRE data not found at {path}, using empty alias tables")
        return group_table, software_table

    parsed_entries = 0
    candidate_entries = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            parsed_entries += 1
            nested = _load_nested_contents(entry)

            entry_type = entry.get("type", "") or entry.get("object_type", "")
            name = (
                entry.get("name", "")
                or nested.get("name", "")
                or entry.get("title", "")
            )
            aliases = entry.get("aliases", []) or nested.get("aliases", []) or nested.get(
                "x_mitre_aliases", []
            )

            if not name:
                continue

            if entry_type in ("intrusion-set", "threat-actor"):
                candidate_entries += 1
                group_table.add(name, aliases)
            elif entry_type in ("malware", "tool"):
                candidate_entries += 1
                software_table.add(name, aliases)

    if candidate_entries == 0:
        logger.info(
            "Loaded MITRE dataset with %s parsed entries but no group/software alias "
            "records; alias tables remain empty for this corpus",
            parsed_entries,
        )
    else:
        logger.info(
            f"Loaded MITRE aliases: {group_table.size} group entries, "
            f"{software_table.size} software entries"
        )
    return group_table, software_table
