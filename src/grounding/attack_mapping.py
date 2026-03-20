"""ATT&CK technique grounding for extracted triples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _normalize_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _load_nested_contents(entry: dict[str, Any]) -> dict[str, Any]:
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


class AttackMapper:
    """Maps extracted behaviors to MITRE ATT&CK technique IDs."""

    def __init__(self, mitre_jsonl_path: str | Path | None = None):
        self.techniques: dict[str, dict] = {}  # technique_id -> {name, description, tactic}
        self.name_to_id: dict[str, str] = {}   # lowercase name -> technique_id
        self.normalized_name_to_id: dict[str, str] = {}

        if mitre_jsonl_path:
            self._load_mitre(mitre_jsonl_path)

    def _load_mitre(self, path: str | Path) -> None:
        """Load ATT&CK techniques from CTIArena mitre.jsonl."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"MITRE data not found: {path}")
            return

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                nested = _load_nested_contents(entry)
                etype = entry.get("type", "") or entry.get("object_type", "")
                tech_id = entry.get("mitre_id")

                if etype and etype != "attack-pattern":
                    continue

                if not tech_id:
                    for ref in entry.get("external_references", []) + nested.get(
                        "external_references", []
                    ):
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id")
                            break

                if not tech_id:
                    continue

                name = entry.get("name", "") or nested.get("name", "") or entry.get("title", "")
                if not name:
                    continue
                description = (
                    entry.get("description", "")
                    or nested.get("description", "")
                )[:200]

                # Extract tactic from kill_chain_phases
                tactics = []
                for phase in entry.get("kill_chain_phases", []) + nested.get(
                    "kill_chain_phases", []
                ):
                    if phase.get("kill_chain_name") == "mitre-attack":
                        tactics.append(phase.get("phase_name", ""))

                self.techniques[tech_id] = {
                    "name": name,
                    "description": description,
                    "tactics": tactics,
                }
                self.name_to_id[name.lower()] = tech_id
                self.normalized_name_to_id[_normalize_name(name)] = tech_id

        logger.info(f"Loaded {len(self.techniques)} ATT&CK techniques")

    def lookup_by_name(self, name: str) -> str | None:
        """Look up a technique ID by exact name match."""
        exact = self.name_to_id.get(name.lower().strip())
        if exact:
            return exact
        return self.lookup_by_name_fuzzy(name)

    def lookup_by_name_fuzzy(self, name: str) -> str | None:
        """Look up a technique ID by normalized exact or token-overlap match."""
        normalized = _normalize_name(name)
        if not normalized:
            return None

        direct = self.normalized_name_to_id.get(normalized)
        if direct:
            return direct

        query_tokens = set(normalized.split())
        if not query_tokens:
            return None

        best_id = None
        best_score = 0.0
        for candidate_name, tech_id in self.normalized_name_to_id.items():
            candidate_tokens = set(candidate_name.split())
            overlap = len(query_tokens & candidate_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens | candidate_tokens), 1)
            if score > best_score:
                best_score = score
                best_id = tech_id

        if best_score >= 0.6:
            return best_id
        return None

    def lookup_by_id(self, tech_id: str) -> dict | None:
        """Get technique details by ID."""
        return self.techniques.get(tech_id)

    def validate_technique_id(self, tech_id: str) -> bool:
        """Check if a technique ID exists in the loaded ATT&CK data."""
        return tech_id in self.techniques
