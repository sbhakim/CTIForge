"""Retrieve few-shot extraction examples from local CTI-Nexus annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.schema.relations import Triple
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


@dataclass
class ExampleRecord:
    doc_id: str
    text: str
    output: str
    token_set: set[str]
    output_token_set: set[str]


class FewShotExampleRetriever:
    """Selects few-shot examples by lexical overlap with the current chunk."""

    def __init__(
        self,
        annotations_dir: str | Path,
        max_examples: int = 3,
        max_chars_per_example: int = 1200,
        max_triples_per_example: int = 6,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.max_examples = max(max_examples, 0)
        self.max_chars_per_example = max(max_chars_per_example, 200)
        self.max_triples_per_example = max(max_triples_per_example, 1)
        self.examples = self._load_examples()

    def _load_examples(self) -> list[ExampleRecord]:
        if self.max_examples == 0 or not self.annotations_dir.exists():
            return []

        records: list[ExampleRecord] = []
        for path in sorted(self.annotations_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("Skipping few-shot example file %s: %s", path.name, exc)
                continue

            text = " ".join((data.get("text") or "").split())
            if not text:
                continue
            triples = data.get("explicit_triplets", [])[: self.max_triples_per_example]
            if not triples:
                continue

            # Build entity type lookup from annotations
            entity_types = {}
            for ent in data.get("entities", []):
                entity_types[ent.get("entity_name", "")] = ent.get("entity_type", "Other")

            formatted_triples = []
            for triple in triples:
                subj = triple.get("subject", "")
                obj = triple.get("object", "")
                formatted_triples.append(
                    {
                        "subject": subj,
                        "subject_type": entity_types.get(subj, "Other"),
                        "relation": triple.get("relation", "related_to"),
                        "object": obj,
                        "object_type": entity_types.get(obj, "Other"),
                        "evidence": "",
                    }
                )

            example_text = text[: self.max_chars_per_example]
            output_json = json.dumps({"triples": formatted_triples}, indent=2)
            records.append(
                ExampleRecord(
                    doc_id=path.stem,
                    text=example_text,
                    output=output_json,
                    token_set=_tokenize(example_text),
                    output_token_set=_tokenize(output_json),
                )
            )

        logger.info("Loaded %s few-shot example records", len(records))
        return records

    def retrieve(self, text: str, exclude_doc_id: str = "") -> list[dict[str, str]]:
        """Return the top lexical-overlap few-shot examples for a chunk."""
        if not self.examples or self.max_examples == 0:
            return []

        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        scored: list[tuple[float, ExampleRecord]] = []
        for example in self.examples:
            if exclude_doc_id and example.doc_id == exclude_doc_id:
                continue
            text_overlap = len(query_tokens & example.token_set)
            output_overlap = len(query_tokens & example.output_token_set)
            if text_overlap == 0 and output_overlap == 0:
                continue
            text_union = len(query_tokens | example.token_set)
            output_union = len(query_tokens | example.output_token_set)
            text_score = text_overlap / text_union if text_union else 0.0
            output_score = output_overlap / output_union if output_union else 0.0
            brevity_bonus = 1.0 / max(len(example.text), 1)
            score = (0.7 * text_score) + (0.25 * output_score) + (25 * brevity_bonus)
            scored.append((score, example))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [record for score, record in scored[: self.max_examples] if score > 0]
        return [{"text": record.text, "output": record.output} for record in selected]
