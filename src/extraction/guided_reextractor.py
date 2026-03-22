"""Type-constraint-guided re-extraction (neuro-symbolic recall booster).

After initial extraction, this module:
1. Collects all extracted entities per chunk
2. Uses the symbolic type constraint table to identify entity pairs
   that COULD have a valid relationship but DON'T have one yet
3. Asks the LLM specifically about those pairs in a focused prompt
4. Returns new triples for validation

This is deeply neuro-symbolic: the symbolic layer (type constraints)
actively GUIDES where the neural layer should focus its attention,
rather than relying on the LLM to find everything in one pass.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from itertools import product

import jinja2
from litellm import completion

from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.schema.type_constraints import TYPE_PAIR_CONSTRAINTS
from src.schema.graph_schema import Chunk
from src.extraction.output_parser import parse_extraction_response
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Relations worth checking for guided re-extraction (skip catch-alls)
_INTERESTING_RELATIONS = frozenset({
    RelationType.USES, RelationType.TARGETS, RelationType.EXPLOITS,
    RelationType.DELIVERS, RelationType.DROPS, RelationType.ATTRIBUTED_TO,
    RelationType.VARIANT_OF, RelationType.LOCATED_IN,
    RelationType.COMMUNICATES_WITH, RelationType.MITIGATED_BY,
})

# Max entity pairs to check per chunk (controls LLM cost)
_MAX_PAIRS_PER_CHUNK = 8


class GuidedReextractor:
    """Use symbolic type constraints to guide targeted re-extraction."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        prompts_dir: str | Path = "prompts",
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        prompts_path = Path(prompts_dir)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_path)),
            autoescape=False,
        )
        self.template = self.jinja_env.get_template("guided_reextraction.jinja")

    def _collect_entities(
        self, triples: list[Triple], chunk_id: str
    ) -> dict[str, EntityType]:
        """Collect unique entities from triples belonging to a chunk."""
        entities: dict[str, EntityType] = {}
        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                continue
            if t.source_chunk_id != chunk_id:
                continue
            entities[t.subject] = t.subject_type
            entities[t.object] = t.object_type
        return entities

    def _get_existing_pairs(
        self, triples: list[Triple], chunk_id: str
    ) -> set[frozenset[str]]:
        """Get existing (subject, object) pairs for dedup."""
        pairs: set[frozenset[str]] = set()
        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                continue
            if t.source_chunk_id != chunk_id:
                continue
            pairs.add(frozenset({t.subject.lower(), t.object.lower()}))
        return pairs

    def _find_candidate_pairs(
        self,
        entities: dict[str, EntityType],
        existing_pairs: set[frozenset[str]],
    ) -> list[dict]:
        """Find entity pairs that satisfy type constraints but have no relationship."""
        entity_list = list(entities.items())
        candidates = []

        for i, (name_a, type_a) in enumerate(entity_list):
            for j, (name_b, type_b) in enumerate(entity_list):
                if i >= j:
                    continue
                # Skip if already related
                pair_key = frozenset({name_a.lower(), name_b.lower()})
                if pair_key in existing_pairs:
                    continue

                # Check if any interesting relation allows this type pair
                for rel, (valid_subj, valid_obj) in TYPE_PAIR_CONSTRAINTS.items():
                    if rel not in _INTERESTING_RELATIONS:
                        continue
                    # Check both directions
                    if (type_a in valid_subj and type_b in valid_obj):
                        candidates.append({
                            "subject": name_a,
                            "subject_type": type_a.value,
                            "object": name_b,
                            "object_type": type_b.value,
                            "possible_relation": rel.value,
                        })
                        break
                    elif (type_b in valid_subj and type_a in valid_obj):
                        candidates.append({
                            "subject": name_b,
                            "subject_type": type_b.value,
                            "object": name_a,
                            "object_type": type_a.value,
                            "possible_relation": rel.value,
                        })
                        break

        return candidates[:_MAX_PAIRS_PER_CHUNK]

    def guided_reextract(
        self,
        chunks: list[Chunk],
        existing_triples: list[Triple],
        source_doc_id: str = "",
    ) -> list[Triple]:
        """Run guided re-extraction on all chunks.

        Returns new triples found by the focused entity-pair queries.
        """
        if self.provider.startswith("local_") or self.model.startswith("huggingface/"):
            logger.debug("Skipping guided re-extraction for local model")
            return []

        all_new: list[Triple] = []
        existing_keys = {
            (t.subject.lower(), t.relation.value, t.object.lower())
            for t in existing_triples
            if t.validation_status != ValidationStatus.REJECTED
        }

        for chunk in chunks:
            entities = self._collect_entities(existing_triples, chunk.chunk_id)
            if len(entities) < 2:
                continue

            existing_pairs = self._get_existing_pairs(existing_triples, chunk.chunk_id)
            candidates = self._find_candidate_pairs(entities, existing_pairs)

            if not candidates:
                continue

            # Build prompt
            entity_list = [
                {"name": name, "type": etype.value}
                for name, etype in entities.items()
            ]
            prompt = self.template.render(
                text=chunk.text,
                entities=entity_list,
                pairs=candidates,
            )

            try:
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a CTI analyst. Output valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "Guided re-extraction failed for chunk %s: %s",
                    chunk.chunk_id, e,
                )
                continue

            new_triples = parse_extraction_response(
                raw_response=raw,
                source_doc_id=source_doc_id,
                source_chunk_id=chunk.chunk_id,
                source_text=chunk.text,
            )

            # Deduplicate against existing triples
            for t in new_triples:
                key = (t.subject.lower(), t.relation.value, t.object.lower())
                key_swap = (t.object.lower(), t.relation.value, t.subject.lower())
                if key not in existing_keys and key_swap not in existing_keys:
                    # Mark as guided re-extraction for provenance
                    t_copy = t.model_copy()
                    t_copy.confidence = 0.75  # Slightly lower baseline confidence
                    all_new.append(t_copy)
                    existing_keys.add(key)

        if all_new:
            logger.info(
                "Guided re-extraction: found %d new triples across %d chunks",
                len(all_new), len(chunks),
            )

        return all_new
