"""Link prediction for disconnected subgraph components (inspired by CTI-Nexus LP).

After initial extraction, the knowledge graph may have disconnected components
- groups of entities with no path between them. This module identifies the
main entity in each disconnected subgraph and asks the LLM to infer
relationships between them, boosting recall.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import jinja2
from litellm import completion

from src.schema.relations import Triple, RelationType, ValidationStatus, normalize_relation
from src.schema.entities import EntityType
from src.extraction.output_parser import parse_extraction_response
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class LinkPredictor:
    """Infer missing relationships between disconnected subgraph components."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_predictions: int = 5,
        prompts_dir: str | Path = "prompts",
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_predictions = max_predictions

        prompts_path = Path(prompts_dir)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_path)),
            autoescape=False,
        )
        self.template = self.jinja_env.get_template("link_prediction.jinja")

    def _build_graph(self, triples: list[Triple]) -> dict[str, set[str]]:
        """Build an undirected adjacency list from active triples."""
        adj: dict[str, set[str]] = defaultdict(set)
        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                continue
            s_key = f"{t.subject_type.value}::{t.subject}"
            o_key = f"{t.object_type.value}::{t.object}"
            adj[s_key].add(o_key)
            adj[o_key].add(s_key)
        return adj

    def _find_components(self, adj: dict[str, set[str]]) -> list[set[str]]:
        """Find connected components via BFS."""
        visited: set[str] = set()
        components: list[set[str]] = []

        for node in adj:
            if node in visited:
                continue
            component: set[str] = set()
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                for neighbor in adj.get(curr, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        return components

    def _get_main_node(self, component: set[str], adj: dict[str, set[str]]) -> str:
        """Return the node with highest degree in a component."""
        return max(component, key=lambda n: len(adj.get(n, set())))

    def _parse_key(self, key: str) -> tuple[str, str]:
        """Parse 'EntityType::name' back to (type_str, name)."""
        parts = key.split("::", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def predict_links(
        self,
        triples: list[Triple],
        doc_text: str,
        source_doc_id: str = "",
    ) -> list[Triple]:
        """Find disconnected components and infer missing links.

        Returns new triples that connect previously disconnected subgraphs.
        Limited to max_predictions to control cost.
        """
        active = [t for t in triples if t.validation_status != ValidationStatus.REJECTED]
        if len(active) < 2:
            return []

        adj = self._build_graph(active)
        components = self._find_components(adj)

        if len(components) <= 1:
            logger.info("Link prediction: graph is already connected, no predictions needed")
            return []

        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        main_component = components[0]
        topic_node = self._get_main_node(main_component, adj)

        # For each smaller component, try to link its main node to the topic node
        predictions: list[Triple] = []
        pairs_tried = 0

        for comp in components[1:]:
            if pairs_tried >= self.max_predictions:
                break

            comp_main = self._get_main_node(comp, adj)
            topic_type, topic_name = self._parse_key(topic_node)
            comp_type, comp_name = self._parse_key(comp_main)

            prompt = self.template.render(
                text=doc_text[:4000],  # Limit context length
                existing_triples=active,
                entity_a=topic_name,
                type_a=topic_type,
                entity_b=comp_name,
                type_b=comp_type,
            )

            try:
                if self.provider.startswith("local_") or self.model.startswith("huggingface/"):
                    # Skip LP for local models, too slow and unreliable
                    logger.debug("Skipping LP for local model")
                    continue

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
                logger.warning("LP call failed for %s <-> %s: %s", topic_name, comp_name, e)
                pairs_tried += 1
                continue

            # Parse the LLM response
            triple = self._parse_lp_response(raw, source_doc_id, topic_name, comp_name)
            if triple:
                predictions.append(triple)
                logger.info(
                    "LP inferred: (%s, %s, %s)",
                    triple.subject, triple.relation.value, triple.object,
                )

            pairs_tried += 1

        if predictions:
            logger.info("Link prediction: inferred %d new relationships", len(predictions))

        return predictions

    def _parse_lp_response(
        self,
        raw: str,
        source_doc_id: str,
        expected_a: str,
        expected_b: str,
    ) -> Triple | None:
        """Parse LLM response into a Triple, with hallucination guard."""
        # Clean markdown fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r"\{[^{}]+\}", raw)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        if data.get("skip"):
            return None

        subj = data.get("subject", "").strip()
        obj = data.get("object", "").strip()
        rel = data.get("relation", "related_to").strip()
        evidence = data.get("evidence", "")

        if not subj or not obj:
            return None

        # Hallucination guard: subject and object must be the expected entities
        expected = {expected_a.lower(), expected_b.lower()}
        actual = {subj.lower(), obj.lower()}
        if not actual.intersection(expected):
            logger.debug("LP hallucination: predicted entities don't match expected")
            return None

        # Parse types
        subj_type_str = data.get("subject_type", "Other")
        obj_type_str = data.get("object_type", "Other")

        try:
            subj_type = EntityType(subj_type_str)
        except ValueError:
            subj_type = EntityType.OTHER
        try:
            obj_type = EntityType(obj_type_str)
        except ValueError:
            obj_type = EntityType.OTHER

        relation = normalize_relation(rel)

        return Triple(
            subject=subj,
            subject_type=subj_type,
            relation=relation,
            object=obj,
            object_type=obj_type,
            evidence_text=evidence if isinstance(evidence, str) else "",
            source_doc_id=source_doc_id,
            source_chunk_id="link_prediction",
            confidence=0.7,  # Lower confidence for inferred links
            validation_status=ValidationStatus.RAW,
        )
