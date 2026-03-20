"""Embedding-based entity alignment (inspired by CTI-Nexus EA stage).

After canonicalization, entities with different surface forms but the same
meaning may still exist as separate nodes (e.g., "APT28" vs "Fancy Bear"
when no alias table entry exists). This module uses sentence-transformer
embeddings to cluster similar entities within the same type and merge them,
boosting both precision and recall in downstream evaluation.
"""

from __future__ import annotations

from collections import defaultdict

from src.schema.entities import EntityType, Entity
from src.schema.relations import Triple, ValidationStatus
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class EntityAligner:
    """Merge similar entities using embedding cosine similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._embedder = None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for entity alignment: %s", self.embedding_model)
        self._embedder = SentenceTransformer(self.embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_embedder()
        import numpy as np
        embeddings = self._embedder.encode(texts, convert_to_numpy=True)
        return embeddings

    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def align_triples(self, triples: list[Triple]) -> list[Triple]:
        """Find similar entities and merge them by rewriting triple names.

        Groups entities by type, computes pairwise embedding similarity,
        clusters entities above threshold, and rewrites all triples to use
        the canonical (most frequent) name from each cluster.
        """
        # Collect all unique entity names grouped by type
        active = [t for t in triples if t.validation_status != ValidationStatus.REJECTED]
        if not active:
            return triples

        type_to_names: dict[EntityType, set[str]] = defaultdict(set)
        for t in active:
            type_to_names[t.subject_type].add(t.subject)
            type_to_names[t.object_type].add(t.object)

        # Build merge map: old_name -> canonical_name (within same type)
        merge_map: dict[tuple[EntityType, str], str] = {}
        merged_count = 0

        for etype, names in type_to_names.items():
            names_list = sorted(names)
            if len(names_list) < 2:
                continue

            # Compute embeddings
            embeddings = self._embed(names_list)

            # Build clusters via greedy similarity matching
            visited = set()
            clusters: list[list[int]] = []

            for i in range(len(names_list)):
                if i in visited:
                    continue
                cluster = [i]
                visited.add(i)
                for j in range(i + 1, len(names_list)):
                    if j in visited:
                        continue
                    sim = self._cosine_sim(embeddings[i], embeddings[j])
                    if sim >= self.similarity_threshold:
                        cluster.append(j)
                        visited.add(j)
                clusters.append(cluster)

            # For each cluster with >1 member, pick the most frequent name
            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                # Count frequency of each name in triples
                cluster_names = [names_list[idx] for idx in cluster]
                freq: dict[str, int] = defaultdict(int)
                for t in active:
                    if t.subject_type == etype and t.subject in cluster_names:
                        freq[t.subject] += 1
                    if t.object_type == etype and t.object in cluster_names:
                        freq[t.object] += 1

                canonical = max(cluster_names, key=lambda n: freq.get(n, 0))
                for name in cluster_names:
                    if name != canonical:
                        merge_map[(etype, name)] = canonical
                        merged_count += 1

        if merged_count == 0:
            return triples

        logger.info(
            "Entity alignment: merged %d entity names into canonical forms",
            merged_count,
        )

        # Rewrite triples and deduplicate post-merge duplicates
        result = []
        seen: set[tuple[str, str, str]] = set()
        deduped = 0
        for t in triples:
            if t.validation_status == ValidationStatus.REJECTED:
                result.append(t)
                continue
            t_copy = t.model_copy()
            key_s = (t.subject_type, t.subject)
            key_o = (t.object_type, t.object)
            if key_s in merge_map:
                t_copy.subject = merge_map[key_s]
            if key_o in merge_map:
                t_copy.object = merge_map[key_o]
            # Deduplicate: after merging, some triples may become identical
            dedup_key = (
                t_copy.subject.lower().strip(),
                t_copy.relation.value,
                t_copy.object.lower().strip(),
            )
            if dedup_key in seen:
                deduped += 1
                continue
            seen.add(dedup_key)
            result.append(t_copy)

        if deduped:
            logger.info("Entity alignment: removed %d post-merge duplicate triples", deduped)

        return result
