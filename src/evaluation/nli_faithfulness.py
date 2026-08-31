"""Natural-language inference as an evaluation instrument (Module: evaluation).

Two distinct uses, kept separate because they answer different questions and
only one of them can be validated against the GRID human-adjudicated set:

  1. NLI *as a matching protocol*. Premise is a gold edge, hypothesis is a
     predicted edge: does the gold graph entail the prediction? This is the same
     question every lexical and embedding protocol answers, so it can be scored
     against human judgement on GRID's calibration items alongside them.

  2. NLI *as gold-free faithfulness*. Premise is the source report text,
     hypothesis is a predicted triple: does the document support the claim? This
     needs no annotations at all, which is the point, but it also means the
     GRID items cannot validate it -- they carry edge-to-edge match labels, not
     text-to-edge entailment labels.

Both run on a locally cached MNLI model, so scoring is offline, free, and
deterministic under a fixed model revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MODEL = "facebook/bart-large-mnli"

# Verbalisation templates for the 12-relation inventory. Each renders a triple
# as a plain declarative clause; MNLI models are trained on ordinary prose, so
# "APT29 uses Cobalt Strike" is a far better hypothesis than "APT29 | uses |
# Cobalt Strike". Templates are deliberately weak in commitment: they assert the
# relation and nothing else.
_TEMPLATES = {
    "uses": "{s} uses {o}.",
    "targets": "{s} targets {o}.",
    "exploits": "{s} exploits {o}.",
    "delivers": "{s} delivers {o}.",
    "communicates_with": "{s} communicates with {o}.",
    "drops": "{s} drops {o}.",
    "attributed_to": "{s} is attributed to {o}.",
    "associated_with": "{s} is associated with {o}.",
    "variant_of": "{s} is a variant of {o}.",
    "located_in": "{s} is located in {o}.",
    "mitigated_by": "{s} is mitigated by {o}.",
    "related_to": "{s} is related to {o}.",
}


def verbalize(subject: str, relation: str, obj: str) -> str:
    """Render a triple as a declarative sentence for use as premise or hypothesis."""
    s = (subject or "").strip()
    o = (obj or "").strip()
    rel = (relation or "").strip().lower().replace(" ", "_")
    template = _TEMPLATES.get(rel)
    if template is None:
        # Unknown or free-text relation: use it verbatim rather than collapsing
        # to related_to, which would discard signal the model can still use.
        readable = rel.replace("_", " ") or "is related to"
        return f"{s} {readable} {o}."
    return template.format(s=s, o=o)


class NLIScorer:
    """Batched MNLI entailment with an on-disk cache.

    Returns P(entailment) for each (premise, hypothesis) pair. The cache is keyed
    on the model name plus the pair, so switching models never returns a stale
    score.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: str | Path | None = "output/nli_cache.json",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._device = device
        self._tokenizer = None
        self._model = None
        self._entail_idx: int | None = None

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.load(open(self.cache_path))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    # -- model -------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self._device)
        self._model = model

        # Never assume label order. MNLI checkpoints differ, and reading it off
        # the config is the difference between measuring entailment and
        # measuring contradiction.
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        if "entailment" not in labels:
            raise RuntimeError(
                f"{self.model_name} does not expose an 'entailment' label; "
                f"got {model.config.id2label}"
            )
        self._entail_idx = labels["entailment"]

    def _key(self, premise: str, hypothesis: str) -> str:
        h = hashlib.sha1(f"{self.model_name}\x00{premise}\x00{hypothesis}".encode()).hexdigest()
        return h

    # -- scoring -----------------------------------------------------------

    def entail_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis), cached across calls."""
        if not pairs:
            return []

        keys = [self._key(p, h) for p, h in pairs]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]

        if todo:
            import torch
            self._ensure_model()
            for start in range(0, len(todo), self.batch_size):
                idxs = todo[start:start + self.batch_size]
                prem = [pairs[i][0] for i in idxs]
                hyp = [pairs[i][1] for i in idxs]
                enc = self._tokenizer(
                    prem, hyp, return_tensors="pt", truncation=True,
                    padding=True, max_length=512,
                ).to(self._device)
                with torch.no_grad():
                    logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self._entail_idx]
                for i, p in zip(idxs, probs.tolist()):
                    self._cache[keys[i]] = float(p)

        return [self._cache[k] for k in keys]

    def entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        return self.entail_probs([(premise, hypothesis)])[0] >= threshold

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self._cache, open(self.cache_path, "w"))

    def __len__(self) -> int:
        return len(self._cache)


def graph_entails_edge(
    scorer: NLIScorer,
    edge: tuple[str, str, str],
    graph: Iterable[tuple[str, str, str]],
    threshold: float = 0.5,
) -> bool:
    """Matching protocol: does any edge of `graph` entail `edge`?

    Every gold edge is scored -- no lexical pre-filter -- so this protocol stays
    independent of the string matchers it is being compared against.
    """
    graph = list(graph)
    if not graph:
        return False
    hypothesis = verbalize(*edge)
    pairs = [(verbalize(*g), hypothesis) for g in graph]
    return max(scorer.entail_probs(pairs)) >= threshold
