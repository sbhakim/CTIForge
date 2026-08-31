"""Tests for the NLI evaluation instrument.

The scorer is exercised through a stub so the suite stays fast and offline; the
model itself is checked once, manually, in the calibration harness.
"""

import json

import pytest

from src.evaluation.nli_faithfulness import (
    NLIScorer,
    graph_entails_edge,
    verbalize,
)


class StubScorer(NLIScorer):
    """NLIScorer with the model replaced by a lookup, to test the plumbing."""

    def __init__(self, table, **kw):
        super().__init__(cache_path=None, **kw)
        self.table = table
        self.calls = 0

    def entail_probs(self, pairs):
        self.calls += len(pairs)
        return [self.table.get((p, h), 0.0) for p, h in pairs]


# -- verbalisation ---------------------------------------------------------

def test_verbalize_uses_relation_template():
    assert verbalize("APT29", "uses", "Cobalt Strike") == "APT29 uses Cobalt Strike."
    assert verbalize("3AM", "variant_of", "LockBit") == "3AM is a variant of LockBit."
    assert verbalize("Clop", "attributed_to", "FIN11") == "Clop is attributed to FIN11."


def test_verbalize_covers_every_declared_relation():
    from src.schema.relations import RelationType
    for rel in RelationType:
        out = verbalize("A", rel.value, "B")
        assert out.startswith("A ") and out.endswith("B.") or out.endswith(".")
        # A template must have fired, not the free-text fallback.
        assert rel.value.replace("_", " ") in out or rel.value in ("uses", "targets")


def test_verbalize_unknown_relation_keeps_the_string():
    # Collapsing to related_to would discard signal the model can still use.
    assert verbalize("X", "beacons to", "Y") == "X beacons to Y."


def test_verbalize_empty_relation_falls_back():
    assert verbalize("A", "", "B") == "A is related to B."


def test_verbalize_strips_whitespace():
    assert verbalize("  APT29 ", "uses", " Mimikatz ") == "APT29 uses Mimikatz."


# -- matching protocol -----------------------------------------------------

def test_graph_entails_edge_true_when_any_gold_edge_entails():
    edge = ("APT29", "uses", "Cobalt Strike")
    graph = [("Lazarus", "uses", "Mimikatz"), ("APT29", "uses", "Cobalt Strike")]
    table = {
        ("Lazarus uses Mimikatz.", "APT29 uses Cobalt Strike."): 0.01,
        ("APT29 uses Cobalt Strike.", "APT29 uses Cobalt Strike."): 0.99,
    }
    assert graph_entails_edge(StubScorer(table), edge, graph, threshold=0.5)


def test_graph_entails_edge_false_below_threshold():
    edge = ("APT29", "uses", "Cobalt Strike")
    graph = [("Lazarus", "uses", "Mimikatz")]
    table = {("Lazarus uses Mimikatz.", "APT29 uses Cobalt Strike."): 0.30}
    assert not graph_entails_edge(StubScorer(table), edge, graph, threshold=0.5)


def test_graph_entails_edge_empty_graph_is_false():
    assert not graph_entails_edge(StubScorer({}), ("A", "uses", "B"), [])


def test_graph_entails_edge_scores_every_gold_edge():
    """No lexical pre-filter: the protocol must stay independent of string
    matching, or the comparison against string matchers is contaminated."""
    scorer = StubScorer({})
    graph = [("A", "uses", "B"), ("C", "uses", "D"), ("E", "uses", "F")]
    graph_entails_edge(scorer, ("X", "uses", "Y"), graph)
    assert scorer.calls == 3


def test_threshold_is_inclusive():
    table = {("A uses B.", "A uses B."): 0.5}
    assert graph_entails_edge(StubScorer(table), ("A", "uses", "B"),
                              [("A", "uses", "B")], threshold=0.5)


# -- cache -----------------------------------------------------------------

def test_cache_key_includes_model_name(tmp_path):
    """A stale score from a different model is worse than no cache at all."""
    a = NLIScorer(model_name="model-a", cache_path=tmp_path / "c.json")
    b = NLIScorer(model_name="model-b", cache_path=tmp_path / "c.json")
    assert a._key("p", "h") != b._key("p", "h")


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "nli.json"
    s = NLIScorer(cache_path=path)
    s._cache[s._key("p", "h")] = 0.77
    s.save_cache()
    assert json.load(open(path))
    reloaded = NLIScorer(cache_path=path)
    assert reloaded.entail_probs([("p", "h")]) == [0.77]


def test_corrupt_cache_is_survivable(tmp_path):
    path = tmp_path / "nli.json"
    path.write_text("{not json")
    s = NLIScorer(cache_path=path)
    assert len(s) == 0


def test_empty_pairs_needs_no_model():
    assert NLIScorer(cache_path=None).entail_probs([]) == []
