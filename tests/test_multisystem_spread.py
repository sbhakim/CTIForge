"""Tests for eval_multisystem_spread.py.

Pins the load-path coercion and the greedy per-document matching, both of which
fail silently rather than loudly if wrong.
"""

import pytest

from benchmarks.eval_multisystem_spread import (
    _as_list,
    f1_for,
    greedy_tp,
    m_exact,
    m_name_compat,
    m_name_rel_eq,
    m_so_pair,
)


class TestAsList:
    def test_string_becomes_single_element(self):
        assert _as_list("APT29") == ["APT29"]

    def test_list_is_expanded_not_stringified(self):
        """grid_end2end emits obj=['.mallox', '.xollam'] -- two facts, not one
        unmatchable string."""
        assert _as_list([".mallox", ".xollam"]) == [".mallox", ".xollam"]

    def test_blank_and_none_drop_out(self):
        assert _as_list("") == []
        assert _as_list("   ") == []
        assert _as_list(None) == []
        assert _as_list(["a", "", "  "]) == ["a"]

    def test_scalar_non_string_is_coerced(self):
        assert _as_list(42) == ["42"]


class TestMatchers:
    GOLD = [("APT29", "uses", "Cobalt Strike")]

    def test_exact_needs_all_three(self):
        assert m_exact(("APT29", "uses", "Cobalt Strike"), self.GOLD[0])
        assert not m_exact(("APT29", "deployed", "Cobalt Strike"), self.GOLD[0])

    def test_so_pair_ignores_relation_and_direction(self):
        assert m_so_pair(("APT29", "anything at all", "Cobalt Strike"), self.GOLD[0])
        assert m_so_pair(("Cobalt Strike", "used by", "APT29"), self.GOLD[0])

    def test_compat_is_superset_of_rel_eq(self):
        e = ("APT29", "uses", "Cobalt Strike")
        if m_name_rel_eq(e, self.GOLD[0]):
            assert m_name_compat(e, self.GOLD[0])

    def test_ordering_is_monotone_in_strictness(self):
        """exact <= rel== <= compat <= so_pair on a paraphrased edge."""
        e = ("APT29 group", "leveraged", "Cobalt Strike")
        g = ("APT29", "uses", "Cobalt Strike")
        assert not m_exact(e, g)
        assert m_so_pair(e, g)


class TestGreedyTp:
    def test_each_gold_consumed_at_most_once(self):
        preds = [("A", "uses", "B"), ("A", "uses", "B")]
        golds = [("A", "uses", "B")]
        assert greedy_tp(preds, golds, m_exact) == 1

    def test_no_match(self):
        assert greedy_tp([("X", "uses", "Y")], [("A", "uses", "B")], m_exact) == 0

    def test_empty_sides(self):
        assert greedy_tp([], [("A", "uses", "B")], m_exact) == 0
        assert greedy_tp([("A", "uses", "B")], [], m_exact) == 0


class TestF1PerDocumentScoping:
    def test_predictions_do_not_match_other_documents_gold(self):
        """The cross-document leakage guard, restated for this harness."""
        preds = {"d1": [("A", "uses", "B")], "d2": [("C", "uses", "D")]}
        gold = {"d1": [("C", "uses", "D")], "d2": [("A", "uses", "B")]}
        assert f1_for(preds, gold, match=m_exact) == 0.0

    def test_perfect_match(self):
        preds = {"d1": [("A", "uses", "B")]}
        gold = {"d1": [("A", "uses", "B")]}
        assert f1_for(preds, gold, match=m_exact) == pytest.approx(1.0)

    def test_missing_document_counts_gold_as_missed(self):
        preds = {}
        gold = {"d1": [("A", "uses", "B")]}
        assert f1_for(preds, gold, match=m_exact) == 0.0
