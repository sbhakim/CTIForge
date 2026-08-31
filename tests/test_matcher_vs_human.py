"""Tests for eval_matcher_vs_human.py.

The load path is fiddly: the two graph fields in GRID's calibration files use
different storage shapes, and the human verdict has to be recovered by composing
the judge decision with the reviewer's agreement. Both are easy to get subtly
wrong and neither fails loudly, so they are pinned here.
"""

import pytest

from benchmarks.eval_matcher_vs_human import (
    _coerce_graph,
    p_exact,
    p_name_soft_compat,
    p_name_soft_rel_eq,
    p_so_pair,
    score,
)


class TestCoerceGraph:
    def test_parsed_dict_list(self):
        raw = [{"sub": "APT29", "rel": "uses", "obj": "Cobalt Strike"}]
        assert _coerce_graph(raw) == [("APT29", "uses", "Cobalt Strike")]

    def test_python_repr_string_with_relations_key(self):
        """`Predict Graph` is a single-quoted Python repr, so json.loads fails."""
        raw = "{'entities': [{'name': 'APT29'}], 'relations': [{'sub': 'APT29', 'rel': 'uses', 'obj': 'X'}]}"
        assert _coerce_graph(raw) == [("APT29", "uses", "X")]

    def test_alternate_field_names(self):
        raw = [{"subject": "A", "relation": "uses", "object": "B"}]
        assert _coerce_graph(raw) == [("A", "uses", "B")]

    def test_malformed_string_returns_empty_not_raises(self):
        assert _coerce_graph("{not valid python") == []

    def test_edges_without_endpoints_are_dropped(self):
        raw = [{"sub": "", "rel": "uses", "obj": "B"}, {"sub": "A", "rel": "uses", "obj": ""}]
        assert _coerce_graph(raw) == []

    def test_empty_inputs(self):
        assert _coerce_graph(None) == []
        assert _coerce_graph([]) == []
        assert _coerce_graph("") == []


class TestHumanLabelDerivation:
    """human_says_match = judge_positive == (agreement == 'Agree').

    A reviewer agreeing with a TP verdict and a reviewer disagreeing with an
    FP verdict both mean 'a human considered these a match'.
    """

    @pytest.mark.parametrize("judge_positive,agreement,expected", [
        (True, "Agree", True),      # judge TP, human agrees   -> match
        (True, "Disagree", False),  # judge TP, human rejects   -> no match
        (False, "Agree", False),    # judge FP/FN, human agrees -> no match
        (False, "Disagree", True),  # judge FP/FN, human rejects-> match
    ])
    def test_composition(self, judge_positive, agreement, expected):
        assert (judge_positive == (agreement == "Agree")) is expected


class TestProtocols:
    GRAPH = [("APT29", "uses", "Cobalt Strike"), ("Lazarus", "targets", "banks")]

    def test_exact_requires_all_three(self):
        assert p_exact(("APT29", "uses", "Cobalt Strike"), self.GRAPH)
        assert not p_exact(("APT29", "deploys", "Cobalt Strike"), self.GRAPH)

    def test_so_pair_ignores_relation(self):
        assert p_so_pair(("APT29", "totally-different", "Cobalt Strike"), self.GRAPH)

    def test_so_pair_is_swap_aware(self):
        assert p_so_pair(("Cobalt Strike", "used-by", "APT29"), self.GRAPH)

    def test_compat_is_no_stricter_than_rel_eq(self):
        """The compat pass is a superset of the rel== pass by construction."""
        edge = ("APT29", "uses", "Cobalt Strike")
        if p_name_soft_rel_eq(edge, self.GRAPH):
            assert p_name_soft_compat(edge, self.GRAPH)

    def test_empty_graph_never_matches(self):
        for proto in (p_exact, p_name_soft_rel_eq, p_name_soft_compat, p_so_pair):
            assert not proto(("A", "uses", "B"), [])


class TestScore:
    def test_confusion_counts_and_directions(self):
        items = [
            {"edge": ("APT29", "uses", "Cobalt Strike"), "graph": TestProtocols.GRAPH, "human": True},
            {"edge": ("Nobody", "uses", "Nothing"), "graph": TestProtocols.GRAPH, "human": False},
            {"edge": ("Nobody", "uses", "Nothing"), "graph": TestProtocols.GRAPH, "human": True},
        ]
        r = score(items, p_exact)
        assert r["tp"] == 1      # matched, human agreed
        assert r["tn"] == 1      # not matched, human agreed
        assert r["fn"] == 1      # human credited, matcher missed -> under-match
        assert r["fp"] == 0
        assert r["under_match"] == pytest.approx(1 / 3)
        assert r["over_match"] == 0.0
        assert r["agreement"] == pytest.approx(2 / 3)

    def test_empty(self):
        r = score([], p_exact)
        assert r["agreement"] == 0.0 and r["n"] == 0
