"""Tests for src/evaluation/llm_judge.py.

The judge is the only protocol with measured agreement against human judgement,
so its verdict parsing and its cache have to be right. A silent parse failure
reads as "zero true positives", which looks like a result rather than a bug.
"""

import json

import pytest

from src.evaluation.llm_judge import (
    CachedJudge,
    _cache_key,
    _fmt_edges,
    _fmt_entities,
    parse_verdicts,
)

EDGES = [("APT29", "uses", "Cobalt Strike"), ("Lazarus", "targets", "banks")]


class TestParseVerdicts:
    def test_plain_json_array(self):
        r = json.dumps([{"index": "p_0", "result": "TP"}, {"index": "p_1", "result": "FP"}])
        assert parse_verdicts(r) == (1, 2)

    def test_fenced_json_block(self):
        r = "Here you go:\n```json\n" + json.dumps(
            [{"index": "p_0", "result": "TP"}, {"index": "p_1", "result": "TP"}]) + "\n```\n<Fin>"
        assert parse_verdicts(r) == (2, 2)

    def test_prose_around_array_is_tolerated(self):
        r = 'Reasoning...\n[{"index":"p_0","result":"FN"}]\nDone. <Fin>'
        assert parse_verdicts(r) == (0, 1)

    def test_tp_prefix_variants_count_as_positive(self):
        r = json.dumps([{"index": "p_0", "result": "TP (supported)"},
                        {"index": "p_1", "result": "tp"}])
        assert parse_verdicts(r) == (2, 2)

    def test_entries_without_result_are_ignored(self):
        r = json.dumps([{"index": "p_0", "result": "TP"}, {"note": "no verdict"}])
        assert parse_verdicts(r) == (1, 1)

    def test_empty_and_malformed_return_zero_zero(self):
        """Must return (0,0) not (0,n) -- otherwise a parse failure silently
        becomes 'judged n edges, none correct'."""
        assert parse_verdicts("") == (0, 0)
        assert parse_verdicts("   ") == (0, 0)
        assert parse_verdicts("no json at all") == (0, 0)
        assert parse_verdicts("[not valid json") == (0, 0)
        assert parse_verdicts(json.dumps({"result": "TP"})) == (0, 0)  # dict, not list


class TestFormatting:
    def test_edges_get_indexed_prefixes(self):
        out = json.loads(_fmt_edges(EDGES, "predict_relationship"))
        assert out[0]["index"] == "predict_relationship_0"
        assert out[1]["index"] == "predict_relationship_1"
        assert out[0]["sub"] == "APT29"

    def test_entities_deduplicated_and_order_preserved(self):
        ents = json.loads(_fmt_entities(EDGES + [("APT29", "targets", "banks")]))
        names = [e["name"] for e in ents]
        assert names == ["APT29", "Cobalt Strike", "Lazarus", "banks"]

    def test_empty_edges(self):
        assert json.loads(_fmt_edges([], "p")) == []
        assert json.loads(_fmt_entities([])) == []


class TestCacheKey:
    def test_key_depends_on_model_and_content(self):
        assert _cache_key("gpt-4o-mini", "x") != _cache_key("gpt-5.4-mini", "x")
        assert _cache_key("gpt-4o-mini", "x") != _cache_key("gpt-4o-mini", "y")

    def test_key_is_stable(self):
        assert _cache_key("m", "c") == _cache_key("m", "c")


class TestCachedJudge:
    def test_cache_hit_avoids_a_call(self, tmp_path):
        j = CachedJudge(model="gpt-4o-mini", cache_dir=tmp_path)
        j._store(_cache_key("gpt-4o-mini", "hello"), "hello", '[{"index":"p_0","result":"TP"}]')
        out = j.ask("hello")
        assert j.stats["hits"] == 1
        assert j.stats["calls"] == 0
        assert parse_verdicts(out) == (1, 1)

    def test_dry_run_makes_no_call_but_counts_input(self, tmp_path):
        j = CachedJudge(model="gpt-4o-mini", cache_dir=tmp_path, dry_run=True)
        assert j.ask("some content") == ""
        assert j.stats["calls"] == 1
        assert j.stats["in_chars"] == len("some content")

    def test_dry_run_does_not_write_cache(self, tmp_path):
        j = CachedJudge(model="gpt-4o-mini", cache_dir=tmp_path, dry_run=True)
        j.ask("content")
        assert list(tmp_path.rglob("*.json")) == []

    def test_cost_estimate_shapes(self, tmp_path):
        j = CachedJudge(model="gpt-4o-mini", cache_dir=tmp_path, dry_run=True)
        j.ask("x" * 4000)
        est = j.estimate_cost()
        assert est["input_tokens"] == 1000
        assert est["usd"] > 0
        assert est["usd_if_prefix_cached"] < est["usd"]

    def test_unknown_model_reports_no_price(self, tmp_path):
        j = CachedJudge(model="some-new-model", cache_dir=tmp_path, dry_run=True)
        j.ask("x")
        assert j.estimate_cost()["usd"] is None

    def test_corrupt_cache_entry_is_treated_as_miss(self, tmp_path):
        j = CachedJudge(model="gpt-4o-mini", cache_dir=tmp_path, dry_run=True)
        key = _cache_key("gpt-4o-mini", "c")
        (j.cache_dir / f"{key}.json").write_text("{ broken")
        assert j._cached(key) is None
