"""Tests for per-document metric scoping (src/evaluation/per_document.py).

Regression guard: pooling every document's predictions and gold into flat lists
before matching lets a prediction from document A be credited against a gold
triple from document B. Measured on the 149-doc GPT-4o caches, pooling inflated
CTIForge triplet TP from 795 to 846 (+6.4%, +0.031 F1) and CTI-Nexus from
1067 to 1130 (+5.9%, +0.032 F1).
"""

import pytest

from src.evaluation.per_document import score_per_document, score_pooled
from src.schema.entities import EntityType
from src.schema.relations import Triple, RelationType


def _t(subject, relation, obj):
    return Triple(
        subject=subject,
        subject_type=EntityType.THREAT_ACTOR,
        relation=relation,
        object=obj,
        object_type=EntityType.MALWARE,
        evidence_text=f"{subject} {relation.value} {obj}",
    )


def _exact_matcher(pred, gold):
    """Minimal exact (subject, relation, object) matcher with the shared contract."""
    gold_keys = [(g.subject, g.relation.value, g.object) for g in gold]
    used = set()
    tp = 0
    for p in pred:
        key = (p.subject, p.relation.value, p.object)
        for i, g in enumerate(gold_keys):
            if i in used:
                continue
            if key == g:
                used.add(i)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold) - tp
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


class TestCrossDocumentLeakage:
    """The core bug: gold from one document crediting a prediction in another."""

    def setup_method(self):
        # doc_a predicts a triple that is gold only in doc_b, and vice versa.
        # Under per-document scoring both are false positives. Under pooling
        # they match each other's gold and are wrongly credited.
        self.pred = {
            "doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")],
            "doc_b": [_t("Lazarus", RelationType.USES, "AppleJeus")],
        }
        self.gold = {
            "doc_a": [_t("Lazarus", RelationType.USES, "AppleJeus")],
            "doc_b": [_t("APT29", RelationType.USES, "Cobalt Strike")],
        }

    def test_per_document_rejects_cross_document_match(self):
        result = score_per_document(self.pred, self.gold, _exact_matcher)
        assert result["tp"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 2
        assert result["f1"] == 0.0
        assert result["scope"] == "per_document"

    def test_pooled_wrongly_credits_cross_document_match(self):
        result = score_pooled(self.pred, self.gold, _exact_matcher)
        assert result["tp"] == 2  # both credited against the wrong document
        assert result["scope"] == "pooled"

    def test_pooling_inflates_relative_to_per_document(self):
        pooled = score_pooled(self.pred, self.gold, _exact_matcher)
        per_doc = score_per_document(self.pred, self.gold, _exact_matcher)
        assert pooled["tp"] > per_doc["tp"]


class TestScoringContract:
    def test_agrees_with_pooling_when_no_leakage_possible(self):
        """With one document there is no cross-document boundary to cross."""
        pred = {"doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")]}
        gold = {"doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")]}
        per_doc = score_per_document(pred, gold, _exact_matcher)
        pooled = score_pooled(pred, gold, _exact_matcher)
        assert per_doc["tp"] == pooled["tp"] == 1
        assert per_doc["f1"] == pytest.approx(1.0)

    def test_micro_average_uses_corpus_wide_denominators(self):
        pred = {
            "doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")],
            "doc_b": [_t("X", RelationType.USES, "Y"), _t("P", RelationType.USES, "Q")],
        }
        gold = {
            "doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")],
            "doc_b": [_t("X", RelationType.USES, "Y")],
        }
        result = score_per_document(pred, gold, _exact_matcher)
        assert result["tp"] == 2
        assert result["fp"] == 1  # 3 predicted, 2 matched
        assert result["fn"] == 0  # 2 gold, both matched
        # Metrics are rounded to 4dp, matching compute_phase1_triplet_f1.
        assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
        assert result["recall"] == pytest.approx(1.0)

    def test_missing_gold_document_counts_predictions_as_false_positives(self):
        pred = {"doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")]}
        result = score_per_document(pred, {}, _exact_matcher)
        assert result["tp"] == 0
        assert result["fp"] == 1

    def test_per_doc_breakdown_is_reported(self):
        pred = {"doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")]}
        gold = {"doc_a": [_t("APT29", RelationType.USES, "Cobalt Strike")]}
        result = score_per_document(pred, gold, _exact_matcher)
        assert len(result["per_doc"]) == 1
        assert result["per_doc"][0]["doc_id"] == "doc_a"
        assert result["per_doc"][0]["pred"] == 1
        assert result["per_doc"][0]["gold"] == 1

    def test_empty_input_does_not_divide_by_zero(self):
        result = score_per_document({}, {}, _exact_matcher)
        assert result["f1"] == 0.0
        assert result["tp"] == 0
