"""Per-document metric aggregation.

Pooling every document's predictions and gold into two flat lists before
matching lets a prediction from document A be credited against a gold triple
from document B. Measured inflation on the 149-doc GPT-4o caches: +6.4% TP
for CTIForge, +5.9% for CTI-Nexus (~+0.03 F1 each).

Matching must be scoped to a document. These helpers do that while reusing
whatever matcher function is passed in, so the matching semantics are
unchanged -- only the scope is corrected.
"""

from __future__ import annotations


def score_per_document(pred_by_doc, gold_by_doc, compute_f1):
    """Micro-average a matcher over per-document matches.

    Args:
        pred_by_doc: dict doc_id -> list[Triple]
        gold_by_doc: dict doc_id -> list[Triple]
        compute_f1:  matcher returning {"tp","precision","recall","f1", ...}

    Returns:
        dict with tp/fp/fn/precision/recall/f1 plus per-document F1 values.
    """
    tp = n_pred = n_gold = 0
    per_doc = []

    for doc_id, preds in pred_by_doc.items():
        golds = gold_by_doc.get(doc_id, [])
        res = compute_f1(preds, golds)
        tp += res["tp"]
        n_pred += len(preds)
        n_gold += len(golds)
        per_doc.append({
            "doc_id": doc_id,
            "pred": len(preds),
            "gold": len(golds),
            "f1": res["f1"],
        })

    fp = n_pred - tp
    fn = n_gold - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "scope": "per_document",
        "per_doc": per_doc,
    }


def score_pooled(pred_by_doc, gold_by_doc, compute_f1):
    """Legacy pooled scoring, kept so the inflation can be reported alongside."""
    all_pred, all_gold = [], []
    for doc_id, preds in pred_by_doc.items():
        all_pred.extend(preds)
        all_gold.extend(gold_by_doc.get(doc_id, []))
    res = dict(compute_f1(all_pred, all_gold))
    res["scope"] = "pooled"
    return res
