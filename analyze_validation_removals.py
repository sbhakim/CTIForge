#!/usr/bin/env python3
"""What does Module C actually remove, and was it right to?

`eval_module_ablation.py` reports F1 per arm but not the composition of the
delta. Two backbones can post the same drop in F1 while removing entirely
different things: one shedding false positives (precision rises) and one
shedding correct triples (precision falls). That distinction is the finding,
so it needs to be computed rather than inferred from ΔF1.

For each document we take the triples present in B-only but absent from B+C --
the ones validation dropped -- and ask, under the harness's own matcher,
whether each matched a gold triple. A removal is CORRECT when the dropped
triple matched nothing in gold.

Usage:
    conda run -n cti python analyze_validation_removals.py output/paper/ablation_149_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_ctinexus_decomposed import compute_phase1_triplet_f1
from src.ingestion.loaders import load_ctinexus_dataset


def key(t: dict) -> tuple:
    return (str(t.get("subject", "")).strip().lower(),
            str(t.get("relation", "")).strip().lower(),
            str(t.get("object", "")).strip().lower())


class _Rel:
    """Minimal stand-in so persisted dicts satisfy the matcher's `.relation.value`."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value


class _Pred:
    """Adapter: the harness matcher takes Triple objects, predictions persist as dicts."""

    __slots__ = ("subject", "object", "relation")

    def __init__(self, d: dict):
        self.subject = str(d.get("subject", ""))
        self.object = str(d.get("object", ""))
        self.relation = _Rel(str(d.get("relation", "")))


def matches_gold(pred: dict, gold: list) -> bool:
    """True if this single prediction is credited against the document's gold."""
    if not gold:
        return False
    res = compute_phase1_triplet_f1([_Pred(pred)], gold)
    if not isinstance(res, dict):
        return False
    tp = res.get("tp", res.get("true_positives", res.get("matched", 0)))
    return bool(tp)


def analyse(run: Path, gold_by_doc: dict) -> dict | None:
    """Decompose what validation changed, using TP/FP counts rather than
    set-difference over triple keys.

    Module C does not only drop triples -- it repairs them (argument swap, IOC
    normalisation). A repaired triple has a different (s, r, o) key, so a naive
    key diff counts it as one removal plus one addition and badly overstates
    both. TP and FP counts are invariant to that: a repair that fixes a triple
    shows up as a TP gain, and one that breaks it as a TP loss.
    """
    res_path = run / "module_ablation_results.json"
    if not res_path.exists():
        return None
    arms = json.load(open(res_path))["arms"]
    if "B-only" not in arms or "B+C" not in arms:
        return None

    b, c = arms["B-only"]["triplet"], arms["B+C"]["triplet"]
    d_tp = c["tp"] - b["tp"]      # negative: correct triples lost
    d_fp = c["fp"] - b["fp"]      # negative: false positives dropped
    dropped_fp, lost_tp = max(0, -d_fp), max(0, -d_tp)
    total = dropped_fp + lost_tp

    return {
        "run": run.name,
        "n_b": b["tp"] + b["fp"], "n_c": c["tp"] + c["fp"],
        "b_f1": b["f1"], "c_f1": c["f1"],
        "d_f1": c["f1"] - b["f1"], "d_p": c["precision"] - b["precision"],
        "d_tp": d_tp, "d_fp": d_fp,
        "removed": total, "correct": dropped_fp,
        "rate": dropped_fp / total if total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--annotations-dir", default="data/annotations/ctinexus")
    args = ap.parse_args()

    docs, gold = load_ctinexus_dataset(args.annotations_dir)
    gold_by_doc = {d.doc_id: gold.get(d.doc_id, []) for d in docs}

    rows = [r for r in (analyse(Path(p), gold_by_doc) for p in args.runs) if r]
    rows.sort(key=lambda r: -(r["rate"] or 0))

    print(f"\n  {'run':34s} {'B F1':>7s} {'B+C F1':>7s} {'dF1':>8s} {'dP':>8s} "
          f"{'dTP':>6s} {'dFP':>6s} {'net':>5s} {'rate':>6s}")
    print("  " + "-" * 96)
    for r in rows:
        fmt = lambda v, w=7, p=4: (f"{v:{w}.{p}f}" if v is not None else " " * w)
        rate_s = "%.0f%%" % (r["rate"] * 100) if r["rate"] is not None else ""
        print(f"  {r['run']:34s} {fmt(r['b_f1'])} {fmt(r['c_f1'])} {fmt(r['d_f1'],8)} "
              f"{fmt(r['d_p'],8)} {r['d_tp']:6d} {r['d_fp']:6d} "
              f"{r['n_c']-r['n_b']:5d} {rate_s:>6s}")
    print("\n  dTP  = change in true positives  (negative: correct triples lost)")
    print("  dFP  = change in false positives (negative: bad triples dropped)")
    print("  net  = change in total predictions")
    print("  rate = dropped FPs / (dropped FPs + lost TPs) -- how well validation aimed\n")


if __name__ == "__main__":
    main()
