#!/usr/bin/env python3
"""Score matching protocols against human judgement.

Every CTI-KG paper reports F1 under its own matcher, and the numbers are not
comparable. This asks the prior question: *which matcher agrees with a human?*

Data: GRID's judge-calibration set (Huang et al., arXiv 2605.16714),
`../Codes/ProjectGRID/eval/llm-judge-calibration-with-human/`. Three reviewers
adjudicated 400 items; 378 remain after dropping "Undecided". Each item is a
single edge plus the opposing graph, an LLM judge verdict, and whether a human
agreed with that verdict.

  Precision item -- `Review edge` is a PREDICTED edge, scored against
                    `Ground Truth Graph`. Question: does it match any gold edge?
  Recall item    -- `Review edge` is a GOLD edge, scored against `Predict Graph`.
                    Question: is it covered by any predicted edge?

The human verdict is recovered by composing the judge decision with the human's
agreement:

    human_says_match = (decision starts with "TP") == (agreement == "Agree")

so TP+Agree and FP/FN+Disagree both mean "a human considered these a match".

Each protocol is then asked the same binary question on the same item, and we
report agreement with the human label. GRID's own judge agreement (86.0%
overall, 80.6% precision / 91.4% recall) is the reference point.

Usage:
    conda run -n cti python eval_matcher_vs_human.py
    conda run -n cti python eval_matcher_vs_human.py --no-embed   # skip ST model
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval_ctinexus_decomposed import _names_match, _normalize
from src.schema.relations import normalize_relation

CALIBRATION_DIR = (
    Path(__file__).resolve().parent.parent
    / "Codes" / "ProjectGRID" / "eval" / "llm-judge-calibration-with-human"
)

# Relation-compatibility set from compute_phase1_triplet_f1. Reproduced rather
# than imported because it is a local in that function.
_COMPAT = {
    ("uses", "targets"), ("uses", "drops"), ("uses", "delivers"),
    ("delivers", "drops"), ("associated_with", "attributed_to"),
    ("associated_with", "uses"), ("associated_with", "exploits"),
    ("variant_of", "associated_with"), ("targets", "exploits"),
    ("communicates_with", "uses"), ("exploits", "mitigated_by"),
    ("related_to", "uses"), ("related_to", "targets"),
    ("related_to", "associated_with"), ("related_to", "variant_of"),
    ("related_to", "attributed_to"), ("attributed_to", "drops"),
    ("related_to", "mitigated_by"), ("exploits", "variant_of"),
}


def _rels_compat(r1: str, r2: str) -> bool:
    return r1 == r2 or (r1, r2) in _COMPAT or (r2, r1) in _COMPAT


def _norm_rel(raw: str) -> str:
    """Map a free-text relation into the 12-type ontology."""
    try:
        return normalize_relation(raw).value
    except Exception:
        return "related_to"


# --------------------------------------------------------------------------
# Protocols. Each answers: does edge `e` match any edge in `graph`?
# `e` and graph entries are (subject, relation, object) raw strings.
# --------------------------------------------------------------------------

def p_exact(e, graph) -> bool:
    key = (_normalize(e[0]), _normalize(e[1]), _normalize(e[2]))
    return any(key == (_normalize(g[0]), _normalize(g[1]), _normalize(g[2])) for g in graph)


def p_name_soft_rel_eq(e, graph) -> bool:
    er = _norm_rel(e[1])
    return any(
        _names_match(e[0], g[0]) and _names_match(e[2], g[2]) and er == _norm_rel(g[1])
        for g in graph
    )


def p_name_soft_compat(e, graph) -> bool:
    """The protocol the manuscript reports: soft names + relation compatibility,
    with a swap-aware pass."""
    er = _norm_rel(e[1])
    for g in graph:
        gr = _norm_rel(g[1])
        if _names_match(e[0], g[0]) and _names_match(e[2], g[2]) and _rels_compat(er, gr):
            return True
        if _names_match(e[0], g[2]) and _names_match(e[2], g[0]) and _rels_compat(er, gr):
            return True
    return False


def p_so_pair(e, graph) -> bool:
    """Relation ignored entirely -- needs no relation normalisation, so it is the
    only lexical protocol free of that lossy step."""
    return any(
        (_names_match(e[0], g[0]) and _names_match(e[2], g[2]))
        or (_names_match(e[0], g[2]) and _names_match(e[2], g[0]))
        for g in graph
    )


class EmbedProtocol:
    """TACTIC-KG-style: cosine similarity over the concatenated triple."""

    def __init__(self, threshold: float, model_name="sentence-transformers/all-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.threshold = threshold
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, object] = {}

    def _embed(self, texts):
        import numpy as np
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True,
                                      show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.array([self._cache[t] for t in texts])

    def __call__(self, e, graph) -> bool:
        import numpy as np
        if not graph:
            return False
        ev = self._embed([" ".join(e)])[0]
        gv = self._embed([" ".join(g) for g in graph])
        return bool((gv @ ev).max() >= self.threshold)


def _coerce_graph(raw) -> list[tuple[str, str, str]]:
    """Normalise the two storage shapes used in the calibration files.

    `Ground Truth Graph` is a parsed list of {sub, rel, obj} dicts.
    `Predict Graph` is a Python-repr STRING of {'entities': [...], 'relations': [...]}
    (single-quoted, so json.loads fails -- ast.literal_eval is required).
    """
    if not raw:
        return []

    if isinstance(raw, str):
        import ast
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []

    if isinstance(raw, dict):
        raw = raw.get("relations") or raw.get("edges") or []

    out = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        sub = g.get("sub") or g.get("subject") or ""
        rel = g.get("rel") or g.get("relation") or ""
        obj = g.get("obj") or g.get("object") or ""
        if sub and obj:
            out.append((str(sub), str(rel), str(obj)))
    return out


def load_items():
    """Return (edge, opposing_graph, human_label, judge_label, type) tuples."""
    rows = []
    for i in (1, 2, 3):
        path = CALIBRATION_DIR / f"reviewer_{i}.json"
        if not path.exists():
            raise SystemExit(f"Calibration data not found: {path}")
        rows += json.load(open(path))

    items = []
    for r in rows:
        agreement = r.get("Human Agreement")
        if agreement not in ("Agree", "Disagree"):
            continue  # drop "Undecided" -- 400 -> 378, matching the GRID paper

        edge = r.get("Review edge") or {}
        e = (edge.get("sub", ""), edge.get("rel", ""), edge.get("obj", ""))
        if not e[0] or not e[2]:
            continue

        if r["Type"] == "Precision":
            graph = _coerce_graph(r.get("Ground Truth Graph"))
        else:
            graph = _coerce_graph(r.get("Predict Graph"))
        if not graph:
            continue  # nothing to match against; cannot score this item

        judge_positive = str(r.get("LLM Decision", "")).strip().upper().startswith("TP")
        human_label = (judge_positive == (agreement == "Agree"))

        items.append({
            "edge": e, "graph": graph, "human": human_label,
            "judge": judge_positive, "type": r["Type"],
        })
    return items


def score(items, protocol) -> dict:
    tp = fp = fn = tn = 0
    for it in items:
        pred = bool(protocol(it["edge"], it["graph"]))
        human = it["human"]
        if pred and human:
            tp += 1
        elif pred and not human:
            fp += 1
        elif not pred and human:
            fn += 1
        else:
            tn += 1
    n = len(items)
    return {
        "agreement": (tp + tn) / n if n else 0.0,
        "over_match": fp / n if n else 0.0,   # matcher credits, human does not
        "under_match": fn / n if n else 0.0,  # human credits, matcher does not
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="skip sentence-transformer protocols")
    ap.add_argument("-o", "--output", default="output/matcher_vs_human.json")
    args = ap.parse_args()

    items = load_items()
    prec = [i for i in items if i["type"] == "Precision"]
    rec = [i for i in items if i["type"] == "Recall"]
    print(f"\n  Loaded {len(items)} adjudicated items "
          f"({len(prec)} precision, {len(rec)} recall) after dropping Undecided")
    print(f"  Human says MATCH on {sum(i['human'] for i in items)}/{len(items)} items\n")

    protocols = [
        ("exact triple", p_exact),
        ("name-soft + rel==", p_name_soft_rel_eq),
        ("name-soft + compat (MANUSCRIPT)", p_name_soft_compat),
        ("S-O pair (rel ignored)", p_so_pair),
    ]
    if not args.no_embed:
        for thr in (0.60, 0.75, 0.85):
            protocols.append((f"embed cos>={thr:.2f}", EmbedProtocol(thr)))

    hdr = f"  {'protocol':34s} {'ALL':>8s} {'prec':>8s} {'rec':>8s} {'over':>7s} {'under':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {}
    for name, fn in protocols:
        a, p, r = score(items, fn), score(prec, fn), score(rec, fn)
        results[name] = {"all": a, "precision_items": p, "recall_items": r}
        print(f"  {name:34s} {a['agreement']:8.3f} {p['agreement']:8.3f} "
              f"{r['agreement']:8.3f} {a['over_match']:7.3f} {a['under_match']:7.3f}")

    # Reference: GRID's own LLM judge, on these same items by construction.
    jall = sum(i["judge"] == i["human"] for i in items) / len(items)
    jp = sum(i["judge"] == i["human"] for i in prec) / len(prec)
    jr = sum(i["judge"] == i["human"] for i in rec) / len(rec)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'GRID LLM judge (reference)':34s} {jall:8.3f} {jp:8.3f} {jr:8.3f}")
    results["GRID LLM judge"] = {"agreement_all": jall, "precision": jp, "recall": jr}

    print("\n  over  = matcher credits a match the human rejected (too lenient)")
    print("  under = human credits a match the matcher missed (too strict)\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"  Written to {out}\n")


if __name__ == "__main__":
    main()
