#!/usr/bin/env python3
"""Protocol sensitivity across ten systems on one fixed backbone.

`eval_protocol_spread.py` varies the matcher over two systems and finds a 0.54
F1 swing -- but CTI-Nexus wins under every protocol, so it can show magnitude and
not rank reversal. This extends the same question to ten systems.

Data: GRID's released prediction cache (Huang et al., arXiv 2605.16714),
`../Codes/ProjectGRID/generated/<system>/generated/ctinexus__*.json`, scored
against `benchmark/ctinexus/source_records.parquet`. All fifty articles are a
verified subset of our 149-document CTI-Nexus benchmark.

Why this cache and not our own: every system in it was run on the SAME backbone
(Qwen3-4B-Instruct-2507). Backbone is therefore held constant and the only thing
varying is the matcher -- exactly the control this question needs. The same
property makes the cache USELESS for claiming one pipeline beats another, and it
is not used for that here.

Everything stays in GRID's native free-text representation; predictions are never
routed through CTIForge's 12-relation ontology except inside the protocols
that need a normalised relation, which is itself one of the variables measured.

Matching is scoped per document (see src/evaluation/per_document.py for why).

Usage:
    conda run -n cti python eval_multisystem_spread.py
    conda run -n cti python eval_multisystem_spread.py --no-embed
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from pathlib import Path

from eval_ctinexus_decomposed import _names_match, _normalize
from eval_matcher_vs_human import _COMPAT, _norm_rel, _rels_compat

GRID_ROOT = Path(__file__).resolve().parent.parent / "Codes" / "ProjectGRID"

DISPLAY = {
    "attackg_plus": "AttacKG+", "cognee": "Cognee", "ctikg": "CTIKG",
    "ctinexus": "CTI-Nexus", "graphiti": "Graphiti", "graphrag": "GraphRAG",
    "grid_end2end": "GRID-E2E", "grid_task_bank": "GRID-TaskBank",
    "knowgl": "KnowGL", "llm_cakg": "LLM-CAKG",
}


# --------------------------------------------------------------------------
# Protocols: does predicted edge `p` match gold edge `g`?
# --------------------------------------------------------------------------

def m_exact(p, g) -> bool:
    return (_normalize(p[0]), _normalize(p[1]), _normalize(p[2])) == \
           (_normalize(g[0]), _normalize(g[1]), _normalize(g[2]))


def m_name_rel_eq(p, g) -> bool:
    return (_names_match(p[0], g[0]) and _names_match(p[2], g[2])
            and _norm_rel(p[1]) == _norm_rel(g[1]))


def m_name_compat(p, g) -> bool:
    pr, gr = _norm_rel(p[1]), _norm_rel(g[1])
    if _names_match(p[0], g[0]) and _names_match(p[2], g[2]) and _rels_compat(pr, gr):
        return True
    return _names_match(p[0], g[2]) and _names_match(p[2], g[0]) and _rels_compat(pr, gr)


def m_so_pair(p, g) -> bool:
    return ((_names_match(p[0], g[0]) and _names_match(p[2], g[2]))
            or (_names_match(p[0], g[2]) and _names_match(p[2], g[0])))


LEXICAL = [
    ("exact triple", m_exact),
    ("name-soft + rel==", m_name_rel_eq),
    ("name-soft + compat", m_name_compat),
    ("S-O pair (rel ignored)", m_so_pair),
]


def greedy_tp(preds, golds, match) -> int:
    """Greedy first-match, scoped to one document."""
    used, tp = set(), 0
    for p in preds:
        for i, g in enumerate(golds):
            if i in used:
                continue
            if match(p, g):
                used.add(i)
                tp += 1
                break
    return tp


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, object] = {}

    def encode(self, triples):
        import numpy as np
        texts = [" ".join(t) for t in triples]
        missing = sorted({t for t in texts if t not in self.cache})
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=256)
            for t, v in zip(missing, vecs):
                self.cache[t] = v
        return np.array([self.cache[t] for t in texts]) if texts else None

    def tp(self, preds, golds, threshold) -> int:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        if not preds or not golds:
            return 0
        P, G = self.encode(preds), self.encode(golds)
        A = ((P @ G.T) >= threshold).astype(float)
        if A.sum() == 0:
            return 0
        r, c = linear_sum_assignment(-A)
        return int(A[r, c].sum())


def load_gold() -> dict[str, list[tuple[str, str, str]]]:
    import pandas as pd
    df = pd.read_parquet(GRID_ROOT / "benchmark" / "ctinexus" / "source_records.parquet")
    out = {}
    for _, r in df.iterrows():
        edges = json.loads(r["ground_truth_json"])
        out[r["file_name"]] = [(e.get("sub", ""), e.get("rel", ""), e.get("obj", ""))
                               for e in edges if e.get("sub") and e.get("obj")]
    return out


def _as_list(v) -> list[str]:
    """Coerce a field to a list of strings.

    A handful of grid_end2end edges carry a list-valued object, e.g.
    obj = ['.mallox', '.xollam'], which genuinely encodes two facts. Expanding
    is the fair reading; stringifying the list would make both unmatchable.
    Affects 3 of 810 edges in one system -- immaterial, but handled explicitly
    so it is not silently mangled.
    """
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if v is None:
        return []
    return [str(v)]


def load_system(system: str) -> dict[str, list[tuple[str, str, str]]]:
    out = {}
    for f in sorted(glob.glob(str(GRID_ROOT / "generated" / system / "generated" / "ctinexus__*"))):
        d = json.load(open(f))
        rels = (d.get("generated_graph") or {}).get("relations") or []
        edges = []
        for r in rels:
            for s in _as_list(r.get("sub")):
                for rel in (_as_list(r.get("rel")) or [""]):
                    for o in _as_list(r.get("obj")):
                        edges.append((s, rel, o))
        out[d["file_name"]] = edges
    return out


def f1_for(preds_by_doc, gold_by_doc, match=None, embedder=None, threshold=None) -> float:
    tp = npred = ngold = 0
    for doc, golds in gold_by_doc.items():
        preds = preds_by_doc.get(doc, [])
        if embedder is not None:
            tp += embedder.tp(preds, golds, threshold)
        else:
            tp += greedy_tp(preds, golds, match)
        npred += len(preds)
        ngold += len(golds)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def load_articles() -> dict[str, str]:
    import pandas as pd
    df = pd.read_parquet(GRID_ROOT / "benchmark" / "ctinexus" / "source_records.parquet")
    return {r["file_name"]: r["content"] for _, r in df.iterrows()}


def judge_f1(preds_by_doc, gold_by_doc, articles, judge) -> float:
    """LLM-judge protocol, micro-averaged over documents.

    This is the only protocol with measured human agreement (Section 15).
    Responses are cached to disk, so re-runs are free.
    """
    from src.evaluation.llm_judge import judge_documents
    items = [(preds_by_doc.get(doc, []), golds, articles.get(doc, ""))
             for doc, golds in gold_by_doc.items()]
    ptp = pn = rtp = rn = 0
    for r in judge_documents(judge, items):
        ptp += r["precision_tp"]; pn += r["n_pred"]
        rtp += r["recall_tp"];    rn += r["n_gold"]
    p = ptp / pn if pn else 0.0
    r_ = rtp / rn if rn else 0.0
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="add the LLM-judge protocol column (uses GRID's calibrated prompt)")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-dry-run", action="store_true",
                    help="estimate judge cost without making any API call")
    ap.add_argument("-o", "--output", default="output/multisystem_spread.json")
    args = ap.parse_args()

    gold = load_gold()
    systems = sorted(
        os.path.basename(d)
        for d in glob.glob(str(GRID_ROOT / "generated" / "*"))
        if os.path.isdir(d) and os.path.basename(d) != "resources"
    )
    preds = {s: load_system(s) for s in systems}

    print(f"\n  {len(systems)} systems, {len(gold)} documents, "
          f"{sum(len(v) for v in gold.values())} gold edges")
    print("  Backbone held constant: Qwen3-4B-Instruct-2507 (GRID release)\n")

    protocols = list(LEXICAL)
    embedder = None
    if not args.no_embed:
        embedder = Embedder()
        for thr in (0.60, 0.75, 0.85):
            protocols.append((f"embed cos>={thr:.2f}", thr))

    scores: dict[str, dict[str, float]] = {}
    for name, spec in protocols:
        scores[name] = {}
        for s in systems:
            if callable(spec):
                scores[name][s] = f1_for(preds[s], gold, match=spec)
            else:
                scores[name][s] = f1_for(preds[s], gold, embedder=embedder, threshold=spec)

    judge_stats = None
    if args.judge or args.judge_dry_run:
        from src.evaluation.llm_judge import CachedJudge
        judge = CachedJudge(model=args.judge_model, dry_run=args.judge_dry_run)
        col = f"LLM judge ({args.judge_model})"
        articles = load_articles()
        scores[col] = {}
        for s in systems:
            scores[col][s] = judge_f1(preds[s], gold, articles, judge)
            print(f"    judged {DISPLAY.get(s, s)}: "
                  f"{judge.stats['calls']} calls / {judge.stats['hits']} cached", flush=True)
        judge_stats = {"model": args.judge_model, **judge.stats, **judge.estimate_cost()}
        print(f"\n  judge usage: {judge_stats}")
        if args.judge_dry_run:
            print("  (dry run -- no API calls made, judge column is meaningless)")
        else:
            protocols.append((col, None))

    # ---- F1 table ----
    pnames = [n for n, _ in protocols]
    print("  " + "F1 by system and protocol".center(30 + 12 * len(pnames)))
    hdr = f"  {'system':16s}" + "".join(f"{n[:11]:>12s}" for n in pnames) + f"{'spread':>9s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in sorted(systems, key=lambda x: -scores[pnames[2]][x]):
        vals = [scores[n][s] for n in pnames]
        print(f"  {DISPLAY.get(s, s):16s}" + "".join(f"{v:12.4f}" for v in vals)
              + f"{max(vals) - min(vals):9.4f}")

    # ---- ranks ----
    print(f"\n  {'Rank by protocol (1 = best)'}")
    hdr2 = f"  {'system':16s}" + "".join(f"{n[:11]:>12s}" for n in pnames)
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    ranks: dict[str, dict[str, int]] = {}
    for n in pnames:
        order = sorted(systems, key=lambda x: -scores[n][x])
        ranks[n] = {s: i + 1 for i, s in enumerate(order)}
    for s in sorted(systems, key=lambda x: ranks[pnames[2]][x]):
        print(f"  {DISPLAY.get(s, s):16s}" + "".join(f"{ranks[n][s]:12d}" for n in pnames))

    # ---- rank reversals ----
    reversals = []
    for a, b in itertools.combinations(systems, 2):
        signs = {n: (scores[n][a] > scores[n][b]) for n in pnames}
        if len(set(signs.values())) > 1:
            wins_a = [n for n in pnames if signs[n]]
            wins_b = [n for n in pnames if not signs[n]]
            gap = max(abs(scores[n][a] - scores[n][b]) for n in pnames)
            reversals.append((DISPLAY.get(a, a), DISPLAY.get(b, b), wins_a, wins_b, gap))

    total_pairs = len(systems) * (len(systems) - 1) // 2
    print(f"\n  RANK REVERSALS: {len(reversals)} of {total_pairs} system pairs "
          f"({len(reversals)/total_pairs:.1%}) change order depending on protocol\n")
    for a, b, wa, wb, gap in sorted(reversals, key=lambda x: -x[4]):
        print(f"    {a} vs {b}   (max gap {gap:.4f})")
        print(f"        {a} wins under: {', '.join(w[:18] for w in wa)}")
        print(f"        {b} wins under: {', '.join(w[:18] for w in wb)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"scores": scores, "ranks": ranks, "judge": judge_stats,
               "reversals": [{"a": a, "b": b, "a_wins": wa, "b_wins": wb, "max_gap": g}
                             for a, b, wa, wb, g in reversals],
               "n_systems": len(systems), "n_docs": len(gold)},
              open(out, "w"), indent=2)
    print(f"\n  Written to {out}\n")


if __name__ == "__main__":
    main()
