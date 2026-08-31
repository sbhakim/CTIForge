#!/usr/bin/env python3
"""Why does validation help hosted backbones and hurt local ones?

RQ2 establishes that Module C raises precision on every hosted backbone and
lowers it on every local one, and that the split tracks neither model capability
nor parameter count. That is a correlation. This asks what the validator is
actually *doing* differently, by comparing the category mix of its actions.

The hypothesis under test: on local backbones a larger share of actions are
schema-level complaints about malformed entity typing (`type_misassignment`,
`impossible_type_pair`) rather than genuine relational errors -- i.e. the
validator misfires on annotation sloppiness instead of catching bad facts.

Reads the B+C arm's log from each run. `eval_error_taxonomy.py` expects
`<run>/error_taxonomy/`, but the ablation writes one directory per arm, so the
paths are resolved here instead.

Usage:
    conda run -n cti python -m benchmarks.analyze_taxonomy_by_backbone
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RUNS = [
    ("MiniMax-M3",   "hosted", "ablation_149_m3"),
    ("GPT-4o",       "hosted", "ablation_149_gpt4o"),
    ("Haiku 4.5",    "hosted", "ablation_149_haiku45"),
    ("Sonnet 4.5",   "hosted", "ablation_149_sonnet45"),
    ("Qwen2.5-7B",   "local",  "ablation_149_local-qwen25"),
    ("Gemma-2-9B",   "local",  "ablation_149_local-gemma2"),
    ("Mistral-7B",   "local",  "ablation_149_mistral_hostedprompt"),
]
BASE = Path("output/paper")
ARM = "error_taxonomy_B_C"

# Actions that explicitly dispute how an entity was typed. Structural issues
# such as empty fields, placeholders, and duplicates are intentionally excluded.
TYPE_ACTIONS = {"type_misassignment", "impossible_type_pair"}


def load(run: str) -> list[dict]:
    p = BASE / run / ARM / "error_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    rows = []
    all_cats: Counter = Counter()
    for label, serving, run in RUNS:
        e = load(run)
        if not e:
            print(f"  [skip] {label}: no log")
            continue
        cats = Counter(x.get("category", "?") for x in e)
        acts = Counter(x.get("action", "?") for x in e)
        all_cats.update(cats)
        type_actions = sum(v for k, v in cats.items() if k in TYPE_ACTIONS)
        rows.append({"label": label, "serving": serving, "n": len(e),
                     "cats": cats, "acts": acts,
                     "type_share": type_actions / len(e)})

    top = [c for c, _ in all_cats.most_common(6)]
    hdr = f"  {'backbone':13s} {'serv':7s} {'actions':>8s} {'rejected':>9s} {'type%':>8s}  "
    hdr += " ".join(f"{c[:11]:>12s}" for c in top)
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        rej = r["acts"].get("rejected", 0) / r["n"]
        line = (f"  {r['label']:13s} {r['serving']:7s} {r['n']:8d} "
                f"{rej:8.1%} {r['type_share']:8.1%}  ")
        line += " ".join(f"{r['cats'].get(c,0)/r['n']:11.1%}" for c in top)
        print(line)

    for serving in ("hosted", "local"):
        grp = [r for r in rows if r["serving"] == serving]
        if grp:
            m = sum(r["type_share"] for r in grp) / len(grp)
            print(f"\n  mean type-action share, {serving:6s}: {m:.1%}")
    print("\n  type% = share of validator actions explicitly disputing entity type")
    print("          (type_misassignment or impossible_type_pair)\n")


if __name__ == "__main__":
    main()
