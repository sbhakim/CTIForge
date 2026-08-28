#!/usr/bin/env python3
"""Aggregate the symbolic validator's error taxonomy into a reportable table.

The manuscript's RQ3 currently summarises the taxonomy qualitatively, on the
grounds that a single snapshot depends on which configuration is scored. That is
true but it is not a reason to omit the numbers: the variation *between*
configurations is itself the finding, because canonicalisation (Module D)
silences a subset of what validation (Module C) flags.

This aggregates one or more runs' `error_taxonomy/error_log.jsonl` into a
per-category table with counts, share, and the action distribution
(rejected / repaired / flagged) for each category.

Note on provenance: `error_logger.save()` was historically reached only from
main.py and src/cli.py, never from eval_head_to_head.py or
eval_ctinexus_decomposed.py -- so no benchmark-scale run before August 2026
produced a log. Both evaluators now persist it.

Usage:
    conda run -n cti python eval_error_taxonomy.py output/<run>
    conda run -n cti python eval_error_taxonomy.py output/runA output/runB --labels B+C B+C+D
    conda run -n cti python eval_error_taxonomy.py --discover      # any run with a log
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

# The 14 categories declared by ErrorCategory, in the order the paper uses.
CATEGORIES = [
    "confidence_lowered", "type_misassignment", "impossible_type_pair",
    "generic_placeholder", "hallucinated_entity", "missing_evidence",
    "malformed_identifier", "repaired_identifier", "repaired_alias",
    "self_loop", "empty_field", "duplicate_entity", "over_extraction",
    "hallucinated_relation",
]
ACTIONS = ["rejected", "repaired", "flagged"]


def load_log(run_dir: Path) -> list[dict]:
    path = run_dir / "error_taxonomy" / "error_log.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def summarise(entries: list[dict]) -> dict:
    by_cat = Counter(e.get("category", "unknown") for e in entries)
    by_action = Counter(e.get("action", "unknown") for e in entries)
    cat_action: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        cat_action[e.get("category", "unknown")][e.get("action", "unknown")] += 1
    docs = {e.get("source_doc_id") for e in entries if e.get("source_doc_id")}
    return {
        "total": len(entries),
        "by_category": dict(by_cat),
        "by_action": dict(by_action),
        "category_action": {k: dict(v) for k, v in cat_action.items()},
        "documents_touched": len(docs),
    }


def print_table(runs: list[tuple[str, dict]]) -> None:
    all_cats = [c for c in CATEGORIES
                if any(s["by_category"].get(c) for _, s in runs)]
    extra = sorted({c for _, s in runs for c in s["by_category"]} - set(CATEGORIES))
    all_cats += extra

    width = 26
    hdr = f"  {'category':{width}s}" + "".join(f"{lbl[:14]:>16s}" for lbl, _ in runs)
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cat in all_cats:
        row = f"  {cat:{width}s}"
        for _, s in runs:
            n = s["by_category"].get(cat, 0)
            pct = n / s["total"] * 100 if s["total"] else 0.0
            row += f"{n:>9d} ({pct:4.1f}%)" if n else f"{'—':>16s}"
        print(row)
    print("  " + "-" * (len(hdr) - 2))
    total_row = f"  {'TOTAL actions':{width}s}"
    for _, s in runs:
        total_row += f"{s['total']:>16d}"
    print(total_row)
    doc_row = f"  {'documents touched':{width}s}"
    for _, s in runs:
        doc_row += f"{s['documents_touched']:>16d}"
    print(doc_row)

    print(f"\n  {'action distribution':{width}s}" + "".join(f"{lbl[:14]:>16s}" for lbl, _ in runs))
    print("  " + "-" * (len(hdr) - 2))
    for act in ACTIONS:
        row = f"  {act:{width}s}"
        for _, s in runs:
            n = s["by_action"].get(act, 0)
            pct = n / s["total"] * 100 if s["total"] else 0.0
            row += f"{n:>9d} ({pct:4.1f}%)" if n else f"{'—':>16s}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="run directories containing error_taxonomy/")
    ap.add_argument("--labels", nargs="*", default=None, help="column labels")
    ap.add_argument("--discover", action="store_true",
                    help="find every run under output/ that has a non-empty log")
    ap.add_argument("--min-entries", type=int, default=1)
    ap.add_argument("-o", "--output", default="output/error_taxonomy_table.json")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.run_dirs]
    if args.discover:
        found = []
        for p in sorted(glob.glob("output/*/error_taxonomy/error_log.jsonl")):
            run = Path(p).parent.parent
            if len(load_log(run)) >= args.min_entries:
                found.append(run)
        dirs = sorted(found, key=lambda d: -len(load_log(d)))
        print(f"\n  Discovered {len(dirs)} runs with >= {args.min_entries} logged actions")

    if not dirs:
        raise SystemExit(
            "No run directories given.\n"
            "Note: error_taxonomy/ is only produced by runs that call\n"
            "error_logger.save(). Before Aug 2026 that excluded\n"
            "eval_head_to_head.py and eval_ctinexus_decomposed.py, so the\n"
            "149-doc benchmark runs have no log. Re-run to generate one."
        )

    labels = args.labels or [d.name for d in dirs]
    runs = []
    for d, lbl in zip(dirs, labels):
        entries = load_log(d)
        if not entries:
            print(f"  [skip] {d} -- no error_taxonomy/error_log.jsonl")
            continue
        runs.append((lbl, summarise(entries)))

    if not runs:
        raise SystemExit("No usable logs found.")

    print_table(runs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({lbl: s for lbl, s in runs}, open(out, "w"), indent=2)
    print(f"\n  Written to {out}\n")


if __name__ == "__main__":
    main()
