#!/usr/bin/env python3
"""Between-pass variability of the validation effect on one backbone.

Each pass is an independent extraction run; within a pass, the B and B+C arms
share that pass's raw triples, so dP is a *paired* difference. The quantity the
manuscript needs is the spread of dP across passes -- not the spread of absolute
F1, which also carries extraction variance the paired design already removes.

  conda run -n cti python -m benchmarks.analyze_seed_variance output/paper/seed_minimax
"""
from __future__ import annotations
import json, sys, glob, os, statistics as st

def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "pass_*", "module_ablation_results.json"))):
        j = json.load(open(f))
        arms = j["arms"]
        b, bc = arms["B-only"]["triplet"], arms["B+C"]["triplet"]
        rows.append({
            "pass": os.path.basename(os.path.dirname(f)),
            "raw": j.get("raw_triples"),
            "B_P": b["precision"], "BC_P": bc["precision"],
            "B_F1": b["f1"], "BC_F1": bc["f1"],
            "dP": round(bc["precision"] - b["precision"], 6),
            "dF1": round(bc["f1"] - b["f1"], 6),
        })
    return rows

def spread(vals):
    if len(vals) < 2:
        return None, None
    return st.mean(vals), st.stdev(vals)   # sample SD, n-1

def main(d):
    rows = load(d)
    if not rows:
        print(f"no completed passes under {d}"); return 1
    print(f"{len(rows)} pass(es) under {d}\n")
    print(f"  {'pass':8s} {'raw':>5s} {'B P':>8s} {'B+C P':>8s} {'dP':>9s} "
          f"{'B F1':>8s} {'B+C F1':>8s} {'dF1':>9s}")
    for r in rows:
        print(f"  {r['pass']:8s} {str(r['raw']):>5s} {r['B_P']:8.4f} {r['BC_P']:8.4f} "
              f"{r['dP']:+9.4f} {r['B_F1']:8.4f} {r['BC_F1']:8.4f} {r['dF1']:+9.4f}")
    print()
    for name, key in (("dP  (validation precision effect)", "dP"),
                      ("dF1 (validation F1 effect)", "dF1"),
                      ("B F1 (absolute, for contrast)", "B_F1")):
        v = [r[key] for r in rows]
        m, sd = spread(v)
        if sd is None:
            print(f"  {name:36s} n<2, no spread"); continue
        print(f"  {name:36s} mean {m:+.4f}   SD {sd:.4f}   range {max(v)-min(v):.4f}")
    print("\n  Note: SD of dP is the paired quantity Figure 4 needs. SD of B F1 is the\n"
          "  unpaired extraction variance and is expected to be larger; do not use it\n"
          "  as a band on the dP axis.")

    # Persist the summary so every figure quoted in the manuscript is traceable
    # to an artifact rather than to this script's stdout.
    summary = {"n_passes": len(rows), "source": d, "passes": rows}
    for key in ("dP", "dF1", "B_F1", "B_P"):
        v = [r[key] for r in rows]
        m, sd = spread(v)
        summary[key] = {"mean": round(m, 4), "sd": round(sd, 4) if sd else None,
                        "min": round(min(v), 4), "max": round(max(v), 4)}
    out = os.path.join(d, "seed_variance_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"  summary written to {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output/paper/seed_minimax"))
