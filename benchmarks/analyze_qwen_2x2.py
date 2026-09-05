#!/usr/bin/env python3
"""Decompose the validation effect into serving and decoding components.

Qwen2.5-7B-Instruct is the one backbone in this study available both locally and
through a hosted endpoint, so it is the only cell where the manuscript's stated
confound -- backbone, decoding and serving covarying -- can actually be broken.
Weights, prompt content and every downstream stage are held fixed; only serving
and decoding vary.

  conda run -n cti python -m benchmarks.analyze_qwen_2x2
"""
from __future__ import annotations
import json, os, sys

B = "output/paper"
CELLS = {
    ("local", "greedy"):   f"{B}/ablation_149_local-qwen25",     # already published
    ("local", "sampled"):  f"{B}/qwen_2x2/local_sampled",
    ("hosted", "greedy"):  f"{B}/qwen_2x2/hosted_greedy",
    ("hosted", "sampled"): f"{B}/qwen_2x2/hosted_sampled",
}

def dp(path):
    f = os.path.join(path, "module_ablation_results.json")
    if not os.path.exists(f):
        return None
    a = json.load(open(f))["arms"]
    b, bc = a["B-only"]["triplet"], a["B+C"]["triplet"]
    return {"dP": bc["precision"] - b["precision"],
            "B_P": b["precision"], "BC_P": bc["precision"],
            "B_F1": b["f1"], "BC_F1": bc["f1"],
            "raw": json.load(open(f)).get("raw_triples")}

def main():
    got = {k: dp(v) for k, v in CELLS.items()}
    missing = [k for k, v in got.items() if v is None]
    print("Qwen2.5-7B 2x2 -- identical weights, identical downstream\n")
    print(f"  {'cell':22s} {'raw':>5s} {'B P':>8s} {'B+C P':>8s} {'dP':>9s}")
    for k in CELLS:
        v = got[k]
        name = f"{k[0]}+{k[1]}"
        if v is None:
            print(f"  {name:22s} {'--':>5s} {'--':>8s} {'--':>8s} {'pending':>9s}")
        else:
            print(f"  {name:22s} {str(v['raw']):>5s} {v['B_P']:8.4f} {v['BC_P']:8.4f} {v['dP']:+9.4f}")
    if missing:
        print(f"\n  {len(missing)} cell(s) still running; effects need all four.")
        return 1

    d = {k: got[k]["dP"] for k in CELLS}
    serving  = ((d[("hosted","greedy")] + d[("hosted","sampled")]) / 2
                - (d[("local","greedy")] + d[("local","sampled")]) / 2)
    decoding = ((d[("local","sampled")] + d[("hosted","sampled")]) / 2
                - (d[("local","greedy")] + d[("hosted","greedy")]) / 2)
    inter    = ((d[("hosted","sampled")] - d[("hosted","greedy")])
                - (d[("local","sampled")] - d[("local","greedy")]))
    print(f"\n  main effect of SERVING  (hosted - local) : {serving:+.4f}")
    print(f"  main effect of DECODING (sampled - greedy): {decoding:+.4f}")
    print(f"  interaction                               : {inter:+.4f}")

    # Persist so the derived main effects quoted in the manuscript are traceable
    # to an artifact rather than to this script's stdout.
    summary = {"cells": {f"{k[0]}+{k[1]}": got[k] for k in CELLS},
               "serving_effect": round(serving, 4),
               "decoding_effect": round(decoding, 4),
               "interaction": round(inter, 4)}
    os.makedirs(f"{B}/qwen_2x2", exist_ok=True)
    with open(f"{B}/qwen_2x2/qwen_2x2_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n  summary written to {B}/qwen_2x2/qwen_2x2_summary.json")

    print("\n  Reading it:")
    if abs(serving) > abs(decoding) * 1.5:
        print("   serving dominates -- the split in Finding 4 survives with the backbone held fixed.")
    elif abs(decoding) > abs(serving) * 1.5:
        print("   decoding dominates -- the split tracks how the model is sampled, not where it runs.")
    else:
        print("   neither dominates -- the two contribute comparably on this backbone.")
    print("   local+greedy carries no sampling noise (byte-identical across runs); the")
    print("   other three cells each carry roughly the 0.0058 per-pass SD measured on")
    print("   MiniMax-M3, so differences below ~0.006 should not be read as effects.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
