"""Protocol-sensitivity harness: hold predictions fixed, vary only the matcher.

Run from the CTIForge repo root:
    conda run -n cti python -m benchmarks.eval_protocol_spread

Reports F1 for each cached prediction set under 7 matching protocols,
including the embedding matcher used by TACTIC-KG (ESORICS 2026).
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from benchmarks.eval_ctinexus_decomposed import _normalize,_names_match
from src.ingestion.loaders import load_ctinexus_dataset
from src.schema.relations import Triple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_COMPAT={("uses","targets"),("uses","drops"),("uses","delivers"),("delivers","drops"),
 ("associated_with","attributed_to"),("associated_with","uses"),("associated_with","exploits"),
 ("variant_of","associated_with"),("targets","exploits"),("communicates_with","uses"),
 ("exploits","mitigated_by"),("related_to","uses"),("related_to","targets"),
 ("related_to","associated_with"),("related_to","variant_of"),("related_to","attributed_to"),
 ("attributed_to","drops"),("related_to","mitigated_by"),("exploits","variant_of")}
def _rc(a,b): return a==b or (a,b) in _COMPAT or (b,a) in _COMPAT

P_EXACT  = lambda p,g: p==g
P_NAME   = lambda p,g: _names_match(p[0],g[0]) and _names_match(p[2],g[2]) and p[1]==g[1]
P_COMPAT = lambda p,g: (_names_match(p[0],g[0]) and _names_match(p[2],g[2]) and _rc(p[1],g[1])) or \
                       (_names_match(p[0],g[2]) and _names_match(p[2],g[0]) and _rc(p[1],g[1]))
P_PAIR   = lambda p,g: _names_match(p[0],g[0]) and _names_match(p[2],g[2])

def greedy(pred,gold,pred_fn):
    used=set();tp=0
    for p in pred:
        for gi,g in enumerate(gold):
            if gi in used: continue
            if pred_fn(p,g): used.add(gi);tp+=1;break
    return tp

MODEL=SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
def embed_tp(pred,gold,thr):
    if not pred or not gold: return 0
    pe=MODEL.encode([" ".join(p) for p in pred],normalize_embeddings=True)
    ge=MODEL.encode([" ".join(g) for g in gold],normalize_embeddings=True)
    S=cosine_similarity(pe,ge); M=(S>=thr).astype(float)
    if M.sum()==0: return 0
    r,c=linear_sum_assignment(-M); return int(M[r,c].sum())

def prf(tp,n,gd):
    p=tp/n if n else 0; r=tp/gd if gd else 0
    return p,r,(2*p*r/(p+r) if p+r else 0)
def keys(ts): return [(_normalize(t.subject),t.relation.value,_normalize(t.object)) for t in ts]

_d,_g=load_ctinexus_dataset(Path("data/annotations/ctinexus"))
gold_by={k:keys(v) for k,v in _g.items()}
SETS={"CTIForge":"output/head_to_head_149_sg_gpt4o/ctiforge_cached_triples.json",
      "CTI-Nexus":"output/head_to_head_149/nexus_cached_triples.json"}
loaded={n:json.load(open(p)) for n,p in SETS.items()}

PROTOCOLS=[("1 exact triple",P_EXACT),("2 name-soft + rel==",P_NAME),
           ("3 name-soft + compat (yours)",P_COMPAT),("4 S-O pair, rel ignored",P_PAIR)]
print(f"\n{'protocol':32s} {'SG F1':>8s} {'NX F1':>8s} {'winner':>8s} {'gap':>8s}")
print("-"*70)
res={}
for label,fn in PROTOCOLS:
    row={}
    for n,raw in loaded.items():
        TP=NP=NG=0
        for doc,tl in raw.items():
            if doc not in gold_by: continue
            pk=keys([Triple(**t) for t in tl]); gk=gold_by[doc]
            TP+=greedy(pk,gk,fn); NP+=len(pk); NG+=len(gk)
        row[n]=prf(TP,NP,NG)
    sg,nx=row["CTIForge"][2],row["CTI-Nexus"][2]
    w="SG" if sg>nx else "NX"
    print(f"{label:32s} {sg:8.4f} {nx:8.4f} {w:>8s} {abs(sg-nx):8.4f}")
    res[label]=(sg,nx)
for thr in (0.60,0.75,0.85):
    row={}
    for n,raw in loaded.items():
        TP=NP=NG=0
        for doc,tl in raw.items():
            if doc not in gold_by: continue
            pk=keys([Triple(**t) for t in tl]); gk=gold_by[doc]
            TP+=embed_tp(pk,gk,thr); NP+=len(pk); NG+=len(gk)
        row[n]=prf(TP,NP,NG)
    sg,nx=row["CTIForge"][2],row["CTI-Nexus"][2]
    w="SG" if sg>nx else "NX"
    print(f"{('5 embed cos>=%.2f (TACTIC)'%thr):32s} {sg:8.4f} {nx:8.4f} {w:>8s} {abs(sg-nx):8.4f}")
    res[f"embed{thr}"]=(sg,nx)
allsg=[v[0] for v in res.values()]; allnx=[v[1] for v in res.values()]
print("-"*70)
print(f"SG F1 range across protocols: {min(allsg):.4f} - {max(allsg):.4f}  (spread {max(allsg)-min(allsg):.4f})")
print(f"NX F1 range across protocols: {min(allnx):.4f} - {max(allnx):.4f}  (spread {max(allnx)-min(allnx):.4f})")
