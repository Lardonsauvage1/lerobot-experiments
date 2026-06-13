"""Succes 500-rollouts 1k->150k : constant 1e-4 vs cosine (vagues SGDR)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
CONST = ["mini_constant","mini_constant_continue","mini_constant_continue2","mini_constant_resume","mini_constant_150k"]
COS   = ["mini_cosine_rollouts_500.csv","mini_cosine_continue/rollouts_500.csv",
         "mini_cosine_continue2/rollouts_500.csv","mini_cosine_150k/rollouts_500.csv"]

def merge(paths, finevar=False):
    d={}
    for p in paths:
        fp = ROOT/p if p.endswith(".csv") else ROOT/p/"rollouts_500.csv"
        for r in csv.DictReader(open(fp)):
            d[int(r["step"])]=(float(r["success_rate"])*100,float(r["ci95_low"])*100,float(r["ci95_high"])*100)
    if finevar:
        for r in csv.DictReader(open(ROOT/"mini_constant_finevar"/"rollouts_500.csv")):
            if int(r["step"])==57000:
                d[57000]=(float(r["success_rate"])*100,float(r["ci95_low"])*100,float(r["ci95_high"])*100)
    st=sorted(d); return st,[d[s][0] for s in st],[d[s][1] for s in st],[d[s][2] for s in st]

fig, ax = plt.subplots(figsize=(14,6))
for name,paths,col,fv in [("constant 1e-4",CONST,"tab:red",True),("cosine (vagues SGDR)",COS,"tab:blue",False)]:
    st,sr,lo,hi = merge(paths, fv)
    ax.plot(st,sr,"-o",color=col,ms=2.5,lw=1,label=name); ax.fill_between(st,lo,hi,color=col,alpha=0.12)
# plateau converge du constant (100-150k)
ax.axhline(66.6, ls="--", c="darkred", lw=1, label="constant convergé ~66.6% (σ≈4, 100-150k)")
for x in (20000,50000,80000): ax.axvline(x, ls=":", c="gray", lw=0.7)
ax.set_title("mini-CNN Can (vision pure, 1.84M) — succès 500-rollouts 1k→150k : LR constant vs cosine (vagues)",
             fontsize=12, weight="bold")
ax.set_xlabel("step"); ax.set_ylabel("succès (%)"); ax.set_ylim(-2,82)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
out=ROOT/"courbe_full_1k_150k.png"; fig.savefig(out,dpi=150)
print(f"écrit : {out}")
for name,paths,fv in [("constant",CONST,True),("cosine",COS,False)]:
    st,sr,_,_=merge(paths,fv); print(f"{name:9s}: {len(st)} pts, max {max(sr):.1f}% @ {st[sr.index(max(sr))]}, fin150k {sr[-1]:.1f}%")
