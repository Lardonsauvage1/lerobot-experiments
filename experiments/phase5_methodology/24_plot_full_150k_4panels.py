"""4 panneaux 1k->150k : cosine (vagues SGDR) vs LR constant.
succes 500-rollouts | loss train(—)+val(- -) log | learning rate (lineaire, sci) | grad_norm log.
"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
LOGS = Path("results/logs/phase5_methodology")

CONST_CSV = ["mini_constant","mini_constant_continue","mini_constant_continue2","mini_constant_resume","mini_constant_150k"]
COS_CSV   = ["mini_cosine_rollouts_500.csv","mini_cosine_continue/rollouts_500.csv",
             "mini_cosine_continue2/rollouts_500.csv","mini_cosine_150k/rollouts_500.csv"]
CONST_LOG = [("run_mini_constant.log",0),("run_mini_constant_continue.log",20000),
             ("run_mini_constant_continue2.log",50000),("run_mini_constant_finevar.log",56000),
             ("run_mini_constant_resume.log",57200),("run_mini_constant_150k.log",80000)]
COS_LOG   = [("run_mini_cosine.log",0),("run_mini_cosine_continue.log",20000),
             ("run_mini_cosine_continue2.log",50000),("run_mini_cosine_150k.log",80000)]

def succ(csvs, finevar=False):
    d={}
    for p in csvs:
        fp = ROOT/p if p.endswith(".csv") else ROOT/p/"rollouts_500.csv"
        for r in csv.DictReader(open(fp)):
            d[int(r["step"])]=(float(r["success_rate"])*100,float(r["ci95_low"])*100,float(r["ci95_high"])*100)
    if finevar:
        for r in csv.DictReader(open(ROOT/"mini_constant_finevar"/"rollouts_500.csv")):
            if int(r["step"])==57000: d[57000]=(float(r["success_rate"])*100,float(r["ci95_low"])*100,float(r["ci95_high"])*100)
    st=sorted(d); return st,[d[s][0] for s in st],[d[s][1] for s in st],[d[s][2] for s in st]

def train(specs):
    s,lo,gn,lr=[],[],[],[]
    for fn,off in specs:
        for line in (LOGS/fn).read_text(errors="ignore").replace("\r","\n").split("\n"):
            m=re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m: s.append(int(m.group(1))+off); lo.append(float(m.group(2))); gn.append(float(m.group(3))); lr.append(float(m.group(4)))
    z=sorted(zip(s,lo,gn,lr)); return [list(t) for t in zip(*z)]

def vloss(specs):
    s,v=[],[]
    for fn,_ in specs:
        for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", (LOGS/fn).read_text(errors="ignore").replace("\r","\n")):
            v.append(float(m.group(1))); s.append(int(m.group(2)))
    z=sorted(zip(s,v)); return [list(t) for t in zip(*z)]

RUNS=[("constant 1e-4","tab:red",succ(CONST_CSV,True),train(CONST_LOG),vloss(CONST_LOG)),
      ("cosine (vagues SGDR)","tab:blue",succ(COS_CSV),train(COS_LOG),vloss(COS_LOG))]

fig,ax=plt.subplots(2,2,figsize=(15,9))
fig.suptitle("mini-CNN sur Can (vision pure, 1.84M) — 1k→150k : cosine (vagues) vs LR constant",fontsize=13,weight="bold")
for name,col,(st,sr,lo,hi),(tst,tl,gn,lr),(vst,vl) in RUNS:
    ax[0,0].plot(st,sr,"-o",color=col,ms=2.5,lw=1,label=name); ax[0,0].fill_between(st,lo,hi,color=col,alpha=0.12)
    ax[0,1].plot(tst,tl,"-",color=col,alpha=0.45,lw=0.8,label=f"{name} train")
    ax[0,1].plot(vst,vl,"--",color=col,alpha=0.9,lw=1.1,label=f"{name} val")
    ax[1,0].plot(tst,lr,"-",color=col,lw=1.3,label=name)
    ax[1,1].plot(tst,gn,"-",color=col,alpha=0.8,lw=0.9,label=name)
for a in ax.flat:
    for x in (20000,50000,80000): a.axvline(x,ls=":",c="gray",lw=0.7)
ax[0,0].set_title("Succès 500-rollouts (+ IC95 Wilson)"); ax[0,0].set_ylabel("succès (%)"); ax[0,0].axhline(2,ls=":",c="lightgray",lw=0.8)
ax[0,1].set_title("loss train (—) + val (- -) — log"); ax[0,1].set_ylabel("loss"); ax[0,1].set_yscale("log")
ax[1,0].set_title("learning rate"); ax[1,0].set_ylabel("LR"); ax[1,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
ax[1,1].set_title("grad_norm — log"); ax[1,1].set_ylabel("grad_norm"); ax[1,1].set_yscale("log")
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout()
out=ROOT/"courbes_minicnn_full_1k_150k.png"; fig.savefig(out,dpi=150)
print(f"écrit : {out}")
for name,_,(st,sr,_,_),_,_ in RUNS: print(f"{name:22s}: {len(st)} pts succès, max {max(sr):.1f}% @ {st[sr.index(max(sr))]}")
