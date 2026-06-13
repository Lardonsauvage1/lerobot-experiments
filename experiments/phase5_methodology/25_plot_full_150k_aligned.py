"""4 panneaux 1k->150k, version ALIGNEE : loss/LR/grad affiches UNIQUEMENT aux steps
ou on a une mesure de rollout (memes checkpoints que le succes) -> points comparables,
pas la bande dense (tous les 100). cosine (vagues SGDR) vs LR constant.
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

def train_map(specs):  # {step: (loss, grdn, lr)}
    d={}
    for fn,off in specs:
        for line in (LOGS/fn).read_text(errors="ignore").replace("\r","\n").split("\n"):
            m=re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m: d[int(m.group(1))+off]=(float(m.group(2)),float(m.group(3)),float(m.group(4)))
    return d

def vloss_map(specs):  # {step: val_loss}
    d={}
    for fn,_ in specs:
        for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", (LOGS/fn).read_text(errors="ignore").replace("\r","\n")):
            d[int(m.group(2))]=float(m.group(1))
    return d

def at(steps, dmap, idx=None):
    """valeurs de dmap aux 'steps' (ignore les steps absents). idx=index du tuple, None=scalaire."""
    xs,ys=[],[]
    for s in steps:
        if s in dmap:
            xs.append(s); ys.append(dmap[s] if idx is None else dmap[s][idx])
    return xs,ys

RUNS=[("constant 1e-4","tab:red",succ(CONST_CSV,True),train_map(CONST_LOG),vloss_map(CONST_LOG)),
      ("cosine (vagues SGDR)","tab:blue",succ(COS_CSV),train_map(COS_LOG),vloss_map(COS_LOG))]

fig,ax=plt.subplots(2,2,figsize=(15,9))
fig.suptitle("mini-CNN Can (1.84M) — 1k→150k, loss/LR/grad UNIQUEMENT aux steps évalués (rollouts)",fontsize=13,weight="bold")
for name,col,(st,sr,lo,hi),tm,vm in RUNS:
    ax[0,0].plot(st,sr,"-o",color=col,ms=2.5,lw=1,label=name); ax[0,0].fill_between(st,lo,hi,color=col,alpha=0.12)
    xt,tl=at(st,tm,0); xt,gn=at(st,tm,1); xt,lr=at(st,tm,2); xv,vl=at(st,vm)
    ax[0,1].plot(xt,tl,"-o",color=col,ms=3,lw=0.8,alpha=0.6,label=f"{name} train")
    ax[0,1].plot(xv,vl,"--s",color=col,ms=3,lw=0.8,alpha=0.95,mfc="none",label=f"{name} val")
    ax[1,0].plot(xt,lr,"-o",color=col,ms=3,lw=1.2,label=name)
    ax[1,1].plot(xt,gn,"-o",color=col,ms=3,lw=0.9,alpha=0.85,label=name)
for a in ax.flat:
    for x in (20000,50000,80000): a.axvline(x,ls=":",c="gray",lw=0.7)
ax[0,0].set_title("Succès 500-rollouts (+ IC95 Wilson)"); ax[0,0].set_ylabel("succès (%)"); ax[0,0].axhline(2,ls=":",c="lightgray",lw=0.8)
ax[0,1].set_title("loss train (●) + val (□) — aux steps évalués, log"); ax[0,1].set_ylabel("loss"); ax[0,1].set_yscale("log")
ax[1,0].set_title("learning rate — aux steps évalués"); ax[1,0].set_ylabel("LR"); ax[1,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
ax[1,1].set_title("grad_norm — aux steps évalués, log"); ax[1,1].set_ylabel("grad_norm"); ax[1,1].set_yscale("log")
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout()
out=ROOT/"courbes_minicnn_full_1k_150k_aligned.png"; fig.savefig(out,dpi=150)
print(f"écrit : {out}")
