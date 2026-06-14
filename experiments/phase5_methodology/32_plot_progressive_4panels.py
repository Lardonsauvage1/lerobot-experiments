"""4 panneaux progressifs 1k -> N (auto-decouverte des tranches de la chaine).
succes 500-rollouts | loss train(—)+val(- -) log | learning rate (lineaire, sci) | grad_norm log.
constant 1e-4 vs cosine (vagues SGDR, restart /50k).
"""
import csv, glob, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
LOGS = Path("results/logs/phase5_methodology")

# logs de base (phases 0->150k) avec offsets fixes
CONST_BASE = [("run_mini_constant.log",0),("run_mini_constant_continue.log",20000),
              ("run_mini_constant_continue2.log",50000),("run_mini_constant_finevar.log",56000),
              ("run_mini_constant_resume.log",57200),("run_mini_constant_150k.log",80000)]
COS_BASE   = [("run_mini_cosine.log",0),("run_mini_cosine_continue.log",20000),
              ("run_mini_cosine_continue2.log",50000),("run_mini_cosine_150k.log",80000)]

def chunk_logs(which):  # tranches de la chaine : run_mini_<which>_<N>k.log, N>=200, offset=(N-50)*1000
    specs = []
    for fp in glob.glob(str(LOGS / f"run_mini_{which}_*k.log")):
        m = re.search(rf"run_mini_{which}_(\d+)k\.log$", fp)
        if m and int(m.group(1)) >= 200:
            n = int(m.group(1)); specs.append((Path(fp).name, (n-50)*1000))
    return sorted(specs, key=lambda x: x[1])

CONST_LOG = CONST_BASE + chunk_logs("constant")
COS_LOG   = COS_BASE + chunk_logs("cosine")

def succ(patterns):
    d = {}
    for pat in patterns:
        for fp in glob.glob(str(ROOT / pat)):
            is_fv = "finevar" in fp
            for r in csv.DictReader(open(fp)):
                s = int(r["step"])
                if is_fv and s != 57000: continue
                d[s] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
    st = sorted(d); return st,[d[s][0] for s in st],[d[s][1] for s in st],[d[s][2] for s in st]

def train(specs):
    s,lo,gn,lr=[],[],[],[]
    for fn,off in specs:
        p = LOGS/fn
        if not p.exists(): continue
        for line in p.read_text(errors="ignore").replace("\r","\n").split("\n"):
            m=re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m: s.append(int(m.group(1))+off); lo.append(float(m.group(2))); gn.append(float(m.group(3))); lr.append(float(m.group(4)))
    z=sorted(zip(s,lo,gn,lr)); return [list(t) for t in zip(*z)]

def vloss(specs):
    s,v=[],[]
    for fn,off in specs:
        p = LOGS/fn
        if not p.exists(): continue
        for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", p.read_text(errors="ignore").replace("\r","\n")):
            s.append(int(m.group(2))); v.append(float(m.group(1)))
    z=sorted(zip(s,v)); return [list(t) for t in zip(*z)]

RUNS=[("constant 1e-4","tab:red",succ(["mini_constant/rollouts_500.csv","mini_constant_*/rollouts_500.csv"]),train(CONST_LOG),vloss(CONST_LOG)),
      ("cosine (vagues SGDR)","tab:blue",succ(["mini_cosine_rollouts_500.csv","mini_cosine_*/rollouts_500.csv"]),train(COS_LOG),vloss(COS_LOG))]
maxstep=max(RUNS[0][2][0][-1],RUNS[1][2][0][-1])

fig,ax=plt.subplots(2,2,figsize=(15,9))
fig.suptitle(f"mini-CNN Can 1k->{maxstep//1000}k — constant vs cosine (restart vague /50k)",fontsize=13,weight="bold")
for name,col,(st,sr,lo,hi),(tst,tl,gn,lr),(vst,vl) in RUNS:
    ax[0,0].plot(st,sr,"-o",color=col,ms=2,lw=0.8,label=name); ax[0,0].fill_between(st,lo,hi,color=col,alpha=0.12)
    ax[0,1].plot(tst,tl,"-",color=col,alpha=0.45,lw=0.7,label=f"{name} train")
    ax[0,1].plot(vst,vl,"--",color=col,alpha=0.9,lw=1.0,label=f"{name} val")
    ax[1,0].plot(tst,lr,"-",color=col,lw=1.1,label=name)
    ax[1,1].plot(tst,gn,"-",color=col,alpha=0.8,lw=0.8,label=name)
for a in ax.flat:
    for x in range(50000,maxstep+1,50000): a.axvline(x,ls=":",c="gray",lw=0.6)
ax[0,0].set_title("Succes 500-rollouts (+IC95)"); ax[0,0].set_ylabel("succes (%)"); ax[0,0].axhline(2,ls=":",c="lightgray",lw=0.8)
ax[0,1].set_title("loss train (—) + val (- -) — log"); ax[0,1].set_ylabel("loss"); ax[0,1].set_yscale("log")
ax[1,0].set_title("learning rate"); ax[1,0].set_ylabel("LR"); ax[1,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
ax[1,1].set_title("grad_norm — log"); ax[1,1].set_ylabel("grad_norm"); ax[1,1].set_yscale("log")
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=7); a.grid(alpha=0.3)
fig.tight_layout()
out=ROOT/"courbes_progressive_4panels.png"; fig.savefig(out,dpi=150)
print(f"ecrit : {out}  (jusqu'a {maxstep})")
