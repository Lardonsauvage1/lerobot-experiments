"""4-panneaux détaillés des runs joint A/B (agentview, sim). Arg: A ou B."""
import re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = sys.argv[1]
CFG = {"A": ("joint_A", "cd_joint_agent_ourconfig", "joint_agent_ourconfig",
             "NOTRE config (R34 + U-Net[128,256,512], no crop)", 77.6, (73.7, 81.0)),
       "B": ("joint_B", "cd_joint_agent_goodconfig", "joint_agent_goodconfig",
             "BON config (R18 + U-Net[512,1024,2048] + crop)", 83.8, (80.3, 86.8))}[W]
tlog, clog, run, desc, succ, ci = CFG
def parse(p, off=0):
    p = Path(f"results/logs/can/{p}.log")
    if not p.exists(): return []
    txt = p.read_text(errors="ignore").replace("\r", "\n"); last=None; out=[]
    for ln in txt.split("\n"):
        m = re.search(r"loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)", ln)
        if m: last = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        v = re.search(r"val_loss:([0-9.]+) valstep:([0-9]+)", ln)
        if v and last: out.append((int(v.group(2))+off, last[0], float(v.group(1)), last[1], last[2]))
    return out
data = parse(tlog, 0) + parse(clog, 40000)
steps=[d[0] for d in data]; tl=[d[1] for d in data]; vl=[d[2] for d in data]; gr=[d[3] for d in data]; lr=[d[4] for d in data]
vmin=min(range(len(vl)), key=lambda i: vl[i]); CDX=40000
fig, ax = plt.subplots(2,2, figsize=(15,8))
fig.suptitle(f"JOINT sim agentview — {desc}\nSUCCÈS 500 rollouts = {succ:.1f}% [{ci[0]}-{ci[1]}]  ({len(data)} points)",
             fontsize=12, fontweight="bold")
def cd(a): a.axvline(CDX,color="#94a3b8",ls="--",lw=1); a.axvspan(CDX,max(steps),color="#f1f5f9",alpha=.6,zorder=0)
a=ax[0,0]; a.plot(steps,tl,"-",color="#2563eb",lw=.7,label="train"); a.plot(steps,vl,"-",color="#dc2626",lw=.7,alpha=.85,label="val")
cd(a); a.set_yscale("log"); a.set_title("(A) Loss train vs val"); a.legend(); a.grid(alpha=.3); a.set_xlabel("step")
b=ax[0,1]; b.plot(steps,vl,"-",color="#dc2626",lw=.7); b.plot(steps[vmin],vl[vmin],"*",color="#f59e0b",ms=18,markeredgecolor="k",label=f"min {vl[vmin]:.4f}")
cd(b); b.set_title("(B) Val-loss"); b.legend(fontsize=9); b.grid(alpha=.3); b.set_xlabel("step")
c=ax[1,0]; c.plot(steps,gr,"-",color="#059669",lw=.6); cd(c); c.set_title("(C) Grad-norm"); c.grid(alpha=.3); c.set_xlabel("step")
d=ax[1,1]; d.plot(steps,lr,"-",color="#7c3aed",lw=1.1); cd(d); d.set_title("(D) Learning rate"); d.grid(alpha=.3); d.set_xlabel("step"); d.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
fig.tight_layout(rect=[0,0,1,0.95])
out=f"results/runs/can/{run}/joint_{W}_4panel.png"; Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=120)
print(f"-> {out} | {len(data)} points | succès {succ}%")
