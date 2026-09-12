#!/usr/bin/env python3
"""4-PANNEAUX mini-CNN Can (CONSTANT) ENRICHI cooldown :
  (1) succès : brut rivière + merge SWA + COOLDOWN LR=0  / (2) loss / (3) lr / (4) grad_norm.
Même format que le 4-panneaux joint (87). Logs constant avec offsets (runs continue).
  venv312/bin/python experiments/phase5_methodology/93_plot_minicnn_cooldown_4panel.py
"""
import csv, glob, re, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/can"; P5 = "results/runs/phase5_methodology"
LOGS = "results/logs/phase5_methodology"
# logs constant + offset de step (numérotation locale -> absolue)
CONST_LOG = [("run_mini_constant.log",0),("run_mini_constant_continue.log",20000),
             ("run_mini_constant_continue2.log",50000),("run_mini_constant_resume.log",57200),
             ("run_mini_constant_150k.log",80000),("run_mini_constant_200k.log",150000),
             ("run_mini_constant_250k.log",200000)]
COS_LOG = [("run_mini_cosine.log",0),("run_mini_cosine_continue.log",20000),
           ("run_mini_cosine_continue2.log",50000),("run_mini_cosine_150k.log",80000),
           ("run_mini_cosine_200k.log",150000),("run_mini_cosine_250k.log",200000)]

def wilson(k, n, z=1.96):
    if n == 0: return 0., 0.
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-h)*100, min(1.,c+h)*100

def smooth(y, w=11):
    y=np.asarray(y,float)
    if len(y)<3: return y
    h=w//2; return np.array([y[max(0,i-h):min(len(y),i+h+1)].mean() for i in range(len(y))])

def load_logs(LOGSET=CONST_LOG):
    # le champ step: est abrégé (3K) -> on prend (loss,grdn,lr) + val_loss appariés par ORDRE,
    # et on reconstruit un step absolu via le valstep (entier complet) + offset de phase.
    steps,tl,vl,gr,lr=[],[],[],[],[]
    for name,off in LOGSET:
        fp=f"{LOGS}/{name}"
        if not Path(fp).exists(): continue
        txt=open(fp,errors="ignore").read()
        tr=re.findall(r"loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)",txt)
        vs=re.findall(r"val_loss:([0-9.]+) valstep:(\d+)",txt)
        n=min(len(tr),len(vs))
        for i in range(n):
            l,g,r_=tr[i]; v,s=vs[i]; s=int(s)
            ab=s if s>=off else s+off    # valstep déjà absolu (>=off) sinon on offset
            steps.append(ab); tl.append(float(l)); gr.append(float(g)); lr.append(float(r_)); vl.append(float(v))
    if not steps: return (np.array([]),)*5
    o=np.argsort(steps)
    return (np.array(steps)[o],np.array(tl)[o],np.array(vl)[o],np.array(gr)[o],np.array(lr)[o])

def prof(path):
    d={}
    if not Path(path).exists(): return d
    for r in csv.DictReader(open(path)):
        if r.get("success_rate"): d[int(float(r["source_k"]))]=(float(r["success_rate"])*100,
            float(r.get("ci95_low") or 0)*100, float(r.get("ci95_high") or 0)*100, int(float(r.get("n") or 500)))
    return d

# brut agrégé
raw={}
for pat in [f"{P5}/mini_constant_rollouts_500.csv",f"{P5}/mini_constant/rollouts_500.csv",
            f"{P5}/mini_constant_continue*/rollouts_500.csv",f"{P5}/mini_constant_resume/rollouts_500.csv",
            f"{P5}/mini_constant_1*0k/rollouts_500.csv",f"{P5}/mini_constant_2*0k/rollouts_500.csv"]:
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            try: raw[int(float(r["step"]))]=float(r["success_rate"])*100
            except: pass
rx=[k for k in sorted(raw) if k<=100000]; ry=[raw[k] for k in rx]
# brut COSINE (agrégé) pour comparaison
rawc={}
for pat in [f"{P5}/mini_cosine_rollouts_500.csv",f"{P5}/mini_cosine/rollouts_500.csv",
            f"{P5}/mini_cosine_continue*/rollouts_500.csv",f"{P5}/mini_cosine_1*0k/rollouts_500.csv"]:
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            try: rawc[int(float(r["step"]))]=float(r["success_rate"])*100
            except: pass
cxr=[k for k in sorted(rawc) if k<=100000]; cyr=[rawc[k] for k in cxr]
cd=prof(f"{R}/cooldown_profile_minicnn.csv"); mg=prof(f"{R}/merge_profile_minicnn.csv")
lst,tl,vl,gr,lr=load_logs(CONST_LOG)
clst,_,_,_,clr=load_logs(COS_LOG)   # LR cosine (vagues SGDR)

fig,ax=plt.subplots(2,2,figsize=(14,9)); ax=ax.flatten()
# (1) succès
if cxr: ax[0].plot([x/1000 for x in cxr],cyr,"-",color="#7fb0e0",lw=1.2,alpha=0.85,label="brut COSINE SGDR (n=500)")
ax[0].plot([x/1000 for x in rx],ry,"--^",color="#b0b0b0",ms=4,lw=1,alpha=0.8,label="brut constant (1 ckpt, n=500)")
if mg:
    xs=sorted(mg); ax[0].plot(xs,[mg[k][0] for k in xs],"-s",color="#d62728",ms=5,lw=1.6,label="merge SWA (5 ckpts)")
if cd:
    xs=sorted(cd); ys=[cd[k][0] for k in xs]
    lo=[cd[k][0]-cd[k][1] for k in xs]; hi=[cd[k][2]-cd[k][0] for k in xs]
    ax[0].errorbar(xs,ys,yerr=[lo,hi],fmt="-o",color="#1f2d3d",ms=6,lw=2,capsize=3,zorder=6,label="cooldown LR=0 (n=500)")
ax[0].axhline(80,color="#27ae60",ls="--",lw=1.1,alpha=0.6,label="plateau cosine SGDR ~80%")
ax[0].set_title("Succès (rollouts Can) % — brut → merge → cooldown"); ax[0].set_ylim(-3,100)
ax[0].set_xlim(0,102); ax[0].grid(alpha=.3); ax[0].set_xlabel("checkpoint de départ (k)"); ax[0].legend(loc="lower right",fontsize=8)
# (2) loss
if len(lst):
    ax[1].plot(lst/1000,smooth(tl),"-",color="#f4a3a3",lw=1.2,label="train (lissée)")
    ax[1].plot(lst/1000,smooth(vl),"-",color="#d62728",lw=1.8,label="val (lissée)")
    ax[1].set_yscale("log"); ax[1].set_title("loss train + val — log"); ax[1].grid(alpha=.3,which="both")
    ax[1].set_xlabel("step (k)"); ax[1].legend(fontsize=8); ax[1].set_xlim(0,102)
    ax[2].plot(lst/1000,lr,"-",color="#b0b0b0",lw=1.6,label="constant 1e-4")
    if len(clst): ax[2].plot(clst/1000,clr,"-",color="#7fb0e0",lw=1.4,label="cosine SGDR (restarts /vague)")
    ax[2].set_title("learning rate — constant vs cosine"); ax[2].grid(alpha=.3); ax[2].set_xlabel("step (k)")
    ax[2].legend(fontsize=8); ax[2].set_xlim(0,102)
    ax[3].plot(lst/1000,gr,"-",color="#d62728",lw=0.6,alpha=.5); ax[3].plot(lst/1000,smooth(gr),"-",color="#d62728",lw=1.8)
    ax[3].set_yscale("log"); ax[3].set_title("grad_norm — log"); ax[3].grid(alpha=.3,which="both"); ax[3].set_xlabel("step (k)"); ax[3].set_xlim(0,102)
fig.suptitle("mini-CNN Can (2M, LR constant) — 4 panneaux + cooldown : brut rebondit, cooldown se pose ~80%",
             fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); out=f"{P5}/courbes_minicnn_cooldown_4panel.png"; fig.savefig(out,dpi=130)
print("Sauvé :",out,f"| brut {len(rx)}, merge {len(mg)}, cooldown {len(cd)}, log {len(lst)} pts")
