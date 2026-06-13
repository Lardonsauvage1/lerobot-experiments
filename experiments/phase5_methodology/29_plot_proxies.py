"""Proxies offline vs succes : action_mse (proxy 3) + MMD (proxy 2), avec correlations.
Repond : un signal SANS simulation predit-il le succes (surtout dans le plateau) ?
"""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")

def load_csv(fn):
    rows = list(csv.DictReader(open(ROOT/fn)));
    return {int(r["step"]): r for r in rows}

def succ_map(csvs, finevar=False):
    d={}
    for p in csvs:
        fp = ROOT/p if p.endswith(".csv") else ROOT/p/"rollouts_500.csv"
        for r in csv.DictReader(open(fp)): d[int(r["step"])]=float(r["success_rate"])*100
    return d

def pearson(x,y):
    n=len(x);
    if n<3: return float("nan")
    mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sx=sum((a-mx)**2 for a in x); sy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sx*sy) if sx>0 and sy>0 else float("nan")

def spearman(x,y):
    def rank(v):
        s=sorted(range(len(v)), key=lambda i:v[i]); r=[0]*len(v)
        for k,i in enumerate(s): r[i]=k
        return r
    return pearson(rank(x), rank(y))

CONST_S = ["mini_constant","mini_constant_continue","mini_constant_continue2","mini_constant_resume","mini_constant_150k"]
COS_S   = ["mini_cosine_rollouts_500.csv","mini_cosine_continue/rollouts_500.csv","mini_cosine_continue2/rollouts_500.csv","mini_cosine_150k/rollouts_500.csv"]
RUNS = [("constant","tab:red","proxies_constant.csv",succ_map(CONST_S),"val_constant_full_merged.csv"),
        ("cosine","tab:blue","proxies_cosine.csv",succ_map(COS_S),"val_cosine_full.csv")]

fig,ax=plt.subplots(2,2,figsize=(15,9))
fig.suptitle("Proxies offline (sans sim) vs succès — action error (proxy 3) & MMD (proxy 2)",fontsize=13,weight="bold")
corr_lines=[]
for name,col,pf,smap,vf in RUNS:
    pr=load_csv(pf); val=load_csv(vf)
    steps=sorted(pr)
    mse=[float(pr[s]["action_mse"]) for s in steps]
    mmd=[float(pr[s]["mmd"]) for s in steps]
    h0=[float(pr[s]["mse_h0"]) for s in steps]; h7=[float(pr[s]["mse_h7"]) for s in steps]
    suc=[smap.get(s,float("nan")) for s in steps]
    vfl=[float(val[s]["val_loss_full"]) if s in val else float("nan") for s in steps]
    ax[0,0].plot(steps,suc,"-o",color=col,ms=3,label=name)
    ax[0,1].plot(steps,mse,"-o",color=col,ms=3,label=f"{name} action_mse")
    ax[1,0].plot(steps,mmd,"-o",color=col,ms=3,label=f"{name} MMD")
    ax[1,1].plot(steps,h0,"--",color=col,lw=1,label=f"{name} h0 (1er pas)")
    ax[1,1].plot(steps,h7,"-",color=col,lw=1.5,label=f"{name} h7 (dernier pas)")
    # correlations |r| avec succes : full range et plateau (>=40k)
    def corr(sig):
        idx=[i for i in range(len(steps)) if not math.isnan(suc[i]) and not math.isnan(sig[i])]
        idxp=[i for i in idx if steps[i]>=40000]
        return pearson([sig[i] for i in idx],[suc[i] for i in idx]), pearson([sig[i] for i in idxp],[suc[i] for i in idxp])
    rmse=corr(mse); rmmd=corr(mmd); rval=corr(vfl)
    corr_lines.append(f"{name}: |r| succès  full / plateau(≥40k)\n"
        f"  action_mse  {abs(rmse[0]):.2f} / {abs(rmse[1]):.2f}\n"
        f"  MMD         {abs(rmmd[0]):.2f} / {abs(rmmd[1]):.2f}\n"
        f"  val_full    {abs(rval[0]):.2f} / {abs(rval[1]):.2f}")

ax[0,0].set_title("Succès 500-rollouts (référence)"); ax[0,0].set_ylabel("succès (%)")
ax[0,1].set_title("proxy 3 : erreur d'action générée (MSE)"); ax[0,1].set_ylabel("action MSE")
ax[1,0].set_title("proxy 2 : MMD (action prédite vs experte) — log"); ax[1,0].set_ylabel("MMD"); ax[1,0].set_yscale("log")
ax[1,1].set_title("compounding : erreur au 1er pas (- -) vs dernier pas (—)"); ax[1,1].set_ylabel("MSE par pas")
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
ax[0,0].text(0.98,0.02,"\n\n".join(corr_lines),transform=ax[0,0].transAxes,fontsize=7,va="bottom",ha="right",
             family="monospace",bbox=dict(boxstyle="round",fc="white",alpha=0.8))
fig.tight_layout()
out=ROOT/"courbe_proxies.png"; fig.savefig(out,dpi=150); print(f"écrit : {out}")
print("\n".join(corr_lines))
