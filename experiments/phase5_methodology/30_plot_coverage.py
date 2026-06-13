"""Compare les 3 metriques offline (sans sim) vs succes, avec correlations :
  action_mse (proxy 3, prediction moyenne) | coverage (par etat, min sur K, multimodal) | MMD marginal.
Repond : la couverture (theoriquement la bonne vu nos donnees) bat-elle empiriquement le MMD marginal ?
"""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
def load(fn): return {int(r["step"]): r for r in csv.DictReader(open(ROOT/fn))}
def succ_map(csvs):
    d={}
    for p in csvs:
        fp=ROOT/p if p.endswith(".csv") else ROOT/p/"rollouts_500.csv"
        for r in csv.DictReader(open(fp)): d[int(r["step"])]=float(r["success_rate"])*100
    return d
def pearson(x,y):
    n=len(x)
    if n<3: return float("nan")
    mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sx=sum((a-mx)**2 for a in x); sy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sx*sy) if sx>0 and sy>0 else float("nan")

CONST_S=["mini_constant","mini_constant_continue","mini_constant_continue2","mini_constant_resume","mini_constant_150k"]
COS_S=["mini_cosine_rollouts_500.csv","mini_cosine_continue/rollouts_500.csv","mini_cosine_continue2/rollouts_500.csv","mini_cosine_150k/rollouts_500.csv"]
RUNS=[("constant","tab:red","proxies_constant_cov.csv",succ_map(CONST_S)),
      ("cosine","tab:blue","proxies_cosine_cov.csv",succ_map(COS_S))]
METRICS=[("action_mse","proxy 3 : MSE moyenne"),("coverage","couverture par état (min sur K)"),("mmd","MMD marginal")]

fig,ax=plt.subplots(2,2,figsize=(15,9))
fig.suptitle("3 métriques offline vs succès — la couverture (par état) bat-elle le MMD marginal ?",fontsize=13,weight="bold")
corr={}
for name,col,pf,smap in RUNS:
    pr=load(pf); steps=sorted(pr)
    suc=[smap.get(s,float("nan")) for s in steps]
    ax[0,0].plot(steps,suc,"-o",color=col,ms=3,label=name)
    for j,(mk,_) in enumerate(METRICS):
        vals=[float(pr[s][mk]) for s in steps]
        r,c=(j+1)//2,(j+1)%2
        ax[r,c].plot(steps,vals,"-o",color=col,ms=3,label=name)
        idx=[i for i in range(len(steps)) if not math.isnan(suc[i])]
        idxp=[i for i in idx if steps[i]>=40000]
        corr[(name,mk)]=(abs(pearson([vals[i] for i in idx],[suc[i] for i in idx])),
                          abs(pearson([vals[i] for i in idxp],[suc[i] for i in idxp])))
ax[0,0].set_title("Succès 500-rollouts (référence)"); ax[0,0].set_ylabel("succès (%)")
for j,(mk,t) in enumerate(METRICS):
    r,c=(j+1)//2,(j+1)%2; ax[r,c].set_title(t); ax[r,c].set_ylabel(mk)
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
# tableau correlations
lines=["|r| avec succès   full / plateau(≥40k)"]
for name in ("constant","cosine"):
    lines.append(f"-- {name} --")
    for mk,_ in METRICS:
        f_,p_=corr[(name,mk)]; lines.append(f"  {mk:11s} {f_:.2f} / {p_:.2f}")
ax[0,0].text(0.98,0.02,"\n".join(lines),transform=ax[0,0].transAxes,fontsize=7,va="bottom",ha="right",
             family="monospace",bbox=dict(boxstyle="round",fc="white",alpha=0.85))
fig.tight_layout(); out=ROOT/"courbe_coverage.png"; fig.savefig(out,dpi=150); print(f"écrit : {out}")
print("\n".join(lines))
