"""Courbe COMPLETE mini-CNN 1k->80k : cosine (vagues SGDR) vs LR constant — 4 panneaux.
succes 500-rollouts | loss train(—)+val(- -) en log | learning rate (lineaire, sci) | grad_norm en log.
NB : succes cosine seulement jusqu'a 50k (50-80k non encore evalue).
"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
LOGS = Path("results/logs/phase5_methodology")

# success : liste de csv ; finevar -> on ne garde que 57000 (sinon amas dense a 57k)
CONST_CSV = [ROOT/"mini_constant"/"rollouts_500.csv", ROOT/"mini_constant_continue"/"rollouts_500.csv",
             ROOT/"mini_constant_continue2"/"rollouts_500.csv", ROOT/"mini_constant_resume"/"rollouts_500.csv"]
COS_CSV   = [ROOT/"mini_cosine_rollouts_500.csv", ROOT/"mini_cosine_continue"/"rollouts_500.csv"]
# logs : (chemin, offset_step) — chaque phase continue logge sa barre tqdm de 0..N
CONST_LOG = [(LOGS/"run_mini_constant.log",0),(LOGS/"run_mini_constant_continue.log",20000),
             (LOGS/"run_mini_constant_continue2.log",50000),(LOGS/"run_mini_constant_finevar.log",56000),
             (LOGS/"run_mini_constant_resume.log",57200)]
COS_LOG   = [(LOGS/"run_mini_cosine.log",0),(LOGS/"run_mini_cosine_continue.log",20000),
             (LOGS/"run_mini_cosine_continue2.log",50000)]

def succ(csvs, finevar=None):
    d = {}
    for p in csvs:
        for r in csv.DictReader(open(p)):
            d[int(r["step"])] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
    if finevar:
        for r in csv.DictReader(open(finevar)):
            if int(r["step"]) == 57000:
                d[57000] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
    st = sorted(d)
    return st, [d[s][0] for s in st], [d[s][1] for s in st], [d[s][2] for s in st]

def train(specs):
    s_l, loss, grdn, lr = [], [], [], []
    for p, off in specs:
        for line in Path(p).read_text(errors="ignore").replace("\r","\n").split("\n"):
            m = re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m:
                s_l.append(int(m.group(1))+off); loss.append(float(m.group(2)))
                grdn.append(float(m.group(3))); lr.append(float(m.group(4)))
    z = sorted(zip(s_l, loss, grdn, lr)); return [list(t) for t in zip(*z)]

def vloss(specs):
    st, vl = [], []
    for p, _ in specs:
        for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", Path(p).read_text(errors="ignore").replace("\r","\n")):
            vl.append(float(m.group(1))); st.append(int(m.group(2)))
    z = sorted(zip(st, vl)); return [list(t) for t in zip(*z)]

RUNS = [
    ("constant 1e-4", "tab:red",  succ(CONST_CSV, ROOT/"mini_constant_finevar"/"rollouts_500.csv"), train(CONST_LOG), vloss(CONST_LOG)),
    ("cosine (vagues SGDR)", "tab:blue", succ(COS_CSV), train(COS_LOG), vloss(COS_LOG)),
]

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("mini-CNN sur Can (vision pure, 1.84M) — 1k→80k : cosine (vagues) vs LR constant", fontsize=13, weight="bold")
for name, col, (st,sr,lo,hi), (tst,tl,gn,lr), (vst,vl) in RUNS:
    suff = "  (succès →50k)" if "cosine" in name else ""
    ax[0,0].plot(st, sr, "-o", color=col, ms=3, label=name+suff); ax[0,0].fill_between(st, lo, hi, color=col, alpha=0.15)
    ax[0,1].plot(tst, tl, "-", color=col, alpha=0.5, lw=0.9, label=f"{name} train")
    ax[0,1].plot(vst, vl, "--", color=col, alpha=0.95, lw=1.2, label=f"{name} val")
    ax[1,0].plot(tst, lr, "-", color=col, lw=1.5, label=name)
    ax[1,1].plot(tst, gn, "-", color=col, alpha=0.8, lw=1, label=name)

for a in ax.flat:
    for x in (20000, 50000): a.axvline(x, ls=":", c="gray", lw=0.8)
ax[0,0].set_title("Succès 500-rollouts (+ IC95 Wilson)"); ax[0,0].set_ylabel("succès (%)"); ax[0,0].axhline(2, ls=":", c="lightgray", lw=0.8)
ax[0,1].set_title("loss train (—) + val (- -) — log"); ax[0,1].set_ylabel("loss"); ax[0,1].set_yscale("log")
ax[1,0].set_title("learning rate"); ax[1,0].set_ylabel("LR"); ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[1,1].set_title("grad_norm — log"); ax[1,1].set_ylabel("grad_norm"); ax[1,1].set_yscale("log")
for a in ax.flat: a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout()
out = ROOT/"courbes_minicnn_full_1k_80k.png"; fig.savefig(out, dpi=150)
print(f"écrit : {out}")
for name,_,(st,sr,_,_),_,_ in RUNS: print(f"{name:22s}: {len(st)} pts succès, max {max(sr):.1f}% @ {st[sr.index(max(sr))]}")
