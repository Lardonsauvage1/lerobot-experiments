"""Phase 6 — Test CAPACITE : ResNet18 (11.2M vision) vs mini-CNN (0.03M) sur la tache
jittee 10cm/10deg (150 demos). Trace succes every-10K + train loss + val_loss + grad_norm.
Robuste aux donnees partielles (ne trace que ce qui existe)."""
import csv, re, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase6_camera"); LOGS = Path("results/logs/phase6_camera")

def metrics(logname):
    p = LOGS / logname
    if not p.exists(): return [], [], [], [], []
    st, lo, gr, lr, vl = [], [], [], [], []
    last = None
    for line in p.read_text(errors="ignore").replace("\r", "\n").split("\n"):
        m = re.search(r"loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
        if m: last = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        mv = re.search(r"val_loss:([\d.]+) valstep:(\d+)", line)
        if mv and last:
            st.append(int(mv.group(2))); lo.append(last[0]); gr.append(last[1]); lr.append(last[2]); vl.append(float(mv.group(1)))
    return st, lo, gr, lr, vl

def succ(patterns):
    pts = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            try:
                for r in csv.DictReader(open(f)):
                    pts[int(r["step"])] = float(r["success_rate"]) * 100
            except Exception: pass
    xs = sorted(pts); return xs, [pts[x] for x in xs]

mini_x, mini_y = succ(["mini_camaug_150/rollouts_500.csv"])
r18_x, r18_y = succ(["r18_eval.csv", "r18_eval_*.csv"])
r34_x, r34_y = succ(["r34_eval_*.csv"])
sep_x, sep_y = succ(["r34sep_eval*.csv"])
ms, ml, mg, mlr, mv = metrics("run_camaug_150.log")
rs, rl, rg, rlr, rv = metrics("run_camaug_150_resnet18.log")
qs, ql, qg, qlr, qv = metrics("run_camaug_150_resnet34.log")
ps, pl, pg, plr, pv = metrics("run_camaug_150_resnet34sep.log")
RED, BLUE, GREEN, PURPLE = "tab:red", "tab:blue", "tab:green", "tab:purple"
MODELS = [("mini-CNN 0.03M", RED, (mini_x, mini_y), (ms, ml, mg, mv)),
          ("ResNet18 11.2M", BLUE, (r18_x, r18_y), (rs, rl, rg, rv)),
          ("ResNet34 21.3M", GREEN, (r34_x, r34_y), (qs, ql, qg, qv)),
          ("ResNet34-sep 42.6M", PURPLE, (sep_x, sep_y), (ps, pl, pg, pv))]

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Phase 6 — CAPACITE : echelle mini-CNN -> ResNet18 -> ResNet34 -> ResNet34-sep — jitter 10cm/10deg, 150 demos", fontsize=11, weight="bold")
ymax = 8
for name, col, (sx, sy), (mx, mlo, mgr, mvl) in MODELS:
    if sx: ax[0, 0].plot(sx, sy, "-o", color=col, ms=5, label=name); ymax = max([ymax] + sy)
    if mx:
        ax[0, 1].plot(mx, mlo, color=col, lw=.7, label=name)
        ax[1, 0].plot(mx, mvl, color=col, lw=.7, label=name)
        ax[1, 1].plot(mx, mgr, color=col, lw=.7, label=name)
ax[0, 0].set_title("succes 500-rollouts (nominal) tous les 10K"); ax[0, 0].set_ylabel("succes %"); ax[0, 0].set_ylim(-1, ymax * 1.15)
ax[0, 1].set_yscale("log"); ax[0, 1].set_title("train loss (log)")
ax[1, 0].set_title("val_loss")
ax[1, 1].set_yscale("log"); ax[1, 1].set_title("grad_norm (log)")
for a in ax.flat: a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); out = ROOT / "courbe_capacite.png"; fig.savefig(out, dpi=150)
print(f"ecrit : {out}")
for name, _, (sx, sy), _ in MODELS:
    print(f"{name} succes:", [(x // 1000, round(y, 1)) for x, y in zip(sx, sy)])
