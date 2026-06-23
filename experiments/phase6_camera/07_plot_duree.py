"""Phase 6 — Test DUREE : mini-CNN 150 (jitte) prolonge bien au-dela de 70K avec warm
restarts (vagues cosine). Trace succes every-10K + train loss + val_loss + grad_norm + LR
sur 0 -> 460K (selon donnees dispo). Robuste aux donnees partielles."""
import csv, re, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase6_camera"); LOGS = Path("results/logs/phase6_camera")

def metrics(lognames):
    st, lo, gr, lr, vl = [], [], [], [], []
    for logname in lognames:
        p = LOGS / logname
        if not p.exists(): continue
        last = None
        for line in p.read_text(errors="ignore").replace("\r", "\n").split("\n"):
            m = re.search(r"loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m: last = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            mv = re.search(r"val_loss:([\d.]+) valstep:(\d+)", line)
            if mv and last:
                st.append(int(mv.group(2))); lo.append(last[0]); gr.append(last[1]); lr.append(last[2]); vl.append(float(mv.group(1)))
    z = sorted(zip(st, lo, gr, lr, vl))
    return ([t[0] for t in z], [t[1] for t in z], [t[2] for t in z], [t[3] for t in z], [t[4] for t in z]) if z else ([], [], [], [], [])

def succ(patterns):
    pts = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            try:
                for r in csv.DictReader(open(f)):
                    pts[int(r["step"])] = float(r["success_rate"]) * 100
            except Exception: pass
    xs = sorted(pts); return xs, [pts[x] for x in xs]

sx, sy = succ(["mini_camaug_150/rollouts_500.csv", "mini_camaug_150_ext/rollouts_500.csv",
               "mini_camaug_150_ext2/rollouts_500.csv", "mini_camaug_150_ext3/rollouts_500.csv"])
st, lo, gr, lr, vl = metrics(["run_camaug_150.log", "run_cont_200000.log", "run_cont_330000.log", "run_cont_460000.log"])
C = "tab:green"

fig, ax = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Phase 6 — DUREE : mini-CNN 150 (jitte) prolonge avec warm restarts (vagues cosine)", fontsize=12, weight="bold")
ax[0, 0].plot(sx, sy, "-o", color=C, ms=4)
ymax = max([8] + sy)
ax[0, 0].set_title("succes 500-rollouts (nominal) tous les 10K"); ax[0, 0].set_ylabel("succes %"); ax[0, 0].set_ylim(-1, ymax * 1.15)
ax[0, 1].plot(st, lo, color=C, lw=.6); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("train loss (log)")
ax[1, 0].plot(st, vl, color=C, lw=.6); ax[1, 0].set_title("val_loss")
ax[1, 1].plot(st, gr, color="tab:gray", lw=.6, label="grad_norm")
ax[1, 1].set_yscale("log"); ax[1, 1].set_ylabel("grad_norm"); ax[1, 1].set_title("grad_norm (gris) + LR (orange)")
axb = ax[1, 1].twinx(); axb.plot(st, lr, color="tab:orange", lw=.8, label="LR"); axb.set_ylabel("LR")
for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]): a.set_xlabel("step global"); a.grid(alpha=.3)
fig.tight_layout(); out = ROOT / "courbe_duree.png"; fig.savefig(out, dpi=150)
print(f"ecrit : {out}")
print("succes:", [(x // 1000, round(y, 1)) for x, y in zip(sx, sy)])
