#!/usr/bin/env python3
"""Frise combinée run31 (0->40k) PUIS fine-tune look-at greffé à 40k (warm-start).
4 panneaux standard (succès+IC95 / loss / lr / grad), trait vertical = début du fine-tune.
  venv312/bin/python experiments/can/59_plot_finetune_vs_run31.py
"""
import csv, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF = 40000  # le fine-tune démarre du checkpoint 40k de run31

R31_DIR = "results/runs/can/31_proprio_birdview_r34_bigunet"
FT_DIR = "results/runs/phase6_camera/lookat_finetune_run31"
R31_LOG = "results/logs/can/run_31_r34_bigunet.log"
# préfère le log concaténé (15k + prolongation) s'il existe
FT_LOG = "results/logs/phase6_camera/run_lookat_ft_all.log" \
    if os.path.exists("results/logs/phase6_camera/run_lookat_ft_all.log") \
    else "results/logs/phase6_camera/run_lookat_ft.log"
OUT = "results/runs/phase6_camera/lookat_finetune_run31/finetune_vs_run31.png"


def smooth(y, w=15):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    h = w // 2
    return np.array([y[max(0, i - h):min(len(y), i + h + 1)].mean() for i in range(len(y))])


def load_rollouts(path):
    by = {}
    for r in csv.DictReader(open(path)):
        by[int(float(r["step"]))] = r
    rows = [by[s] for s in sorted(by)]
    g = lambda r, k, d=None: float(r[k]) if r.get(k) not in (None, "") else d
    st = np.array([int(float(r["step"])) for r in rows])
    sr = np.array([g(r, "success_rate") * 100 for r in rows])
    lo = np.array([g(r, "ci95_low", g(r, "success_rate")) * 100 for r in rows])
    hi = np.array([g(r, "ci95_high", g(r, "success_rate")) * 100 for r in rows])
    return st, sr, lo, hi


def parse_step(s):
    return int(float(s[:-1]) * 1000) if s.endswith("K") else int(s)


def load_log(path):
    txt = open(path, errors="ignore").read()
    tr = re.findall(r"step:([0-9.]+K?) smpl.*?loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)", txt)
    vl = re.findall(r"val_loss:([0-9.]+) valstep:([0-9]+)", txt)
    st = np.array([parse_step(t[0]) for t in tr])
    tl = np.array([float(t[1]) for t in tr])
    gr = np.array([float(t[2]) for t in tr])
    lr = np.array([float(t[3]) for t in tr])
    vst = np.array([int(v[1]) for v in vl])
    vlo = np.array([float(v[0]) for v in vl])
    o = np.argsort(st)
    vo = np.argsort(vst)
    return st[o], tl[o], gr[o], lr[o], vst[vo], vlo[vo]


# --- succès ---
r_st, r_sr, r_lo, r_hi = load_rollouts(f"{R31_DIR}/rollouts_50.csv")
f_st, f_sr, f_lo, f_hi = load_rollouts(f"{FT_DIR}/rollouts_100.csv")
e500 = list(csv.DictReader(open(f"{R31_DIR}/eval500_best.csv")))
e_st = int(float(e500[0]["step"])); e_sr = float(e500[0]["success_rate"]) * 100

# --- logs ---
rs, rtl, rgr, rlr, rvs, rvl = load_log(R31_LOG)
fs, ftl, fgr, flr, fvs, fvl = load_log(FT_LOG)

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
BLUE, ORNG, GRN = "#1f77b4", "#ff7f0e", "#2ca02c"
K = 1000.0


def mark(a):
    a.axvline(OFF / K, color="red", ls="--", lw=1.3, alpha=0.7)


# 1) succès
a = ax[0, 0]
a.fill_between(r_st / K, r_lo, r_hi, color=BLUE, alpha=0.15)
a.plot(r_st / K, r_sr, "-o", color=BLUE, ms=3, lw=1.6, label="run31 (cartésien, clean) n=50")
a.plot(e_st / K, e_sr, "*", color=GRN, ms=17, label=f"run31 @500 = {e_sr:.1f}%")
a.fill_between((f_st + OFF) / K, f_lo, f_hi, color=ORNG, alpha=0.18)
a.plot((f_st + OFF) / K, f_sr, "-o", color=ORNG, ms=3, lw=1.6, label="fine-tune look-at (clean) n=100")
mark(a)
a.text(OFF / K, 5, " warm-start →\n début fine-tune", color="red", fontsize=8, va="bottom")
a.set_ylim(-3, 103); a.set_title("Succès (rollouts) %")
a.set_xlabel("step (k) — run31 puis fine-tune greffé à 40k"); a.legend(loc="center right", fontsize=8)

# 2) loss
a = ax[0, 1]
a.plot(rs / K, smooth(rtl), color=BLUE, alpha=0.5, lw=1, label="run31 train")
a.plot(rvs / K, smooth(rvl), color=BLUE, lw=1.8, label="run31 val")
a.plot((fs + OFF) / K, smooth(ftl), color=ORNG, alpha=0.5, lw=1, label="fine-tune train")
a.plot((fvs + OFF) / K, smooth(fvl), color=ORNG, lw=1.8, label="fine-tune val")
mark(a); a.set_yscale("log"); a.set_title("loss train + val — log")
a.set_xlabel("step (k)"); a.legend(fontsize=7)

# 3) lr
a = ax[1, 0]
a.plot(rs / K, rlr, color=BLUE, lw=1.6, label="run31 (cosine)")
a.plot((fs + OFF) / K, flr, color=ORNG, lw=1.6, label="fine-tune (constant 1e-4)")
mark(a); a.set_yscale("log"); a.set_title("learning rate — log")
a.set_xlabel("step (k)"); a.legend(fontsize=8)

# 4) grad
a = ax[1, 1]
a.plot(rs / K, smooth(rgr), color=BLUE, lw=1.4, label="run31")
a.plot((fs + OFF) / K, smooth(fgr), color=ORNG, lw=1.4, label="fine-tune")
mark(a); a.set_yscale("log"); a.set_title("grad_norm — log")
a.set_xlabel("step (k)"); a.legend(fontsize=8)

fig.suptitle("run31 (cartésien) → fine-tune look-at greffé à 40k (warm-start) — 4 panneaux", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=110)
print(f"Sauvé : {OUT}")
print(f"run31 succès: {len(r_st)} pts (0-40k), @500={e_sr:.1f}% ; fine-tune: {len(f_st)} pts (greffés {OFF//1000}-{(OFF+f_st.max())//1000}k)")
