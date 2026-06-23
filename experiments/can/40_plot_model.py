"""GRAPHE STANDARD d'un modèle — 4 panneaux (succès+IC95 / loss / learning rate / grad_norm).

C'est le format par défaut quand on demande « le graphe d'un modèle ».
Auto-détecte les rollouts du run-dir (500r prioritaire, sinon 50r ; + eval500_best.csv en
surimpression « étoile fiable » si présent) et lit loss/lr/grad depuis le(s) log(s).

Usage :
  venv312/bin/python experiments/can/40_plot_model.py <run_dir> <log_glob> [--title T] [--out PNG]
Ex :
  ... 40_plot_model.py results/runs/can/31_proprio_birdview_r34_bigunet "results/logs/can/run_31_r34_bigunet.log"
"""
import argparse, csv, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C, CL, STAR = "#d62728", "#f4a3a3", "#2ca02c"


def smooth(y, w=15):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    h = w // 2
    return np.array([y[max(0, i - h):min(len(y), i + h + 1)].mean() for i in range(len(y))])


def load_rollouts(path):
    rows = list(csv.DictReader(open(path)))
    g = lambda r, k, d=None: float(r[k]) if r.get(k) not in (None, "") else d
    st = np.array([int(float(r["step"])) for r in rows])
    sr = np.array([g(r, "success_rate") * 100 for r in rows])
    lo = np.array([g(r, "ci95_low", g(r, "success_rate")) * 100 for r in rows])
    hi = np.array([g(r, "ci95_high", g(r, "success_rate")) * 100 for r in rows])
    n = int(g(rows[0], "n", 0)) if rows else 0
    return st, sr, lo, hi, n


def load_log(log_glob):
    steps, tl, vl_, gr, lr = [], [], [], [], []
    seen = set()
    for lg in sorted(glob.glob(log_glob)):
        txt = open(lg, errors="ignore").read()
        tr = re.findall(r"(?<!val_)loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)", txt)
        vl = re.findall(r"val_loss:([0-9.]+) valstep:(\d+)", txt)
        for i in range(min(len(tr), len(vl))):
            s = int(vl[i][1])
            if s in seen:
                continue
            seen.add(s); steps.append(s)
            tl.append(float(tr[i][0])); vl_.append(float(vl[i][0]))
            gr.append(float(tr[i][1])); lr.append(float(tr[i][2]))
    o = np.argsort(steps)
    return (np.array(steps)[o], np.array(tl)[o], np.array(vl_)[o], np.array(gr)[o], np.array(lr)[o]) if steps \
        else (np.array([]),) * 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("log_glob", nargs="?", default="")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = Path(a.run_dir)

    # --- succès : 500r prioritaire, sinon 50r ; + eval500_best en overlay ---
    main_csv = rd / "rollouts_500.csv"
    rollN = 500
    if not main_csv.exists():
        main_csv = rd / "rollouts_50.csv"; rollN = 50
    st, sr, lo, hi, n = load_rollouts(main_csv) if main_csv.exists() else ([], [], [], [], 0)
    e500 = rd / "eval500_best.csv"

    lst, tl, vl, gr, lr = load_log(a.log_glob) if a.log_glob else ([],) * 5
    title = a.title or rd.name
    out = a.out or str(rd / "full_curves.png")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.flatten()
    # 1) succès + IC95
    if len(st):
        ax[0].fill_between(st / 1000, lo, hi, color=C, alpha=0.18)
        ax[0].plot(st / 1000, sr, "-o", color=C, ms=5, label=f"{rollN} rollouts (±IC95)")
    if rollN != 500 and e500.exists():
        st5, sr5, lo5, hi5, _ = load_rollouts(e500)
        ax[0].errorbar(st5 / 1000, sr5, yerr=[sr5 - lo5, hi5 - sr5], fmt="*", color=STAR,
                       ms=18, capsize=4, zorder=5, label="500 rollouts (fiable)")
        for x, y in zip(st5 / 1000, sr5):
            ax[0].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(-42, -2),
                           fontsize=10, color=STAR, fontweight="bold")
    ax[0].set_title("Succès (rollouts) %"); ax[0].set_ylim(-3, 102); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("step (k)"); ax[0].legend(loc="lower right", fontsize=8)
    # 2) loss train+val (log)
    if len(lst):
        ax[1].plot(lst / 1000, smooth(tl), "-", color=CL, lw=1.3, label="train (lissée)")
        ax[1].plot(lst / 1000, smooth(vl), "-", color=C, lw=2, label="val (lissée)")
        ax[1].set_yscale("log"); ax[1].set_title("loss train + val — log")
        ax[1].grid(alpha=.3, which="both"); ax[1].set_xlabel("step (k)"); ax[1].legend(fontsize=8)
        ax[2].plot(lst / 1000, lr, "-", color=C, lw=1.5); ax[2].set_title("learning rate")
        ax[2].grid(alpha=.3); ax[2].set_xlabel("step (k)")
        ax[3].plot(lst / 1000, gr, "-", color=C, lw=0.7, alpha=.5)
        ax[3].plot(lst / 1000, smooth(gr), "-", color=C, lw=2)
        ax[3].set_yscale("log"); ax[3].set_title("grad_norm — log")
        ax[3].grid(alpha=.3, which="both"); ax[3].set_xlabel("step (k)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=130)
    print(f"Sauvé : {out}  (succès {len(st)} pts @{rollN}r, log {len(lst)} pts)")


if __name__ == "__main__":
    main()
