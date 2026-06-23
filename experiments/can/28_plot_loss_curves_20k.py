"""Trace train_loss + val_loss pour les modèles retrainés 10k → 20k via chain 27.

Sources :
- Log 10k initial : results/logs/can/run_<run>.log (lignes "step:X loss:Y" et "val_loss:Y valstep:X")
- Log 20k resume  : results/logs/can/run_27_resume_all.log (mêmes lignes, steps 10000+)

Sortie : results/runs/can/<run>/loss_curve_20k.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Patterns lerobot
RE_TRAIN = re.compile(r"step:(\d+(?:K)?)\s+smpl:.*?\bloss:([\d.]+)")
RE_VAL   = re.compile(r"val_loss:([\d.]+)\s+valstep:(\d+)")


def parse_step(s):
    """'1K' -> 1000, '15000' -> 15000."""
    if s.endswith("K"):
        return int(s[:-1]) * 1000
    return int(s)


def extract(log_path):
    """Renvoie (train_x, train_y, val_x, val_y) ; train_x est inféré séquentiellement
    car les logs lerobot arrondissent step à 'NK' (perdent la précision)."""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    if not Path(log_path).exists():
        return train_steps, train_losses, val_steps, val_losses
    txt = Path(log_path).read_text()
    # Train : on prend tous les step/loss, plus tard on déduit la résolution
    for m in RE_TRAIN.finditer(txt):
        train_steps.append(parse_step(m.group(1)))
        train_losses.append(float(m.group(2)))
    for m in RE_VAL.finditer(txt):
        val_steps.append(int(m.group(2)))
        val_losses.append(float(m.group(1)))
    return train_steps, train_losses, val_steps, val_losses


def filter_for_run(log_27, run_name):
    """Le log chain 27 contient les sorties de TOUS les runs séquentiels.
    On garde uniquement le segment entre '[27]... TRAIN RESUME <run>' et le suivant."""
    if not Path(log_27).exists():
        return ""
    txt = Path(log_27).read_text()
    # Trouve début segment
    start_pat = rf"\[27\][^\n]*TRAIN RESUME {re.escape(run_name)}"
    next_pat  = rf"\[27\][^\n]*TRAIN RESUME (?!{re.escape(run_name)})"
    m_start = re.search(start_pat, txt)
    if not m_start:
        return ""
    start = m_start.end()
    m_next = re.search(next_pat, txt[start:])
    end = start + m_next.start() if m_next else len(txt)
    return txt[start:end]


def plot_one(run_dir, run_name, log_orig, log_27):
    out_png = Path(run_dir) / "loss_curve_20k.png"

    # Phase 0-10k : log d'origine
    tx0, ty0, vx0, vy0 = extract(log_orig)
    # Phase 10k-20k : log chain 27 (filtré par run_name)
    seg_27 = filter_for_run(log_27, run_name)
    txt27_path = Path("/tmp") / f"_{run_name}_seg27.log"
    txt27_path.write_text(seg_27)
    tx1, ty1, vx1, vy1 = extract(str(txt27_path))

    # Concat — les val_steps sont absolus dans les 2 logs
    val_x = vx0 + vx1
    val_y = vy0 + vy1
    # Train : les step:NK perdent la précision. On reconstruit en utilisant
    # juste les ratios + le nb total de lignes pour positionner.
    # Plus simple : on plot avec X = index séquentiel × log_freq.
    # Phase 0-10k : log_freq=50 (par défaut Mac), donc steps = 50, 100, 150, ...
    train_x = list(range(50, 50 * (len(ty0) + 1), 50))[: len(ty0)]
    # Phase 10k-20k via chain 27 : log_freq=200 (chain a spécifié 200)
    train_x_resume = [10000 + 200 * (i + 1) for i in range(len(ty1))]
    train_x = train_x + train_x_resume
    train_y = ty0 + ty1

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train_x, train_y, color="#1f77b4", lw=0.8, alpha=0.7, label="train_loss")
    ax.plot(val_x, val_y, color="#d62728", lw=2.0, marker=".", markersize=5, label="val_loss")
    ax.axvline(10000, color="gray", ls="--", lw=1, alpha=0.6, label="resume 10k → 20k")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_xlim(0, 20500)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"{run_name}  (val_loss : {min(val_y):.4f} @ step {val_x[val_y.index(min(val_y))]})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"  → {out_png}")
    return out_png


def main():
    log_27 = "results/logs/can/run_27_resume_all.log"
    for run_name, log_orig in [
        ("16_proprio_birdview",         "results/logs/can/run_16_proprio_birdview.log"),
        ("26_proprio_birdview_resnet34","results/logs/can/run_26_chain.log"),
    ]:
        run_dir = f"results/runs/can/{run_name}"
        if not Path(run_dir).exists():
            print(f"  skip {run_name} : run_dir absent")
            continue
        print(f"== {run_name} ==")
        plot_one(run_dir, run_name, log_orig, log_27)


if __name__ == "__main__":
    main()
