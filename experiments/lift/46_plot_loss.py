"""Parse le log lerobot-train et trace loss + grad_norm + lr.

Le log contient des lignes du type :
  INFO ... ot_train.py:439 step:200 smpl:6K ep:132 epch:0.66 loss:0.618 grdn:3.721 lr:2.0e-05
Loggées toutes les 200 steps. On les parse en ordre et reconstruit la courbe.
"""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

LOG_PATH = Path("results/logs/lift/run_46_diffusion.log")
OUT_PATH = Path("results/runs/lift/46_diffusion_official/loss_curves.png")

# Regex pour extraire loss/grdn/lr (le step de cette ligne est rond style "1K", on l'ignore)
RX = re.compile(r"loss:([\d.eE+-]+)\s+grdn:([\d.eE+-]+)\s+lr:([\d.eE+-]+)")


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    # Le log a des \r de tqdm — découper les vraies lignes par \n
    lines = text.split("\n")
    steps, losses, grdns, lrs = [], [], [], []
    idx = 0
    for line in lines:
        m = RX.search(line)
        if not m:
            continue
        idx += 1
        steps.append(idx * 200)   # logged every 200 steps starting at 200
        losses.append(float(m.group(1)))
        grdns.append(float(m.group(2)))
        lrs.append(float(m.group(3)))
    return steps, losses, grdns, lrs


def plot(steps, losses, grdns, lrs, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.5)
    ax.set_ylabel("Loss (MSE noise pred)")
    ax.set_yscale("log")
    ax.set_title(f"Diffusion Policy training — Lift PH ({len(steps)} log points, {steps[-1]} steps)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, grdns, color="#ff7f0e", linewidth=1.5)
    ax.set_ylabel("Grad norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, lrs, color="#2ca02c", linewidth=1.5)
    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"✓ Saved : {out}")


if __name__ == "__main__":
    if not LOG_PATH.exists():
        sys.exit(f"Log introuvable : {LOG_PATH}")
    steps, losses, grdns, lrs = parse_log(LOG_PATH)
    if not steps:
        sys.exit("Aucune ligne de loss trouvée dans le log.")
    print(f"Points parsés : {len(steps)}  | step range : {steps[0]} → {steps[-1]}")
    print(f"Loss range    : {min(losses):.4f} → {max(losses):.4f}   (final {losses[-1]:.4f})")
    print(f"Grad norm     : {min(grdns):.3f} → {max(grdns):.3f}     (final {grdns[-1]:.3f})")
    print(f"LR            : {min(lrs):.2e} → {max(lrs):.2e}         (final {lrs[-1]:.2e})")
    plot(steps, losses, grdns, lrs, OUT_PATH)
