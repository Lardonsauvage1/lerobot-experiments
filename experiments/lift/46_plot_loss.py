"""Parse le log lerobot-train et trace loss + grad_norm + lr.

Le log contient des lignes du type :
  INFO ... ot_train.py:439 step:200 smpl:6K ep:132 epch:0.66 loss:0.618 grdn:3.721 lr:2.0e-05
Loggées toutes les 200 steps. On les parse en ordre et reconstruit la courbe.
"""

import argparse
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Regex pour extraire loss/grdn/lr (le step de cette ligne est rond style "1K", on l'ignore)
RX = re.compile(r"loss:([\d.eE+-]+)\s+grdn:([\d.eE+-]+)\s+lr:([\d.eE+-]+)")
# Lignes val-loss injectées par 50_train_valloss.py : "val_loss:0.0762 valstep:1500"
VAL_RX = re.compile(r"val_loss:([\d.eE+-]+)\s+valstep:(\d+)")


def parse_log(path: Path, log_freq: int = 200):
    text = path.read_text(errors="ignore")
    # Le log a des \r de tqdm — découper les vraies lignes par \n
    lines = text.split("\n")
    steps, losses, grdns, lrs = [], [], [], []
    val_steps, val_losses = [], []
    idx = 0
    for line in lines:
        mv = VAL_RX.search(line)
        if mv:
            val_losses.append(float(mv.group(1))); val_steps.append(int(mv.group(2)))
            continue
        m = RX.search(line)
        if not m:
            continue
        idx += 1
        steps.append(idx * log_freq)   # logged every log_freq steps, starting at log_freq
        losses.append(float(m.group(1)))
        grdns.append(float(m.group(2)))
        lrs.append(float(m.group(3)))
    return steps, losses, grdns, lrs, val_steps, val_losses


def plot(steps, losses, grdns, lrs, out: Path, val_steps=None, val_losses=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.5, label="train")
    if val_steps:
        ax.plot(val_steps, val_losses, color="#d62728", linewidth=1.3, label="val (test)")
        ax.legend()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/logs/lift/run_46_diffusion.log")
    ap.add_argument("--out", default="results/runs/lift/46_diffusion_official/loss_curves.png")
    ap.add_argument("--log-freq", type=int, default=200)
    args = ap.parse_args()
    LOG_PATH, OUT_PATH = Path(args.log), Path(args.out)
    if not LOG_PATH.exists():
        sys.exit(f"Log introuvable : {LOG_PATH}")
    steps, losses, grdns, lrs, val_steps, val_losses = parse_log(LOG_PATH, args.log_freq)
    if not steps:
        sys.exit("Aucune ligne de loss trouvée dans le log.")
    print(f"Points parsés : {len(steps)}  | step range : {steps[0]} → {steps[-1]}")
    print(f"Loss range    : {min(losses):.4f} → {max(losses):.4f}   (final {losses[-1]:.4f})")
    print(f"Grad norm     : {min(grdns):.3f} → {max(grdns):.3f}     (final {grdns[-1]:.3f})")
    print(f"LR            : {min(lrs):.2e} → {max(lrs):.2e}         (final {lrs[-1]:.2e})")
    if val_steps:
        print(f"Val loss      : {len(val_steps)} points, {min(val_losses):.4f} → {max(val_losses):.4f} (final {val_losses[-1]:.4f})")
    plot(steps, losses, grdns, lrs, OUT_PATH, val_steps, val_losses)
