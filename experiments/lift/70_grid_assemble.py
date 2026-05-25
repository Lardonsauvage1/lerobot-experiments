"""Assemble la grille 15 (taille U-Net × données) depuis 67 (mini-CNN [32,64,128]) + 69
(mini-CNN [64,128,256], [128,256,512]) — tous 500 rollouts appariés @ 4 pas, IC95 Wilson.

Sortie : results/runs/lift/grid_data_x_unet_500.{json,png}
"""
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import matplotlib.pyplot as plt

DIMS = ["[32,64,128]", "[64,128,256]", "[128,256,512]"]
NS = [10, 20, 50, 100, 150]
OUT = Path("results/runs/lift/grid_data_x_unet_500")


def main():
    d32 = json.load(open("results/runs/lift/67_dataeff_500.json"))["results"]
    for r in d32:
        r["down_dims"] = "[32,64,128]"
    d69 = json.load(open("results/runs/lift/69_grid_500.json"))["results"]
    grid = {(r["down_dims"], r["n_demos"]): r for r in (d32 + d69)}

    # json unifié
    rows = []
    for d in DIMS:
        for N in NS:
            r = grid.get((d, N))
            if r:
                rows.append({"down_dims": d, "n_demos": N, "success_rate": r["success_rate"],
                             "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                             "t_success_median": r["t_success_median"], "n": r["n"],
                             "n_success": r["n_success"]})
    OUT.with_suffix(".json").write_text(json.dumps(
        {"steps": 4, "n_eval": 500, "ci": "wilson95", "results": rows}, indent=2))

    # plot : succès vs N, une courbe par taille de U-Net, barres d'erreur Wilson
    colors = {"[32,64,128]": "#2ca02c", "[64,128,256]": "#1f77b4", "[128,256,512]": "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d in DIMS:
        xs, ys, lo, hi = [], [], [], []
        for N in NS:
            r = grid.get((d, N))
            if not r:
                continue
            xs.append(N); ys.append(r["success_rate"] * 100)
            lo.append((r["success_rate"] - r["ci95_low"]) * 100)
            hi.append((r["ci95_high"] - r["success_rate"]) * 100)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=colors[d], capsize=4, label=f"U-Net {d}")
    ax.set_xscale("log")
    ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("nb de démos d'entraînement (échelle log)")
    ax.set_ylabel("succès (%) @ 4 pas, 500 rollouts appariés (IC95 Wilson)")
    ax.set_ylim(70, 102); ax.grid(True, alpha=0.3)
    ax.set_title("Grille taille U-Net × données (mini-CNN) — Lift, 500 rollouts")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=120)
    print(f"JSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
