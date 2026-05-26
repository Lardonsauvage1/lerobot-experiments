"""Assemble la grille PAS × DONNÉES (modèle [32,64,128] mini-CNN, 500 rollouts, Wilson)
depuis 67 (colonne @4 pas) + 71 (pas {2,10,20,50}).

Sortie : results/runs/lift/grid_steps_x_data_500.{json,png}
"""
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import matplotlib.pyplot as plt

STEPS = [2, 4, 10, 20, 50]
NS = [150, 100, 50, 20, 10]
OUT = Path("results/runs/lift/grid_steps_x_data_500")


def main():
    d4 = json.load(open("results/runs/lift/67_dataeff_500.json"))["results"]  # @4 pas
    for r in d4:
        r["steps"] = 4
    d71 = json.load(open("results/runs/lift/71_steps_data_500.json"))["results"]
    grid = {(r["n_demos"], r["steps"]): r for r in (d4 + d71)}

    rows = []
    for N in NS:
        for s in STEPS:
            r = grid.get((N, s))
            if r:
                rows.append({"n_demos": N, "steps": s, "success_rate": r["success_rate"],
                             "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                             "t_success_median": r["t_success_median"], "n": r["n"],
                             "n_success": r["n_success"]})
    OUT.with_suffix(".json").write_text(json.dumps(
        {"model": "[32,64,128] mini-CNN", "n_eval": 500, "ci": "wilson95", "results": rows}, indent=2))

    # texte : grille succès % [IC95]
    print("=== GRILLE PAS × DONNÉES — succès % [IC95] ===")
    print("pas \\ N | " + " | ".join(f"N={N:>3}" for N in NS))
    for s in STEPS:
        cells = []
        for N in NS:
            r = grid.get((N, s))
            cells.append(f"{r['success_rate']*100:4.1f} [{r['ci95_low']*100:4.1f}-{r['ci95_high']*100:5.1f}]"
                         if r else "    --     ")
        print(f"  {s:>3}   | " + " | ".join(cells))

    # plot : succès vs pas, une courbe par N
    colors = {150: "#2ca02c", 100: "#1f77b4", 50: "#9467bd", 20: "#ff7f0e", 10: "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for N in NS:
        xs, ys, lo, hi = [], [], [], []
        for s in STEPS:
            r = grid.get((N, s))
            if not r:
                continue
            xs.append(s); ys.append(r["success_rate"] * 100)
            lo.append((r["success_rate"] - r["ci95_low"]) * 100)
            hi.append((r["ci95_high"] - r["success_rate"]) * 100)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=colors[N], capsize=4, label=f"N={N} démos")
    ax.set_xscale("log"); ax.set_xticks(STEPS); ax.set_xticklabels(STEPS)
    ax.set_xlabel("pas de diffusion (num_inference_steps, échelle log)")
    ax.set_ylabel("succès (%), 500 rollouts appariés (IC95 Wilson)")
    ax.set_ylim(0, 102); ax.grid(True, alpha=0.3)
    ax.set_title("Pas de diffusion × données — [32,64,128] mini-CNN, Lift 500 rollouts")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=120)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
