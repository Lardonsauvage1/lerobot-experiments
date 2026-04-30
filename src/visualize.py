"""
Outils de visualisation pour comparer les expériences.
Génère des graphes standardisés : loss vs epoch, vs temps, vs FLOPs, comparatifs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from .tracker import load_all_runs

RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_run_losses(run: dict, save_dir: str = None):
    """Génère les 3 graphes de loss pour un run : vs epoch, vs temps, vs FLOPs.

    save_dir : chemin du dossier où sauvegarder (ex: run_dir).
               Si None, utilise results/
    """
    train = run["train_losses"]
    test = run["test_losses"]

    if save_dir is None:
        save_dir = str(RESULTS_DIR)
    Path(save_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss vs epoch
    axes[0].plot(train, label="Train", linewidth=2)
    axes[0].plot(test, label="Test", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Loss vs temps
    tpe = run["time_per_epoch_s"]
    times = [tpe * (i + 1) for i in range(len(train))]
    axes[1].plot(times, train, label="Train", linewidth=2)
    axes[1].plot(times, test, label="Test", linewidth=2)
    axes[1].set_xlabel("Temps (secondes)")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Loss vs Temps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Loss vs FLOPs
    if run.get("flops_total_train", 0) > 0:
        flops_per_epoch = run["flops_total_train"] / len(train)
        flops = [flops_per_epoch * (i + 1) for i in range(len(train))]
        axes[2].plot(flops, train, label="Train", linewidth=2)
        axes[2].plot(flops, test, label="Test", linewidth=2)
        axes[2].set_xlabel("FLOPs")
        axes[2].set_ylabel("MSE Loss")
        axes[2].set_title("Loss vs FLOPs")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "FLOPs non mesuré", ha="center", va="center")
        axes[2].set_title("Loss vs FLOPs")

    plt.suptitle(f"{run['experiment']} — {run['architecture']}", fontsize=14)
    plt.tight_layout()
    filepath = f"{save_dir}/losses.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Graphe : {filepath}")
    return filepath


def plot_compare(runs: list[dict] = None, filter_experiment: str = None,
                 metric: str = "test", x_axis: str = "epoch", save_dir: str = None):
    """Superpose les courbes de loss de plusieurs runs.

    x_axis : "epoch", "time", ou "flops"
    metric : "train" ou "test"
    """
    if runs is None:
        runs = load_all_runs()
    if filter_experiment:
        runs = [r for r in runs if r["experiment"] == filter_experiment]
    if not runs:
        print("Aucun run trouvé.")
        return

    if save_dir is None:
        save_dir = str(RESULTS_DIR)
    Path(save_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for run in runs:
        losses = run[f"{metric}_losses"]
        label = f"{run['experiment']} — {run['architecture']}"

        if x_axis == "epoch":
            x = list(range(1, len(losses) + 1))
            xlabel = "Epoch"
        elif x_axis == "time":
            tpe = run["time_per_epoch_s"]
            x = [tpe * (i + 1) for i in range(len(losses))]
            xlabel = "Temps (secondes)"
        elif x_axis == "flops":
            if run.get("flops_total_train", 0) == 0:
                continue
            fpe = run["flops_total_train"] / len(losses)
            x = [fpe * (i + 1) for i in range(len(losses))]
            xlabel = "FLOPs"
        else:
            raise ValueError(f"x_axis invalide: {x_axis}")

        ax.plot(x, losses, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"MSE Loss ({metric})")
    ax.set_title(f"Comparaison — Loss {metric} vs {x_axis}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    exp_name = filter_experiment or "all"
    filepath = f"{save_dir}/compare_{exp_name}_{metric}_vs_{x_axis}.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Graphe comparatif : {filepath}")
    return filepath
