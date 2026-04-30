"""
Tracker d'expériences — enregistre et compare les runs.
Toutes les expériences sont sauvegardées dans un seul fichier results/all_runs.jsonl
avec un format standardisé.
"""

import json
import time
import os
import torch
import imageio
import numpy as np
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
ALL_RUNS_FILE = RESULTS_DIR / "all_runs.jsonl"


def count_flops(model, input_shapes: dict) -> int:
    """Estime les FLOPs d'un forward pass.

    input_shapes : dict avec les shapes des entrées, ex:
        {"image": (1, 3, 64, 64), "joints": (1, 7)}

    Compte les multiplications-additions des couches Linear et Conv2d.
    C'est une estimation (ignore les activations, normalizations, etc.)
    mais c'est suffisant pour comparer des modèles entre eux.
    """
    flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * in_features * out_features (mul + add)
            flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            # FLOPs = 2 * Cout * Cin * Kh * Kw * Hout * Wout
            # On estime Hout, Wout à partir de l'input shape
            # Approximation : on suppose que les conv gardent la taille (padding=same)
            # puis MaxPool divise par 2
            pass  # On va calculer avec un forward pass réel

    # Méthode plus fiable : mesurer avec un forward pass
    # On utilise torch.utils.flop_counter si disponible
    try:
        from torch.utils.flop_counter import FlopCounterMode
        # Créer des inputs factices
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape)

        with FlopCounterMode(display=False) as counter:
            # Forward pass — on doit adapter selon le modèle
            model.eval()
            with torch.no_grad():
                if "images" in dummy_inputs and "joints" in dummy_inputs:
                    model(dummy_inputs["images"], dummy_inputs["joints"])
                elif "image" in dummy_inputs and "agent_pos" in dummy_inputs:
                    model(image=dummy_inputs["image"], agent_pos=dummy_inputs["agent_pos"])
                elif "images_seq" in dummy_inputs and "joints_seq" in dummy_inputs:
                    model(dummy_inputs["images_seq"], dummy_inputs["joints_seq"])

        flops = counter.get_total_flops()
    except Exception:
        # Fallback : estimation manuelle pour les Linear et Conv2d
        flops = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                flops += 2 * module.in_features * module.out_features
            elif isinstance(module, torch.nn.Conv2d):
                flops += 2 * module.in_channels * module.out_channels * \
                         module.kernel_size[0] * module.kernel_size[1]

    return int(flops)


def measure_inference_time(model, dummy_inputs: dict, n_runs: int = 200) -> float:
    """Mesure le temps d'inférence moyen en millisecondes."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            if "images" in dummy_inputs and "joints" in dummy_inputs:
                model(dummy_inputs["images"], dummy_inputs["joints"])
            elif "image" in dummy_inputs and "agent_pos" in dummy_inputs:
                model(image=dummy_inputs["image"], agent_pos=dummy_inputs["agent_pos"])
            elif "images_seq" in dummy_inputs and "joints_seq" in dummy_inputs:
                model(dummy_inputs["images_seq"], dummy_inputs["joints_seq"])

    # Mesure
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            if "images" in dummy_inputs and "joints" in dummy_inputs:
                model(dummy_inputs["images"], dummy_inputs["joints"])
            elif "image" in dummy_inputs and "agent_pos" in dummy_inputs:
                model(image=dummy_inputs["image"], agent_pos=dummy_inputs["agent_pos"])
            elif "images_seq" in dummy_inputs and "joints_seq" in dummy_inputs:
                model(dummy_inputs["images_seq"], dummy_inputs["joints_seq"])

    elapsed_ms = (time.time() - start) / n_runs * 1000
    return round(elapsed_ms, 3)


def save_run(run_data: dict):
    """Sauvegarde un run dans all_runs.jsonl."""
    RESULTS_DIR.mkdir(exist_ok=True)

    run_data["timestamp"] = datetime.now().isoformat()

    with open(ALL_RUNS_FILE, "a") as f:
        f.write(json.dumps(run_data) + "\n")

    print(f"Run sauvegardé dans {ALL_RUNS_FILE}")
    return run_data


def build_run_data(
    experiment: str,
    architecture: str,
    dataset: str,
    n_episodes: int | str,
    n_params: int,
    size_mb: float,
    seed: int,
    # Entraînement
    train_losses: list[float],
    test_losses: list[float],
    train_time_s: float,
    epoch_times: list[float],
    flops_total_train: int,
    n_train_samples: int,
    n_test_samples: int,
    # Inférence
    inference_time_ms: float,
    flops_inference: int,
    frames_per_call: int = 1,
    # Simulation (optionnel)
    success_rate: float | None = None,
    avg_coverage: float | None = None,
    avg_reward: float | None = None,
    # Bras robotique (optionnel)
    mae_per_joint_deg: list[float] | None = None,
    mae_mean_deg: float | None = None,
    max_error_per_joint_deg: list[float] | None = None,
    # Extra
    extra: dict | None = None,
) -> dict:
    """Construit un run_data standardisé."""

    # Paliers de performance
    thresholds = [0.1, 0.05, 0.03, 0.02, 0.01]
    cumul_times = []
    t = 0
    for et in epoch_times:
        t += et
        cumul_times.append(t)

    paliers = {}
    for thr in thresholds:
        reached = [(ep, cumul_times[ep]) for ep, l in enumerate(test_losses) if l <= thr]
        if reached:
            ep, sec = reached[0]
            paliers[f"loss_le_{thr}"] = {"epoch": ep + 1, "time_s": round(sec, 1)}

    run = {
        # Identité
        "experiment": experiment,
        "architecture": architecture,
        "dataset": dataset,
        "n_episodes": n_episodes,
        "n_params": n_params,
        "size_mb": size_mb,
        "seed": seed,
        # Entraînement
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_time_s": round(train_time_s, 1),
        "time_per_epoch_s": round(sum(epoch_times) / len(epoch_times), 2) if epoch_times else 0,
        "flops_total_train": flops_total_train,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "final_train_loss": train_losses[-1] if train_losses else 0,
        "final_test_loss": test_losses[-1] if test_losses else 0,
        "best_test_loss": min(test_losses) if test_losses else 0,
        "paliers": paliers,
        # Inférence
        "inference_time_ms": inference_time_ms,
        "inference_time_per_frame_ms": round(inference_time_ms / frames_per_call, 3),
        "flops_inference": flops_inference,
        "flops_per_frame": flops_inference // frames_per_call if frames_per_call > 0 else 0,
        "frames_per_call": frames_per_call,
    }

    # Simulation
    if success_rate is not None:
        run["success_rate"] = success_rate
        run["avg_coverage"] = avg_coverage
        run["avg_reward"] = avg_reward

    # Bras robotique
    if mae_per_joint_deg is not None:
        run["mae_per_joint_deg"] = mae_per_joint_deg
        run["mae_mean_deg"] = mae_mean_deg
        run["max_error_per_joint_deg"] = max_error_per_joint_deg

    # Extra
    if extra:
        run["extra"] = extra

    return run


def save_video(frames: list, filepath: str, fps: int = 10):
    """Sauvegarde une liste de frames (numpy arrays) en vidéo MP4."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    writer = imageio.get_writer(filepath, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        if isinstance(frame, np.ndarray):
            writer.append_data(frame)
    writer.close()
    print(f"  Vidéo : {filepath}")


def get_run_dir(experiment_name: str, label: str = "") -> Path:
    """Retourne le chemin du dossier d'un run."""
    safe_label = label.replace(" ", "_").replace("+", "_").replace("/", "_") if label else ""
    suffix = f"_{safe_label}" if safe_label else ""
    run_dir = RESULTS_DIR / "runs" / f"{experiment_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model(model, run_dir: Path, model_config: dict = None):
    """Sauvegarde les poids du modèle + sa config de construction.

    model_config : dict avec les arguments du constructeur, ex:
        {"class": "CNNRNNPolicy", "cnn_channels": [16, 32], ...}
    """
    # Poids
    weights_path = run_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"  Poids sauvegardés : {weights_path}")

    # Config pour recréer le modèle
    if model_config:
        config_path = run_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        print(f"  Config modèle    : {config_path}")


def save_checkpoint(model, optimizer, epoch, train_losses, test_losses,
                    epoch_times, run_dir: Path):
    """Sauvegarde un checkpoint pour reprendre l'entraînement plus tard."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "epoch_times": epoch_times,
    }
    path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: Path, model, optimizer=None):
    """Charge un checkpoint ou un model.pt.

    Si c'est un checkpoint complet : reprend epoch, losses, optimizer.
    Si c'est un model.pt (state_dict seul) : charge les poids, repart de 0 pour le reste.
    """
    data = torch.load(path, weights_only=False)

    if "model_state_dict" in data:
        # Checkpoint complet
        model.load_state_dict(data["model_state_dict"])
        if optimizer and "optimizer_state_dict" in data:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        epoch = data["epoch"]
        print(f"  Checkpoint complet chargé : epoch {epoch + 1}")
        return (
            epoch,
            data["train_losses"],
            data["test_losses"],
            data["epoch_times"],
        )
    else:
        # model.pt simple (state_dict)
        model.load_state_dict(data)
        print(f"  Poids chargés depuis model.pt (pas de checkpoint, optimizer réinitialisé)")
        return (0, [], [], [])


def generate_run_info(run_data: dict, model, run_dir: Path):
    """Génère le fichier run_info.md avec la description complète du run."""

    # Architecture détaillée
    arch_lines = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU, torch.nn.LSTM)):
            arch_lines.append(f"  {name}: {module}")

    arch_str = "\n".join(arch_lines) if arch_lines else str(model)

    # Résultats
    results_lines = []
    results_lines.append(f"Loss train finale : {run_data['final_train_loss']:.6f}")
    results_lines.append(f"Loss test finale  : {run_data['final_test_loss']:.6f}")
    results_lines.append(f"Meilleure test    : {run_data['best_test_loss']:.6f}")
    results_lines.append(f"Temps entraînement: {run_data['train_time_s']}s")
    results_lines.append(f"Temps/epoch       : {run_data['time_per_epoch_s']}s")
    results_lines.append(f"Inférence         : {run_data['inference_time_ms']}ms ({run_data['inference_time_per_frame_ms']}ms/frame)")
    results_lines.append(f"FLOPs/inférence   : {run_data['flops_inference']:,}")
    results_lines.append(f"FLOPs/frame       : {run_data['flops_per_frame']:,}")

    if "success_rate" in run_data:
        results_lines.append(f"Success rate      : {run_data['success_rate']:.0%}")
        results_lines.append(f"Coverage moyen    : {run_data.get('avg_coverage', 0):.1%}")
        results_lines.append(f"Reward moyen      : {run_data.get('avg_reward', 0):.2f}")

    if "mae_mean_deg" in run_data:
        results_lines.append(f"MAE moyenne       : {run_data['mae_mean_deg']:.2f}°")
        if run_data.get("mae_per_joint_deg"):
            for i, mae in enumerate(run_data["mae_per_joint_deg"]):
                results_lines.append(f"  Joint {i}         : {mae:.2f}°")

    # Paliers
    palier_lines = []
    for key, val in run_data.get("paliers", {}).items():
        palier_lines.append(f"  {key} : epoch {val['epoch']} ({val['time_s']}s)")

    content = f"""# {run_data['experiment']} — {run_data['architecture']}

## Expérience
{run_data.get('extra', {}).get('description', run_data['experiment'])}

## Modèle
Type : {run_data['architecture']}
Paramètres : {run_data['n_params']:,}
Taille : {run_data['size_mb']} Mo

### Architecture détaillée
```
{arch_str}
```

## Dataset
{run_data['dataset']} — {run_data['n_episodes']} épisodes
Samples train : {run_data['n_train_samples']} / test : {run_data['n_test_samples']}
Seed : {run_data['seed']}

## Résultats
{chr(10).join(results_lines)}

## Paliers de performance
{chr(10).join(palier_lines) if palier_lines else "Aucun palier atteint"}

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
"""

    filepath = run_dir / "run_info.md"
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Run info : {filepath}")


def save_eval_videos(all_frames: list, run_dir: Path, fps: int = 10):
    """Sauvegarde les vidéos des 3 premiers épisodes d'évaluation en MP4."""
    for ep_idx, frames in enumerate(all_frames[:3]):
        filepath = str(run_dir / f"ep{ep_idx}.mp4")
        save_video(frames, filepath, fps=fps)


def load_all_runs() -> list[dict]:
    """Charge tous les runs."""
    if not ALL_RUNS_FILE.exists():
        return []
    runs = []
    with open(ALL_RUNS_FILE) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    return runs


def compare_all(filter_experiment: str | None = None):
    """Affiche un tableau comparatif de tous les runs."""
    runs = load_all_runs()
    if filter_experiment:
        runs = [r for r in runs if r["experiment"] == filter_experiment]
    if not runs:
        print("Aucun run trouvé.")
        return

    print(f"\n{'='*100}")
    print(f"COMPARAISON — {len(runs)} runs")
    print(f"{'='*100}")

    # Header
    header = f"{'#':<3} {'Expérience':<20} {'Architecture':<25} {'Params':>8} {'Loss test':>10} {'Inf/frame':>10} {'FLOPs/frame':>12}"
    has_sim = any("success_rate" in r for r in runs)
    has_arm = any("mae_mean_deg" in r for r in runs)
    if has_sim:
        header += f" {'Success':>8} {'Coverage':>8}"
    if has_arm:
        header += f" {'MAE (°)':>8}"
    print(header)
    print("-" * len(header))

    for i, run in enumerate(runs):
        line = f"{i+1:<3} {run['experiment']:<20} {run['architecture']:<25} "
        line += f"{run['n_params']:>8,} {run['final_test_loss']:>10.6f} "
        line += f"{run['inference_time_per_frame_ms']:>9.2f}ms "
        line += f"{run['flops_per_frame']:>12,}"
        if has_sim:
            sr = run.get('success_rate', '')
            cov = run.get('avg_coverage', '')
            line += f" {sr if sr == '' else f'{sr:.0%}':>8} {cov if cov == '' else f'{cov:.1%}':>8}"
        if has_arm:
            mae = run.get('mae_mean_deg', '')
            line += f" {mae if mae == '' else f'{mae:.2f}':>8}"
        print(line)
