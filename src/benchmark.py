"""
Benchmark — charger des modèles pré-entraînés LeRobot et extraire
leurs caractéristiques pour les comparer avec nos expériences.
"""

import time
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def load_pretrained(repo_id: str, device: str = "cpu"):
    """Charge un modèle pré-entraîné depuis le Hub Hugging Face.

    Exemples de repo_id:
        - "lerobot/diffusion_pusht"
        - "lerobot/act_aloha_sim_insertion_human"
        - "lerobot/act_aloha_sim_transfer_cube_human"
    """
    config = PreTrainedConfig.from_pretrained(repo_id)
    config.device = device
    PolicyClass = get_policy_class(config.type)
    policy = PolicyClass.from_pretrained(repo_id, config=config)
    policy.eval()
    return policy, config


def model_info(policy, name: str = "") -> dict:
    """Extrait les infos clés d'un modèle (taille, paramètres, etc.)."""
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in policy.parameters()) / (1024 * 1024)

    # Compter les couches
    n_layers = len([n for n, _ in policy.named_modules() if "linear" in n.lower()
                    or "conv" in n.lower()])

    info = {
        "name": name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2),
        "n_linear_conv_layers": n_layers,
    }
    return info


def print_model_info(info: dict):
    """Affiche les infos d'un modèle de façon lisible."""
    print(f"\n{'='*50}")
    print(f"Modèle : {info['name']}")
    print(f"{'='*50}")
    print(f"  Paramètres totaux     : {info['total_params']:>12,}")
    print(f"  Paramètres entraînables: {info['trainable_params']:>12,}")
    print(f"  Taille (Mo)           : {info['size_mb']:>12.2f}")
    print(f"  Couches linear/conv   : {info['n_linear_conv_layers']:>12}")


def compare_with_reference(experiment_info: dict, reference_infos: list[dict]):
    """Compare un modèle expérimental avec des modèles de référence."""
    print(f"\n{'='*70}")
    print("COMPARAISON : ton modèle vs références")
    print(f"{'='*70}")

    header = f"{'Modèle':<30} {'Params':>12} {'Taille (Mo)':>12} {'Couches':>10}"
    print(header)
    print("-" * 70)

    # Ton modèle en premier
    exp = experiment_info
    print(f"{'>> ' + exp['name']:<30} {exp['total_params']:>12,} {exp['size_mb']:>12.2f} {exp['n_linear_conv_layers']:>10}")

    # Références
    for ref in reference_infos:
        print(f"{'   ' + ref['name']:<30} {ref['total_params']:>12,} {ref['size_mb']:>12.2f} {ref['n_linear_conv_layers']:>10}")

    # Ratio
    if reference_infos:
        print(f"\n--- Ratios (ton modèle / référence) ---")
        for ref in reference_infos:
            if ref['total_params'] > 0:
                ratio = exp['total_params'] / ref['total_params']
                print(f"  vs {ref['name']}: {ratio:.4f}x paramètres ({ratio*100:.1f}%)")


def measure_inference_speed(policy, sample_input: dict, n_runs: int = 100) -> float:
    """Mesure le temps d'inférence moyen en ms."""
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            policy.select_action(sample_input)

    # Mesure
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            policy.select_action(sample_input)
    elapsed = (time.time() - start) / n_runs * 1000  # en ms

    return round(elapsed, 2)
