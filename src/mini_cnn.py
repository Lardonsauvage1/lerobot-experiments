"""Mini-CNN vision pour Diffusion Policy — remplace ResNet18 (11.2 M) par ~0.03 M.

LeRobot fige le backbone vision sur un modèle torchvision (ResNet). Ici on swappe juste
`rgb_encoder.backbone` par un petit CNN (3 convs, GroupNorm) et on recalcule le spatial-softmax.
Le `feature_dim` reste 64 (= 2 × spatial_softmax_num_keypoints) → le U-Net est inchangé.

⚠️ Le checkpoint sauvé contient les poids mini-CNN, mais la config dit toujours `resnet18` →
`from_pretrained` reconstruirait un ResNet et planterait au load. Utiliser `load_minicnn_policy`.
"""

import torch
import torch.nn as nn


def build_tiny_backbone() -> nn.Sequential:
    """3→16→32→64, stride 2, GroupNorm(8). ~24k params. Sortie (B,64,12,12) pour une entrée 96×96."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 16), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
    )


def swap_to_tiny_cnn(policy) -> None:
    """Remplace in-place le backbone ResNet18 du rgb_encoder par le mini-CNN + recalcule le pool."""
    from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax, get_output_shape

    enc = policy.diffusion.rgb_encoder
    device = next(policy.parameters()).device
    # construire + dry-run sur CPU (get_output_shape crée un dummy CPU), PUIS déplacer sur device
    enc.backbone = build_tiny_backbone()
    img_shape = next(iter(policy.config.image_features.values())).shape  # (C, H, W)
    fmap = get_output_shape(enc.backbone, (1, *img_shape))[1:]
    enc.pool = SpatialSoftmax(fmap, num_kp=policy.config.spatial_softmax_num_keypoints)
    enc.backbone.to(device)
    enc.pool.to(device)
    # enc.out (Linear 64→64) et enc.feature_dim (64) restent valides


# Checkpoint ResNet [32,64,128] (config identique au run mini-CNN) servant de gabarit de construction.
CONFIG_TEMPLATE = "results/runs/lift/51_unet_d32_64_128/checkpoints/007500/pretrained_model"


def load_minicnn_policy(ckpt: str, device, config_template: str = CONFIG_TEMPLATE):
    """Charge une policy mini-CNN de façon robuste :
    build depuis un checkpoint ResNet de MÊME config (from_pretrained marche) → swap mini-CNN
    → load_state_dict des poids mini-CNN par-dessus. Évite DiffusionConfig.from_pretrained (bug draccus)."""
    from pathlib import Path
    from safetensors.torch import load_file
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(config_template).to(device).eval()  # build (config identique)
    swap_to_tiny_cnn(policy)  # backbone ResNet -> mini-CNN (poids aléatoires)
    sd = load_file(str(Path(ckpt) / "model.safetensors"), device=str(device))
    missing, unexpected = policy.load_state_dict(sd, strict=False)  # charge les vrais poids mini-CNN
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing:
        print(f"  [load_minicnn] ⚠️ poids manquants ({len(real_missing)}): {real_missing[:5]}")
    if unexpected:
        print(f"  [load_minicnn] ⚠️ poids inattendus ({len(unexpected)}): {unexpected[:5]}")
    return policy


def load_minicnn_from_ckpt(ckpt, device):
    """Charge un mini-CNN depuis la config PROPRE du checkpoint (n'importe quelle archi/dims).

    Plus robuste que load_minicnn_policy (pas de template figé) : `PreTrainedConfig.from_pretrained`
    dispatche via le champ `type` (contourne le bug draccus de `DiffusionConfig.from_pretrained`),
    on build une policy random depuis cette config → swap mini-CNN → load_state_dict(strict=False).
    Marche pour Lift (19D) comme Can (12D), tout down_dims. Le normalizer vient du state_dict.
    """
    from pathlib import Path
    from safetensors.torch import load_file
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.policies import PreTrainedConfig

    import os as _os
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = DiffusionPolicy(cfg)
    if _os.environ.get("NO_SWAP") == "1":   # checkpoint ResNet18 natif (temoin capacite) : pas de swap
        print("  [load_minicnn_ckpt] NO_SWAP=1 -> ResNet18 natif conserve")
    else:
        swap_to_tiny_cnn(policy)
    sd = load_file(str(Path(ckpt) / "model.safetensors"), device="cpu")
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing:
        print(f"  [load_minicnn_ckpt] ⚠️ poids manquants ({len(real_missing)}): {real_missing[:5]}")
    if unexpected:
        print(f"  [load_minicnn_ckpt] ⚠️ poids inattendus ({len(unexpected)}): {unexpected[:5]}")
    return policy.to(device).eval()
