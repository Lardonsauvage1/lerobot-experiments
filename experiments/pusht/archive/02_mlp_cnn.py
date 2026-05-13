"""
Expérience 02 — MLP + CNN pour prédire les actions avec les images.

Différence avec 01 : on ajoute un CNN qui traite l'image de la caméra.
Le CNN extrait des "features visuelles" (ex: position de l'assiette),
puis on les combine avec les positions articulaires pour prédire l'action.

Architecture :
    Image (3, 64, 64) → [Conv2d + ReLU + Pool] × N → Flatten → features image
    Joints (6,) ──────────────────────────────────────────────┐
                                                              ├→ MLP → Action (7)
    Features image ───────────────────────────────────────────┘

Dataset : lerobot/utokyo_xarm_pick_and_place
  Tâche : saisir une assiette blanche et la poser sur la rouge
  3 caméras disponibles : "image" (côté), "image2" (haut), "hand_image" (pince)

=== VARIANTES À TESTER ===
1. CAMERA          — quelle caméra utiliser ("image", "image2", "hand_image")
2. CNN_CHANNELS    — nombre de filtres par couche conv ([16, 32] par défaut)
3. CNN_FEATURE_DIM — taille du vecteur de features image en sortie du CNN
4. HIDDEN_LAYERS   — couches du MLP après concaténation
5. IMAGE_SIZE      — taille de l'image redimensionnée (64 par défaut)
6. USE_JOINTS      — utiliser aussi les joints ? (True/False)
7. FREEZE_CNN      — geler le CNN après un certain nombre d'epochs
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.tracker import save_run
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES À MODIFIER
# ============================================================

EXPERIMENT_NAME = "02_mlp_cnn"
DATASET = "lerobot/utokyo_xarm_pick_and_place"

# CNN
CAMERAS = ["image"]             # Liste : ["image"], ["image2"], ["hand_image"], ou les 3
IMAGE_SIZE = 64                # Redimensionner l'image à cette taille
CNN_CHANNELS = [16, 32]        # Filtres par couche conv
CNN_FEATURE_DIM = 64           # Taille du vecteur features image

# MLP (après concaténation features image + joints)
HIDDEN_LAYERS = [128, 64]
ACTIVATION = "relu"

# Entrées
USE_JOINTS = True              # Utiliser aussi les positions articulaires
USE_JOINTS_IDX = [0, 1, 2, 3, 4, 5]  # Quels joints

# Entraînement
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64                # Plus petit car les images prennent de la mémoire
NORMALIZE = True

PREDICT_JOINTS = [0, 1, 2, 3, 4, 5, 6]
N_EPISODES = None

# ============================================================

ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "pince"]


def get_activation(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]


class SimpleCNN(nn.Module):
    """CNN simple pour extraire des features d'une image.

    Comment ça marche :
    - Conv2d : fait glisser un petit filtre (3×3) sur l'image pour détecter des motifs
      (bords, couleurs, formes). Chaque filtre produit une "feature map".
    - ReLU : garde les valeurs positives, met les négatives à 0
    - MaxPool2d : réduit la taille en gardant la valeur max dans chaque zone 2×2
    - À la fin, on aplatit tout et on passe dans un Linear pour obtenir un vecteur fixe

    Exemple avec IMAGE_SIZE=64 et CNN_CHANNELS=[16, 32] :
        (3, 64, 64)  → Conv(3→16) + ReLU + Pool  → (16, 32, 32)
                     → Conv(16→32) + ReLU + Pool → (32, 16, 16)
                     → Flatten → (32×16×16 = 8192)
                     → Linear(8192 → 64) → features image (64,)
    """

    def __init__(self, channels: list[int], feature_dim: int, image_size: int):
        super().__init__()

        conv_layers = []
        in_channels = 3  # RGB
        for out_channels in channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculer la taille après les convolutions
        reduced_size = image_size // (2 ** len(channels))
        flat_dim = channels[-1] * reduced_size * reduced_size

        self.fc = nn.Linear(flat_dim, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)  # (batch, channels * h * w)
        x = F.relu(self.fc(x))
        return x


class CNNMLPPolicy(nn.Module):
    """Politique qui combine vision (CNN) et proprioception (joints).

    Avec 1 caméra :
        image → CNN → features (64D) ─────────┐
        joints (6D) ──────────────────────────┤ concat → MLP → action (7D)

    Avec 3 caméras :
        image1 → CNN_1 → features (64D) ─────┐
        image2 → CNN_2 → features (64D) ─────┤
        image3 → CNN_3 → features (64D) ─────┤ concat → MLP → action (7D)
        joints (6D) ─────────────────────────┘
    """

    def __init__(self, n_cameras, cnn_channels, cnn_feature_dim, image_size,
                 joint_dim, output_dim, hidden_layers, activation, use_joints):
        super().__init__()

        self.use_joints = use_joints
        self.n_cameras = n_cameras

        # Un CNN par caméra
        self.cnns = nn.ModuleList([
            SimpleCNN(cnn_channels, cnn_feature_dim, image_size)
            for _ in range(n_cameras)
        ])

        # Dimension d'entrée du MLP = features de chaque caméra + joints
        mlp_input_dim = cnn_feature_dim * n_cameras + (joint_dim if use_joints else 0)

        layers = []
        prev_dim = mlp_input_dim
        act_fn = get_activation(activation)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, images, joints=None):
        # images est une liste de tenseurs, un par caméra
        features = [cnn(img) for cnn, img in zip(self.cnns, images)]
        x = torch.cat(features, dim=1)

        if self.use_joints and joints is not None:
            x = torch.cat([x, joints], dim=1)

        return self.mlp(x)


def load_data():
    """Charge le dataset avec images et joints."""
    print(f"Chargement du dataset {DATASET}...")
    episodes = list(range(N_EPISODES)) if N_EPISODES else None
    dataset = LeRobotDataset(DATASET, episodes=episodes)

    print(f"  Tâche    : {dataset[0]['task']}")
    print(f"  Frames   : {len(dataset)}")
    print(f"  Épisodes : {dataset.num_episodes}")
    print(f"  Caméras  : {CAMERAS}")

    image_keys = [f"observation.images.{cam}" for cam in CAMERAS]

    # Une liste par caméra
    all_images = [[] for _ in CAMERAS]
    joints = []
    actions = []

    for i in range(len(dataset)):
        sample = dataset[i]
        for c, key in enumerate(image_keys):
            img = sample[key]
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)
            all_images[c].append(img)

        joints.append(sample["observation.state"][USE_JOINTS_IDX])
        actions.append(sample["action"][PREDICT_JOINTS])

    X_imgs = [torch.stack(imgs).float() for imgs in all_images]
    X_joints = torch.stack(joints).float()
    Y = torch.stack(actions).float()

    for c, cam in enumerate(CAMERAS):
        print(f"  {cam:>15} : {X_imgs[c].shape}")
    print(f"  Joints         : {X_joints.shape}")
    print(f"  Actions        : {Y.shape}")

    # Normaliser joints et actions (pas les images, déjà en 0-1)
    norm_params = {}
    if NORMALIZE:
        j_mean, j_std = X_joints.mean(0), X_joints.std(0)
        y_mean, y_std = Y.mean(0), Y.std(0)
        j_std = torch.where(j_std > 1e-6, j_std, torch.ones_like(j_std))
        y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
        X_joints = (X_joints - j_mean) / j_std
        Y = (Y - y_mean) / y_std
        norm_params = {"j_mean": j_mean, "j_std": j_std, "y_mean": y_mean, "y_std": y_std}
        print("  Joints et actions normalisés")

    return X_imgs, X_joints, Y, norm_params


def train(model, X_imgs, X_joints, Y):
    """Entraîne le modèle."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_imgs[0])
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            img_batch = [X[batch] for X in X_imgs]
            joint_batch = X_joints[batch] if USE_JOINTS else None
            y_batch = Y[batch]

            pred = model(img_batch, joint_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            test_imgs = [X[test_idx] for X in X_imgs]
            test_pred = model(test_imgs, X_joints[test_idx] if USE_JOINTS else None)
            test_loss = criterion(test_pred, Y[test_idx]).item()

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train_loss:.6f} — "
                  f"test: {test_loss:.6f}")

    return train_losses, test_losses, test_idx


def evaluate(model, X_imgs, X_joints, Y, test_idx, norm_params):
    """Évalue en unités réelles."""
    model.eval()
    with torch.no_grad():
        test_imgs = [X[test_idx] for X in X_imgs]
        pred = model(test_imgs, X_joints[test_idx] if USE_JOINTS else None)

    if norm_params:
        y_std = norm_params["y_std"]
        y_mean = norm_params["y_mean"]
        pred = pred * y_std + y_mean
        Y_real = Y[test_idx] * y_std + y_mean
    else:
        Y_real = Y[test_idx]

    errors = (pred - Y_real).abs()
    names = [ACTION_NAMES[j] for j in PREDICT_JOINTS]

    print(f"\n{'='*60}")
    print("ERREUR PAR JOINT (sur le test set)")
    print(f"{'='*60}")
    print(f"  {'Joint':<12} {'MAE (rad)':>10} {'MAE (deg)':>10} {'Max (deg)':>10}")
    print(f"  {'-'*44}")
    for i, name in enumerate(names):
        mae_rad = errors[:, i].mean().item()
        mae_deg = mae_rad * 180 / 3.14159
        max_deg = errors[:, i].max().item() * 180 / 3.14159
        if name == "pince":
            print(f"  {name:<12} {mae_rad:>10.4f} {'—':>10} {'—':>10}   (0=ouvert, 1=fermé)")
        else:
            print(f"  {name:<12} {mae_rad:>10.4f} {mae_deg:>9.2f}° {max_deg:>9.2f}°")

    mae_total = errors.mean().item()
    print(f"\n  MAE moyenne : {mae_total:.4f} rad ({mae_total * 180 / 3.14159:.2f}°)")


def plot_losses(train_losses, test_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(test_losses, label="Test", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (normalisée)")
    ax1.set_title(f"Loss — CNN+MLP (caméras: {CAMERAS})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    start = min(10, len(train_losses) // 3)
    ax2.plot(range(start, len(train_losses)), train_losses[start:], label="Train", linewidth=2)
    ax2.plot(range(start, len(test_losses)), test_losses[start:], label="Test", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss (normalisée)")
    ax2.set_title(f"Zoom (epoch {start}+)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = f"results/{EXPERIMENT_NAME}_loss.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Graphe sauvegardé : {filepath}")


def main():
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"{'='*60}")
    print(f"  Dataset         : {DATASET}")
    print(f"  Caméras         : {CAMERAS}")
    print(f"  Image size      : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  CNN channels    : {CNN_CHANNELS}")
    print(f"  CNN feature dim : {CNN_FEATURE_DIM}")
    print(f"  MLP layers      : {HIDDEN_LAYERS}")
    print(f"  Use joints      : {USE_JOINTS}")
    print(f"  Epochs          : {EPOCHS}")
    print()

    # 1. Charger
    X_imgs, X_joints, Y, norm_params = load_data()

    # 2. Créer le modèle
    joint_dim = len(USE_JOINTS_IDX) if USE_JOINTS else 0
    output_dim = len(PREDICT_JOINTS)

    model = CNNMLPPolicy(
        n_cameras=len(CAMERAS),
        cnn_channels=CNN_CHANNELS,
        cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE,
        joint_dim=joint_dim,
        output_dim=output_dim,
        hidden_layers=HIDDEN_LAYERS,
        activation=ACTIVATION,
        use_joints=USE_JOINTS,
    )

    info = model_info(model, name=f"CNN+MLP ({'+'.join(CAMERAS)})")
    print(f"\nArchitecture :")
    for cam in CAMERAS:
        print(f"  {cam} ({IMAGE_SIZE}×{IMAGE_SIZE}) → CNN {CNN_CHANNELS} → {CNN_FEATURE_DIM} features")
    total_feat = CNN_FEATURE_DIM * len(CAMERAS)
    if USE_JOINTS:
        print(f"  Joints (6D) + {len(CAMERAS)} × {CNN_FEATURE_DIM} features = {total_feat + 6}D → MLP {HIDDEN_LAYERS} → Action ({output_dim}D)")
    else:
        print(f"  {len(CAMERAS)} × {CNN_FEATURE_DIM} features = {total_feat}D → MLP {HIDDEN_LAYERS} → Action ({output_dim}D)")
    print(f"  Paramètres : {info['total_params']:,}")
    print(f"  Taille     : {info['size_mb']:.2f} Mo")
    print()

    # 3. Entraîner
    print("Entraînement...")
    train_losses, test_losses, test_idx = train(model, X_imgs, X_joints, Y)

    # 4. Évaluer
    evaluate(model, X_imgs, X_joints, Y, test_idx, norm_params)

    # 5. Graphe
    plot_losses(train_losses, test_losses)

    # 6. Sauvegarder
    config = {
        "dataset": DATASET,
        "cameras": str(CAMERAS),
        "image_size": IMAGE_SIZE,
        "cnn_channels": str(CNN_CHANNELS),
        "cnn_feature_dim": CNN_FEATURE_DIM,
        "hidden_layers": str(HIDDEN_LAYERS),
        "activation": ACTIVATION,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "use_joints": USE_JOINTS,
        "n_params": info["total_params"],
        "size_mb": info["size_mb"],
    }
    metrics = {
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "best_test_loss": min(test_losses),
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    save_run(EXPERIMENT_NAME, config, metrics)

    print(f"\nPour comparer : compare_runs('{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Caméras à utiliser (image, image2, hand_image)")
    parser.add_argument("--no-joints", action="store_true",
                        help="Ne pas utiliser les positions articulaires")
    args = parser.parse_args()

    if args.cameras:
        CAMERAS = args.cameras
    if args.no_joints:
        USE_JOINTS = False

    main()
