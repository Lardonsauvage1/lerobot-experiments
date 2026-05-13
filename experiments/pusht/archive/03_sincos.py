"""
Expérience 03 — Représentation sin/cos des angles.

Problème : le joint 3 va de -π à +π. Pour le réseau, -179° et +179° sont
très éloignés (écart de 358°) alors qu'en réalité c'est 2° de différence.

Solution : représenter chaque angle θ par (sin(θ), cos(θ)).
  - Avantage : pas de discontinuité, -179° et +179° sont proches
  - Inconvénient : 1 angle → 2 valeurs (double la dimension)

On teste sur la config 1 caméra côté + joints.

Runs :
  1. sin/cos sur le joint 3 uniquement (le problématique)
  2. sin/cos sur tous les joints (entrée ET sortie)
  (baseline = run 1 de l'exp 02, sans sin/cos)
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
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "03_sincos"
DATASET = "lerobot/utokyo_xarm_pick_and_place"

# CNN
CAMERAS = ["image"]
IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

# MLP
HIDDEN_LAYERS = [128, 64]
ACTIVATION = "relu"

# Entrées
USE_JOINTS = True
USE_JOINTS_IDX = [0, 1, 2, 3, 4, 5]

# sin/cos
SINCOS_INPUT = [3]      # Indices des joints d'ENTRÉE à convertir en sin/cos
SINCOS_OUTPUT = [3]     # Indices des joints de SORTIE à convertir en sin/cos

# Entraînement
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
NORMALIZE = True

PREDICT_JOINTS = [0, 1, 2, 3, 4, 5, 6]
N_EPISODES = None

# ============================================================

ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "pince"]


def get_activation(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]


def angles_to_sincos(tensor, indices):
    """Remplace les colonnes `indices` par sin/cos (double la dimension pour ces colonnes).

    Exemple : tensor (N, 6) avec indices=[3]
      → colonnes 0,1,2 restent, colonne 3 → sin(3),cos(3), colonnes 4,5 restent
      → résultat (N, 7)
    """
    parts = []
    for i in range(tensor.shape[1]):
        if i in indices:
            parts.append(torch.sin(tensor[:, i:i+1]))
            parts.append(torch.cos(tensor[:, i:i+1]))
        else:
            parts.append(tensor[:, i:i+1])
    return torch.cat(parts, dim=1)


def sincos_to_angles(tensor, indices, original_dim):
    """Inverse de angles_to_sincos : reconstruit les angles à partir de sin/cos.

    Utilise atan2(sin, cos) pour retrouver l'angle.
    """
    result = []
    src_col = 0
    for i in range(original_dim):
        if i in indices:
            sin_val = tensor[:, src_col:src_col+1]
            cos_val = tensor[:, src_col+1:src_col+2]
            angle = torch.atan2(sin_val, cos_val)
            result.append(angle)
            src_col += 2
        else:
            result.append(tensor[:, src_col:src_col+1])
            src_col += 1
    return torch.cat(result, dim=1)


class SimpleCNN(nn.Module):
    def __init__(self, channels, feature_dim, image_size):
        super().__init__()
        conv_layers = []
        in_ch = 3
        for out_ch in channels:
            conv_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        reduced = image_size // (2 ** len(channels))
        self.fc = nn.Linear(channels[-1] * reduced * reduced, feature_dim)

    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class CNNMLPPolicy(nn.Module):
    def __init__(self, n_cameras, cnn_channels, cnn_feature_dim, image_size,
                 joint_dim, output_dim, hidden_layers, activation, use_joints):
        super().__init__()
        self.use_joints = use_joints
        self.cnns = nn.ModuleList([
            SimpleCNN(cnn_channels, cnn_feature_dim, image_size) for _ in range(n_cameras)
        ])
        mlp_input = cnn_feature_dim * n_cameras + (joint_dim if use_joints else 0)
        layers = []
        prev = mlp_input
        act_fn = get_activation(activation)
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, images, joints=None):
        feats = torch.cat([cnn(img) for cnn, img in zip(self.cnns, images)], dim=1)
        if self.use_joints and joints is not None:
            feats = torch.cat([feats, joints], dim=1)
        return self.mlp(feats)


def load_data():
    print(f"Chargement du dataset {DATASET}...")
    episodes = list(range(N_EPISODES)) if N_EPISODES else None
    dataset = LeRobotDataset(DATASET, episodes=episodes)
    print(f"  Tâche    : {dataset[0]['task']}")
    print(f"  Frames   : {len(dataset)}")

    image_keys = [f"observation.images.{cam}" for cam in CAMERAS]
    all_images = [[] for _ in CAMERAS]
    joints = []
    actions = []

    for i in range(len(dataset)):
        sample = dataset[i]
        for c, key in enumerate(image_keys):
            img = F.interpolate(sample[key].unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)
            all_images[c].append(img)
        joints.append(sample["observation.state"][USE_JOINTS_IDX])
        actions.append(sample["action"][PREDICT_JOINTS])

    X_imgs = [torch.stack(imgs).float() for imgs in all_images]
    X_joints = torch.stack(joints).float()
    Y = torch.stack(actions).float()

    # Convertir en sin/cos
    sincos_in_local = [USE_JOINTS_IDX.index(i) for i in SINCOS_INPUT if i in USE_JOINTS_IDX]
    sincos_out_local = [PREDICT_JOINTS.index(i) for i in SINCOS_OUTPUT if i in PREDICT_JOINTS]

    print(f"  Joints avant sin/cos : {X_joints.shape}")
    X_joints = angles_to_sincos(X_joints, sincos_in_local)
    print(f"  Joints après sin/cos : {X_joints.shape} (indices {SINCOS_INPUT} convertis)")

    print(f"  Actions avant sin/cos: {Y.shape}")
    Y = angles_to_sincos(Y, sincos_out_local)
    print(f"  Actions après sin/cos: {Y.shape} (indices {SINCOS_OUTPUT} convertis)")

    # Normaliser
    norm_params = {}
    if NORMALIZE:
        j_mean, j_std = X_joints.mean(0), X_joints.std(0)
        y_mean, y_std = Y.mean(0), Y.std(0)
        j_std = torch.where(j_std > 1e-6, j_std, torch.ones_like(j_std))
        y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
        X_joints = (X_joints - j_mean) / j_std
        Y = (Y - y_mean) / y_std
        norm_params = {"j_mean": j_mean, "j_std": j_std, "y_mean": y_mean, "y_std": y_std}

    norm_params["sincos_out_local"] = sincos_out_local
    norm_params["original_output_dim"] = len(PREDICT_JOINTS)
    return X_imgs, X_joints, Y, norm_params


def train_model(model, X_imgs, X_joints, Y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_imgs[0])
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            pred = model([X[batch] for X in X_imgs], X_joints[batch] if USE_JOINTS else None)
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            test_pred = model([X[test_idx] for X in X_imgs], X_joints[test_idx] if USE_JOINTS else None)
            test_loss = criterion(test_pred, Y[test_idx]).item()

        train_losses.append(avg_train)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train: {avg_train:.6f} — test: {test_loss:.6f}")

    return train_losses, test_losses, test_idx


def evaluate(model, X_imgs, X_joints, Y, test_idx, norm_params):
    model.eval()
    with torch.no_grad():
        pred = model([X[test_idx] for X in X_imgs], X_joints[test_idx] if USE_JOINTS else None)

    # Dénormaliser
    if "y_std" in norm_params:
        pred = pred * norm_params["y_std"] + norm_params["y_mean"]
        Y_real = Y[test_idx] * norm_params["y_std"] + norm_params["y_mean"]
    else:
        Y_real = Y[test_idx]

    # Reconvertir sin/cos → angles
    sincos_out = norm_params["sincos_out_local"]
    orig_dim = norm_params["original_output_dim"]
    pred_angles = sincos_to_angles(pred, sincos_out, orig_dim)
    Y_angles = sincos_to_angles(Y_real, sincos_out, orig_dim)

    # Erreur angulaire (gère le wrap pour les angles)
    errors = torch.zeros_like(pred_angles)
    names = [ACTION_NAMES[j] for j in PREDICT_JOINTS]
    for i, name in enumerate(names):
        if PREDICT_JOINTS[i] in SINCOS_OUTPUT:
            # Erreur angulaire : plus court chemin sur le cercle
            diff = pred_angles[:, i] - Y_angles[:, i]
            errors[:, i] = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
        else:
            errors[:, i] = (pred_angles[:, i] - Y_angles[:, i]).abs()

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
    return mae_total


def plot_losses(train_losses, test_losses, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(test_losses, label="Test", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (normalisée)")
    ax1.set_title(f"Loss — {label}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    start = min(10, len(train_losses) // 3)
    ax2.plot(range(start, len(train_losses)), train_losses[start:], label="Train", linewidth=2)
    ax2.plot(range(start, len(test_losses)), test_losses[start:], label="Test", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"Zoom (epoch {start}+)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_label = label.replace(' ', '_').replace('/', '_').replace('[', '').replace(']', '')
    filepath = f"results/{EXPERIMENT_NAME}_loss_{safe_label}.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Graphe : {filepath}")


def main():
    sincos_label = f"sin/cos joints {SINCOS_INPUT}"
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"{'='*60}")
    print(f"  Caméras         : {CAMERAS}")
    print(f"  sin/cos entrée  : joints {SINCOS_INPUT}")
    print(f"  sin/cos sortie  : joints {SINCOS_OUTPUT}")
    print(f"  Epochs          : {EPOCHS}")
    print()

    X_imgs, X_joints, Y, norm_params = load_data()

    joint_dim = X_joints.shape[1]
    output_dim = Y.shape[1]

    model = CNNMLPPolicy(
        n_cameras=len(CAMERAS), cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, joint_dim=joint_dim, output_dim=output_dim,
        hidden_layers=HIDDEN_LAYERS, activation=ACTIVATION, use_joints=USE_JOINTS,
    )

    info = model_info(model, name=f"CNN+MLP sincos")
    print(f"\nArchitecture :")
    print(f"  Image → CNN → {CNN_FEATURE_DIM} features")
    print(f"  Joints ({joint_dim}D, avec sin/cos) + features → MLP {HIDDEN_LAYERS} → Sortie ({output_dim}D)")
    print(f"  Paramètres : {info['total_params']:,}")
    print()

    print("Entraînement...")
    train_losses, test_losses, test_idx = train_model(model, X_imgs, X_joints, Y)
    evaluate(model, X_imgs, X_joints, Y, test_idx, norm_params)
    plot_losses(train_losses, test_losses, sincos_label)

    config = {
        "dataset": DATASET, "cameras": str(CAMERAS),
        "sincos_input": str(SINCOS_INPUT), "sincos_output": str(SINCOS_OUTPUT),
        "hidden_layers": str(HIDDEN_LAYERS), "epochs": EPOCHS,
        "n_params": info["total_params"],
    }
    metrics = {
        "final_train_loss": train_losses[-1], "final_test_loss": test_losses[-1],
        "best_test_loss": min(test_losses), "train_losses": train_losses, "test_losses": test_losses,
    }
    save_run(EXPERIMENT_NAME, config, metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sincos-joints", nargs="+", type=int, default=None,
                        help="Indices des joints à convertir en sin/cos (entrée et sortie)")
    args = parser.parse_args()
    if args.sincos_joints is not None:
        SINCOS_INPUT = args.sincos_joints
        SINCOS_OUTPUT = args.sincos_joints
    main()
