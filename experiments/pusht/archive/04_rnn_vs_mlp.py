"""
Expérience 04 — CNN+RNN vs CNN+MLP à taille égale.

Le MLP voit 1 frame isolée et prédit l'action.
Le RNN reçoit une séquence de frames et accumule un état interne (mémoire).

Architecture MLP (baseline) :
    Image → CNN → features (64D)
    features + joints (7D, sin/cos) → MLP → action (8D, sin/cos)

Architecture RNN :
    Pour chaque frame t de la séquence :
        Image_t → CNN → features_t (64D)
        features_t + joints_t → RNN (état caché mis à jour)
    État caché final → Linear → action (8D, sin/cos)

Runs :
  1. CNN + MLP (baseline, 1 frame)
  2. CNN + RNN, séquence de 5 frames
  3. CNN + RNN, séquence de 10 frames
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.tracker import save_run
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "04_rnn_vs_mlp"
DATASET = "lerobot/utokyo_xarm_pick_and_place"

CAMERAS = ["image"]
IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

USE_JOINTS_IDX = [0, 1, 2, 3, 4, 5]
SINCOS_JOINTS = [3]
PREDICT_JOINTS = [0, 1, 2, 3, 4, 5, 6]

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
NORMALIZE = True

# Spécifique à ce script — modifié via CLI
MODEL_TYPE = "mlp"     # "mlp" ou "rnn"
SEQ_LEN = 1            # 1 pour MLP, 5 ou 10 pour RNN

# ============================================================

ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "pince"]


# === Utilitaires sin/cos ===

def angles_to_sincos(tensor, indices):
    parts = []
    for i in range(tensor.shape[-1]):
        if i in indices:
            parts.append(torch.sin(tensor[..., i:i+1]))
            parts.append(torch.cos(tensor[..., i:i+1]))
        else:
            parts.append(tensor[..., i:i+1])
    return torch.cat(parts, dim=-1)


def sincos_to_angles(tensor, indices, original_dim):
    result = []
    src = 0
    for i in range(original_dim):
        if i in indices:
            angle = torch.atan2(tensor[..., src:src+1], tensor[..., src+1:src+2])
            result.append(angle)
            src += 2
        else:
            result.append(tensor[..., src:src+1])
            src += 1
    return torch.cat(result, dim=-1)


# === Modèles ===

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
    """CNN + MLP (baseline, 1 frame)."""

    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 joint_dim, output_dim, hidden_size):
        super().__init__()
        self.cnn = SimpleCNN(cnn_channels, cnn_feature_dim, image_size)
        input_dim = cnn_feature_dim + joint_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, images, joints):
        # images: (batch, 3, H, W), joints: (batch, joint_dim)
        feats = self.cnn(images)
        x = torch.cat([feats, joints], dim=1)
        return self.mlp(x)


class CNNRNNPolicy(nn.Module):
    """CNN + RNN (avec mémoire sur la séquence).

    Pour chaque frame de la séquence :
      1. Le CNN extrait des features de l'image
      2. On concatène avec les joints
      3. Le RNN met à jour son état caché

    À la fin de la séquence, l'état caché contient un résumé
    de tout ce qui s'est passé → on prédit l'action.
    """

    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 joint_dim, output_dim, hidden_size, n_rnn_layers=1):
        super().__init__()
        self.cnn = SimpleCNN(cnn_channels, cnn_feature_dim, image_size)
        input_dim = cnn_feature_dim + joint_dim
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, images_seq, joints_seq):
        # images_seq: (batch, seq_len, 3, H, W)
        # joints_seq: (batch, seq_len, joint_dim)
        batch, seq_len = images_seq.shape[:2]

        # Traiter toutes les images d'un coup
        imgs_flat = images_seq.reshape(batch * seq_len, *images_seq.shape[2:])
        feats_flat = self.cnn(imgs_flat)
        feats = feats_flat.reshape(batch, seq_len, -1)

        # Concaténer features + joints pour chaque timestep
        x = torch.cat([feats, joints_seq], dim=2)  # (batch, seq_len, input_dim)

        # Passer dans le RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)

        # Prendre la sortie du dernier timestep
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_size)
        return self.fc_out(last_hidden)


# === Data ===

def load_data_single():
    """Charge les données frame par frame (pour MLP)."""
    dataset = LeRobotDataset(DATASET)
    print(f"  Tâche    : {dataset[0]['task']}")
    print(f"  Frames   : {len(dataset)}")

    images, joints, actions = [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        img = F.interpolate(s[f"observation.images.{CAMERAS[0]}"].unsqueeze(0),
                            size=IMAGE_SIZE, mode="bilinear").squeeze(0)
        images.append(img)
        joints.append(s["observation.state"][USE_JOINTS_IDX])
        actions.append(s["action"][PREDICT_JOINTS])

    X_img = torch.stack(images).float()
    X_j = torch.stack(joints).float()
    Y = torch.stack(actions).float()

    # sin/cos
    sincos_j = [USE_JOINTS_IDX.index(i) for i in SINCOS_JOINTS if i in USE_JOINTS_IDX]
    sincos_a = [PREDICT_JOINTS.index(i) for i in SINCOS_JOINTS if i in PREDICT_JOINTS]
    X_j = angles_to_sincos(X_j, sincos_j)
    Y = angles_to_sincos(Y, sincos_a)

    # Normaliser
    j_mean, j_std = X_j.mean(0), X_j.std(0)
    y_mean, y_std = Y.mean(0), Y.std(0)
    j_std = torch.where(j_std > 1e-6, j_std, torch.ones_like(j_std))
    y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
    X_j = (X_j - j_mean) / j_std
    Y = (Y - y_mean) / y_std

    norm = {"y_mean": y_mean, "y_std": y_std,
            "sincos_out": sincos_a, "orig_dim": len(PREDICT_JOINTS)}
    return X_img, X_j, Y, norm


def load_data_sequences(seq_len):
    """Charge les données en séquences (pour RNN).

    On découpe chaque épisode en séquences de `seq_len` frames consécutives.
    La cible = l'action de la dernière frame de la séquence.
    """
    dataset = LeRobotDataset(DATASET)
    print(f"  Tâche    : {dataset[0]['task']}")
    print(f"  Frames   : {len(dataset)}, séquences de {seq_len}")

    # D'abord charger tout
    all_imgs, all_joints, all_actions, all_eps = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        img = F.interpolate(s[f"observation.images.{CAMERAS[0]}"].unsqueeze(0),
                            size=IMAGE_SIZE, mode="bilinear").squeeze(0)
        all_imgs.append(img)
        all_joints.append(s["observation.state"][USE_JOINTS_IDX])
        all_actions.append(s["action"][PREDICT_JOINTS])
        all_eps.append(s["episode_index"].item())

    all_imgs = torch.stack(all_imgs).float()
    all_joints = torch.stack(all_joints).float()
    all_actions = torch.stack(all_actions).float()

    # sin/cos
    sincos_j = [USE_JOINTS_IDX.index(i) for i in SINCOS_JOINTS if i in USE_JOINTS_IDX]
    sincos_a = [PREDICT_JOINTS.index(i) for i in SINCOS_JOINTS if i in PREDICT_JOINTS]
    all_joints = angles_to_sincos(all_joints, sincos_j)
    all_actions = angles_to_sincos(all_actions, sincos_a)

    # Normaliser
    j_mean, j_std = all_joints.mean(0), all_joints.std(0)
    y_mean, y_std = all_actions.mean(0), all_actions.std(0)
    j_std = torch.where(j_std > 1e-6, j_std, torch.ones_like(j_std))
    y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
    all_joints = (all_joints - j_mean) / j_std
    all_actions = (all_actions - y_mean) / y_std

    # Découper en séquences (respecter les épisodes)
    seq_imgs, seq_joints, seq_actions = [], [], []
    eps_array = all_eps

    i = 0
    while i + seq_len <= len(all_imgs):
        # Vérifier que toute la séquence est dans le même épisode
        if eps_array[i] == eps_array[i + seq_len - 1]:
            seq_imgs.append(all_imgs[i:i + seq_len])
            seq_joints.append(all_joints[i:i + seq_len])
            seq_actions.append(all_actions[i + seq_len - 1])  # action de la dernière frame
            i += 1
        else:
            i += 1
            continue

    X_img = torch.stack(seq_imgs)    # (N, seq_len, 3, H, W)
    X_j = torch.stack(seq_joints)    # (N, seq_len, joint_dim)
    Y = torch.stack(seq_actions)     # (N, output_dim)

    print(f"  Séquences valides : {len(X_img)}")

    norm = {"y_mean": y_mean, "y_std": y_std,
            "sincos_out": sincos_a, "orig_dim": len(PREDICT_JOINTS)}
    return X_img, X_j, Y, norm


# === Train & Eval ===

def train_model(model, X_img, X_j, Y, is_rnn):
    import time
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_img)
    n_train_frames = n  # nombre de frames/séquences utilisées
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    epoch_times = []
    start_total = time.time()

    for epoch in range(EPOCHS):
        start_epoch = time.time()
        model.train()
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            pred = model(X_img[batch], X_j[batch])
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            test_pred = model(X_img[test_idx], X_j[test_idx])
            test_loss = criterion(test_pred, Y[test_idx]).item()

        elapsed_epoch = time.time() - start_epoch
        elapsed_total = time.time() - start_total

        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(elapsed_epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — "
                  f"{elapsed_epoch:.1f}s/epoch — total: {elapsed_total:.0f}s")

    total_time = time.time() - start_total
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    timing = {
        "total_time_s": round(total_time, 1),
        "avg_epoch_time_s": round(avg_epoch_time, 2),
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "epoch_times": [round(t, 2) for t in epoch_times],
    }

    print(f"\n  Temps total      : {total_time:.1f}s")
    print(f"  Temps/epoch      : {avg_epoch_time:.2f}s")
    print(f"  Samples train    : {len(train_idx)}")
    print(f"  Samples test     : {len(test_idx)}")

    return train_losses, test_losses, test_idx, timing


def evaluate(model, X_img, X_j, Y, test_idx, norm):
    model.eval()
    with torch.no_grad():
        pred = model(X_img[test_idx], X_j[test_idx])

    pred = pred * norm["y_std"] + norm["y_mean"]
    Y_real = Y[test_idx] * norm["y_std"] + norm["y_mean"]

    pred_a = sincos_to_angles(pred, norm["sincos_out"], norm["orig_dim"])
    Y_a = sincos_to_angles(Y_real, norm["sincos_out"], norm["orig_dim"])

    errors = torch.zeros_like(pred_a)
    for i in range(pred_a.shape[1]):
        if PREDICT_JOINTS[i] in SINCOS_JOINTS:
            diff = pred_a[:, i] - Y_a[:, i]
            errors[:, i] = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
        else:
            errors[:, i] = (pred_a[:, i] - Y_a[:, i]).abs()

    print(f"\n{'='*60}")
    print("ERREUR PAR JOINT")
    print(f"{'='*60}")
    print(f"  {'Joint':<12} {'MAE (rad)':>10} {'MAE (deg)':>10} {'Max (deg)':>10}")
    print(f"  {'-'*44}")
    for i, j in enumerate(PREDICT_JOINTS):
        name = ACTION_NAMES[j]
        mae_r = errors[:, i].mean().item()
        mae_d = mae_r * 180 / 3.14159
        max_d = errors[:, i].max().item() * 180 / 3.14159
        if name == "pince":
            print(f"  {name:<12} {mae_r:>10.4f} {'—':>10} {'—':>10}   (0=ouvert, 1=fermé)")
        else:
            print(f"  {name:<12} {mae_r:>10.4f} {mae_d:>9.2f}° {max_d:>9.2f}°")

    mae = errors.mean().item()
    print(f"\n  MAE moyenne : {mae:.4f} rad ({mae * 180 / 3.14159:.2f}°)")
    return mae


def plot_losses(train_losses, test_losses, timing, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs epoch
    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(test_losses, label="Test", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title(f"Loss vs Epoch — {label}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss vs temps
    cumul_times = [sum(timing["epoch_times"][:i+1]) for i in range(len(timing["epoch_times"]))]
    ax2.plot(cumul_times, train_losses, label="Train", linewidth=2)
    ax2.plot(cumul_times, test_losses, label="Test", linewidth=2)
    ax2.set_xlabel("Temps (secondes)")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"Loss vs Temps — {label}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = label.replace(" ", "_").replace("+", "_").replace("=", "")
    filepath = f"results/{EXPERIMENT_NAME}_{safe}.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Graphe : {filepath}")


def main():
    is_rnn = MODEL_TYPE == "rnn"
    label = f"{'RNN' if is_rnn else 'MLP'} seq={SEQ_LEN}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")
    print(f"  Modèle : {MODEL_TYPE.upper()}")
    print(f"  Séquence : {SEQ_LEN} frames")
    print()

    # Charger les données
    if is_rnn:
        X_img, X_j, Y, norm = load_data_sequences(SEQ_LEN)
    else:
        X_img, X_j, Y, norm = load_data_single()

    joint_dim = X_j.shape[-1]
    output_dim = Y.shape[-1]

    # Créer le modèle avec ~même nombre de paramètres
    if is_rnn:
        # GRU hidden_size ajusté pour ~547k params
        # CNN ≈ 538k, il reste ~9k pour le RNN
        # GRU(71 input, h hidden): 3*(71*h + h*h + 2*h) + h*8 params
        # On veut ~9k → h≈32 donne ~10k, OK
        model = CNNRNNPolicy(
            cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
            image_size=IMAGE_SIZE, joint_dim=joint_dim, output_dim=output_dim,
            hidden_size=32, n_rnn_layers=1,
        )
    else:
        model = CNNMLPPolicy(
            cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
            image_size=IMAGE_SIZE, joint_dim=joint_dim, output_dim=output_dim,
            hidden_size=128,
        )

    info = model_info(model, name=label)
    print(f"\n  Architecture : {MODEL_TYPE.upper()}")
    print(f"  Paramètres   : {info['total_params']:,}")
    print(f"  Taille       : {info['size_mb']:.2f} Mo")
    print()

    # Entraîner
    print("Entraînement...")
    train_losses, test_losses, test_idx, timing = train_model(model, X_img, X_j, Y, is_rnn)

    # Évaluer
    mae = evaluate(model, X_img, X_j, Y, test_idx, norm)

    # Graphe
    plot_losses(train_losses, test_losses, timing, label)

    # Paliers de performance : à quelle epoch atteint-on certains seuils de loss ?
    thresholds = [0.1, 0.05, 0.03, 0.02]
    print(f"\n  Paliers de performance (test loss) :")
    cumul_times = [sum(timing["epoch_times"][:i+1]) for i in range(len(timing["epoch_times"]))]
    for t in thresholds:
        reached = [(ep, cumul_times[ep]) for ep, l in enumerate(test_losses) if l <= t]
        if reached:
            ep, sec = reached[0]
            print(f"    Loss ≤ {t} : epoch {ep+1} ({sec:.1f}s)")
        else:
            print(f"    Loss ≤ {t} : non atteint")

    # Sauvegarder
    config = {
        "model_type": MODEL_TYPE, "seq_len": SEQ_LEN,
        "n_params": info["total_params"], "size_mb": info["size_mb"],
        "n_train_samples": timing["n_train_samples"],
    }
    metrics = {
        "final_train_loss": train_losses[-1], "final_test_loss": test_losses[-1],
        "best_test_loss": min(test_losses), "mae_degrees": mae * 180 / 3.14159,
        "total_time_s": timing["total_time_s"],
        "avg_epoch_time_s": timing["avg_epoch_time_s"],
        "train_losses": train_losses, "test_losses": test_losses,
    }
    save_run(EXPERIMENT_NAME, config, metrics)
    print(f"\nPour comparer : compare_runs('{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "rnn"], default="mlp")
    parser.add_argument("--seq-len", type=int, default=1)
    args = parser.parse_args()
    MODEL_TYPE = args.model
    SEQ_LEN = args.seq_len
    main()
