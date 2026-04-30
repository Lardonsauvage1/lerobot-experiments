"""
Expérience 01 — MLP basique pour prédire les actions d'un bras robotique.

Dataset : lerobot/utokyo_xarm_pick_and_place
  Robot : xArm 6 axes + pince
  Tâche : saisir une assiette blanche et la poser sur une assiette rouge
  observation.state (8D) : 6 joints + 2 valeurs inutilisées (toujours 0)
  action (7D) : 6 commandes articulaires + 1 pince (0=ouvert, 1=fermé)

Le modèle : un MLP (Multi-Layer Perceptron) — des couches de neurones empilées.
  - Chaque couche = multiplication matricielle + activation non-linéaire
  - Pas de traitement d'image — on utilise seulement les positions articulaires
  - Entrée : position actuelle des 6 joints (en radians)
  - Sortie : prochaine commande (6 joints + pince)

=== VARIANTES À TESTER ===
1. HIDDEN_LAYERS  — nombre et taille des couches cachées
2. ACTIVATION     — fonction d'activation (relu, tanh, sigmoid)
3. LEARNING_RATE  — vitesse d'apprentissage
4. USE_JOINTS     — quels joints utiliser en entrée
5. PREDICT_JOINTS — quels joints prédire en sortie
6. N_EPISODES     — combien d'épisodes utiliser
7. NORMALIZE      — normaliser les données ou pas
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.tracker import save_run
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES À MODIFIER POUR CHAQUE EXPÉRIENCE
# ============================================================

EXPERIMENT_NAME = "01_mlp_basique"
DATASET = "lerobot/utokyo_xarm_pick_and_place"

HIDDEN_LAYERS = [64, 64]       # Taille des couches cachées
ACTIVATION = "relu"            # relu, tanh, sigmoid
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 256
NORMALIZE = True               # Normaliser les données

# Indices des joints à utiliser/prédire
# State : [joint0, joint1, joint2, joint3, joint4, joint5, (vide), (vide)]
# Action: [joint0, joint1, joint2, joint3, joint4, joint5, pince]
USE_JOINTS = [0, 1, 2, 3, 4, 5]         # Joints du state à utiliser (exclut les 2 vides)
PREDICT_JOINTS = [0, 1, 2, 3, 4, 5, 6]  # Joints de l'action à prédire (6 joints + pince)
N_EPISODES = None                        # None = tout, ou un nombre

# ============================================================

STATE_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "(vide)", "(vide)"]
ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "pince"]


def get_activation(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]


class MLP(nn.Module):
    """Réseau de neurones simple (Multi-Layer Perceptron).

    Architecture :
        Entrée → [Linear + Activation] × N couches → Linear → Sortie

    Chaque couche Linear fait : sortie = entrée × poids + biais
    L'activation ajoute de la non-linéarité (sinon tout s'annule en une seule couche).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int],
                 activation: str = "relu"):
        super().__init__()

        layers = []
        prev_dim = input_dim
        act_fn = get_activation(activation)

        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data():
    """Charge le dataset et prépare les tenseurs X (observations) et Y (actions)."""
    print(f"Chargement du dataset {DATASET}...")
    episodes = list(range(N_EPISODES)) if N_EPISODES else None
    dataset = LeRobotDataset(DATASET, episodes=episodes)

    print(f"  Tâche    : {dataset[0]['task']}")
    print(f"  Frames   : {len(dataset)}")
    print(f"  Épisodes : {dataset.num_episodes}")

    # Extraire les données
    states = []
    actions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        states.append(sample["observation.state"])
        actions.append(sample["action"])

    X = torch.stack(states).float()
    Y = torch.stack(actions).float()

    # Sélectionner les joints
    X = X[:, USE_JOINTS]
    Y = Y[:, PREDICT_JOINTS]

    names_in = [STATE_NAMES[j] for j in USE_JOINTS]
    names_out = [ACTION_NAMES[j] for j in PREDICT_JOINTS]
    print(f"  Entrée ({len(USE_JOINTS)}D)  : {names_in}")
    print(f"  Sortie ({len(PREDICT_JOINTS)}D) : {names_out}")

    # Normalisation
    norm_params = {}
    if NORMALIZE:
        x_mean, x_std = X.mean(dim=0), X.std(dim=0)
        y_mean, y_std = Y.mean(dim=0), Y.std(dim=0)
        x_std = torch.where(x_std > 1e-6, x_std, torch.ones_like(x_std))
        y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
        X = (X - x_mean) / x_std
        Y = (Y - y_mean) / y_std
        norm_params = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
        print("  Données normalisées")

    return X, Y, norm_params


def train(model, X, Y):
    """Entraîne le modèle et retourne les losses par epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Split train/test (80/20)
    n = len(X)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    Y_train, Y_test = Y[idx[:split]], Y[idx[split:]]

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            x_batch = X_train[batch_idx]
            y_batch = Y_train[batch_idx]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, Y_test).item()

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train_loss:.6f} — "
                  f"test: {test_loss:.6f}")

    return train_losses, test_losses, X_test, Y_test


def evaluate(model, X_test, Y_test, norm_params):
    """Évalue le modèle et affiche l'erreur en radians et degrés par joint."""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)

    # Dénormaliser si besoin
    if norm_params:
        y_std = norm_params["y_std"]
        y_mean = norm_params["y_mean"]
        pred = pred * y_std + y_mean
        Y_real = Y_test * y_std + y_mean
    else:
        Y_real = Y_test

    # Erreur par joint
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
        unit = "°" if name != "pince" else ""
        if name == "pince":
            print(f"  {name:<12} {mae_rad:>10.4f} {'—':>10} {'—':>10}   (0=ouvert, 1=fermé)")
        else:
            print(f"  {name:<12} {mae_rad:>10.4f} {mae_deg:>9.2f}° {max_deg:>9.2f}°")

    mae_total = errors.mean().item()
    print(f"\n  MAE moyenne : {mae_total:.4f} rad ({mae_total * 180 / 3.14159:.2f}°)")


def plot_losses(train_losses, test_losses):
    """Génère le graphe des courbes de loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(test_losses, label="Test", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (normalisée)")
    ax1.set_title(f"Loss — MLP {HIDDEN_LAYERS}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    start = min(20, len(train_losses) // 3)
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
    print(f"\n  Graphe sauvegardé : {filepath}")


def main():
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"{'='*60}")
    print(f"  Dataset         : {DATASET}")
    print(f"  Couches cachées : {HIDDEN_LAYERS}")
    print(f"  Activation      : {ACTIVATION}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Normalisation   : {NORMALIZE}")
    print(f"  N épisodes      : {N_EPISODES or 'tout'}")
    print()

    # 1. Charger les données
    X, Y, norm_params = load_data()

    # 2. Créer le modèle
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = MLP(input_dim, output_dim, HIDDEN_LAYERS, ACTIVATION)

    print(f"\nArchitecture :")
    print(f"  Entrée ({input_dim}) → ", end="")
    for h in HIDDEN_LAYERS:
        print(f"[{h} neurones + {ACTIVATION}] → ", end="")
    print(f"Sortie ({output_dim})")
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Taille     : {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024:.1f} Ko")
    print()

    # 3. Entraîner
    print("Entraînement...")
    train_losses, test_losses, X_test, Y_test = train(model, X, Y)

    # 4. Évaluer en unités réelles
    evaluate(model, X_test, Y_test, norm_params)

    # 5. Graphe
    plot_losses(train_losses, test_losses)

    # 6. Sauvegarder
    info = model_info(model, name=f"MLP {HIDDEN_LAYERS}")
    config = {
        "dataset": DATASET,
        "hidden_layers": str(HIDDEN_LAYERS),
        "activation": ACTIVATION,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "normalize": NORMALIZE,
        "use_joints": str(USE_JOINTS),
        "predict_joints": str(PREDICT_JOINTS),
        "n_episodes": N_EPISODES or "all",
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

    print(f"\nPour comparer les runs :")
    print(f"  from src.tracker import compare_runs")
    print(f"  compare_runs('{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    main()
