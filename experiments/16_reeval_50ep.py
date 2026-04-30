"""
Expérience 16 — Réévaluer les meilleurs modèles avec 50 épisodes.

On charge chaque model.pt et on l'évalue sur 50 épisodes avec les mêmes seeds fixes.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, measure_inference_time
)
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "16_reeval_50ep"
IMAGE_SIZE = 64
SEED = 42
N_EVAL_EPISODES = 50
MAX_STEPS = 300
# ============================================================


# === Tous les modèles qu'on a utilisés ===

class SimpleCNN_Color(nn.Module):
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
        self.feature_dim = feature_dim
    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class SimpleCNN_Gray(nn.Module):
    def __init__(self, channels, feature_dim, image_size):
        super().__init__()
        conv_layers = []
        in_ch = 1
        for out_ch in channels:
            conv_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        reduced = image_size // (2 ** len(channels))
        self.fc = nn.Linear(channels[-1] * reduced * reduced, feature_dim)
        self.feature_dim = feature_dim
    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class ChunkPolicy(nn.Module):
    """CNN + MLP chunk."""
    def __init__(self, cnn, hidden_layers, chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = cnn
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        mlp_input = cnn.feature_dim + pos_dim
        output_dim = action_dim * chunk_size
        layers = []
        prev = mlp_input
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, image, agent_pos):
        feats = self.cnn(image)
        x = torch.cat([feats, agent_pos], dim=1)
        return self.mlp(x).reshape(-1, self.chunk_size, self.action_dim)


class VAEEncoder(nn.Module):
    def __init__(self, action_dim, chunk_size, z_dim):
        super().__init__()
        input_dim = action_dim * chunk_size
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
    def forward(self, actions_flat):
        h = self.net(actions_flat)
        return self.fc_mu(h), self.fc_logvar(h)


class VAETransformerPolicy(nn.Module):
    def __init__(self, cnn, d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, z_dim, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = cnn
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.d_model = d_model
        self.vae_encoder = VAEEncoder(action_dim, chunk_size, z_dim)
        self.obs_proj = nn.Linear(cnn.feature_dim + pos_dim + z_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, action_dim)
    def forward(self, image, agent_pos, target_actions=None):
        batch = image.shape[0]
        cnn_feats = self.cnn(image)
        if target_actions is not None:
            actions_flat = target_actions.reshape(batch, -1)
            mu, logvar = self.vae_encoder(actions_flat)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = torch.randn(batch, self.z_dim, device=image.device)
        obs = torch.cat([cnn_feats, agent_pos, z], dim=1)
        context = self.obs_proj(obs).unsqueeze(1)
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)
        return self.action_head(decoded)


class TransformerChunkPolicy(nn.Module):
    def __init__(self, cnn, d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = cnn
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model
        self.obs_proj = nn.Linear(cnn.feature_dim + pos_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, action_dim)
    def forward(self, image, agent_pos):
        batch = image.shape[0]
        cnn_feats = self.cnn(image)
        obs = torch.cat([cnn_feats, agent_pos], dim=1)
        context = self.obs_proj(obs).unsqueeze(1)
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)
        return self.action_head(decoded)


def load_model_from_dir(model_dir, grayscale=False):
    """Charge un modèle depuis son dossier."""
    model_dir = Path(model_dir)
    with open(model_dir / "model_config.json") as f:
        config = json.load(f)

    model_class = config.get("class", "")
    cnn_channels = config.get("cnn_channels", [16, 32])
    cnn_feature_dim = config.get("cnn_feature_dim", 64)
    chunk_size = config.get("chunk_size", 20)
    in_channels = config.get("in_channels", 3)

    if in_channels == 1 or grayscale:
        cnn = SimpleCNN_Gray(cnn_channels, cnn_feature_dim, IMAGE_SIZE)
    else:
        cnn = SimpleCNN_Color(cnn_channels, cnn_feature_dim, IMAGE_SIZE)

    if "VAE" in model_class:
        model = VAETransformerPolicy(
            cnn=cnn, d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4), n_layers=config.get("n_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            chunk_size=chunk_size, z_dim=config.get("z_dim", 16),
        )
    elif "Transformer" in model_class:
        model = TransformerChunkPolicy(
            cnn=cnn, d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4), n_layers=config.get("n_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            chunk_size=chunk_size,
        )
    elif "Chunk" in model_class:
        model = ChunkPolicy(
            cnn=cnn, hidden_layers=config.get("hidden_layers", [128, 64]),
            chunk_size=chunk_size,
        )
    elif "PushT" in model_class:
        # Exp 08 : PushTPolicy = chunk=1
        model = ChunkPolicy(
            cnn=cnn, hidden_layers=config.get("hidden_layers", [128, 64]),
            chunk_size=1,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    return model, config


def load_norm():
    from src.data_cache import load_pusht_cached
    _, _, _, norm, _ = load_pusht_cached(n_episodes=50, image_size=IMAGE_SIZE, seq_len=1)
    return norm


def eval_model(model, norm, chunk_size, grayscale=False):
    import gymnasium as gym
    import gym_pusht  # noqa

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []
    model.eval()

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        max_coverage = 0
        action_buffer = []

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")
                if grayscale:
                    img = img.mean(dim=1, keepdim=True)

                with torch.no_grad():
                    chunk_pred = model(img, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        status = "✓" if success else "✗"
        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1:2d}/{N_EVAL_EPISODES} : {status} coverage={max_coverage:.1%}")
        if frames:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])
    coverages = [r["max_coverage"] for r in results]

    print(f"    Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"    Coverage moyen : {avg_coverage:.1%}")
    print(f"    Coverage min/max : {min(coverages):.1%} / {max(coverages):.1%}")

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
        "coverages": coverages,
    }, all_frames


# Liste des modèles à réévaluer
MODELS = [
    # Exp 08 — CNN scratch vs pretrained
    {"name": "08 CNN scratch + MLP", "dir": "results/runs/08_pretrained_cnn_scratch", "grayscale": False},
    # Exp 09 — Action chunking (MLP)
    {"name": "09 MLP chunk=1", "dir": "results/runs/09_action_chunking_chunk1", "grayscale": False},
    {"name": "09 MLP chunk=5", "dir": "results/runs/09_action_chunking_chunk5", "grayscale": False},
    {"name": "09 MLP chunk=10", "dir": "results/runs/09_action_chunking_chunk10", "grayscale": False},
    {"name": "09 MLP chunk=20", "dir": "results/runs/09_action_chunking_chunk20", "grayscale": False},
    # Exp 10 — Transformer
    {"name": "10 Transf 1L 50ep", "dir": "results/runs/10_transformer_1layer", "grayscale": False},
    {"name": "10 Transf 2L 50ep", "dir": "results/runs/10_transformer_2layer", "grayscale": False},
    {"name": "10 Transf 2L 100ep", "dir": "results/runs/10_transformer_2layer_ep100", "grayscale": False},
    {"name": "10 Transf 2L 206ep 50ep", "dir": "results/runs/10_transformer_2layer_ep206", "grayscale": False},
    {"name": "10 Transf 2L 206ep 100ep", "dir": "results/runs/10_transformer_2layer_ep206_ep100epochs", "grayscale": False},
    {"name": "10 Transf 2L 206ep 200ep", "dir": "results/runs/10_transformer_2layer_ep206_ep200epochs", "grayscale": False},
    {"name": "10 Transf 2L lr=5e-3", "dir": "results/runs/10_transformer_2layer_ep206_ep50epochs_lr0.005", "grayscale": False},
    {"name": "10 Transf 2L lr=1e-2", "dir": "results/runs/10_transformer_2layer_ep206_ep50epochs_lr0.01", "grayscale": False},
    # Exp 14 — VAE
    {"name": "14 VAE kl=0.01", "dir": "results/runs/14_vae_transformer_vae_z16_kl0.01", "grayscale": False},
    {"name": "14 VAE kl=0.1", "dir": "results/runs/14_vae_transformer_vae_z16_kl0.1", "grayscale": False},
    {"name": "14 VAE kl=0.1 anneal", "dir": "results/runs/14_vae_transformer_vae_z16_kl0.1_annealing", "grayscale": False},
    # Exp 15 — Grayscale
    {"name": "15 NB CNN[16,32]", "dir": "results/runs/15_grayscale_gray_cnn[16,32]", "grayscale": True},
    {"name": "15 NB CNN[8,16]", "dir": "results/runs/15_grayscale_gray_cnn[8,16]", "grayscale": True},
    {"name": "15 NB CNN[4,8]", "dir": "results/runs/15_grayscale_gray_cnn[4,8]", "grayscale": True},
]


def main():
    print(f"\n{'='*60}")
    print(f"RÉÉVALUATION — {N_EVAL_EPISODES} épisodes par modèle")
    print(f"{'='*60}")

    norm = load_norm()
    all_results = []

    for m in MODELS:
        print(f"\n--- {m['name']} ---")
        print(f"  Dossier : {m['dir']}")

        if not Path(m["dir"]).exists():
            print(f"  SKIP — dossier introuvable")
            continue

        run_dir_name = m["name"].replace(" ", "_").replace(",", "").replace("[", "").replace("]", "").replace("=", "")
        existing_run_dir = Path("results/runs") / f"{EXPERIMENT_NAME}_{run_dir_name}"
        if (existing_run_dir / "ep0.mp4").exists():
            print(f"  SKIP — déjà évalué")
            # Charger le résultat existant
            continue

        model, config = load_model_from_dir(m["dir"], grayscale=m["grayscale"])
        chunk_size = config.get("chunk_size", 20)
        info = model_info(model, name=m["name"])

        eval_results, all_frames = eval_model(model, norm, chunk_size, grayscale=m["grayscale"])

        # Sauvegarder
        run_dir = get_run_dir(EXPERIMENT_NAME, m["name"].replace(" ", "_").replace(",", "").replace("[", "").replace("]", ""))
        if all_frames:
            save_eval_videos(all_frames, run_dir, fps=10)

        all_results.append({
            "name": m["name"],
            "params": info["total_params"],
            "coverage": eval_results["avg_coverage"],
            "success_rate": eval_results["success_rate"],
        })

    # Tableau comparatif final
    print(f"\n{'='*60}")
    print(f"COMPARAISON FINALE ({N_EVAL_EPISODES} épisodes)")
    print(f"{'='*60}")
    print(f"  {'Modèle':<35} {'Params':>8} {'Coverage':>10} {'Success':>10}")
    print(f"  {'-'*65}")
    for r in sorted(all_results, key=lambda x: -x["coverage"]):
        print(f"  {r['name']:<35} {r['params']:>8,} {r['coverage']:>9.1%} {r['success_rate']:>9.0%}")


if __name__ == "__main__":
    main()
