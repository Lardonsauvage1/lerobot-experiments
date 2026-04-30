"""
Expérience 19 — Générer les vidéos pour les checkpoints RNN fenêtre glissante.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from collections import deque
from src.tracker import save_eval_videos, get_run_dir

# ============================================================
IMAGE_SIZE = 64
CHUNK_SIZE = 20
N_EVAL_EPISODES = 3  # Juste pour les vidéos
MAX_STEPS = 300
# ============================================================


class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=64, image_size=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = image_size // 4
        self.fc = nn.Linear(32 * reduced * reduced, feature_dim)
        self.feature_dim = feature_dim
    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class RNNWindowChunkPolicy(nn.Module):
    def __init__(self, cnn_feature_dim, image_size, rnn_hidden, rnn_layers,
                 chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        input_dim = cnn_feature_dim + pos_dim
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True)
        self.fc_out = nn.Linear(rnn_hidden, action_dim * chunk_size)

    def forward(self, images_seq, pos_seq):
        batch, window = images_seq.shape[:2]
        imgs_flat = images_seq.reshape(batch * window, *images_seq.shape[2:])
        feats = self.cnn(imgs_flat).reshape(batch, window, -1)
        x = torch.cat([feats, pos_seq], dim=2)
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        out = self.fc_out(last_hidden)
        return out.reshape(-1, self.chunk_size, self.action_dim)


def load_norm():
    from src.data_cache import load_pusht_cached
    _, _, _, norm, _ = load_pusht_cached(n_episodes=50, image_size=IMAGE_SIZE, seq_len=1)
    return norm


def eval_with_videos(model, norm, window_size, label):
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Éval vidéos : {label}...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    all_frames = []
    model.eval()

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        action_buffer = []
        img_buffer = deque(maxlen=window_size)
        pos_buffer = deque(maxlen=window_size)

        for step in range(MAX_STEPS):
            frames.append(env.render())

            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            agent_pos_norm = (agent_pos - norm["p_mean"]) / norm["p_std"]
            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)

            img_buffer.append(img)
            pos_buffer.append(agent_pos_norm)

            if len(action_buffer) == 0:
                while len(img_buffer) < window_size:
                    img_buffer.appendleft(img_buffer[0])
                    pos_buffer.appendleft(pos_buffer[0])

                imgs = torch.stack(list(img_buffer)).unsqueeze(0)
                poss = torch.stack(list(pos_buffer)).unsqueeze(0)

                with torch.no_grad():
                    chunk_pred = model(imgs, poss)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        coverage = info.get("coverage", 0)
        print(f"    Episode {ep+1} : coverage={coverage:.1%}")
        all_frames.append(frames)

    env.close()

    run_dir = get_run_dir("19_rnn_videos", label)
    save_eval_videos(all_frames, run_dir, fps=10)
    print(f"    Vidéos dans : {run_dir}")


def main():
    norm = load_norm()

    checkpoints = [
        {"path": "results/runs/18_rnn_window_window5/checkpoint_ep100.pt", "window": 5, "label": "window5_ep100"},
        {"path": "results/runs/18_rnn_window_window5/checkpoint_ep200.pt", "window": 5, "label": "window5_ep200"},
        {"path": "results/runs/18_rnn_window_window5/checkpoint_ep300.pt", "window": 5, "label": "window5_ep300"},
        {"path": "results/runs/18_rnn_window_window10/checkpoint_ep100.pt", "window": 10, "label": "window10_ep100"},
    ]

    for cp in checkpoints:
        print(f"\n--- {cp['label']} ---")

        model = RNNWindowChunkPolicy(
            cnn_feature_dim=64, image_size=IMAGE_SIZE,
            rnn_hidden=128, rnn_layers=1, chunk_size=CHUNK_SIZE,
        )

        checkpoint = torch.load(cp["path"], weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        eval_with_videos(model, norm, cp["window"], cp["label"])

    print("\nTerminé.")


if __name__ == "__main__":
    main()
