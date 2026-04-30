"""
Cache de datasets pré-traités.

Au premier chargement : charge depuis LeRobot, redimensionne les images,
normalise, découpe en séquences, et sauvegarde le tout en .pt sur disque.

Aux chargements suivants : charge directement le .pt (~2s au lieu de ~30s).
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE_DIR = Path(__file__).parent.parent / "data_cache"


def get_device():
    """Retourne le meilleur device disponible."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cache_key(dataset_name: str, n_episodes: int | None, image_size: int,
               seq_len: int, filter_reward: bool) -> str:
    """Génère un nom de fichier cache unique pour cette config."""
    ep_str = f"ep{n_episodes}" if n_episodes else "all"
    filt_str = "filtered" if filter_reward else "raw"
    seq_str = f"seq{seq_len}" if seq_len > 1 else "single"
    ds_name = dataset_name.replace("/", "_")
    return f"{ds_name}_{ep_str}_{image_size}px_{seq_str}_{filt_str}"


def load_pusht_cached(
    dataset_name: str = "lerobot/pusht",
    n_episodes: int | None = 50,
    image_size: int = 64,
    seq_len: int = 1,
    filter_reward: bool = False,
    normalize: bool = True,
) -> tuple:
    """Charge le dataset PushT, avec cache sur disque.

    Retourne : (X_img, X_pos, Y, norm_params, metadata)
      - Si seq_len=1 : X_img (N, 3, H, W), X_pos (N, 2), Y (N, 2)
      - Si seq_len>1 : X_img (N, seq, 3, H, W), X_pos (N, seq, 2), Y (N, 2)
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(dataset_name, n_episodes, image_size, seq_len, filter_reward)
    cache_file = CACHE_DIR / f"{key}.pt"

    if cache_file.exists():
        print(f"  Cache trouvé : {cache_file}")
        data = torch.load(cache_file, weights_only=True)
        print(f"  {data['metadata']['info']}")
        return data["X_img"], data["X_pos"], data["Y"], data["norm"], data["metadata"]

    # Pas de cache → charger depuis LeRobot
    print(f"  Pas de cache, chargement depuis {dataset_name}...")
    episodes = list(range(n_episodes)) if n_episodes else None
    dataset = LeRobotDataset(dataset_name, episodes=episodes)

    n_frames = len(dataset)
    print(f"  {n_frames} frames, {dataset.num_episodes} épisodes")

    # Extraire tout d'un coup
    all_imgs = []
    all_pos = []
    all_actions = []
    all_eps = []
    all_rewards = []

    for i in range(n_frames):
        s = dataset[i]
        img = F.interpolate(
            s["observation.image"].unsqueeze(0), size=image_size, mode="bilinear"
        ).squeeze(0)
        all_imgs.append(img)
        all_pos.append(s["observation.state"])
        all_actions.append(s["action"])
        all_eps.append(s["episode_index"].item())
        all_rewards.append(s["next.reward"].item())

    all_imgs = torch.stack(all_imgs).float()
    all_pos = torch.stack(all_pos).float()
    all_actions = torch.stack(all_actions).float()

    # Filtrer si demandé
    n_before = len(all_imgs)
    if filter_reward:
        keep = [i for i, r in enumerate(all_rewards) if r > 0]
        all_imgs = all_imgs[keep]
        all_pos = all_pos[keep]
        all_actions = all_actions[keep]
        all_eps = [all_eps[i] for i in keep]
        print(f"  Filtre reward>0 : {len(keep)}/{n_before} frames gardées ({len(keep)/n_before:.0%})")

    # Normaliser
    norm = {}
    if normalize:
        p_mean, p_std = all_pos.mean(0), all_pos.std(0)
        y_mean, y_std = all_actions.mean(0), all_actions.std(0)
        p_std = torch.where(p_std > 1e-6, p_std, torch.ones_like(p_std))
        y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
        all_pos = (all_pos - p_mean) / p_std
        all_actions = (all_actions - y_mean) / y_std
        norm = {"p_mean": p_mean, "p_std": p_std, "y_mean": y_mean, "y_std": y_std}

    # Séquences ou frames individuelles
    if seq_len > 1:
        seq_imgs, seq_pos, seq_actions = [], [], []
        i = 0
        while i + seq_len <= len(all_imgs):
            if all_eps[i] == all_eps[i + seq_len - 1]:
                seq_imgs.append(all_imgs[i:i + seq_len])
                seq_pos.append(all_pos[i:i + seq_len])
                seq_actions.append(all_actions[i + seq_len - 1])
                i += 1
            else:
                i += 1

        X_img = torch.stack(seq_imgs)
        X_pos = torch.stack(seq_pos)
        Y = torch.stack(seq_actions)
        info = f"{len(X_img)} séquences (len={seq_len}), {image_size}px"
    else:
        X_img = all_imgs
        X_pos = all_pos
        Y = all_actions
        info = f"{len(X_img)} frames, {image_size}px"

    if filter_reward:
        info += ", filtré reward>0"

    metadata = {
        "dataset": dataset_name,
        "n_episodes": n_episodes,
        "n_frames_raw": n_frames,
        "n_frames_used": len(all_imgs),
        "image_size": image_size,
        "seq_len": seq_len,
        "filter_reward": filter_reward,
        "info": info,
    }

    print(f"  {info}")

    # Sauvegarder en cache
    torch.save({
        "X_img": X_img, "X_pos": X_pos, "Y": Y,
        "norm": norm, "metadata": metadata,
    }, cache_file)
    print(f"  Cache sauvegardé : {cache_file} ({cache_file.stat().st_size / 1e6:.0f} Mo)")

    return X_img, X_pos, Y, norm, metadata
