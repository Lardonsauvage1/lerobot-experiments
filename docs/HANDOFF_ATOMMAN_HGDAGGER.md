# Passation — Boucle HG-DAgger « pomme » 100 % sur atomman

> **But de cette note :** tout ce qu'on a appris + les prochaines étapes concrètes, pour reprendre le
> travail **directement sur atomman** (ou avec un Claude lancé sur atomman). Auto-suffisante : pas besoin
> de la conversation précédente. Rédigée 2026-07-27.

---

## 0. Décision stratégique (le "pourquoi")

On **abandonne HIL-SERL / RL** (trop lourd : classifieur de récompense + critique + RLPD, instable, GPU-gourmand)
au profit de l'**imitation interactive = HG-DAgger / Sirius**. Raison : pour un pick-and-place, **égaler un bon
téléopérateur suffit** — pas besoin de *dépasser* l'humain (le seul avantage du RL). Et l'imitation :
- garde **ta Diffusion Policy** qui marche déjà,
- **zéro récompense, zéro critique** → supervisé, stable,
- c'est exactement ta boucle actuelle (datasets de correction + réentraînement), juste **resserrée**.

**Cible : tout faire sur atomman** (mono-machine) : entraîner + infférer + boucler.

---

## 1. La méthode HG-DAgger / Sirius (actionnable)

Boucle :
```
1. La policy (Diffusion) tourne sur le robot.
2. Tu la regardes. Tant qu'elle se débrouille -> tu ne touches à rien (rien enregistré).
3. Dès qu'elle DÉRAPE -> tu reprends la main (le "gate"), tu corriges jusqu'à re-stabiliser.
4. SEUL ce segment de reprise est enregistré comme nouvelle donnée experte.
5. Périodiquement : réentraîner (fine-tune) sur démos + toutes les corrections agrégées.
6. Redéployer. Répéter. Le TAUX D'INTERVENTION qui baisse = signal de convergence.
```

**Pondération façon Sirius** (améliore l'efficacité) :
- correction humaine → **poids fort**,
- transitions robot **juste avant** une intervention (= menaient à l'échec) → **poids faible**,
- robot autonome OK → poids normal.
→ attaque directement le problème "le robot dérive dans la table après une saisie ratée".

**Critère d'arrêt** : PAS la val-loss (décorrélée du succès en diffusion). On entraîne un budget fixe,
EMA, et on **juge au rollout réel** (succès + nb d'interventions/session). On arrête la boucle quand une
session ne nécessite quasi aucune reprise.

Évolution moderne si un jour la collecte devient pénible : **DMD** (Diffusion Meets DAgger) = synthétiser
des états de rattrapage depuis la **caméra poignet** au lieu de les rencontrer en vrai.

---

## 2. Ce qu'on a MESURÉ sur atomman (2026-07-27)

**Matériel** : Intel Core Ultra 9 185H (22 threads), **93 Go RAM**, iGPU **Intel Arc** (Xe-LPG, 128 EUs,
mémoire *partagée* avec le CPU ~87 Go), Level-Zero + OpenCL présents. Disque : 455 Go libres.

### Inférence — sur l'iGPU Arc via **OpenVINO** (déjà en prod)
Chemin : `~/ros2_ws/tools/pc/roby_ov.py` (`patch_policy_igpu`) porte **le U-Net** (96 % des params) en IR
OpenVINO sur `device="GPU"` ; backbone/scheduler/norm restent en torch CPU. `roby_infer_cart.py --igpu GPU --steps N`.

Latence inférence complète (DDIM-10, 1 cam, mesurée) :
| taille | CPU | Arc GPU |
|---|---|---|
| 30M | **46 ms** | 99 ms |
| 78M | 123 ms | 98 ms |
| 263M | 373 ms | **153 ms** |

→ **L'inférence n'est jamais le mur** (à 10 Hz, budget ~800 ms, tout passe même sur CPU). L'Arc ne gagne que
pour les **gros** modèles (≥78M). Petit modèle (≤30M) : **le CPU est même plus rapide** (surcoût transfert iGPU).
Inférence Arc **pendant** entraînement CPU parallèle = **+2 % de latence seulement** (unités séparées).

### Entraînement — sur l'iGPU Arc via **IPEX** (⚠️ PAS torch-xpu natif)
- ❌ **torch-xpu natif** (torch 2.13+xpu, oneMKL 2026) : le **matmul plante** (`could not make an engine with allocator`). NE PAS UTILISER.
- ✅ **IPEX** (torch 2.8.0+xpu + intel-extension-for-pytorch 2.8.10, oneMKL 2025.1) : **matmul OK**, entraînement OK.
- **bf16 = LE levier** (iGPU limité par la bande passante) : débit conv2d 159 → **280 éch/s** (×1,75).
- `ipex.optimize` fp32 = 0 gain ; channels_last = négligeable/négatif.
- ⚠️ **Augmenter le batch RALENTIT** (B=128→1024 : 280→215 éch/s) = iGPU **saturé dès B=128** = plafond mémoire.
  (Sur un GPU dédié type gb10 ce serait l'inverse.)
- **Meilleur : bf16 + ipex, B=128 = 280 éch/s = ×3,6 vs CPU** (77 éch/s fp32).
  → entraînement pomme ~0,55 → **~2 steps/s** ; fine-tune 1000 pas **~30 → ~8 min**.

**Conclusion vitesse** : atomman peut tout faire en mono-machine. gb10 (GPU dédié) reste plus rapide à
l'entraînement (profite des gros batches, pas l'iGPU), mais atomman-seul est **viable** (bf16 obligatoire).

---

## 3. Taille du modèle (idées reçues corrigées)

Répartition params (Diffusion Policy 1 caméra, ResNet-18 + U-Net), **vision = plancher fixe 11,2 M** :
| down_dims | TOTAL | vision | U-Net |
|---|---|---|---|
| (64,128,256) | 16 M | 11,2 | 4,9 |
| (128,256,256) — "m" du papier | 21 M | 11,2 | 9,7 |
| (128,256,512) | 28 M | 11,2 | 17,3 |
| (256,512,1024) — **défaut papier "l"** | 78 M | 11,2 | 67 |

- Le **défaut du papier Diffusion Policy = ~78 M** (≈ ton ancien "gros" 86M). "18M canonique" était FAUX (c'est leur config *Medium*, une ablation).
- MAIS : leur ablation montre que ~20M marche, et **ton propre résultat Lift** montre qu'un petit modèle tient les perfs sur TA tâche.
- **Cible de départ : ~30M** (down_dims (128,256,512)), 1 caméra. Rapide à infférer/entraîner. À valider par balayage rollout (30/78M) si doute.
- Latence CPU = dominée par le **ResNet-18** (11M) → réduire le U-Net n'accélère quasi pas ; pour aller plus vite il faudrait un backbone plus léger (ex. ResNet-10).

---

## 4. Prochaines étapes concrètes (le plan)

### Phase 1 — Valider le pipeline d'entraînement XPU (dérisque, ~15 min)
1. Installer lerobot dans le venv IPEX **sans casser torch 2.8+xpu** :
   `~/ipex_test_venv/bin/pip install lerobot==0.5.1 --no-deps` puis ajouter les deps manquantes HORS torch
   (`pip install` des deps une à une : draccus, datasets, huggingface_hub, imageio, av, opencv-python, einops, etc.
   — vérifier `python -c "import lerobot"` et surtout `torch.__version__` reste `2.8.0+xpu` après).
2. **Smoke test** : entraîner un ~30M sur le dataset JOINT déjà prêt `~/lerobot-experiments/data_cache/lerobot_apple_joint_224_clean`,
   `device=xpu`, bf16, ~200 pas. Confirmer : loss descend, pas d'erreur dtype/scheduler.
   - lerobot 0.5.1 gère déjà `xpu` (`lerobot/utils/device_utils.py`).
   - Appliquer `model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)` + `torch.autocast("xpu", torch.bfloat16)` dans la boucle.
   - Si le CLI `lerobot-train` ne gère pas proprement xpu+ipex, écrire une **boucle custom** (LeRobotDataset + DiffusionPolicy.forward + ipex.optimize + autocast), comme le faisait `bc_pretrain`.

### Phase 2 — Données cartésiennes (⚠️ PRÉALABLE BLOQUANT)
- **Il n'y a PAS de dataset cartésien au format LeRobot sur atomman** — seulement du **mcap brut** dans
  `~/roby_datasets/` (les 3 : `batch_collect_..._cart`, `batch_recovery_..._cart`, `batch_recovery_far_..._cart`)
  et des datasets LeRobot **joint** (`lerobot_apple_joint_224[_clean]`).
- `~/roby_dataset_to_cartesian.py` ajoute juste le topic `/tcp_pose` aux bags (mcap→mcap), **pas** LeRobot.
- **À RÉSOUDRE avec l'utilisateur** : où sont les datasets cartésiens LeRobot (ceux de `cart_combined_128`) ?
  (a) rapatrier depuis Mac/gb10, (b) lancer la conversion mcap→LeRobot sur atomman (retrouver/porter le convertisseur), (c) l'utilisateur pointe son script.

### Phase 3 — Modèle de base cartésien
- Entraîner le ~30M **cartésien** (action 7D = pose TCP 6D + pince), 3 datasets, xpu+bf16, budget long + EMA.

### Phase 4 — Déploiement + boucle HG-DAgger
- Convertir le U-Net en OpenVINO IR (`roby_ov.patch_policy_igpu(pol, model_dir, "GPU")`) → déployer via `roby_infer_cart`.
- Mettre en place : enregistrement des corrections (le "gate" au clavier/téléop existe déjà) → fine-tune xpu bf16
  (pondération Sirius) → redéploiement. Logger le **taux d'intervention/session** (courbe de convergence).

---

## 5. Pièges & règles (à respecter)

- 🚫 **NE JAMAIS toucher au système/réseau d'atomman** (pas de `sudo`, pas d'`apt`, pas de driver). Uniquement des venvs/scripts user.
- 🚫 **NE PAS casser le venv de déploiement** `~/lerobot-experiments/venv` (torch-cpu + OpenVINO, fait tourner le robot). Travailler dans **`~/ipex_test_venv`** (torch 2.8+xpu + IPEX, déjà installé, 7,5 Go).
- ⚙️ **Entraînement GPU = IPEX + bf16 uniquement** (le torch-xpu natif plante sur matmul). Batch **128** (plus gros = plus lent sur l'iGPU).
- 🐞 **Piège SSH** : les commandes ssh qui *backgroundent* un process avalent souvent la sortie ; **ne pas piper vers `grep` un flux qui peut hang** (grep bufferise → sortie invisible). Écrire dans un log + `cat`/`grep` le fichier ensuite, ou utiliser `timeout` distant (atomman a `timeout`, pas le Mac).
- 🐞 **Piège pgrep self-match** : `pgrep -f "pip install torch"` matche la commande ssh elle-même. Utiliser un pattern bracketé (`inte[l]`) ou surveiller le log.
- ✅ **Inférence Arc = OpenVINO** ; **Entraînement Arc = IPEX**. Ne pas confondre. Les deux sur le même iGPU en parallèle = contention non mesurée (préférer inférence et entraînement sur unités distinctes, ou séquentiel).

---

## 6. Chemins clés sur atomman
- Venv entraînement XPU : `~/ipex_test_venv/bin/python` (torch 2.8.0+xpu, ipex 2.8.10, numpy). **PAS de lerobot encore.**
- Venv déploiement : `~/lerobot-experiments/venv/bin/python` (torch-cpu, openvino 2026.2, lerobot 0.5.1). **Ne pas modifier.**
- Datasets bruts : `~/roby_datasets/` (mcap). Datasets LeRobot joint : `~/lerobot-experiments/data_cache/lerobot_apple_joint_224[_clean]`.
- Déploiement/inférence : `~/ros2_ws/tools/pc/roby_ov.py`, `roby_infer_cart.py` ; `~/roby_dataset_to_cartesian.py`.
- Modèles déployables : `~/deployable_models/` (dont `cart_combined_128`).
- Test : `ssh atomman` (Tailscale SSH). GPU OpenVINO : `ov.Core().available_devices == ['CPU','GPU']`.

---

## 7. RECETTE — Fabriquer un dataset LeRobot (mcap brut → format LeRobot)

**Données brutes** (`~/roby_datasets/<batch>/ep_XXX/*.mcap`, rosbag2/mcap). Topics utiles :
- `/head_camera/left/image_raw/compressed` + `/head_camera/right/image_raw/compressed` : JPEG 640×480 (décoder `cv2.imdecode`). **2 vues fixes** (pas embarquées).
- `/joint_states` : 5 angles rad (`position`, ordre joint_1..5).
- `/tcp_pose` : **pose cartésienne 6D** `[x,y,z,rvx,rvy,rvz]` — présente seulement dans les batches `_cart` (ajoutée par `roby_dataset_to_cartesian.py`).
- `/gripper` : Bool **événementiel** (3 msgs/épisode : OUVRE / FERME à la prise / OUVRE à la dépose) → **maintenir la dernière valeur** (hold) pour une cible dense.

**Étapes de conversion** (à écrire dans un script, ex. `~/roby_bag_to_lerobot.py`) :
1. **Lire chaque bag** avec `rosbag2_py.SequentialReader` (storage_id="mcap"), désérialiser les messages par topic avec leurs timestamps.
2. **Ré-échantillonner à une cadence commune** (ex. **15 Hz**) : pour chaque tick, prendre le message le plus proche (nearest) de chaque topic. Caméras ~60 Hz, joints ~100 Hz → aligner par timestamp.
3. **Décoder les images** (`cv2.imdecode` du JPEG), redimensionner à la résolution cible (ex. **128×128**, ou garder 1 seule caméra — voir §3 : on part **1 caméra** pour le ~30M).
4. **Construire obs/action** :
   - **obs(t)** = `{ image (3,128,128), state }` où `state` = tcp_pose(t) 6D (cartésien) *ou* joints(t) 5D.
   - **action(t)** = pose/joints du **pas suivant** (open-loop → la consigne == le prochain état) **+ gripper (hold)**. Cartésien : **action 7D** = tcp_pose(t+1) 6D + gripper 1D.
5. **Écrire au format LeRobot** avec l'API `LeRobotDataset.create` (v0.5.1) :
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
FEATURES = {
    "observation.image":  {"dtype": "video",   "shape": (3,128,128), "names": ["c","h","w"]},
    "observation.state":  {"dtype": "float32", "shape": (6,),  "names": ["x","y","z","rvx","rvy","rvz"]},  # cartésien
    "action":             {"dtype": "float32", "shape": (7,),  "names": ["x","y","z","rvx","rvy","rvz","grip"]},
}
ds = LeRobotDataset.create(repo_id="local/apple_cart_128", fps=15, features=FEATURES,
                           root="~/lerobot-experiments/data_cache/lerobot_apple_cart_128")
for ep in episodes:                      # 1 bag = 1 épisode
    for t in ticks_alignés:
        ds.add_frame({"observation.image": img_chw_uint8,
                      "observation.state": tcp_pose_t.astype("float32"),
                      "action": np.concatenate([tcp_pose_tp1, [gripper_hold]]).astype("float32")},
                     task="pick_apple")
    ds.save_episode()                    # écrit meta/data/videos, incrémente episode_index
# .create écrit meta/info.json ; save_episode remplit data/*.parquet + videos/*.mp4
```
   ⚠️ Vérifier la **signature exacte** de `create`/`add_frame`/`save_episode` dans lerobot **0.5.1** (l'API bouge entre versions ; ex. `task=` peut être requis dans `add_frame` ou `save_episode`).
6. **Répéter pour les 3 datasets** `_cart` (classique + 2 recovery), soit dans un seul dataset multi-tâche, soit 3 datasets à concaténer à l'entraînement (le classique + les corrections, façon HG-DAgger).
7. **Normalisation** : LeRobot calcule les stats (min/max, mean/std) à partir du dataset ; s'assurer que `meta/stats` est bien généré (ou le régénérer).

> Astuce : s'inspirer du script qui a produit `lerobot_apple_joint_224_clean` (même pipeline, mais **joint** au lieu de **cartésien** — remplacer `/joint_states` par `/tcp_pose` pour l'état/action, et garder les images). Chercher ce script sur le Mac (venv312) ou dans `~/lerobot-experiments/experiments/`.

---

## 8. RECETTE — Optimiser le GPU (entraînement Arc, IPEX + bf16)

**Setup** (dans le venv `~/ipex_test_venv`, déjà fait) :
```bash
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/xpu   # torch ASSORTI (oneMKL 2025.1 = matmul OK)
```

**Boucle d'entraînement optimisée** (les réglages qui comptent, mesurés) :
```python
import torch, intel_extension_for_pytorch as ipex
DEV = "xpu"
model = policy.to(DEV)                                   # DiffusionPolicy LeRobot
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# LE point clé : ipex.optimize + bf16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

for batch in loader:                                     # batch_size = 128 (voir ci-dessous)
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
    optimizer.zero_grad()
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):   # <-- bf16 = ×1,75
        out = model.forward(batch)
        loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]
    loss.backward()
    optimizer.step()
    torch.xpu.synchronize()                              # pour mesurer/juger la cadence
```

**Réglages MESURÉS (à respecter) :**
- ✅ **bf16 obligatoire** (`ipex.optimize(dtype=bf16)` + `autocast("xpu", bf16)`) → ×1,75, LE gros gain.
- ✅ **batch_size = 128**. ⚠️ **Ne PAS monter le batch** : l'iGPU est saturé, 256/512/1024 sont **plus lents** (débit qui baisse). Contre-intuitif mais mesuré.
- ⛔ `ipex.optimize` en fp32 = 0 gain ; `channels_last` = négligeable/négatif → inutile ici.
- Résultat attendu : **~2 steps/s** (vs ~0,55 en CPU fp32), soit ~8 min pour 1000 pas de fine-tune.

**Pièges bf16/IPEX :**
- **EMA** : garder l'EMA sur les poids **fp32 maîtres** (ipex.optimize garde un master fp32 en interne). Sauver les **checkpoints en fp32** (repasser `model.float()` / utiliser les poids master avant `save`).
- Vérifier au **smoke test** que la loss descend et qu'aucune couche (normalisation, scheduler de diffusion) ne casse en bf16 (buffers dtype). Si un op pose problème, l'exclure de l'autocast.
- **Inférence** : ne PAS réutiliser IPEX/XPU pour l'inférence en prod → c'est **OpenVINO** (`roby_ov`) qui sert (plus rapide + validé). IPEX/XPU = entraînement seulement.
- **Ne pas** faire tourner entraînement (IPEX) et inférence (OpenVINO) **en même temps sur l'Arc** sans mesurer la contention (même iGPU).
