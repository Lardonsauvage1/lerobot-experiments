# Phase 4 — Compression du Diffusion Policy pour la latence

> Le run 46 a résolu Lift à **100 % de succès** avec un Diffusion Policy LeRobot de **263.7 M params**. Cette phase cherche le **plus petit / plus rapide modèle gardant les perfs**, avec une **priorité explicite sur la latence d'inférence** (préparer le déploiement temps réel sur le bras 5 axes).

## Point de départ (run 46)

| Composant | Params | Part |
|---|---|---|
| **U-Net débruiteur** (`down_dims [512,1024,2048]`) | 252.5 M | **95.8 %** |
| Vision (ResNet18 + spatial-softmax) | 11.2 M | 4.2 % |
| **Total** | **263.7 M** | ~1.05 Go fp32 |

Config : `n_obs_steps=2`, `horizon=16`, `n_action_steps=8`, DDPM `num_train_timesteps=100`, `num_inference_steps` ramené à **10** à l'eval, image 96×96 + état 19D, action 7D.

**Tout le poids est dans le U-Net** → c'est le levier principal. La vision (ResNet18) n'est que 4 %.

## Décisions cadrées (2026-05-20)

- **Métrique = latence d'inférence** (pas que le nb de params). Sur diffusion, une décision « pleine » = *N passes U-Net* → deux leviers **multiplicatifs** : nb de pas de débruitage × taille du U-Net.
- **On garde l'image** (ResNet18). **Pas de state-only** (malgré l'échec des runs 33-36 en non-diffusion, on ne re-tente pas ici).
- **Critère dé-saturé** : le succès binaire sature à 100 % dès 3 000 steps → inutilisable seul comme signal de recherche. Preuve : succès plat 100 % de 3K→12K, mais avg `max_z` (marge de levage) 0.918 → 1.006. On ajoute des métriques **continues**.

## Leviers (ordre par ROI latence)

1. **Réduire `num_inference_steps`** — gratuit, sans réentraînement. Chaque pas = 1 forward U-Net. Sweep 10→8→5→4→3→2→1, garder le minimum qui tient les perfs. Référence littérature : LightDP tourne à 4 pas (vs 100). Levier n°1.
2. **Sweep `down_dims` du U-Net** — réentraînement. `[512,1024,2048]` (défaut, 252 M) → `[256,512,1024]` (~63 M) → `[128,256,512]` (~16 M U-Net) → `[64,128,256]` (~4 M U-Net, la vision 11 M devient alors le plancher). Réduction ~quadratique.
3. **Quantization fp16** (et tenter int8) sur le gagnant — re-bench latence + re-vérifier les perfs.

## Phase 0 — socle de comparaison propre (à coder en premier, réutilisé partout)

Protocole figé, identique pour **tous** les modèles (baseline, rétrécis, quantifiés) :

- **Split train 150 / val 50** (figé, **contigu** : train = démos 0–149, val = 150–199). Choix délibéré : simuler le régime peu-de-données du bras réel. *Contigu et non aléatoire* car `lerobot-train` active l'EpisodeAwareSampler (via `drop_n_last_frames`) qui indexe les frames dans l'espace original ; un train non préfixe-0 déborde le `hf_dataset` sous-sélectionné (IndexError). ⚠️ Le run 46 a vu les 200 démos → non auditable, gardé comme « preuve max » mais **hors comparaison** ; le baseline du sweep doit être **réentraîné sur 150**.
- **Init states de rollout = ceux du val set** (50 départs jamais entraînés → teste la généralisation). Seed env fixe.
- **Métriques par épisode** (continues, pas juste binaire) :
  - **succès** (taux),
  - **temps-jusqu'au-succès** (nb de steps avant de soulever) — se dégrade *en douceur* avant le succès → détecteur précoce,
  - **marge de levage / stabilité du maintien** (`max_z` au-dessus du seuil, fraction de steps maintenue).
- **Early-stop par modèle** : tracer la **val noise-MSE** par checkpoint ; arrêt quand elle remonte. Chaque modèle (surtout pruné) s'arrête à *son* optimum, pas à un nb de steps figé.
- ⚠️ **La loss diffusion (prédiction de bruit) est un proxy bruité du succès rollout** → ne pas s'y fier seule. **Croiser** val-loss (détecteur d'overfit / quand arrêter) et métriques de rollout (qui est vraiment le meilleur).

**Critère de validité** : un modèle est acceptable s'il reste dans une tolérance du baseline sur (succès, temps-au-succès, marge) — pas seulement « encore 100 % ».

## Procédure box GPU (réentraînements)

Le Mac est lent pour l'entraînement (run 46 : **~16 h / 12K steps sur MPS**, ~3-4 s/step). La **box Linux + GPU** (`ssh gpu` → 192.168.1.95) est l'option rapide, mais quand elle est indisponible on entraîne **sur le Mac** (`--policy.device=mps`, en arrière-plan). Le Mac reste de toute façon pour l'éval (harnais validé sur MPS).

**Split figé** : `results/runs/lift/phase4_split.json` (**contigu**, train 0–149 / val 150–199) — versionné, identique des deux côtés. Liste train prête pour la CLI : `results/runs/lift/phase4_train_episodes.txt`.

**Commande baseline 150** (mettre `--policy.device=cuda` sur la box, `mps` sur le Mac ; sur la box précéder de `git pull`) :
```bash
venv312/bin/lerobot-train \
  --policy.type=diffusion --policy.repo_id=local/lift_ph_diffusion --policy.push_to_hub=false \
  --policy.device=mps \
  --dataset.repo_id=local/lift_ph --dataset.root=data_cache/lerobot_lift_ph \
  --dataset.episodes="$(cat results/runs/lift/phase4_train_episodes.txt)" \
  --output_dir=results/runs/lift/47_baseline_150 \
  --steps=12000 --batch_size=32 --save_freq=3000 --eval_freq=999999 \
  --num_workers=4 --wandb.enable=false --seed=42 \
  2>&1 | tee results/logs/lift/run_47_baseline_150.log
```
→ checkpoints 3K / 6K / 9K / 12K dans `results/runs/lift/47_baseline_150/checkpoints/`.

*(Si entraîné sur la box : rapatrier d'abord les checkpoints sur le Mac via `scp -r <box>:.../47_baseline_150/checkpoints results/runs/lift/47_baseline_150/` — lourds + gitignorés, pas git.)*

**Éval propre de chaque checkpoint** (sur le Mac) sur le val set figé :
```bash
for s in 003000 006000 009000 012000; do
  python -u experiments/lift/47_phase0_eval.py \
    --checkpoint results/runs/lift/47_baseline_150/checkpoints/$s/pretrained_model \
    --out results/runs/lift/47_baseline_150/phase0_eval_$s.json
done
```
On trace ensuite succès + t_success + max_z + hold_fraction **par checkpoint** → courbe de stop (quand les métriques rollout plafonnent) + val noise-MSE (détecteur d'overfit, script séparé).

## Résultats finaux ✅ (phase bouclée)

> **Point de fonctionnement : U-Net `[32,64,128]` + vision mini-CNN → 1.65 M params, 100 % de succès à 4 pas de diffusion, ~44 ms/décision.** vs baseline 263.7 M @ 10 pas (938 ms) → **÷160 params, ÷21 latence, perf identique** (au plafond expert).

Trois leviers, dans l'ordre où ils ont payé :
1. **Largeur du U-Net** (`down_dims`) : 252 M → 1.6 M (**÷160**) sans perte. Plancher de capacité = `[32,64,128]` ; en dessous (`[16,32,64]`), falaise à ~2 %.
2. **Pas de diffusion** : 10 → **4** sans perte (cassure à 2 pas). Plancher universel, indépendant de la taille.
3. **Vision** : ResNet18 (11.2 M) → **mini-CNN 0.03 M** from scratch, sans perte (Lift est visuellement simple). Débloque taille + vitesse d'entraînement (÷4) ; latence ~inchangée (le U-Net domine à l'inférence).

**Tableau maître complet** (grille latence × succès, sweep, plafond démos, mini-CNN) : [`../results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md).

**Frontière suivante éventuelle** : rien d'évident côté compression (vision déjà à 0.03 M). Les pistes restantes sont ailleurs (sim-to-real, efficacité données, tâche plus dure).

## Outils déjà en place

- `experiments/lift/46_bench_inference.py` — bench latence MPS/CPU, sample vs queue-pop, extrapolation par épisode. Réutilisé tel quel.
- `experiments/lift/46_eval_all_checkpoints.py` — base de l'éval comparative (⚠️ à refactorer : tire des init states **différents** par checkpoint → biais ; la Phase 0 corrige ça avec des états figés).
- `experiments/lift/46_plot_loss.py` — parse le log lerobot-train (loss/grad/lr) ; **overlay train-vs-val** si le log contient des lignes `val_loss:` (cf. ci-dessous). Paramétrique : `--log --out --log-freq`.
- `experiments/lift/47_phase0_eval.py` — rollout d'un checkpoint sur le val set figé (métriques continues).
- `experiments/lift/48_val_loss.py` — val noise-MSE d'un checkpoint (post-hoc).
- `experiments/lift/49_phase0_curve.py` — balayage : rollout + val-loss sur tous les checkpoints d'un run → `phase0_curve.{json,png}`. Sauvegarde incrémentale.
- `experiments/lift/50_train_valloss.py` — **copie patchée de `lerobot-train`** : logge la **val-loss en continu** (toutes les `log_freq` steps) sur le val set = épisodes hors `dataset.episodes` (auto-cohérent). Recette identique (même `update_policy`/`preprocessor`), juste une passe val read-only ajoutée. **À utiliser pour les retrains du sweep** (le baseline 150 a été fait avec lerobot-train standard → val seulement aux 8 checkpoints). Mêmes args que lerobot-train.
- `src/lift_data.py` / `src/lift_eval.py` — chargement HDF5 + split/env/rollout (métriques continues).

## Référence du run 46

Diffusion Policy entraîné via `lerobot-train` (15K steps prévus, checkpoints tous les 3K, arrêté/évalué à 12K). Tous les checkpoints à 100 % sur 20 ép. (init states aléatoires — comparaison biaisée) :

| step | succès | avg max_z |
|------|--------|-----------|
| 3000 | 100 % | 0.918 |
| 6000 | 100 % | 0.984 |
| 9000 | 100 % | 0.978 |
| 12000 | 100 % | 1.006 |

Détail des 5 bugs d'eval résolus pour atteindre 100 % : voir `README.md` § « Run 46 ».
