# Lanceurs d'entraînement de septembre 2026 (poste `sam-AtomMan`)

Copiés **tels quels** depuis le répertoire personnel du PC le 2026-09-13, où ils
n'étaient versionnés nulle part. Ce sont des lanceurs **ponctuels** : chemins absolus de
cette machine (`/home/sam`, `~/ipex_test_venv`, `~/roby_xpu_env.sh` → dépôt
`roby-le-gentil-robot`, `tools/pc/roby_xpu_env.sh`). Ils ne sont pas faits pour être relancés
sans adaptation ; ils disent **comment** les modèles ont été produits.

**Ce qui fait foi pour un modèle** : le `train_config.json` enregistré par LeRobot dans
chacun de ses checkpoints (`outputs/<run>/checkpoints/<pas>/pretrained_model/`). Les
fichiers ci-dessous sont ceux passés au lancement ; ils ont pu être édités ensuite.

## Modèles du bras réel (Diffusion Policy, pomme, 2 caméras 128 px)

Correspondance lue dans chaque fichier (`output_dir`, `steps`, `dataset.repo_id`,
`policy.pretrained_path`) :

| Config | Run produit | Pas | Corpus | Reprise de |
|---|---|---|---|---|
| `train_config_propre_std.json` | `essai_std_20260907` | 1 500 | `apple_propre_2cam_128` | — |
| `train_config_bench.json` | `bench_bf16` | 80 | `apple_propre_2cam_128` | — |
| `train_config_long.json` | `propre_long_20260908` | 45 000 | `apple_propre_2cam_128` | — |
| `train_config_cooldown.json` | `propre_long_cooldown` | 1 200 | `apple_propre_2cam_128` | `propre_long_20260908/036000` |
| `train_config_pince.json` | `pince_const_20260909` (LR constant) | 36 000 | `apple_pince_2cam_128` | — |
| `train_config_pince_cooldown.json` | `pince_cooldown_20260909` (recuit du 36 000) | 4 000 | `apple_pince_2cam_128` | `pince_const_20260909/036000` |
| `train_config_cooldown_0NN000.json` (6, 12, 18, 24, 30) | `pince_cooldown_0NN000` (recuits) | 4 000 | `apple_pince_2cam_128` | `pince_const_20260909/0NN000` |

Scripts d'enchaînement :

- `plan_263M_20260907.sh` : enchaînement automatique du 263M sur le corpus propre.
- `plan_cooldown_20260908.sh` : arrêt de l'entraînement long à 18 h puis recuit.
- `lance_pince.sh` : lancement de `pince_const_20260909`.
- `enchaine_pince.sh` : attend la fin de la phase à LR constant, puis lance le recuit.
- `chaine_cooldowns.sh` : recuit de chaque checkpoint de la phase à LR constant, du plus
  avancé au moins avancé.

Les recuits 36 000, 30 000 et 24 000 et `propre_long_20260908/036000` ont été essayés sur
le bras réel dans la nuit du 2026-09-12 au 13 (EXP-20260913-NM-001, NIC_forge, en revue).

## Banc « Can occluded » (simulation, 2026-09-11)

`run_keypoints.sh`, `run_kpamp.sh`, `run_oracle.sh`, `run_small.sh`, `run_clair.sh` :
balayages sur le banc d'occlusion de la canette (`src/can_occlusion.py`), entraînés avec
`experiments/lift/50_train_valloss.py`. Sorties dans `results/runs/can/` (non versionnées).
