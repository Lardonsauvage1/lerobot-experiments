# Phase 5 — Confiance dans nos mesures

> **Question centrale** : nos métriques d'entraînement (`val_loss` train log, `val_loss` eval, courbes de loss) sont-elles fiables pour décider quand arrêter et quel checkpoint choisir, et si non, comment les améliorer ?

C'est une phase **méta** : on n'essaie pas de battre un score, on essaie de **savoir si on peut faire confiance à nos scores**.

## Sujet expérimental

**26 ResNet34 + birdview** entraîné **from scratch** sur Can :
- Vision : 2× ResNet34 séparés (~42 M)
- U-Net : `[64,128,256]` (~5 M)
- État : 9D proprio (vision pure)
- Caméras : agentview + birdview
- Dataset : `lerobot_can_ph_proprio_birdview` (200 démos ph, 150 train / 50 val)
- Total ≈ 48 M params

**Pourquoi ce modèle** : c'est celui qui a montré le plus gros écart 10k vs 20k (+28 pt), donc le plus pédagogique pour étudier la trajectoire complète d'apprentissage.

## Protocole de mesure

### Plage et résolution
- **Train 30 000 steps** (suffisant pour dépasser convergence supposée à ~25k)
- **Save toutes les 100 steps** = **300 checkpoints**

### Mesures à chaque step de log (toutes les 100 steps)

#### Cheap (pendant le train, inline)
| Mesure | Définition | Coût |
|---|---|---|
| `train_loss` | loss du batch courant | 0 (déjà calculé) |
| `train_loss_smooth` | moyenne mobile fenêtre 100 | 0 |
| `val_loss_live_1batch` | 1 batch random de val | ~1 s |
| `val_loss_full_1seed` | moyenne sur les 50 ép. complètes (1 seed) | ~5 s |
| `val_loss_full_5seeds` | mean ± std sur 5 dataloader seeds différents (tous les 500 steps) | ~25 s |

**Coût total inline** : ~30 sec × 300 = 2.5 h cumulés. Acceptable.

#### Cher (eval worker, après train)
| Mesure | Définition | Fréquence | Coût total |
|---|---|---|---|
| `rollout_50` | succès sur les 50 ép. val | tous les 200 steps (= 150 fois) | ~12.5 h |
| `rollout_500` | succès sur 500 init figés (`can_eval500.npy`) | tous les 1 000 steps (= 30 fois) | ~25 h |
| `rollout_50_variance` | rollout 50 répété 5× sur le MÊME ckpt | 5 ckpts clés (5k, 10k, 15k, 20k, 30k) | ~2 h |

**Coût total worker** : ~40 h Mac MPS sequentiel.

### Total compute estimé
| | Heures |
|---|---|
| Train + mesures cheap | ~12-15 h |
| Worker rollouts | ~40 h |
| **Total** | **~52-55 h** Mac MPS (~2.5 jours autopilote) |

### Stockage disque
- 300 ckpts × ~200 MB = **~60 GB** sur disque (272 GB dispo, OK)

## Architecture du code

```
experiments/phase5_methodology/
├── README.md                    # Ce protocole (copie)
├── 01_train_dense.py            # Train modifié + cheap measures inline
├── 02_eval_worker.py            # Worker séquentiel sur ckpts sauvés
├── 03_variance_pure.py          # Mesures de variance répétée sur ckpts clés
└── 04_analyse.py                # Corrélation, variance, heuristiques sélection

results/runs/phase5_methodology/
└── 26_resnet34_dense/
    ├── checkpoints/             # 300 ckpts (steps 100..30000)
    ├── losses.csv               # mesures cheap par step
    ├── rollouts_50.csv          # par ckpt × 200
    ├── rollouts_500.csv         # par ckpt × 1000
    ├── variance_pure.csv        # 5 ckpts clés × 5 mesures
    └── train_config.json
```

## Analyses finales (à produire dans `04_analyse.py`)

### Q1 — Variance pure de chaque métrique
- **`val_loss_live_1batch`** : distribution de N=300 mesures sur la trajectoire (proxy variance d'un signal aléatoire)
- **`val_loss_full_5seeds`** : `std` à chaque step = variance entre seeds → mesure réelle
- **`rollout_50_variance`** : sur 5 ckpts, comparer succès entre les 5 répétitions → IC95 attendu vs observé

### Q2 — Corrélation val_loss ↔ succès rollout
- Pearson et Spearman entre :
  - `val_loss_live_1batch` ↔ `rollout_50` (regard du chercheur en live)
  - `val_loss_full_1seed` ↔ `rollout_50` (mesure standard eval)
  - `val_loss_full_5seeds_mean` ↔ `rollout_500` (le plus rigoureux)
- Par régime de train : début (0-10k), milieu (10k-20k), fin (20k-30k)

### Q3 — Heuristiques de sélection de checkpoint
Comparer rétrospectivement quelle heuristique aurait choisi le **vrai** meilleur ckpt (= celui avec le meilleur rollout 500) :
1. `best_step = argmin(val_loss_live_1batch)` — heuristique actuelle bruitée
2. `best_step = argmin(val_loss_full_1seed)` — variante stable
3. `best_step = argmin(val_loss_full_5seeds_mean)` — gold standard cheap
4. `best_step = argmin(moyenne_mobile(val_loss, fenêtre=5))` — lisser le bruit
5. `best_step = argmax(rollout_50)` — proxy direct mais bruité (IC95 ±13 pt)
6. `best_step = argmax(rollout_500)` — vrai best (référence)

→ Mesurer l'**écart entre le best ckpt sélectionné par chaque heuristique et le best réel** (en succès % et en steps).

### Q4 — Convergence et règles d'arrêt
- À partir de quand `rollout_500` plateau ?
- `train_loss` ou `val_loss_full` permettent-ils de détecter ce plateau ?
- Quelle règle d'arrêt aurait été optimale ?

### Q5 — Caveat sur le sujet
- 26 ResNet34 est UN modèle. Les résultats généralisent-ils à un mini-CNN, à ResNet18, à un U-Net plus gros ?
- → Ce qu'on conclura sera **calibré sur ce sujet**, et on devra **être prudent** sur la généralisation.

## Sortie attendue

1. **`results/runs/phase5_methodology/analyses/`** : courbes (variance, corrélation, comparaison heuristiques), tableaux
2. **`docs/METHODOLOGIE.md`** : doc dédiée avec recommandations pour TOUTE la suite du projet
3. Mise à jour de `COMPRESSION.md` et futures docs avec un **caveat méthodologique standardisé**

## Suite

À l'issue de cette phase, on aura :
- Une **règle de sélection de checkpoint** consensuelle pour le reste du projet
- Une **règle d'arrêt d'entraînement** quantifiée
- Une **note précise** sur la fiabilité (et les pièges) de val_loss en BC
- Du matériel pour potentiellement écrire une réflexion plus large sur la méthodologie d'éval en BC

→ Permettra ensuite de **redémarrer une phase Can propre** (« phase 6 » ou simplement « Can v2 ») avec une méthodologie solide.
