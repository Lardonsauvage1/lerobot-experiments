# Temps d'entraînement — modèle calibré (Diffusion Policy, Mac M1)

But : prédire le temps d'entraînement en fonction de l'architecture et savoir
quand on tape la **falaise mémoire**. Calibré sur des mesures réelles (tâche Can,
Diffusion Policy, images 96², batch 32, M1 8 cœurs GPU / 16 Go RAM unifiée).

## Méthode de mesure (importante)

Les logs séparent `updt_s` (calcul GPU pur du step) et `data_s` (chargement).

- On utilise **`updt_s`**, pas le temps mur (barre de progression) : `updt_s` est
  **immunisé contre les pauses PC** (machine éteinte / veille → trou dans le temps
  mur, mais pas dans la mesure du step exécuté).
- On prend le **p10** (10ᵉ percentile) et non la médiane : il neutralise les steps
  gonflés par un **PC non dédié** (autres apps). Signature de contention = p75 ≫ p10.
- `data_s` ≈ 1 % du step **tant que le dataset tient en RAM** (cache). Voir falaise.

## Table calibrée (p10 `updt_s`, batch 32, 96², M1)

| Architecture | runs (Can) | p10 calcul pur |
|---|---|---|
| 1 cam · R18 · U-Net[64,128,256] | 02, 15 | **0,48 s** |
| 2 cam **partagé** · R18 · [64,128,256] | 06 | 0,85 s |
| 2 cam **séparé** · R18 · [64,128,256] | 08, 13, 16 | **1,13 s** |
| 2 cam séparé · R18 · **[128,256,512]** | 14, 25 | **1,23 s** |
| 2 cam séparé · **R34** · [64,128,256] | 26 | **1,66 s** |
| 2 cam séparé · R18 · [64] · **résolution 160²** | 23 | **2,91 s** |

Des runs indépendants de même archi retombent au même p10 (0,48/0,47 ;
1,13/1,15/1,13 ; 1,22/1,25) → le p10 capture bien le coût réel.

## Coefficients (effet de chaque levier sur `t_step`)

| Levier | Mesuré | Note |
|---|---|---|
| 1 → 2 caméras (séparé) | **×2,35** | un peu sur-linéaire (la conditioning grossit) |
| Encodeur **partagé** vs séparé | **×0,75** | partagé = ~25 % moins cher en calcul aussi |
| U-Net [64] → [128,256,512] (×3) | **+9 %** | quasi gratuit (conv 1D, horizon court) |
| Backbone **R18 → R34** | **×1,47** | ≈ 2× FLOPs vision, dilué par le reste |
| **Résolution 96 → 160** | **×2,57** | loi `∝ res²` (prédit ×2,78) ✅ |
| n_obs_steps | ∝ n_obs | empile n images à encoder |
| Batch | ∝ B par step | mais ~constant par époque |
| Nombre d'exemples | **0** sur `t_step` | fixe N_steps/époque (sauf si dépasse le cache RAM) |

**Enseignement clé : le temps suit les FLOPs de la VISION, pas le compte de
paramètres.** Le run 25 (41 M) est plus rapide que le 16 (28 M) car ses params en
plus sont dans le U-Net (quasi gratuit). Ce qui coûte = conv 2D sur images
(nb caméras × backbone × résolution²).

## Formule

```
t_step ≈ FLOPs_par_step / débit_GPU_effectif          (+ data ≈ négligeable si ça tient en RAM)
FLOPs_par_step ≈ 3 × B × n_obs × [ n_cam × F_backbone(res) + F_unet ]
   3 = avant + arrière ;  n_cam × F_backbone = terme dominant ;  F_unet ≈ négligeable
F_backbone(res) ≈ F_backbone(224) × (res/224)²
   ResNet18 @224 ≈ 1,8 GFLOP ;  ResNet34 @224 ≈ 3,6 GFLOP
t_total ≈ N_steps × t_step   (+ ~15 % pour val-loss + sauvegardes checkpoints)
```

### Estimations M1 (plancher p10 × N, +15 % overhead)
- `02` (1 cam) → 20k : ~2,7 h
- `16` (2 cam birdview) → 20k : ~6,3 h
- `26` (R34) → 20k : ~9,2 h
- `birdview_160` → ~16 h **et** risque de falaise → à éviter sur M1

## ⚠️ La falaise mémoire (RAM unifiée 16 Go)

Deux régimes, pas un continuum :
- **Tient en RAM** : `t_step` lisse, formule ci-dessus valable.
- **Dépasse la RAM** : swap SSD → `t_step` ×50–400, en dents de scie. Falaise, pas pente.

Ce qui remplit les 16 Go (les poids ne sont PAS le terme dominant) :
```
RAM ≈ poids×(1 + 2 Adam)×4o   +   ACTIVATIONS (∝ batch × res² × canaux × profondeur)   +   cache dataset   +   OS
            ~petit                  ~LE GROS                                              ~petit ici
```

Preuve dans les mesures — les **deux** plus grosses empreintes sont les **seules** à exploser :

| Run | empreinte | max step |
|---|---|---|
| `23_birdview_160` (160² = 2,78× activations) | la pire | **1291 s** (×440) |
| `25_big` (gros U-Net) | grosse | 104 s (×74) |
| tous les autres | normale | ≤ 2,2 s |

→ Le vrai danger mémoire est la **résolution** (activations ∝ res²), plus encore que la
taille du U-Net. Si ça swappe : baisser **batch** puis **résolution** en premier, ou
passer sur une machine à VRAM dédiée (atomman).

## Prédire sur une autre machine
- Théorique : `t_autre ≈ t_M1 × (débit_M1 / débit_autre)`, mais l'efficacité varie
  énormément (MPS ~2-5 % du pic FP32 vs CUDA ~30-50 %) → le ratio TFLOPS brut
  **sur-estime** le M1. Les gros modèles vont bien plus vite sur CUDA que ce ratio ne dit.
- Fiable : **mesurer 100 steps** sur la machine cible, lire le `s/step` médian,
  `t_total = N_steps × s/step`. Les formules servent à prédire l'effet d'un changement
  d'**archi** sur une machine donnée, pas à comparer deux machines dans l'absolu.

---
*Calibré le 2026-06-22 à partir des logs `results/logs/can/run_*.log`.*
