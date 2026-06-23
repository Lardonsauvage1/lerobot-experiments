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

> 🔄 **Calibrage auto-extensible** : `experiments/can/41_calibrate_train_time.py` régénère `results/runs/phase5_methodology/train_time_calib.md` (p10 s/step par signature d'archi, **resserré à chaque nouveau run**). À relancer après chaque entraînement. La table ci-dessous est l'analyse figée ; le `.md` généré est la version à jour.

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
| Batch | ∝ B par step | **par époque : décroît** (amortit l'overhead fixe — voir § Optimisation) |
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

## Optimisation de la vitesse d'entraînement

Benchmark dédié (`experiments/can/34_bench_compile_mlx.py` : ResNet18 ×2 + tête 1D, sweep de batch, 4 conditions, même archi des deux côtés). On décompose `t_step = O + c·B` (**O** = overhead fixe par step, **c** = calcul par échantillon) :

| condition | O fixe | c /éch. | part fixe @B16 | verdict |
|---|---|---|---|---|
| PyTorch eager | 0,077 s | 7,9 ms | **38 %** | référence |
| PyTorch sans synchro `.item()` | 0,081 s | 8,4 ms | 38 % | **aucun gain** (piste morte) |
| `torch.compile` | — | — | — | **échoue sur MPS** (inductor casse sur ResNet) |
| MLX `mx.compile` | 0,045 s | 9,9 ms | 22 % | **match nul** (voir ci-dessous) |

À batch 16, le step est **38 % overhead fixe + 62 % calcul** (et non « overhead-dominé » : à cette échelle le calcul des convolutions domine).

**Trois leviers logiciels qui NE marchent PAS sur Mac :**
- `torch.compile` : cassé pour les ResNet sur MPS.
- retirer les synchros `.item()` par step : aucun gain (même un poil pire) → la boucle n'est pas bloquée par ces synchros.
- **MLX** : sa fusion réduit l'overhead (O 0,045 vs 0,077) **mais ses kernels de convolution sont plus lents** (c 9,9 vs 7,9 ms) → les deux s'annulent à B16, MLX devient plus lent à B32. Pas de raison de migrer.

**Le seul vrai levier sur Mac = le BATCH** (amortir l'overhead). Temps **par échantillon** = `O/B + c` :

| batch | temps/éch. | vs B16 |
|---|---|---|
| 16 | 0,0127 s | — |
| 32 | 0,0103 s | −19 % |
| 64 | 0,0091 s | −28 % |
| 128 | 0,0085 s | −33 % |
| ∞ (plancher = `c`) | 0,0079 s | **−38 % (max absolu)** |

> **Plafond de gain = la fraction d'overhead au batch courant (~38 % à B16).** On ne peut amortir que l'overhead : ×2 batch ≈ −19 %, ×4 ≈ −28 %.

**Grossir le batch fait-il converger en moins de steps ?** Oui, **sous la taille de batch critique (CBS)** : en dessous du CBS, doubler le batch (avec LR ajusté) **réduit proportionnellement les steps à résultat égal** — la trajectoire est préservée *en fonction des exemples vus* [McCandlish 2018 ; Goyal 2017]. Au-dessus → rendements décroissants. **On est très loin sous le CBS** (batch 16-64 ; et le CBS est *d'autant plus grand que le gradient est bruité* → notre tâche multimodale Can = gradient très bruité = CBS élevé). → le gain batch est **réel, borné par la falaise mémoire, pas par le CBS**. *Caveat Adam : ajuster le LR modestement (√ratio + warmup), pas de saut brutal.*

**Combiner « bons kernels + zéro overhead » ?** C'est exactement le rôle de `torch.compile` / CUDA-graphs (garder les kernels rapides ET fusionner les lancements) — mais **indisponible sur MPS**. Le « combo du pauvre » sur Mac = **le plus gros batch que la RAM permet** (on approche le plancher de calcul `c`).

**Conclusion pratique :**
- **petits modèles** (mini-CNN, ResNet18 simple) : pousser le batch au max sous 16 Go → **~20-35 %** de temps gagné.
- **gros modèles** (R34 + gros U-Net) : déjà **coincés à batch 16** par la mémoire → **~0 gain sur Mac** → les mettre sur **atomman/CUDA** (VRAM + `torch.compile`/graph fonctionnels).
- abandonner `torch.compile` / MLX / suppression-de-synchro sur Mac.

Sources : [McCandlish et al., *An Empirical Model of Large-Batch Training* (1812.06162)](https://arxiv.org/pdf/1812.06162) · [*Critical Batch Size Revisited* (2505.23971)](https://arxiv.org/abs/2505.23971) · [Smith et al., *Don't Decay the LR, Increase the Batch Size* (ICLR 2018)](https://openreview.net/pdf?id=B1Yy1BxCZ).

## Prédire sur une autre machine
- Théorique : `t_autre ≈ t_M1 × (débit_M1 / débit_autre)`, mais l'efficacité varie
  énormément (MPS ~2-5 % du pic FP32 vs CUDA ~30-50 %) → le ratio TFLOPS brut
  **sur-estime** le M1. Les gros modèles vont bien plus vite sur CUDA que ce ratio ne dit.
- Fiable : **mesurer 100 steps** sur la machine cible, lire le `s/step` médian,
  `t_total = N_steps × s/step`. Les formules servent à prédire l'effet d'un changement
  d'**archi** sur une machine donnée, pas à comparer deux machines dans l'absolu.

---
*Calibré le 2026-06-22 (table/coefficients) et 2026-06-23 (§ Optimisation : benchmark 4 conditions, batch critique). Logs `results/logs/can/run_*.log`, benchmark `run_34_benchmark.log`.*
