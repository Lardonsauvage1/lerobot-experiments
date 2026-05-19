# lift/40_lift_anti_overfit — ResNet18(gelé) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, WD=0.0005)

## Expérience
Anti-overfit combo : Dropout 0.4 + LayerNorm + AdamW(wd=5e-4) + LS 0.1. Final eval 200 eps.

## Modèle
Type : ResNet18(gelé) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, WD=0.0005)
Paramètres : 1,714,168
Taille : 6.54 Mo

### Architecture détaillée
```
  body.0: Linear(in_features=531, out_features=256, bias=True)
  body.4: Linear(in_features=256, out_features=256, bias=True)
  bin_head: Linear(in_features=256, out_features=2940, bias=True)
  residual_head: Linear(in_features=256, out_features=2940, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 1.271921
Loss test finale  : 1.384577
Meilleure test    : 1.370696
Temps entraînement: 387.3s
Temps/epoch       : 0.64s
Inférence         : 0.2133500576019287ms (0.011ms/frame)
FLOPs/inférence   : 3,428,336
FLOPs/frame       : 171,416
Success rate      : 35%
Coverage moyen    : 84.8%
Reward moyen      : 5.32

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
