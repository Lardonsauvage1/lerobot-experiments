# lift/45_lift_dinov2 — DINOv2-small(gelé) + MLP 403 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, WD=0.0005)

## Expérience
Anti-overfit combo : Dropout 0.4 + LayerNorm + AdamW(wd=5e-4) + LS 0.1. Final eval 200 eps.

## Modèle
Type : DINOv2-small(gelé) + MLP 403 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, WD=0.0005)
Paramètres : 1,681,400
Taille : 6.41 Mo

### Architecture détaillée
```
  body.0: Linear(in_features=403, out_features=256, bias=True)
  body.4: Linear(in_features=256, out_features=256, bias=True)
  bin_head: Linear(in_features=256, out_features=2940, bias=True)
  residual_head: Linear(in_features=256, out_features=2940, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 1.256692
Loss test finale  : 1.387149
Meilleure test    : 1.360754
Temps entraînement: 439.2s
Temps/epoch       : 0.7s
Inférence         : 0.1811838150024414ms (0.009ms/frame)
FLOPs/inférence   : 3,362,800
FLOPs/frame       : 168,140
Success rate      : 32%
Coverage moyen    : 84.7%
Reward moyen      : 4.54

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
