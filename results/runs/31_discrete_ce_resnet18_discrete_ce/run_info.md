# 31_discrete_ce — ResNet18(gelé) → Transformer 2L → Discret(64 bins + résidu) chunk=20

## Expérience
BeT-simple : k-means N=64 train-only + CE sur bin + MSE sur résidu par bin. bin_centers en buffer. Best model sauvegardé.

## Modèle
Type : ResNet18(gelé) → Transformer 2L → Discret(64 bins + résidu) chunk=20
Paramètres : 181,376
Taille : 0.69 Mo

### Architecture détaillée
```
  obs_proj: Linear(in_features=512, out_features=64, bias=True)
  transformer_decoder.layers.0.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.0.linear2: Linear(in_features=256, out_features=64, bias=True)
  transformer_decoder.layers.1.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.1.linear2: Linear(in_features=256, out_features=64, bias=True)
  bin_head: Linear(in_features=64, out_features=64, bias=True)
  residual_head: Linear(in_features=64, out_features=128, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 1.131853
Loss test finale  : 1.211265
Meilleure test    : 1.211265
Temps entraînement: 1651.9s
Temps/epoch       : 9.17s
Inférence         : 0.9044694900512695ms (0.045ms/frame)
FLOPs/inférence   : 362,752
FLOPs/frame       : 18,137
Success rate      : 0%
Coverage moyen    : 35.0%
Reward moyen      : 28.21

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
