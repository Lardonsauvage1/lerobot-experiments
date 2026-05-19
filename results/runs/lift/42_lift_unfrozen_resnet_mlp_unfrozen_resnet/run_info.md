# lift/42_lift_unfrozen_resnet — ResNet18(trainable, BN gelée, lr=1e-05) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, head_lr=0.001)

## Expérience
ResNet18 fine-tuné (lr=1e-5) + anti-overfit (DO 0.4, LN, AdamW, LS 0.1). LeRobot-style.

## Modèle
Type : ResNet18(trainable, BN gelée, lr=1e-05) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, head_lr=0.001)
Paramètres : 12,890,680
Taille : 49.17 Mo

### Architecture détaillée
```
  backbone.0: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  backbone.4.0.conv1: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.4.0.conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.4.1.conv1: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.4.1.conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.5.0.conv1: Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  backbone.5.0.conv2: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.5.0.downsample.0: Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
  backbone.5.1.conv1: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.5.1.conv2: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.6.0.conv1: Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  backbone.6.0.conv2: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.6.0.downsample.0: Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
  backbone.6.1.conv1: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.6.1.conv2: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.7.0.conv1: Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  backbone.7.0.conv2: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.7.0.downsample.0: Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
  backbone.7.1.conv1: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  backbone.7.1.conv2: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
Loss train finale : nan
Loss test finale  : nan
Meilleure test    : 1.386965
Temps entraînement: 2904.3s
Temps/epoch       : 25.72s
Inférence         : 5.302724838256836ms (0.265ms/frame)
FLOPs/inférence   : 25,781,360
FLOPs/frame       : 1,289,068
Success rate      : 0%
Coverage moyen    : 82.1%
Reward moyen      : 0.00

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
