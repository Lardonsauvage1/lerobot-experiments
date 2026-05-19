# lift/43_lift_frozen_bn_stable — ResNet18(trainable, BN gelée, lr=1e-05) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, head_lr=0.001)

## Expérience
ResNet18 fine-tuné (lr=1e-5) + anti-overfit (DO 0.4, LN, AdamW, LS 0.1). LeRobot-style.

## Modèle
Type : ResNet18(trainable, BN gelée, lr=1e-05) + MLP 531 → 256 → 256 (LN,DO=0.4) → CE+résidu (LS=0.1, head_lr=0.001)
Paramètres : 12,881,080
Taille : 49.14 Mo

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
Loss train finale : 1.142109
Loss test finale  : 1.491170
Meilleure test    : 1.373576
Temps entraînement: 3153.3s
Temps/epoch       : 28.29s
Inférence         : 3.867940902709961ms (0.193ms/frame)
FLOPs/inférence   : 25,762,160
FLOPs/frame       : 1,288,108
Success rate      : 10%
Coverage moyen    : 83.3%
Reward moyen      : 1.83

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
