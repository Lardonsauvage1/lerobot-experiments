# 08_pretrained_cnn — ResNet18 gelé + MLP [128, 64]

## Expérience
CNN pré-entraîné (ResNet18 gelé) + MLP sur PushT

## Modèle
Type : ResNet18 gelé + MLP [128, 64]
Paramètres : 11,226,306
Taille : 42.82 Mo

### Architecture détaillée
```
  cnn.backbone.0: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  cnn.backbone.4.0.conv1: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.4.0.conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.4.1.conv1: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.4.1.conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.5.0.conv1: Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  cnn.backbone.5.0.conv2: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.5.0.downsample.0: Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
  cnn.backbone.5.1.conv1: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.5.1.conv2: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.6.0.conv1: Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  cnn.backbone.6.0.conv2: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.6.0.downsample.0: Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
  cnn.backbone.6.1.conv1: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.6.1.conv2: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.7.0.conv1: Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  cnn.backbone.7.0.conv2: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.7.0.downsample.0: Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
  cnn.backbone.7.1.conv1: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.backbone.7.1.conv2: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  cnn.fc: Linear(in_features=512, out_features=64, bias=True)
  mlp.0: Linear(in_features=66, out_features=128, bias=True)
  mlp.2: Linear(in_features=128, out_features=64, bias=True)
  mlp.4: Linear(in_features=64, out_features=2, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4988 / test : 1247
Seed : 42

## Résultats
Loss train finale : 0.007959
Loss test finale  : 0.019966
Meilleure test    : 0.019142
Temps entraînement: 7369.5s
Temps/epoch       : 147.39s
Inférence         : 6.264ms (6.264ms/frame)
FLOPs/inférence   : 22,432,896
FLOPs/frame       : 22,432,896
Success rate      : 0%
Coverage moyen    : 5.1%
Reward moyen      : 14.63

## Paliers de performance
  loss_le_0.1 : epoch 1 (44.1s)
  loss_le_0.05 : epoch 2 (85.2s)
  loss_le_0.03 : epoch 5 (212.6s)
  loss_le_0.02 : epoch 30 (6865.6s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
