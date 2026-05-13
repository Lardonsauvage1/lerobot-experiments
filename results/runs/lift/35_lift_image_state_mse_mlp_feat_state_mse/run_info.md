# lift/35_lift_image_state_mse — ResNet18(gelé) + MLP 531 → 256 → 256 → 140

## Expérience
Lift baseline image + state (équivalent run 24 PushT). MLP sur features ResNet pré-calculées.

## Modèle
Type : ResNet18(gelé) + MLP 531 → 256 → 256 → 140
Paramètres : 237,964
Taille : 0.91 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=531, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=256, bias=True)
  net.4: Linear(in_features=256, out_features=140, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 0.113868
Loss test finale  : 0.560396
Meilleure test    : 0.458478
Temps entraînement: 581.8s
Temps/epoch       : 0.13s
Inférence         : 0.05121469497680664ms (0.003ms/frame)
FLOPs/inférence   : 475,928
FLOPs/frame       : 23,796
Success rate      : 18%
Coverage moyen    : 83.7%
Reward moyen      : 2.08

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
