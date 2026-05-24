# Phase 1-2 — PushT : enquête « loss vs performance »

> Pousser un cube en 2D vers une cible (PushT). On y découvre que **une loss basse ≠ un bon robot**, et que **la formulation de la sortie** (régression → classification discrète) compte plus que les features. Meilleur résultat : 46.5 % coverage.
> Liens : **Phase 1-2 (ici)** · [Phase 3 ▶ LIFT](LIFT.md) · [index](README.md)

## Contexte

Premières ~28 expériences (MLP, RNN, Transformer, action chunking). Best : **46.5 % coverage** (run 24, Transformer 2L sur image + position). Trois constats gênants :
1. **Aucun succès** (`success_rate = 0` partout — coverage moyen ≥ 30 % mais tâche jamais finalisée à >95 %).
2. **Loss et coverage décorrélés** : des runs à très basse loss (MSE 0.0012) avaient le **pire** coverage (4.6 %).
3. **Hypothèse** : sur PushT (tâche multimodale — plusieurs actions valides au même état), la MSE pousse à prédire la **moyenne** des actions humaines → « moyenne morte » qui ne marche pour aucune solution.

La série 29-32 isole l'effet de la **formulation de la sortie**, à input et backbone constants (image seule, ResNet18 gelé + Transformer 2L).

## Le parcours (runs 08 → 32)

### Phase 1 — exploration : trouver une archi qui marche (runs ~08-28)

| Jalon | Coverage | Idée |
|---|---|---|
| MLP / RNN sur image + état | ~27 % | baselines |
| Action chunking (prédire N actions futures) | aide | run 09 |
| **Transformer 2L + features ResNet** (image + position) | **46.5 %** ⭐ | run 24 — best de la phase 1 |
| Entraînement long (10 000 epochs) | n'aide pas | run 26 (loss ↓ mais coverage stagne) |
| Historique de positions (hist 5/10) | aide peu | runs 27-28 |

Bilan phase 1 : **le Transformer + chunking bat largement les MLP** (l'archi compte plus que l'historique), mais **0 % de succès** partout — coverage ~46 % au mieux, la tâche n'est jamais bouclée. D'où l'enquête ciblée de la phase 2.

### Phase 2 — enquête « formulation de la sortie » (runs 29-32)

Input et backbone figés (image seule, ResNet18 gelé + Transformer 2L) ; **seules la dernière couche et la loss changent**.

| Run | Sortie | Coverage | Idée |
|---|---|---|---|
| 29 | MSE (régression) | 31.8 % | image seule, baseline régression |
| 30 | MDN (mélange gaussien K=5) | 40.9 % | capter la multimodalité par sampling |
| 31 | **CE + résidu, bins absolus** (k-means 64) | **44.9 %** | classer parmi 64 bins → décision tranchée |
| 32 | CE + résidu, bins de **delta** | 21.3 % | contrôle en vitesse plutôt qu'en position |

## Résultats

- **CE + résidu (run 31, 44.9 %) rattrape « image + position » (run 24, 46.5 %)** — sans donner la position en entrée. Le vrai obstacle n'était donc pas l'absence de position, mais le **mode collapse de la MSE**.
- **Premier succès du projet** : 3/200 épisodes à epoch 75 du run 31 (aucun run précédent n'avait passé `is_success`).
- Les bins de **delta** (run 32) **dégradent** (21.3 %) — résultat inattendu, voir Leçons.

## Leçons clés

1. **La MSE est piégée par le mode collapse** sur tâche multimodale — propriété mathématique, pas un bug : elle optimise la moyenne, le coverage récompense les modes décidés.
2. **La formulation de la sortie peut compenser une feature manquante** : CE discret (image seule) ≈ MSE (image + position).
3. **Corrélation loss↔coverage selon la formulation** : MSE décorrélée · MDN partielle (σ rétrécit sans gain) · **CE+résidu bien corrélée** (commettre une décision discrète aligne loss et perf).
4. **Le delta est conditionnel à l'état** : prédire « de combien bouger » exige de savoir « où je suis / ce que j'ai commandé » — info que le ResNet gelé n'extrait pas → le delta régresse malgré sa promesse théorique.

## Détails techniques

- **Bins absolus → saccadé** : prédiction de chunk parallèle (t=0 et t=1 indépendants) + bins traités comme catégories sans notion de distance → téléportations entre timesteps consécutifs.
- **« delta_0 fantôme »** (run 32) : le 1er delta d'un chunk mélange décision humaine et retard simulateur ; on ne cluster que les vrais deltas action→action.
- **Bruit d'éval** : ±3.7 pt à 50 ép., ±1.7 pt à 200 ép. → ne pas surinterpréter les petites différences.

*(Récit run-par-run détaillé + vidéos : dans l'historique git de ce fichier — ex-`JOURNEY.md`.)*

## Suite → [Phase 3 : Lift](LIFT.md)

PushT plafonne ~45 % sans succès fiable. On passe à **Robomimic Lift** (bras Panda 7-DoF, plus proche du bras réel) et à la **Diffusion Policy** — qui résout intrinsèquement le multimodal qu'on contournait ici à la main.
