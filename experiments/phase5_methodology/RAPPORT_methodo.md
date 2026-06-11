# Phase 5 — Compte rendu méthodologie (Can) — BROUILLON

> Étude méthodologique sur tâche dure (Can, ResNet34 + birdview) : **comment évaluer/converger proprement** quand le succès par checkpoint est très bruité. Modèle dense sauvé tous les 100 steps jusqu'à 30k, puis prolongé (continue train) vers 50k.

## Convention statistique — Intervalle de confiance à 95 % (IC95 %)

On utilise l'**IC95 % de Wilson** partout (robuste près de 0/1, contrairement à Wald).

**Définition retenue (lecture « compatibilité ») :**
> L'**IC95 %** = l'ensemble des valeurs du paramètre **compatibles** avec les données de l'échantillon.
> Une valeur est « incompatible » si, pour cette valeur, la probabilité d'observer les données obtenues dans l'échantillon de l'étude est faible (→ le « 95 % » de l'intervalle de confiance).

Appliqué ici : le paramètre = la **vraie probabilité de succès** p (= le succès qu'on mesurerait avec un nombre **infini** de rollouts). Sur N rollouts (k succès), l'IC95 % encadre p.
- Largeur ∝ **1/√N** → pour diviser l'intervalle par 2, il faut **×4 rollouts**.
- À p≈30 % : ±12.7 pts (N=50) → **±4.0 pts (N=500)** → ±2.0 pts (N=2000).
- À p≈90 % (plateau) : **±2.6 pts (N=500)**.

## Ingrédients rassemblés (pour rédaction)

1. **Graphe** `courbes_methodo.png/pdf` — 2 panneaux : succès (50 vs 500 rollouts) + toutes les loss (train, val_loss live, val_loss full 1/5 seeds), complet jusqu'à 30k.
2. **Succès 50 vs 500 rollouts** : à 50 ép. l'IC95 vaut ±~12 pts (mesure inutilisable pour choisir un checkpoint) ; à 500, ±~4 pts (fiable). Données : `rollouts_500.csv` (03), `rollouts_50.csv` (04).
3. **Découplage loss ↔ succès** : train/val loss au plancher dès ~5k alors que le succès continue de monter jusqu'à ~25k → **la loss ne prédit pas le succès**. Formule de la loss = MSE de prédiction du bruit (DDPM, `prediction_type="epsilon"`) :
   L = E[ ‖ε − ε_θ(√ᾱ_t·a + √(1−ᾱ_t)·ε, t, o)‖² ] — erreur de débruitage 1-pas moyennée, ≠ succès (échantillonnage itératif en boucle fermée).
4. **Comparaison des mesures de val_loss** : live (1 batch → très bruitée, inutilisable) vs full 1 seed ≈ full 5 seeds (lisses) → 1 seed suffit. Données : `val_losses.csv` (05), `val_loss_1seed_fast.csv` (08).
5. **Variance de la mesure 500-rollouts** (`variance_500_seeds.csv`, 09) : checkpoint 10000, seeds de départs différents → seed0 30.4 % / seed1 30.6 % / seed2 (en cours). La variance seed-à-seed est **bien plus petite que l'IC95** → l'IC95 capture correctement l'incertitude.
6. **Variance intrinsèque du rollout-50** (`variance_pure.csv`, 06) : 5 répétitions sur checkpoints clés.
7. **Convergence** (à faire après le continue train) : éval 500-rollouts des checkpoints 30k→50k pour voir si le succès dépasse le plateau ~90 %.

## Constat fort — le succès n'est PAS lisse, et on ne peut pas interpoler

Deux phénomènes **distincts** se cumulent dans la dentelure des courbes :

**(a) Bruit de mesure** (n fini). Chaque point 50-rollouts a un IC95 (Wilson, n=50) de **±~10-13 pts** au centre. Vérification empirique : sur 30 checkpoints, **97 %** des mesures 500-rollouts (vérité fiable) tombent dans l'IC95 du point 50 correspondant → la couverture de l'IC95 est correcte, les 50-rollouts sont **non biaisés** (centrés sur la vérité), juste imprécis. Graphes : `ic95_50rollouts.png`, `ic95_50_vs_500.png`.

**(b) Instabilité RÉELLE du modèle** (au-delà du bruit). Entre checkpoints 500 **voisins** (1000 steps) : |Δsuccès| médian 3 pts mais **jusqu'à 28 pts**, et **10/29 écarts dépassent 8 pts** (= > 2× l'IC95 de ±4) → ce sont de **vraies différences de modèle**. À 1000 steps d'écart le modèle peut être réellement bien meilleur ou bien pire.

**Conséquence — interpoler entre points 500 est INVALIDE.** La droite reliant deux points 500 consécutifs, évaluée aux steps intermédiaires, **sort de l'IC95(n=50) du point réel dans 28 % des cas** (vs ~5 % attendu si l'interpolation était valide). La courbe 500 paraît « plus lisse » seulement parce qu'elle est échantillonnée moins souvent (tous les 1000 steps) ; les points 50 (tous les 200) montrent que ça saute aussi entre eux. **Le trait qui relie les points est un artefact visuel, pas une trajectoire.**

→ **« Convergé » = une bande, pas une valeur** : à partir de ~20k, succès ∈ [~84 %, ~94 %] ; un checkpoint pris seul peut être à l'un ou l'autre bout.

## Zoom résolution 100 steps (10k-11k) — l'instabilité est intrinsèque

On a évalué **11 checkpoints tous les 100 steps** entre 10k et 11k, **chacun à 500 rollouts** (fiable ±4 pts). Résultat : le succès oscille de **24 % à 57 %** (amplitude **33 pts**, écart-type 11 pts), IC95 souvent disjoints entre voisins (ex. 10700=24 % → 10800=55 %, **+30 pts en 100 steps**). Données : `rollouts_fine_10_11k.csv`, graphe `courbes_fine_10_11k.png`.

**Découplage avec TOUS les signaux d'entraînement.** Dans cette même zone : la **train/val loss** est plate (~0.05), le **learning rate** est lisse (~7.5e-5, décroissance cosine), le **grad_norm** est plat (~0.62). Aucun ne bouge pendant que le succès zigzague de 33 pts. → L'instabilité n'est **pas** un artefact du LR ni de la dynamique de gradient : c'est une **sensibilité intrinsèque du succès** aux micro-changements des poids (à LR constant, les poids continuent de bouger via les gradients stochastiques × Adam, et le succès — fonction non-lisse des poids en boucle fermée — y est hyper-sensible). Graphe global 3 panneaux : `courbes_methodo.png`.

## Conclusion méthodo (à formaliser)
Sur tâche dure :
1. Évaluer le **succès**, pas la loss (elle est au plancher dès ~5k alors que le succès triple ensuite).
2. Avec **assez de rollouts** (500, pas 50) — l'IC95 de Wilson encadre alors correctement la vraie valeur (couverture ~95 % vérifiée).
3. **Évaluer le checkpoint EXACT** qu'on déploiera : pas d'interpolation, pas de confiance au voisinage (instabilité réelle jusqu'à 28 pts entre checkpoints voisins).
4. La val_loss full 1 seed suffit comme signal lisse secondaire (≈ 5 seeds), mais ne remplace pas le rollout.
