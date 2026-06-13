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

## La loss au plancher ne veut PAS dire « convergé » — un mini-CNN passe de 2 % à 74 % après 20k

Expérience décisive (mini-CNN 1.84M params, vision 0.03M, Can vision pure). Deux runs entraînés à l'identique jusqu'à 20k (cosine vs LR constant), **loss train ET val au plancher ~0.05 dès ~5k**, succès **~2 %**. Conclusion naïve : « modèle trop faible, convergé à 2 % ». **FAUX.**

Le run cosine d'origine avait **annealé son LR à ~2e-9 (≈ 0) à 20k** → poids gelés, succès figé à 2 % alors que la loss *paraissait* parfaite. En prolongeant 20k→50k :
- **LR constant 1e-4** : succès **2 % → 74 %** (pic @46k ; plateau bruité 50-74 %). Données : `mini_constant_continue/rollouts_500.csv`.
- **cosine « vague » (warm restart SGDR, ~0→1e-4→0)** : 2 % → 46 % (sa 2ᵉ moitié re-décroît le LR vers 0 → re-affame le modèle). Données : `mini_cosine_continue/rollouts_500.csv`.

→ Le mini-CNN n'était **pas trop faible** (60× moins de params vision que le ResNet34 et il atteint ~74 % vs ~90 %) : il était **affamé de LR**. La loss était **aveugle à +72 points** de capacité réelle, et n'indiquait même pas qu'il restait tout ça à gagner. Graphe complet 1k→50k (4 panneaux) : `courbes_minicnn_full_1k_50k.png` ; comparatif 0→20k : `courbes_minicnn_cos_vs_const.png`.

**Le LR constant bat le warm-restart** ici : pour décoller d'un plancher dû à l'annealing, un LR soutenu non-nul est ce qui compte ; la vague aide mais sa redescente bride le gain.

## Zoom résolution 10 steps — le succès est rugueux JUSQU'EN BAS (pas d'échelle lisse)

Pour vérifier la variance sans ambiguïté de trajectoire : on a densifié une fenêtre de 200 pas **jamais entraînée auparavant** (premier passage, 1 ckpt **tous les 10 pas**, run constant 1e-4 à 57000→57200), chacun à **500 rollouts**. Données : `mini_constant_finevar/rollouts_500.csv`, graphe `courbe_finevar_57k.png`.

Résultat : succès **42.2 % → 75.4 %**, amplitude **33 pts**, σ 8 pts — **sur 200 pas seulement**. Sauts entre voisins **distants de 10 pas** : 57040=72.4 % → 57050=56.6 % (**−16 pts**), 57190=57.2 % → 57200=42.2 % (**−15 pts**). **8 écarts sur 20 ont des IC95 disjoints** (n=500, ±4 pts) → ce sont de **vraies différences de modèle**, pas du bruit.

**L'instabilité ne se lisse à aucune échelle.** À 100 pas (10k-11k) : 33 pts d'amplitude ; en zoomant ×10 (10 pas) : **encore ~33 pts**. **10 pas de gradient** (LR 1e-4) suffisent à bouger le succès de 15+ points. → Aucune confiance possible même à un voisin à 10 pas ; le seul succès connu est celui du checkpoint exactement évalué.

## Conclusion méthodo (à formaliser)
Sur tâche dure :
1. Évaluer le **succès**, pas la loss (elle est au plancher dès ~5k alors que le succès triple ensuite).
2. Avec **assez de rollouts** (500, pas 50) — l'IC95 de Wilson encadre alors correctement la vraie valeur (couverture ~95 % vérifiée).
3. **Évaluer le checkpoint EXACT** qu'on déploiera : pas d'interpolation, pas de confiance au voisinage — l'instabilité est réelle et **ne se lisse à aucune échelle** (33 pts d'amplitude aussi bien à 100 pas qu'à 10 pas).
4. La val_loss full 1 seed suffit comme signal lisse secondaire (≈ 5 seeds), mais ne remplace pas le rollout.
5. **Loss au plancher ≠ convergé.** Une loss plate ne dit ni le succès ni qu'il reste de la marge : un mini-CNN « loss parfaite, 2 % » cachait 72 points récupérables par simple LR soutenu. **Ne jamais arrêter un entraînement sur la loss.**
6. **Surveiller le LR de fin** : un scheduler qui anneal à ~0 gèle le modèle bien avant son potentiel. Sur tâche dure, un LR constant (ou un warm-restart) débloque le plancher.

## Sim-to-real — fragilité catastrophique au déplacement des caméras

**Question :** le modèle survit-il si les caméras ne sont pas exactement où elles étaient pendant la collecte des démos (cas du transfert vers un vrai robot) ?

**Protocole :** on réutilise un modèle 2-cams déjà entraîné (agentview + birdview + proprio), sans ré-entraîner. À chaque épisode, on décale **aléatoirement et indépendamment les deux caméras** (translation + rotation, direction aléatoire, amplitude ≤ niveau) ; la caméra reste fixe pendant l'épisode (= caméra re-fixée un peu de travers pour ce déploiement), différente à chaque épisode. Sweep de niveaux, 500 rollouts chacun. Scripts : `20_eval_camshift.py` (éval), `23_camshift_gallery.py` (visuel). Données : `camshift_46k.csv`, `camshift_45k.csv` ; graphes `courbe_camshift.png`, `camshift_gallery.png`.

**Résultat — effondrement immédiat :**

| décalage (aléatoire/épisode) | ckpt 46000 (baseline 74 %) | ckpt 45000 (baseline 56 %) |
|---|---|---|
| 0 (baseline)        | 70.2 % | 59.2 % |
| ≤2 cm / ≤2°         | **12.6 %** | **7.8 %** |
| ≤5 cm / ≤5°         | 3.0 %  | 2.8 % |
| ≤10 cm / ≤10°       | 2.2 %  | 1.0 % |

- **Un décalage de 2 cm + 2° fait chuter le succès de ~70 % à ~13 %**. À 5 cm/5°, le modèle est mort (~3 %). Il a **mémorisé le point de vue exact** : aucune tolérance.
- **Hypersensibilité à l'imperceptible** : sur la galerie, les vues à ≤2 cm/2° sont quasi identiques à l'œil humain — pourtant le succès s'est déjà effondré. Le modèle est perdu là où un humain ne voit presque rien.
- **Comparaison de 2 checkpoints voisins** (74 % vs 56 %) : le pic à 74 % garde un poil plus à 2 cm/2° (13 % vs 8 %) mais les deux s'effondrent pareil. L'écart d'instabilité entre checkpoints est **écrasé** par la catastrophe du déplacement caméra.

**Implication sim-to-real :** tel quel, cette politique exigerait de re-fixer les caméras au millimètre / sous-degré près de la pose de collecte — irréaliste sur un vrai robot. Levier à tester en priorité : **augmentation de caméra à l'entraînement** (poses jittées pendant l'apprentissage) puis re-mesure de cette même courbe de dégradation.
