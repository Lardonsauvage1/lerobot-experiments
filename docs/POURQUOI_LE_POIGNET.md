# Pourquoi ajouter une caméra de poignet dégrade le modèle — et quand ça l'améliore

Question de départ : les implémentations officielles de Diffusion Policy utilisent la caméra
`robot0_eye_in_hand` sur Robomimic et obtiennent de meilleurs résultats. Chez nous, l'ajouter
faisait perdre jusqu'à 61 points. Pourquoi ?

## Les mesures

Toutes sur les mêmes 500 états initiaux figés, McNemar apparié. Plancher de bruit inter-run
mesuré sur 3 répliques identiques : **écart-type 6,5 pts, étendue 13 pts**.

| caméras | réussite |
|---|---|
| côté seule | 76,6 · 80,4 % · (96,0 % avec la recette standard) |
| côté + dessus | 80,0 · 82,0 % |
| **côté + poignet** | **54,2 · 61,6 · 48,6 %** |
| dessus seule | 20,4 % |
| **dessus + poignet** | **46,2 %** (+25,8 pts, p = 2·10⁻¹⁴) |
| poignet seul (entraîné seul) | 0,0 % |

Le poignet fait donc **perdre 20 points** à côté d'une bonne caméra fixe, et **gagner 26 points**
à côté d'une mauvaise. Ce n'est pas le capteur qui est bon ou mauvais : c'est le contexte.

## Le diagnostic — ablation de modalité

On rejoue les mêmes états avec le même modèle, en masquant une caméra (mise à zéro après
normalisation = image moyenne). Si masquer ne change rien, le modèle ne s'en servait pas.

| modèle | complet | sans la fixe | sans le poignet |
|---|---|---|---|
| côté + poignet | 60,5 % | 37,0 % (−23,5) | **0,0 %** (−60,5) |
| dessus + poignet | 48,5 % | 46,0 % (−2,5) | **0,0 %** (−48,5) |

## Ce que ça établit

**1. Le modèle devient totalement dépendant du poignet.** Dans les deux cas, le masquer ramène
à zéro. Alors qu'un modèle entraîné sur la caméra de côté seule atteint 78 %, le modèle
bi-caméra ne sait plus agir sans le poignet — il n'a jamais appris à s'en passer.

**2. Ce n'est pas « une caméra est ignorée ».** Dans le bras côté + poignet, retirer la vue de
côté coûte 23,5 points : elle est bel et bien utilisée. Le réseau exploite ses deux entrées et
fait pourtant moins bien que sa propre sous-architecture à une caméra — le cas décrit par
*What Makes Training Multi-modal Classification Networks Hard?* (Wang et al., CVPR 2020).

**3. La vue de dessus, elle, est bien ignorée** (−2,5 pts seulement). Le gain de 26 points ne
vient donc pas d'une complémentarité : le poignet **remplace** la vue de dessus. Le vecteur
d'état contenant la pose du préhenseur, poignet + proprioception suffit à se localiser.

## Le mécanisme

Le gros plan du poignet est **l'entrée la plus prédictive de l'action immédiate** : il encode
directement la position de la cible relativement à la pince. C'est le raccourci le plus facile,
donc celui que la descente de gradient s'approprie en premier. Le réseau bâtit une
représentation conjointe ancrée dessus, et ne développe jamais la capacité de s'appuyer sur la
vue globale seule.

Quand la vue globale était bonne, on a échangé une représentation robuste contre une
représentation fragile : **catastrophe**. Quand elle était mauvaise, le raccourci se trouvait
être aussi le meilleur signal disponible : **gain**.

## Pourquoi l'officiel y arrive et pas nous — hypothèse non testée

Notre configuration est identique à l'officielle sur quinze paramètres d'architecture
(horizon, n_obs_steps, ordonnanceur, prediction_type, clip_sample, kernel_size, n_groups,
GroupNorm, 32 keypoints, ResNet18, lr, weight_decay, betas…). Le crop est correctement géré
par LeRobot — aléatoire à l'entraînement, centré à l'évaluation.

Il reste **deux** écarts :

| | officiel | nous |
|---|---|---|
| budget d'entraînement | 3050 époques ≈ **70,8 M échantillons** | 2,56 M au mieux — **÷28** |
| pas de débruitage à l'éval | 100 | 10 |

L'hypothèse principale est le budget : une branche visuelle supplémentaire, c'est 11 M de
paramètres de plus à faire converger, et rien ne garantit qu'elle y arrive dans le budget qui
suffit à une seule caméra. On comparerait alors un modèle convergé à un modèle inachevé.
**Égaler le budget officiel demanderait 307 h sur notre matériel — infaisable.**

## Ce qu'on en retient pour le robot réel

Sur le robot, il n'y a pas de vue extérieure qui voit tout, et le bras masque la caméra fixe.
C'est **exactement** le régime « dessus + poignet », où le poignet a gagné 26 points. Le résultat
qui compte pour le déploiement est donc le positif, pas le négatif.

---

# Bilan de la campagne (2026-09-14 → 16)

**Vingt modèles, 500 rollouts chacun, états initiaux figés, comparaisons appariées.**

## La loi

| état de la caméra fixe | apport du poignet |
|---|---:|
| bonne (vue de côté) | **−24 pts** |
| dégradée par le crop | **+13 pts** |
| masquée par le bras (vue de dessus) | **+26 pts** |

**La valeur de la caméra embarquée est inversement proportionnelle à la qualité de la vue
globale.** Trois régimes, trois mesures indépendantes.

## Cinq remèdes essayés, aucun ne marche

| | réussite | vs référence 54,8 % |
|---|---:|---|
| dropout de caméra | 64,2 % | +9,4 — dans le bruit |
| tête auxiliaire par caméra | 45,8 % | −9,0 — dans le bruit |
| routeur de caméras (porte apprise) | 42,2 % | −12,6 — sous les 3 répliques |
| crop aléatoire | 34,8 % | **−20,0** ⚠️ voir ci-dessous |
| budget ×10 (100 000 steps) | 60,0 % | +5,2 — insuffisant |

## ⚠️ Nuance sur le crop — à ne pas généraliser

Le crop **n'est pas mauvais en soi**. Notre meilleur modèle Can, `wristcap84_A` à **96 %**,
en utilise un (76←84). Le bras G qui s'effondre à 21,6 % a le même type d'augmentation mais
**huit fois moins de budget** : 0,32 M échantillons contre 2,56 M, un U-Net dix fois plus
petit et 150 démonstrations au lieu de 196.

L'explication la plus économique est classique : **l'augmentation rend l'apprentissage plus
difficile et ne paie qu'avec assez d'entraînement.** À 10 000 steps le modèle subit le bruit
sans avoir le temps d'en tirer la robustesse.

Ce qui est donc établi ici : **ajouter du crop à un petit budget dégrade massivement.**
Ce qui ne l'est pas : que le crop soit inutile. Tester proprement demanderait de faire varier
le crop À budget constant et élevé — ce qu'on n'a pas fait.

## Trois enseignements qui dépassent le sujet

**1. Le plancher de bruit d'abord.** Trois entraînements STRICTEMENT identiques donnent
54,2 · 61,6 · 48,6 — **13 points d'étendue**. Sans cette mesure, trois des résultats
ci-dessus auraient été annoncés comme des effets. McNemar compare deux *modèles*, jamais
deux *configurations*.

**2. La val-loss et la boucle fermée pointent en sens opposés.** Le bras avec crop a la
**meilleure** val-loss du lot (0,0557) et la pire réussite (34,8 %). Le crop rend l'image
peu fiable comme repère de position absolue ; le réseau se rabat sur la proprioception, qui
prédit très bien l'action suivante dans une démonstration lisse et ne sert à rien pour se
corriger en boucle fermée.

**3. Un mécanisme peut être utilisé par le réseau et nuire quand même.** Le `gamma` du
routeur, initialisé à 0 dans une forme résiduelle, est monté à 0,45 : la descente de gradient
a *choisi* d'ouvrir la porte. Elle a réduit la perte d'entraînement et dégradé la réussite.

## Ce qu'on en retient pour le robot réel

Une caméra extérieure bien placée suffit — la deuxième n'apporte rien de démontrable
(+2,5 pts, p = 0,40). Mais **là où le bras masque la vue fixe, le poignet vaut 26 points**,
et c'est exactement la situation du robot. Aucune augmentation par crop, sous aucun prétexte.
