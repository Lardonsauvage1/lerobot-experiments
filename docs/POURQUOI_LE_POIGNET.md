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

---

# ⭐ LA SOLUTION : encodeur gelé (2026-09-17)

Après **sept remèdes infructueux**, un protocole fonctionne — et il retourne complètement
la conclusion de cette enquête.

## Le protocole

1. Entraîner un modèle **mono-caméra** sur la vue globale seule (78,5 %).
2. L'insérer dans une architecture bi-caméra avec **les colonnes du poignet à zéro**
   (`141_init_from_mono.py`). Équivalence vérifiée : sortie du U-Net identique au bit près,
   et modèle totalement insensible à l'image du poignet.
3. **Geler** l'encodeur de la vue globale (`FREEZE_CAM0=1`, 11,2 M params sur 27,8).
4. N'entraîner que la branche poignet et le décodeur.

Le poignet ne peut alors plus **remplacer** la représentation qui fonctionne : il ne peut que
la **compléter**, ou rester muet.

## Le résultat

| configuration | moyenne | répliques |
|---|---:|---|
| **poignet + encodeur GELÉ** | **88,7 %** | 86,2 · 91,2 |
| caméra de côté seule | 78,5 % | 76,6 · 80,4 |
| poignet, entraînement conjoint | 54,8 % | 54,2 · 61,6 · 48,6 |

**+10,2 points**, séparation complète — les deux répliques gelées dépassent les deux
références, sans chevauchement. (Welch p = 0,091 à n = 2 par groupe : sous-puissance, pas
absence d'effet. Un test de rang ne pourrait pas descendre sous 0,167 à cet effectif.
Une 3ᵉ graine est en cours.)

## Ce que ça change

**Le problème n'a jamais été le capteur, c'était l'entraînement conjoint.** Laissé libre, le
poignet accapare la représentation — il est l'entrée la plus prédictive de l'action immédiate,
donc le raccourci que la descente de gradient préfère — et détruit ce qui marchait : −24 points.
Forcé de se greffer sur une représentation verrouillée, il ajoute +10 points.

Ça réconcilie enfin nos mesures avec la littérature : les systèmes publiés utilisent bien la
caméra de poignet, et l'écart venait de la **façon d'entraîner**, pas de l'architecture, ni du
budget, ni de l'augmentation.

## Les sept remèdes qui ont échoué avant

budget ×10 · dropout de caméra · têtes auxiliaires · routeur de caméras appris ·
crop aléatoire · épisodes de rattrapage · corrections seules.

Aucun ne s'attaquait à la cause : tous laissaient le poignet participer à la construction de
la représentation.

---

# Bilan final de la campagne (17 septembre 2026)

**23 modèles entraînés, 500 rollouts chacun, états initiaux figés.**

| configuration | réussite | |
|---|---:|---|
| **poignet sur encodeur GELÉ** | **86,2 %** | 4 bras : 86,2 · 91,2 · 83,0 · 84,2 |
| encodeur gelé + routeur | 83,2 % | le routeur n'ajoute rien |
| côté + dessus (2 cam. extérieures) | 81,0 % | 80,0 · 82,0 |
| caméra de côté seule | 78,5 % | 76,6 · 80,4 |
| + 200 démonstrations expertes | 79,4 % | le volume est neutre |
| + corrections scriptées | 72,0 % | |
| + rattrapages complets | 66,8 % | |
| poignet en entraînement CONJOINT | 54,8 % | 54,2 · 61,6 · 48,6 |
| + dropout de caméra | 64,2 % | |
| + têtes auxiliaires | 45,8 % | |
| + routeur appris | 42,2 % | |
| + crop aléatoire | 34,8 % | |

## Le résultat

Le même capteur passe de **−24 points** (conjoint) à **+7,7 points** (gelé). Rien d'autre ne
change : même architecture, mêmes données, même budget. Seule change la possibilité qu'a le
poignet de s'approprier la représentation visuelle.

(Welch p = 0,066 — la puissance est limitée par le groupe de CONTRÔLE, qui n'a que 2 répliques.
Les 4 bras gelés dépassent les 2 références sans chevauchement.)

## Le routeur : bonne stratégie, mauvaise représentation

Le diagnostic par phase (`147_routeur_par_phase.py`) montre que la porte apprise est **sensée** :

| phase | poids de la vue globale |
|---|---:|
| approche | 0,386 |
| **saisie** | **0,241** — le poignet domine, c'est la précision du dernier centimètre |
| **transport** | **0,556** — la vue globale domine, il faut trouver le bac |

Elle ne papillonne pas : une bascule tous les 84 pas. Et son profil temporel décroît
régulièrement, de 0,45 au départ à 0,07 à l'arrivée.

Pourtant ce bras est le **pire** de tous (42,2 %). La politique de routage était bonne ; c'est
la représentation qu'elle pilotait qui était détruite.

Et posé sur un encodeur gelé, le routeur devient **inutile** : son `gamma` tombe de 0,45 à
0,077 et le résultat (83,2 %) ne dépasse pas le gel seul. Une fois la représentation protégée,
le modèle n'a plus besoin d'arbitrage explicite.

## Ce qu'on retient pour le robot réel

1. **Une caméra fixe bien placée + son encodeur gelé + le poignet greffé dessus.** C'est la
   recette, et elle ne demande ni capteur ni budget supplémentaire.
2. **Ne jamais entraîner les deux caméras ensemble depuis zéro.**
3. **Les corrections scriptées n'aident pas** — mais des corrections *démontrées par un humain*
   restent non testées, et c'est le principe de HG-DAgger.
