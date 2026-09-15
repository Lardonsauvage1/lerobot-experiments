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
