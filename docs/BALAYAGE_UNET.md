# Taille du décodeur : ce que dit le balayage

*Can, 500 rollouts par modèle, 16 modèles. Config cible : 2 caméras — vue extérieure que le bras
masque + caméra embarquée au poignet — protocole à encodeur gelé. Tout figé sauf `down_dims` :
ResNet18, 96 px, 150 démos, batch 32, 10k pas + recuit 5k, LR constante 1e-4, EMA.*

> Le tableau de tous les leviers est dans `TABLEAU_LEVIERS.md`, **généré** par
> `experiments/can/160_tableau_leviers.py --md` — ne rien y écrire à la main, la régénération
> l'efface. Ce document-ci porte l'analyse.

## Le résultat, apparié proprement

Même graine, même batch, même budget, **poignet NEUF des deux côtés** :

| graine | `[64,128,256]` | `[128,256,512]` | échecs en moins |
|---|---|---|---|
| 42 | 79 | 31 | **−60,8 %** |
| 43 | 73 | 43 | **−41,1 %** |
| 44 | 76 | **27** | **−64,5 %** |
| **moyenne** | **76,0** | **33,7** | **−55,7 %** |

Groupe : `[73, 76, 79]` contre `[27, 31, 43]` — **séparation complète**, trois graines de chaque côté.

⚠️ **Et pourtant p = 0,10, non démontré.** Le plancher du test de permutation bilatéral vaut
`2 / C(2n, n)` : une séparation parfaite ne produit que deux arrangements extrêmes sur le total.

| répliques par groupe | arrangements | plancher |
|---|---|---|
| 2 contre 2 | 6 | 0,333 |
| 3 contre 3 | 20 | **0,100** |
| **4 contre 4** | **70** | **0,029** |
| 5 contre 5 | 252 | 0,008 |

**Il faut donc QUATRE répliques par groupe**, pas trois. Une affirmation antérieure de ce document
— « une 3e graine ferait tomber le plancher à 0,05 » — était **fausse** : le calcul avait été fait
de tête. La 3e graine renforce néanmoins le résultat au lieu de le diluer (l'estimation passe de
51,3 % à 55,7 %, et le nouveau bras à 27 échecs est le meilleur des six).

⚠️ **Le verdict intermédiaire (p = 0,042) était biaisé** : le groupe de référence mélangeait deux
initialisations du poignet (I/I2/I3 = **repris**, J = **neuf**) alors que le balayage est tout en
**neuf**. Il avait aussi fait croire à une hétérogénéité (−2 % à la graine 43) qui n'existe pas :
la référence utilisée, I2, avait un poignet repris **et** était un tirage exceptionnellement bon.

## La taille a un OPTIMUM, elle ne se maximise pas

Échecs sur 500, deux graines par case :

| décodeur | mono | bi-caméra gelé |
|---|---|---|
| `[64,128,256]` | 107,5 *(98·117)* | 76,0 *(79·73·76, homogène)* |
| **`[128,256,512]`** | **49,5** *(44·55)* | **37,0** *(31·43)* |
| `[256,512,1024]` | 68,0 *(70·66)* | 44,5 *(44·45)* |

Le **second** doublement rajoute des échecs : +37 % en mono, +20 % en bi-caméra, sans recouvrement
entre graines dans les deux cas. Les trois tailles voient le **même budget** (320k échantillons)
pour ~28, 41 et 90 M de paramètres. La recette standard à 96 % emploie un décodeur encore plus
large — mais avec **2,56 M d'échantillons, 8× plus**.

**→ À budget fixé, la taille du décodeur s'accorde au budget ; la pousser au-delà coûte.**

## ⭐⭐ Le plancher de bruit du projet était surestimé

| groupe de référence | moyenne | écart-type | étendue |
|---|---|---|---|
| variantes **mélangées** | 86,2 % | 3,62 pts | 8,2 pts |
| **strictement homogène** | 84,8 % | **0,60 pts** | **1,2 pts** |

84,2 · 85,4 · 84,8 % sur trois graines. **L'essentiel des « 4 à 8 points de bruit inter-run » qui
bornent les conclusions de ce dépôt venait du MÉLANGE DE VARIANTES, pas du hasard d'entraînement.**
À configuration réellement constante, trois entraînements tiennent dans 1,2 point — moins que le
bruit d'échantillonnage attendu (1,61 pt à 85 %).

⚠️ Les 7,4 points entre deux runs « identiques » mesurés en juin restent réels, mais à **58 % de
réussite** (variance binomiale maximale) sur une configuration **instable** (poignet en
concaténation naïve). Le bruit dépend du **niveau** ET de la **stabilité de la configuration** —
ce n'est pas une constante du projet, il faut le recalculer.

## ⭐ Levier découvert au passage : l'initialisation du poignet

| graine | poignet **repris** | poignet **neuf** | coût du neuf |
|---|---|---|---|
| 42 | 69 échecs | 79 | +14,5 % |
| 43 | 44 échecs | 73 | **+65,9 %** (McNemar p = 0,0015) |

Reprendre l'encodeur du poignet d'un modèle antérieur **aide**, et l'ancienne référence en
bénéficiait sans que ce soit mesuré ni voulu.

## Reproduire

```bash
bash experiments/can/158_balayage_unet.sh   # balayage (atomman)
bash experiments/can/162_ref_fresh.sh       # références homogènes, poignet neuf
python experiments/can/159_eval_unet.py --ckpt <ckpt> --tag <nom> --n 500   # bi-caméra
python experiments/can/161_eval_mono.py --ckpt <ckpt> --tag <nom> --n 500   # mono
python experiments/can/160_tableau_leviers.py --md                          # tableau
```
