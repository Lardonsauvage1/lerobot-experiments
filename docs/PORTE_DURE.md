# Porte dure : n'encoder qu'une caméra sur deux, sans perdre en réussite

*Campagne du 17/09/2026 — tâche Robomimic Can, 500 rollouts par bras, états figés partagés.*

## La question

Le volet caméra avait établi qu'ajouter une vue poignet **dégrade** la policy à toute capacité
(agentview seule 67 % contre 42 % avec le poignet), et que le protocole à **encodeur gelé** répare
cela : on part d'un mono-caméra qui marche, on l'insère dans l'architecture bi-caméra, on gèle
l'encodeur de la vue globale, et le poignet ne peut plus que compléter — **86,2 %** sur quatre graines.

Restait le coût. Chaque caméra traîne son propre encodeur de **11,20 M de paramètres sur 27,79 M**,
déroulé à chaque pas de contrôle. Réduire le U-Net ne rapportait que 0,459 M — le décodeur n'est pas
compressible. **L'encodeur, lui, l'est : si on savait quelle vue est inutile à cet instant, on
pourrait ne pas la calculer.**

## Le mécanisme

Une porte lit les features du pas **t−1** — déjà calculées — plus la proprioception, et choisit **une
seule** caméra à encoder au pas t. La vue écartée n'est pas encodée du tout : on rejoue ses features
de t−1. C'est le « reuses historical information » de HCMS (arXiv 2104.09760), et cela respecte les
deux contraintes sans lesquelles rien n'est déployable :

- **causalité** — on ne peut pas encoder une image pour décider si on va l'encoder ; la décision ne
  lit donc que du passé déjà disponible ;
- **entrées réelles** — features et proprioception, rien de la géométrie de la simulation. C'est la
  leçon de l'oracle 12D : une méthode validée avec une béquille ne transfère pas.

Entraînement : Gumbel-softmax `hard=True` (sélection binaire en avant, gradient en arrière), plus la
perte d'équilibrage de charge qui interdit non pas une porte tranchée, mais une porte **toujours
tranchée du même côté**. Activation : `HARDGATE=1` dans `experiments/lift/50_train_valloss.py`.

## Résultats

| configuration | réussite @500 | passes d'encodeur |
|---|---|---|
| encodeur gelé, **sans porte** (4 graines) | **86,2 %** *(83,0 – 91,2)* | 100 % |
| **porte dure entraînée** (2 graines) | **83,4 %** *(82,2 · 84,6)* | **75 %** |
| porte souple K, durcie après coup | 83,6 % | 75 % |
| porte souple K | 81,6 % | 100 % |

**Comparaison de groupes** (un entraînement = une observation, seul test qui sépare l'effet du réglage
du hasard d'entraînement) : **−2,8 pts, Welch t = −1,27, df = 3,9, p = 0,275**. L'écart n'est pas
démontré. Le test n'a pas non plus la puissance de garantir l'absence totale de coût : à 2 graines
contre 4, il faudrait ~5 pts pour trancher.

Durcissement **sans réentraînement**, apparié sur les mêmes 500 états (McNemar exact) : **+2,0 pts,
p = 0,348** — et 92 épisodes sur 500 changent d'issue pour un écart net de 10. C'est du brassage,
pas un gain.

## Ce que la porte a appris

| phase | porte dure **entraînée** | porte K **durcie après coup** |
|---|---|---|
| APPROCHE (29 % des pas) | vue globale **75 %** | vue globale 30 % |
| SAISIE (45 %) | vue globale **19 %** | vue globale 82 % |
| TRANSPORT (26 %) | vue globale 26 % | vue globale 65 % |

Profil temporel de la porte entraînée : **monotone, 0,90 → 0,05**. Celui de K durci : en dents de
scie, sans direction.

**La porte regarde de loin pour s'approcher, puis bascule sur le poignet pour saisir et transporter.**
Elle a retrouvé seule, depuis le seul signal de réussite, la conclusion que la campagne caméra avait
mis trois semaines à extraire — le poignet ne sert qu'en manipulation fine et au contact. Elle ne
clignote pas (une bascule tous les 57 pas) et elle tranche vraiment (écarts-types de 0,39-0,44 sur des
moyennes de 0,19 et 0,75 : une porte à 0 ou à 1 selon la phase, pas une hésitation autour de 0,5).
**Politique remarquablement reproductible** — deux entraînements indépendants :

| | graine 42 | graine 43 |
|---|---|---|
| APPROCHE, vue globale | 0,750 | 0,731 |
| SAISIE, vue globale | 0,192 | 0,240 |
| TRANSPORT, vue globale | 0,258 | 0,195 |
| bascule tous les | 57 pas | 58 pas |
| vue globale dominante | 37 % | 38 % |

Profils temporels superposables (0,90 → 0,05 et 0,87 → 0,05). Sur un montage où deux entraînements
identiques peuvent différer de **7,4 points de réussite**, cette stabilité dit que la politique de
routage n'est pas un artefact d'une graine : c'est une propriété de la tâche que le mécanisme
retrouve à chaque fois.

## ⚠️ La limite, qui est le vrai enseignement

**K durci, dont la politique est l'exacte inverse et incohérente, réussit 83,6 % — mieux que les deux
modèles à politique sensée.** Un routage absurde n'échoue pas.

Donc **l'économie est bon marché parce que les deux vues sont redondantes à un pas d'intervalle, pas
parce que la porte choisit bien**. À ce régime, mécanisme appris et tirage arbitraire sont
indiscernables. La politique par phase reste la preuve que le mécanisme apprend quelque chose de vrai
sur la tâche, mais elle ne se paie pas encore en réussite.

**Suite à tester** : allonger la durée pendant laquelle une vue reste coupée (k pas d'affilée, ou des
segments entiers). Le retard s'accumule, la redondance ne suffit plus, et la qualité du choix devrait
enfin compter — c'est là que la politique par phase devrait payer.

## Portée pratique

25 % des passes d'encodeur supprimées sans perte démontrable. Avec un cache entre pas de contrôle —
les features de t−1 sont déjà en mémoire au pas suivant au lieu d'être recalculées comme dans cette
implémentation — **l'économie monte à 50 % des passes pour exactement le même comportement**.

## Reproduire

```bash
# entraînement (atomman) — 2 graines
bash experiments/can/150_cam2_L_hardgate.sh
# évaluation 500 rollouts
python experiments/can/148_porte_dure.py --n 500 --mode entrainee --ckpt <ckpt> --tag <nom>
python experiments/can/149_analyse_porte_dure.py          # McNemar apparié dur vs doux
# diagnostic par phase
python experiments/can/147_routeur_par_phase.py --run cam2_L_hardgate_s42 --n 40 --dure
```
