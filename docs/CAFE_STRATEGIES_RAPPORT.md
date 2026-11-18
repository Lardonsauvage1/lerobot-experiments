# Varier les stratégies d'un geste dans les démos : compte rendu (18/11/2026)

## Question

Au réel, il est difficile de garder une stratégie unique d'une démo à l'autre : plusieurs opérateurs, des trajectoires et des orientations différentes. Mélanger plusieurs façons de faire le même geste dégrade-t-il l'apprentissage ? Et à partir de quelle part une stratégie mal apprise « empoisonne-t-elle » un jeu de démos ?

## Protocole

- **Geste** : fermer le tiroir, dans la tâche café (sim MimicGen durcie avec Roby 5 axes).
- **Simulation** : 10 Hz, latence de 1 à 2 décisions en RTC, image vieillie d'une décision. Variations de couleur (±0,1) et de lumière (±0,15). Biais articulaire de 0,06 rad, zone morte, jeu.
- **Stratégies** : 4 stratégies d'expert (S1 à S4), avec des trajectoires, des méthodes et des orientations différentes. L'expert réussit 100 % avec chacune.
- **Démos** : 80 par jeu, enregistrées sous biais (pose lue), avec les mêmes départs pour tous les jeux. Un contrôle qualité bloquant passe avant chaque entraînement.
- **Modèles** : Diffusion Policy, recette identique, 2 graines d'entraînement par jeu.
- **Évaluation** : 200 essais par modèle (2 graines d'éval × 100), avec observateurs parfaits (oracle) et la même limite de 13,4 s par essai. On donne la réussite au 1er essai et la réussite avec relances (jusqu'à 6 essais).
- **Rendu** : toutes les évals ont été refaites le 17/11 après la correction d'un défaut de rendu (voir la section Infrastructure). Aucun épisode n'a été perdu.

## Résultats

### Stratégies seules

| Jeu (80 démos) | 1er essai | Avec relances | Échecs (relances) |
|---|---|---|---|
| S1 | 185 (92,5 %) | 191 (95,5 %) | 9 |
| S2 | 197 (98,5 %) | 200 (100 %) | 0 |
| S3 | 83 (41,5 %) | 99 (49,5 %) | 101 |
| S4 | 200 (100 %) | 200 (100 %) | 0 |

S3 est réussie à 100 % par l'expert, mais le modèle ne l'apprend qu'à moitié.

### Mélanges, comparés à la prédiction pondérée

La prédiction pondérée est la moyenne des taux de chaque stratégie seule (avec relances), pondérée par la part de cette stratégie dans le jeu.

| Jeu | Composition (S1/S2/S3/S4) | 1er essai | Avec relances | Prédit | Écart |
|---|---|---|---|---|---|
| S1+S2 | 40/40/0/0 | 190 (95 %) | 195 (97,5 %) | 97,8 % | −0,2 |
| S1+S4 | 40/0/0/40 | 186 (93 %) | 194 (97 %) | 97,8 % | −0,8 |
| 0 % de S3 | 27/28/0/27 | 193 (96,5 %) | 195 (97,5 %) | 98,5 % | −1,0 |
| 10 % de S3 | 25/25/8/24 | 189 (94,5 %) | 194 (97 %) | 93,7 % | +3,3 |
| 25 % de S3 | 20/20/20/20 | 169 (84,5 %) | 177 (88,5 %) | 86,2 % | +2,2 |
| 50 % de S3 | 13/13/40/14 | 166 (83 %) | 171 (85,5 %) | 74,0 % | **+11,5** |
| 40 démos saines du jeu à 50 % | 13/13/0/14 | 182 (91 %) | 189 (94,5 %) | 98,5 % | −4,0 |

La composition est comptée sur les chaînes générées. Le jeu à 0 % compte 82 chaînes pour 80 démos gardées ; l'écart de 2 n'a pas été vérifié.

### Contrôle : poison ou manque de démos ?

Dans le jeu à 50 %, chaque bonne stratégie n'a qu'environ 13 démos. Pour séparer les deux effets, on a entraîné un modèle sur les 40 démos S1/S2/S4 de ce jeu seules : mêmes démos, mêmes départs, même nombre de pas.

| Jeu | Échecs / 200 |
|---|---|
| 40 démos S1/S2/S4 | 11 |
| Les mêmes 40 + 40 démos S3 | 29 |
| 80 démos S1/S2/S4 | 5 |

Ajouter les démos d'une stratégie mal apprise fait baisser le taux, comme le prévoit la pondération. Doubler le nombre de bonnes démos ne gagne que 6 échecs.

## Conclusion

**Varier les méthodes ne change rien en soi.** Mélanger deux ou trois stratégies que le modèle apprend bien donne le même résultat qu'une stratégie seule : 97 à 97,5 % contre 95,5 à 100 %.

**Le taux global est à peu près prévisible.** C'est la moyenne des taux de réussite de chaque méthode, pondérée par sa part dans le jeu de démos. La prédiction tient à ±4 points sur 6 jeux sur 7.

**Mécanisme, observé par Sam sur les vidéos** : le modèle ne fait pas la moyenne des comportements. Il en choisit un, apparemment au hasard mais en proportion de ce qu'il a vu dans le jeu, puis l'applique jusqu'au bout. C'est ce qui rend le résultat prévisible : une stratégie présente à x % est jouée environ x % du temps, avec son propre taux de réussite.

Conséquence pratique : ce n'est pas la variété des démos qui compte, mais la part des stratégies difficiles à apprendre. Au réel, il faut repérer et limiter les façons de faire que le modèle n'apprend pas bien. Dans cet essai, 10 % de la mauvaise stratégie ne coûte rien de mesurable.

## Limites

- **À 50 % de S3, le modèle fait nettement mieux que la prédiction** : 29 échecs contre 52 prédits. À cette proportion, il semble choisir S3 moins souvent que sa part dans le jeu, ou mieux l'exécuter. La règle proportionnelle est donc une bonne approximation, pas une loi exacte.
- **Le choix d'une stratégie par épisode a été observé sur des vidéos**, quelques épisodes par modèle. Il n'a pas été mesuré sur les 200 essais. Pour le mesurer, il faudrait classer automatiquement la stratégie jouée à partir de la trajectoire.
- **Portée de l'essai** : un seul geste (fermer le tiroir), 80 démos, 2 graines d'entraînement par jeu. Avec ce bruit, un écart de moins de 8 points sur 200 essais n'est pas significatif.
- **Rejeu non reproductible** : rejouer un épisode ne reproduit pas exactement son issue (la sim n'est pas déterministe d'une exécution à l'autre). Les vidéos sont des exemples, pas les épisodes exacts du tableau.

## Infrastructure : défaut de rendu corrigé pendant l'essai

Des images vides (« rendus perdus ») faussaient une partie des évals. La cause : quand le ramasse-miettes détruisait un ancien contexte de rendu robosuite, sa libération (`mjr_freeContext`) s'exécutait dans le contexte GL en service et effaçait ses framebuffers.

Le correctif libère chaque contexte dans son propre contexte, puis rend le contexte en service courant avant chaque rendu. Il est activé par `ORDO_RENDU_PROPRE=1` dans `experiments/cafe/22_eval_ordo_pret_cafe.py`. Sur la scène café sans modèle, on passe de 376 images vides sur 1 500 à 0 sur 6 000. Toutes les évals de ce rapport ont été refaites avec le correctif.

## Fichiers

- Expert et stratégies : `src/cafe/expert.py` (`CAFE_TIROIR_STRAT=1..4` ou `mix:a,b,…`).
- Évaluation : `experiments/cafe/22_eval_ordo_pret_cafe.py`.
- Sous-jeu de démos : `experiments/cafe/49_sous_jeu_strat.py`, chaîné par `experiments/cafe/50_fs50v_gb10.sh`.
- Vidéos (sur la machine de Sam) : `results/videos/cafe_strat/` (mosaïque des 8 modèles ; réussites, relances et échecs du jeu à 50 %).
- Journal détaillé : `docs/CAFE_CADRE.md`.
