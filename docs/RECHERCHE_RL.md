# Recherche — RL (renforcement) pour piloter le bras

> Notes de veille (juin 2026). Piste suggérée par un prof : utiliser du RL (« Q matrice ») mêlé à des réseaux, éventuellement modulaire. Contexte projet : on fait de l'**imitation** (diffusion policies, LeRobot) sur Robomimic Lift/Can ; objectif réel à terme. Ce fichier = pour reprendre le sujet plus tard.

---

## 1. Vocabulaire de base (RL vs ce qu'on fait)

- **Imitation (ce qu'on fait)** : on montre des démos, le robot **copie**. Stable, peu de données, **mais plafonné par la qualité des démos** (ne dépasse pas le démonstrateur).
- **RL (renforcement)** : le robot **essaie**, reçoit une **récompense** (réussi = +1), apprend par **essai-erreur**. Peut **dépasser** l'humain, **mais** demande beaucoup d'essais.
- **« Q matrice »** = Q-learning : table **Q(état, action)** = score « à quel point cette action est bonne **à long terme** ». Forme classique = tableau ; version moderne = **réseau de neurones**.

## 2. Le mur des actions continues (pourquoi le Q-learning pur ne va pas sur un bras)

- Q-learning a besoin de **comparer une liste finie d'actions** pour prendre la meilleure (`max`).
- Un bras = **actions continues** (infinité de mouvements) → impossible de toutes les lister.
- **Solutions** :
  - discrétiser → explose avec les degrés de liberté (mauvais pour 5 axes + pince) ;
  - **acteur-critique** : un réseau **propose** une action (acteur) + un réseau **juge** sa valeur (critique). → **DDPG → TD3 → SAC** (SAC/TD3 = standards modernes) ;
  - Q factorisé (DecQN).
- 3 familles : *value-based* (Q) ≠ *policy-gradient* ≠ *acteur-critique* (les 2 combinés). Pour un bras → **acteur-critique continu**.

## 3. Clarification clé : récompense ≠ critique (Q)

Souvent confondu :
| | C'est quoi | D'où ça vient |
|---|---|---|
| **Récompense** | signal **immédiat** (+1 si réussi) | **donnée par la tâche**, jamais un réseau |
| **Critique / juge (Q)** | « quelle est ma chance de réussir **au final** si je fais cette action ? » | **appris** par un réseau, **à partir** des récompenses |
| **Acteur / proposeur** | propose directement l'action | réseau ajouté pour le continu |

→ On **ne remplace pas** la récompense par un réseau. Le **juge = la Q matrice en réseau** : c'est **ça** le lien avec le Q-learning. (Analogie échecs : la récompense = gagner la partie ; le juge = « ce coup donne de bonnes chances de gagner » = Q.)

## 4. Marier RL + imitation (le plus réaliste pour nous)

On garde la diffusion policy, on ajoute du RL par-dessus (efficace en échantillons) :
- **Offline RL** : apprendre depuis nos démos sans nouvel essai → **IQL, CQL, AWAC**.
- **Offline → online** : démarrer offline puis affiner → **IQL, AWAC, RLPD** (RLPD : ~×2.5 d'efficacité).
- **Residual RL** : garder la politique en boîte noire + apprendre une **petite correction** → ~200× moins d'échantillons réels (ResFiT 2025, préliminaire).

## 5. Les 2 architectures modulaires (la question au prof)

| | « un réseau **par tâche** » | « un réseau qui **choisit** » |
|---|---|---|
| Nom | **Mixture-of-Experts (MoE)** | **RL hiérarchique** |
| Mécanisme | experts + **gating** qui route selon l'entrée | **chef** qui **séquence** des sous-skills dans le temps |
| Réfs | Cheng RA-L 2023 | Options (Sutton 1999, MDP+options=SMDP), FeUdal Networks |
| Décide | *quel expert pour CETTE entrée* | *quel skill lancer MAINTENANT* |
| Quand | beaucoup de tâches apparentées | tâche longue décomposable |

→ **Pas urgent pour nous** (tâche unique). Utile quand plusieurs gestes à enchaîner.

## 6. Le RL peut-il débloquer un plateau ?

Dépend de la **cause** :
- plateau = **plafond des démos / pas de rattrapage** → **OUI**, le RL dépasse les démos + apprend à se corriger.
- plateau = **capacité** (réseau trop petit) → **NON**, le RL n'ajoute pas de capacité, il faut agrandir l'archi.
- plateau = **info manquante** → non.

Le RL attaque un **levier différent** (signal d'entraînement) de notre levier habituel (capacité/archi). Test pour savoir : si agrandir l'archi débloque → c'était la capacité ; sinon → plafond démos → RL pertinent. **Notre continue train Can 30k→50k** teste justement si « plus d'imitation » aide.

## 7. UTD élevé : applicable à l'imitation ? → NON

- **UTD** = nb de mises à jour par donnée collectée. Suppose une **boucle de collecte**.
- En **imitation** : dataset **fixe**, cible **figée** (« copie ») → « UTD élevé » = juste plus d'epochs = **sur-apprentissage** (déjà vu sur nos courbes).
- L'UTD marche en RL car la **cible s'améliore** (Q bootstrappe) + **récompense** + nouvelles données.
- **Twist** : l'**offline RL** a aussi un dataset fixe mais **profite** de l'UTD élevé — car sa cible s'améliore et il peut **dépasser** les données. Donc pour profiter de cette machinerie : passer nos démos en **objectif offline RL**, pas cranker l'imitation.

---

## 8. ⭐ HIL-SERL — l'article phare (RL qui bat l'imitation sur vrai bras)

*Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL* (Science Robotics 2024-25). [arXiv 2410.21845](https://arxiv.org/html/2410.21845v1) · [projet](https://hil-serl.github.io/)

**Résultat** : RL qui part de démos + corrections humaines → **~2× le succès et 1.8× plus rapide que l'imitation seule**, sur tâches de précision réelles. Preuve que le RL casse le plateau des démos.

### Tailles des réseaux (tout est PETIT)
| Composant | Taille | Source |
|---|---|---|
| Encodeur visuel | **ResNet-10** (~5 M), ImageNet pré-entraîné, **128×128** | HIL-SERL/SERL |
| MLP post-encodeur | 2 couches, **256** | SERL |
| Critique (Q) | **ensemble de 10** réseaux, MLP **256 × 2-3 couches**, **+ LayerNorm** | RLPD |
| Acteur | MLP **256 × ~2 couches** (sort pose 6D delta) | SERL |
| Récompense | classifieur binaire (200 pos / 1000 nég, >95 %) | HIL-SERL |

Hyperparams RLPD : ensemble **E=10**, largeur **256**, **UTD=20** (état), **batch 256**, **γ=0.99**, **Adam 3e-4**, **LayerNorm critiques**.
→ Total **~6-8 M params** (vs notre ResNet-34 ~21 M + gros U-Net). **Petit réseau + signal RL fort.**
Sources : [SERL](https://arxiv.org/html/2401.16013v2) · [RLPD (tables)](https://ar5iv.labs.arxiv.org/html/2302.02948v4)

### Les 3 ingrédients de RLPD (le cœur)
1. **Ensemble de 10 critiques** : un seul juge **surévalue** des actions non vérifiées → l'acteur exploite ces fausses bonnes notes → instable. 10 juges + **consensus prudent (min)** → mate la surévaluation.
2. **LayerNorm (dans les critiques)** : le juge s'entraîne sur une **cible mouvante** (bootstrapping) → ses sorties peuvent **exploser** sur des actions jamais vues. LayerNorm garde les notes **bornées/calibrées**. Trick clé.
3. **UTD élevé (=20)** : **20 mises à jour par échantillon réel** → on extrait 20× plus d'apprentissage → **peu d'essais réels** (vital sur robot).
- **Package** : UTD élevé = efficacité **mais** instable → ensemble + LayerNorm = garde-fous qui le rendent stable. Les trois vont **ensemble**.

### Comment éviter le « robot qui fait n'importe quoi » au début
- **Pas de BC préalable**. Les **20-30 démos** vont dans un buffer ; RLPD échantillonne **50/50 démos/online dès le départ** → la politique est orientée par les démos d'emblée.
- ⭐ **Humain dans la boucle** (le « HIL ») : un humain tient une **SpaceMouse** (palet 3D, 6 DDL = la pince). Quand le robot va faire une bêtise, il **attrape le palet et téléopère** le bras quelques secondes (le contrôle bascule politique→humain), puis **rend la main**. **Analogie auto-école double commande**. Interventions **fréquentes au début, décroissantes** ensuite.
- **Données** : « état → action de l'humain » = bons exemples (buffer démos) ; les mauvaises transitions juste avant = buffer RL (apprendre de ses erreurs). → l'humain **corrige pile là où ça échoue**, c'est la donnée la plus précieuse.
- **Sécurité physique** : contrôle en **impédance** (bras souple, ne casse pas), **espace de travail borné**, actions **relatives** à la pince.

---

## 9. Reco pour notre projet (synthèse)

- **Ne pas repartir d'une Q matrice** (incompatible actions continues + trop d'essais).
- **Garder la diffusion policy** et ajouter du RL par-dessus : **residual RL** ou **fine-tuning offline (IQL/AWAC)**, backbone **SAC/TD3**.
- Le « réseau qui choisit » (hiérarchique/MoE) = **plus tard** (multi-skills).
- **Prérequis réel** pour reproduire HIL-SERL : dispositif **humain-dans-la-boucle** (téléop + reprise) + **sécurité physique**. C'est là que se joue la faisabilité, plus que l'algo.

## 10. Lacunes / à creuser plus tard
- **TD3 vs SAC vs DDPG** comme backbone : pas tranché.
- **Sim-to-real** Robomimic → bras 5 axes (domain randomization, residual réel, CQL).
- **Residual RL vs IQL/AWAC** : lequel préserve le mieux la diffusion policy au moindre coût d'échantillons.
- État de l'art manip réelle 2023-2025 RL vs diffusion : peu de comparaisons directes (à part HIL-SERL).

## Biblio (RL vs imitation, bras robotiques)
- ⭐ **HIL-SERL** : [paper](https://hil-serl.github.io/static/hil-serl-paper.pdf) · [arXiv](https://arxiv.org/html/2410.21845v1) · [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.ads5033)
- Survey imitation manip : [arXiv 2508.17449](https://arxiv.org/html/2508.17449v1)
- Survey deep RL manip : [Sensors/MDPI 2023](https://www.mdpi.com/1424-8220/23/7/3762)
- RL + imitation depuis pixels (DeepMind 2018) : [arXiv 1802.09564](https://arxiv.org/pdf/1802.09564)
- VLA-RL (2025) : [arXiv 2505.18719](https://arxiv.org/pdf/2505.18719)
- Thèse sample-efficient manip (Northeastern) : [PDF](https://repository.library.northeastern.edu/files/neu:4f197h37b/fulltext.pdf)
- Survey interactive imitation : [Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1682437/full)
- Liste curée : [Awesome-Robotics-Manipulation](https://github.com/BaiShuanghao/Awesome-Robotics-Manipulation)
- SERL : [arXiv 2401.16013](https://arxiv.org/html/2401.16013v2) · RLPD : [ar5iv 2302.02948](https://ar5iv.labs.arxiv.org/html/2302.02948v4)
