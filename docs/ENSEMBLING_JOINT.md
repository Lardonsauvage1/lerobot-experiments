# Joint vs cartésien, temporal ensembling & stabilisation des poids (Can)

> **En une phrase.** L'espace d'action **joint** (moteur) est très instable comparé au **cartésien** (OSC, 94,8 %) ; le **temporal ensembling** le rattrape ponctuellement (jusqu'à 90 %) mais c'est un *multiplicateur*, pas une solution stable. On étudie EMA et SWA pour stabiliser l'entraînement lui-même.

Structure : Contexte · Le parcours · Résultats · Leçons clés · Détails techniques · Suite.

---

## Contexte

Même tâche (Can, PickPlaceCan, Panda 7-DOF), même archi (ResNet34 + gros U-Net, 61 M), deux **espaces d'action** :
- **Cartésien (OSC)** — run31, feedback dans l'espace tâche → **94,8 %@500**, exécution lisse.
- **Joint (JOINT_POSITION absolu, kp=50)** — le réseau prédit directement les angles articulaires → **~15 %** au dernier checkpoint, convergence qui **oscille violemment** (12-56 % d'un checkpoint à l'autre, avec des checkpoints carrément cassés à 0 %).

## Le parcours

1. **Constat** : le joint sous-performe massivement et n'a **pas de plateau stable** (≠ cartésien qui monte proprement vers 95 % et y reste).
2. **Temporal ensembling** (façon ACT, à l'inférence, sans réentraînement) : on interroge la policy à chaque pas, on récupère le chunk d'actions complet, et on fait une **moyenne pondérée** `exp(-m·âge)` des chunks recouvrants. Ça lisse l'exécution saccadée.
3. **Recherche des causes d'instabilité** (sources vérifiées) et **stabilisation de l'entraînement** : EMA (moyenne mobile des poids) et SWA (moyennage de checkpoints).

## Résultats

### Temporal ensembling (sans → avec, m=0.01, n=50/30)

**JOINT** (exécution saccadée) :

| checkpoint | sans | avec | Δ |
|---|---|---|---|
| 30k (meilleur) | 56 | **90** | +34 |
| 40k | 12 | **52** | +40 |
| 50k (+10k) | 46 | 43 | ~0 (déjà stable) |
| 34k (cassé) | 0 | **0** | +0 |

m-sweep sur 40k : m=0.01 → 52 % ; m=0.1 → 33 % (**plus de lissage = mieux** ici).

**CARTÉSIEN** (run31, OSC — déjà lisse) :

| checkpoint | sans | avec | Δ |
|---|---|---|---|
| 24k | 73 | **87** | +13 |
| 30k | 77 | **90** | +13 |
| 40k (plafond) | 97 | 97 | +0 |

### SWA — moyennage de poids de checkpoints (n=50)

On moyenne les **poids** de plusieurs checkpoints joint (sans réentraînement) et on évalue le modèle résultant. Le meilleur checkpoint individuel est à **56 %**.

| Combo | Membres (taux indiv.) | Moyenne des poids |
|---|---|---|
| best3 | 56,48,48 | **70** |
| best5 | 56,48,48,46,38 | 64 |
| med3 | 34,34,30 | **76** ⬆️ |
| late5 (5 derniers) | 0,38,48,34,46 | 70 |
| rand5 (aléatoire) | 12,0,38,18,0 | **62** ⬆️ |
| **late10 (10 derniers)** | 34,0,30,12,12,0,38,48,34,46 | **88** 🏆 |
| goodonly9 (les ≥30 %) | 38,48,56,34,30,38,48,34,46 | 72 |
| null3 | 0,0,0 | 6 |
| best_null | 56,0 | **8** ⬇️ |

**Conclusions :**
1. **Le SWA marche spectaculairement** : `late10` = **88 %** (vs meilleur checkpoint individuel 56 %), **un seul modèle stable, gratuit, déployable**. Presque le niveau du temporal ensembling (90 %) mais sans truc d'inférence ni checkpoint chanceux.
2. **Moyenner les poids ≠ moyenner les réussites** : presque toutes les moyennes **dépassent leur meilleur membre** (`med3` 30-34 % → 76 % ; `rand5` membres ~14 % moyenne → 62 %). Le moyennage trouve un **minimum plus plat/meilleur**.
3. **Un checkpoint cassé empoisonne une PETITE moyenne** : `best_null` (56+0) → 8 %, `null3` → 6 %. Mais **noyé dans une grande moyenne il est dilué** : `late10` contient 2 cassés (34k, 42k = 0) et fait quand même 88 %.
4. **Plus de checkpoints (queue LR-constant) = mieux** : late10 (88) > goodonly9 (72) > best5 (64). → la bonne pratique = moyenner ~10 checkpoints tardifs.

Détails : `results/runs/can/joint_r34_bigunet/swa_table.csv`.

### Matrice schedule × SWA (quand le SWA aide-t-il ?)

Test : appliquer le SWA à des modèles **constant** vs **cosine**, gros (cartésien) et petit (mini-CNN), avec/sans marge.

| Modèle | brut | + SWA | lecture |
|---|---|---|---|
| **joint (constant)** | 15 | **88** (+ensembling **96**) | rebond + marge → **gros gain** ; les 2 techniques s'empilent |
| **cart. cosine** (run31) | 94,8 | late5 **97** / late10 **89** / mid **74** | clusterisé → **no-op** (late5≈94,8) ; late10 baisse (mélange de *stages*) ; mid n'excède pas son meilleur membre |
| **cart. constant** (run31-jumeau) | ~50 | late5 **77** / late10 **76** | rebond → **+26 pts**, MAIS **reste sous cosine (94,8)** |
| **mini cosine** | 81 | 78 | clusterisé + capacité saturée → no-op |
| **mini constant** | 76 | 68 | capacité saturée → pas de marge |

**Conclusions de la matrice :**
1. **Le SWA aide quand l'étalement « REBONDIT » + marge** (joint 15→88, cart-constant 50→77). Mesuré : le constant est 16× plus étalé que le cosine.
2. **Neutre quand clusterisé** (cosine annélé : cos_late5 97≈94,8 ; mini-cosine 78≈81) → le cosine a déjà fait le settling, **rien à moyenner**.
3. **Peut DÉGRADER si on moyenne à travers les *stages***: cos_late10 (89) < cos_late5 (97) car il inclut des checkpoints plus précoces/différents ; et cos_mid (étalé mais « qui AVANCE », pas « qui rebondit ») n'excède pas son meilleur membre (74≈74). → **étalement-qui-avance ≠ étalement-qui-rebondit.**
4. ⚠️ **NUANCE clé** : « constant + SWA ≈ cosine » n'est que **PARTIEL**. Sur le gros cartésien, constant+SWA **récupère beaucoup** (50→77) mais **ne rejoint PAS** cosine (94,8) — résidu ~18 pts. → Le SWA **répare le non-settling, pas le sous-entraînement** : le run constant lui-même sous-converge vs cosine à 40k (le constant décolle plus lentement, cf. [`SCHEDULE.md`](SCHEDULE.md)). Pour fermer le résidu : plus de steps constant, ou meilleur merge (fenêtre + poids dérivés).

### EMA — *(en cours, à compléter)*

EMA (decay 0.9999) activée sur le push joint **50k→80k** (env `EMA=1` dans `50_train_valloss.py`) → comparaison **poids bruts vs EMA sur la même plage**. → *résultats à compléter quand le push tourne.*

## Leçons clés

1. **L'espace d'action compte énormément** : cartésien (OSC) 94,8 % ≫ joint ~15 %, à archi/données identiques.
2. **Temporal ensembling = multiplicateur, pas créateur.** Gain ∝ **marge × instabilité** : fort sur une policy saccadée avec de la marge (joint 12→52, cartésien 24k 73→87) ; **nul au plafond** (cartésien 97→97) ou **sur un checkpoint cassé** (joint 34k 0→0). Technique **générale** (pas joint-only), mais **KO en multimodal** (cf. PushT 0,67→0,40 : moyenner deux modes valides = action invalide).
3. **Le SWA donne une vraie solution joint stable** : moyenner ~10 checkpoints tardifs (`late10`) → **88 %**, un seul modèle déployable, gratuit (post-hoc, sans réentraînement) — bien au-dessus du meilleur checkpoint isolé (56 %) et sans dépendre d'un checkpoint chanceux comme le temporal ensembling. Limite : un checkpoint cassé empoisonne une *petite* moyenne (best_null 56+0 → 8 %), mais est dilué dans une grande.

## Détails techniques

### Pourquoi le joint est instable (causes candidates, sourcées)

1. **Variété de configurations multimodale (cause #1)** — un bras **redondant 7-DOF** a plusieurs configs articulaires pour la même pose d'outil (espace nul) → cibles articulaires **multimodales** pour des états quasi-identiques → la diffusion moyenne les modes. La pose d'outil (cartésien), elle, est unimodale. *(Demystifying Action Space Design, arXiv 2602.23408 : l'espace joint vit sur une « variété de configurations complexe, non-linéaire et multimodale ».)*
2. **Discontinuité d'angle (360°=0°)** — représentations d'angle discontinues → erreurs de régression ; remède sin/cos ou 6D *(Zhou et al., CVPR 2019)*. Probablement **secondaire** ici (pick-place borné).
3. **Amplification cinématique + pas de feedback tâche** — petite erreur articulaire amplifiée en bout de bras ; le PD articulaire ne corrige pas dans l'espace tâche (≠ OSC).
4. **Pas d'EMA dans LeRobot** — confirmé (rien dans config/modeling/train loop), alors que Diffusion Policy original (Chi 2023) en fait un élément de stabilité → on évalue des poids SGD bruts qui rebondissent.

### Remèdes pour entraîner sur une tâche « instable »
- **EMA** des poids (standard Diffusion Policy) — ajouté à `experiments/lift/50_train_valloss.py` via `EMA=1 EMA_DECAY=0.9999`.
- **SWA / moyennage post-hoc** de checkpoints (moyenne des **poids**, pas des réussites → souvent un minimum plus plat, meilleur qu'un checkpoint isolé).
- **Actions delta (chunk-wise)** au lieu d'absolues — arXiv 2602.23408 : delta 89,6 % vs absolu 69,0 % (joint).
- **Encodage sin/cos** des angles si le wraparound s'avère réel.

### Scripts
- Temporal ensembling : `experiments/can/47_eval_joint_ensemble.py` (joint), `48_eval_cartesian_ensemble.py` (cartésien).
- Push + EMA : `experiments/can/62_push_joint_mac2.sh`, éval brut+EMA `63_eval_push_joint.sh`, comparaison `64_plot_raw_vs_ema.py`.
- SWA : `experiments/can/65_swa_make.py` (moyenne des poids), `65_swa_eval.sh` (table de combos).

### Sources
- [Demystifying Action Space Design (arXiv 2602.23408)](https://arxiv.org/abs/2602.23408)
- [On the Continuity of Rotation Representations, Zhou et al. CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf)
- [Diffusion Policy, Chi et al. (arXiv 2303.04137)](https://arxiv.org/pdf/2303.04137)
- [EMA of Weights in Deep Learning (arXiv 2411.18704)](https://arxiv.org/html/2411.18704v1)

## Suite
- Compléter les tableaux EMA et SWA quand les évals tournent (push joint 50k→80k, combos SWA).
- Si EMA/SWA donnent un point **stable et haut** → début de vraie solution joint ; sinon → confirme que l'instabilité est structurelle à l'espace joint (et le cartésien reste le choix de déploiement).
