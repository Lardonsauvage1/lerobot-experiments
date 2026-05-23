# Phase 2 (PushT) — enquête « loss vs performance » (runs 29 à 32)

> 📍 **Où on en est** : ceci est le **chapitre PushT** du projet. Il couvre l'enquête sur la formulation de la sortie (runs 29-32). La suite : [`LIFT.md`](LIFT.md) (passage à Robomimic Lift, Diffusion Policy 100 %) puis [`COMPRESSION.md`](COMPRESSION.md) (compression). Index : [`README.md`](README.md).
>
> Récit narratif d'une enquête : pourquoi la loss ne reflète-t-elle pas la performance en simulation, et comment y remédier sans changer d'algorithme entier ?
>
> Cette série de 4 runs explore systématiquement la **formulation de la sortie** d'un modèle d'imitation, à input constant (image seule, mêmes features ResNet18 gelées, même Transformer 2L). Seule la dernière couche + la fonction de loss changent.

## Contexte

Avant cette série, le projet avait exploré 28 expériences (MLP, RNN, Transformer, action chunking, etc.) et atteint un best coverage de **46.5%** avec le run `24_precomputed_features` (Transformer 2L, image + agent_pos, MSE regression).

Trois constats gênants à ce moment :
1. **Aucun succès** : `success_rate = 0` sur tous les runs (coverage moyen ≥ 30% mais jamais la tâche finalisée à >95%)
2. **Loss vs coverage décorrélés** : certains runs avec très basse loss (`07_data_filtering`, MSE 0.0012) avaient le pire coverage (4.6%). Le run 24 à 46.5% coverage avait une loss train 50× plus élevée (0.06)
3. **Hypothèse** : sur PushT (tâche multimodale — plusieurs actions valides au même état), la MSE pousse le modèle à prédire la **moyenne** des actions humaines → "moyenne morte" qui ne marche pour aucune des solutions valides

L'objectif de la série 29-32 : **isoler l'effet de la formulation de la sortie** en gardant tout le reste identique.

---

## Run 29 — image seule, MSE

**Question** : si on enlève `agent_pos` de l'entrée, à quel point ça pénalise ?

**Setup** : copie de run 24, on retire juste `agent_pos`. ResNet18 frozen + Transformer 2L + Linear(64, 2) régression. 100 epochs, MSE loss.

**Résultat** :
- Train loss finale : 0.067 (vs 0.061 pour run 24 → +9%)
- Test loss finale : 0.090 (vs 0.072 pour run 24 → +24%)
- **Coverage : 31.8% (vs 46.5% pour run 24 → -14.7pts)**
- Success rate : 0%

**Lecture** : `agent_pos` apportait ~15pts de coverage, mais seulement +24% de loss. La loss sous-estime largement la perte de performance. Le modèle "image seule" tient bien la régression numérique sans tenir la performance comportementale. **Premier signe que la loss MSE n'est pas représentative**.

**Vidéos** :
- `ep0` (seed 0) : max coverage **27.4%**, final 0% → s'approche puis perd
- `ep1` (seed 42) : max **72.0%**, final 0% → presque réussi puis dégrade
- `ep2` (seed 84) : max **80.5%**, final 80.5% → maintient à 80% mais ne passe pas le seuil de 95% de `is_success`

---

## Run 30 — image seule, MDN (Mixture Density Network)

**Question** : si on remplace la régression MSE par une distribution gaussienne mixte (K=5 modes), est-ce que la multimodalité est mieux capturée ?

**Setup** : même backbone qu'en run 29, mais sortie = K=5 gaussiennes diagonales par timestep `(π_i, μ_i, log_σ_i)`. Loss = NLL d'un mélange. À l'inférence : sample → pioche un mode selon les `π`, puis tire dans sa gaussienne.

**Résultat** :
| Checkpoint | Train NLL | Test NLL | Coverage |
|---|---|---|---|
| epoch 25 | +0.30 | +0.24 | 23.5% |
| epoch 50 | -0.15 | -0.14 | 32.8% |
| epoch 75 | -0.37 | -0.35 | **40.9%** ← best |
| epoch 100 | -0.50 | -0.40 | 34.9% |
| Final (sample) | | | 35.4% |

**Lectures clés** :
- **NLL négative = normal** : densité gaussienne (≠ probabilité) peut dépasser 1 quand σ est petit. Une NLL de -0.5 signifie densité moyenne ≈ 1.65 au point cible.
- **Confirmation forte de la décorrélation loss↔coverage** : entre epoch 75 et 100, NLL baisse encore de -0.37 → -0.50 (-35%), mais coverage **chute** de 40.9% → 34.9%. Le modèle apprend à mieux fitter le dataset, mais ce "mieux" ne se traduit pas en simulation.
- **Gain modeste** : 35.4% final (+3.6pts vs run 29), 40.9% au best (+9.1pts). Pas la révolution attendue.

**Vidéos** : très sautillantes — sampling indépendant par timestep entre 5 modes différents → trajectoire incohérente. Identifié comme limitation structurelle du sampling MDN parallèle.

---

## Run 31 — image seule, classification discrète + résidu (BeT-simple)

**Question** : si au lieu de prédire des coordonnées continues, on **discrétise** l'espace d'action en bins et on classifie ?

**Setup** :
- K-means N=64 sur les premières actions de chaque chunk du **train set seul** (pas de fuite test). Centres en pixels, dénormalisés. Cache : `data_cache/action_bins_n64_seed42_train.pt`.
- Sortie modèle : `bin_logits` (64 logits softmax) + `residuals` (64 × 2 résidus, un par bin possible).
- Loss : `CE(bin_logits, true_bin) + MSE(residual[true_bin], true_offset_from_center)`.
- À l'inférence : argmax (ou sample) du bin → ajouter le résidu correspondant → action prédite.

**Améliorations expérimentales** apportées par rapport au run 30 :
- `bin_centers` stockés en `register_buffer` du modèle (state_dict propre, rechargement trivial)
- Évaluation aux checkpoints en mode `argmax` (déterministe) — pour mesurer proprement la corrélation loss↔coverage ; évaluation finale en mode `sample` pour capturer la multimodalité
- Sauvegarde du best checkpoint (basé sur coverage en argmax)

**Résultat** :
| Checkpoint | Train loss | Test loss | CE | MSE résidu | Coverage argmax |
|---|---|---|---|---|---|
| epoch 25 | 1.7055 | 1.6969 | 1.69 | 0.018 | 36.2% |
| epoch 50 | 1.3546 | 1.4286 | 1.34 | 0.017 | 41.6% |
| epoch 75 | 1.2135 | 1.2947 | 1.20 | 0.016 | 41.3% (3/200 succès !) |
| epoch 100 | 1.1319 | 1.2113 | 1.12 | 0.016 | **44.9%** ← best |
| Final (sample) | | | | | 35.0% |

**Lectures clés** :
- **Corrélation loss↔coverage devenue MONOTONE** : loss baisse régulièrement de 1.70 → 1.13 (-33%), coverage monte régulièrement 36.2% → 44.9% (+8.7pts). **Première fois sur le projet**.
- **Pourquoi** : la CE force le modèle à *commettre* une décision discrète (un bin). Pas de "moyenne mortelle" comme en MSE. Pas non plus de σ qui rétrécit comme en MDN.
- **Premier succès du projet** : 3/200 = 1.5% à epoch 75. Aucun run précédent n'avait passé `is_success = True`. Le modèle Discret a réellement terminé 3 épisodes.
- **Discret quasi-rattrape "image+pos"** : 44.9% argmax vs 46.5% du run 24. Donc le vrai obstacle n'était pas l'absence de `agent_pos`, c'était le mode collapse de la MSE.
- **Argmax > Sample sur ce setup** : 44.9% vs 35.0%. Sampling stochastique introduit du bruit qui dégrade. Contrairement à MDN où le sampling était nécessaire pour capturer la multimodalité, ici les bins déjà discrétisés portent la structure modale.

**Vidéos** : nettement plus décidées qu'en run 24/29 (actions tranchantes), **mais saccadées**. Au sein d'un chunk de 20 timesteps, le modèle peut prédire t=0→bin 12 ("(266, 305)") et t=1→bin 47 ("(450, 100)") → l'agent semble se téléporter de 200 pixels entre deux actions consécutives.

**Cause du saccadé** : les bins sont des **positions absolues**. La CE est invariante à la distance géographique entre bins (elle traite les bins comme des classes catégoriques "chat vs chien", pas comme des points sur un plan). Et la prédiction des 20 timesteps est **parallèle** — pas de contrainte temporelle.

---

## Run 32 — image seule, bins de delta sur grille uniforme

**Question** : et si au lieu de bins absolus ("où aller"), on utilisait des bins de **delta** ("comment bouger") ?

### Construction des bins

Vérification empirique sur le dataset (~412k deltas action-to-action) :
- Distribution **bornée** : 99% des deltas dans ±32 pixels par axe, max 136px
- **Quasi-symétrique** : 42-45% des deltas sont positifs (~50% pour parfaite symétrie). Mean ≈ 0, std ≈ 9.3 px par axe.
- **Concentration au centre** : 50% des deltas dans ±5px → distribution gaussienne autour de 0.

Le prior "symétrique borné" justifie de **hand-designer la grille** plutôt que k-means. Grille uniforme 8×8 = 64 bins, bornes au 99.5e percentile (~±0.38 en espace normalisé ≈ ±39 px).

### Insight crucial : "delta_0 fantôme"

Itération de design importante : on a d'abord pensé inclure dans le clustering `delta_0 = a_0 - agent_pos_at_chunk_start`. **Mais ce delta n'existe pas dans la tête de l'humain** — c'est un artefact du chunking, qui mélange la décision humaine (`a_0 - a_{-1}`) avec le retard du simulateur (`a_{-1} - agent_pos`). Stats du delta_0 fantôme : std 20.7 (vs 9.3 pour action→action), donc 2× plus large que les "vrais" deltas humains.

Solution adoptée :
- **Training** : ne cluster que les vrais deltas humains (action→action). Skip les chunks démarrant à `i=0` d'un épisode (pas de `a_{i-1}` disponible). Perte de ~1% des samples.
- **Inférence** : reconstruction cumulative. Le "previous action" mémorisé = la dernière action commandée au simulateur. Au tout début d'épisode (1× par épisode), bootstrap avec `previous_action = agent_pos`.

### Reconstruction à l'inférence

```
step 0 d'épisode : action_0 = agent_pos + delta_0_predicted (bootstrap)
step t ≥ 1      : action_t = previous_commanded_action + delta_t_predicted
```

L'`agent_pos` est utilisé uniquement comme **frame de référence** pour la reconstruction, jamais comme input du modèle → le test "image seule" reste équitable.

### Effet attendu sur le saccadé

Par construction, chaque delta est borné par la taille typique d'un pas humain (~±20 pixels au P95). Même si le modèle se trompe de bin entre t=0 et t=1, le saut résultant est borné. Plus de téléportation possible. Trade-off : un **drift cumulatif** peut apparaître si delta_0 est mal prédit (toute la trajectoire est décalée).

### Résultat

| Checkpoint | Train loss | Test loss | CE | MSE résidu | Coverage argmax |
|---|---|---|---|---|---|
| epoch 25 | 1.6449 | 1.6860 | 1.64 | 0.0010 | 16.5% |
| epoch 50 | 1.4521 | 1.5155 | 1.45 | 0.0010 | 15.5% |
| epoch 75 | 1.3580 | 1.4478 | 1.36 | 0.0010 | 17.4% |
| epoch 100 | 1.3015 | 1.3914 | 1.30 | 0.0009 | **21.3%** ← best |
| Final (sample) | | | | | 21.1% |

**Résultat inattendu** : le delta dégrade fortement le coverage (21.3% vs 44.9% pour run 31). Le saccadé a probablement disparu (les actions sont bornées par la taille typique d'un step humain, ±40px max), mais la performance globale chute.

**Lectures et hypothèses** :

1. **Hypothèse principale — prédire un delta exige de connaître son état actuel**.
   - Run 31 (bins absolus) : "given this image, where should I be next?" → fonction de l'état actuel uniquement, le modèle peut prédire une position absolue sans savoir précisément où il est.
   - Run 32 (bins delta) : "given this image, by how much should I change my command from the previous one?" → fonction de **(état actuel + commande précédente)**.

   Le modèle n'a **pas accès** à la commande précédente (input = image seule). Il doit l'inférer depuis l'image, mais ResNet18 ImageNet n'est pas optimisé pour localiser un petit cercle bleu, et "ce qui a été commandé une étape avant" n'est pas dans l'image actuelle. Le modèle prédit donc des deltas "génériques" (petits, près du centre de la grille — MSE résidu = 0.001 indique que le modèle ne sort presque que les centres de bins), pas adaptés à l'état précis.

2. **Hypothèse secondaire — perte du prior global**. En run 31, les bins absolus étaient des **destinations typiques** (k-means → autour du T, des bords de la cible). Le modèle apprenait "dans cet état, vise cet endroit du carré". En run 32, on a perdu cette info globale — uniquement "comment bouger localement". Le modèle peut être lisse mais perdu globalement.

3. **Drift cumulatif** : 20 deltas accumulés peuvent diverger. Une erreur de delta_0 décale toutes les actions suivantes.

**Ce qu'il aurait fallu pour exploiter le delta** :
- Donner `agent_pos` en input (recompose le "savoir où je suis")
- Donner `previous_action` en input (recompose le "savoir ce que j'ai commandé")
- Décodeur autoregressif (les actions précédentes deviennent l'input naturel)
- Dégeler le ResNet pour qu'il apprenne à localiser l'agent

**Conclusion structurelle** : changer la représentation de sortie sans changer l'input et la backbone a une limite. Le **delta est intrinsèquement conditionnel à l'état** — il fallait fournir cet état, ou laisser le réseau l'apprendre.

---

## Synthèse de la série

### Ce qu'on a appris

1. **Sur PushT (et probablement la plupart des tâches multimodales), la MSE-régression est piégée par le mode collapse**. Pas un bug d'implémentation : c'est la propriété mathématique fondamentale de la MSE sur distributions multimodales.

2. **Changer la formulation de la sortie peut rattraper l'effet d'enlever une feature importante** : run 31 (image seule, CE) ≈ run 24 (image + pos, MSE). C'est puissant et contre-intuitif au premier abord.

3. **Différentes formulations ont des corrélations loss↔coverage très différentes** :
   - MSE : décorrélée (la loss optimise la moyenne, le coverage récompense les modes décidés)
   - NLL/MDN : partiellement décorrélée (les σ peuvent rétrécir artificiellement sans gain en simulation)
   - CE + résidu : **bien corrélée** (commettre une décision discrète aligne directement loss et performance en sim)

4. **La cross-entropy traite les bins comme catégories sans notion de proximité** : c'est sa force (commettre une décision tranchée) mais aussi sa limite (saccadé entre bins absolus géographiquement éloignés). Le passage aux bins de delta résout ça structurellement.

5. **Le bon design des bins fait gagner plus que le tuning d'hyperparams** : bins absolus k-means (run 31) vs bins delta grille uniforme (run 32) → trade-off "décision absolue" vs "lissité par construction".

### Limites identifiées (encore ouvertes)

1. **Prédiction parallèle dans le chunk** : même avec bins delta, t=0 et t=1 sont prédits indépendamment. Un décodeur autoregressif (= BeT propre) ajouterait la cohérence temporelle.
2. **ResNet18 ImageNet n'est pas optimisé pour PushT** : il pourrait mal extraire la position de l'agent (petit cercle bleu) des features. Tester avec ResNet déglacé serait pertinent.
3. **Coverage moyen vs success rate** : 44.9% de coverage moyen mais 0% de success — le modèle frôle souvent la solution sans la verrouiller à la fin. Métrique `final_coverage` ou `coverage_held_above_threshold` serait plus stricte.

### Pour aller plus loin

- **Décodeur autoregressif** sur run 31 ou 32 : t=1 voit t=0 → cohérence temporelle imposée → BeT propre
- **Diffusion Policy** : la SOTA actuelle, ~84-91% success en littérature. Inférence ~10-100× plus lente, donc ~14h-127h sur Mac CPU pour 100 epochs équivalents. À tourner sur Colab/Kaggle.
- **VQ-BeT** : remplacer le k-means/grille par un VQ-VAE qui apprend des tokens d'action. Plus propre que les bins hand-designed, performances ~87% SOTA.
- **Receding horizon** sur run 31 ou 32 : exécuter seulement les 8 premières actions du chunk puis re-prédire. Gratuit (pas de retraining), corrige les dérives en fin de chunk.

---

## Reproductibilité

Chaque run est lancé en mode unbuffered (suivi live possible via `tail -f`) :

```bash
venv312/bin/python -u experiments/pusht/29_image_only.py  2>&1 | tee results/logs/pusht/run_29.log
venv312/bin/python -u experiments/pusht/30_mdn.py         2>&1 | tee results/logs/pusht/run_30.log
venv312/bin/python -u experiments/pusht/31_discrete_ce.py 2>&1 | tee results/logs/pusht/run_31.log
venv312/bin/python -u experiments/pusht/32_delta_grid.py  2>&1 | tee results/logs/pusht/run_32.log
```

> Note : ces scripts ont été déplacés vers `experiments/pusht/` lors de la phase 3 (introduction de Lift). Les commits antérieurs à la phase 3 utilisaient le chemin `experiments/29_*.py` directement.

Hyperparams identiques pour les 4 runs :
- ResNet18 gelé (pré-entraîné ImageNet) + Transformer 2L (d_model=64, 4 heads, FFN 256)
- Chunk size 20, batch 64, LR 1e-3, 100 epochs, seed 42
- 200 épisodes d'éval (variance ±1.7pt), 3 vidéos sauvegardées (seeds 0, 42, 84)

Tous les paramètres entraînés tiennent dans ~170k-180k. ResNet18 ajoute 11.2M de paramètres gelés.

Données : `data_cache/resnet18_features_ep206_64px.pt` (features ResNet pré-calculées, ~10s/epoch d'entraînement sur Mac CPU au lieu de ~3min si on recalculait à chaque epoch).
