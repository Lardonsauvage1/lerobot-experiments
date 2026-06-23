# Phase 6 — Robustesse au déplacement caméra (cap sim-to-real)

> **Question centrale** : un vrai bras n'aura jamais sa caméra exactement à la position de la sim. Nos politiques vision-pure résistent-elles à un **déplacement de caméra**, et si non, **comment** les rendre robustes ?

C'est la première phase tournée **explicitement vers le réel** : on ne cherche plus à battre un score en sim, mais à savoir ce qui survit à un changement de point de vue.

Liens : [◀ Compression (4)](COMPRESSION.md) · [Can / méthodologie (5)](PHASE5_PROTOCOLE.md) · **Caméra (6)** · [index](README.md)

## Contexte

Après la compression (Lift) et l'étude de confiance dans nos mesures (Can, phase 5), un constat : **tous nos modèles sont entraînés avec une caméra parfaitement fixe**. En réel, la caméra sera re-fixée « à peu près » au même endroit — quelques cm, quelques degrés d'écart. Cette phase mesure la fragilité à ce décalage et teste l'**augmentation caméra** comme remède.

## Le parcours

1. **Test de fragilité (`camshift`)** — on évalue le modèle Can 74 % (mono-cam agentview+birdview) en **décalant les caméras** par épisode (translation cm / rotation °). Résultat brutal : **74 % → 13 %** dès **2 cm / 2°**. La politique est collée à la géométrie exacte de la caméra d'entraînement.

2. **Idée : augmentation caméra** — entraîner avec des caméras qui **bougent** (jitter aléatoire par épisode, fixe pendant l'épisode = « caméra re-fixée de travers »). Dataset jitté à **10 cm / 10°** construit en rejouant les états robomimic et en re-rendant avec `cam_pos`/`cam_quat` décalés.
   - ⚠️ **Piège trouvé en route** : `env.reset()` **recrée `env.sim`** → un handle `model` capturé *avant* le reset est périmé. Le premier jet appliquait le jitter à un modèle jeté et rendait au nominal. Corrigé en re-capturant `m = env.sim.model` **après** chaque reset (diff image vérifiée : ~3 → 40).

3. **Échec du mini-CNN** — même architecture comprimée (mini-CNN, 0.03 M vision) entraînée sur le dataset jitté : **~3 % de succès même au nominal**, et ce pour **10 / 30 / 50 / 150 démos**. La loss descend pourtant normalement au plancher (~0.04). Découplage loss↔succès total et permanent.

4. **Deux hypothèses** : (a) **durée** — il faut plus d'entraînement ; (b) **capacité** — l'encodeur 0.03 M est trop petit pour une localisation *invariante à la caméra*.

5. **Test durée** — mini-CNN prolongé par **warm restarts cosine** (vagues 130 k) jusqu'à 200 k : **plat à 3-4 %**. Hypothèse durée **réfutée**.

6. **Test capacité** — même tâche jittée, encodeur **ResNet18 natif (11.2 M)** : succès qui **bondit à ~20 % au 30 k** et se stabilise à **~18 %**. Le mini-CNN, lui, ne bouge jamais. **La capacité est le mur.**

7. **Échelle de capacité (en cours)** — pour savoir si on peut faire mieux que 18 %, on monte : **ResNet34 (21 M)** et **ResNet34 à encodeurs séparés (42 M)**, même tâche, même protocole.

## Résultats chiffrés

**Fragilité (camshift, modèle 74 % non augmenté)**

| décalage | 0 (nominal) | 2 cm/2° | 5 cm/5° | 10 cm/10° |
|---|---|---|---|---|
| succès | 74 % | **13 %** | ~3 % | ~2 % |

**Augmentation × capacité (jitter 10 cm/10°, 150 démos, succès @ plateau)**

| encodeur vision | params | succès |
|---|---|---|
| mini-CNN | 0.03 M | **~3 %** |
| ResNet18 | 11.2 M | **~18 %** |
| ResNet34 | 21.3 M | *en cours* |
| ResNet34 séparés | 42.6 M | *en cours* |

## Leçons clés

- **L'augmentation caméra à pleine amplitude exige de la capacité.** Un encodeur minuscule ne peut pas apprendre à localiser l'objet sous une caméra qui bouge → il abandonne la vision et s'effondre. Ce n'est **pas** que 10 cm/10° est inapprenable, c'est que le mini-CNN est trop petit.
- **La loss est totalement aveugle** : mini-CNN (3 %) et ResNet18 (18 %) ont des `train_loss` et `val_loss` **quasi identiques** (~0.04). Seuls les rollouts voient l'écart de 6×.
- **Capacité ET magnitude** : même à 11 M, ça plafonne à 18 %, loin des 74 % sur tâche non-jittée → 10 cm/10° reste intrinsèquement dur ; la taille aide mais ne suffit pas (échelle en cours pour quantifier).
- **La durée ne remplace pas la capacité** : +130 k steps de warm restart sur le petit modèle n'ont rien débloqué.

## Détails techniques

- **Jitter** : `sim.model.cam_pos[i] += dir·U(0,10cm)` et `cam_quat[i] = aa_quat(axe, U(0,10°)) ⊗ q0`, par épisode, puis `sim.forward()`. Re-capturer `env.sim.model` après chaque `reset()`.
- **Backbones natifs** : drapeau `NO_SWAP=1` dans `61_train_minicnn.py` pour **garder** le ResNet (pas de swap vers mini-CNN) ; idem côté chargement d'éval (`src/mini_cnn.load_minicnn_from_ckpt`).
- **Éval** : 500 rollouts tous les 10 k (`12_eval_parallel.py`, partagé avec la phase 5), au nominal ; comparaison entre tailles d'encodeur.
- **Hooks de reprise** (warm restart) : `MINI_INIT_CKPT` / `MINI_SCHED=cosine_wave` / `MINI_START_STEP` (le resume natif LeRobot est cassé après un swap).

## Scripts

`experiments/phase6_camera/` : `01_eval_camshift` · `02_plot_camshift` · `03_camshift_gallery` · `04_make_camaug_dataset` · `05_plot_camshift_aug` · `06_plot_capacite` · `07_plot_duree`. (`12_eval_parallel.py` reste en `phase5_methodology/`, partagé.)

## Suite

- Terminer l'échelle de capacité (21 M / 42 M) : le plafond de 18 % monte-t-il ?
- Si oui → modèles encore plus gros / plus de démos. Si non → le plafond vient de la magnitude/des données.
- Implication réelle : choisir le compromis **capacité × robustesse** pour le bras 5 axes visé.
