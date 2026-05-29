# Phase 5 — Can (PickPlaceCan) : tâche plus dure + virage transférabilité réel

> Après Lift (résolu + compressé à ~99 %), on étend à une tâche **pick-and-place** plus longue : attraper une canette et la déposer dans un bac. Au fil de la phase, **un constat majeur a fait pivoter le projet** : tous nos résultats donnaient les **coordonnées de l'objet** au modèle — une info qu'un vrai bras n'a pas. On a donc basculé en **vision pure** (image + proprio uniquement), seule voie crédible pour le bras 5 axes visé.
> Liens : [◀ Phase 4 Compression](COMPRESSION.md) · **Phase 5 (ici)** · [index](README.md)

## Contexte

**Robomimic Can / `PickPlaceCan`** : le bras Panda saisit une canette sur une table et la **dépose dans le bon compartiment** d'un bac — saisie **+ transport + dépôt** (vs Lift = juste soulever). ~2× plus d'étapes, horizon doublé (démos ~118 steps vs 59).

Dataset : **200 démos humaines** (ph), action 7D, image agentview 96px.

⚠️ **Single-object, pas du tri.** Le bac a 4 compartiments car la scène est héritée de la tâche complète **PickPlace** (canette, lait, céréales, pain → 4 bacs). `PickPlaceCan` n'en garde **qu'un** (la canette, toujours vers son compartiment ; les 3 autres restent vides). Le tri 4 objets n'est pas dispo en démos humaines dans Robomimic.

## Le parcours

| Étape | État |
|---|---|
| 1. Télécharger Can ph + convertir → LeRobot (12D avec can_pos) | ✅ |
| 2. Baseline (mini-CNN [64,128,256] + état 12D avec coords canette) | ✅ → **48 %** |
| 3. **Réalisation clé** : `can_pos` est une béquille sim indisponible au réel | 💡 |
| 4. **Vision pure** : converter `--proprio` → état 9D (sans can_pos) | ✅ |
| 5. Modèle vision-only mono-cam (ResNet18 + [64,128,256] + 9D) | ✅ → **72 %** |
| 6. Diagnostic des échecs → ambiguïté de profondeur d'une seule vue | 💡 |
| 7. Tentative 2 caméras (agentview + wrist), encodeur **partagé** | ✅ → 52 % (régression) |
| 8. 2 caméras, **encodeurs séparés** (1 ResNet18 par vue) | ⏳ |

## Résultats

### Baseline (12D avec `can_pos`) — 48 %
Compressé `[64,128,256]` mini-CNN, 12k steps, état 12D incluant la position absolue de la canette. **48 % [35-62]** sur 50 val @ 10 pas. Can est apprenable mais pas résolu — la tâche est plus dure que Lift, ou le modèle est sous-dimensionné, ou l'état trop minimal.

### Pivot : pourquoi pas de coords objet
**Un vrai bras 5 axes n'aura JAMAIS la position de la canette donnée gratuitement** — il faudrait un système de perception pour la calculer depuis la caméra. Si on s'appuie sur cette béquille en sim, **rien ne transfère**. Le seul setup honnête côté transfert :
- **Image** (caméra réelle) + **proprioception** (encodeurs articulaires + cinématique).
- **Pas de pose objet** dans l'état.

Donc on rebascule en 9D (proprio seul) et on laisse la vision faire le travail de localisation.

### Mono-caméra vision-only (`02_proprio`) — **72 %**
ResNet18 (11.2 M) + U-Net `[64,128,256]` (5 M) + état 9D, 10k steps. **72 % [58-83]** sur 50 val @ 10 pas, t_succ 107. Can est résoluble en vision pure à un niveau utile.

Plot loss train/val : [`../results/runs/can/02_proprio/loss_curve.png`](../results/runs/can/02_proprio/loss_curve.png) (convergence propre, pas d'overfit).

### Diagnostic des échecs → ambiguïté de profondeur
Sur les ~28 % d'échecs, un pattern frappant : **le bras part vers les bacs sans avoir vraiment saisi la canette**. Avec une seule vue de face (agentview), le modèle a du mal à évaluer **où exactement** est la canette en profondeur.

Exemple typique (ep161 du modèle `02_proprio`) :

![échec d'un seul angle de caméra : le bras passe à côté de la canette](../results/runs/can/02_proprio/videos/proprio_FAIL_ep161.gif)

→ Sur les 300 steps disponibles, le bras n'arrive jamais à un grasp propre. C'est exactement le genre d'échec qu'une **2e vue** est censée corriger.

### 2 caméras (agentview + wrist), encodeur partagé (`06_proprio_wrist`) — 52 %
Ajout de `robot0_eye_in_hand` (vue du poignet), mêmes 9D + ResNet18 **partagé** entre les 2 vues, 10k steps. **52 % [38-65]** — **régression apparente** (IC larges, chevauchent le mono-cam, donc pas une vraie régression statistique, mais pas l'amélioration espérée).

Hypothèse : un seul ResNet18 doit gérer 2 distributions visuelles très différentes (vue large vs plongée poignet), il fait un **compromis qui dilue la capacité** → moins bonne représentation par vue.

### 2 caméras, encodeurs séparés (`08_proprio_wrist_sep`) — ⏳
1 ResNet18 par caméra (22.4 M de vision au lieu de 11.2). Chaque encodeur se spécialise sur sa vue. *Entraînement en cours.*

## Leçons clés

1. **Donner les coords d'un objet en sim = béquille sans transfert au réel.** Tout score obtenu avec une telle béquille **n'est pas comparable** à ce qu'un vrai bras pourra faire. L'expérience honnête est image + proprio seul.
2. **Lift se résout très bien en vision pure** (~99 % avec coords → ~81 % sans, sur 500 rollouts). Encourageant pour Lift, **moins pour Can dur** (~72 % en mono-cam).
3. **Une seule vue de face = ambiguïté de profondeur** sur les tâches de manipulation. Diagnostic empirique sur les échecs Can mono-cam.
4. **Ajouter une caméra n'est pas gratuit** : avec un encodeur **partagé**, on peut en fait **régresser** parce que la capacité visuelle est diluée. Encodeurs **séparés** par caméra semblent indispensables pour bénéficier vraiment de l'info multi-vue (à confirmer avec `08`).

## Détails techniques

- **Pipeline conversion** : `src/can_to_lerobot.py` avec `--proprio` (état 9D) et `--wrist` (ajoute `robot0_eye_in_hand`). Convention multi-cam = `observation.images.<nom>` (détectée par Diffusion Policy comme VISUAL multiples).
- **Éval** : `src/can_eval.py` (env PickPlaceCan, `make_env`, `load_init_states`, `MAX_STEPS=300`). État construit côté live env (`object[7:10]` pour `can_pos` quand utilisé) — différent de la convention dataset (`object[0:3]`) à cause du mismatch robosuite 1.4 ↔ 1.5.
- **Risque connu (résolu)** : robosuite 1.5 réordonne l'`object` et la partie relative est instable → solution = **abandonner la partie relative** et utiliser seulement les composantes fiables (proprio ± can_pos absolue).
- **Données** : `data_cache/robomimic_can_ph/` (HDF5 source), `data_cache/lerobot_can_ph[_proprio][_wrist]/` (datasets convertis), `data_cache/can_demos/` (vidéos d'exemples expertes).
- **Scripts numérotés** : `01_train_baseline.sh` · `02_train_proprio.sh` · `03_eval_proprio.py` · `04/05_proprio_videos*.py` · `06_train_proprio_wrist.sh` · `07_eval_proprio_wrist.py` · `08_train_proprio_wrist_sep.sh` · `09_wrist_videos.py`.

## Suite

- Verdict `08` (encodeurs séparés) en attente → décidera si la voie multi-cam vaut le coût.
- Si ça marche : grilles rigoureuses 500 rollouts (réutiliser outillage phase 4) pour comparer proprement vision-only mono-cam vs multi-cam.
- Paliers plus durs en réserve : **Square** (insertion précise — données téléchargées) ou **Tool Hang** (très long horizon).
