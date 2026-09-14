# Bilan — recuit 24 000 (`pince_cooldown_024000` / checkpoint `004000`), nuit du 2026-09-13

## Résultat

**3 réussites sur 8 essais** — comptage de Niels, communiqué le 2026-09-13 vers 02:25.

Observations de Niels :

- il fait des erreurs **dans des parties évidentes** de la tâche ;
- **il lâche la pomme pendant la ligne droite** (le transport) ;
- **moins bon que le précédent** (recuit 30 000), à score égal (3 / 8 tous les deux).

**Aucun essai enregistré** (case « Enregistrer les essais » décochée) : ni bag, ni vidéo.

## Conditions

- Recuit de 4 000 pas du checkpoint 24 000 de `pince_const_20260909` (même entraînement,
  même recette de recuit, même corpus `apple_pince_2cam_128` que les recuits 36 000 et
  30 000). État/action 7D/7D, caméra `fixed` (= topic `right`).
- Inférence : RTC (gel 4, guidage 4), iGPU OpenVINO (latence DRY médiane 230 ms), CPU
  0-11, 15 Hz, 10 pas de diffusion. Export OpenVINO `unet_ov.xml` daté du 2026-09-10
  à 16:01 (CORRECTION du 2026-09-13 : la première version de ce bilan disait « créé le
  2026-09-13 à ~02:10 », c'était FAUX ; l'heure citée est probablement celle du chargement dans le panneau, non vérifié).
- Pose de départ commandée `depart_infer_comp` ; bras non recalé au nid (pose réelle non
  contrôlée).
- Le 2026-09-10, ce recuit avait fait 0 / 9, résultat alors jugé **confondu** (départ à
  14 cm en y et 17 cm en z de la médiane du corpus, lumière hors enveloppe — PR #40).

## Série des recuits du même entraînement (du plus au moins avancé), même soirée

| Recuit | Réussites (comptage Niels) | Observation |
|---|---|---|
| 36 000 | 9 / 14 | meilleur de la série ; + 3 échecs identiques de la SERIE-20260913-0117, hors comptage |
| 30 000 | 3 / 8 | moins bon que le 36 000 |
| **24 000** | **3 / 8** | erreurs dans des parties évidentes, lâche la pomme dans la ligne droite ; moins bon que le 30 000 |

## À ne pas conclure trop vite

Petits nombres (8 à 14 essais), pose réelle non contrôlée, aucun essai enregistré pour le
30 000 et le 24 000. La tendance « plus d'entraînement avant recuit = mieux » est
cohérente avec le banc hors ligne de l'EXP-20260910-NM-002 (non acceptée), mais elle n'est
pas établie ici. Aucun EXP, VAL ni promotion sans Issue et décision humaine.
