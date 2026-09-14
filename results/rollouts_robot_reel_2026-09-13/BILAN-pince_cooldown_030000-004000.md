# Bilan — recuit 30 000 (`pince_cooldown_030000` / checkpoint `004000`), nuit du 2026-09-13

## Résultat

**3 réussites sur 8 essais** — comptage de Niels, communiqué le 2026-09-13 vers 02:10.
Jugement de Niels : **moins bon que le recuit 36 000** (9 / 14, même série de recuits).

**Aucun essai enregistré** (case « Enregistrer les essais » décochée) : ni bag, ni vidéo.

## Conditions

- Recuit de 4 000 pas du checkpoint 30 000 de `pince_const_20260909` (même entraînement,
  même recette de recuit et même corpus `apple_pince_2cam_128` que les recuits 36 000 et
  24 000). État/action 7D/7D, caméra `fixed` (= topic `right`).
- Inférence : RTC (gel 4, guidage 4), iGPU OpenVINO (latence DRY médiane 226 ms), CPU
  0-11, 15 Hz, 10 pas de diffusion. Export OpenVINO `unet_ov.xml` daté du 2026-09-10
  à 15:31 (CORRECTION du 2026-09-13 : la première version de ce bilan disait « créé le
  2026-09-13 à ~01:56 », c'était FAUX ; l'heure citée est probablement celle du chargement dans le panneau, non vérifié).
- Pose de départ commandée `depart_infer_comp` ; bras non recalé au nid (pose réelle non
  contrôlée).

## Place dans la série des recuits (même entraînement, du plus au moins avancé)

| Recuit | Réussites (comptage Niels) |
|---|---|
| 36 000 | 9 / 14 |
| **30 000** | **3 / 8** |
| 24 000 | à venir |

Pour mémoire, le 2026-09-10 ce recuit avait été estimé « ~25 % » (estimation, pas un
comptage, dans une séance jugée non concluante à cause de la lumière — PR #40).

## À ne pas conclure trop vite

Petits nombres, pose réelle non contrôlée. Aucun EXP, VAL ni promotion sans Issue et
décision humaine.
