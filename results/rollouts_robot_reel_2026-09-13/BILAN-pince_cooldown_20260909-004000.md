# Bilan — recuit 36 000 (`pince_cooldown_20260909` / checkpoint `004000`), nuit du 2026-09-13

## Résultat

**9 réussites sur 14 essais** — comptage de Niels, communiqué le 2026-09-13 vers 01:40
(après les essais). C'est un **comptage**, pas une estimation.

Précisions de Niels (01:45) :

- ce comptage **n'est pas lié aux enregistrements** : tous les essais n'ont pas été
  enregistrés, et on ne sait pas quels essais comptés ont un bag ;
- les **3 échecs identiques de la série** `SERIE-20260913-0117` **ne font pas partie** de
  ces 14 essais ; ils s'ajoutent à eux.

Ce modèle a donc tourné au moins 17 fois cette nuit (14 comptés + 3 de la série), dont
11 enregistrements (6 par le veilleur, 5 par le panneau), sans correspondance connue entre
les deux listes.

## Conditions de ces essais

- Modèle : recuit du checkpoint 36 000 de `pince_const_20260909`, corpus
  `apple_pince_2cam_128`, état/action 7D/7D, caméra `fixed` (= topic `right`).
- Inférence : RTC (gel 4, guidage 4), iGPU OpenVINO (latence ~240-260 ms), CPU 0-11,
  15 Hz, 10 pas de diffusion.
- Pose de départ commandée : `depart_infer_comp`, à 6,2 cm de la médiane des départs du
  corpus (dz +5,4 cm), dans sa dispersion (~12 cm). **Pose réelle variable** d'un essai à
  l'autre (constaté sur la série) : bras non recalé au nid pendant la séance.
- Luminance caméra `right` pendant les essais enregistrés : 105-112, dans l'enveloppe du
  corpus (97,9-123,6). Pas de mesure pour les essais non enregistrés.

## Ce qui est connu par enregistrement

- 3 échecs : la série `SERIE-20260913-0117` (01:19:51, 01:21:03, 01:23:15), observés par
  Niels, **hors du comptage 9/14**.
- Sans résultat individuel : 00:55:22, 00:57:08, 00:58:02, 01:08:50, 01:10:26, 01:11:51,
  01:22:11, 01:24:18.

## À ne pas conclure trop vite

- Si l'on voulait un taux sur tous les essais connus, il faudrait compter les 3 échecs
  de la série (9 sur 17) — **à décider par Niels** : la série a été faite exprès avec la
  pomme toujours à la même position, ce qui n'est pas la même expérience.
- Pas de comparaison directe avec le 5/14 du premier modèle le 2026-09-08 : autre modèle,
  autre inférence (CPU, async ce jour-là), autre séance.
- Aucun EXP, VAL ni promotion de modèle n'en découle sans Issue et décision humaine
  (règles NIC_forge).
