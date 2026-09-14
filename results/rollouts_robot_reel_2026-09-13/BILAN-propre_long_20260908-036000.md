# Bilan — premier modèle (`propre_long_20260908` / checkpoint `036000`), nuit du 2026-09-13

## Résultat

**5 réussites sur 10 essais** — comptage de Niels, communiqué le 2026-09-13 vers 01:50,
pour la série d'essais faite après le recuit 36 000 (panneau rechargé avec ce modèle à
01:38). Jugement de Niels : **« bien moins bien que le précédent »**, c'est-à-dire que le
recuit 36 000 (9 réussites sur 14, voir `BILAN-pince_cooldown_20260909-004000.md`).

**Aucun de ces 10 essais n'a été enregistré** (case « Enregistrer les essais » décochée
après la relance du panneau) : ni bag, ni vidéo, ni mesure de lumière pour eux.

## Historique de ce même checkpoint (vérifié : même chemin dans chaque source)

| Date | Réussites | Inférence | Source |
|---|---|---|---|
| 2026-09-08 | **5 / 14** | CPU, async (exec-offset 6) | `brasRobot/docs/experiences/2026-09-08_premier_rollout_autonome.md` — « premier modèle du projet à mener la pomme sur la cible en autonomie » |
| 2026-09-10 | **0 / 10** | iGPU, async | PR #40 (EXP-20260910-NM-002, non acceptée) — lumière de la séance hors enveloppe du corpus (128,9 > 123,8) |
| 2026-09-13 | **5 / 10** | iGPU, RTC | ce bilan |

Ce n'est pas le premier modèle *entraîné* du projet (d'autres l'ont précédé et échouaient),
mais le premier à avoir réussi la tâche en autonomie. Les trois séances diffèrent
(inférence, lumière, pose réelle de départ) : les chiffres ne se comparent pas
directement entre eux.

## Conditions de ces essais

- Modèle : corpus `apple_propre_2cam_128`, état/action 6D/7D, caméra `fixed` (= topic
  `right`). C'est le modèle du premier rollout autonome (5/14 le 2026-09-08).
- Inférence : RTC (gel 4, guidage 4), iGPU OpenVINO, CPU 0-11, 15 Hz, 10 pas de
  diffusion (réglages du panneau au moment du chargement ; non relevés essai par essai
  faute d'enregistrement).
- Pose de départ commandée : `depart_infer_comp`. Bras **non recalé au nid** pendant la
  séance : la pose réelle peut s'écarter de la commande (constaté sur la série du recuit).

## Autres essais de ce modèle cette nuit (enregistrés, sans résultat individuel)

- mode async (exec-offset 6) : 00:01:56, 00:11:46, 00:12:28 ;
- mode RTC : 00:31:08, 00:32:09, 00:33:13.
Ils ne font pas partie du comptage 5/10.

## À ne pas conclure trop vite

- La comparaison 5/10 contre 9/14 est celle de Niels, **même soirée, même réglage
  d'inférence (RTC + iGPU)** — c'est la comparaison la plus propre de la nuit. Elle reste
  sur de petits nombres, et la pose réelle de départ n'était pas contrôlée.
- Le 5/14 du 2026-09-08 n'est pas directement comparable (CPU, mode async, autre séance).
- Aucun EXP, VAL ni promotion de modèle n'en découle sans Issue et décision humaine.
