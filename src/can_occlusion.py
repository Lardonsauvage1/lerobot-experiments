"""Occlusion synthétique de la canette — banc « Can-Occluded » (étape 0 du chantier mémoire).

But : reproduire EN SIMULATION le mode d'échec dominant du robot réel (note atomman
`2026-09-08_premier_rollout_autonome.md`, §2) — *le bras masque la cible, le modèle la perd
et enchaîne la phase suivante comme s'il l'avait attrapée*.

MÉTHODE — on ne peint PAS un masque sur l'image
------------------------------------------------
Peindre un rectangle noir/gris créerait un artefact que le réseau n'a JAMAIS vu à
l'entraînement : on mesurerait alors une chute due au hors-distribution, pas à la perte
d'information. À la place on **retire la canette de la scène et on re-rend** :

    sauver qpos(Can_joint0) -> téléporter la canette loin sous le sol -> sim.forward()
    -> render -> restaurer qpos -> sim.forward()

L'image obtenue est photométriquement PARFAITE (fond correct, bras correct, ombres
correctes) et ne diffère de l'originale que par l'absence de la canette. C'est exactement
« je ne la vois plus », sans aucun autre changement.

⚠️ `sim.forward()` et NON `sim.step()` : on recalcule les positions dérivées sans faire
avancer la physique. L'état dynamique est restauré à l'identique après le rendu.

DÉCLENCHEMENT — géométrique, paramétré par un seul nombre
---------------------------------------------------------
L'occlusion s'active quand le préhenseur est **au-dessus** de la canette et à moins de
`radius` mètres d'elle en XY :

    ||eef_xy - can_xy|| < radius   ET   eef_z > can_z

C'est la géométrie exacte de l'échec réel (le bras s'interpose entre la caméra extérieure
et la zone de travail au moment de l'approche). `radius` est un **paramètre continu** :
0 = jamais d'occlusion (témoin), grand = la canette disparaît dès qu'on s'en approche.
On trace donc une COURBE de robustesse, comme on l'a fait pour le décalage caméra.

Cas particulier assumé : une fois la canette saisie, l'eef reste à moins de `radius`
→ occlusion permanente. C'est fidèle (la pince cache réellement l'objet) et sans
conséquence (une fois saisie, sa position est celle de la pince).
"""

import json
from pathlib import Path

import numpy as np

CAN_JOINT = "Can_joint0"
FAR_AWAY = np.array([0.0, 0.0, -5.0])  # sous le sol, hors de tous les champs de caméra

# Hauteur de la canette POSÉE sur la table : 0,8600 m, constante sur les 250 épisodes
# (mesurée, écart-type nul). Au-delà de +2 cm elle est tenue par la pince.
CAN_Z_REST = 0.86
LIFT_MARGIN = 0.02


# --------------------------------------------------------------------- rendu sans canette
def get_sim(env):
    """`env.env.sim` (env robomimic, côté éval) ou `env.sim` (env robosuite brut, côté
    conversion du dataset). Les deux chemins existent dans le projet — on absorbe ici."""
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "sim"):
        return inner.sim
    return env.sim


def _qpos_slice(sim, joint_name=CAN_JOINT):
    start, end = sim.model.get_joint_qpos_addr(joint_name)
    return slice(start, end)


def render_without_can(env, height, width, camera_name="agentview", joint_name=CAN_JOINT):
    """Rend la scène EXACTEMENT comme d'habitude, mais canette retirée. Restaure après."""
    sim = get_sim(env)
    sl = _qpos_slice(sim, joint_name)
    saved = np.array(sim.data.qpos[sl], copy=True)
    try:
        sim.data.qpos[sl][:3] = FAR_AWAY
        sim.forward()
        img = sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        img = img.copy()
    finally:
        sim.data.qpos[sl] = saved
        sim.forward()
    return img


def render_normal(env, height, width, camera_name="agentview"):
    return get_sim(env).render(height=height, width=width, camera_name=camera_name)[::-1].copy()


# ------------------------------------------------------------------- condition d'occlusion
def occluded_from_xyz(eef, can, radius, *, require_above=True, stop_when_lifted=True):
    """Condition d'occlusion depuis les positions brutes — utilisable côté DATASET, où
    `can_pos` vit dans object[0:3] (format 1.4) et non object[7:10] (env live 1.5).

    ⚠️ CORRECTION du 2026-09-09 — `stop_when_lifted`. La version initiale n'occultait que
    sur la proximité XY : or une fois la canette SAISIE, la pince reste par construction à
    moins de `radius` d'elle, donc l'occlusion ne se coupait plus JAMAIS. Le dataset
    d'entraînement montrait la canette disparaître à l'instant de la préhension, puis tout
    le transport et le dépôt exécutés sans jamais la revoir — 44,7 % de la durée des
    réussites. J'avais écrit dans ce fichier que ce cas était « sans conséquence » : la
    baseline entraînée ainsi est tombée à 3,3 % de succès contre 37,6 % pour le témoin.

    La correction est aussi la plus juste physiquement : quand le robot TIENT l'objet,
    celui-ci monte avec le bras et redevient parfaitement visible. On n'occulte donc que
    la canette encore POSÉE (z <= CAN_Z_REST + LIFT_MARGIN).
    """
    if radius <= 0:
        return False
    eef = np.asarray(eef).flatten()
    can = np.asarray(can).flatten()
    if stop_when_lifted and can[2] > CAN_Z_REST + LIFT_MARGIN:
        return False
    if radius >= 1e3:
        return True
    if require_above and eef[2] <= can[2]:
        return False
    return float(np.linalg.norm(eef[:2] - can[:2])) < radius


def can_is_occluded(obs, radius, *, require_above=True):
    """Le bras masque-t-il la canette à cet instant ?

    radius <= 0        -> jamais (témoin sans occlusion)
    radius >= 1e3      -> toujours (borne basse : la canette n'est jamais visible)
    """
    return occluded_from_xyz(np.asarray(obs["robot0_eef_pos"]),
                             np.asarray(obs["object"]).flatten()[7:10],  # live env (cf. can_eval)
                             radius, require_above=require_above)


def make_render_fn(radius, camera_name="agentview"):
    """Fabrique le `render_fn` à passer à rollout_eval, pour un niveau d'occlusion donné."""
    def render_fn(env, obs, image_size):
        if can_is_occluded(obs, radius):
            return render_without_can(env, image_size, image_size, camera_name)
        return render_normal(env, image_size, image_size, camera_name)
    return render_fn


# ------------------------------------------------------------------------------- la sonde
def z_cycles(z, min_amp=0.02):
    """Cycles descente->remontée d'amplitude > min_amp (m) dans la hauteur de l'effecteur.

    Marqueur de la boucle d'échec INDÉPENDANT de toute géométrie de zone : il capte
    « il remonte et redescend » même quand le bras ne sort jamais du voisinage de la cible
    — précisément le cas qui rendait `n_approaches` aveugle.
    """
    z = np.asarray(z, dtype=float)
    if z.size < 2:
        return 0
    ext, direction = [float(z[0])], 0
    for v in z[1:]:
        v = float(v)
        if direction >= 0 and v < ext[-1] - min_amp:
            ext.append(v); direction = -1
        elif direction <= 0 and v > ext[-1] + min_amp:
            ext.append(v); direction = 1
        elif (direction < 0 and v < ext[-1]) or (direction > 0 and v > ext[-1]):
            ext[-1] = v
    return max((len(ext) - 1) // 2, 0)


class OcclusionProbe:
    """Mesure ce que l'occlusion fait subir à la policy, épisode par épisode.

    Le taux de succès global DILUE l'effet qu'on cherche : il mêle les épisodes où la
    canette n'a jamais été masquée à ceux où elle l'a été. On instrumente donc :

      occl_steps / occl_fraction : combien de temps la cible a été invisible
      n_occl_events              : nombre de passages visible -> invisible
      n_approaches               : nombre d'ENTRÉES dans la zone de saisie.
        ⚠️ ANGLE MORT MESURÉ (2026-09-09) : cette métrique compte les transitions
        dehors->dedans. Or le modèle qui boucle ne SORT jamais de la zone — il
        descend, ferme sur du vide, remonte de quelques centimètres, redescend.
        Résultat : n_approaches = 1,09 sur les échecs contre 1,03 au témoin, ce qui
        laissait croire à tort que la boucle ne se reproduisait pas. Le diagnostic
        sur trajectoires (115_diag_failure_mode.py) montre l'inverse : 3,70 cycles
        haut/bas et 3,41 fermetures de pince par échec, 100 % des échecs avec >=2
        cycles. -> utiliser `z_cycles` ci-dessous, PAS n_approaches.
      z_cycles                   : ⭐ LE bon marqueur de la boucle. Cycles
        descente->remontée de l'effecteur d'amplitude > 2 cm, comptés sans aucune
        référence à la position de la canette — donc insensible au réglage des
        seuils de zone.
      recovered                  : l'épisode a subi >=1 occlusion ET a fini par réussir.
    """

    APPROACH_XY = 0.06   # m — au-dessus de la canette en XY
    APPROACH_Z = 0.05    # m — hauteur eef au-dessus de la canette sous laquelle on "descend"

    def __init__(self, radius):
        self.radius = radius
        self._reset_episode()

    def _reset_episode(self):
        self.occl_steps = 0
        self.total_steps = 0
        self.n_occl_events = 0
        self.n_approaches = 0
        self._was_occluded = False
        self._was_approaching = False
        self.occ_flags = []        # par pas : la cible était-elle masquée ?
        self.appr_flags = []       # par pas : le bras descendait-il sur la zone de saisie ?
        self.eef_z = []            # hauteur de l'effecteur -> cycles haut/bas

    def start_episode(self):
        self._reset_episode()

    def observe(self, env, obs):
        self.total_steps += 1
        occ = can_is_occluded(obs, self.radius)
        if occ:
            self.occl_steps += 1
            if not self._was_occluded:
                self.n_occl_events += 1
        self._was_occluded = occ
        self.occ_flags.append(bool(occ))

        eef = np.asarray(obs["robot0_eef_pos"]).flatten()
        can = np.asarray(obs["object"]).flatten()[7:10]
        approaching = (float(np.linalg.norm(eef[:2] - can[:2])) < self.APPROACH_XY
                       and (eef[2] - can[2]) < self.APPROACH_Z)
        if approaching and not self._was_approaching:
            self.n_approaches += 1
        self._was_approaching = approaching
        self.appr_flags.append(bool(approaching))
        self.eef_z.append(float(eef[2]))

    def summary(self, success):
        n = max(self.total_steps, 1)
        return {
            "z_cycles": z_cycles(self.eef_z),
            "occl_steps": self.occl_steps,
            "occl_fraction": self.occl_steps / n,
            "n_occl_events": self.n_occl_events,
            "n_approaches": self.n_approaches,
            "recovered": bool(success and self.n_occl_events > 0),
        }


def aggregate_occlusion(per_ep):
    """Agrégats dédiés mémoire. `recovery_rate` = succès PARMI les épisodes réellement occlus
    — c'est LA métrique du chantier, le succès global la dilue."""
    occluded = [e for e in per_ep if e.get("n_occl_events", 0) > 0]
    return {
        "n_episodes_occluded": len(occluded),
        "recovery_rate": (sum(e["success"] for e in occluded) / len(occluded)) if occluded else None,
        "occl_fraction_mean": float(np.mean([e.get("occl_fraction", 0.0) for e in per_ep])) if per_ep else 0.0,
        "n_approaches_mean": float(np.mean([e.get("n_approaches", 0) for e in per_ep])) if per_ep else 0.0,
        "z_cycles_mean_failed": (
            float(np.mean([e.get("z_cycles", 0) for e in per_ep if not e["success"]]))
            if any(not e["success"] for e in per_ep) else None),
        "n_approaches_mean_failed": (
            float(np.mean([e.get("n_approaches", 0) for e in per_ep if not e["success"]]))
            if any(not e["success"] for e in per_ep) else None),
    }


# ------------------------------------------------------- enregistrement des trajectoires
class TrajectoryRecorder:
    """Garde les trajectoires des rollouts (proprio, actions exécutées, drapeaux).

    POURQUOI — deux usages, tous deux gratuits en collecte de données (contrainte C2) :

    1. **Diagnostic de la mémoire CAMP.** Le module mémoire est pré-entraîné sur des
       trajectoires EXPERTES, qui ne contiennent jamais d'échec. Rien ne l'a donc poussé à
       distinguer « je remonte avec la canette » de « je remonte après l'avoir ratée » —
       or c'est exactement la distinction dont dépend toute l'utilité de CAMP chez nous.
       Avec ces enregistrements on peut le VÉRIFIER : le code m_t de la 2e approche
       diffère-t-il de celui de la 1re ? Sinon la mémoire ne porte rien.

    2. **Données de pré-entraînement supplémentaires.** Les rollouts qui bouclent SONT des
       trajectoires d'échec — la matière première qui manque, produite par le robot seul,
       en simulation, sans une seule capture de plus. Source illimitée.

    Longueurs variables -> tout est concaténé avec un tableau d'offsets `ep_start`.
    On garde aussi `can_pos` (vérité terrain sim, gratuite) : indispensable pour l'oracle-
    mémoire et pour savoir, après coup, si le modèle visait juste ou faux.
    """

    def __init__(self):
        self.proprio, self.action, self.can_pos = [], [], []
        self.occ, self.appr, self.ep_start, self.ep_meta = [], [], [], []
        self._n = 0

    def start_episode(self):
        self.ep_start.append(self._n)

    def step(self, proprio, action, can_pos):
        self.proprio.append(np.asarray(proprio, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.can_pos.append(np.asarray(can_pos, dtype=np.float32))
        self._n += 1

    def end_episode(self, probe, meta):
        self.occ.extend(probe.occ_flags[:self._n - self.ep_start[-1]])
        self.appr.extend(probe.appr_flags[:self._n - self.ep_start[-1]])
        self.ep_meta.append(meta)

    def save(self, path, **extra):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            proprio=np.stack(self.proprio) if self.proprio else np.zeros((0, 0), np.float32),
            action=np.stack(self.action) if self.action else np.zeros((0, 0), np.float32),
            can_pos=np.stack(self.can_pos) if self.can_pos else np.zeros((0, 3), np.float32),
            occluded=np.asarray(self.occ, dtype=bool),
            approaching=np.asarray(self.appr, dtype=bool),
            ep_start=np.asarray(self.ep_start, dtype=np.int64),
            ep_meta=np.asarray(json.dumps(self.ep_meta)),
            **extra)
        return path
