import sys; sys.path.insert(0,".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
import numpy as np
from src import can_eval

STEP = 0.0105          # m par unité d'action (calibré)
BIN  = np.array([0.200, 0.376, 0.980])   # calibré : 10/10
GRASP_Z = 0.877        # hauteur de fermeture des démos expertes
APPROACH_DZ = 0.10

def can_of(obs):  return np.asarray(obs["object"]).flatten()[7:10]
def eef_of(obs):  return np.asarray(obs["robot0_eef_pos"]).flatten()

def go(target, eef, grip, gain=0.8):
    d = np.clip((target - eef) / STEP * gain, -1, 1)
    return np.array([d[0], d[1], d[2], 0, 0, 0, grip], dtype=np.float32)

def scripted_episode(env, state, perturb=None, seed=0):
    """Retourne (succès, liste d'actions, liste d'obs). perturb = décalage (m) appliqué
    juste avant la descente, pour fabriquer un RATTRAPAGE."""
    rng = np.random.default_rng(seed)
    obs = env.reset_to(dict(states=state))
    acts, obss = [], []
    def step(a):
        nonlocal obs
        acts.append(a.copy()); obss.append(obs)
        obs = env.step(a)[0]
    can = can_of(obs)
    # 1) se placer au-dessus
    for _ in range(60):
        e = eef_of(obs); t = np.array([can[0], can[1], can[2] + APPROACH_DZ])
        if np.linalg.norm(e[:2] - t[:2]) < 0.004 and abs(e[2]-t[2]) < 0.01: break
        step(go(t, e, -1))
    # 1bis) PERTURBATION : on s'écarte volontairement -> l'épisode devient un rattrapage
    if perturb is not None:
        off = np.array([perturb[0], perturb[1], perturb[2]])
        tgt = eef_of(obs) + off
        for _ in range(25):
            e = eef_of(obs)
            if np.linalg.norm(e - tgt) < 0.006: break
            step(go(tgt, e, -1))
        # on redescend un peu comme si on allait saisir, puis on referme dans le vide
        for _ in range(12): step(go(eef_of(obs) + np.array([0,0,-0.02]), eef_of(obs), -1))
        for _ in range(4):  step(np.array([0,0,0,0,0,0,1], dtype=np.float32))   # ferme à vide
        # ⚠️ si cette fermeture a RÉELLEMENT attrapé la canette, rouvrir apprendrait au
        # modèle à lâcher une prise réussie. On invalide l'épisode plutôt que de le montrer.
        for _ in range(6):
            e = eef_of(obs); step(go(e + np.array([0,0,0.05]), e, 1))
        if can_of(obs)[2] > 0.90:
            return None, acts, obss          # None = à jeter, pas un échec
        for _ in range(4):  step(np.array([0,0,0,0,0,0,-1], dtype=np.float32))  # rouvre
        # RATTRAPAGE : remonter puis se recentrer
        can = can_of(obs)
        for _ in range(40):
            e = eef_of(obs); t = np.array([can[0], can[1], can[2] + APPROACH_DZ])
            if np.linalg.norm(e[:2]-t[:2]) < 0.004 and abs(e[2]-t[2]) < 0.012: break
            step(go(t, e, -1))
    # 2) descendre
    can = can_of(obs)
    for _ in range(45):
        e = eef_of(obs); t = np.array([can[0], can[1], GRASP_Z])
        if abs(e[2] - t[2]) < 0.004 and np.linalg.norm(e[:2]-t[:2]) < 0.006: break
        step(go(t, e, -1, gain=0.5))
    # 3-4) fermer, VÉRIFIER, et recommencer si la canette n'a pas suivi.
    #      C'est le geste que les démonstrations expertes ne contiennent jamais,
    #      et précisément celui qui manque au modèle : constater l'échec et réessayer.
    lifted = False
    for attempt in range(3):
        for _ in range(12): step(np.array([0,0,0,0,0,0,1], dtype=np.float32))
        for _ in range(25):
            e = eef_of(obs)
            if e[2] > 1.02: break
            step(go(np.array([e[0], e[1], 1.05]), e, 1))
        if can_of(obs)[2] > 0.90:          # la canette a bien été soulevée
            lifted = True; break
        # RATTRAPAGE : rouvrir, se recentrer sur la canette, redescendre
        for _ in range(6): step(np.array([0,0,0,0,0,0,-1], dtype=np.float32))
        can = can_of(obs)
        for _ in range(45):
            e = eef_of(obs); t = np.array([can[0], can[1], can[2] + APPROACH_DZ])
            if np.linalg.norm(e[:2]-t[:2]) < 0.004 and abs(e[2]-t[2]) < 0.012: break
            step(go(t, e, -1))
        for _ in range(45):
            e = eef_of(obs); t = np.array([can[0], can[1], GRASP_Z])
            if abs(e[2]-t[2]) < 0.004 and np.linalg.norm(e[:2]-t[:2]) < 0.006: break
            step(go(t, e, -1, gain=0.5))
    if not lifted:
        return False, acts, obss
    # 5) transporter
    for _ in range(70):
        e = eef_of(obs)
        if np.linalg.norm(e - BIN) < 0.03: break
        step(go(BIN, e, 1))
    # 6) lâcher
    for _ in range(10): step(np.array([0,0,0,0,0,0,-1], dtype=np.float32))
    # filtre indépendant du mécanisme : compter les FRONTS montants de la canette.
    # Un rattrapage propre n'en a qu'un ; « attrape, lâche, rattrape » en a deux.
    z = np.array([np.asarray(o["object"]).flatten()[9] for o in obss])
    up = z > 0.90
    fronts = int(np.sum((~up[:-1]) & up[1:]))
    if fronts > 1:
        return None, acts, obss              # à jeter
    return bool(env.is_success()["task"]), acts, obss

if __name__ == "__main__":
    import importlib.util
    spec=importlib.util.spec_from_file_location("v","experiments/can/10_vision_500_rollouts.py")
    v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    states=v.make_eval_states()
    env=can_eval.make_env()
    ok=0
    for i in range(12):
        s,a,_=scripted_episode(env, states[i])
        ok+=s
        print(f"    ép {i:2d} : {'OK ' if s else 'RATE'}  ({len(a)} pas)", flush=True)
    print(f"\n  contrôleur scripté SANS perturbation : {ok}/12")
