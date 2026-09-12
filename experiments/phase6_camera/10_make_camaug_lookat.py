"""Dataset Can AUGMENTE version LOOK-AT (corrige les défauts du camaug d'origine).

Pourquoi : le 1er camaug (04) tournait la caméra autour d'un AXE ALEATOIRE -> introduisait
du ROLL (image qui tourne sur elle-même), variation irréaliste et incohérente -> modèle à 2%.
La littérature qui marche (arXiv 2511.09932 : ~0 -> 0.6-1.0) échantillonne des poses caméra
ORIENTEES VERS LA SCENE (look-at), sans roll parasite.

Ici, par épisode et PAR CAMERA INDEPENDANTE (réaliste : 2 caméras dérivent chacune de leur côté) :
  - position = canonique + dérive (uniforme 0 -> TRANS_MAX, direction aléatoire)
  - orientation = LOOK-AT du plan de travail (ré-orientation vers la cible) + pan/tilt (0 -> ROT_MAX)
    SANS ROLL (rotation seulement dans le plan horizontal/vertical de la caméra, jamais autour
    de l'axe optique).
Mêmes démos / actions / états que can_to_lerobot — SEULES les images changent.

Sortie : data_cache/lerobot_can_ph_proprio_birdview_lookat  (repo local/can_ph_proprio_birdview_lookat)
"""
import sys, math
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
from pathlib import Path
import h5py, numpy as np

HDF5_PATH = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUT = Path("data_cache/lerobot_can_ph_proprio_birdview_lookat")
REPO_ID = "local/can_ph_proprio_birdview_lookat"
IMAGE_SIZE = 96; FPS = 20
TASK = "Pick up the can and place it in the bin."
TRANS_MAX = 0.20            # 20 cm (couvre la cible 10 cm avec marge ; uniforme 0->max inclut le proche-canonique)
ROT_MAX = math.radians(15)  # 15 deg de pan/tilt (couvre 10° avec marge), SANS roll
Z_TABLE = 0.85              # hauteur du plan de travail (cible du look-at)
SEED = 42


def aa_quat(ax, ang):
    ax = np.asarray(ax, float); n = np.linalg.norm(ax)
    if n < 1e-9 or abs(ang) < 1e-9: return np.array([1.0, 0, 0, 0])
    ax = ax/n; h = ang/2; s = math.sin(h)
    return np.array([math.cos(h), s*ax[0], s*ax[1], s*ax[2]])

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def q2m(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def m2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t+1)*2; w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return np.array([w,x,y,z])

def lookat_quat(eye, tgt):
    z = eye - tgt; z = z/np.linalg.norm(z)                       # +Z s'éloigne de la cible (cam regarde -Z)
    up = np.array([0,0,1.]) if abs(z[2]) < 0.95 else np.array([0,1.,0])
    x = np.cross(up, z); x = x/np.linalg.norm(x)
    y = np.cross(z, x)
    return m2q(np.stack([x, y, z], axis=1))

def cam_id(m, name):
    try: return m.camera_name2id(name)
    except Exception:
        for i in range(m.ncam):
            try:
                if m.camera(i).name == name: return i
            except Exception: pass
    raise ValueError(name)

def perturb_lookat(p0, q0, rng):
    """Position dérivée + look-at + pan/tilt sans roll. Retourne (new_pos, new_quat)."""
    f = q2m(q0) @ np.array([0, 0, -1.])                          # direction de visée canonique
    t = (Z_TABLE - p0[2]) / f[2]; target = p0 + f * t           # cible = intersection rayon/plan
    dvec = rng.normal(size=3); dvec = dvec/(np.linalg.norm(dvec)+1e-9) * rng.uniform(0, TRANS_MAX)
    new_pos = p0 + dvec
    base = lookat_quat(new_pos, target)
    R = q2m(base); xax, yax = R[:, 0], R[:, 1]                   # axes caméra droite/haut
    phi = rng.uniform(0, 2*math.pi); axis = math.cos(phi)*xax + math.sin(phi)*yax  # plan -> pan/tilt, PAS de roll
    ang = rng.uniform(0, ROT_MAX)
    return new_pos, qmul(aa_quat(axis, ang), base)


def main(smoke=False, n_demos=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs
    global OUT, REPO_ID
    if n_demos:
        OUT = Path(f"data_cache/lerobot_can_ph_proprio_birdview_lookat{n_demos}")
        REPO_ID = f"local/can_ph_proprio_birdview_lookat{n_demos}"
    if OUT.exists():
        print(f"⚠️  {OUT} existe déjà. Supprime-le pour relancer."); return
    base = ["eef_pos_x","eef_pos_y","eef_pos_z","eef_quat_w","eef_quat_x","eef_quat_y","eef_quat_z","gripper_l","gripper_r"]
    img_feat = {"dtype":"video","shape":(3,IMAGE_SIZE,IMAGE_SIZE),"names":["channels","height","width"]}
    features = {
        "observation.images.agentview": img_feat, "observation.images.birdview": img_feat,
        "observation.state": {"dtype":"float32","shape":(9,),"names":base},
        "action": {"dtype":"float32","shape":(7,),"names":["dx","dy","dz","drx","dry","drz","gripper"]},
    }
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=features, root=OUT,
                               robot_type="panda", use_videos=True, video_backend="pyav")
    env = rs.make(env_name="PickPlaceCan", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=False, camera_names="agentview", camera_heights=IMAGE_SIZE,
                  camera_widths=IMAGE_SIZE, control_freq=FPS, reward_shaping=False)
    ids = None

    with h5py.File(HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if smoke: demos = demos[:2]
        elif n_demos: demos = demos[:n_demos]
        print(f"{len(demos)} démos -> LOOK-AT (dérive <= {TRANS_MAX*100:.0f}cm + pan/tilt <= {math.degrees(ROT_MAX):.0f}°, sans roll, par épisode)...")
        for d_idx, dn in enumerate(demos):
            demo = f["data"][dn]; states = demo["states"][:]; actions = demo["actions"][:]; o = demo["obs"]
            state9 = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:], o["robot0_gripper_qpos"][:]], axis=1)
            env.reset()
            m = env.sim.model
            if ids is None:
                ids = {c: cam_id(m, c) for c in ("agentview", "birdview")}
            rng = np.random.default_rng(SEED + d_idx)
            for c, i in ids.items():
                p0, q0 = m.cam_pos[i].copy(), m.cam_quat[i].copy()
                new_pos, new_quat = perturb_lookat(p0, q0, rng)
                m.cam_pos[i] = new_pos; m.cam_quat[i] = new_quat
            env.sim.forward()
            for k in range(states.shape[0]):
                env.sim.set_state_from_flattened(states[k]); env.sim.forward()
                fr = {"observation.state": state9[k].astype(np.float32),
                      "action": actions[k].astype(np.float32), "task": TASK}
                for key, cam in [("observation.images.agentview","agentview"),("observation.images.birdview","birdview")]:
                    img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name=cam)[::-1]
                    fr[key] = np.ascontiguousarray(img.transpose(2,0,1))
                ds.add_frame(fr)
            ds.save_episode()
            if (d_idx+1) % 20 == 0: print(f"  {d_idx+1}/{len(demos)}", flush=True)
    env.close()
    print(f"\n✓ Dataset look-at : {OUT} | frames={ds.num_frames} épisodes={ds.num_episodes}")


if __name__ == "__main__":
    nd = None
    for i, a in enumerate(sys.argv):
        if a == "--n-demos": nd = int(sys.argv[i+1])
    main(smoke="--smoke" in sys.argv, n_demos=nd)
