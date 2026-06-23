"""Dataset Can AUGMENTE : meme pipeline que can_to_lerobot (proprio + agentview + birdview),
mais les 2 cameras sont DECALEES aleatoirement PAR EPISODE (translation <=10cm + rotation <=10deg,
direction aleatoire, fixe pendant tout l'episode = comme une camera re-fixee de travers).
But : entrainer un modele robuste au deplacement de camera (sim-to-real).

Sortie : data_cache/lerobot_can_ph_proprio_birdview_camaug  (repo local/can_ph_proprio_birdview_camaug)
"""
import sys, math
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
from pathlib import Path
import h5py, numpy as np

HDF5_PATH = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUT = Path("data_cache/lerobot_can_ph_proprio_birdview_camaug")
REPO_ID = "local/can_ph_proprio_birdview_camaug"
IMAGE_SIZE = 96; FPS = 20
TASK = "Pick up the can and place it in the bin."
TRANS_MAX = 0.10            # 10 cm
ROT_MAX = math.radians(10)  # 10 deg
SEED = 42


def aa_quat(ax, ang):
    ax = np.asarray(ax, float); n = np.linalg.norm(ax)
    if n < 1e-9 or abs(ang) < 1e-9: return np.array([1.0, 0, 0, 0])
    ax /= n; h = ang/2; s = math.sin(h)
    return np.array([math.cos(h), s*ax[0], s*ax[1], s*ax[2]])

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def cam_id(m, name):
    try: return m.camera_name2id(name)
    except Exception:
        for i in range(m.ncam):
            try:
                if m.camera(i).name == name: return i
            except Exception: pass
    raise ValueError(name)


def main(smoke=False, n_demos=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs
    global OUT, REPO_ID
    if n_demos:  # dataset reduit (ex. 60 demos pour le run 30-demos : train 0-29, val 30-59)
        OUT = Path(f"data_cache/lerobot_can_ph_proprio_birdview_camaug{n_demos}")
        REPO_ID = f"local/can_ph_proprio_birdview_camaug{n_demos}"
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
    ids = None  # capturés APRÈS le 1er reset (reset() recrée env.sim -> un m capturé avant est périmé !)

    with h5py.File(HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if smoke: demos = demos[:2]
        elif n_demos: demos = demos[:n_demos]
        print(f"{len(demos)} démos à augmenter (jitter <= {TRANS_MAX*100:.0f}cm / {math.degrees(ROT_MAX):.0f}°, par épisode)...")
        for d_idx, dn in enumerate(demos):
            demo = f["data"][dn]; states = demo["states"][:]; actions = demo["actions"][:]; o = demo["obs"]
            state9 = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:], o["robot0_gripper_qpos"][:]], axis=1)
            env.reset()
            m = env.sim.model  # FRAIS après reset (sinon périmé)
            if ids is None:
                ids = {c: cam_id(m, c) for c in ("agentview", "birdview")}
            # --- jitter PAR EPISODE (independant par camera), fixe pour tout l'episode ---
            rng = np.random.default_rng(SEED + d_idx)
            for c, i in ids.items():
                p0, q0 = m.cam_pos[i].copy(), m.cam_quat[i].copy()  # pose par défaut du sim courant
                dvec = rng.normal(size=3); dvec = dvec/(np.linalg.norm(dvec)+1e-9) * rng.uniform(0, TRANS_MAX)
                m.cam_pos[i] = p0 + dvec
                ax = rng.normal(size=3); ang = rng.uniform(0, ROT_MAX)
                m.cam_quat[i] = qmul(aa_quat(ax, ang), q0)
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
    print(f"\n✓ Dataset augmenté : {OUT} | frames={ds.num_frames} épisodes={ds.num_episodes}")


if __name__ == "__main__":
    nd = None
    for i, a in enumerate(sys.argv):
        if a == "--n-demos": nd = int(sys.argv[i+1])
    main(smoke="--smoke" in sys.argv, n_demos=nd)
