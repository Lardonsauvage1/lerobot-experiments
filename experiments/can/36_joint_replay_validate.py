"""ÉTAPE 0 (de-risk) — valider l'espace d'action ARTICULAIRE avant tout entraînement.

Rejoue en OPEN-LOOP les cibles articulaires des démos (action = joint_pos[t+1] absolu,
+ gripper de l'action OSC d'origine) dans un env contrôlé en JOINT_POSITION absolu.
Si Can réussit en replay → l'espace d'action est valide, une politique peut l'apprendre.
Sinon → ajuster (kp, ranges) avant de perdre des heures d'entraînement.

Usage : venv312/bin/python -u experiments/can/36_joint_replay_validate.py [--n 20] [--kp 50]
"""
import argparse, sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np
import h5py

HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"


def build_joint_env(kp):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_joint_pos", "robot0_gripper_qpos", "object"], "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5)
    # contrôleur arm = JOINT_POSITION ABSOLU (action = cibles joints brutes en rad)
    arm = {
        "type": "JOINT_POSITION", "input_type": "absolute",
        "kp": kp, "damping_ratio": 1, "impedance_mode": "fixed",
        "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
        "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2,
        "input_max": 1, "input_min": -1, "output_max": 1, "output_min": -1,
        "gripper": {"type": "GRIP"},
    }
    env_meta["env_kwargs"]["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kp", type=float, default=50)
    a = ap.parse_args()

    env = build_joint_env(a.kp)
    f = h5py.File(HDF5, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[: a.n]
    print(f"[36] replay open-loop JOINT_POSITION absolu, kp={a.kp}, {len(demos)} démos", flush=True)

    succ = 0
    for di, dn in enumerate(demos):
        d = f[f"data/{dn}"]
        jpos = d["obs/robot0_joint_pos"][:]          # (T,7)
        grip = d["actions"][:, 6]                    # (T,) commande gripper OSC
        init = d["states"][0]
        env.reset()
        env.reset_to({"states": init})
        ok = False
        T = jpos.shape[0]
        for t in range(T - 1):
            action = np.concatenate([jpos[t + 1], [grip[t]]]).astype(np.float32)  # 7 joints abs + grip
            env.step(action)
            if env.is_success()["task"]:
                ok = True; break
        succ += ok
        print(f"  démo {di:2d}: {'OK' if ok else 'ÉCHEC'}  ({t+1}/{T-1} steps)", flush=True)

    print(f"\n[36] REPLAY SUCCÈS : {succ}/{len(demos)} = {succ/len(demos):.0%}  (kp={a.kp})", flush=True)
    print("→ si ~100% : espace d'action articulaire VALIDE, on peut entraîner.", flush=True)


if __name__ == "__main__":
    main()
