#!/usr/bin/env python3
"""Test de reconversion DELTA : rejoue les démos Robomimic en appliquant cible = jpos_courant + delta,
où delta = jpos_demo[i+1] - jpos_demo[i]. Si le robot réussit la tâche -> la reconversion delta de
l'éval (37 --delta) est CORRECTE (donc A échoue pour une vraie raison). Si 0% -> bug de reconversion.
Compare aussi le replay ABSOLU (cible = jpos_demo[i+1]) comme référence (doit marcher).
  venv312/bin/python experiments/can/97_replay_delta_test.py --n 10
"""
import argparse, numpy as np, h5py
import robosuite as rs
from robosuite.controllers import load_composite_controller_config

HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"

def build_env(kp):
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    arm = {"type": "JOINT_POSITION", "input_type": "absolute", "kp": kp, "damping_ratio": 1,
           "impedance_mode": "fixed", "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
           "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2, "gripper": {"type": "GRIP"}}
    cfg["body_parts"]["right"] = arm
    env = rs.make("PickPlaceCan", robots="Panda", controller_configs=cfg,
                  has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                  reward_shaping=False, control_freq=20, ignore_done=True, hard_reset=False)
    return env

def run(mode, n, kp):
    env = build_env(kp)
    ok = 0
    with h5py.File(HDF5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[:n]
        for dn in demos:
            d = f["data"][dn]
            states = d["states"][:]; acts = d["actions"][:]
            jpos = d["obs"]["robot0_joint_pos"][:]   # (T,7)
            env.reset()
            env.sim.set_state_from_flattened(states[0]); env.sim.forward()
            obs = env._get_observations()
            success = False
            for i in range(states.shape[0]):
                jt_next = jpos[i + 1] if i + 1 < jpos.shape[0] else jpos[i]
                if mode == "absolu":
                    tgt = jt_next
                else:  # delta : jpos_courant (obs du step précédent) + (jt_next - jpos[i])
                    cur = np.asarray(obs["robot0_joint_pos"]).flatten()
                    tgt = cur + (jt_next - jpos[i])
                a = np.concatenate([tgt, [acts[i, 6]]]).astype(np.float32)
                obs = env.step(a)[0]
                if env._check_success():
                    success = True; break
            ok += int(success)
    env.close()
    return ok, n

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--kp", type=float, default=50)
    a = ap.parse_args()
    for mode in ["absolu", "delta"]:
        k, n = run(mode, a.n, a.kp)
        print(f"REPLAY {mode:7} : {k}/{n} = {100*k/n:.0f}% succès")
    print("-> si delta ~= absolu (haut) : reconversion CORRECTE (A échoue vraiment) ; si delta=0 : BUG reconversion")
