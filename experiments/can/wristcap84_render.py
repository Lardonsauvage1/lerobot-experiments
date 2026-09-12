"""Rendu Can 224px pour WRISTCAP+aug : agentview + robot0_eye_in_hand (poignet), SANS jitter.
Produit EN IMAGES (use_videos=False -> prêt gb10) et EN UN SEUL passage sim :
  - can_ph_proprio_84_img        : agentview seule            (bras A')
  - can_ph_proprio_wrist_84_img  : agentview + poignet        (bras B')
State 9D proprio, action 7D cartésien, 200 démos. Sur Principal (robosuite).
Smoke : python wristcap224_render.py --smoke  (2 démos)."""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
from pathlib import Path
import h5py, numpy as np

HDF5_PATH = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
IMG = 84; FPS = 20
TASK = "Pick up the can and place it in the bin."
DS_AGENT = ("local/can_ph_proprio_84_img", Path("data_cache/lerobot_can_ph_proprio_84_img"))
DS_WRIST = ("local/can_ph_proprio_wrist_84_img", Path("data_cache/lerobot_can_ph_proprio_wrist_84_img"))


def cam_id(m, name):
    try: return m.camera_name2id(name)
    except Exception:
        for i in range(m.ncam):
            try:
                if m.camera(i).name == name: return i
            except Exception: pass
    raise ValueError(name)


def main(smoke=False):
    import shutil
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs
    base = ["eef_pos_x","eef_pos_y","eef_pos_z","eef_quat_w","eef_quat_x","eef_quat_y","eef_quat_z","gripper_l","gripper_r"]
    img_feat = {"dtype":"image","shape":(3,IMG,IMG),"names":["channels","height","width"]}
    st = {"dtype":"float32","shape":(9,),"names":base}
    ac = {"dtype":"float32","shape":(7,),"names":["dx","dy","dz","drx","dry","drz","gripper"]}
    for _, root in (DS_AGENT, DS_WRIST):
        if root.exists(): shutil.rmtree(root)
    common = dict(fps=FPS, robot_type="panda", use_videos=False,
                  image_writer_processes=0, image_writer_threads=4, metadata_buffer_size=1)
    ds_a = LeRobotDataset.create(repo_id=DS_AGENT[0], root=DS_AGENT[1],
        features={"observation.images.agentview":img_feat, "observation.state":st, "action":ac}, **common)
    ds_w = LeRobotDataset.create(repo_id=DS_WRIST[0], root=DS_WRIST[1],
        features={"observation.images.agentview":img_feat, "observation.images.wrist":img_feat,
                  "observation.state":st, "action":ac}, **common)

    env = rs.make(env_name="PickPlaceCan", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=False, camera_names="agentview", camera_heights=IMG,
                  camera_widths=IMG, control_freq=FPS, reward_shaping=False)
    ids = None
    with h5py.File(HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if smoke: demos = demos[:2]
        print(f"{len(demos)} démos -> 224px agentview+poignet (2 datasets)")
        for d_idx, dn in enumerate(demos):
            demo = f["data"][dn]; states = demo["states"][:]; actions = demo["actions"][:]; o = demo["obs"]
            state9 = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:], o["robot0_gripper_qpos"][:]], axis=1)
            env.reset()
            m = env.sim.model
            if ids is None:
                ids = {c: cam_id(m, c) for c in ("agentview", "robot0_eye_in_hand")}
            for k in range(states.shape[0]):
                env.sim.set_state_from_flattened(states[k]); env.sim.forward()
                av = np.ascontiguousarray(env.sim.render(height=IMG, width=IMG, camera_name="agentview")[::-1].transpose(2,0,1))
                wr = np.ascontiguousarray(env.sim.render(height=IMG, width=IMG, camera_name="robot0_eye_in_hand")[::-1].transpose(2,0,1))
                s = state9[k].astype(np.float32); a = actions[k].astype(np.float32)
                ds_a.add_frame({"observation.images.agentview":av, "observation.state":s, "action":a, "task":TASK})
                ds_w.add_frame({"observation.images.agentview":av, "observation.images.wrist":wr,
                                "observation.state":s, "action":a, "task":TASK})
            ds_a.save_episode(); ds_w.save_episode()
            if (d_idx+1) % 20 == 0: print(f"  {d_idx+1}/{len(demos)}", flush=True)
    env.close()
    print(f"✓ A' {DS_AGENT[1]} frames={ds_a.num_frames} ép={ds_a.num_episodes}")
    print(f"✓ B' {DS_WRIST[1]} frames={ds_w.num_frames} ép={ds_w.num_episodes}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
