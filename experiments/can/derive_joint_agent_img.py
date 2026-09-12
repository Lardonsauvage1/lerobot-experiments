"""Dérive can_ph_joint_birdview -> can_ph_joint_agent_96_img : agentview SEULE, en IMAGES (gb10-ready)."""
import shutil
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
src = LeRobotDataset("local/can_ph_joint_birdview", root="data_cache/lerobot_can_ph_joint_birdview")
DROOT = Path("data_cache/lerobot_can_ph_joint_agent_96_img")
if DROOT.exists(): shutil.rmtree(DROOT)
sf = src.meta.features
feats = {
  "observation.images.agentview": {"dtype": "image", "shape": (3, 96, 96), "names": ["channels", "height", "width"]},
  "observation.state": {"dtype": "float32", "shape": tuple(sf["observation.state"]["shape"]), "names": sf["observation.state"].get("names")},
  "action": {"dtype": "float32", "shape": tuple(sf["action"]["shape"]), "names": sf["action"].get("names")},
}
dst = LeRobotDataset.create(repo_id="local/can_ph_joint_agent_96_img", fps=int(src.fps), features=feats, root=DROOT,
                            use_videos=False, image_writer_processes=0, image_writer_threads=4, metadata_buffer_size=1)
def hwc(t):
    a = t.numpy() if hasattr(t, "numpy") else t
    if a.ndim == 3 and a.shape[0] in (1, 3): a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8: a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)
TASK = src[0].get("task") if isinstance(src[0].get("task"), str) else "pick and place the can"
cur = 0
for i in range(src.num_frames):
    f = src[i]; ep = int(f["episode_index"])
    if ep != cur: dst.save_episode(); cur = ep; print(f"ép {ep}", flush=True) if ep % 40 == 0 else None
    dst.add_frame({"observation.images.agentview": hwc(f["observation.images.agentview"]),
                   "observation.state": f["observation.state"].numpy(), "action": f["action"].numpy(), "task": TASK})
dst.save_episode()
print(f"TERMINE: {dst.meta.total_episodes} ép, action {tuple(feats['action']['shape'])}, state {tuple(feats['observation.state']['shape'])}")
