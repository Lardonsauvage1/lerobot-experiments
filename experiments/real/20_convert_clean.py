#!/usr/bin/env python
"""Re-conversion NETTOYÉE des mcaps -> LeRobot `local/apple_joint_224_clean`.
Identique à 02_convert_all.py MAIS rogne les frames statiques (~50 début / ~10 fin par épisode,
= 20% de frames mortes) en gardant MARGE=3 frames avant/après le mouvement (pour garder le
démarrage/arrêt du geste). Seuil mouvement = 1° vs pose de repos."""
import shutil
from pathlib import Path
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FS, IMG, MARGIN, THR = 15.0, 224, 3, np.deg2rad(1.0)
TASK = "pick the white apple and place it at the drop point"
SRC = Path("/Users/nielsmurawka/Documents/VScodeProject/NeuroneImitationCarote/Demis_dataset/batch_689428411_20260712_025841")
REPO, ROOT = "local/apple_joint_224_clean", Path("data_cache/lerobot_apple_joint_224_clean")
TS = get_typestore(Stores.ROS2_JAZZY)
LEFT,RIGHT,JOINTS,GRIP = ("/head_camera/left/image_raw/compressed","/head_camera/right/image_raw/compressed","/joint_states","/gripper")

def read_bag(bag):
    jt,jv,lt,lraw,rt,rraw,gt,gv=[],[],[],[],[],[],[],[]
    with Reader(bag) as r:
        for c,ts,raw in r.messages():
            t=c.topic
            if t==JOINTS: m=TS.deserialize_cdr(raw,c.msgtype); jt.append(ts); jv.append(np.asarray(m.position,np.float32))
            elif t==LEFT: m=TS.deserialize_cdr(raw,c.msgtype); lt.append(ts); lraw.append(bytes(m.data))
            elif t==RIGHT: m=TS.deserialize_cdr(raw,c.msgtype); rt.append(ts); rraw.append(bytes(m.data))
            elif t==GRIP: m=TS.deserialize_cdr(raw,c.msgtype); gt.append(ts); gv.append(bool(m.data))
    jt=np.array(jt); o=np.argsort(jt); jt,jv=jt[o],[jv[i] for i in o]
    return dict(jt=jt,jv=np.stack(jv),lt=np.array(lt),lraw=lraw,rt=np.array(rt),rraw=rraw,gt=np.array(gt),gv=gv)

def decode(j):
    im=cv2.imdecode(np.frombuffer(j,np.uint8),cv2.IMREAD_COLOR); im=cv2.resize(im,(IMG,IMG),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
def nearest(times,t):
    i=np.searchsorted(times,t)
    return 0 if i==0 else len(times)-1 if i>=len(times) else (i if abs(times[i]-t)<abs(times[i-1]-t) else i-1)
def grip_hold(gt,gv,t):
    p=False
    for te,ve in zip(gt,gv):
        if te<=t: p=ve
        else: break
    return p

def episode_frames(bag):
    s=read_bag(bag)
    t0=max(s["jt"][0],s["lt"][0],s["rt"][0]); t1=min(s["jt"][-1],s["lt"][-1],s["rt"][-1])
    n=int((t1-t0)/1e9*FS); grid=t0+(np.arange(n)/FS*1e9).astype(np.int64)
    joints=np.stack([s["jv"][nearest(s["jt"],t)] for t in grid]).astype(np.float32)
    grip=np.array([grip_hold(s["gt"],s["gv"],t) for t in grid],np.float32)
    # --- fenêtre de mouvement (sur joints) ---
    dev0=np.abs(joints-joints[0]).max(1); lead=int(np.argmax(dev0>THR)) if (dev0>THR).any() else 0
    devN=np.abs(joints-joints[-1]).max(1); mov=np.where(devN>THR)[0]; mend=int(mov[-1]) if len(mov) else n-1
    lo=max(0,lead-MARGIN); hi=min(n-1,mend+MARGIN)          # bornes gardées (indices d'obs)
    frames=[]
    for t in range(lo,min(hi,n-2)+1):                       # action(t)=pose suivante -> t<=n-2
        frames.append(dict(left=decode(s["lraw"][nearest(s["lt"],grid[t])]),
                           right=decode(s["rraw"][nearest(s["rt"],grid[t])]),
                           state=joints[t].copy(),
                           action=np.concatenate([joints[t+1],grip[t+1:t+2]]).astype(np.float32)))
    return frames, n, lo, hi

def main():
    eps=sorted(p for p in SRC.glob("ep_*") if p.is_dir())
    if ROOT.exists(): shutil.rmtree(ROOT)
    feats={
      "observation.images.left":{"dtype":"video","shape":(3,IMG,IMG),"names":["channels","height","width"]},
      "observation.images.right":{"dtype":"video","shape":(3,IMG,IMG),"names":["channels","height","width"]},
      "observation.state":{"dtype":"float32","shape":(5,),"names":[f"joint_{i+1}" for i in range(5)]},
      "action":{"dtype":"float32","shape":(6,),"names":[f"jt_{i+1}" for i in range(5)]+["gripper"]}}
    ds=LeRobotDataset.create(repo_id=REPO,fps=int(FS),features=feats,root=ROOT,robot_type="real5dof",
                             use_videos=True,image_writer_processes=0,image_writer_threads=4)
    tot=0
    for bag in eps:
        try: frames,n,lo,hi=episode_frames(bag)
        except Exception as e: print(f"!! {bag.name} ECHEC {e}"); continue
        for fr in frames:
            ds.add_frame({"observation.images.left":fr["left"],"observation.images.right":fr["right"],
                          "observation.state":fr["state"],"action":fr["action"],"task":TASK})
        ds.save_episode(); tot+=len(frames)
        print(f"{bag.name}: {n} brutes -> gardé [{lo}:{hi}] = {len(frames)} frames (cumul {tot})",flush=True)
    print(f"TERMINE: {ds.meta.total_episodes} ép, {tot} frames -> {ROOT}")

if __name__=="__main__": main()
