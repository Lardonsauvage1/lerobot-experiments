"""Galerie visuelle de la perturbation camera : 4 niveaux x 5 angles aleatoires = 20 images.
Meme scene (1 etat fige), seule la camera agentview bouge (translation + rotation aleatoires)."""
import sys, os, math; sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
from src import can_eval

CAM = "agentview"; RES = 256
LEVELS = [(0,0), (2,2), (5,5), (10,10)]  # (trans_cm, rot_deg)
NCOL = 5

def aa_quat(ax, ang):
    ax=np.asarray(ax,float); n=np.linalg.norm(ax)
    if n<1e-9 or abs(ang)<1e-9: return np.array([1.0,0,0,0])
    ax/=n; h=ang/2; s=math.sin(h); return np.array([math.cos(h),s*ax[0],s*ax[1],s*ax[2]])
def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

states = np.load("results/runs/can/can_eval500.npy")
env = can_eval.make_env(); env.reset_to(dict(states=states[0]))
m = env.env.sim.model
try: cid = m.camera_name2id(CAM)
except Exception:
    cid = next(i for i in range(m.ncam) if m.camera(i).name==CAM)
p0 = m.cam_pos[cid].copy(); q0 = m.cam_quat[cid].copy()

fig, axes = plt.subplots(len(LEVELS), NCOL, figsize=(NCOL*2.4, len(LEVELS)*2.4))
for r,(tcm,rdeg) in enumerate(LEVELS):
    for c in range(NCOL):
        rng = np.random.default_rng(1000*r + c)
        if tcm>0:
            d = rng.normal(size=3); d = d/(np.linalg.norm(d)+1e-9)*rng.uniform(0,tcm/100.0)
        else: d = np.zeros(3)
        m.cam_pos[cid] = p0 + d
        if rdeg>0:
            ax_ = rng.normal(size=3); ang = rng.uniform(0, math.radians(rdeg))
            m.cam_quat[cid] = qmul(aa_quat(ax_,ang), q0)
        else: m.cam_quat[cid] = q0
        env.env.sim.forward()
        img = env.env.sim.render(height=RES, width=RES, camera_name=CAM)[::-1]
        a = axes[r][c]; a.imshow(img); a.set_xticks([]); a.set_yticks([])
        if c==0:
            lab = "baseline (0)" if tcm==0 else f"≤{tcm}cm / ≤{rdeg}°"
            a.set_ylabel(lab, fontsize=11, weight="bold")
m.cam_pos[cid]=p0; m.cam_quat[cid]=q0
try: env.env.close()
except Exception: pass
fig.suptitle(f"Perturbation caméra {CAM} — 4 niveaux × 5 décalages aléatoires (même scène)", fontsize=13, weight="bold")
fig.tight_layout()
out = "results/runs/phase5_methodology/camshift_gallery.png"; fig.savefig(out, dpi=130)
print(f"écrit : {out}")
