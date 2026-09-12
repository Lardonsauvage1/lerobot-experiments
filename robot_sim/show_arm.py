#!/usr/bin/env python
"""Visualiseur du bras (5 joints) à une pose articulaire donnée, à partir de l'URDF + STL.
Usage:  python show_arm.py j1 j2 j3 j4 j5      (angles en radians ; défaut = 0)
        python show_arm.py --deg 30 -20 -40 0 15   (en degrés)
Rend robot_sim/pose.png . Autonome (parse URDF + lit les STL binaires, matplotlib)."""
import sys, os, struct, xml.etree.ElementTree as ET
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "robot.urdf")

def rpy_to_R(r, p, y):
    cx,sx=np.cos(r),np.sin(r); cy,sy=np.cos(p),np.sin(p); cz,sz=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx

def T(xyz, rpy):
    M=np.eye(4); M[:3,:3]=rpy_to_R(*rpy); M[:3,3]=xyz; return M

def axis_rot(axis, ang):
    a=np.array(axis,float); a/=np.linalg.norm(a); x,y,z=a; c,s=np.cos(ang),np.sin(ang); C=1-c
    R=np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
    M=np.eye(4); M[:3,:3]=R; return M

def load_stl(path):
    with open(path,"rb") as f:
        f.read(80); n=struct.unpack("<I",f.read(4))[0]
        buf=np.frombuffer(f.read(n*50), dtype=np.uint8).reshape(n,50)
    v=buf[:,12:48].copy().view("<f4").reshape(n,3,3)   # [n_tri,3 sommets,3 coords]
    return v

def parse_urdf(path):
    root=ET.parse(path).getroot()
    joints={}   # child_link -> (parent, type, axis, origin_T, name)
    for j in root.findall("joint"):
        parent=j.find("parent").get("link"); child=j.find("child").get("link")
        o=j.find("origin"); xyz=[float(x) for x in (o.get("xyz","0 0 0")).split()] if o is not None else [0,0,0]
        rpy=[float(x) for x in (o.get("rpy","0 0 0")).split()] if o is not None else [0,0,0]
        ax=j.find("axis"); axis=[float(x) for x in ax.get("xyz").split()] if ax is not None else [0,0,1]
        joints[child]=dict(parent=parent, type=j.get("type"), axis=axis, T=T(xyz,rpy), name=j.get("name"))
    visuals={}  # link -> list of (kind, data, origin_T, scale)
    for lk in root.findall("link"):
        vs=[]
        for v in lk.findall("visual"):
            o=v.find("origin"); xyz=[float(x) for x in (o.get("xyz","0 0 0")).split()] if o is not None else [0,0,0]
            rpy=[float(x) for x in (o.get("rpy","0 0 0")).split()] if o is not None else [0,0,0]
            g=v.find("geometry"); mesh=g.find("mesh"); box=g.find("box")
            if mesh is not None:
                fn=mesh.get("filename").split("/")[-1]; sc=[float(x) for x in mesh.get("scale","1 1 1").split()]
                vs.append(("mesh", os.path.join(HERE,"meshes",fn), T(xyz,rpy), sc))
            elif box is not None:
                vs.append(("box", [float(x) for x in box.get("size").split()], T(xyz,rpy), [1,1,1]))
        visuals[lk.get("link" if False else "name")]=vs
    return joints, visuals

def link_world(link, joints, q):
    """transform monde du repère d'un lien, en remontant jusqu'à world."""
    M=np.eye(4)
    chain=[]
    while link in joints:
        chain.append(link); link=joints[chain[-1]]["parent"]
    for lk in reversed(chain):
        j=joints[lk]; Mj=j["T"].copy()
        if j["type"]=="revolute":
            Mj=Mj@axis_rot(j["axis"], q.get(j["name"],0.0))
        M=M@Mj
    return M

def box_tris(size):
    sx,sy,sz=[s/2 for s in size]
    c=np.array([[ -sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],
                [-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])
    faces=[[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]]
    return np.array([[c[i] for i in f] for f in faces])

def main():
    args=sys.argv[1:]
    deg=False
    if args and args[0]=="--deg": deg=True; args=args[1:]
    vals=[float(x) for x in args[:5]] + [0.0]*(5-len(args))
    if deg: vals=[np.deg2rad(v) for v in vals]
    q={f"joint_{i+1}":vals[i] for i in range(5)}
    print("pose (rad):", {k:round(v,3) for k,v in q.items()})

    joints,visuals=parse_urdf(URDF)
    tris_all=[]
    for link, vs in visuals.items():
        Mw=link_world(link, joints, q)
        for kind,data,Ov,sc in vs:
            V = load_stl(data) if kind=="mesh" else box_tris(data)
            V = V*np.array(sc)                                  # scale mm->m
            flat=V.reshape(-1,3)
            world=(Mw@Ov@np.c_[flat, np.ones(len(flat))].T).T[:,:3]
            tris_all.append(world.reshape(-1,3,3))
    tris=np.concatenate(tris_all,0)
    print(f"{len(tris)} triangles")

    fig=plt.figure(figsize=(9,9)); ax=fig.add_subplot(111,projection="3d")
    pc=Poly3DCollection(tris, facecolor="#9aa7b4", edgecolor="none", alpha=1.0)
    # ombrage simple par normale.z
    n=np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]); n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
    shade=0.4+0.6*np.clip(n[:,2]*0.5+0.5,0,1)
    pc.set_facecolor(np.c_[np.outer(shade,[0.6,0.67,0.72])])
    ax.add_collection3d(pc)
    # sol + repère
    allp=tris.reshape(-1,3); c=allp.mean(0); r=np.ptp(allp,0).max()/2*1.1
    for lim,setter in [((c[0]-r,c[0]+r),ax.set_xlim),((c[1]-r,c[1]+r),ax.set_ylim),((0,2*r),ax.set_zlim)]:
        setter(*lim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (haut)")
    ax.view_init(elev=20, azim=-60); ax.set_box_aspect((1,1,1))
    ax.set_title(f"Bras — pose {[round(np.rad2deg(v)) for v in vals]}°", fontweight="bold")
    out=os.path.join(HERE,"pose.png"); fig.savefig(out,dpi=110,bbox_inches="tight"); print("->",out)

if __name__=="__main__":
    main()
