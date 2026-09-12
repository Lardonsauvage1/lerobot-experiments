#!/usr/bin/env python
"""Exporte la géométrie (meshes décimés, bakés dans le repère du lien) + la chaîne cinématique
en JS pour le visualiseur navigateur. Sortie : robot_sim/arm_data.js  (const ARM = {...})."""
import os, struct, json, xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "robot.urdf")
MAXTRI = 2200          # triangles max par lien (décimation pour un rendu fluide)

def rpy_to_R(r,p,y):
    cx,sx=np.cos(r),np.sin(r); cy,sy=np.cos(p),np.sin(p); cz,sz=np.cos(y),np.sin(y)
    return (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            @np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]))
def T(xyz,rpy):
    M=np.eye(4); M[:3,:3]=rpy_to_R(*rpy); M[:3,3]=xyz; return M

def load_stl(path):
    with open(path,"rb") as f:
        f.read(80); n=struct.unpack("<I",f.read(4))[0]
        buf=np.frombuffer(f.read(n*50),dtype=np.uint8).reshape(n,50)
    return buf[:,12:48].copy().view("<f4").reshape(n,3,3)

def box_tris(size):
    sx,sy,sz=[s/2 for s in size]
    c=np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])
    F=[[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]]
    return np.array([[c[i] for i in f] for f in F])

root=ET.parse(URDF).getroot()
# chaine
chain=[]
for j in root.findall("joint"):
    o=j.find("origin"); xyz=[float(x) for x in (o.get("xyz","0 0 0")).split()] if o is not None else [0,0,0]
    rpy=[float(x) for x in (o.get("rpy","0 0 0")).split()] if o is not None else [0,0,0]
    ax=j.find("axis"); axis=[float(x) for x in ax.get("xyz").split()] if ax is not None else [0,0,1]
    chain.append(dict(name=j.get("name"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                      type=j.get("type"), axis=axis, xyz=xyz, rpy=rpy))
# meshes bakés (repère du lien = scale + visual origin appliqués)
links={}
for lk in root.findall("link"):
    tris=[]
    for v in lk.findall("visual"):
        o=v.find("origin"); xyz=[float(x) for x in (o.get("xyz","0 0 0")).split()] if o is not None else [0,0,0]
        rpy=[float(x) for x in (o.get("rpy","0 0 0")).split()] if o is not None else [0,0,0]
        g=v.find("geometry"); mesh=g.find("mesh"); box=g.find("box"); cyl=g.find("cylinder")
        if mesh is not None:
            fn=mesh.get("filename").split("/")[-1]; sc=np.array([float(x) for x in mesh.get("scale","1 1 1").split()])
            V=load_stl(os.path.join(HERE,"meshes",fn))*sc
        elif box is not None:
            V=box_tris([float(x) for x in box.get("size").split()])
        elif cyl is not None:
            rad=float(cyl.get("radius")); h=float(cyl.get("length")); m=16
            ang=np.linspace(0,2*np.pi,m,endpoint=False); tt=[]
            for i in range(m):
                a,b=ang[i],ang[(i+1)%m]
                p1=[rad*np.cos(a),rad*np.sin(a),-h/2]; p2=[rad*np.cos(b),rad*np.sin(b),-h/2]
                p3=[rad*np.cos(a),rad*np.sin(a),h/2]; p4=[rad*np.cos(b),rad*np.sin(b),h/2]
                tt+= [[p1,p2,p3],[p2,p4,p3],[[0,0,-h/2],p2,p1],[[0,0,h/2],p3,p4]]
            V=np.array(tt)
        else:
            continue
        Mo=T(xyz,rpy)
        flat=V.reshape(-1,3); w=(Mo@np.c_[flat,np.ones(len(flat))].T).T[:,:3]
        tris.append(w.reshape(-1,3,3))
    if not tris: continue
    A=np.concatenate(tris,0)
    if len(A)>MAXTRI:                                   # décimation (sous-échantillonnage régulier)
        idx=np.linspace(0,len(A)-1,MAXTRI).astype(int); A=A[idx]
    links[lk.get("name")]=np.round(A,4).reshape(len(A),9).tolist()   # [tri][9] = 3 sommets

data=dict(chain=chain, links=links)
out=os.path.join(HERE,"arm_data.js")
with open(out,"w") as f:
    f.write("const ARM = "+json.dumps(data)+";")
ntri=sum(len(v) for v in links.values())
print(f"-> {out}  ({len(links)} liens, {ntri} triangles, {os.path.getsize(out)//1024} Ko)")
