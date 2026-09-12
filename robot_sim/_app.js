// ============ moteur 3D (Canvas 2D, painter's algorithm) + FK 5 axes ============
"use strict";
const LIMITS=[[-180,180],[-92,120],[-172,37],[-180,180],[-92,92]]; // butées joint_1..5 (deg)
const JOINTS=["joint_1","joint_2","joint_3","joint_4","joint_5"];
const DEG=Math.PI/180, RAD=180/Math.PI;

// ---- matrices 4x4 (col-major flat[16]) ----
const I4=()=>[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
function mul(a,b){const c=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
  for(let k=0;k<4;k++)s+=a[k*4+i]*b[j*4+k];c[j*4+i]=s;}return c;}
function rpyR(r,p,y){const cx=Math.cos(r),sx=Math.sin(r),cy=Math.cos(p),sy=Math.sin(p),cz=Math.cos(y),sz=Math.sin(y);
  // Rz*Ry*Rx
  return [cz*cy, sz*cy, -sy, 0,
          cz*sy*sx-sz*cx, sz*sy*sx+cz*cx, cy*sx, 0,
          cz*sy*cx+sz*sx, sz*sy*cx-cz*sx, cy*cx, 0,
          0,0,0,1];}
function T(xyz,rpy){const m=rpyR(rpy[0],rpy[1],rpy[2]);m[12]=xyz[0];m[13]=xyz[1];m[14]=xyz[2];return m;}
function axisRot(ax,a){let[x,y,z]=ax;const n=Math.hypot(x,y,z)||1;x/=n;y/=n;z/=n;
  const c=Math.cos(a),s=Math.sin(a),C=1-c;
  return [c+x*x*C, y*x*C+z*s, z*x*C-y*s, 0,
          x*y*C-z*s, c+y*y*C, z*y*C+x*s, 0,
          x*z*C+y*s, y*z*C-x*s, c+z*z*C, 0, 0,0,0,1];}
function apply(m,p){return [m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12],
                            m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13],
                            m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14]];}

// ---- FK : repère monde de chaque lien ----
function linkWorld(anglesRad){
  const W={world:I4()};
  for(const j of ARM.chain){
    const ji=JOINTS.indexOf(j.name);
    const rot = (j.type==="revolute")? axisRot(j.axis, ji>=0?anglesRad[ji]:0) : I4();
    W[j.child]=mul(W[j.parent]||I4(), mul(T(j.xyz,j.rpy), rot));
  }
  return W;
}

// ---- caméra orbitale ----
let az=-1.05, el=0.42, dist=1.45, target=[0,0,0.42];
function lookAt(eye,ctr,up){
  let f=[ctr[0]-eye[0],ctr[1]-eye[1],ctr[2]-eye[2]];let fn=Math.hypot(...f);f=f.map(v=>v/fn);
  let s=[f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]];let sn=Math.hypot(...s);s=s.map(v=>v/sn);
  let u=[s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]];
  return [s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
          -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]),
          -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]),
           (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]),1];}
function eyePos(){return [target[0]+dist*Math.cos(el)*Math.cos(az),
                         target[1]+dist*Math.cos(el)*Math.sin(az),
                         target[2]+dist*Math.sin(el)];}

// ---- rendu ----
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
let W=0,H=0,DPR=Math.min(devicePixelRatio||1,2);
function resize(){const r=cv.getBoundingClientRect();W=r.width;H=r.height;
  cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);render();}
const MESH=[0.60,0.68,0.78], GRIP=[0.86,0.64,0.30], L=[0.4,0.5,0.75]; // couleurs + lumière
(function(){const n=Math.hypot(...L);for(let i=0;i<3;i++)L[i]/=n;})();
let curAngles=[0,0,0,0,0];

function render(){
  const V=lookAt(eyePos(),target,[0,0,1]);
  const foc=H*1.15;                                   // focale px
  const Wl=linkWorld(curAngles);
  const tri=[];
  for(const link in ARM.links){
    const M=Wl[link]; if(!M) continue;
    const col=(link==="link_gripper")?GRIP:MESH;
    for(const t of ARM.links[link]){
      const a=apply(M,[t[0],t[1],t[2]]), b=apply(M,[t[3],t[4],t[5]]), c=apply(M,[t[6],t[7],t[8]]);
      // normale monde -> ombrage
      const ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2], vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
      let nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx; const nn=Math.hypot(nx,ny,nz)||1;
      let d=(nx*L[0]+ny*L[1]+nz*L[2])/nn; d=Math.abs(d); const sh=0.30+0.70*d;
      // -> caméra
      const A=apply(V,a),B=apply(V,b),Cc=apply(V,c);
      if(A[2]>-0.02&&B[2]>-0.02&&Cc[2]>-0.02) continue; // derrière la caméra
      const px=(p)=>[W/2+foc*p[0]/(-p[2]), H/2-foc*p[1]/(-p[2])];
      tri.push({z:(A[2]+B[2]+Cc[2])/3, p:[px(A),px(B),px(Cc)],
                c:`rgb(${col[0]*sh*255|0},${col[1]*sh*255|0},${col[2]*sh*255|0})`});
    }
  }
  tri.sort((x,y)=>x.z-y.z);                            // loin -> près
  ctx.clearRect(0,0,W,H);
  // sol
  const gy=px3(apply(V,[0,0,0]),foc); if(gy) drawGround(V,foc);
  for(const t of tri){const p=t.p;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);
    ctx.lineTo(p[1][0],p[1][1]);ctx.lineTo(p[2][0],p[2][1]);ctx.closePath();
    ctx.fillStyle=t.c;ctx.fill();}
  // TCP
  const tcpW=Wl["tcp"]; if(tcpW){const p=apply(V,[tcpW[12],tcpW[13],tcpW[14]]);
    if(p[2]<-0.02){const s=[W/2+foc*p[0]/(-p[2]),H/2-foc*p[1]/(-p[2])];
    ctx.beginPath();ctx.arc(s[0],s[1],4,0,7);ctx.fillStyle="#f2a33c";ctx.fill();}}
}
function px3(p,foc){if(p[2]>=-0.02)return null;return [W/2+foc*p[0]/(-p[2]),H/2-foc*p[1]/(-p[2])];}
function drawGround(V,foc){ctx.strokeStyle="rgba(90,110,130,.18)";ctx.lineWidth=1;
  const R=0.6,st=0.1;for(let g=-R;g<=R+1e-6;g+=st){
    for(const seg of [[[g,-R,0],[g,R,0]],[[-R,g,0],[R,g,0]]]){
      const a=px3(apply(V,seg[0]),foc),b=px3(apply(V,seg[1]),foc);
      if(a&&b){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}}}}

// ---- FK effecteur pour l'affichage TCP ----
function updateTCP(){const W=linkWorld(curAngles),m=W["tcp"];
  document.getElementById("tcp").innerHTML= m?
    `position : <b>x</b> ${m[12].toFixed(3)}  <b>y</b> ${m[13].toFixed(3)}  <b>z</b> ${m[14].toFixed(3)} m`:"–";}

// ============ UI ============
const jointsDiv=document.getElementById("joints"), sliders=[], vals=[];
for(let i=0;i<5;i++){
  const row=document.createElement("div");row.className="jrow";
  row.innerHTML=`<label>j${i+1}</label>
    <input type="range" min="${LIMITS[i][0]}" max="${LIMITS[i][1]}" step="0.5" value="0">
    <span class="val">0°</span>`;
  jointsDiv.appendChild(row);
  const sl=row.querySelector("input"),vl=row.querySelector(".val");sliders.push(sl);vals.push(vl);
  sl.addEventListener("input",()=>{setPoseDeg(sliders.map(s=>+s.value));document.getElementById("hud_frame").textContent="manuel";});
}
function setPoseDeg(deg){curAngles=deg.map(d=>d*DEG);
  for(let i=0;i<5;i++){sliders[i].value=deg[i];vals[i].textContent=deg[i].toFixed(0)+"°";}
  updateTCP();render();}

// unité deg/rad
let unit="deg";
document.getElementById("unit").addEventListener("click",e=>{if(!e.target.dataset.u)return;
  unit=e.target.dataset.u;[...e.currentTarget.children].forEach(b=>b.classList.toggle("on",b.dataset.u===unit));});

// trajectoire
let traj=[];                                          // [[j1..j5 rad, grip], ...]
function parseTraj(){const lines=document.getElementById("traj").value.trim().split("\n");
  const out=[];for(const ln of lines){const n=ln.trim().split(/[\s,]+/).map(Number).filter(x=>!isNaN(x));
    if(n.length>=5){const a=n.slice(0,5).map(v=>unit==="deg"?v*DEG:v);a.push(n[5]??0);out.push(a);}}
  return out;}
function loadTrajInto(arr){traj=arr;const fr=document.getElementById("frame");
  fr.max=Math.max(0,traj.length-1);fr.value=0;seek(0);
  document.getElementById("framev").textContent=traj.length?`1 / ${traj.length}`:"– / –";}
function fillTA(arr){const ta=document.getElementById("traj");
  ta.value=arr.map(p=>p.slice(0,5).map(v=>unit==="deg"?(v*RAD).toFixed(1):v.toFixed(3)).join(" ")).join("\n");}
document.getElementById("loadDemo").addEventListener("click",()=>{
  stop();fillTA(DEMO);loadTrajInto(DEMO.map(p=>p.slice()));});
function loadPred1s(poses,label){                                            // 16 pas prédits, 1/s
  stop();fillTA(poses);loadTrajInto(poses.map(p=>p.slice()));
  const sp=document.getElementById("speed");sp.value=1;document.getElementById("speedv").textContent="1/s";
  document.getElementById("hud_frame").textContent="prédiction "+label+" (16 pas @ 1/s)";}
document.getElementById("loadPred").addEventListener("click",()=>loadPred1s(PRED,"ep0 f0"));
// 5 endroits au hasard : boutons prédit + réel
const pb=document.getElementById("predBtns"), rb=document.getElementById("realBtns");
PREDS.forEach(pr=>{
  const b=document.createElement("button");b.textContent=pr.label;b.style.padding="5px 9px";
  b.addEventListener("click",()=>loadPred1s(pr.poses,pr.label+" (prédit)"));pb.appendChild(b);
  if(REAL[pr.label]){const r=document.createElement("button");r.textContent=pr.label;r.style.padding="5px 9px";
    r.addEventListener("click",()=>loadPred1s(REAL[pr.label],pr.label+" (réel)"));rb.appendChild(r);}
});

function seek(i){i=Math.max(0,Math.min(traj.length-1,i|0));if(!traj.length)return;
  const p=traj[i];setPoseDeg(p.slice(0,5).map(v=>v*RAD));
  document.getElementById("frame").value=i;
  document.getElementById("framev").textContent=`${i+1} / ${traj.length}`;
  document.getElementById("hud_frame").textContent=`trajectoire ${i+1}/${traj.length}`;}
document.getElementById("frame").addEventListener("input",e=>{stop();seek(+e.target.value);});

// lecture
let playing=false,rafid=0,acc=0,last=0,idx=0;
const playBtn=document.getElementById("play");
function stop(){playing=false;playBtn.textContent="▶ Lire";cancelAnimationFrame(rafid);}
function play(){if(!traj.length){loadTrajInto(parseTraj());if(!traj.length)return;}
  playing=true;playBtn.textContent="❚❚ Pause";idx=+document.getElementById("frame").value;last=performance.now();acc=0;loop();}
function loop(){if(!playing)return;const now=performance.now(),dt=(now-last)/1000;last=now;
  const sp=+document.getElementById("speed").value;acc+=dt*sp;
  while(acc>=1){acc-=1;idx++;if(idx>=traj.length){if(document.getElementById("loop").checked)idx=0;else{seek(traj.length-1);stop();return;}}}
  seek(idx);rafid=requestAnimationFrame(loop);}
playBtn.addEventListener("click",()=>{if(playing)stop();else{ if(document.getElementById("traj").value.trim()&&!traj.length)loadTrajInto(parseTraj()); play();}});
document.getElementById("speed").addEventListener("input",e=>document.getElementById("speedv").textContent=e.target.value+"/s");
document.getElementById("traj").addEventListener("change",()=>loadTrajInto(parseTraj()));

// boutons vue
document.getElementById("resetView").addEventListener("click",()=>{az=-1.05;el=0.42;dist=1.45;render();});
document.getElementById("resetPose").addEventListener("click",()=>{stop();setPoseDeg([0,0,0,0,0]);document.getElementById("hud_frame").textContent="manuel";});

// orbite souris
let drag=false,lx=0,ly=0;
cv.addEventListener("pointerdown",e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);});
cv.addEventListener("pointermove",e=>{if(!drag)return;az-=(e.clientX-lx)*0.008;el+=(e.clientY-ly)*0.008;
  el=Math.max(-1.4,Math.min(1.4,el));lx=e.clientX;ly=e.clientY;render();});
cv.addEventListener("pointerup",()=>drag=false);
cv.addEventListener("wheel",e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0011);
  dist=Math.max(0.6,Math.min(4,dist));render();},{passive:false});

window.addEventListener("resize",resize);
setPoseDeg([0,0,0,0,0]);resize();
