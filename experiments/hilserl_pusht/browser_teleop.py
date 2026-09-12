#!/usr/bin/env python
"""Téléop Push-T dans le NAVIGATEUR (pont web local) — remplace pygame/pynput.
- Toggle CONTRÔLE (espace / bouton) : ON = tu pilotes, OFF = la policy agit.
- Pendant le contrôle, les pauses (aucune flèche) ne sont PAS enregistrées (le wrapper attend sans stepper).
Interface : connect/set_frame/get_teleop_events/get_action/disconnect.
Convention flèches : dx=+droite/-gauche, dy=+bas/-haut (repère pixel Push-T).
"""
import threading
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from lerobot.teleoperators.utils import TeleopEvents

HTML = """<!doctype html><html><head><meta charset=utf-8><title>Push-T HIL-SERL</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;text-align:center;margin:0;padding:12px}
img{image-rendering:pixelated;width:480px;height:480px;border:1px solid #333;background:#000}
#ctl{font-size:18px;font-weight:bold;padding:6px 14px;border-radius:6px;display:inline-block;margin:8px}
.on{background:#16a34a;color:#fff}.off{background:#374151;color:#bbb}
button{font-size:16px;padding:8px 16px;margin:6px;border-radius:6px;border:1px solid #555;background:#2563eb;color:#fff;cursor:pointer}
.k{display:inline-block;padding:2px 8px;margin:2px;border:1px solid #555;border-radius:4px}.kon{background:#2563eb;color:#fff}
p{color:#888;font-size:12px}</style></head><body>
<h3>Push-T — HIL-SERL</h3>
<img id=v src="/frame"><br>
<span id=ctl class=off>CONTRÔLE : OFF (la policy agit)</span> <span id=fl></span>
<button onclick="toggle()">Activer / désactiver (Espace)</button>
<div><span class=k id=kl>←</span><span class=k id=ku>↑</span><span class=k id=kd>↓</span><span class=k id=kr>→</span>
&nbsp; <span class=k>s = succès</span><span class=k>q = fin épisode</span></div>
<p>Espace = prendre/rendre le contrôle. En contrôle ON, les flèches déplacent le poussoir ;
si tu ne presses rien, la sim ATTEND (pause non enregistrée) — prends ton temps.</p>
<script>
let control=false,dx=0,dy=0,S=false,Q=false,down={};
function send(){fetch('/keys?control='+(control?1:0)+'&dx='+dx+'&dy='+dy+'&s='+(S?1:0)+'&q='+(Q?1:0));}
function flash(t){let f=document.getElementById('fl');f.textContent=t;f.style.color='#fde047';setTimeout(()=>f.textContent='',700);}
function ui(){
 for(const[i,k]of[['kl','ArrowLeft'],['kr','ArrowRight'],['ku','ArrowUp'],['kd','ArrowDown']])
   document.getElementById(i).className='k'+(down[k]?' kon':'');
 let c=document.getElementById('ctl');
 c.textContent='CONTRÔLE : '+(control?'ON (tu pilotes)':'OFF (la policy agit)');c.className=control?'on':'off';
}
function upd(){dx=(down.ArrowRight?1:0)-(down.ArrowLeft?1:0);dy=(down.ArrowDown?1:0)-(down.ArrowUp?1:0);ui();send();}
function toggle(){control=!control;ui();send();}
addEventListener('keydown',e=>{
 if(e.key===' '){toggle();e.preventDefault();return;}
 if(e.key.startsWith('Arrow')){down[e.key]=1;e.preventDefault();upd();}
 else if(e.key==='s'){S=true;send();flash('✓ succès envoyé');setTimeout(()=>{S=false;},300);}
 else if(e.key==='q'){Q=true;send();flash('✗ fin épisode');setTimeout(()=>{Q=false;},300);}});
addEventListener('keyup',e=>{if(e.key.startsWith('Arrow')){down[e.key]=0;e.preventDefault();upd();}});
setInterval(()=>{document.getElementById('v').src='/frame?t='+Date.now();},80);  // ~12 fps
setInterval(send,200);  // heartbeat (garde l'état contrôle côté serveur)
ui();
</script></body></html>"""


class BrowserTeleop:
    def __init__(self, port=8000):
        self.port = port
        self._lock = threading.Lock()
        self._png = cv2.imencode(".png", np.zeros((96, 96, 3), np.uint8))[1].tobytes()
        self._st = {"control": False, "dx": 0.0, "dy": 0.0, "s": False, "q": False}
        self._dbg = {"pos": [0.0, 0.0], "act": [0.0, 0.0], "interv": False}
        self._srv = None

    def set_debug(self, pos, act, interv):
        with self._lock:
            self._dbg = {"pos": [float(pos[0]), float(pos[1])], "act": [float(act[0]), float(act[1])], "interv": bool(interv)}

    def connect(self):
        tele = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    body = HTML.encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif u.path == "/frame":
                    with tele._lock:
                        png = tele._png
                    self.send_response(200); self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(png)))
                    self.end_headers(); self.wfile.write(png)
                elif u.path == "/keys":
                    q = parse_qs(u.query)
                    with tele._lock:
                        tele._st["control"] = q.get("control", ["0"])[0] == "1"
                        tele._st["dx"] = float(q.get("dx", ["0"])[0])
                        tele._st["dy"] = float(q.get("dy", ["0"])[0])
                        if q.get("s", ["0"])[0] == "1":
                            tele._st["s"] = True
                        if q.get("q", ["0"])[0] == "1":
                            tele._st["q"] = True
                    self.send_response(200); self.end_headers()
                elif u.path == "/state":
                    import json as _j
                    with tele._lock:
                        body = _j.dumps(tele._dbg).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()

        class _Srv(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True
        self._srv = _Srv(("127.0.0.1", self.port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"[browser-teleop] >>> OUVRE http://127.0.0.1:{self.port} dans ton navigateur <<<", flush=True)

    def set_frame(self, rgb):
        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        png = cv2.imencode(".png", bgr)[1].tobytes()
        with self._lock:
            self._png = png

    def read(self):
        """(control, dx, dy, success, terminate) ; consomme success/terminate (momentanés)."""
        with self._lock:
            s = dict(self._st)
            self._st["s"] = False; self._st["q"] = False
        return bool(s["control"]), s["dx"], s["dy"], bool(s["s"]), bool(s["q"])

    def disconnect(self):
        if self._srv:
            self._srv.shutdown()
