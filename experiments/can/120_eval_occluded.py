#!/usr/bin/env python
"""Éval d'un checkpoint sur le banc Can-Occluded — avec ou sans mémoire CAMP EN LIGNE.

C'est ici que CAMP existe vraiment : à l'entraînement les codes m_t sont précalculés, mais
au déploiement il faut faire tourner le LSTM **pas à pas**, comme le fera le vrai robot.
Ce script est donc aussi la répétition générale du déploiement.

  sans mémoire :  --ckpt <ck> --radius 0.03
  avec mémoire :  --ckpt <ck> --radius 0.03 --camp results/runs/can/camp/memory_L64_K32_m32.pt

⚠️ Points de cohérence avec l'entraînement, chacun a son piège :
  - `prev_action` = action RÉELLEMENT exécutée au pas précédent, ZÉRO au premier pas
    (au pas 0 l'action n'est pas encore décidée — d'où le padding par zéros côté train) ;
  - l'état caché est REMIS À ZÉRO à chaque épisode, comme `policy.reset()` ;
  - `n_obs_steps` codes empilés, clampés au début de l'épisode — même convention que
    `precompute_memory`, sinon la policy reçoit un vecteur qu'elle n'a jamais vu.
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import can_eval, can_occlusion, camp

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vis500)

OUT = Path("results/runs/can/occluded")


class OnlineMemory:
    """Fait avancer le module mémoire pas à pas et fabrique l'état augmenté."""

    def __init__(self, ckpt_path, action_dim, state_dim, n_obs_steps, device):
        ck = torch.load(ckpt_path, weights_only=False)
        a = ck["args"]
        self.model = camp.CampMemory(action_dim, state_dim, n_coef=a["K"],
                                     mem_dim=a["mem_dim"], codebook_size=a["codebook"]).to(device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.device, self.n_obs, self.mem_dim = device, n_obs_steps, a["mem_dim"]
        self.action_dim = action_dim

    def reset(self):
        self.hx = None
        self.prev = torch.zeros(1, 1, self.action_dim, device=self.device)  # zéro au 1er pas
        self.hist = []

    @torch.no_grad()
    def step(self, proprio):
        p = torch.tensor(proprio, device=self.device)[None, None]
        _, m, _, _, self.hx = self.model(p, self.prev, hx=self.hx)
        self.hist.append(m[0, 0])
        # n_obs_steps derniers codes, clampés au début de l'épisode (comme au précalcul)
        stack = [self.hist[max(0, len(self.hist) - 1 - k)] for k in range(self.n_obs - 1, -1, -1)]
        return torch.cat([torch.tensor(proprio, device=self.device), stack[-1]])

    def set_prev_action(self, a):
        self.prev = torch.tensor(a, device=self.device)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--camp", default=None, help="module mémoire (active CAMP en ligne)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--grasp-guard", dest="grasp_guard", type=float, default=0.0,
                    help="TEST CAUSAL : bloque la fermeture de pince tant que l'effecteur est "
                         "à plus de N cm au-dessus de la canette (0 = désactivé). Si le succès "
                         "monte, fermer trop haut CAUSE l'échec ; sinon c'est un symptôme.")
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()[:a.n]

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    # ⚠️ un checkpoint entraîné sur atomman embarque device='xpu' dans son préprocesseur,
    # inexistant ici -> le pipeline refuse de s'instancier (la policy, elle, bascule seule).
    # On réécrit le device dans les JSON du checkpoint (idempotent).
    import json as _json
    for _f in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        _p = Path(a.ckpt) / _f
        if _p.exists():
            _c = _json.loads(_p.read_text())
            _txt = _json.dumps(_c)
            if '"xpu"' in _txt or '"cuda"' in _txt:
                _p.write_text(_txt.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))
                print(f"  device réécrit dans {_f} -> {device.type}", flush=True)

    # ⚠️ KPAMP : le checkpoint a un spatial softmax modifié (3 valeurs/keypoint au lieu de 2)
    # et un Linear d'entrée 3*kp. `from_pretrained` reconstruirait le modèle STANDARD et le
    # chargement des poids échouerait. On détecte le cas dans le safetensors et on applique
    # le même patch qu'à l'entraînement AVANT de charger.
    import torch.nn as _nn
    from safetensors.torch import load_file as _load
    _w = _load(str(Path(a.ckpt) / "model.safetensors"))
    _k = "diffusion.rgb_encoder.out.weight"
    _cfgj = _json.loads((Path(a.ckpt) / "config.json").read_text())
    _nkp = int(_cfgj.get("spatial_softmax_num_keypoints", 32))
    _is_kpamp = _k in _w and _w[_k].shape[1] == _nkp * 3
    if _is_kpamp:
        print(f"  checkpoint KPAMP détecté ({_nkp} keypoints x 3)", flush=True)

    policy = DiffusionPolicy.from_pretrained(a.ckpt) if not _is_kpamp else None
    if _is_kpamp:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as _DP
        from lerobot.configs.policies import PreTrainedConfig as _PC
        _cfg = _PC.from_pretrained(a.ckpt)
        policy = _DP(_cfg)
        _enc = policy.diffusion.rgb_encoder
        _base = _enc.pool

        class _SSAmp(_nn.Module):
            def __init__(self, base):
                super().__init__(); self.base = base
            def forward(self, f):
                if self.base.nets is not None:
                    f = self.base.nets(f)
                B, K, H, W = f.shape
                flat = f.reshape(B * K, H * W)
                att = _nn.functional.softmax(flat, dim=-1)
                xy = att @ self.base.pos_grid
                amp = flat.max(dim=-1, keepdim=True).values
                return torch.cat([xy, amp], dim=-1).reshape(B, K, 3)

        _enc.pool = _SSAmp(_base)
        _enc.out = _nn.Linear(_nkp * 3, _enc.out.out_features)
        _missing = policy.load_state_dict(_w, strict=False)
        assert not _missing.unexpected_keys, f"poids inattendus : {_missing.unexpected_keys[:3]}"
        print(f"  poids KPAMP chargés (manquants : {len(_missing.missing_keys)})", flush=True)
    policy = policy.to(device).eval()
    policy.diffusion.num_inference_steps = a.steps
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    # ORACLE 12D : le checkpoint attend proprio(9) + can_pos(3). On complète depuis l'env —
    # c'est la BÉQUILLE assumée de ce bras : la position vraie, même quand la canette est
    # masquée à l'image. Détection automatique sur la dimension déclarée.
    _sdim = int(policy.config.input_features["observation.state"].shape[0])
    _oracle = _sdim == 12
    if _oracle:
        print("  ORACLE 12D : can_pos ajoutée à l'état (position vraie, même occluse)", flush=True)

    mem = None
    if a.camp:
        n_obs = int(policy.config.n_obs_steps)
        mem = OnlineMemory(a.camp, 7, 9, n_obs, device)
        exp = int(policy.config.input_features["observation.state"].shape[0])
        got = 9 + mem.mem_dim
        assert exp == got, f"state attendu par la policy {exp}D != {got}D fourni par CAMP"
        print(f"CAMP en ligne : state 9D + {mem.mem_dim}D = {got}D | n_obs_steps={n_obs}", flush=True)

    render_fn = can_occlusion.make_render_fn(a.radius)
    probe = can_occlusion.OcclusionProbe(a.radius)
    rec_traj = can_occlusion.TrajectoryRecorder()
    per, t0 = [], time.time()
    for i in range(0, len(states), 50):
        env = can_eval.make_env()
        for ep in range(i, min(i + 50, len(states))):
            obs = env.reset_to(dict(states=states[ep]))
            policy.reset(); probe.start_episode(); rec_traj.start_episode()
            if mem: mem.reset()
            t_succ = None
            max_can_z = 0.0
            for step_i in range(can_eval.MAX_STEPS):
                probe.observe(env, obs)
                img = render_fn(env, obs, 96)
                pro = vis500.state_proprio(obs)
                if _oracle:
                    pro = np.concatenate([pro, np.asarray(obs["object"]).flatten()[7:10]]).astype(np.float32)
                st = mem.step(pro) if mem else torch.tensor(pro, device=device)
                obs_d = pre({"observation.image": torch.from_numpy(img).permute(2, 0, 1)
                             .float().unsqueeze(0).to(device) / 255.0,
                             "observation.state": st.unsqueeze(0).to(device)})
                with torch.no_grad():
                    act = policy.select_action(obs_d)
                act = np.clip(post(act).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
                if a.grasp_guard > 0:
                    # on n'aide PAS à viser : on interdit seulement de refermer trop haut.
                    _cp = np.asarray(obs["object"]).flatten()[7:10]
                    _dz = float(np.asarray(obs["robot0_eef_pos"]).flatten()[2] - _cp[2])
                    if act[6] > 0 and _dz > a.grasp_guard:
                        act[6] = -1.0
                if mem: mem.set_prev_action(act)
                cp = np.asarray(obs["object"]).flatten()[7:10]
                max_can_z = max(max_can_z, float(cp[2]))
                rec_traj.step(pro[:9], act, cp)
                obs = env.step(act)[0]
                if env.is_success()["task"]:
                    t_succ = step_i; break
            r = {"success": t_succ is not None, "t_success": t_succ}
            # ⭐ indicateur PRÉCOCE : la canette a-t-elle décollé ? Bien plus sensible que le
            # succès binaire — un modèle en cours d'apprentissage soulève avant de réussir.
            r["can_lifted"] = bool(max_can_z > can_occlusion.CAN_Z_REST + can_occlusion.LIFT_MARGIN)
            r["max_can_z"] = float(max_can_z)
            r.update(probe.summary(r["success"]))
            rec_traj.end_episode(probe, {"ep": ep, "success": r["success"], "t_success": t_succ,
                                         "n_approaches": r["n_approaches"],
                                         "n_occl_events": r["n_occl_events"]})
            per.append(r)
        try: env.env.close()
        except Exception: pass
        del env
        k = sum(e["success"] for e in per)
        print(f"  [{i}-{i+50}] cumul {k}/{len(per)} = {k/len(per):.1%}", flush=True)

    k, n = sum(e["success"] for e in per), len(per)
    lo, hi = vis500.wilson_ci(k, n)
    out = {"tag": a.tag, "ckpt": a.ckpt, "radius": a.radius, "camp": a.camp,
           "n": n, "n_success": k, "success_rate": k / n, "ci95": [lo, hi],
           "z_cycles_failed": float(np.mean([e["z_cycles"] for e in per if not e["success"]]))
                              if any(not e["success"] for e in per) else None,
           "can_lifted_frac": float(np.mean([e["can_lifted"] for e in per])),
           "max_can_z_mean": float(np.mean([e["max_can_z"] for e in per])),
           "elapsed_min": (time.time() - t0) / 60}
    out.update(can_occlusion.aggregate_occlusion(per))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{a.tag}.json").write_text(json.dumps(out, indent=2))
    rec_traj.save(OUT / f"traj_{a.tag}.npz", radius=a.radius, ckpt=a.ckpt)
    print(f"\n=> {a.tag} : {k}/{n} = {k/n:.1%} [{lo:.1%}, {hi:.1%}] "
          f"| CANETTE SOULEVÉE {out['can_lifted_frac']:.1%} "
          f"| cycles/échec {out['z_cycles_failed']} | {out['elapsed_min']:.0f} min", flush=True)


if __name__ == "__main__":
    main()
