#!/usr/bin/env python
"""Wrapper gym_pusht (image) au format HIL-SERL de LeRobot + intervention CLAVIER cartésienne.

- obs : {"pixels": (H,W,3) uint8, "agent_pos": (2,)}  (== ce que VanillaObservationProcessorStep attend)
- action : Box(0,512,(2,)) = position cible absolue du poussoir
- reward : coverage intégré de Push-T (pas de classifieur)
- intervention : KeyboardEndEffectorTeleop (flèches -> delta_x/y) ; on pousse l'action exécutée dans info.

Le pipeline HIL-SERL est câblé pour gym_hil : on remplace `make_robot_env` par `make_pusht_robot_env`
(voir run_actor_pusht.py) et on met `env.name="gym_hil"` dans la config pour prendre la branche sim.
"""
import os
import time
import gymnasium as gym
import gym_pusht  # noqa: F401  (enregistre gym_pusht/PushT-v0)
import numpy as np

from lerobot.teleoperators.keyboard.teleop_keyboard import (
    KeyboardEndEffectorTeleop,
    KeyboardEndEffectorTeleopConfig,
)
from lerobot.teleoperators.utils import TeleopEvents

try:
    from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
except Exception:
    TELEOP_ACTION_KEY = "teleop_action"

IMG_SIZE = 96          # doit égaler input_features image de la config SAC
MAX_STEPS = 500
STEP_PX = float(os.environ.get("PUSHT_STEP", "45"))   # pas du poussoir par tick (px), réglable
KX, KY = -1.0, +1.0    # signes clavier->déplacement (repère pixel y-bas) ; flip si inversé


def _unnorm(a):
    """action policy [-1,1] -> cible env Push-T [0,512] (l'action_processor ne le fait pas dans ce montage)."""
    return np.clip((np.asarray(a, np.float32).reshape(-1)[:2] + 1.0) * 256.0, 0.0, 512.0)


def _norm(x):
    """cible env [0,512] -> action normalisée [-1,1] (pour enregistrer, cohérent avec le buffer/policy)."""
    return np.clip(np.asarray(x, np.float32).reshape(-1)[:2] / 256.0 - 1.0, -1.0, 1.0)


class PushTHILEnv(gym.Env):
    def __init__(self, task="PushT-v0", fps=10, img_size=IMG_SIZE, max_steps=MAX_STEPS, use_keyboard=True):
        import os
        render = os.environ.get("PUSHT_RENDER", "rgb_array")   # "human" = fenêtre visible pour intervenir
        self._env = gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render,
            observation_width=img_size, observation_height=img_size, max_episode_steps=max_steps,
        )
        self.observation_space = self._env.observation_space
        self.action_space = gym.spaces.Box(low=0.0, high=512.0, shape=(2,), dtype=np.float32)
        self.metadata = getattr(self._env, "metadata", {})
        self._prev_pos = np.zeros(2, dtype=np.float32)
        # --- HIL-SERL phase 1 : classifieur de récompense (remplace le coverage triché de l'env) ---
        self._clf = None; self._clf_hits = 0
        clf_path = os.environ.get("REWARD_CLF", "")
        if clf_path and os.path.exists(clf_path):
            from reward_classifier import RewardClassifier
            self._clf = RewardClassifier(clf_path, device="cpu")
            print(f"[reward-clf] chargé {clf_path} (seuil {self._clf.thresh})", flush=True)
        self._tele = None; self._browser = False; self._kx, self._ky = KX, KY
        if os.environ.get("PUSHT_BROWSER") == "1":
            from browser_teleop import BrowserTeleop
            self._tele = BrowserTeleop(port=int(os.environ.get("PUSHT_PORT", "8000")))
            self._tele.connect(); self._browser = True; self._kx, self._ky = 1.0, 1.0  # convention navigateur
        elif use_keyboard:
            self._tele = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig(use_gripper=False))
            self._tele.connect()   # pynput : nécessite Accessibilité + Input Monitoring sur macOS

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        self._clf_hits = 0
        info = dict(info); info["is_intervention"] = False
        return obs, info

    def step(self, policy_action):
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)[:2]
        is_intervention = False
        success = terminate = rerecord = False
        executed = _unnorm(policy_action)
        if self._browser:
            # toggle contrôle ; pause (contrôle ON sans flèche) = on attend SANS stepper (skip idle),
            # MAIS bornée à ~1 s pour ne pas casser le flux gRPC (keepalive = un pas policy).
            t0 = time.time()
            while True:
                self._tele.set_frame(self._env.render())
                control, dx, dy, success, terminate = self._tele.read()
                if success or terminate:
                    is_intervention = control
                    executed = (np.clip(self._prev_pos + STEP_PX * np.array([dx, dy], np.float32), 0.0, 512.0)
                                if control and (dx or dy) else self._prev_pos.copy())
                    break
                if not control:                       # policy agit
                    is_intervention = False; executed = _unnorm(policy_action); break
                if dx or dy:                          # tu bouges -> on step et on enregistre
                    is_intervention = True
                    executed = np.clip(self._prev_pos + STEP_PX * np.array([dx, dy], np.float32), 0.0, 512.0)
                    break
                if time.time() - t0 > 1.0:            # keepalive : pause longue -> on GÈLE le poussoir (reste sur place), garde le gRPC vivant sans bouger
                    is_intervention = False; executed = self._prev_pos.copy(); break
                time.sleep(0.05)                      # pause courte : attend, n'enregistre rien
        elif self._tele is not None:
            ev = self._tele.get_teleop_events()
            is_intervention = bool(ev.get(TeleopEvents.IS_INTERVENTION, False))
            success = bool(ev.get(TeleopEvents.SUCCESS, False))
            terminate = bool(ev.get(TeleopEvents.TERMINATE_EPISODE, False))
            rerecord = bool(ev.get(TeleopEvents.RERECORD_EPISODE, False))
            if is_intervention:
                a = self._tele.get_action()
                dx, dy = float(a.get("delta_x", 0.0)), float(a.get("delta_y", 0.0))
                executed = np.clip(self._prev_pos + STEP_PX * np.array([self._kx * dx, self._ky * dy], np.float32), 0.0, 512.0)

        obs, reward, terminated, truncated, info = self._env.step(executed)
        self._prev_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        if self._browser:
            self._tele.set_debug(self._prev_pos, executed, is_intervention)

        info = dict(info)
        info[TELEOP_ACTION_KEY] = _norm(executed)   # action enregistrée en [-1,1] (cohérent buffer/policy)
        info["is_intervention"] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord
        # SUCCÈS = classifieur de récompense appris (HIL-SERL phase 1), anti-rebond 2 frames.
        # 's' force le succès (override si le classifieur rate) ; 'q' abandonne sans récompense.
        clf_success = False
        if self._clf is not None:
            p = self._clf.prob(obs["pixels"])
            self._clf_hits = self._clf_hits + 1 if p >= self._clf.thresh else 0
            clf_success = self._clf_hits >= 2          # 2 frames consécutives -> pas de faux positif isolé
        success_final = bool(success or clf_success)
        info[TeleopEvents.SUCCESS] = success_final
        info[TeleopEvents.TERMINATE_EPISODE] = bool(terminate or success_final)
        done = bool(success_final or terminate)
        sparse_reward = 1.0 if success_final else 0.0
        return obs, sparse_reward, done, bool(truncated), info

    def render(self):
        return self._env.render()

    def close(self):
        if self._tele is not None:
            try: self._tele.disconnect()
            except Exception: pass
        self._env.close()


def make_pusht_robot_env(cfg=None):
    """Signature compatible avec lerobot.rl.actor.make_robot_env : renvoie (env, None).
    PUSHT_HEADLESS=1 désactive le clavier (test automatique sans intervention)."""
    import os
    task = getattr(cfg, "task", "PushT-v0") if cfg is not None else "PushT-v0"
    fps = getattr(cfg, "fps", 10) if cfg is not None else 10
    use_kb = os.environ.get("PUSHT_HEADLESS", "0") != "1"
    return PushTHILEnv(task=task, fps=fps, use_keyboard=use_kb), None
