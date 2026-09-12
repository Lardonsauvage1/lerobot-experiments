#!/usr/bin/env python
"""
Nœud ROS2 d'inférence — tâche "prise de pomme" (Diffusion Policy, LeRobot).
=========================================================================
Nœud de RÉFÉRENCE : câble EXACTEMENT le prétraitement de l'entraînement (c'était LE bug :
sans le `preprocessor`/normalisation, le modèle sort une pose constante ~1 rad à côté).

Chaîne : 2 caméras compressées + /joint_states  ->  obs normalisée  ->  select_action  ->
         /arm_controller/joint_trajectory (5 joints) + /gripper (Bool).

ENVIRONNEMENT : nécessite un Python qui a À LA FOIS rclpy (ROS2) ET torch+lerobot (le venv).
  Deux options :
    (a) installer rclpy dans le venv lerobot, ou
    (b) `pip install lerobot torch` dans l'env ROS2, ou
    (c) sourcer ROS2 puis lancer avec le venv en ajoutant les paths ROS2 au PYTHONPATH.
  Lancer :  python roby_infer_node.py   (ROS2 sourcé)

ARCHITECTURE temps réel : l'inférence (réseau diffusion ~540 ms sur CPU) tourne dans un THREAD
séparé qui remplit un tampon d'actions ; un timer publie à 15 Hz depuis ce tampon (tient last
si vide). Ça découple le calcul lourd de la cadence de contrôle.
"""
import threading
from collections import deque

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# ============================ CONFIG (à ajuster) ============================
MODEL_PATH   = "/home/sam/lerobot-experiments/deployable_models/apple/B_cd5k_from12000/brut"
DEVICE       = "cpu"                 # atomman = CPU (pas de CUDA)
CONTROL_HZ   = 15.0                  # cadence de contrôle = fps des données (NE PAS changer)
IMG_SIZE     = 224
GRIPPER_THR  = 0.5                   # action[5] > seuil => FERMER
JOINT_NAMES  = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
TOPIC_LEFT   = "/head_camera/left/image_raw/compressed"
TOPIC_RIGHT  = "/head_camera/right/image_raw/compressed"
TOPIC_JOINTS = "/joint_states"
TOPIC_TRAJ   = "/arm_controller/joint_trajectory"
TOPIC_GRIP   = "/gripper"
POINT_DT     = 2.0 / CONTROL_HZ      # time_from_start d'un setpoint (léger lissage)
# ===========================================================================


def decode_resize(jpeg_bytes) -> torch.Tensor:
    """CompressedImage JPEG -> tensor [3,224,224] float[0,1] RGB CHW (EXACTEMENT comme le dataset)."""
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)                 # BGR 480x640
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                # <-- BGR->RGB OBLIGATOIRE
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # CHW, [0,1]
    return t


class AppleInferNode(Node):
    def __init__(self):
        super().__init__("apple_infer_node")
        self.get_logger().info(f"Chargement modèle : {MODEL_PATH} (device {DEVICE})")
        self.dev = torch.device(DEVICE)
        self.policy = DiffusionPolicy.from_pretrained(MODEL_PATH).eval().to(self.dev)
        self.policy.reset()
        # ⚠️ override device_processor : checkpoint entraîné en mps -> forcer le device local
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=MODEL_PATH,
            preprocessor_overrides={"device_processor": {"device": DEVICE}})
        self.img_keys = [k for k in self.policy.config.input_features if "image" in k]

        # état partagé (dernières obs) + tampon d'actions
        self._lock = threading.Lock()
        self._left = None; self._right = None; self._joints = None
        self._buf = deque(maxlen=64)
        self._last_action = None
        self._last_grip = None

        # QoS best_effort pour les images compressées (comme à l'enregistrement)
        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, TOPIC_LEFT,  self._cb_left,  qos_img)
        self.create_subscription(CompressedImage, TOPIC_RIGHT, self._cb_right, qos_img)
        self.create_subscription(JointState, TOPIC_JOINTS, self._cb_joints, 10)

        self.pub_traj = self.create_publisher(JointTrajectory, TOPIC_TRAJ, 10)
        self.pub_grip = self.create_publisher(Bool, TOPIC_GRIP, 10)

        # thread d'inférence + timer de publication 15 Hz
        self._run = True
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()
        self.create_timer(1.0 / CONTROL_HZ, self._publish_tick)
        self.get_logger().info("Nœud prêt. En attente des 2 caméras + /joint_states…")

    # ---- callbacks : on cache juste la dernière donnée de chaque flux ----
    def _cb_left(self, msg):
        with self._lock: self._left = bytes(msg.data)
    def _cb_right(self, msg):
        with self._lock: self._right = bytes(msg.data)
    def _cb_joints(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            j = [msg.position[idx[n]] for n in JOINT_NAMES]      # ordre joint_1..joint_5 garanti
        except KeyError:
            j = list(msg.position[:5])                            # fallback : 5 premiers
        with self._lock: self._joints = np.asarray(j, dtype=np.float32)

    def _get_obs(self):
        with self._lock:
            if self._left is None or self._right is None or self._joints is None:
                return None
            left, right, joints = self._left, self._right, self._joints.copy()
        obs = {
            self.img_keys[0]: decode_resize(left).unsqueeze(0).to(self.dev),   # .left
            self.img_keys[1]: decode_resize(right).unsqueeze(0).to(self.dev),  # .right
            "observation.state": torch.from_numpy(joints).unsqueeze(0).to(self.dev),
        }
        return obs

    # ---- thread d'inférence : produit des actions ~15 Hz (bursty), remplit le tampon ----
    def _infer_loop(self):
        period = 1.0 / CONTROL_HZ
        import time
        while self._run:
            t0 = time.perf_counter()
            obs = self._get_obs()
            if obs is None:
                time.sleep(0.05); continue
            with torch.no_grad():
                # ⚠️⚠️ pre() = LA correction : normalise l'obs. post() dé-normalise l'action.
                a = self.post(self.policy.select_action(self.pre(obs)))
            a = a.squeeze(0).cpu().numpy()
            with self._lock:
                self._buf.append(a)
            dt = time.perf_counter() - t0
            if dt < period:                      # pacer à ~15 Hz (les appels "file" sont rapides)
                time.sleep(period - dt)

    # ---- timer 15 Hz : publie une action depuis le tampon (tient la dernière si vide) ----
    def _publish_tick(self):
        with self._lock:
            a = self._buf.popleft() if self._buf else self._last_action
            if a is not None: self._last_action = a
        if a is None:
            return
        # joints -> JointTrajectory
        jt = JointTrajectory(); jt.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in a[:5]]
        pt.time_from_start.sec = int(POINT_DT)
        pt.time_from_start.nanosec = int((POINT_DT % 1.0) * 1e9)
        jt.points = [pt]
        jt.header.stamp = self.get_clock().now().to_msg()
        self.pub_traj.publish(jt)
        # gripper -> Bool (publie seulement au changement, comme à l'enregistrement)
        close = bool(a[5] > GRIPPER_THR)
        if close != self._last_grip:
            self.pub_grip.publish(Bool(data=close))
            self._last_grip = close

    def destroy_node(self):
        self._run = False
        super().destroy_node()


def main():
    rclpy.init()
    node = AppleInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
