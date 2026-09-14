"""Conversion du dataset Robomimic Can (PickPlaceCan) PH au format LeRobot.

Même pipeline que lift_to_lerobot.py (replay des états dans le sim → rendu agentview
→ écriture LeRobot), adapté à Can : env PickPlaceCan, object 14D (state total 23D),
horizon ~2× plus long que Lift.

  python -u src/can_to_lerobot.py            # conversion complète (200 démos)
  python -u src/can_to_lerobot.py --smoke    # 2 démos, vérif rendu + dataset.create
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path

import h5py
import numpy as np

from src import can_occlusion

HDF5_PATH = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUTPUT_ROOT = Path("data_cache/lerobot_can_ph")
REPO_ID = "local/can_ph"
DEFAULT_IMAGE_SIZE = 96
CAMERA_NAME = "agentview"
FPS = 20
TASK_DESCRIPTION = "Pick up the can and place it in the bin."
# State Can = proprio + position canette = 12D (voir src/can_eval.py pour le pourquoi).
# ⚠️ Côté DATASET (1.4), la can_pos est dans object[0:3] (côté env live 1.5 : object[7:10]).
STATE_DIM = 3 + 4 + 2 + 3  # eef_pos + eef_quat + gripper_qpos + can_pos(3) = 12


def convert(proprio=False, wrist=False, wrist_only=False, birdview=False,
            birdview_only=False, birdview_wrist=False,
            image_size=DEFAULT_IMAGE_SIZE, smoke=False, joint=False, delta=False, coherent=False,
            occlude=0.0):
    """proprio=True : state 9D = proprio SEULE (sans can_pos), test transférabilité réel.
    wrist=True : agentview + robot0_eye_in_hand (2 cams), features observation.images.X.
    wrist_only=True : robot0_eye_in_hand SEUL sous observation.image (mono-cam wrist).
    birdview=True : agentview + birdview (2 cams scene-fixées, parallaxe vraie).
       Mutuellement exclusif avec wrist/wrist_only.
    birdview_only=True : birdview SEULE sous observation.image (mono-cam vue de dessus).
    birdview_wrist=True : birdview + robot0_eye_in_hand (2 cams).
       Ces deux modes servent le cas RÉEL : là-bas il n'y a pas de vue extérieure qui voit
       tout, et le bras — qui arrive par le haut — masque justement la vue de dessus. C'est
       la seule configuration où la caméra embarquée peut apporter une information que la
       caméra fixe n'a pas.
    image_size : résolution de rendu (default 96 = standard Robomimic ; 128/160/224 possibles).
    occlude : rayon d'occlusion en MÈTRES (0 = désactivé). La canette est retirée de la
       scène au rendu quand le préhenseur est au-dessus d'elle à moins de `occlude` m en XY
       (voir src/can_occlusion.py). Sert au banc Can-Occluded : le robot réel a été entraîné
       SUR des données où la cible disparaît — entraîner en clair et n'occlure qu'à l'éval
       mesurerait un décalage train/test, pas le problème de mémoire."""
    _modes = [wrist, wrist_only, birdview, birdview_only, birdview_wrist]
    if sum(_modes) > 1:
        raise ValueError("--wrist, --wrist-only, --birdview, --birdview-only, "
                         "--birdview-wrist sont mutuellement exclusifs")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs

    IMAGE_SIZE = image_size
    occ_tag = f"_occ{int(round(occlude * 100)):02d}" if occlude > 0 else ""
    suffix = ("_joint" if joint else "") + ("_delta" if delta else "") + ("_coh" if coherent else "") + ("_proprio" if proprio else "") + ("_wrist" if wrist else "") + \
             ("_wristonly" if wrist_only else "") + ("_birdview" if birdview else "") + \
             ("_birdviewonly" if birdview_only else "") + ("_birdviewwrist" if birdview_wrist else "") + \
             (f"_{image_size}" if image_size != DEFAULT_IMAGE_SIZE else "") + occ_tag
    out_root = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + suffix)
    repo_id = REPO_ID + suffix
    out = out_root.with_name(out_root.name + "_smoke") if smoke else out_root
    if out.exists():
        print(f"⚠️  {out} existe déjà. Supprime-le pour relancer.")
        return

    base_names = ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_w", "eef_quat_x",
                  "eef_quat_y", "eef_quat_z", "gripper_l", "gripper_r"]
    if joint:
        state_dim = 9
        state_names = [f"joint_{k}" for k in range(7)] + ["gripper_l", "gripper_r"]  # 7 joints + gripper
    elif proprio:
        state_dim, state_names = 9, base_names                          # SANS can_pos
    else:
        state_dim, state_names = 12, base_names + ["can_x", "can_y", "can_z"]

    img_feat = {"dtype": "video", "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
                "names": ["channels", "height", "width"]}
    if wrist:
        # Multi-caméras : convention LeRobot 'observation.images.X' (sera détecté par
        # diffusion policy comme features VISUAL multiples, vision backbone partagé).
        image_features = {"observation.images.agentview": img_feat,
                          "observation.images.wrist": img_feat}
    elif birdview:
        image_features = {"observation.images.agentview": img_feat,
                          "observation.images.birdview": img_feat}
    elif birdview_wrist:
        image_features = {"observation.images.birdview": img_feat,
                          "observation.images.wrist": img_feat}
    else:
        image_features = {"observation.image": img_feat}
    features = {
        **image_features,
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": ({"dtype": "float32", "shape": (8,),
                    "names": [f"jt_{k}" for k in range(7)] + ["gripper"]} if joint else
                   {"dtype": "float32", "shape": (7,),
                    "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]}),
    }

    print(f"Création dataset LeRobot : {out} (proprio={proprio}, wrist={wrist}, "
          f"state={state_dim}D, cams={list(image_features.keys())})")
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=out,
                                    robot_type="panda", use_videos=True, video_backend="pyav")

    print(f"Setup env Robosuite PickPlaceCan{' [COHÉRENT: replay OSC dynamique 1.5.2]' if coherent else ''}...")
    ctrl_cfg = None
    if coherent:
        # contrôleur OSC_POSE (format 1.5.2) pour rejouer les actions originales -> jpos LIVE cohérents
        from robosuite.controllers import load_composite_controller_config
        ctrl_cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        ctrl_cfg["body_parts"]["right"] = {
            "type": "OSC_POSE", "input_type": "delta", "input_ref_frame": "base", "interpolation": None,
            "ramp_ratio": 0.2, "gripper": {"type": "GRIP"}, "kp": 150, "damping_ratio": 1,
            "impedance_mode": "fixed", "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None, "uncouple_pos_ori": True, "control_delta": True}
    env = rs.make(env_name="PickPlaceCan", robots="Panda",
                  has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=False, camera_names=CAMERA_NAME,
                  camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
                  control_freq=FPS, reward_shaping=False,
                  **({"controller_configs": ctrl_cfg} if coherent else {}))

    with h5py.File(HDF5_PATH, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if smoke:
            demo_names = demo_names[:2]
        print(f"\n{len(demo_names)} démos à convertir...")
        for d_idx, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            states_demo = demo["states"][:]
            actions_demo = demo["actions"][:]
            o = demo["obs"]
            jpos = o["robot0_joint_pos"][:] if joint else None  # (T,7) pour state + action joint
            if joint:
                state_low = np.concatenate([jpos, o["robot0_gripper_qpos"][:]], axis=1)  # 9D joint
            elif proprio:
                state_low = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                                            o["robot0_gripper_qpos"][:]], axis=1)  # 9D
            else:
                # 12D = eef_pos(3) + eef_quat(4) + gripper(2) + can_pos = object[0:3] (dataset 1.4)
                state_low = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                                            o["robot0_gripper_qpos"][:], o["object"][:, 0:3]], axis=1)
            assert state_low.shape[1] == state_dim, f"state dim {state_low.shape[1]} != {state_dim}"

            def render(i_render):  # rend les images à l'état COURANT de l'env
                hide = occlude > 0 and can_occlusion.occluded_from_xyz(
                    o["robot0_eef_pos"][i_render], o["object"][i_render, 0:3], occlude)

                def _shot(cam):
                    if hide:
                        return can_occlusion.render_without_can(env, IMAGE_SIZE, IMAGE_SIZE, cam)
                    return env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name=cam)[::-1]

                fr = {}
                if wrist or birdview or birdview_wrist:
                    if wrist:
                        cam_pairs = [("observation.images.agentview", "agentview"),
                                     ("observation.images.wrist", "robot0_eye_in_hand")]
                    elif birdview:
                        cam_pairs = [("observation.images.agentview", "agentview"),
                                     ("observation.images.birdview", "birdview")]
                    else:   # birdview_wrist
                        cam_pairs = [("observation.images.birdview", "birdview"),
                                     ("observation.images.wrist", "robot0_eye_in_hand")]
                    for key, cam in cam_pairs:
                        img = _shot(cam)
                        fr[key] = np.ascontiguousarray(img.transpose(2, 0, 1))
                else:
                    cam = ("robot0_eye_in_hand" if wrist_only else
                           "birdview" if birdview_only else CAMERA_NAME)
                    img = _shot(cam)
                    fr["observation.image"] = np.ascontiguousarray(img.transpose(2, 0, 1))
                return fr

            if coherent:
                # --- REPLAY DYNAMIQUE OSC (1.5.2) : jpos LIVE cohérents avec l'éval ---
                assert joint, "--coherent requiert --joint"
                env.reset(); env.sim.set_state_from_flattened(states_demo[0]); env.sim.forward()
                jl, grip, imgs = [], [], []
                for i in range(actions_demo.shape[0]):
                    ob = env._get_observations()
                    jl.append(np.asarray(ob["robot0_joint_pos"]).flatten())
                    grip.append(np.asarray(ob["robot0_gripper_qpos"]).flatten())
                    imgs.append(render(i))
                    env.step(actions_demo[i].astype(np.float32))          # rejoue l'action OSC originale
                obf = env._get_observations(); jl.append(np.asarray(obf["robot0_joint_pos"]).flatten())
                jl = np.stack(jl)  # (T+1, 7) trajectoire jpos LIVE cohérente
                for i in range(actions_demo.shape[0]):
                    jt = jl[i + 1] - jl[i] if delta else jl[i + 1]         # delta ou absolu, repère 1.5.2
                    act = np.concatenate([jt, [actions_demo[i, 6]]]).astype(np.float32)
                    st = np.concatenate([jl[i], grip[i]]).astype(np.float32)  # state = jpos_live + gripper
                    dataset.add_frame({"observation.state": st, "action": act,
                                       "task": TASK_DESCRIPTION, **imgs[i]})
            else:
                # --- mode legacy : set_state par frame (jpos HDF5) ---
                env.reset()
                for i in range(states_demo.shape[0]):
                    env.sim.set_state_from_flattened(states_demo[i]); env.sim.forward()
                    if joint:
                        jt = jpos[i + 1] if i + 1 < jpos.shape[0] else jpos[i]
                        if delta: jt = jt - jpos[i]
                        act = np.concatenate([jt, [actions_demo[i, 6]]]).astype(np.float32)
                    else:
                        act = actions_demo[i].astype(np.float32)
                    dataset.add_frame({"observation.state": state_low[i].astype(np.float32),
                                       "action": act, "task": TASK_DESCRIPTION, **render(i)})
            dataset.save_episode()
            if (d_idx + 1) % 20 == 0:
                print(f"  {d_idx + 1}/{len(demo_names)} démos", flush=True)

    env.close()
    print(f"\n✓ Dataset créé : {out} | frames={dataset.num_frames} épisodes={dataset.num_episodes}")
    if smoke:
        # vérif : sauver une image pour inspection visuelle
        import imageio
        f = h5py.File(HDF5_PATH, "r")
        d0 = f["data"]["demo_0"]
        env2 = rs.make(env_name="PickPlaceCan", robots="Panda", has_renderer=False,
                       has_offscreen_renderer=True, use_camera_obs=False, camera_names=CAMERA_NAME,
                       camera_heights=256, camera_widths=256, control_freq=FPS)
        env2.reset(); env2.sim.set_state_from_flattened(d0["states"][len(d0["states"])//2]); env2.sim.forward()
        mid = env2.sim.render(height=256, width=256, camera_name=CAMERA_NAME)[::-1]
        imageio.imwrite("data_cache/can_render_check.png", mid)
        print("  rendu mi-démo sauvé : data_cache/can_render_check.png (vérifie que c'est bien la scène Can)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proprio", action="store_true", help="state 9D proprio seule (sans can_pos)")
    ap.add_argument("--wrist", action="store_true", help="ajoute la caméra robot0_eye_in_hand (2 vues)")
    ap.add_argument("--wrist-only", dest="wrist_only", action="store_true",
                    help="UNIQUEMENT la wrist camera (mono-cam, sous observation.image)")
    ap.add_argument("--birdview", action="store_true",
                    help="agentview + birdview (2 cams scene-fixées, parallaxe)")
    ap.add_argument("--birdview-only", dest="birdview_only", action="store_true",
                    help="birdview SEULE sous observation.image (mono-cam vue de dessus)")
    ap.add_argument("--birdview-wrist", dest="birdview_wrist", action="store_true",
                    help="birdview + poignet : le cas RÉEL (pas de vue extérieure complète)")
    ap.add_argument("--image-size", dest="image_size", type=int, default=DEFAULT_IMAGE_SIZE,
                    help=f"résolution de rendu (default {DEFAULT_IMAGE_SIZE} ; suffixe ajouté si != default)")
    ap.add_argument("--joint", action="store_true",
                    help="espace ARTICULAIRE : state=joint_pos(7)+gripper(2), action=joint_pos[t+1] abs(7)+gripper")
    ap.add_argument("--delta", action="store_true",
                    help="avec --joint : action = DELTA articulaire (jpos[t+1]-jpos[t]) au lieu d'absolu")
    ap.add_argument("--coherent", action="store_true",
                    help="avec --joint : replay OSC dynamique -> jpos LIVE cohérents 1.5.2 (fix mismatch version)")
    ap.add_argument("--occlude", type=float, default=0.0,
                    help="rayon d'occlusion en m (ex 0.10) : la canette disparaît du rendu quand "
                         "le préhenseur est au-dessus d'elle à moins de ce rayon. Banc Can-Occluded.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    convert(proprio=a.proprio, wrist=a.wrist, wrist_only=a.wrist_only, birdview=a.birdview,
            birdview_only=a.birdview_only, birdview_wrist=a.birdview_wrist,
            image_size=a.image_size, smoke=a.smoke, joint=a.joint, delta=a.delta,
            coherent=a.coherent, occlude=a.occlude)
