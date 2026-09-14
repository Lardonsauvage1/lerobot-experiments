#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # --- PATCH RELJOINT : actions articulaires RELATIVES chunk-wise (ancre = obs courante) ---
    # Override les stats de normalisation des dims joints (0..DIMS-1) par les stats RELATIVES
    # (delta chunk-wise), gripper reste absolu. Le batch est rendu relatif avant le preprocessor.
    import os as _os  # local (nom réassigné plus loin dans la fonction -> local partout)
    if _os.environ.get("RELJOINT") == "1":
        import json as _json, numpy as _np
        _dims = int(_os.environ.get("RELJOINT_DIMS", "7"))
        _rs_path = _os.environ.get("RELJOINT_STATS",
            "data_cache/lerobot_can_ph_joint_birdview/meta/relstats_chunkwise.json")
        _rs = _json.load(open(_rs_path))
        _sa = dataset.meta.stats["action"]
        for _k in ("min", "max", "mean", "std"):
            _was_t = torch.is_tensor(_sa[_k])
            _v = (_sa[_k].cpu().numpy() if _was_t else _np.asarray(_sa[_k])).astype(_np.float32).copy()
            _v[:_dims] = _np.asarray(_rs[_k], dtype=_np.float32)[:_dims]
            _sa[_k] = torch.as_tensor(_v) if _was_t else _v
        logging.info(f"[RELJOINT] actions joints 0-{_dims-1} RELATIVES chunk-wise ; "
                     f"stats action override depuis {_rs_path} (gripper absolu)")

    # --- PATCH CAMP : mémoire compressée de l'historique d'ACTIONS concaténée à l'état ---
    # CAMP-lite (variante B) : le module mémoire est PRÉ-ENTRAÎNÉ ET GELÉ, donc on précalcule
    # m_t une fois pour toutes les frames. La boucle d'entraînement reste alors RIGOUREUSEMENT
    # celle d'un run normal — c'est ce qui satisfait la contrainte "pas plus long à entraîner".
    # cf. src/camp.py et docs/recherche/MEMOIRE.md.
    #   CAMP=1 CAMP_CKPT=results/runs/can/camp/memory_L64_K32_m32.pt
    _camp_M = None
    _camp_bank = None     # ⚠️ doit exister même sans CAMP : il est lu à chaque checkpoint
    if _os.environ.get("CAMP") == "1":
        import numpy as _np
        _camp_ft = _os.environ.get("CAMP_FINETUNE") == "1"   # variante A (papier) si 1
        # ce script vit dans experiments/lift/ ; la racine du repo n'est pas sur sys.path
        # quand il est lancé directement -> `from src import camp` échouait
        # (ModuleNotFoundError, nuit du 2026-09-09). On l'ajoute ici, sans dépendre du lanceur.
        import sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from src import camp as _camp
        _ck = torch.load(_os.environ["CAMP_CKPT"], weights_only=False)
        _ca = _ck["args"]
        _sdim = int(dataset.meta.features["observation.state"]["shape"][0])
        _adim = int(dataset.meta.features["action"]["shape"][0])
        _mem = _camp.CampMemory(_adim, _sdim, n_coef=_ca["K"], mem_dim=_ca["mem_dim"],
                                codebook_size=_ca["codebook"])
        _mem.load_state_dict(_ck["state_dict"])
        # variante A (papier) : le LSTM est DÉGELÉ et finetuné conjointement à lr x alpha.
        # variante B (défaut) : module gelé, codes précalculés une fois.
        for _prm in _mem.parameters():
            _prm.requires_grad_(_camp_ft)
        _nobs = int(getattr(cfg.policy, "n_obs_steps", 1))
        # ⚠️ précalcul sur TOUS les épisodes, pas seulement le train : LeRobot conserve
        # l'index GLOBAL dans les sous-ensembles (vérifié : ép. 150-151 -> index 17252+),
        # or la val-loss tire d'un LeRobotDataset séparé. Un M dimensionné sur le seul train
        # partirait hors bornes dès la première val-loss.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LRD
        _full = _LRD(cfg.dataset.repo_id, root=cfg.dataset.root,
                     delta_timestamps=dataset.delta_timestamps)
        if _camp_ft:
            # déroulement À LA VOLÉE depuis le début de l'épisode -> le gradient atteint le LSTM.
            # Coût mesuré : 66 ms/step à batch 32 (vs ~950 ms pour le step complet) = +7 %.
            _mem = _mem.to(cfg.policy.device)
            _camp_bank = _camp.EpisodeBank(_full, torch.device(cfg.policy.device))
            _camp_M = _camp.precompute_memory(_full, _mem.cpu(), _nobs, torch.device("cpu"))
            _mem = _mem.to(cfg.policy.device)
            logging.info("[CAMP] mode FINETUNING CONJOINT (variante A du papier) : LSTM dégelé")
        else:
            _camp_M = _camp.precompute_memory(_full, _mem, _nobs, torch.device("cpu"))
        del _full
        _md = int(_ca["mem_dim"])

        # la policy dérive ses shapes de ds_meta -> on étend la feature AVANT make_policy
        dataset.meta.features["observation.state"]["shape"] = (_sdim + _md,)
        # ... et les stats de normalisation, sinon le normaliseur reçoit 9 dims pour 41
        _flat = _camp_M[:, -1, :].numpy()
        _ext = {"min": _flat.min(0), "max": _flat.max(0), "mean": _flat.mean(0),
                "std": _flat.std(0) + 1e-6}
        for _q in (1, 10, 50, 90, 99):
            _ext[f"q{_q:02d}"] = _np.percentile(_flat, _q, axis=0)
        _ss = dataset.meta.stats["observation.state"]
        for _k, _v in list(_ss.items()):
            if _k == "count":
                continue
            _was_t = torch.is_tensor(_v)
            _arr = (_v.cpu().numpy() if _was_t else _np.asarray(_v)).astype(_np.float32)
            _add = _ext.get(_k, _np.zeros(_md, dtype=_np.float32)).astype(_np.float32)
            _new = _np.concatenate([_arr, _add])
            _ss[_k] = torch.as_tensor(_new) if _was_t else _new
        logging.info(f"[CAMP] mémoire GELÉE {_os.environ['CAMP_CKPT']} | "
                     f"state {_sdim}D -> {_sdim + _md}D | {_camp_M.shape[0]} frames précalculées "
                     f"| n_obs_steps={_nobs}")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # --- PATCH KPAMP : ajoute l'AMPLITUDE d'activation à chaque keypoint ------------------
    # Le spatial softmax NORMALISE : quelle que soit la force de l'activation, il sort une
    # distribution qui somme à 1, donc toujours 2 coordonnées. Il ne peut JAMAIS signaler
    # « ce motif est absent ». Mesuré le 2026-09-11 : quand la canette disparaît de l'image,
    # les keypoints se REPLACENT silencieusement (déplacement moyen 3,3 px, max 42,8 px) et
    # le dénoiseur reçoit une représentation d'apparence normale décrivant une scène qu'il
    # n'a pas. C'est peut-être le mécanisme derrière « il relève le bras comme s'il l'avait ».
    #
    # KPAMP=1 conserve, pour chaque keypoint, le LOGIT MAXIMAL de son canal avant softmax —
    # exactement l'information que la normalisation jette. 3 nombres par keypoint au lieu
    # de 2. Aucun autre changement : le Linear de sortie ramène au même feature_dim.
    if _os.environ.get("KPAMP") == "1":
        import torch.nn as _nn
        # ⚠️ au COOLDOWN, make_policy a déjà chargé --policy.pretrained_path dans un modèle
        # STANDARD -> les poids KPAMP (Linear 3*kp) ne rentrent pas et ça plante.
        # On patche l'architecture PUIS on recharge les poids du checkpoint.
        _reload = getattr(getattr(cfg, "policy", None), "pretrained_path", None)
        _enc = policy.diffusion.rgb_encoder
        _pool = _enc.pool

        class _SpatialSoftmaxAmp(_nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, features):
                if self.base.nets is not None:
                    features = self.base.nets(features)
                B, K, H, W = features.shape
                flat = features.reshape(B * K, H * W)
                att = _nn.functional.softmax(flat, dim=-1)
                xy = att @ self.base.pos_grid                      # (B*K, 2) — inchangé
                amp = flat.max(dim=-1, keepdim=True).values        # (B*K, 1) — l'info jetée
                return torch.cat([xy, amp], dim=-1).reshape(B, K, 3)

        _kp = int(policy.config.spatial_softmax_num_keypoints)
        _enc.pool = _SpatialSoftmaxAmp(_pool)
        _old = _enc.out
        _enc.out = _nn.Linear(_kp * 3, _old.out_features).to(_old.weight.device)
        with torch.no_grad():                                      # on repart des poids appris
            _enc.out.weight.zero_()
            _enc.out.weight[:, : _kp * 2] = _old.weight
            _enc.out.bias.copy_(_old.bias)
        logging.info(f"[KPAMP] amplitude ajoutée : {_kp} keypoints x 3 = {_kp*3} -> "
                     f"Linear({_kp*3}, {_old.out_features})")
        if _reload:
            from safetensors.torch import load_file as _lf
            import os as _os3
            _mp = _os3.path.join(str(_reload), "model.safetensors")
            if _os3.path.exists(_mp):
                _r = policy.load_state_dict(_lf(_mp), strict=False)
                logging.info(f"[KPAMP] poids rechargés depuis {_reload} "
                             f"(manquants {len(_r.missing_keys)}, inattendus {len(_r.unexpected_keys)})")


    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # ===================== PATCH AUXHEAD : têtes auxiliaires par caméra =====================
    # Problème visé : un modèle 2 caméras CONTIENT le modèle 1 caméra (poids du 2e encodeur à
    # zéro) et fait pourtant moins bien -> ce n'est pas l'expressivité, c'est l'optimisation.
    # Mécanisme documenté : compétition entre modalités / gradient starvation (Wang et al.
    # CVPR 2020 « What Makes Training Multi-modal Classification Networks Hard? » ; Pezeshki
    # et al. NeurIPS 2021). La branche la plus vite prédictive capte le gradient et affame
    # l'autre. Le dropout de caméra agit sur l'ENTRÉE et n'a rien donné (+2,6 pts, p=0,22) ;
    # ici on agit sur le GRADIENT : chaque encodeur reçoit sa propre tête de régression et sa
    # propre perte, donc chaque branche doit être prédictive SEULE.
    #
    # Coût : ~0,39 M par caméra pendant l'entraînement, et **0 au déploiement** — les têtes
    # ne sont volontairement PAS attachées à la policy, donc absentes du state_dict et des
    # checkpoints. Le modèle sauvegardé est bit-à-bit de même structure que sans AUXHEAD.
    _aux_w = float(_os.environ.get("AUXHEAD", "0") or 0)
    _aux_heads = None
    if _aux_w > 0:
        import einops as _ein
        from lerobot.utils.constants import ACTION as _K_ACT, OBS_IMAGES as _K_IMG, OBS_STATE as _K_ST
        _dm = policy.diffusion
        if not getattr(_dm, "rgb_encoder", None) or not isinstance(_dm.rgb_encoder, torch.nn.ModuleList):
            raise RuntimeError("AUXHEAD exige use_separate_rgb_encoder_per_camera=true")
        _n_cam = len(_dm.rgb_encoder)
        _fdim = _dm.rgb_encoder[0].feature_dim
        _sdim = cfg.policy.robot_state_feature.shape[0]
        _nobs = cfg.policy.n_obs_steps
        _H = cfg.policy.horizon
        _adim = cfg.policy.action_feature.shape[0]
        _hid = int(_os.environ.get("AUXHEAD_HIDDEN", "512"))
        _aux_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(_nobs * (_fdim + _sdim), _hid), torch.nn.ReLU(),
                torch.nn.Linear(_hid, _hid), torch.nn.ReLU(),
                torch.nn.Linear(_hid, _H * _adim))
            for _ in range(_n_cam)]).to(device)
        _np_aux = sum(q.numel() for q in _aux_heads.parameters())
        logging.info(f"[AUXHEAD] {_n_cam} tête(s), poids={_aux_w}, {_np_aux/1e6:.3f} M params "
                     f"(entraînement seulement, absentes du checkpoint)")

        # On réimplémente le conditionnement pour MÉMORISER les features par caméra.
        # Strictement équivalent à l'original dans le cas « encodeurs séparés ».
        _orig_prep = _dm._prepare_global_conditioning

        def _prep_stash(batch, _dm=_dm, _ein=_ein):
            B, S = batch[_K_ST].shape[:2]
            imgs = _ein.rearrange(batch[_K_IMG], "b s n ... -> n (b s) ...")
            per = [enc(im) for enc, im in zip(_dm.rgb_encoder, imgs, strict=True)]
            _dm._percam = [_ein.rearrange(f, "(b s) d -> b s d", b=B, s=S) for f in per]
            img_features = _ein.rearrange(torch.cat(per), "(n b s) ... -> b s (n ...)", b=B, s=S)
            return torch.cat([batch[_K_ST], img_features], dim=-1).flatten(start_dim=1)

        _dm._percam = None
        _dm._prepare_global_conditioning = _prep_stash

        _orig_closs = _dm.compute_loss

        def _closs_aux(batch, _dm=_dm, _heads=_aux_heads, _w=_aux_w):
            main = _orig_closs(batch)
            feats = getattr(_dm, "_percam", None)
            if not feats or not _dm.training:
                return main
            tgt = batch[_K_ACT]
            st = batch[_K_ST]
            aux = 0.0
            for h, f in zip(_heads, feats, strict=True):
                x = torch.cat([f, st], dim=-1).flatten(start_dim=1)
                aux = aux + torch.nn.functional.mse_loss(h(x).reshape(tgt.shape), tgt)
            return main + _w * aux / len(_heads)

        _dm.compute_loss = _closs_aux

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if _aux_heads is not None:
        # même précaution que CAMP : ajouter le groupe AVANT de (re)créer le scheduler,
        # sinon LambdaLR garde une lambda par groupe et casse au premier .step().
        optimizer.add_param_group({"params": list(_aux_heads.parameters()),
                                   "lr": float(cfg.optimizer.lr)})
        _o2, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        del _o2
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.optimizer = optimizer
            lr_scheduler.base_lrs = [g["lr"] for g in optimizer.param_groups]
            if hasattr(lr_scheduler, "lr_lambdas") and len(lr_scheduler.lr_lambdas) < len(optimizer.param_groups):
                lr_scheduler.lr_lambdas = list(lr_scheduler.lr_lambdas) + \
                    [lr_scheduler.lr_lambdas[-1]] * (len(optimizer.param_groups) - len(lr_scheduler.lr_lambdas))
    if _camp_M is not None and _os.environ.get("CAMP_FINETUNE") == "1":
        # ⚠️ ajouter le groupe APRÈS la création du scheduler casse LambdaLR (il garde une
        # lambda par groupe -> "zip() argument 2 is shorter"). On l'ajoute donc ici, puis on
        # RECRÉE le scheduler pour qu'il voie les deux groupes.
        # ⚠️ le lr des param_groups vaut 0 au démarrage (warmup) -> on lit cfg, pas le groupe.
        _alpha = float(_os.environ.get("CAMP_ALPHA", "0.1"))
        _base = float(cfg.optimizer.lr)
        optimizer.add_param_group({"params": [q for q in _mem.parameters() if q.requires_grad],
                                   "lr": _base * _alpha})
        _o2, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        del _o2
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.optimizer = optimizer
            lr_scheduler.base_lrs = [g["lr"] for g in optimizer.param_groups]
            if hasattr(lr_scheduler, "lr_lambdas") and len(lr_scheduler.lr_lambdas) < len(optimizer.param_groups):
                lr_scheduler.lr_lambdas = list(lr_scheduler.lr_lambdas) + \
                    [lr_scheduler.lr_lambdas[-1]] * (len(optimizer.param_groups) - len(lr_scheduler.lr_lambdas))
        logging.info(f"[CAMP] LSTM dans l'optimiseur : lr = {_base} x {_alpha} = {_base*_alpha}")
    # --- LR CONSTANT (decision methodo : pas de cosine) : CONST_LR=1 desactive le scheduler ---
    import os as _os
    if _os.environ.get("CONST_LR"):
        _clr = float(_os.environ.get("CONST_LR_VALUE", "1e-4"))
        for _g in optimizer.param_groups:
            _g["lr"] = _clr
        lr_scheduler = None  # update_policy ne fait .step() que si lr_scheduler is not None -> LR fixe
        if is_main_process:
            logging.info(f"[CONST_LR] scheduler desactive -> LR constant = {_clr}")

    # --- COOLDOWN LR (test WSD) : COOLDOWN_STEPS=N decroit le LR de COOLDOWN_LR0 -> 0 lineairement ---
    #     A utiliser SANS resume (charge les poids via --policy.pretrained_path, optimiseur frais),
    #     pour annealer proprement depuis un plateau constant et mesurer si on depasse le SWA/merge.
    if _os.environ.get("COOLDOWN_STEPS") and not _os.environ.get("ENCODER_LR"):  # ENCODER_LR gère son propre cooldown par groupe
        from torch.optim.lr_scheduler import LambdaLR
        _n_cd = int(_os.environ["COOLDOWN_STEPS"])
        _lr0 = float(_os.environ.get("COOLDOWN_LR0", "1e-4"))
        for _g in optimizer.param_groups:
            _g["lr"] = _lr0
        lr_scheduler = LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / _n_cd))
        if is_main_process:
            logging.info(f"[COOLDOWN] LR lineaire {_lr0} -> 0 sur {_n_cd} steps")

    # --- ENCODER_LR : LR par GROUPE pour backbone PRÉ-ENTRAÎNÉ (test B) ---
    #     encodeur vision (rgb_encoder.*) a un LR plus bas que le reste pour ne pas detruire ImageNet.
    #     ENCODER_LR = LR encodeur (ex 1e-5) ; UNET_LR = LR du reste (defaut 1e-4). LR constant (pas de scheduler).
    if _os.environ.get("ENCODER_LR"):
        import torch as _torch
        _enc_lr = float(_os.environ["ENCODER_LR"])
        _base_lr = float(_os.environ.get("UNET_LR", "1e-4"))
        _enc_p, _other_p = [], []
        for _n, _p in policy.named_parameters():
            if not _p.requires_grad:
                continue
            (_enc_p if "rgb_encoder" in _n else _other_p).append(_p)
        _def = dict(optimizer.defaults); _def.pop("lr", None)
        optimizer = type(optimizer)(
            [{"params": _enc_p, "lr": _enc_lr}, {"params": _other_p, "lr": _base_lr}],
            lr=_base_lr, **_def)
        # COOLDOWN par GROUPE : LambdaLR multiplie l'initial_lr de CHAQUE groupe -> ratio 1e-5/1e-4 préservé -> 0
        if _os.environ.get("COOLDOWN_STEPS"):
            from torch.optim.lr_scheduler import LambdaLR
            _n_cd = int(_os.environ["COOLDOWN_STEPS"])
            lr_scheduler = LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / _n_cd))
            if is_main_process:
                logging.info(f"[ENCODER_LR+COOLDOWN] 2 groupes ({_enc_lr}/{_base_lr}) -> 0 lineaire sur {_n_cd} steps")
        else:
            lr_scheduler = None  # LR constant par groupe
            if is_main_process:
                logging.info(f"[ENCODER_LR] 2 groupes : encodeur={_enc_lr} ({len(_enc_p)} tenseurs) / reste={_base_lr} ({len(_other_p)} tenseurs)")

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    # --- PATCH val-loss : val = épisodes du dataset PAS dans cfg.dataset.episodes (auto-cohérent) ---
    val_iter = None
    if cfg.dataset.episodes:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        val_episodes = sorted(set(range(dataset.meta.total_episodes)) - set(cfg.dataset.episodes))
        if val_episodes:
            val_ds = LeRobotDataset(
                cfg.dataset.repo_id, root=cfg.dataset.root,
                episodes=val_episodes, delta_timestamps=dataset.delta_timestamps,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True,
            )
            val_iter = cycle(val_loader)
            if is_main_process:
                logging.info(f"[val-loss] {len(val_episodes)} episodes val, {val_ds.num_frames} frames")
    # --- END PATCH ---

    # --- PATCH EMA (optionnel) : EMA=1 maintient une moyenne mobile expo des poids ---
    #     sauvée à chaque save_freq dans <output_dir>_ema (permet d'évaluer brut vs EMA
    #     sur EXACTEMENT la même plage d'entraînement). EMA_DECAY par défaut 0.9999.
    import os as _os_ema
    from pathlib import Path as _PathEma
    _ema_params = None
    if bool(_os_ema.environ.get("EMA")):
        _ema_decay = float(_os_ema.environ.get("EMA_DECAY", "0.9999"))
        _ema_model0 = accelerator.unwrap_model(policy)
        _ema_params = {n: p.detach().clone().float()
                       for n, p in _ema_model0.named_parameters() if p.requires_grad}
        if is_main_process:
            logging.info(f"[EMA] activé decay={_ema_decay} ({len(_ema_params)} tenseurs) "
                         f"-> {cfg.output_dir}_ema")
    # --- END PATCH EMA ---

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    # PATCH RELJOINT : rend le batch relatif (action[:, :, :dims] -= obs.state[:, -1, :dims]) AVANT normalisation
    _reljoint = _os.environ.get("RELJOINT") == "1"
    _rj_dims = int(_os.environ.get("RELJOINT_DIMS", "7"))
    def _to_relative(b):
        if not _reljoint:
            return b
        st, act = b.get("observation.state"), b.get("action")
        if st is None or act is None:
            return b
        anchor = st[:, -1:, :_rj_dims]            # (B, 1, dims)
        act = act.clone()
        act[:, :, :_rj_dims] = act[:, :, :_rj_dims] - anchor
        b = dict(b); b["action"] = act
        return b

    def _add_memory(b):
        """Concatène m_t (32 dims) à observation.state, avant le preprocessor donc avant
        normalisation. `b["index"]` = index GLOBAL de la frame courante ; les n_obs_steps
        codes ont été pré-empilés et clampés aux frontières d'épisode par precompute_memory."""
        if _camp_M is None:
            return b
        st = b.get("observation.state")
        if st is None or "index" not in b:
            return b
        if _camp_bank is not None:
            # (B, mem) au pas courant, gradient inclus ; on réplique sur les n_obs_steps
            m_t, _lvq = _camp_bank.memory_for(_mem, b["episode_index"], b["frame_index"])
            # le batch brut est encore sur CPU ici (le transfert se fait dans le preprocessor)
            # alors que la banque vit sur le device -> ramener explicitement.
            mm = m_t.unsqueeze(1).expand(-1, st.shape[1], -1).to(st.device, st.dtype)
        else:
            mm = _camp_M[b["index"].cpu()].to(st.device, st.dtype)
        b = dict(b)
        b["observation.state"] = torch.cat([st, mm], dim=-1)
        return b

    # PATCH CAMDROP : dropout de caméra (anti-copycat sur la caméra de poignet).
    # Hypothèse visée : l'image du poignet est une fonction quasi déterministe de la pose de
    # la pince, déjà présente dans observation.state. Elle prédit donc l'action du moment
    # sans rien apprendre de l'approche -> raccourci causal (copycat, Wen et al. 2020).
    # Remède : avec probabilité p, masquer UNE caméra tirée au hasard, pour que le réseau ne
    # puisse jamais se reposer sur une seule. Appliqué APRÈS le preprocessor : sur une entrée
    # normalisée, 0 correspond exactement à l'image moyenne — un signal « absent » neutre,
    # pas une image noire (qui serait, elle, une valeur extrême).
    # ⚠️ jamais appliqué à la val-loss (chemin séparé ligne ~733) : on veut mesurer le modèle
    # complet, pas le modèle amputé.
    _camdrop_p = float(_os.environ.get("CAMDROP", "0") or 0)
    if _camdrop_p > 0:
        logging.info(f"[CAMDROP] dropout de caméra actif : p={_camdrop_p} "
                     f"(une caméra masquée au hasard sur {_camdrop_p:.0%} des échantillons)")

    def _cam_dropout(b):
        if _camdrop_p <= 0:
            return b
        keys = sorted(k for k in b if k.startswith("observation.images"))
        if len(keys) < 2:
            return b                      # mono-caméra : rien à masquer
        ref = b[keys[0]]
        n = ref.shape[0]
        drop = torch.rand(n, device=ref.device) < _camdrop_p
        which = torch.randint(len(keys), (n,), device=ref.device)
        if not bool(drop.any()):
            return b
        b = dict(b)
        for j, k in enumerate(keys):
            m = drop & (which == j)
            if bool(m.any()):
                x = b[k].clone()
                x[m] = 0.0                # = image moyenne après normalisation
                b[k] = x
        return b

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = _cam_dropout(preprocessor(_add_memory(_to_relative(next(dl_iter)))))
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if _ema_params is not None:
            _m_ema = accelerator.unwrap_model(policy)
            for _n, _p in _m_ema.named_parameters():
                if _n in _ema_params:
                    _ema_params[_n].mul_(_ema_decay).add_(_p.detach().float(), alpha=1.0 - _ema_decay)
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            # --- PATCH val-loss : passe read-only sur un batch val (même preprocessor) ---
            if val_iter is not None:
                policy.eval()
                with torch.no_grad():
                    val_loss, _ = policy.forward(preprocessor(_add_memory(_to_relative(next(val_iter)))))
                policy.train()
                logging.info(f"val_loss:{float(val_loss):.4f} valstep:{step}")
            # --- END PATCH ---
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                if _camp_bank is not None:
                    # ⭐ variante A : le LSTM a été MODIFIÉ par le finetuning conjoint. Sans
                    # cette sauvegarde, l'éval rechargerait le module pré-entraîné d'origine
                    # et la variante A serait indiscernable de la B — échec SILENCIEUX.
                    import os as _os2
                    _mp = _os2.path.join(str(checkpoint_dir), "memory_finetuned.pt")
                    torch.save({"state_dict": {k: v.cpu() for k, v in _mem.state_dict().items()},
                                "args": _ck["args"]}, _mp)
                    logging.info(f"[CAMP] LSTM finetuné sauvé -> {_mp}")
                update_last_checkpoint(checkpoint_dir)
                if _ema_params is not None:
                    # sauve les poids EMA dans <output_dir>_ema/checkpoints/<step> (swap in/out, exception-safe)
                    try:
                        _ema_ckpt = _PathEma(
                            str(checkpoint_dir).replace(str(cfg.output_dir), str(cfg.output_dir) + "_ema", 1)
                        )
                        _m_ema = accelerator.unwrap_model(policy)
                        _bak = {n: p.detach().clone() for n, p in _m_ema.named_parameters() if n in _ema_params}
                        try:
                            for _n, _p in _m_ema.named_parameters():
                                if _n in _ema_params:
                                    _p.data.copy_(_ema_params[_n].to(_p.dtype))
                            save_checkpoint(
                                checkpoint_dir=_ema_ckpt, step=step, cfg=cfg, policy=_m_ema,
                                optimizer=optimizer, scheduler=lr_scheduler,
                                preprocessor=preprocessor, postprocessor=postprocessor,
                            )
                        finally:
                            for _n, _p in _m_ema.named_parameters():
                                if _n in _bak:
                                    _p.data.copy_(_bak[_n])
                        logging.info(f"[EMA] checkpoint EMA sauvé: {_ema_ckpt}")
                    except Exception as _e_ema:
                        logging.warning(f"[EMA] save échoué (ignoré): {_e_ema}")
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
