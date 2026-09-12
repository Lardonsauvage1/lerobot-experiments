#!/usr/bin/env python
"""Lance l'actor HIL-SERL sur Push-T en remplaçant make_robot_env par notre wrapper."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
_orig=torch.load
def _load(*a,**k):
    k["weights_only"]=False
    return _orig(*a,**k)
torch.load=_load
import lerobot.policies.sac.configuration_sac  # noqa: F401  (enregistre le type sac)
import lerobot.rl.actor as A
from pusht_hil_env import make_pusht_robot_env
A.make_robot_env = make_pusht_robot_env   # patch le nom que actor.py détient
A.actor_cli()
