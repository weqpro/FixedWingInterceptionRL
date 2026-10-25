from __future__ import annotations

from pathlib import Path
import numpy as np
from stable_baselines3 import PPO


def make_ppo_policy(model_path: str | Path):
    """
    Loads a trained Stable-Baselines3 PPO model and returns a policy function (obs -> action).
    Applies the exact same observation scaling (Step 1) used during training.
    """
    model = PPO.load(model_path)

    # Identical deterministic feature scales used inside PPOTrainingWrapper (Step 1)
    scales = np.array([
        1.0,   # roll rad
        1.0,   # pitch rad
        1.0,   # sin_psi
        1.0,   # cos_psi
        1.0,   # p
        1.0,   # q
        1.0,   # r
        0.02,  # airspeed
        0.001, # height AGL
        0.05,  # vertical speed
        0.001, # target forward
        0.001, # target right
        0.001  # target down
    ], dtype=np.float32)

    def policy(obs: np.ndarray) -> np.ndarray:
        # Step 1: scale observation before loading to model
        scaled_obs = obs * scales
        action, _ = model.predict(scaled_obs, deterministic=True)
        return action.astype(np.float32)

    return policy