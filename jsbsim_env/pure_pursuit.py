import math
import numpy as np

# Import indices from the new environment structure
from jsbsim_env.envs.intercept import (
    OBS_P, OBS_PITCH, OBS_Q, OBS_ROLL,
    OBS_TARGET_DOWN, OBS_TARGET_FORWARD, OBS_TARGET_RIGHT
)

def make_pure_pursuit(
    k_roll: float = 1.0,
    max_bank_deg: float = 35.0,
    kp_roll: float = 2.0,
    kd_roll: float = 0.5,
    kp_pitch: float = 3.0,
    kd_pitch: float = 0.5,
):
    """Proportional-Derivative Pure Pursuit guidance policy for the new C172 2D environment."""
    max_bank = math.radians(max_bank_deg)

    def policy(obs: np.ndarray) -> np.ndarray:
        fwd = obs[OBS_TARGET_FORWARD]
        right = obs[OBS_TARGET_RIGHT]
        down = obs[OBS_TARGET_DOWN]

        # Lateral / Roll Control
        bearing = math.atan2(right, fwd)
        desired_roll = float(np.clip(k_roll * bearing, -max_bank, max_bank))
        aileron = kp_roll * (desired_roll - obs[OBS_ROLL]) - kd_roll * obs[OBS_P]

        # Longitudinal / Pitch Control (JSBSim elevator: negative = nose UP)
        horizontal = math.hypot(fwd, right)
        desired_pitch = math.atan2(-down, horizontal) if horizontal > 1e-3 else 0.0
        elevator = -(kp_pitch * (desired_pitch - obs[OBS_PITCH]) - kd_pitch * obs[OBS_Q])

        return np.clip([aileron, elevator], -1.0, 1.0).astype(np.float32)

    return policy