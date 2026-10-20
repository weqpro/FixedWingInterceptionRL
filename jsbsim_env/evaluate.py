"""Policy evaluation on a fixed set of scenarios: success rate, TTI, miss distance, smoothness."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

import jsbsim_env  # noqa: F401  (registers the env)
from jsbsim_env.envs import InterceptConfig, InterceptEnv

# from jsbsim_env.envs.intercept import (
#     OBS_PITCH, OBS_P, OBS_Q, OBS_ROLL,
#     OBS_TARGET_DOWN, OBS_TARGET_FORWARD, OBS_TARGET_RIGHT,
# )

Policy = Callable[[np.ndarray], np.ndarray]


def run_episode(env: InterceptEnv, policy: Policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    min_distance = info["distance_m"]
    prev_action = None
    deltas = []

    while True:
        action = np.clip(np.asarray(policy(obs), dtype=np.float64), -1.0, 1.0)
        if prev_action is not None:
            deltas.append(np.abs(action - prev_action).mean())  # |a_t - a_{t-1}|
        prev_action = action

        obs, _, terminated, truncated, info = env.step(action)
        # use the env's own minimum if it provides one, otherwise track per agent step
        min_distance = min(min_distance, info.get("min_distance_m", info["distance_m"]))
        if terminated or truncated:
            break

    return {
        "success": bool(info["is_success"]),
        "tti_s": info["sim_time_s"] if info["is_success"] else math.nan,
        "min_distance_m": min_distance,
        "smoothness": float(np.mean(deltas)) if deltas else 0.0,
        "end_reason": info["end_reason"],
    }


def evaluate_policy(
    policy: Policy,
    seeds: Sequence[int],
    config: InterceptConfig | None = None,
) -> dict:
    """Run one episode per seed. The same seeds give the same scenarios for every policy."""
    env = InterceptEnv(config)
    try:
        episodes = [run_episode(env, policy, s) for s in seeds]
    finally:
        env.close()

    success = np.array([e["success"] for e in episodes])
    tti = np.array([e["tti_s"] for e in episodes])
    min_dist = np.array([e["min_distance_m"] for e in episodes])
    smooth = np.array([e["smoothness"] for e in episodes])

    return {
        "success_rate": float(success.mean()),
        "tti_mean_s": float(np.nanmean(tti)) if success.any() else math.nan,
        "miss_distance_mean_m": float(min_dist[~success].mean()) if (~success).any() else math.nan,
        "control_smoothness": float(smooth.mean()),
        "episodes": episodes,
    }


# ---------------------------------------------------------------- policies


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def make_random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


# --- Pure Pursuit (to be enabled later; gains are untuned, and the signs of
#     aileron/elevator must be checked in the simulator) -------------------
#
# def make_pure_pursuit(k_roll=1.0, max_bank=math.radians(35), kp_roll=2.0, kd_roll=0.5,
#                       kp_pitch=3.0, kd_pitch=0.5) -> Policy:
#     def policy(obs: np.ndarray) -> np.ndarray:
#         fwd, right, down = obs[OBS_TARGET_FORWARD], obs[OBS_TARGET_RIGHT], obs[OBS_TARGET_DOWN]
#         bearing = math.atan2(right, fwd)                       # + = target on the right
#         desired_roll = float(np.clip(k_roll * bearing, -max_bank, max_bank))
#         aileron = kp_roll * (desired_roll - obs[OBS_ROLL]) - kd_roll * obs[OBS_P]
#
#         horizontal = math.hypot(fwd, right)
#         desired_pitch = math.atan2(-down, horizontal)          # + = target above
#         elevator = kp_pitch * (desired_pitch - obs[OBS_PITCH]) - kd_pitch * obs[OBS_Q]
#         return np.clip([aileron, elevator], -1.0, 1.0).astype(np.float32)
#     return policy


# --- PPO (Stable-Baselines3, after training) ------------------------------
#
# from stable_baselines3 import PPO
#
# def make_ppo_policy(path: str) -> Policy:
#     model = PPO.load(path)
#     return lambda obs: model.predict(obs, deterministic=True)[0]


if __name__ == "__main__":
    seeds = list(range(20))
    policies: dict[str, Policy] = {
        "zero": zero_policy,
        "random": make_random_policy(),
        # "pure_pursuit": make_pure_pursuit(),
        # "ppo": make_ppo_policy("models/ppo_intercept"),
    }
    for name, policy in policies.items():
        m = evaluate_policy(policy, seeds)
        print(
            f"{name:>12}: SR={m['success_rate']:.2f}  TTI={m['tti_mean_s']:.1f}s  "
            f"miss={m['miss_distance_mean_m']:.0f}m  smooth={m['control_smoothness']:.3f}"
        )
