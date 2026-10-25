"""Policy evaluation on a fixed set of scenarios: success rate, TTI, miss distance, smoothness."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

import jsbsim_env
from jsbsim_env.envs.intercept import ACT_DIM, InterceptConfig, InterceptEnv
from jsbsim_env.pure_pursuit import make_pure_pursuit
from jsbsim_env.ppo_policy import make_ppo_policy

Policy = Callable[[np.ndarray], np.ndarray]


def run_episode(env: InterceptEnv, policy: Policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    min_distance = info["distance_m"]
    prev_action = None
    deltas = []

    while True:
        action = np.clip(np.asarray(policy(obs), dtype=np.float64), -1.0, 1.0)
        if prev_action is not None:
            deltas.append(np.abs(action - prev_action).mean())
        prev_action = action

        obs, _, terminated, truncated, info = env.step(action)
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


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros(ACT_DIM, dtype=np.float32)


def make_random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)


if __name__ == "__main__":
    seeds = list(range(20))
    policies: dict[str, Policy] = {
        "zero": zero_policy,
        "random": make_random_policy(),
        "pure_pursuit": make_pure_pursuit(),
    }
    
    ppo_model_path = Path("models/ppo_intercept.zip")
    if ppo_model_path.exists():
        policies["ppo"] = make_ppo_policy(ppo_model_path)

    for name, policy in policies.items():
        m = evaluate_policy(policy, seeds)
        print(
            f"{name:>12}: SR={m['success_rate']:.2f}  TTI={m['tti_mean_s']:.1f}s  "
            f"miss={m['miss_distance_mean_m']:.0f}m  smooth={m['control_smoothness']:.3f}"
        )