import numpy as np

import jsbsim_env  # noqa: F401
from jsbsim_env.envs import InterceptEnv
from jsbsim_env.evaluate import evaluate_policy, zero_policy
from jsbsim_env.visualize import animate_paths, plot_paths

# --- Pure Pursuit / PPO (enable when ready) ---
# from jsbsim_env.evaluate import make_pure_pursuit, make_ppo_policy
# (or import Pure Pursuit from your teammate's module, e.g.
#  from jsbsim_env.pure_pursuit import make_pure_pursuit)

POLICIES = {
    "zero": zero_policy,
    # "pure_pursuit": make_pure_pursuit(),
    # "ppo": make_ppo_policy("models/ppo_intercept"),
}
DEMO_POLICY = "zero"  # which policy to show in 3D: "zero" / "pure_pursuit" / "ppo"

EVAL_SEEDS = list(range(20))
DEMO_SEED = 0
DEMO_TARGET_OFFSET = (600.0, 50.0, 0.0)  # forward, right, down, m


def fly(env, policy, seed, options=None):
    """Run one episode, return the aircraft path, the target point and the final info."""
    obs, info = env.reset(seed=seed, options=options)
    plane = [info["aircraft_ned_m"]]
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        plane.append(info["aircraft_ned_m"])
        if terminated or truncated:
            break
    return np.array(plane), info["target_ned_m"], info


if __name__ == "__main__":
    # 1. metrics for every enabled policy over a fixed set of scenarios
    for name, policy in POLICIES.items():
        m = evaluate_policy(policy, EVAL_SEEDS)
        print(
            f"{name:>12}: SR={m['success_rate']:.2f}  TTI={m['tti_mean_s']:.1f}s  "
            f"miss={m['miss_distance_mean_m']:.0f}m  smooth={m['control_smoothness']:.3f}"
        )

    # 2. one episode of the chosen policy for the 3D visualisation
    env = InterceptEnv()
    plane, target, info = fly(
        env, POLICIES[DEMO_POLICY], DEMO_SEED, options={"target_offset_m": DEMO_TARGET_OFFSET}
    )
    dt = env.config.agent_dt_s
    env.close()

    title = f"{DEMO_POLICY}: {info['end_reason']}, distance = {info['distance_m']:.0f} m"

    plot_paths(
        {"plane": plane},
        points={"target": [target]},
        out="flight.html",
        title=title,
        frame="ned",
    )
    animate_paths(
        {"plane": plane, "target": np.tile(target, (len(plane), 1))},
        dt=dt,
        out="flight_live.html",
        title=title,
        frame="ned",
    )