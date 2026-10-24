import numpy as np

import jsbsim_env  # noqa: F401
from jsbsim_env.envs import InterceptEnv
from jsbsim_env.evaluate import zero_policy
from jsbsim_env.plot3d import animate_paths, plot_paths


def fly(env, policy, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    plane = [info["aircraft_ned_m"]]
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        plane.append(info["aircraft_ned_m"])
        if terminated or truncated:
            break
    return np.array(plane), info["target_ned_m"], info


if __name__ == "__main__":
    env = InterceptEnv()
    plane, target, info = fly(
        env, zero_policy, seed=0, options={"target_offset_m": (600.0, 50.0, 0.0)}
    )
    env.close()
    print(info["end_reason"], info["distance_m"])

    title = f"{info['end_reason']}, distance = {info['distance_m']:.0f} m"

    # static 3D view (rotate with the mouse)
    plot_paths(
        {"plane": plane},
        points={"target": [target]},
        out="flight.html",
        title=title,
        frame="ned",
    )

    # animation: the target stays in place, so repeat its point for every sample
    target_path = np.tile(target, (len(plane), 1))
    animate_paths(
        {"plane": plane, "target": target_path},
        dt=env.config.agent_dt_s,
        out="flight_live.html",
        title=title,
        frame="ned",
    )
