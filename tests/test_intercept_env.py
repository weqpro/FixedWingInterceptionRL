import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import jsbsim_env  # noqa: F401  (registers the env)
from jsbsim_env.envs import EndReason, InterceptConfig, InterceptEnv
from jsbsim_env.envs.intercept import (
    ACT_AILERON,
    ACT_DIM,
    ACT_ELEVATOR,
    ACT_RUDDER,
    ACT_THROTTLE,
    FT_TO_M,
    OBS_AGL,
    OBS_AIRSPEED,
    OBS_COS_PSI,
    OBS_DIM,
    OBS_R,
    OBS_SIN_PSI,
    OBS_TARGET_DOWN,
    OBS_TARGET_FORWARD,
    OBS_TARGET_RIGHT,
    _segment_point_distance,
)

ZERO_ACTION = np.zeros(ACT_DIM, dtype=np.float32)


def make_action(**controls):
    """Action with the named controls set, e.g. ``make_action(rudder=0.5)``."""
    index = {
        "aileron": ACT_AILERON,
        "elevator": ACT_ELEVATOR,
        "rudder": ACT_RUDDER,
        "throttle": ACT_THROTTLE,
    }
    action = np.zeros(ACT_DIM, dtype=np.float32)
    for name, value in controls.items():
        action[index[name]] = value
    return action


def run_episode(env, action, **reset_kwargs):
    obs, info = env.reset(**reset_kwargs)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return obs, reward, total, terminated, truncated, info


@pytest.fixture
def env():
    e = InterceptEnv()
    yield e
    e.close()


def test_check_env(env):
    check_env(env, skip_render_check=True)


def test_gym_make():
    e = gym.make("jsbsim_env/Intercept-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    e.close()


def test_same_seed_is_deterministic():
    def trajectory(seed):
        e = InterceptEnv()
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(123)
        out = [obs]
        for _ in range(30):
            obs, reward, *_ = e.step(rng.uniform(-0.3, 0.3, size=ACT_DIM).astype(np.float32))
            out.append(np.append(obs, reward))
        return out

    for a, b in zip(trajectory(7), trajectory(7), strict=True):
        np.testing.assert_array_equal(a, b)


def test_zero_action_holds_trim_until_timeout(env):
    obs, _ = env.reset(seed=0)
    start_agl = obs[OBS_AGL]
    for _ in range(env.config.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs)) and math.isfinite(reward)
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info["end_reason"] is None
    assert not info["is_success"]
    assert abs(obs[OBS_AGL] - start_agl) < 100.0


def test_success_when_target_straight_ahead(env):
    _, _, total, terminated, _, info = run_episode(
        env, ZERO_ACTION, seed=0, options={"target_offset_m": (120.0, 0.0, 0.0)}
    )
    assert terminated
    assert info["end_reason"] is EndReason.SUCCESS
    assert info["is_success"]
    assert info["min_distance_m"] < env.config.success_radius_m
    # distance_m is at the end of the sim step, up to one step (~0.5 m) past
    # the closest point.
    assert info["distance_m"] < env.config.success_radius_m + 1.0
    assert total > env.config.success_bonus


def test_pass_between_sim_steps_counts_as_success(env):
    """A pass 0.98 m from the target, midway between two sim-step positions.

    Both end points are just over 1 m away, so only the segment check sees it.
    """
    obs, info = env.reset(seed=0, options={"target_offset_m": (1e5, 0.0, 0.0)})
    start = info["aircraft_ned_m"]
    # The zero action keeps the trimmed commands, so raw sim steps fly the
    # same path as env.step(ZERO_ACTION).
    path = [start]
    for _ in range(31):
        env._fdm.run()
        path.append(env._aircraft_ned())
    p0, p1 = path[30], path[31]
    along = (p1 - p0) / np.linalg.norm(p1 - p0)
    side = np.cross(along, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    target = 0.5 * (p0 + p1) + 0.98 * side
    assert min(np.linalg.norm(target - p0), np.linalg.norm(target - p1)) > 1.0

    psi = math.radians(env.config.init_heading_deg)
    north, east, down = target - start
    offset = (
        math.cos(psi) * north + math.sin(psi) * east,
        -math.sin(psi) * north + math.cos(psi) * east,
        down,
    )
    _, _, _, terminated, _, info = run_episode(
        env, ZERO_ACTION, seed=0, options={"target_offset_m": offset}
    )
    assert terminated and info["end_reason"] is EndReason.SUCCESS
    assert info["min_distance_m"] == pytest.approx(0.98, abs=0.01)


def test_min_distance_is_closest_approach(env):
    _, info = env.reset(seed=0)
    distances = [info["distance_m"]]
    while True:
        _, _, terminated, truncated, info = env.step(make_action(aileron=0.05))
        distances.append(info["distance_m"])
        if terminated or truncated:
            break
    assert info["min_distance_m"] <= min(distances)
    assert info["min_distance_m"] > min(distances) - 1.0  # within one sim step


@pytest.mark.parametrize(
    "p0, p1, q, expected",
    [
        ((0, 0, 0), (2, 0, 0), (1, 1, 0), 1.0),
        ((0, 0, 0), (2, 0, 0), (-3, 4, 0), 5.0),
        ((0, 0, 0), (2, 0, 0), (5, 0, 4), 5.0),
        ((1, 1, 1), (1, 1, 1), (1, 1, 3), 2.0),
    ],
)
def test_segment_point_distance(p0, p1, q, expected):
    p0, p1, q = (np.array(v, dtype=float) for v in (p0, p1, q))
    assert _segment_point_distance(p0, p1, q) == pytest.approx(expected)


def test_crash_on_full_nose_down():
    env = InterceptEnv(InterceptConfig(init_altitude_ft=500.0, target_down_m=(-50.0, 50.0)))
    _, last_reward, _, terminated, _, info = run_episode(env, make_action(elevator=1.0), seed=0)
    assert terminated
    assert info["end_reason"] is EndReason.CRASH
    assert not info["is_success"]
    assert last_reward < -env.config.failure_penalty + 1.0


def test_target_offset_in_heading_frame():
    env = InterceptEnv(InterceptConfig(init_heading_deg=210.0))
    obs, _ = env.reset(seed=0, options={"target_offset_m": (1000.0, -300.0, 50.0)})
    np.testing.assert_allclose(
        obs[[OBS_TARGET_FORWARD, OBS_TARGET_RIGHT, OBS_TARGET_DOWN]],
        [1000.0, -300.0, 50.0],
        atol=1.0,
    )


def test_heading_crossing_north_is_continuous():
    env = InterceptEnv(InterceptConfig(init_heading_deg=355.0))
    env.reset(seed=0)
    prev = None
    for _ in range(50):
        obs, *_ = env.step(make_action(aileron=0.3))
        if prev is not None:
            step = obs[[OBS_SIN_PSI, OBS_COS_PSI]] - prev[[OBS_SIN_PSI, OBS_COS_PSI]]
            assert np.hypot(*step) < 0.2
        prev = obs


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(ZERO_ACTION)


def test_step_after_episode_end_raises(env):
    run_episode(env, ZERO_ACTION, seed=0, options={"target_offset_m": (120.0, 0.0, 0.0)})
    with pytest.raises(RuntimeError):
        env.step(ZERO_ACTION)


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(ACT_DIM - 1),
        np.zeros(ACT_DIM + 1),
        make_action(aileron=np.nan),
        make_action(throttle=np.inf),
    ],
)
def test_invalid_action_raises(env, action):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_out_of_range_action_is_clipped(env):
    env.reset(seed=0)
    obs, reward, *_ = env.step(np.array([5.0, -5.0, 5.0, -5.0]))
    assert np.all(np.isfinite(obs)) and math.isfinite(reward)


def test_zero_action_keeps_rudder_and_throttle_at_trim(env):
    env.reset(seed=0)
    trim_rudder = env._fdm["fcs/rudder-cmd-norm"]
    trim_throttle = env._fdm["fcs/throttle-cmd-norm"]
    assert 0.0 < trim_throttle < 1.0
    env.step(ZERO_ACTION)
    assert env._fdm["fcs/rudder-cmd-norm"] == pytest.approx(trim_rudder)
    assert env._fdm["fcs/throttle-cmd-norm"] == pytest.approx(trim_throttle)


@pytest.mark.parametrize("throttle, expected", [(-1.0, 0.0), (1.0, 1.0), (5.0, 1.0)])
def test_throttle_extremes(env, throttle, expected):
    env.reset(seed=0)
    env.step(make_action(throttle=throttle))
    assert env._fdm["fcs/throttle-cmd-norm"] == pytest.approx(expected)


def airspeed_after(action, steps=50):
    env = InterceptEnv()
    env.reset(seed=0, options={"target_offset_m": (5000.0, 0.0, 0.0)})
    for _ in range(steps):
        obs, *_ = env.step(action)
    env.close()
    return obs[OBS_AIRSPEED]


def test_throttle_changes_airspeed():
    trimmed = airspeed_after(ZERO_ACTION)
    assert airspeed_after(make_action(throttle=1.0)) > trimmed + 1.0
    assert airspeed_after(make_action(throttle=-1.0)) < trimmed - 1.0


def test_rudder_yaws_aircraft(env):
    # JSBSim convention: positive rudder yaws the nose left.
    env.reset(seed=0)
    for _ in range(5):
        pos, *_ = env.step(make_action(rudder=0.5))
    env.reset(seed=0)
    for _ in range(5):
        neg, *_ = env.step(make_action(rudder=-0.5))
    assert pos[OBS_R] < -0.01 and neg[OBS_R] > 0.01


def test_flat_planet_is_loaded(env):
    env.reset(seed=0)
    assert env._fdm["inertial/sea-level-radius_ft"] * FT_TO_M == pytest.approx(1e10)
    g = env._fdm["accelerations/gravity-ft_sec2"] * FT_TO_M
    assert g == pytest.approx(9.80665, rel=1e-6)


def test_frame_matches_integrated_ned_velocity(env):
    """Position from the frame equals the integral of JSBSim's NED velocity."""
    _, info = env.reset(seed=0, options={"target_offset_m": (1e5, 0.0, 0.0)})
    fdm = env._fdm
    dt = 1.0 / env.config.sim_hz

    def ned_velocity():
        axes = ("north", "east", "down")
        return np.array([fdm[f"velocities/v-{axis}-fps"] for axis in axes]) * FT_TO_M

    integrated = info["aircraft_ned_m"].copy()
    displacement = np.zeros((2, 3))  # min, max
    max_error = 0.0
    for k in range(300):
        # Stepped by hand to integrate every sim step.
        fdm["fcs/aileron-cmd-norm"] = env._trim_aileron + (0.05 if k < 150 else -0.05)
        fdm["fcs/elevator-cmd-norm"] = -0.05
        fdm["fcs/throttle-cmd-norm"] = env._throttle_cmd(0.5 if k % 100 < 50 else -0.5)
        for _ in range(env.config.sim_steps_per_action):
            v0 = ned_velocity()
            assert fdm.run()
            integrated += 0.5 * (v0 + ned_velocity()) * dt
            max_error = max(max_error, np.abs(integrated - env._aircraft_ned()).max())
        d = env._aircraft_ned() - info["aircraft_ned_m"]
        displacement = np.array([np.minimum(displacement[0], d), np.maximum(displacement[1], d)])
        # Ground contact makes velocity jump within a sim step, which the
        # trapezoid can't follow, so the manoeuvre must stay airborne.
        assert fdm["position/h-agl-ft"] * FT_TO_M > 500.0
    assert np.all(displacement[1] - displacement[0] > 100.0)
    assert max_error < 1e-3


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 210.0])
def test_flight_is_independent_of_heading(heading_deg):
    """No Earth rotation or curvature: the same flight in any direction."""

    def displacement_in_heading_frame(heading):
        env = InterceptEnv(InterceptConfig(init_heading_deg=heading))
        _, info = env.reset(seed=0, options={"target_offset_m": (1e5, 0.0, 0.0)})
        start = info["aircraft_ned_m"]
        for _ in range(300):
            _, _, _, _, info = env.step(make_action(aileron=0.1, rudder=0.05))
        env.close()
        north, east, down = info["aircraft_ned_m"] - start
        psi = math.radians(heading)
        return np.array(
            [
                math.cos(psi) * north + math.sin(psi) * east,
                -math.sin(psi) * north + math.cos(psi) * east,
                down,
            ]
        )

    np.testing.assert_allclose(
        displacement_in_heading_frame(heading_deg),
        displacement_in_heading_frame(45.0),
        atol=0.01,
    )


def test_down_is_minus_height_above_ground(env):
    env.reset(seed=0)
    for _ in range(20):
        _, _, _, _, info = env.step(make_action(elevator=0.2))
    assert info["aircraft_ned_m"][2] == pytest.approx(-env._fdm["position/h-agl-ft"] * FT_TO_M)


@pytest.mark.parametrize(
    "options", [{"target_offset_m": (1.0, 2.0)}, {"target_offset_m": (np.nan, 0, 0)}]
)
def test_invalid_target_offset_raises(env, options):
    with pytest.raises(ValueError):
        env.reset(seed=0, options=options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sim_steps_per_action": 0},
        {"max_episode_steps": 1.5},
        {"success_radius_m": -1.0},
        {"target_forward_m": (2000.0, 800.0)},
        {"init_altitude_ft": 10.0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        InterceptConfig(**kwargs)


def test_unsupported_render_mode_raises():
    with pytest.raises(ValueError):
        InterceptEnv(render_mode="ansi")


def test_rgb_array_render():
    env = InterceptEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(ZERO_ACTION)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    env.close()
