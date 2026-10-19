import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import jsbsim_env  # noqa: F401  (registers the env)
from jsbsim_env.envs import EndReason, InterceptConfig, InterceptEnv
from jsbsim_env.envs.intercept import (
    OBS_DIM,
    OBS_TARGET_DOWN,
    OBS_TARGET_FORWARD,
    OBS_TARGET_RIGHT,
)

ZERO_ACTION = np.zeros(2, dtype=np.float32)


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
            obs, reward, *_ = e.step(rng.uniform(-0.3, 0.3, size=2).astype(np.float32))
            out.append(np.append(obs, reward))
        return out

    for a, b in zip(trajectory(7), trajectory(7), strict=True):
        np.testing.assert_array_equal(a, b)


def test_zero_action_holds_trim_until_timeout(env):
    obs, _ = env.reset(seed=0)
    start_agl = obs[8]
    for _ in range(env.config.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs)) and math.isfinite(reward)
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info["end_reason"] is None
    assert not info["is_success"]
    assert abs(obs[8] - start_agl) < 100.0  # roughly level flight


def test_success_when_target_straight_ahead(env):
    _, _, total, terminated, _, info = run_episode(
        env, ZERO_ACTION, seed=0, options={"target_offset_m": (120.0, 0.0, 0.0)}
    )
    assert terminated
    assert info["end_reason"] is EndReason.SUCCESS
    assert info["is_success"]
    assert info["distance_m"] < env.config.success_radius_m
    assert total > env.config.success_bonus


def test_crash_on_full_nose_down():
    env = InterceptEnv(InterceptConfig(init_altitude_ft=500.0, target_down_m=(-50.0, 50.0)))
    _, last_reward, _, terminated, _, info = run_episode(
        env, np.array([0.0, -1.0], dtype=np.float32), seed=0
    )
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
        obs, *_ = env.step(np.array([0.3, 0.0], dtype=np.float32))
        if prev is not None:
            assert np.hypot(obs[2] - prev[2], obs[3] - prev[3]) < 0.2
        prev = obs


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(ZERO_ACTION)


def test_step_after_episode_end_raises(env):
    run_episode(env, ZERO_ACTION, seed=0, options={"target_offset_m": (120.0, 0.0, 0.0)})
    with pytest.raises(RuntimeError):
        env.step(ZERO_ACTION)


@pytest.mark.parametrize("action", [np.zeros(3), np.array([np.nan, 0.0]), np.array([0.0, np.inf])])
def test_invalid_action_raises(env, action):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_out_of_range_action_is_clipped(env):
    env.reset(seed=0)
    obs, reward, *_ = env.step(np.array([5.0, -5.0]))
    assert np.all(np.isfinite(obs)) and math.isfinite(reward)


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
        {"origin_lat_deg": 95.0},
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
