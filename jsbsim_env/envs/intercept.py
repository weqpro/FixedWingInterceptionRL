"""Fixed-wing interception of a stationary target, simulated with JSBSim.

The agent controls aileron and elevator as deltas around the level-flight trim
and must fly the aircraft into a sphere around the target before running out of
time or hitting the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import gymnasium as gym
import jsbsim
import numpy as np
from gymnasium import spaces

FT_TO_M = 0.3048

# WGS84 ellipsoid
_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


@dataclass(frozen=True)
class InterceptConfig:
    """All tunables of :class:`InterceptEnv`."""

    # Aircraft and initial conditions
    aircraft: str = "c172x"
    origin_lat_deg: float = 49.8
    origin_lon_deg: float = 24.0
    init_altitude_ft: float = 3000.0  # above sea level
    init_airspeed_kts: float = 100.0  # calibrated
    init_heading_deg: float = 90.0

    # Timing
    sim_hz: float = 120.0
    sim_steps_per_action: int = 12
    max_episode_steps: int = 600

    # Termination
    success_radius_m: float = 50.0
    min_agl_m: float = 15.0

    # Target spawn, relative to the initial position and heading
    target_forward_m: tuple[float, float] = (800.0, 2000.0)
    target_right_m: tuple[float, float] = (-500.0, 500.0)
    target_down_m: tuple[float, float] = (-100.0, 100.0)

    # Reward
    progress_scale: float = 0.01  # reward per metre of distance closed
    success_bonus: float = 10.0
    failure_penalty: float = 10.0

    def __post_init__(self) -> None:
        if not -90.0 < self.origin_lat_deg < 90.0:
            raise ValueError(f"origin_lat_deg must be in (-90, 90), got {self.origin_lat_deg}")
        if not -180.0 <= self.origin_lon_deg <= 180.0:
            raise ValueError(f"origin_lon_deg must be in [-180, 180], got {self.origin_lon_deg}")
        for name in (
            "init_airspeed_kts",
            "sim_hz",
            "success_radius_m",
            "min_agl_m",
            "progress_scale",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("success_bonus", "failure_penalty"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        for name in ("sim_steps_per_action", "max_episode_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in ("target_forward_m", "target_right_m", "target_down_m"):
            low, high = getattr(self, name)
            if not low <= high:
                raise ValueError(f"{name} must be (low, high) with low <= high, got {(low, high)}")
        if self.init_altitude_ft * FT_TO_M <= self.min_agl_m:
            raise ValueError("init_altitude_ft must be above min_agl_m")
        if self.init_altitude_ft * FT_TO_M - self.target_down_m[1] <= self.min_agl_m:
            raise ValueError("target_down_m allows targets below min_agl_m")

    @property
    def agent_dt_s(self) -> float:
        return self.sim_steps_per_action / self.sim_hz


class EndReason(StrEnum):  # NOTE: check if needed
    SUCCESS = "success"
    CRASH = "crash"
    NUMERICAL_ERROR = "numerical_error"
    SIM_FAILURE = "sim_failure"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Observation layout (float32, SI units)
#
#   0  roll             rad
#   1  pitch            rad
#   2  sin(heading)
#   3  cos(heading)     heading as sin/cos so there is no 2pi -> 0 jump
#   4  p                rad/s  roll rate
#   5  q                rad/s  pitch rate
#   6  r                rad/s  yaw rate
#   7  true airspeed    m/s
#   8  height AGL       m
#   9  vertical speed   m/s, + = climbing
#  10  target forward   m  \
#  11  target right     m   } target offset in the heading frame (yaw-only
#  12  target down      m  /  rotation of NED), independent of world heading
# ---------------------------------------------------------------------------
OBS_ROLL = 0
OBS_PITCH = 1
OBS_SIN_PSI = 2
OBS_COS_PSI = 3
OBS_P = 4
OBS_Q = 5
OBS_R = 6
OBS_AIRSPEED = 7
OBS_AGL = 8
OBS_VSPEED = 9
OBS_TARGET_FORWARD = 10
OBS_TARGET_RIGHT = 11
OBS_TARGET_DOWN = 12
OBS_DIM = 13

ACT_AILERON = 0
ACT_ELEVATOR = 1
ACT_DIM = 2


class InterceptEnv(gym.Env[np.ndarray, np.ndarray]):
    """Fly a fixed-wing aircraft into a sphere around a stationary target.

    Action: ``[aileron, elevator]`` in [-1, 1], added to the level-flight trim.
    Rudder holds its trimmed value; throttle and pitch trim are left as set by
    the trim routine. Out-of-range actions are clipped.

    Reward: ``progress_scale * (distance closed this step)``, plus
    ``success_bonus`` on reaching the target or minus ``failure_penalty`` on
    crash, numerical error or simulator failure.

    Info: ``distance_m``, ``end_reason`` (:class:`EndReason` or None) and
    ``sim_time_s``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, config: InterceptConfig | None = None, render_mode: str | None = None
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.config = config if config is not None else InterceptConfig()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

        cfg = self.config
        lat0 = math.radians(cfg.origin_lat_deg)
        denom = 1.0 - _WGS84_E2 * math.sin(lat0) ** 2
        # Metres per radian of latitude (meridional radius) and of longitude
        # (prime-vertical radius times cos(lat)) on the tangent plane at the origin.
        self._m_per_rad_lat = _WGS84_A * (1.0 - _WGS84_E2) / denom**1.5
        self._m_per_rad_lon = _WGS84_A / math.sqrt(denom) * math.cos(lat0)

        self._fdm: jsbsim.FGFDMExec | None = None
        self._trim_aileron = 0.0
        self._trim_rudder = 0.0
        self._target_ned = np.zeros(3)
        self._prev_distance = 0.0
        self._steps = 0
        self._episode_over = False

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        ``options["target_offset_m"]``: optional ``(forward, right, down)`` of
        the target relative to the initial position and heading, in metres.
        When omitted, the offset is sampled from the config ranges.
        """
        super().reset(seed=seed)
        cfg = self.config

        self._fdm = self._new_fdm()
        start = self._aircraft_ned()

        if options is not None and "target_offset_m" in options:
            offset = np.asarray(options["target_offset_m"], dtype=np.float64)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError(
                    f"target_offset_m must be 3 finite numbers, got {options['target_offset_m']!r}"
                )
        else:
            offset = np.array(
                [
                    self.np_random.uniform(*cfg.target_forward_m),
                    self.np_random.uniform(*cfg.target_right_m),
                    self.np_random.uniform(*cfg.target_down_m),
                ]
            )
        psi0 = math.radians(cfg.init_heading_deg)
        forward, right, down = offset
        self._target_ned = start + np.array(
            [
                forward * math.cos(psi0) - right * math.sin(psi0),
                forward * math.sin(psi0) + right * math.cos(psi0),
                down,
            ]
        )

        self._steps = 0
        self._episode_over = False
        self._prev_distance = self._distance()
        return self._get_obs(), self._info(self._prev_distance, None)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._fdm is None:
            raise RuntimeError("Call reset() before step()")
        if self._episode_over:
            raise RuntimeError("Episode is over; call reset() before step()")

        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACT_DIM,):
            raise ValueError(f"Action must have shape ({ACT_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Action must be finite, got {action}")
        action = np.clip(action, -1.0, 1.0)

        cfg = self.config
        fdm = self._fdm
        fdm["fcs/aileron-cmd-norm"] = float(
            np.clip(self._trim_aileron + action[ACT_AILERON], -1.0, 1.0)
        )
        # Elevator trim lives in fcs/pitch-trim-cmd-norm, which the FCS adds on
        # top of this command, so the zero action is exactly the trimmed state.
        fdm["fcs/elevator-cmd-norm"] = float(action[ACT_ELEVATOR])
        fdm["fcs/rudder-cmd-norm"] = self._trim_rudder

        reason: EndReason | None = None
        distance = self._prev_distance
        for _ in range(cfg.sim_steps_per_action):
            if not fdm.run():
                reason = EndReason.SIM_FAILURE
                break
            distance = self._distance()
            if not math.isfinite(distance):
                reason = EndReason.NUMERICAL_ERROR
                break
            # Checked every sim step so a fast pass through the sphere counts.
            if distance < cfg.success_radius_m:
                reason = EndReason.SUCCESS
                break
            if fdm["position/h-agl-ft"] * FT_TO_M < cfg.min_agl_m:
                reason = EndReason.CRASH
                break

        obs = self._get_obs()
        if reason is None and not np.all(np.isfinite(obs)):
            reason = EndReason.NUMERICAL_ERROR

        if reason in (EndReason.NUMERICAL_ERROR, EndReason.SIM_FAILURE):
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            distance = self._prev_distance
            reward = -cfg.failure_penalty
        else:
            reward = cfg.progress_scale * (self._prev_distance - distance)
            if reason is EndReason.SUCCESS:
                reward += cfg.success_bonus
            elif reason is EndReason.CRASH:
                reward -= cfg.failure_penalty
        self._prev_distance = distance

        self._steps += 1
        terminated = reason is not None
        truncated = not terminated and self._steps >= cfg.max_episode_steps
        if truncated:
            reason = EndReason.TIMEOUT
        self._episode_over = terminated or truncated

        return obs, float(reward), terminated, truncated, self._info(distance, reason)

    def close(self) -> None:
        self._fdm = None
        super().close()

    # ------------------------------------------------------------ internals

    def _new_fdm(self) -> jsbsim.FGFDMExec:
        """Create, initialise and trim a fresh JSBSim instance.

        A fresh instance per episode avoids state leaking between episodes
        (engine, actuators, trim), so resets are deterministic.
        """
        cfg = self.config
        jsbsim.FGJSBBase().debug_lvl = 0  # silence the per-instance banner
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        if not fdm.load_model(cfg.aircraft):
            raise RuntimeError(f"Could not load JSBSim model {cfg.aircraft!r}")
        fdm.disable_output()  # after load_model: the model file re-enables output
        fdm.set_dt(1.0 / cfg.sim_hz)

        fdm["ic/lat-geod-deg"] = cfg.origin_lat_deg
        fdm["ic/long-gc-deg"] = cfg.origin_lon_deg
        fdm["ic/h-sl-ft"] = cfg.init_altitude_ft
        fdm["ic/vc-kts"] = cfg.init_airspeed_kts
        fdm["ic/psi-true-deg"] = cfg.init_heading_deg
        if not fdm.run_ic():
            raise RuntimeError("JSBSim failed to apply initial conditions")
        fdm["propulsion/set-running"] = -1  # start all engines
        try:
            fdm["simulation/do_simple_trim"] = 1  # full trim for level flight
        except jsbsim.TrimFailureError as e:
            raise RuntimeError(
                f"JSBSim could not trim {cfg.aircraft!r} at "
                f"{cfg.init_airspeed_kts} kts, {cfg.init_altitude_ft} ft"
            ) from e

        self._trim_aileron = fdm["fcs/aileron-cmd-norm"]
        self._trim_rudder = fdm["fcs/rudder-cmd-norm"]
        return fdm

    def _aircraft_ned(self) -> np.ndarray:
        """Aircraft position in metres on the tangent plane at the config origin.

        JSBSim's distance-from-start properties are unsigned, so lat/lon are
        converted here instead.
        """
        assert self._fdm is not None
        cfg = self.config
        dlat = self._fdm["position/lat-geod-deg"] - cfg.origin_lat_deg
        dlon = (self._fdm["position/long-gc-deg"] - cfg.origin_lon_deg + 180.0) % 360.0 - 180.0
        return np.array(
            [
                math.radians(dlat) * self._m_per_rad_lat,
                math.radians(dlon) * self._m_per_rad_lon,
                -self._fdm["position/h-sl-meters"],
            ]
        )

    def _distance(self) -> float:
        return float(np.linalg.norm(self._target_ned - self._aircraft_ned()))

    def _get_obs(self) -> np.ndarray:
        assert self._fdm is not None
        fdm = self._fdm
        psi = fdm["attitude/psi-rad"]
        sin_psi, cos_psi = math.sin(psi), math.cos(psi)
        north, east, down = self._target_ned - self._aircraft_ned()

        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[OBS_ROLL] = fdm["attitude/phi-rad"]
        obs[OBS_PITCH] = fdm["attitude/theta-rad"]
        obs[OBS_SIN_PSI] = sin_psi
        obs[OBS_COS_PSI] = cos_psi
        obs[OBS_P] = fdm["velocities/p-rad_sec"]
        obs[OBS_Q] = fdm["velocities/q-rad_sec"]
        obs[OBS_R] = fdm["velocities/r-rad_sec"]
        obs[OBS_AIRSPEED] = fdm["velocities/vt-fps"] * FT_TO_M
        obs[OBS_AGL] = fdm["position/h-agl-ft"] * FT_TO_M
        obs[OBS_VSPEED] = fdm["velocities/h-dot-fps"] * FT_TO_M
        obs[OBS_TARGET_FORWARD] = cos_psi * north + sin_psi * east
        obs[OBS_TARGET_RIGHT] = -sin_psi * north + cos_psi * east
        obs[OBS_TARGET_DOWN] = down
        return obs

    def _info(self, distance: float, reason: EndReason | None) -> dict[str, Any]:
        assert self._fdm is not None
        return {
            "distance_m": distance,
            "end_reason": reason,
            "sim_time_s": self._fdm["simulation/sim-time-sec"],
        }
