from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np

@dataclass(frozen=True)
class GeoPos:
    """Geodetic position"""
    lat_deg: float
    lon_deg: float
    alt_m: float  # above mean sea level


@dataclass(frozen=True, slots=True)
class NedVec:
    """3-vector in the local NED frame. Unit is given by the holding attribute."""
    north: float
    east: float
    down: float
 
    @property
    def up(self) -> float:
        """Upward component (height for a position, climb rate for a velocity)."""
        return -self.down
 
    def norm(self) -> float:
        return math.sqrt(self.north * self.north + self.east * self.east + self.down * self.down)
 
    def as_array(self) -> np.ndarray:
        return np.array((self.north, self.east, self.down), dtype=np.float64)
 
    @classmethod
    def from_array(cls, v: np.ndarray) -> NedVec:
        n, e, d = v
        return cls(float(n), float(e), float(d))
 
    @classmethod
    def zero(cls) -> NedVec:
        return cls(0.0, 0.0, 0.0)
 
    def __add__(self, o: NedVec) -> NedVec:
        return NedVec(self.north + o.north, self.east + o.east, self.down + o.down)
 
    def __sub__(self, o: NedVec) -> NedVec:
        return NedVec(self.north - o.north, self.east - o.east, self.down - o.down)
 
    def __neg__(self) -> NedVec:
        return NedVec(-self.north, -self.east, -self.down)
 
    def __mul__(self, k: float) -> NedVec:
        return NedVec(self.north * k, self.east * k, self.down * k)
 
    __rmul__ = __mul__

@dataclass(frozen=True, slots=True)
class BodyRates:
    """Angular rates about the body axes [rad/s]: p (roll), q (pitch), r (yaw)."""
    p: float
    q: float
    r: float
 
    def as_array(self) -> np.ndarray:
        return np.array((self.p, self.q, self.r), dtype=np.float64)
 
    @classmethod
    def from_array(cls, v: np.ndarray) -> BodyRates:
        p, q, r = v
        return cls(float(p), float(q), float(r))

@dataclass(frozen=True)
class Attitude:
    """Euler angles of the body relative to NED"""
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    @classmethod
    def from_deg(cls, roll: float, pitch: float, yaw: float) -> Attitude:
        return cls(
            roll * math.pi / 180.0,
            pitch * math.pi / 180.0,
            yaw * math.pi / 180.0
        )

    @classmethod
    def from_array(cls, a: np.ndarray) -> Attitude:
        r, p, y = a
        return cls(float(r), float(p), float(y))

    @classmethod
    def level(cls, heading_deg: float = 0.0) -> Attitude:
        return cls(0.0, 0.0, math.radians(heading_deg))

    def as_array(self) -> np.ndarray:
        return np.array((self.roll_rad, self.pitch_rad, self.yaw_rad), dtype=np.float64)

    def to_dcm(self) -> np.ndarray:
        """
        Body-to-NED rotation matrix C_nb = Rz(yaw) @ Ry(pitch) @ Rx(roll).
 
        Use as ``v_ned = C_nb @ v_body``; the inverse is ``C_nb.T``.
        """

        sr, cr = math.sin(self.roll_rad), math.cos(self.roll_rad)
        sp, cp = math.sin(self.pitch_rad), math.cos(self.pitch_rad)
        sy, cy = math.sin(self.yaw_rad), math.cos(self.yaw_rad)
        sp_cy = sp * cy
        sp_sy = sp * sy
        return np.array(
            (
                (cp * cy, sr * sp_cy - cr * sy, cr * sp_cy + sr * sy),
                (cp * sy, sr * sp_sy + cr * cy, cr * sp_sy - sr * cy),
                (-sp, sr * cp, cr * cp),
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class InitialConditions:
    pos: GeoPos
    airspeed_mps: float
    attitude: Attitude = field(default_factory=lambda: Attitude(0.0, 0.0, 0.0))
    trim: bool = True


@dataclass(frozen=True, slots=True)
class Controls:
    """Control inputs
 
    aileron, elevator, rudder are normalized to [-1, 1]; throttle to [0, 1].
    """
    aileron: float
    elevator: float
    rudder: float
    throttle: float
 
    def as_array(self) -> np.ndarray:
        return np.array((self.aileron, self.elevator, self.rudder, self.throttle), dtype=np.float64)
 
    @classmethod
    def from_array(cls, a: np.ndarray, clip: bool = True) -> Controls:
        """Build from [aileron, elevator, rudder, throttle]; clips to valid ranges by default.
 
        Use this for untrusted input such as policy outputs.
        """
        ail, ele, rud, thr = (float(x) for x in a)
        if clip:
            ail = min(1.0, max(-1.0, ail))
            ele = min(1.0, max(-1.0, ele))
            rud = min(1.0, max(-1.0, rud))
            thr = min(1.0, max(0.0, thr))
        return cls(ail, ele, rud, thr)


@dataclass(frozen=True, slots=True)
class AircraftState:
    """Snapshot of the aircraft at time t_s. Safe to use as policy observation input."""
    t_s: float
    geo: GeoPos  # for FlightGear
    pos_m: NedVec  # relative to the LocalFrame origin
    vel_mps: NedVec  # ground-relative
    attitude: Attitude
    body_rates: BodyRates
    alpha_rad: float
    beta_rad: float
    airspeed_mps: float  # air-relative
    controls: Controls  # actural surface positions and throttle


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """Conditions at the aircraft's position at time t_s.
 
    PRIVILEGED: for recording, residual learning and evaluation only.
    Never pass this to an observation builder.
    """
    t_s: float
    wind_mps: NedVec  # total wind incl. turbulence, NED


@dataclass(frozen=True, slots=True)
class TargetState:
    t_s: float
    pos_m: NedVec
    vel_mps: NedVec
