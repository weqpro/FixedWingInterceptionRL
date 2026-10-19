"""2D renderer for InterceptEnv: top-down view and altitude profile."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle


class InterceptRenderer:
    def __init__(self, render_mode: str, success_radius_m: float, fps: int) -> None:
        self.render_mode = render_mode
        self.success_radius_m = success_radius_m
        self.fps = fps
        self._plt = None

        if render_mode == "human":
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            self.fig = plt.figure(figsize=(11, 5), layout="constrained")
        else:  # rgb_array: offscreen, no window
            self.fig = Figure(figsize=(11, 5), layout="constrained")
            FigureCanvasAgg(self.fig)
        self.ax_top, self.ax_alt = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
        self.reset()

    def reset(self) -> None:
        self._t: list[float] = []
        self._east: list[float] = []
        self._north: list[float] = []
        self._alt: list[float] = []
        self._heading_rad = 0.0
        self._target_ned = np.zeros(3)
        self._distance_m = 0.0
        self._reason: str | None = None

    def record(self, t, ned, heading_rad, target_ned, distance_m, reason) -> None:
        self._t.append(float(t))
        self._north.append(float(ned[0]))
        self._east.append(float(ned[1]))
        self._alt.append(float(-ned[2]))
        self._heading_rad = float(heading_rad)
        self._target_ned = np.asarray(target_ned)
        self._distance_m = float(distance_m)
        self._reason = str(reason) if reason is not None else None

    def draw(self) -> np.ndarray | None:
        if not self._t:
            return None
        r = self.success_radius_m
        t, e, n, h = map(np.asarray, (self._t, self._east, self._north, self._alt))
        t_n, t_e, t_alt = self._target_ned[0], self._target_ned[1], -self._target_ned[2]

        # --- top-down view (x = East, y = North) ---
        ax = self.ax_top
        ax.clear()
        ax.plot(e, n, color="tab:blue", lw=1.5)
        ax.plot(e[0], n[0], "o", color="gray", ms=5, label="start")
        ax.add_patch(Circle((t_e, t_n), r, fill=False, color="tab:red", lw=1.5))
        ax.plot(t_e, t_n, "x", color="tab:red", ms=8, label="target")
        # triangle rotated by heading (clockwise from North)
        ax.plot(
            e[-1], n[-1], ls="none", color="tab:blue", ms=16,
            marker=(3, 0, -np.degrees(self._heading_rad)),
        )
        xs = np.append(e, [t_e - r, t_e + r])
        ys = np.append(n, [t_n - r, t_n + r])
        half = max(np.ptp(xs), np.ptp(ys)) / 2 * 1.15 + 50.0
        ax.set_xlim(xs.mean() - half, xs.mean() + half)
        ax.set_ylim(ys.mean() - half, ys.max() * 0 + ys.mean() + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East, m")
        ax.set_ylabel("North, m")
        ax.set_title("Top view")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        # --- altitude profile ---
        ax = self.ax_alt
        ax.clear()
        ax.plot(t, h, color="tab:blue", lw=1.5, label="aircraft")
        ax.axhline(t_alt, color="tab:red", ls="--", label="target")
        ax.set_xlabel("time, s")
        ax.set_ylabel("altitude (MSL), m")
        ax.set_title("Altitude")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        title = f"t = {t[-1]:.1f} s   distance = {self._distance_m:.0f} m"
        if self._reason is not None:
            title += f"   [{self._reason.upper()}]"
        self.fig.suptitle(title)

        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
            self._plt.pause(1.0 / self.fps)
            return None
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self) -> None:
        if self._plt is not None:
            self._plt.close(self.fig)