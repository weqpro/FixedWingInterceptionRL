"""3D flight visualisation: builds interactive HTML pages from ready-made trajectories.

Pure plotting library: no simulation, no env imports.
"""

import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

COLORS = ["royalblue", "darkorange", "seagreen", "crimson", "purple", "goldenrod", "teal", "gray"]


def _points(p, frame):
    a = np.asarray(p, dtype=float).reshape(-1, 3)
    if frame == "ned":  # -> east, north, up
        a = np.stack([a[:, 1], a[:, 0], -a[:, 2]], -1)
    elif frame != "enu":
        raise ValueError('frame must be "enu" or "ned"')
    return a


def _named(items):
    """dict {name: points} or a list of point sequences -> list of (name, points)."""
    if items is None:
        return []
    if isinstance(items, dict):
        return list(items.items())
    return [(f"path {i}", p) for i, p in enumerate(items)]


def _scene(arrays, margin=10.0):
    allp = np.concatenate(arrays)
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    span = hi - lo
    return dict(xaxis=dict(range=[lo[0], hi[0]], title="East m"), yaxis=dict(range=[lo[1], hi[1]], title="North m"),
                zaxis=dict(range=[lo[2], hi[2]], title="Up m"), aspectmode="manual",
                aspectratio=dict(x=1, y=span[1] / span[0], z=span[2] / span[0]))


def _save(fig, out, show):
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    path = Path(out).resolve()
    if show:
        webbrowser.open(path.as_uri())
    return path


def plot_paths(paths, points=None, out="paths.html", title="", frame="enu", show=True):
    """Static 3D view.

    paths:  {name: points} or [points, ...] — drawn as lines (trajectories), start marked with a dot
    points: {name: points} or [points, ...] — drawn as markers only (gates, waypoints, hits, ...)
    out:    html file to write; show=True opens it in the browser. Returns the file path.
    """
    fig, arrays = go.Figure(), []
    for i, (name, p) in enumerate(_named(paths)):
        a = _points(p, frame); arrays.append(a)
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines", name=name,
                                   line=dict(width=5, color=color)))
        fig.add_trace(go.Scatter3d(x=a[:1, 0], y=a[:1, 1], z=a[:1, 2], mode="markers", showlegend=False,
                                   marker=dict(size=5, color=color), hovertext=f"{name} start", hoverinfo="text"))
    for j, (name, p) in enumerate(_named(points)):
        a = _points(p, frame); arrays.append(a)
        fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="markers", name=name,
                                   marker=dict(size=4, color=COLORS[(j + 4) % len(COLORS)])))
    fig.update_layout(title=title, scene=_scene(arrays), legend=dict(x=0, y=1))
    return _save(fig, out, show)


def animate_paths(paths, dt=0.1, out="paths_live.html", title="", frame="enu", stride=1, trail=True, show=True):
    """Real-time playback: every path is a moving object sampled every dt seconds (shorter paths stop at their end).

    paths:  {name: points} or [points, ...], sample k of every path is at time k * dt
    stride: draw every stride-th sample (keeps big files light; playback stays real-time)
    trail:  draw the path flown so far behind each object
    Controls in the page: play at 1x or 0.25x, pause, time slider. Returns the file path.
    """
    named = [(name, _points(p, frame)) for name, p in _named(paths)]
    T = max(len(a) for _, a in named)
    steps = list(range(0, T, stride)) + ([T - 1] if (T - 1) % stride else [])

    def frame_data(k):
        data = []
        for i, (name, a) in enumerate(named):
            e = min(k, len(a) - 1)
            color = COLORS[i % len(COLORS)]
            seg = a[:e + 1] if trail else a[e:e + 1]
            data.append(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                                     line=dict(width=4, color=color), showlegend=False))
            data.append(go.Scatter3d(x=a[e:e + 1, 0], y=a[e:e + 1, 1], z=a[e:e + 1, 2], mode="markers+text",
                                     text=[name], marker=dict(size=6, color=color), textfont=dict(color=color),
                                     showlegend=False))
        return data

    status = lambda k: f"{title}<br>t = {k * dt:.1f} s" if title else f"t = {k * dt:.1f} s"
    fig = go.Figure(data=frame_data(0), frames=[go.Frame(data=frame_data(k), name=str(k),
                                                         layout=dict(title=status(k))) for k in steps])
    ms = stride * dt * 1000
    play = lambda speed: dict(label=f"▶ {speed}x", method="animate", args=[None, dict(
        frame=dict(duration=ms / speed, redraw=True), transition=dict(duration=0), fromcurrent=True)])
    fig.update_layout(
        title=status(0), scene=_scene([a for _, a in named]),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", buttons=[
            play(1), play(0.25), dict(label="❚❚", method="animate",
                                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.25, len=0.75, y=0, currentvalue=dict(prefix="t = ", suffix=" s"), steps=[
            dict(label=f"{k * dt:.1f}", method="animate",
                 args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for k in steps])])
    return _save(fig, out, show)