"""
sim/visualizer.py

Matplotlib visualisation helpers for the environment.

This module currently covers the STATIC track layout: the drivable band,
boundaries, centerline, and start/finish marker. The centerline can optionally
be coloured by a per-sample reference speed (heatmap) to sanity-check the speed
profile. Dynamic overlays (car pose, reference horizon, telemetry) come later.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation


def _closed(arr):
    """Append the first row so a sampled closed loop plots without a seam gap."""
    return np.concatenate([arr, arr[:1]], axis=0)


def _speed_line(ax, track, speed, cmap):
    """Draw the centerline as line segments coloured by reference speed."""
    pts = _closed(np.column_stack([track.cx, track.cy]))
    sp = np.concatenate([speed, speed[:1]])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)      # (N, 2, 2)
    seg_speed = 0.5 * (sp[:-1] + sp[1:])                  # colour per segment
    lc = LineCollection(segments, cmap=cmap, zorder=3)
    lc.set_array(seg_speed)
    lc.set_linewidth(3.0)
    ax.add_collection(lc)
    return lc


def draw_track(track, ax=None, title=None, speed=None,
               band_color="#cdd6dd", edge_color="0.2",
               center_color="tab:blue", start_color="crimson",
               speed_cmap="viridis"):
    """Render a Track's static layout on a matplotlib axis.

    Draws the drivable area as a filled band between the left and right
    boundaries, both boundary lines, and a distinct start/finish marker.
    The centerline is drawn dashed, or — if ``speed`` (a per-sample array the
    same length as the centerline) is given — as a heatmap coloured by
    reference speed with a colorbar. Aspect ratio is locked to equal.

    Returns the axis it drew on (created if ax is None).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    lb = track.left_bound
    rb = track.right_bound

    # Drivable area: trace one boundary loop forward and the other backward into
    # a single polygon. Reversing the second loop makes the loops wind opposite
    # ways, so the infield is carved out and only the band between the
    # boundaries is filled. The two connector segments overlap at the seam.
    band_x = np.concatenate([lb[:, 0], rb[::-1, 0]])
    band_y = np.concatenate([lb[:, 1], rb[::-1, 1]])
    ax.fill(band_x, band_y, color=band_color, zorder=0, label="drivable area")

    # Boundaries, closed by wrapping the first sample so the seam is a join.
    lbc = _closed(lb)
    rbc = _closed(rb)
    ax.plot(lbc[:, 0], lbc[:, 1], "-", color=edge_color, lw=1.5, label="boundaries")
    ax.plot(rbc[:, 0], rbc[:, 1], "-", color=edge_color, lw=1.5)

    # Centerline: plain dashed line, or a speed heatmap.
    if speed is None:
        cc = _closed(np.column_stack([track.cx, track.cy]))
        ax.plot(cc[:, 0], cc[:, 1], "--", color=center_color, lw=1.3,
                label="centerline")
    else:
        speed = np.asarray(speed, dtype=float)
        if speed.shape[0] != len(track.cx):
            raise ValueError("speed length must match the centerline length")
        lc = _speed_line(ax, track, speed, speed_cmap)
        cbar = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("reference speed [m/s]")

    # Start/finish point (sample 0).
    ax.plot(track.cx[0], track.cy[0], marker="*", color=start_color,
            markersize=16, zorder=5, label="start/finish")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    ax.autoscale_view()
    return ax


class LiveView:
    """Animated live view of the running simulation over a static track.

    The static track (band, boundaries, centerline) is drawn once as the
    backdrop. Each frame updates the car pose (a marker plus a heading line),
    its trailing path, an optional MPC prediction horizon, and a HUD showing
    lap, lap time, and speed.

    Wire it into an integration loop in either of two ways:

    * call ``update(car_state, horizon, lap_info)`` yourself each step, or
    * pass an iterable of ``(car_state, horizon, lap_info)`` frames to
      ``animate(...)`` and let matplotlib's FuncAnimation drive it.

    ``car_state`` is the [x, y, v, theta] state vector (see docs/INTERFACE.md);
    ``horizon`` is an optional (N+1, 2) array of predicted positions (None to
    hide it); ``lap_info`` is an optional dict with ``lap`` and ``lap_time``.
    """

    def __init__(self, track, trail_maxlen=2000, heading_len=None, figsize=(9, 7)):
        self.track = track
        self.fig, self.ax = plt.subplots(figsize=figsize)
        draw_track(track, ax=self.ax)  # static backdrop

        extent = max(np.ptp(track.cx), np.ptp(track.cy))
        self.heading_len = heading_len or max(2.0 * track.half_width, 0.03 * extent)
        self._trail = deque(maxlen=trail_maxlen)
        self._anim = None

        # Animated artists (drawn on top of the static backdrop).
        (self.trail_ln,) = self.ax.plot([], [], "-", color="tab:blue", lw=1.5,
                                        alpha=0.7, zorder=4, label="path")
        (self.horizon_ln,) = self.ax.plot([], [], "-o", color="tab:orange",
                                          ms=3, lw=1.5, zorder=6, label="MPC horizon")
        (self.car_pt,) = self.ax.plot([], [], "o", color="crimson", ms=11, zorder=8)
        (self.heading_ln,) = self.ax.plot([], [], "-", color="crimson", lw=2.5,
                                          zorder=8)
        self.hud = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace", zorder=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

        self.ax.legend(loc="upper right", fontsize=8)
        self._artists = [self.trail_ln, self.horizon_ln, self.car_pt,
                         self.heading_ln, self.hud]

    def reset(self):
        """Clear the trailing path (e.g. before starting a new run)."""
        self._trail.clear()

    def init_artists(self):
        """FuncAnimation init_func: blank every animated artist."""
        self.trail_ln.set_data([], [])
        self.horizon_ln.set_data([], [])
        self.car_pt.set_data([], [])
        self.heading_ln.set_data([], [])
        self.hud.set_text("")
        return self._artists

    def update(self, car_state, horizon=None, lap_info=None):
        """Update all animated artists for one frame; returns them (for blit)."""
        cs = np.asarray(car_state, dtype=float)
        x, y = float(cs[0]), float(cs[1])
        speed = float(cs[2]) if cs.shape[0] >= 3 else float("nan")
        theta = float(cs[3]) if cs.shape[0] >= 4 else 0.0

        self.car_pt.set_data([x], [y])
        self.heading_ln.set_data([x, x + self.heading_len * np.cos(theta)],
                                 [y, y + self.heading_len * np.sin(theta)])

        self._trail.append((x, y))
        tx, ty = zip(*self._trail)
        self.trail_ln.set_data(tx, ty)

        if horizon is not None and len(horizon) > 0:
            h = np.asarray(horizon, dtype=float)
            self.horizon_ln.set_data(h[:, 0], h[:, 1])
        else:
            self.horizon_ln.set_data([], [])

        self.hud.set_text(self._hud_text(lap_info, speed))
        return self._artists

    @staticmethod
    def _hud_text(lap_info, speed):
        info = lap_info or {}
        lap = info.get("lap", "-")
        lap_time = info.get("lap_time", float("nan"))
        return (f"lap   : {lap}\n"
                f"time  : {lap_time:6.2f} s\n"
                f"speed : {speed:5.2f} m/s")

    def _on_frame(self, frame):
        car_state, horizon, lap_info = frame
        return self.update(car_state, horizon, lap_info)

    def animate(self, frames, interval=30, blit=True, repeat=False):
        """Drive the view from an iterable of (car_state, horizon, lap_info).

        Returns the FuncAnimation (also kept on the instance so it is not
        garbage-collected before plt.show()).
        """
        self.reset()
        self._anim = FuncAnimation(
            self.fig, self._on_frame, frames=frames, init_func=self.init_artists,
            interval=interval, blit=blit, repeat=repeat, cache_frame_data=False)
        return self._anim


def _centerline_drive(track, dt=0.1, n_laps=2, N=15):
    """Fake telemetry: a car following the centerline at the reference speed.

    Yields (car_state, horizon, lap_info) frames. The "predicted horizon" is a
    stand-in built from the track's own reference lookup (no real controller
    yet), which is exactly the (N+1, 2) shape LiveView expects.
    """
    v = track.v_ref
    P = len(track.cx)
    i = 0
    sim_t = 0.0
    lap = 1
    lap_t0 = 0.0
    laps_done = 0
    while laps_done < n_laps:
        state = np.array([track.cx[i], track.cy[i], v[i], track.theta[i]])
        horizon = track.get_reference(state, N, dt)[:, :2]
        yield state, horizon, {"lap": lap, "lap_time": sim_t - lap_t0}

        step = max(1, int(round(v[i] * dt / track.ds)))
        nxt = (i + step) % P
        sim_t += dt
        if nxt < i:  # wrapped past the start/finish seam -> lap complete
            laps_done += 1
            lap += 1
            lap_t0 = sim_t
        i = nxt


if __name__ == "__main__":
    try:
        from .track import Track
    except ImportError:  # run directly as `python sim/visualizer.py`
        from track import Track

    oval = Track.oval()
    view = LiveView(oval)
    view.ax.set_title("Live view demo — fake car on the oval centerline")
    # Keep a reference to the animation so it is not garbage-collected.
    anim = view.animate(_centerline_drive(oval, dt=0.1, n_laps=2, N=15), interval=20)
    plt.show()
