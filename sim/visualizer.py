"""
sim/visualizer.py

Matplotlib visualisation helpers for the environment.

This module currently covers the STATIC track layout: the drivable band,
boundaries, centerline, and start/finish marker. The centerline can optionally
be coloured by a per-sample reference speed (heatmap) to sanity-check the speed
profile. Dynamic overlays (car pose, reference horizon, telemetry) come later.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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


if __name__ == "__main__":
    try:
        from sim.track import Track
        from sim.speed_profile import speed_profile
    except ImportError:  # run directly as `python sim/visualizer.py`
        from track import Track
        from speed_profile import speed_profile

    presets = [
        ("Oval", Track.oval()),
        ("Circuit", Track.circuit()),
        ("Figure-eight", Track.figure_eight()),
    ]

    # Static layouts.
    for name, trk in presets:
        fig, ax = plt.subplots(figsize=(8, 7))
        draw_track(trk, ax=ax, title=f"{name} track")
        fig.tight_layout()

    # Oval coloured by its reference speed. The 3 m/s default would clamp this
    # gently-curved oval flat, so the demo raises v_max to make the curvature
    # dependence visible: slower into the tight ends, faster on the gentle ones.
    oval = Track.oval()
    v_ref = speed_profile(oval, v_max=8.0, a_lat_max=2.0)
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_track(oval, ax=ax, title="Oval — reference speed profile", speed=v_ref)
    fig.tight_layout()

    plt.show()
