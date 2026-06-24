"""
obstacles.py

Random obstacle generator for the racetrack environment.

Obstacles are circles described by ``{"x", "y", "r"}`` dicts (metres) -- the format
the simulator draws and the MPC controller consumes as avoidance constraints.
"""

from __future__ import annotations

import numpy as np


def generate_obstacles(track, radius=2.0, n_min=3, n_max=10, rng=None):
    """Randomly place same-radius circular obstacles inside the track corridor.

    For each obstacle a random point is picked on the track's inner boundary, then
    a random point along that point's normal, kept far enough from both edges that
    the whole circle stays inside the track band.

    Parameters
    ----------
    track   : Track instance (provides cx, cy, normals, half_width, width).
    radius  : float, common radius of every obstacle, metres.
    n_min,
    n_max   : int, inclusive bounds on the random obstacle count.
    rng     : optional numpy Generator for reproducibility; a fresh one is used
              when None (so each run produces a different layout).

    Returns
    -------
    list of {"x", "y", "r"} dicts, length in [n_min, n_max].
    """
    rng = np.random.default_rng() if rng is None else rng
    if radius >= track.half_width:
        raise ValueError(
            f"radius {radius} >= half_width {track.half_width}: no room to place obstacles"
        )

    n = int(rng.integers(n_min, n_max + 1))  # n_max inclusive
    n_samples = len(track.cx)
    obstacles = []
    for _ in range(n):
        # Random point on the inner boundary (left_bound for a CCW track):
        # left-pointing normals face inward, so centerline + half_width * normal.
        i = int(rng.integers(0, n_samples))
        nx, ny = track.normals[i]
        inner_x = track.cx[i] + track.half_width * nx
        inner_y = track.cy[i] + track.half_width * ny

        # Step across the corridor along the normal. Bounding the distance to
        # [radius, width - radius] keeps the entire circle within the track.
        d = rng.uniform(radius, track.width - radius)
        obstacles.append(
            {
                "x": float(inner_x - d * nx),
                "y": float(inner_y - d * ny),
                "r": float(radius),
            }
        )
    return obstacles


if __name__ == "__main__":
    # Numeric sanity check against the default oval.
    from ackermann_car.sim.track import Track

    track = Track.oval(n=100, width=15.0)
    obs = generate_obstacles(track, rng=np.random.default_rng(0))
    center = np.column_stack([track.cx, track.cy])
    print(f"generated {len(obs)} obstacles (radius {obs[0]['r']} m):")
    for o in obs:
        d_center = np.hypot(center[:, 0] - o["x"], center[:, 1] - o["y"]).min()
        inside = d_center <= track.half_width - o["r"] + 1e-9
        print(f"  ({o['x']:7.2f}, {o['y']:7.2f})  lateral={d_center:5.2f} m  inside={inside}")
