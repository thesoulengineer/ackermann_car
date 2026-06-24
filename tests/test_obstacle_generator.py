"""
tests/test_obstacle_generator.py

Unit tests for the random obstacle generator (ackermann_car.sim.obstacles).
"""

from __future__ import annotations

import numpy as np
import pytest

from ackermann_car.sim.obstacles import generate_obstacles
from ackermann_car.sim.track import Track


def _track():
    # Small oval keeps spline construction fast; default width=10 -> half_width=5.
    return Track.oval(n=16)


def test_count_within_range():
    track = _track()
    for seed in range(25):
        obs = generate_obstacles(track, rng=np.random.default_rng(seed))
        assert 3 <= len(obs) <= 10


def test_count_respects_custom_bounds():
    track = _track()
    for seed in range(15):
        obs = generate_obstacles(track, n_min=5, n_max=5, rng=np.random.default_rng(seed))
        assert len(obs) == 5


def test_all_same_radius():
    track = _track()
    radius = 2.0
    obs = generate_obstacles(track, radius=radius, rng=np.random.default_rng(0))
    assert all(o["r"] == radius for o in obs)


def test_dict_format():
    track = _track()
    obs = generate_obstacles(track, rng=np.random.default_rng(0))
    for o in obs:
        assert set(o.keys()) == {"x", "y", "r"}
        assert all(isinstance(o[k], float) for k in ("x", "y", "r"))


def test_obstacles_within_track():
    """Every obstacle's whole circle must stay inside the corridor: the centre's
    perpendicular distance to the centerline is at most half_width - radius."""
    track = _track()
    radius = 2.0
    center = np.column_stack([track.cx, track.cy])
    for seed in range(10):
        obs = generate_obstacles(track, radius=radius, rng=np.random.default_rng(seed))
        for o in obs:
            min_dist = np.hypot(center[:, 0] - o["x"], center[:, 1] - o["y"]).min()
            assert min_dist <= track.half_width - radius + 1e-6


def test_deterministic_with_seed():
    track = _track()
    a = generate_obstacles(track, rng=np.random.default_rng(42))
    b = generate_obstacles(track, rng=np.random.default_rng(42))
    assert a == b


def test_radius_too_large_raises():
    track = _track()  # half_width = 5.0
    with pytest.raises(ValueError):
        generate_obstacles(track, radius=5.0)
