"""
tests/test_dynamics.py

Unit tests for the kinematic bicycle dynamics model.
"""

from __future__ import annotations

import numpy as np
from ackermann_car.sim.car import KinematicBicycleModel


def test_straight_line_acceleration():
    """Verify that moving straight with positive acceleration increases velocity and position."""
    model = KinematicBicycleModel(wheelbase=0.3)
    # State: [x, y, v, theta]
    state = np.array([0.0, 0.0, 1.0, 0.0])
    # Action: [a, delta] (accelerating straight)
    action = np.array([2.0, 0.0])
    dt = 0.1

    new_state = model.step(state, action, dt)

    # Velocity should increase: 1.0 + 2.0 * 0.1 = 1.2
    assert np.isclose(new_state[2], 1.2)
    # Position X should increase: 0.0 + 1.0 * cos(0.0) * 0.1 = 0.1
    assert np.isclose(new_state[0], 0.1)
    # Position Y should remain 0
    assert np.isclose(new_state[1], 0.0)
    # Heading should remain 0
    assert np.isclose(new_state[3], 0.0)


def test_velocity_clamping():
    """Verify that the model prevents negative velocity (clamping to >= 0)."""
    model = KinematicBicycleModel(wheelbase=0.3)
    state = np.array([0.0, 0.0, 0.1, 0.0])
    # Strong deceleration
    action = np.array([-2.0, 0.0])
    dt = 0.1

    new_state = model.step(state, action, dt)

    assert new_state[2] == 0.0


def test_steering_left():
    """Verify that steering left (+delta) results in positive heading change."""
    model = KinematicBicycleModel(wheelbase=0.3)
    state = np.array([0.0, 0.0, 2.0, 0.0])
    # Steering left, no acceleration
    action = np.array([0.0, 0.1])
    dt = 0.1

    new_state = model.step(state, action, dt)

    # heading rate = v/L * tan(delta) = 2.0/0.3 * tan(0.1)
    expected_dtheta = (2.0 / 0.3) * np.tan(0.1) * 0.1
    assert np.isclose(new_state[3], expected_dtheta)


def test_theta_wrapping():
    """Verify that heading angle wraps correctly inside [-pi, pi]."""
    model = KinematicBicycleModel(wheelbase=0.3)
    # State close to pi
    state = np.array([0.0, 0.0, 3.0, np.pi - 0.01])
    # Action steering left to push it past pi
    action = np.array([0.0, 0.2])
    dt = 0.1

    new_state = model.step(state, action, dt)

    # Heading should wrap to negative side
    assert -np.pi <= new_state[3] <= np.pi
    assert new_state[3] < 0.0
