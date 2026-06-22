"""
tests/test_obstacles.py

Unit tests for obstacle avoidance behavior in the MPC Controller.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
import pytest
from ackermann_car.controllers.mpc import MPCController


def test_no_obstacles():
    """Verify that MPCController solves optimally and tracks the path when no obstacles are present."""
    N = 15
    dt = 0.1
    mpc = MPCController(N=N, dt=dt)

    # Reference trajectory: straight line along x-axis from x=0.0 to x=1.5
    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N)])
    half_width = 2.0

    # Initial state is exactly on reference at step 0
    state = ref[0]

    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=None
    )

    # Solver should converge to OPTIMAL
    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    
    # Action and predicted horizon shapes must be correct
    assert action.shape == (2,)
    assert predicted.shape == (N + 1, 2)
    
    # In the absence of obstacles, the car should track the straight reference line closely
    # The lateral deviation (y) should remain close to 0.0
    assert np.allclose(predicted[:, 1], 0.0, atol=0.1)


def test_obstacle_avoidance():
    """Verify that the MPC controller successfully steers away from an obstacle on the path."""
    N = 15
    dt = 0.1
    mpc = MPCController(N=N, dt=dt)

    # Reference trajectory: straight line along x-axis from x=0.0 to x=1.5
    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N)])
    half_width = 2.0

    # Place an obstacle directly on the path at x=0.6, y=0.0 with radius 0.2
    # Safe distance will be r + 0.55 = 0.75m.
    obstacles = [{"x": 0.6, "y": 0.0, "r": 0.2}]
    
    state = ref[0]

    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=obstacles
    )

    # Solver should be feasible and optimal
    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    # The predicted horizon should steer away from the obstacle (deviating laterally in y)
    # Since the obstacle is at y=0, the car should steer either left or right (y != 0)
    max_lateral_deviation = np.max(np.abs(predicted[:, 1]))
    assert max_lateral_deviation > 0.1, "Car did not steer to avoid the obstacle!"
    
    # Also verify that by the second half of the horizon (k >= 8), the vehicle has steered
    # to a safe lateral distance, so the slack variable for the active obstacle drops to zero.
    slack_values = mpc._obs_slack.value
    assert slack_values is not None
    active_slack = slack_values[0::mpc.M_max]
    assert np.all(active_slack[8:] < 0.05)


def test_unavoidable_obstacle():
    """Verify that if an obstacle is completely unavoidable (e.g. blocking the entire track),
    the soft constraint (slack variable) allows the QP to remain feasible rather than crashing.
    """
    N = 15
    dt = 0.1
    mpc = MPCController(N=N, dt=dt)

    # Reference trajectory: straight line along x-axis from x=0.0 to x=1.5
    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N)])
    
    # Narrow track: half width of 0.2m, but a huge obstacle of radius 5.0m directly on the track
    half_width = 0.2
    obstacles = [{"x": 0.6, "y": 0.0, "r": 5.0}]
    
    state = ref[0]

    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=obstacles
    )

    # Thanks to the slack variable formulation, the QP remains FEASIBLE (optimal/optimal_inaccurate)
    # instead of returning INFEASIBLE and failing.
    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    
    # The slack variables should be positive, reflecting the fact that the constraint was violated
    # because it was physically impossible to stay on track and avoid the huge obstacle.
    slack_values = mpc._obs_slack.value
    assert slack_values is not None
    assert np.any(slack_values > 0.1), "Slack variable did not activate for unavoidable obstacle!"
