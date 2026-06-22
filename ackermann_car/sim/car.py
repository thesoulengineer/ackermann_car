"""
sim/car.py

Kinematic bicycle model dynamics and state integration for the Ackermann car.
"""

from __future__ import annotations

import numpy as np


class KinematicBicycleModel:
    """Kinematic bicycle model representing a flat Ackermann-steering car.

    State representation:
        state = [x, y, v, theta]
            x, y  : Position, meters
            v     : Longitudinal speed, m/s (non-negative)
            theta : Heading angle, radians

    Control inputs:
        action = [a, delta]
            a     : Acceleration, m/s^2
            delta : Front steering angle, radians
    """

    def __init__(self, wheelbase: float = 0.3):
        """Initialize the car model.

        Parameters
        ----------
        wheelbase : float
            Distance L between front and rear axles (default 0.3m for scale car).
        """
        self.L = wheelbase

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """Integrate the dynamics equations for one timestep dt using Euler integration.

        Parameters
        ----------
        state  : (4,) array of [x, y, v, theta]
        action : (2,) array of [a, delta]
        dt     : float, timestep in seconds

        Returns
        -------
        new_state : (4,) array of updated [x, y, v, theta]
        """
        x, y, v, theta = state
        a, delta = action

        # Continuous-time derivatives
        # dX/dt = v * cos(theta)
        # dY/dt = v * sin(theta)
        # dv/dt = a
        # dtheta/dt = (v / L) * tan(delta)
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dv = a
        dtheta = (v / self.L) * np.tan(delta)

        # Euler step
        new_x = x + dx * dt
        new_y = y + dy * dt
        # Clamp velocity to >= 0 as defined in speed limits (no backward motion)
        new_v = max(0.0, v + dv * dt)
        new_theta = theta + dtheta * dt

        # Normalize heading to [-pi, pi]
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        return np.array([new_x, new_y, new_v, new_theta], dtype=float)
