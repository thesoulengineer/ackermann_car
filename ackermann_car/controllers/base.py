"""
controllers/base_controller.py

Abstract base class for all vehicle controllers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """Abstract base class defining the interface for Ackermann car controllers."""

    @abstractmethod
    def solve(
        self,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
        obstacles: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for the control actions given reference trajectory and track boundaries.

        Parameters
        ----------
        ref              : (N+1, 4) array of reference states [x, y, v, theta]
        boundary_normals : (N, 2) array of track boundary normals
        half_width       : float, track half-width
        obstacles        : list of dicts, optional, containing 'x', 'y', 'r' for each obstacle

        Returns
        -------
        control_action    : (2,) array containing [a, delta] (acceleration, steering)
        predicted_horizon : (N+1, 2) array of predicted [x, y] positions
        """
        pass
