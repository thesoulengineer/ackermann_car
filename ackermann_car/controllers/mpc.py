"""
controllers/mpc_controller.py

Model Predictive Controller for the Ackermann car project.
Includes mathematically corrected soft-constrained obstacle avoidance halfplanes.
"""

from __future__ import annotations

import logging
import warnings

import cvxpy as cp
import numpy as np

from .base import BaseController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical / model constants
# ---------------------------------------------------------------------------
WHEELBASE = 0.3  # L — distance between axles, metres

# ---------------------------------------------------------------------------
# Control limits
# ---------------------------------------------------------------------------
A_MAX = 2.0  # maximum acceleration, m/s²
A_MIN = -2.0  # maximum braking deceleration, m/s²
DELTA_MAX = 0.5  # maximum steering angle, radians

DA_MAX = 0.5  # max change in acceleration per step (slew rate)
DDELTA_MAX = 0.3  # max change in steering angle per step

SPEED_MIN = 0.0
SPEED_MAX = 3.5

# ---------------------------------------------------------------------------
# MPC tuning weights
# ---------------------------------------------------------------------------
Q_DIAG = np.array([20.0, 20.0, 2.0, 10.0])  # slightly higher weights for stable tracking
P_SCALE = 8.0
P_DIAG = Q_DIAG * P_SCALE

R_DIAG = np.array([0.5, 1.0])
RD_DIAG = np.array([0.1, 0.5])

WALL_MARGIN = 0.05

E0_CLIP_POS = 4.0
E0_CLIP_VEL = 2.0
E0_CLIP_ANG = 1.0


class MPCController(BaseController):
    """Receding-horizon MPC for the Ackermann car with corrected obstacle halfplanes."""

    def __init__(self, N: int = 25, dt: float = 0.1):
        self.N = N
        self.dt = dt

        self.Q = np.diag(Q_DIAG)
        self.P = np.diag(P_DIAG)
        self.R = np.diag(R_DIAG)
        self.Rd = np.diag(RD_DIAG)

        self._u_prev: np.ndarray = np.zeros((N, 2))
        self._u_last_applied: np.ndarray = np.zeros(2)

        self.M_max = 8

        self._build_problem()
        logger.info(f"MPCController ready (N={N}, dt={dt}, L={WHEELBASE} m)")

    def solve(
        self,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy interface for teammate compatibility."""
        ref = np.asarray(ref, dtype=float)
        state_at_ref = ref[0].copy()
        A_list, B_list = self._linearise_along_ref(ref)
        self._load_parameters(
            state_at_ref, ref, A_list, B_list, boundary_normals, half_width, obstacles=None
        )
        self._warm_start()

        try:
            self._problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                polish=True,
            )
        except cp.SolverError as exc:
            warnings.warn(f"[MPCController] Solver error: {exc}. Falling back.")
            return self._fallback(ref)

        status = self._problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            warnings.warn(f"[MPCController] Solver status: {status}. Falling back.")
            return self._fallback(ref)

        u_opt = self._u_var.value
        e_opt = self._e_var.value
        if u_opt is None or e_opt is None:
            return self._fallback(ref)

        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        self._u_last_applied = u_opt[0]

        predicted_xy = ref[:, :2] + e_opt[:, :2]
        command = np.array(
            [predicted_xy[1, 0], predicted_xy[1, 1], ref[1, 2], ref[1, 3]], dtype=float
        )
        return command, predicted_xy

    def solve_control(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
        obstacles: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve MPC problem and return raw control inputs."""
        ref = np.asarray(ref, dtype=float)
        state = np.asarray(state, dtype=float)

        A_list, B_list = self._linearise_along_ref(ref)
        self._load_parameters(
            state, ref, A_list, B_list, boundary_normals, half_width, obstacles=obstacles
        )
        self._warm_start()

        try:
            self._problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=8000,
                polish=True,
            )
        except cp.SolverError as exc:
            warnings.warn(f"[MPCController] Solver error: {exc}. Falling back.")
            return self._fallback_control(ref)

        status = self._problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            warnings.warn(f"[MPCController] Solver status '{status}'. Falling back.")
            return self._fallback_control(ref)

        u_opt = self._u_var.value
        e_opt = self._e_var.value
        if u_opt is None or e_opt is None:
            return self._fallback_control(ref)

        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        self._u_last_applied = u_opt[0]

        predicted_xy = ref[:, :2] + e_opt[:, :2]
        action = u_opt[0]

        return action, predicted_xy

    def _build_problem(self):
        """Construct the CVXPY optimization problem."""
        N = self.N

        # Decision variables
        self._e_var = cp.Variable((N + 1, 4), name="e")
        self._u_var = cp.Variable((N, 2), name="u")

        # Parameters
        self._e0_par = cp.Parameter(4, name="e0")
        self._A_par = cp.Parameter((N * 4, 4), name="A")
        self._B_par = cp.Parameter((N * 4, 2), name="B")

        self._n_par = cp.Parameter((N, 2), name="normals")
        self._w_par = cp.Parameter(nonneg=True, name="half_width_eff")
        self._u_prev_par = cp.Parameter(2, name="u_prev")

        # Obstacle avoidance parameters
        self._obs_A_par = cp.Parameter((N * self.M_max, 2), name="obs_A")
        self._obs_b_par = cp.Parameter(N * self.M_max, name="obs_b")

        # Soft constraints slack variables
        self._obs_slack = cp.Variable(N * self.M_max, nonneg=True, name="obs_slack")

        cost = 0
        constraints = []

        # Initial state feedback
        constraints += [self._e_var[0] == self._e0_par]

        for k in range(N):
            A_k = self._A_par[k * 4 : (k + 1) * 4, :]
            B_k = self._B_par[k * 4 : (k + 1) * 4, :]

            # Linearized discrete dynamics
            constraints += [self._e_var[k + 1] == A_k @ self._e_var[k] + B_k @ self._u_var[k]]

            # Costs
            cost += cp.quad_form(self._e_var[k], self.Q)
            cost += cp.quad_form(self._u_var[k], self.R)

            du = self._u_var[k] - (self._u_prev_par if k == 0 else self._u_var[k - 1])
            cost += cp.quad_form(du, self.Rd)

            # Control bounds
            constraints += [
                self._u_var[k, 0] >= A_MIN,
                self._u_var[k, 0] <= A_MAX,
                self._u_var[k, 1] >= -DELTA_MAX,
                self._u_var[k, 1] <= DELTA_MAX,
            ]

            # Slew rate limits
            constraints += [
                du[0] >= -DA_MAX,
                du[0] <= DA_MAX,
                du[1] >= -DDELTA_MAX,
                du[1] <= DDELTA_MAX,
            ]

            # Wall boundary constraints
            e_pos = self._e_var[k + 1, :2]
            n_k = self._n_par[k, :]
            constraints += [
                n_k @ e_pos <= self._w_par,
                -n_k @ e_pos <= self._w_par,
            ]

            # Soft obstacle constraints with high linear penalty
            for j in range(self.M_max):
                idx = k * self.M_max + j
                constraints += [
                    self._obs_A_par[idx, :] @ e_pos <= self._obs_b_par[idx] + self._obs_slack[idx]
                ]
                cost += 100000.0 * self._obs_slack[idx] + 5000.0 * cp.square(self._obs_slack[idx])

        # Terminal cost
        cost += cp.quad_form(self._e_var[N], self.P)

        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def _linearise_along_ref(
        self,
        ref: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        N = self.N
        dt = self.dt
        L = WHEELBASE

        A_list = []
        B_list = []

        for k in range(N):
            v_r = float(ref[k, 2])
            theta_r = float(ref[k, 3])

            dfdx = np.array(
                [
                    [0, 0, np.cos(theta_r), -v_r * np.sin(theta_r)],
                    [0, 0, np.sin(theta_r), v_r * np.cos(theta_r)],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
            A_k = np.eye(4) + dt * dfdx

            dfdu = np.array(
                [
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, v_r / L if L > 1e-9 else 0.0],
                ]
            )
            B_k = dt * dfdu

            A_list.append(A_k)
            B_list.append(B_k)

        return A_list, B_list

    def _load_parameters(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        boundary_normals: np.ndarray,
        half_width: float,
        obstacles: list[dict] | None = None,
    ):
        N = self.N

        # Initial state error
        e0 = np.asarray(state, dtype=float) - ref[0]
        e0[3] = (e0[3] + np.pi) % (2 * np.pi) - np.pi
        e0[:2] = np.clip(e0[:2], -E0_CLIP_POS, E0_CLIP_POS)
        e0[2] = np.clip(e0[2], -E0_CLIP_VEL, E0_CLIP_VEL)
        e0[3] = np.clip(e0[3], -E0_CLIP_ANG, E0_CLIP_ANG)
        self._e0_par.value = e0

        # packed parameters
        self._A_par.value = np.vstack(A_list)
        self._B_par.value = np.vstack(B_list)

        normals_N = np.asarray(boundary_normals[:N], dtype=float)
        norms = np.linalg.norm(normals_N, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._n_par.value = normals_N / norms

        self._w_par.value = max(0.0, float(half_width) - WALL_MARGIN)
        self._u_prev_par.value = self._u_last_applied.copy()

        # Corrected mathematical formulation for obstacle halfplanes
        obs_A_val = np.zeros((N * self.M_max, 2))
        obs_b_val = np.full(N * self.M_max, 1e5)

        if obstacles is not None:
            for j, obs in enumerate(obstacles[: self.M_max]):
                obs_pos = np.array([obs["x"], obs["y"]], dtype=float)
                r_obs = float(obs["r"])
                # Vehicle safety margin: obstacle radius + car half-width (0.16) + buffer (0.15)
                d_min = r_obs + 0.23
                influence = 3.0 * d_min

                for k in range(N):
                    p_ref = ref[k + 1, :2]
                    dp = p_ref - obs_pos
                    dist = np.linalg.norm(dp)

                    idx = k * self.M_max + j

                    if dist < 1e-6:
                        dp = state[:2] - obs_pos
                        dist = np.linalg.norm(dp)
                        if dist < 1e-6:
                            dp = np.array([1.0, 0.0])
                            dist = 1.0

                    if dist > influence:
                        # Obstacle is too far to influence current horizon step
                        obs_A_val[idx, :] = 0.0
                        obs_b_val[idx] = 1e5
                        continue

                    # Avoidance unit normal pointing away from the obstacle
                    n_avoid = dp / dist

                    # In error space: -n_avoid^T e_pos <= dist - d_min
                    obs_A_val[idx, :] = -n_avoid
                    obs_b_val[idx] = dist - d_min

        self._obs_A_par.value = obs_A_val
        self._obs_b_par.value = obs_b_val

    def _warm_start(self):
        self._u_var.value = self._u_prev
        self._e_var.value = np.zeros((self.N + 1, 4))

    def _fallback(
        self,
        ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        command = ref[1] if len(ref) > 1 else ref[0]
        predicted = ref[:, :2]
        return np.asarray(command, dtype=float), np.asarray(predicted, dtype=float)

    def _fallback_control(self, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = self._u_prev[0].copy()
        action[0] = float(np.clip(action[0], A_MIN, A_MAX))
        action[1] = float(np.clip(action[1], -DELTA_MAX, DELTA_MAX))
        predicted = ref[:, :2]
        return action, predicted
