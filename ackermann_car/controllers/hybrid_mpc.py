"""
controllers/hybrid_mpc.py

Hybrid MPC controller using a highly stable iterative LTV-MPC (iMPC) formulation solved
entirely in the vehicle's local (ego) frame, adapted from the reference cvxpy_mpc module.

This implementation features dynamic reference speed scaling when approaching large
obstacles to allow safe cornering, blended normal fields, and a hysteresis bypass latch.
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
WHEELBASE = 0.3
A_MAX = 2.0
A_MIN = -2.0
DELTA_MAX = 0.5

DA_MAX = 10.0
DDELTA_MAX = 2.0
SPEED_MAX = 3.5

# ---------------------------------------------------------------------------
# MPC tuning weights (diagonal elements for along, cross, v, theta errors)
# ---------------------------------------------------------------------------
Q_DIAG = np.array([1.0, 15.0, 10.0, 20.0])  # [along, cross, v, theta]
P_SCALE = 8.0
P_DIAG = Q_DIAG * P_SCALE
R_DIAG = np.array([0.1, 5.0])  # [a, delta]
RD_DIAG = np.array([0.05, 5.0])  # control rate penalties

WALL_MARGIN = 0.05
OBSTACLE_MARGIN = 0.1
SLACK_PENALTY = 1e4
QUAD_SLACK_PENALTY = 1.0


class HybridMPCController(BaseController):
    def __init__(
        self,
        N: int = 10,
        dt: float = 0.1,
        enable_walls: bool = True,
        enable_obstacles: bool = True,
        max_iter: int = 2,
        solver: str = "OSQP",
        rho: float = SLACK_PENALTY,
        q_diag: np.ndarray = Q_DIAG,
        r_diag: np.ndarray = R_DIAG,
        rd_diag: np.ndarray = RD_DIAG,
        p_scale: float = P_SCALE,
        da_max: float = DA_MAX,
        ddelta_max: float = DDELTA_MAX,
        obs_margin: float = OBSTACLE_MARGIN,
        obstacle_activation_range: float = 8.0,
        blend_distance: float = 10.0,
        speed_scale_distance: float = 5.0,
        speed_scale_radius: float = 1.2,
        speed_scale_min: float = 0.5,
    ):
        self.N = N
        self.dt = dt
        self.enable_walls = enable_walls
        self.enable_obstacles = enable_obstacles
        self.max_iter = max_iter
        self.solver = solver
        self.rho = rho

        self.wheelbase = WHEELBASE
        self.width = 0.3  # Scale vehicle width
        self.v_max = SPEED_MAX
        self.a_max = A_MAX
        self.a_min = A_MIN
        self.delta_max = DELTA_MAX
        self.da_max = da_max
        self.ddelta_max = ddelta_max
        self.wall_margin = WALL_MARGIN
        self.obs_margin = obs_margin
        # Obstacle-avoidance shaping knobs (tunable; defaults reproduce the original literals).
        self.obstacle_activation_range = obstacle_activation_range
        self.blend_distance = blend_distance
        self.speed_scale_distance = speed_scale_distance
        self.speed_scale_radius = speed_scale_radius
        self.speed_scale_min = speed_scale_min

        # Cost matrices (tunable via constructor; defaults reproduce the module constants).
        self.Q = np.diag(q_diag)
        self.P = np.diag(np.asarray(q_diag, dtype=float) * p_scale)
        self.R = np.diag(r_diag)
        self.Rd = np.diag(rd_diag)

        # Precompute square-root cost matrices for CVXPY DPP-compliance (sum_squares formulation)
        self.Q_matrix = np.sqrt(self.Q)
        self.P_matrix = np.sqrt(self.P)
        self.R_matrix = np.sqrt(self.R)
        self.Rd_matrix = np.sqrt(self.Rd)

        # Memory for warm starts and hysteresis decision latches
        self._u_prev = np.zeros((2, N))
        self._u_last_applied = np.zeros(2)
        self._X_prev = None  # Stores previous trajectory in ego-frame
        self._bypass_side = None  # Persistent lateral bypass direction latch

        # OSQP Polishing option (managed by benchmarks in test_obstacles.py)
        self._osqp_polish = False

        self._build_problem()
        logger.info(
            f"HybridMPCController initialized in Ego Frame (N={N}, dt={dt}, "
            f"solver={solver}, walls={enable_walls}, obstacles={enable_obstacles})"
        )

    def _build_problem(self):
        """Constructs the parameterized DPP-compliant optimization problem."""
        N = self.N

        # CVXPY Variables (solved in ego-frame)
        self._states = cp.Variable((4, N + 1), name="states")
        self._controls = cp.Variable((2, N), name="controls")

        # Unified slack variable vector for soft constraints (walls and obstacles)
        self._S_var = cp.Variable(N, nonneg=True, name="S")

        # CVXPY Parameters (runtime placeholders)
        self._initial_state = cp.Parameter(4, name="x0")
        self._last_command = cp.Parameter(2, name="u_last")

        # LTV model parameters (A_k, B_k, C_k) for each step
        self._A_params = [cp.Parameter((4, 4), name=f"A_{k}") for k in range(N)]
        self._B_params = [cp.Parameter((4, 2), name=f"B_{k}") for k in range(N)]
        self._C_params = [cp.Parameter(4, name=f"C_{k}") for k in range(N)]

        # Pre-calculated reference path descriptors mapped to ego-frame
        self._cos_reference = cp.Parameter(N + 1, name="cos_ref")
        self._sin_reference = cp.Parameter(N + 1, name="sin_ref")
        self._along_reference = cp.Parameter(N + 1, name="along_ref")
        self._cross_reference = cp.Parameter(N + 1, name="cross_ref")
        self._velocity_reference = cp.Parameter(N + 1, name="v_ref")
        self._heading_reference = cp.Parameter(N + 1, name="theta_ref")

        # Half-plane linear parameters for obstacle avoidance
        self._obstacle_normal_x = cp.Parameter(N, name="obs_nx")
        self._obstacle_normal_y = cp.Parameter(N, name="obs_ny")
        self._obstacle_safe_distance = cp.Parameter(N, name="obs_dist")

        # Boundary tracking wall parameters (rotated to ego-frame)
        self._wall_normal_x = cp.Parameter(N, name="wall_nx")
        self._wall_normal_y = cp.Parameter(N, name="wall_ny")
        self._wall_half_width = cp.Parameter(N, name="wall_width")
        self._wall_ref_proj = cp.Parameter(N, name="wall_ref_proj")

        cost = 0
        constraints = []

        # Initial state equality constraint (starts at [0, 0, v, 0] in ego frame)
        constraints += [self._states[:, 0] == self._initial_state]

        # Dynamics constraints
        for k in range(N):
            constraints += [
                self._states[:, k + 1]
                == self._A_params[k] @ self._states[:, k]
                + self._B_params[k] @ self._controls[:, k]
                + self._C_params[k]
            ]

        # Tracking cost over the horizon
        for k in range(N):
            along_track_error = (
                self._cos_reference[k] * self._states[0, k]
                + self._sin_reference[k] * self._states[1, k]
                - self._along_reference[k]
            )
            cross_track_error = (
                -self._sin_reference[k] * self._states[0, k]
                + self._cos_reference[k] * self._states[1, k]
                - self._cross_reference[k]
            )
            error = cp.vstack(
                [
                    along_track_error,
                    cross_track_error,
                    self._states[2, k] - self._velocity_reference[k],
                    self._states[3, k] - self._heading_reference[k],
                ]
            )
            cost += cp.sum_squares(self.Q_matrix @ error)

            # Actuation effort costs
            cost += cp.sum_squares(self.R_matrix @ self._controls[:, k])
            if k == 0:
                cost += cp.sum_squares(self.Rd_matrix @ (self._controls[:, 0] - self._last_command))
            else:
                cost += cp.sum_squares(
                    self.Rd_matrix @ (self._controls[:, k] - self._controls[:, k - 1])
                )

            # Soft constraint slack penalties (balanced linear and quadratic elements)
            if self.enable_obstacles or self.enable_walls:
                cost += self.rho * self._S_var[k] + QUAD_SLACK_PENALTY * cp.square(self._S_var[k])
                constraints += [self._S_var[k] <= 1e6]

            # Obstacle avoidance constraints (Half-plane linearization)
            if self.enable_obstacles:
                constraints += [
                    self._obstacle_normal_x[k] * self._states[0, k + 1]
                    + self._obstacle_normal_y[k] * self._states[1, k + 1]
                    >= self._obstacle_safe_distance[k] - self._S_var[k]
                ]

            # Wall boundary constraints
            if self.enable_walls:
                signed_dist = (
                    self._wall_normal_x[k] * self._states[0, k + 1]
                    + self._wall_normal_y[k] * self._states[1, k + 1]
                )
                constraints += [
                    signed_dist
                    <= self._wall_half_width[k] + self._wall_ref_proj[k] + self._S_var[k],
                    -signed_dist
                    <= self._wall_half_width[k] - self._wall_ref_proj[k] + self._S_var[k],
                ]

        # Terminal state cost
        terminal_along_track_error = (
            self._cos_reference[-1] * self._states[0, -1]
            + self._sin_reference[-1] * self._states[1, -1]
            - self._along_reference[-1]
        )
        terminal_cross_track_error = (
            -self._sin_reference[-1] * self._states[0, -1]
            + self._cos_reference[-1] * self._states[1, -1]
            - self._cross_reference[-1]
        )
        terminal_error = cp.vstack(
            [
                terminal_along_track_error,
                terminal_cross_track_error,
                self._states[2, -1] - self._velocity_reference[-1],
                self._states[3, -1] - self._heading_reference[-1],
            ]
        )
        cost += cp.sum_squares(self.P_matrix @ terminal_error)

        # State bounds
        constraints += [
            self._states[2, :] <= self.v_max,
            self._states[2, :] >= 0.0,
        ]

        # Actuation limits
        constraints += [
            self._controls[0, :] >= self.a_min,
            self._controls[0, :] <= self.a_max,
            self._controls[1, :] >= -self.delta_max,
            self._controls[1, :] <= self.delta_max,
        ]

        # Actuation rate limits
        constraints += [
            cp.abs(self._controls[0, 0] - self._last_command[0]) <= self.da_max * self.dt,
            cp.abs(self._controls[1, 0] - self._last_command[1]) <= self.ddelta_max * self.dt,
        ]
        for k in range(1, N):
            constraints += [
                cp.abs(self._controls[0, k] - self._controls[0, k - 1]) <= self.da_max * self.dt,
                cp.abs(self._controls[1, k] - self._controls[1, k - 1])
                <= self.ddelta_max * self.dt,
            ]

        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def _compute_linear_model_matrices(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates discrete LTV system matrices around the trajectory guess."""
        v = x_bar[2]
        theta = x_bar[3]
        a = u_bar[0]
        delta = u_bar[1]

        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)

        # Continuous Jacobians
        A = np.zeros((4, 4))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        A[3, 2] = td / self.wheelbase

        A_lin = np.eye(4) + self.dt * A

        B = np.zeros((4, 2))
        B[2, 0] = 1.0
        B[3, 1] = v / (self.wheelbase * (cd**2) + 1e-9)

        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.wheelbase])

        C_lin = self.dt * (f_xu - A @ x_bar - B @ u_bar)

        return A_lin, B_lin, C_lin

    def solve_control(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
        obstacles: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the MPC optimization by projecting the global path onto the ego frame.

        Parameters
        ----------
        state : (4,) global state array [x, y, v, theta]
        ref : (N+1, 4) global reference states [x, y, v_ref, theta_ref]
        boundary_normals : (N+1, 2) left boundary normals
        half_width : float, track width / 2
        obstacles : list of dicts with 'x', 'y', 'r' keys

        Returns
        -------
        action : (2,) [acceleration, steering]
        predicted_xy : (N+1, 2) predicted global trajectory positions
        """
        state = np.asarray(state, dtype=float)
        ref = np.asarray(ref, dtype=float)
        normals = np.asarray(boundary_normals, dtype=float)

        if len(normals) < self.N + 1:
            last = normals[-1] if len(normals) > 0 else np.array([1.0, 0.0])
            pad = np.tile(last, (self.N + 1 - len(normals), 1))
            normals = np.vstack([normals, pad])
        normals = normals[: self.N + 1]

        # -------------------------------------------------------------------
        # 1. Transform global reference trajectory to vehicle-local ego-frame
        # -------------------------------------------------------------------
        xref = ref.T.copy()
        dx = xref[0, :] - state[0]
        dy = xref[1, :] - state[1]
        cos_theta = np.cos(-state[3])
        sin_theta = np.sin(-state[3])

        xref_ego = np.zeros_like(xref)
        xref_ego[0, :] = dx * cos_theta - dy * sin_theta
        xref_ego[1, :] = dy * cos_theta + dx * sin_theta
        xref_ego[2, :] = xref[2, :]
        xref_ego[3, :] = xref[3, :] - state[3]

        xref_ego[3, :] = (xref_ego[3, :] + np.pi) % (2 * np.pi) - np.pi
        xref_ego[3, :] = xref_ego[3, 0] + np.unwrap(xref_ego[3, :] - xref_ego[3, 0])

        # -------------------------------------------------------------------
        # 2. Process Obstacle coordinate transformations and Evasion Latching
        # -------------------------------------------------------------------
        obstacle_ego = None
        if self.enable_obstacles and obstacles:
            closest_obs = None
            min_dist = float("inf")
            for obs in obstacles:
                ox, oy, r = obs["x"], obs["y"], obs["r"]
                dist = np.hypot(ox - state[0], oy - state[1]) - r
                if dist < min_dist:
                    min_dist = dist
                    closest_obs = (ox, oy, r, 0.0, 0.0)

            # Process closest active obstacle within horizon range (8.0 meters)
            if closest_obs is not None and min_dist < self.obstacle_activation_range:
                gx, gy, r, vx, vy = closest_obs
                dx_obs = gx - state[0]
                dy_obs = gy - state[1]
                obstacle_ego = (
                    dx_obs * cos_theta - dy_obs * sin_theta,
                    dy_obs * cos_theta + dx_obs * sin_theta,
                    r,
                    vx * cos_theta - vy * sin_theta,
                    vy * cos_theta + vx * sin_theta,
                )

        # Reset bypass latch if no active obstacle is in front or has been cleared
        if obstacle_ego is None:
            self._bypass_side = None
        else:
            ox_e, oy_e, r_e, vx_e, vy_e = obstacle_ego

            # Reset bypass memory if obstacle has successfully moved behind us
            if ox_e < -1.0:
                self._bypass_side = None

            # Latch bypass direction for consistent, stable avoidance:
            # If the obstacle is centered or slightly to the left, bypass on the left (side = 1.0)
            # to stay aligned with the CCW track flow, otherwise bypass on the right (side = -1.0)
            if self._bypass_side is None:
                self._bypass_side = 1.0 if oy_e < 0.1 else -1.0

        # KEY ADAPTIVE CONTROL: Scale down target velocity when approaching massive obstacles
        # to guarantee the vehicle slows down smoothly before making the evasive steering maneuver.
        if obstacle_ego is not None:
            ox_e, oy_e, r_e, _, _ = obstacle_ego
            dist_to_obs_edge = max(0.0, ox_e - r_e)
            # If the vehicle is close to a large roadblock, scale down reference speed
            if dist_to_obs_edge < self.speed_scale_distance and r_e > self.speed_scale_radius:
                speed_scaler = np.clip(
                    dist_to_obs_edge / self.speed_scale_distance, self.speed_scale_min, 1.0
                )
                xref_ego[2, :] *= speed_scaler

        # -------------------------------------------------------------------
        # 3. Update DPP tracking parameter values
        # -------------------------------------------------------------------
        cos_val = np.cos(xref_ego[3, :])
        sin_val = np.sin(xref_ego[3, :])
        self._cos_reference.value = cos_val
        self._sin_reference.value = sin_val
        self._along_reference.value = cos_val * xref_ego[0, :] + sin_val * xref_ego[1, :]
        self._cross_reference.value = -sin_val * xref_ego[0, :] + cos_val * xref_ego[1, :]
        self._velocity_reference.value = xref_ego[2, :]
        self._heading_reference.value = xref_ego[3, :]

        # Initial state in its own ego frame is always at the origin
        self._initial_state.value = np.array([0.0, 0.0, state[2], 0.0])
        self._last_command.value = self._u_last_applied.copy()

        # -------------------------------------------------------------------
        # 4. Process normal vectors in local frame (used for walls and obstacle fallback)
        # -------------------------------------------------------------------
        normals_ego = np.zeros_like(normals)
        normals_ego[:, 0] = normals[:, 0] * cos_theta - normals[:, 1] * sin_theta
        normals_ego[:, 1] = normals[:, 1] * cos_theta + normals[:, 0] * sin_theta

        # Wall constraints affect states k+1 (steps 1 to N+1)
        wall_nx = normals_ego[1 : self.N + 1, 0]
        wall_ny = normals_ego[1 : self.N + 1, 1]

        if self.enable_walls:
            self._wall_normal_x.value = wall_nx
            self._wall_normal_y.value = wall_ny
            self._wall_half_width.value = np.ones(self.N) * max(
                0.0, float(half_width) - self.wall_margin
            )

            # Precompute projections of the local reference path onto local normals
            ref_x = xref_ego[0, 1 : self.N + 1]
            ref_y = xref_ego[1, 1 : self.N + 1]
            self._wall_ref_proj.value = wall_nx * ref_x + wall_ny * ref_y
        else:
            self._wall_normal_x.value = np.zeros(self.N)
            self._wall_normal_y.value = np.zeros(self.N)
            self._wall_half_width.value = np.zeros(self.N)
            self._wall_ref_proj.value = np.zeros(self.N)

        # -------------------------------------------------------------------
        # 5. Iterative iMPC optimization loop
        # -------------------------------------------------------------------
        if self._X_prev is not None and self._X_prev.shape[1] == self.N + 1:
            x_guess = np.roll(self._X_prev, -1, axis=1)
            x_guess[:, -1] = self._X_prev[:, -1]
            u_guess = np.roll(self._u_prev, -1, axis=1)
            u_guess[:, -1] = self._u_prev[:, -1]
        else:
            x_guess = xref_ego.copy()
            u_guess = np.zeros((2, self.N))

        x_guess[:, 0] = np.array([0.0, 0.0, state[2], 0.0])

        solved = False
        for _ in range(self.max_iter):
            # Update discrete linear system parameters
            for k in range(self.N):
                x_bar = x_guess[:, k]
                u_bar = u_guess[:, k]
                A_k, B_k, C_k = self._compute_linear_model_matrices(x_bar, u_bar)
                self._A_params[k].value = A_k
                self._B_params[k].value = B_k
                self._C_params[k].value = C_k

            # Recalculate obstacle half-plane normal parameters dynamically around the current guess
            obs_nx = np.zeros(self.N)
            obs_ny = np.zeros(self.N)
            obs_dist = np.zeros(self.N)

            if obstacle_ego is not None:
                ox_e, oy_e, r_e, vx_e, vy_e = obstacle_ego
                for k in range(self.N):
                    pred_ox = ox_e + vx_e * k * self.dt
                    pred_oy = oy_e + vy_e * k * self.dt

                    p_guess = x_guess[:2, k + 1]
                    dx = p_guess[0] - pred_ox
                    dy = p_guess[1] - pred_oy
                    dist_rad = np.hypot(dx, dy)
                    dist_rad = dist_rad if dist_rad > 1e-5 else 1e-5

                    nx_rad = dx / dist_rad
                    ny_rad = dy / dist_rad

                    # Longitudinal distance to obstacle center
                    d_long = pred_ox - p_guess[0]

                    # KEY STRATEGY: Smooth continuous blend with the track perpendicular normal.
                    # We use a start blend distance of 10.0 meters so that the lateral steering
                    # gradient starts acting far ahead, easing steering changes significantly.
                    d_start = self.blend_distance
                    w = np.clip(1.0 - d_long / d_start, 0.0, 1.0)

                    # Determine bypass side consistently based on the latched bypass memory
                    side = self._bypass_side if self._bypass_side is not None else 1.0
                    nx_perp = wall_nx[k] * side
                    ny_perp = wall_ny[k] * side

                    # If the rotated track normal is too small, fallback to pure ego Y axis
                    if np.hypot(nx_perp, ny_perp) < 1e-3:
                        nx_perp = 0.0
                        ny_perp = side

                    # Blend the radial and perpendicular normals smoothly
                    nx = (1.0 - w) * nx_rad + w * nx_perp
                    ny = (1.0 - w) * ny_rad + w * ny_perp

                    # Normalize the blended normal
                    norm_val = np.hypot(nx, ny)
                    nx = nx / norm_val
                    ny = ny / norm_val

                    obs_nx[k] = nx
                    obs_ny[k] = ny
                    obs_dist[k] = nx * pred_ox + ny * pred_oy + r_e

                self._obstacle_normal_x.value = obs_nx
                self._obstacle_normal_y.value = obs_ny
                self._obstacle_safe_distance.value = obs_dist + (self.width / 2.0 + self.obs_margin)
            else:
                self._obstacle_normal_x.value = np.ones(self.N)
                self._obstacle_normal_y.value = np.zeros(self.N)
                self._obstacle_safe_distance.value = np.ones(self.N) * -1000.0

            # Warm start solver variables
            self._states.value = x_guess
            self._controls.value = u_guess

            try:
                if self.solver.upper() == "OSQP":
                    polishing_val = getattr(self, "_osqp_polish", False)
                    self._problem.solve(
                        solver=cp.OSQP,
                        warm_start=False,
                        verbose=False,
                        eps_abs=1e-3,
                        eps_rel=1e-3,
                        max_iter=4000,
                        polishing=polishing_val,
                    )
                else:  # CLARABEL
                    self._problem.solve(
                        solver=cp.CLARABEL,
                        warm_start=True,
                        verbose=False,
                        tol_gap_abs=1e-3,
                        tol_gap_rel=1e-3,
                        tol_feas=1e-3,
                    )
            except cp.SolverError as exc:
                warnings.warn(f"[HybridMPC] Solver error: {exc}. Falling back.")
                break

            status = self._problem.status
            if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self._states.value is None:
                warnings.warn(f"[HybridMPC] Solver status '{status}'. Falling back.")
                break

            X_sol = self._states.value
            U_sol = self._controls.value

            # Convergence criteria
            if np.max(np.abs(X_sol - x_guess)) < 1e-2:
                x_guess = X_sol
                u_guess = U_sol
                solved = True
                break

            x_guess = X_sol
            u_guess = U_sol
            solved = True

        if not solved or self._states.value is None:
            return self._fallback_control(ref)

        # Store finalized values for the next step's warm start
        self._X_prev = x_guess
        self._u_prev = u_guess
        self._u_last_applied = u_guess[:, 0]

        # -------------------------------------------------------------------
        # 6. Transform predicted local path back to global frame coordinates
        # -------------------------------------------------------------------
        predicted_xy_ego = x_guess[:2, :]
        cos_glob = np.cos(state[3])
        sin_glob = np.sin(state[3])

        predicted_xy = np.zeros((self.N + 1, 2))
        predicted_xy[:, 0] = (
            predicted_xy_ego[0, :] * cos_glob - predicted_xy_ego[1, :] * sin_glob + state[0]
        )
        predicted_xy[:, 1] = (
            predicted_xy_ego[0, :] * sin_glob + predicted_xy_ego[1, :] * cos_glob + state[1]
        )

        action = u_guess[:, 0].copy()
        return action, predicted_xy

    def solve(self, ref, boundary_normals, half_width, obstacles=None):
        """Standard interface compatibility wrapper."""
        state = ref[0].copy()
        action, predicted_xy = self.solve_control(
            state, ref, boundary_normals, half_width, obstacles
        )
        command = np.array(
            [
                predicted_xy[1, 0],
                predicted_xy[1, 1],
                ref[1, 2],
                ref[1, 3],
            ],
            dtype=float,
        )
        return command, predicted_xy

    def _fallback_control(self, ref):
        """Recovery logic returning last safe actions (Emergency braking)."""
        action = np.zeros(2)
        action[0] = self.a_min  # Apply full deceleration for safety
        action[1] = 0.0  # Maintain steering centered
        return action, ref[:, :2]
