"""
controllers/mpc_controller.py

Model Predictive Controller for the Ackermann car project.

This module is Teammate B's (Jamiu's) deliverable.  It plugs directly into
run.py by replacing `mpc_placeholder` — the call signature is identical.

The controller receives the environment's public interface outputs:

    ref              (N+1, 4) reference trajectory from Track.get_reference
    boundary_normals (M, 2)   left-pointing unit normals from Track.get_boundary_data
    half_width       float    track half-width in metres

and returns:

    command          (4,)     [x, y, v, θ] target for Teammate A's integrator
    predicted_horizon(N+1, 2) predicted car positions over the horizon (for LiveView)

The maths behind every design choice lives in MPC_FORMULATION.md.

Dependencies:
    pip install cvxpy osqp numpy scipy
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import cvxpy as cp
from controllers.base_controller import BaseController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical / model constants
# ---------------------------------------------------------------------------

WHEELBASE = 0.3        # L  — distance between axles, metres (scale model car)

# ---------------------------------------------------------------------------
# Control limits  (see §4 of MPC_FORMULATION.md)
# ---------------------------------------------------------------------------

A_MAX     =  2.0       # maximum acceleration, m/s²
A_MIN     = -2.0       # maximum braking deceleration (negative), m/s²
DELTA_MAX =  0.5       # maximum steering angle, radians (~29°)

DA_MAX    =  0.5       # max change in acceleration per step  (slew rate)
DDELTA_MAX = 0.3       # max change in steering angle per step

SPEED_MIN =  0.0       # car cannot go backwards in this setup
SPEED_MAX =  3.5       # hard speed cap (slightly above v_ref max for headroom)

# ---------------------------------------------------------------------------
# MPC tuning weights  (see §3 of MPC_FORMULATION.md)
# ---------------------------------------------------------------------------

# Q — state tracking error weights: [X, Y, v, θ]
# We care most about position (X, Y), moderately about heading (θ), least speed.
Q_DIAG = np.array([10.0, 10.0, 1.0, 5.0])

# P — terminal cost weights: same structure as Q but larger to stabilise horizon.
# A common heuristic is P = Q * (N / 2), which we replicate here.
P_SCALE = 8.0
P_DIAG  = Q_DIAG * P_SCALE

# R — control effort weights: [a, δ]
# Larger values discourage large inputs.  δ is weighted more because large
# steering angles at speed are unsafe.
R_DIAG = np.array([0.1, 0.5])

# R_d — input-rate (smoothness) weights: [a, δ]
# Penalise rapid changes in throttle and steering for physical actuator health.
RD_DIAG = np.array([0.05, 0.1])

# Wall-constraint safety margin (metres).  The car must stay at least this far
# from the track boundary.
WALL_MARGIN = 0.05

# Maximum position error (metres) to clamp e_0 so the QP remains feasible
# when the car has drifted far off-track.
E0_CLIP_POS = 4.0   # clip x,y error to ±4 m
E0_CLIP_VEL = 2.0   # clip v error to ±2 m/s
E0_CLIP_ANG = 1.0   # clip theta error to ±1 rad


class MPCController(BaseController):
    """
    Receding-horizon MPC for the Ackermann car.

    Usage
    -----
    Construct once before the simulation loop, then call at every step:

        mpc = MPCController(N=15, dt=0.1)
        action, horizon = mpc.solve_control(state, ref, boundary_normals, half_width)

    The controller owns N and dt (it passes them to get_reference), matching
    the interface contract in INTERFACE.md §4.

    Parameters
    ----------
    N  : int   Prediction horizon length (number of steps).
    dt : float Timestep, seconds.

    Internals
    ---------
    The CVXPY problem is built once in __init__ and parameters are updated at
    each call to solve().  This avoids the overhead of re-constructing the
    problem every step (CVXPY compilation is slow; solving is fast).

    KEY DESIGN: e_0 (initial error state) is a CVXPY Parameter filled with
    the actual measured state error at each step.  This closes the feedback
    loop so the MPC corrects real deviations instead of assuming the car is
    always perfectly on the reference.
    """

    def __init__(self, N: int = 15, dt: float = 0.1):
        self.N  = N
        self.dt = dt

        # Build weight matrices from diagonal vectors.
        self.Q  = np.diag(Q_DIAG)
        self.P  = np.diag(P_DIAG)
        self.R  = np.diag(R_DIAG)
        self.Rd = np.diag(RD_DIAG)

        # Warm-start storage: previous solution shifted by one step.
        # Initialised to zeros; updated after each successful solve.
        self._u_prev: np.ndarray = np.zeros((N, 2))    # [[a_0, δ_0], …]
        self._u_last_applied: np.ndarray = np.zeros(2)  # [a, δ] from last step

        # Maximum number of obstacles supported simultaneously in the horizon
        self.M_max = 3

        # Build the CVXPY problem (once).
        self._build_problem()
        logger.info(f"MPCController ready (N={N}, dt={dt}, L={WHEELBASE} m)")

    # -----------------------------------------------------------------------
    # Public API — legacy interface (teammate compatibility)
    # -----------------------------------------------------------------------

    def solve(
        self,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem for one timestep and return the control output.

        Parameters
        ----------
        ref              : (N+1, 4)  reference trajectory from Track.get_reference
        boundary_normals : (M, 2)    unit normals from Track.get_boundary_data
        half_width       : float     track half-width in metres

        Returns
        -------
        command          : (4,)      [x, y, v, θ] — first predicted state after
                                     applying the optimal u_0.  This is the
                                     replacement for the old mpc_placeholder's
                                     `command` (run.py line: `command, predicted = …`).
        predicted_horizon: (N+1, 2)  predicted [x, y] positions over the horizon
                                     (for the LiveView MPC overlay).
        """
        ref = np.asarray(ref, dtype=float)

        # Use e_0=0 for legacy callers (assumes car is at reference)
        state_at_ref = ref[0].copy()
        A_list, B_list = self._linearise_along_ref(ref)
        self._load_parameters(state_at_ref, ref, A_list, B_list,
                              boundary_normals, half_width, obstacles=None)
        self._warm_start()

        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                                eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polish=True)
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
        command = np.array([predicted_xy[1, 0], predicted_xy[1, 1],
                            ref[1, 2], ref[1, 3]], dtype=float)
        return command, predicted_xy

    # -----------------------------------------------------------------------
    # Public API — decoupled ZMQ interface (state-feedback MPC)
    # -----------------------------------------------------------------------

    def solve_control(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        boundary_normals: np.ndarray,
        half_width: float,
        obstacles: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem for one timestep and return the control action [a, delta].

        This is the primary interface for decoupled execution over ZeroMQ.
        Unlike the legacy ``solve`` method, this properly closes the feedback loop
        by computing the initial error state e_0 = state − ref[0].

        Parameters
        ----------
        state            : (4,)      actual measured vehicle state [x, y, v, theta]
        ref              : (N+1, 4)  reference trajectory from Track.get_reference
        boundary_normals : (M, 2)    unit normals from Track.get_boundary_data
        half_width       : float     track half-width in metres
        obstacles        : list of dicts (unused in CVXPY; used for logging)

        Returns
        -------
        action           : (2,)      [a, delta] — optimal acceleration and steering angle.
        predicted_horizon: (N+1, 2)  predicted [x, y] positions over the horizon.
        """
        ref = np.asarray(ref, dtype=float)
        state = np.asarray(state, dtype=float)

        # Step 1: linearise dynamics along the reference trajectory
        A_list, B_list = self._linearise_along_ref(ref)

        # Step 2: load all CVXPY parameters — including actual e_0
        self._load_parameters(state, ref, A_list, B_list, boundary_normals, half_width, obstacles=obstacles)

        # Step 3: warm-start the solver
        self._warm_start()

        # Step 4: solve the QP
        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                                eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polish=True)
        except cp.SolverError as exc:
            warnings.warn(f"[MPCController] Solver error: {exc}. Falling back.")
            return self._fallback_control(ref)

        status = self._problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            warnings.warn(f"[MPCController] Solver status '{status}'. Falling back.")
            return self._fallback_control(ref)

        # Step 5: extract and return results
        u_opt = self._u_var.value
        e_opt = self._e_var.value
        if u_opt is None or e_opt is None:
            return self._fallback_control(ref)

        # Update warm-start cache (shift solution one step forward)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        self._u_last_applied = u_opt[0]

        # Absolute predicted positions = reference + error correction
        predicted_xy = ref[:, :2] + e_opt[:, :2]   # (N+1, 2)
        action = u_opt[0]                            # [a, delta] — first optimal input

        return action, predicted_xy

    # -----------------------------------------------------------------------
    # CVXPY problem construction (called once in __init__)
    # -----------------------------------------------------------------------

    def _build_problem(self):
        """
        Declare CVXPY variables and parameters, then assemble the QP.

        Variables (what the solver optimises over)
        ------------------------------------------
        e_var  (N+1, 4)  error states  e_k = x_k − x_ref_k
        u_var  (N,   2)  control inputs  u_k = [a_k, δ_k]

        Parameters (values injected at each solve)
        ------------------------------------------
        e0_par  (4,)      ACTUAL initial error state (closes the feedback loop)
        A_par  (N, 4, 4)  linearised state-transition Jacobians
        B_par  (N, 4, 2)  linearised input Jacobians
        n_par  (N, 2)     boundary normals (one per horizon step)
        w_par  scalar     effective half-width (half_width − WALL_MARGIN)
        u_prev_par (2,)   last applied input (for Δu rate constraint at k=0)
        """
        N = self.N

        # --- Decision variables ---
        self._e_var = cp.Variable((N + 1, 4), name="e")   # error states
        self._u_var = cp.Variable((N,     2), name="u")   # control inputs

        # --- Parameters (filled in _load_parameters at each solve) ---

        # Initial error state — set to actual (state - ref[0]) each step.
        # This is the key parameter that closes the feedback loop.
        self._e0_par = cp.Parameter(4, name="e0")

        # Linearisation matrices (one per step along the horizon).
        self._A_par = cp.Parameter((N * 4, 4), name="A")  # packed: row i*4:(i+1)*4
        self._B_par = cp.Parameter((N * 4, 2), name="B")  # same packing

        # Boundary normals and effective half-width.
        self._n_par  = cp.Parameter((N, 2), name="normals")
        self._w_par  = cp.Parameter(nonneg=True, name="half_width_eff")

        # Previous input (for Δu constraint at the first step).
        self._u_prev_par = cp.Parameter(2, name="u_prev")

        # Linear parameters for obstacle avoidance
        self._obs_A_par = cp.Parameter((N * self.M_max, 2), name="obs_A")
        self._obs_b_par = cp.Parameter(N * self.M_max, name="obs_b")

        # Slack variables for soft obstacle constraints
        self._obs_slack = cp.Variable(N * self.M_max, nonneg=True, name="obs_slack")

        # --- Build cost and constraints ---
        cost        = 0
        constraints = []

        # FEEDBACK: Initial error state = actual deviation from reference.
        # This replaces the old hard-coded `e_var[0] == 0` assumption.
        constraints += [self._e_var[0] == self._e0_par]

        for k in range(N):
            # Slice this step's A and B matrices out of the packed parameters.
            A_k = self._A_par[k * 4 : (k + 1) * 4, :]   # (4, 4)
            B_k = self._B_par[k * 4 : (k + 1) * 4, :]   # (4, 2)

            # --- Linearised dynamics constraint  e_{k+1} = A_k e_k + B_k u_k ---
            # (see §2 of MPC_FORMULATION.md)
            constraints += [
                self._e_var[k + 1] == A_k @ self._e_var[k] + B_k @ self._u_var[k]
            ]

            # --- Stage cost: tracking + effort + smoothness ---
            cost += cp.quad_form(self._e_var[k], self.Q)   # tracking error
            cost += cp.quad_form(self._u_var[k], self.R)   # control effort
            # Smoothness: penalise rapid input changes
            du = self._u_var[k] - (self._u_prev_par if k == 0 else self._u_var[k - 1])
            cost += cp.quad_form(du, self.Rd)

            # --- Input bound constraints ---
            constraints += [
                self._u_var[k, 0] >= A_MIN,
                self._u_var[k, 0] <= A_MAX,
                self._u_var[k, 1] >= -DELTA_MAX,
                self._u_var[k, 1] <=  DELTA_MAX,
            ]

            # --- Input-rate (slew-rate) constraints ---
            constraints += [
                du[0] >= -DA_MAX,
                du[0] <=  DA_MAX,
                du[1] >= -DDELTA_MAX,
                du[1] <=  DDELTA_MAX,
            ]

            # --- Wall / boundary constraints ---
            # |n_k^T e_pos| ≤ w_eff  (written as two linear inequalities)
            e_pos = self._e_var[k + 1, :2]
            n_k   = self._n_par[k, :]
            constraints += [
                 n_k @ e_pos <= self._w_par,
                -n_k @ e_pos <= self._w_par,
            ]

            # Obstacle constraints (linearized halfplanes)
            for j in range(self.M_max):
                idx = k * self.M_max + j
                constraints += [
                    self._obs_A_par[idx, :] @ e_pos <= self._obs_b_par[idx] + self._obs_slack[idx]
                ]
                cost += 100000.0 * self._obs_slack[idx] + 10000.0 * cp.square(self._obs_slack[idx])

        # --- Terminal cost  e_N^T P e_N ---
        cost += cp.quad_form(self._e_var[N], self.P)

        # --- Assemble the CVXPY problem ---
        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    # -----------------------------------------------------------------------
    # Helper: linearise the bicycle model along the reference
    # -----------------------------------------------------------------------

    def _linearise_along_ref(
        self,
        ref: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute linearised (A_k, B_k) matrices at each reference step.

        See §2 of MPC_FORMULATION.md for the derivation.

        Returns
        -------
        A_list : list of N (4, 4) arrays
        B_list : list of N (4, 2) arrays
        """
        N  = self.N
        dt = self.dt
        L  = WHEELBASE

        A_list = []
        B_list = []

        for k in range(N):
            v_r     = float(ref[k, 2])   # reference speed at step k
            theta_r = float(ref[k, 3])   # reference heading at step k

            # A_k = I + dt * ∂f/∂x  evaluated at (x_ref_k, u_ref_k)
            # (see MPC_FORMULATION.md §2 for the full Jacobian)
            #
            # With δ_ref ≈ 0, the 4th-row Jacobian entry (∂(dθ/dt)/∂v) = tan(0)/L = 0,
            # and ∂(dθ/dt)/∂θ = 0 as well, so the bottom-right 2×2 block is zero.
            dfdx = np.array([
                [0, 0,  np.cos(theta_r), -v_r * np.sin(theta_r)],
                [0, 0,  np.sin(theta_r),  v_r * np.cos(theta_r)],
                [0, 0,  0,                0                     ],
                [0, 0,  0,                0                     ],
            ])
            A_k = np.eye(4) + dt * dfdx

            # B_k = dt * ∂f/∂u  evaluated at (x_ref_k, u_ref_k)
            # Columns: [∂f/∂a, ∂f/∂δ]
            #
            # ∂X/∂a = 0, ∂X/∂δ = 0
            # ∂Y/∂a = 0, ∂Y/∂δ = 0
            # ∂v/∂a = 1, ∂v/∂δ = 0
            # ∂θ/∂a = 0, ∂θ/∂δ = v_r / L  (from the turn-rate equation with δ_ref=0,
            #             derivative of tan(δ) w.r.t. δ at 0 is sec²(0) = 1)
            dfdu = np.array([
                [0,          0        ],
                [0,          0        ],
                [1,          0        ],
                [0,  v_r / L if L > 1e-9 else 0.0],
            ])
            B_k = dt * dfdu

            A_list.append(A_k)
            B_list.append(B_k)

        return A_list, B_list

    # -----------------------------------------------------------------------
    # Helper: load CVXPY parameters for the current timestep
    # -----------------------------------------------------------------------

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
        """Pack all CVXPY parameter values from the current environment data."""
        N = self.N

        # --- Compute and set the initial error state e_0 = state - ref[0] ---
        # This is the feedback mechanism: the MPC now "knows" where the car
        # actually is relative to the reference, so it can correct real deviations.
        e0 = np.asarray(state, dtype=float) - ref[0]
        # Normalize the heading error to [-pi, pi]
        e0[3] = (e0[3] + np.pi) % (2 * np.pi) - np.pi
        # Clamp errors to keep the QP feasible even if car is far off-track
        e0[:2] = np.clip(e0[:2], -E0_CLIP_POS, E0_CLIP_POS)
        e0[2]  = np.clip(e0[2],  -E0_CLIP_VEL, E0_CLIP_VEL)
        e0[3]  = np.clip(e0[3],  -E0_CLIP_ANG, E0_CLIP_ANG)
        self._e0_par.value = e0

        # --- Pack A and B into the flat (N*4, 4) and (N*4, 2) parameter shapes. ---
        self._A_par.value = np.vstack(A_list)   # (N*4, 4)
        self._B_par.value = np.vstack(B_list)   # (N*4, 2)

        # --- Boundary normals: take the first N rows ---
        # Safety: normalise in case the environment doesn't guarantee unit length.
        normals_N = np.asarray(boundary_normals[:N], dtype=float)
        norms = np.linalg.norm(normals_N, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._n_par.value = normals_N / norms

        # Effective half-width after subtracting the safety margin.
        self._w_par.value = max(0.0, float(half_width) - WALL_MARGIN)

        # Previous applied input (for slew-rate constraint at k=0).
        self._u_prev_par.value = self._u_last_applied.copy()

        # Construction of linear obstacle constraints
        obs_A_val = np.zeros((N * self.M_max, 2))
        obs_b_val = np.ones(N * self.M_max) * 1e5  # Inactive constraint by default

        if obstacles is not None:
            for j, obs in enumerate(obstacles[:self.M_max]):
                obs_pos = np.array([obs["x"], obs["y"]])
                r_obs = obs["r"]
                # Car safety radius = 0.55m
                d_min = r_obs + 0.55

                for k in range(N):
                    # Linearization at the reference horizon k+1
                    p_ref = ref[k+1, :2]
                    n_k = self._n_par.value[k]  # Unit track normal at step k

                    # Vector from obstacle to reference position
                    dp = p_ref - obs_pos
                    d_lat = np.dot(n_k, dp)

                    # Determine avoidance direction (left or right)
                    lateral_dist = d_lat
                    if np.abs(lateral_dist) < 0.05:
                        # Obstacle is on the centerline, use actual state lateral deviation
                        lateral_dist = np.dot(n_k, state[:2] - obs_pos)
                    if np.abs(lateral_dist) < 0.05:
                        # Default to left if still close to zero
                        direction = 1.0
                    else:
                        direction = np.sign(lateral_dist)

                    u = direction * n_k
                    c = -u
                    r = np.abs(d_lat) - d_min

                    idx = k * self.M_max + j
                    obs_A_val[idx, :] = c
                    obs_b_val[idx] = r

        self._obs_A_par.value = obs_A_val
        self._obs_b_par.value = obs_b_val

    # -----------------------------------------------------------------------
    # Helper: warm-start the solver
    # -----------------------------------------------------------------------

    def _warm_start(self):
        """
        Initialise the CVXPY variable values with the shifted previous solution.

        Shifting: drop u_0 (just applied), keep u_1…u_{N-1}, repeat u_{N-1}
        as a neutral guess for the new last step.  This gives the solver a
        feasible (or near-feasible) starting point, which OSQP uses internally.
        """
        self._u_var.value = self._u_prev     # (N, 2) — shifted solution
        self._e_var.value = np.zeros((self.N + 1, 4))  # errors start near zero

    # -----------------------------------------------------------------------
    # Fallback: pure pursuit of next reference point
    # -----------------------------------------------------------------------

    def _fallback(
        self,
        ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return when the MPC solve fails (legacy `solve` method).

        Falls back to the same behaviour as the original mpc_placeholder:
        command the next reference point and return the reference path as the
        'predicted' horizon.  This keeps the simulation running safely.
        """
        command = ref[1] if len(ref) > 1 else ref[0]
        predicted = ref[:, :2]
        return np.asarray(command, dtype=float), np.asarray(predicted, dtype=float)

    def _fallback_control(self, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback control for solve_control when the MPC solve fails.

        Returns a neutral action (small positive acceleration, zero steering)
        to keep the car moving along the reference rather than braking to a stop.
        """
        # Apply small positive acceleration and zero steering to follow the reference
        action = np.array([0.5, 0.0], dtype=float)
        predicted = ref[:, :2]
        return action, predicted


# ---------------------------------------------------------------------------
# Module self-test: drop-in replacement check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick smoke-test: confirm the controller matches the mpc_placeholder
    signature expected by run.py and returns sensible shapes/types.

    Run from the project root with:
        python controllers/mpc_controller.py
    """
    import sys, os
    # Allow importing sim/ from the project root.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from sim.track import Track
    except ImportError:
        print("ERROR: could not import sim.track — run from the project root.")
        sys.exit(1)

    N, dt = 15, 0.1
    track = Track.oval()

    # Simulate a few steps with a fake state on the starting line.
    state = np.array([track.cx[0], track.cy[0], track.v_ref[0], track.theta[0]])
    mpc   = MPCController(N=N, dt=dt)

    print("MPCController smoke-test:")
    print(f"  N={N}, dt={dt}, wheelbase L={WHEELBASE} m")

    from sim.car import KinematicBicycleModel
    car = KinematicBicycleModel(wheelbase=WHEELBASE)

    for step in range(5):
        ref                      = track.get_reference(state, N, dt)
        boundary_normals, half_w = track.get_boundary_data(track.last_index, N)
        action, predicted        = mpc.solve_control(state, ref, boundary_normals, half_w)

        assert action.shape    == (2,),          f"action shape {action.shape}"
        assert predicted.shape == (N + 1, 2),    f"predicted shape {predicted.shape}"
        print(f"  step {step+1}: action=[a={action[0]:.3f}, δ={action[1]:.3f} rad]  "
              f"status={mpc._problem.status}")

        # Advance the state using the real kinematic model
        state = car.step(state, action, dt)

    print("  All shape/type checks passed.")
