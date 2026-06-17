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

import warnings
import numpy as np
import cvxpy as cp


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


class MPCController:
    """
    Receding-horizon MPC for the Ackermann car.

    Usage
    -----
    Construct once before the simulation loop, then call at every step:

        mpc = MPCController(N=15, dt=0.1)
        command, horizon = mpc.solve(ref, boundary_normals, half_width)

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
        self._u_prev: np.ndarray = np.zeros((N, 2))   # [[a_0, δ_0], …]
        self._u_last_applied: np.ndarray = np.zeros(2) # [a, δ] from last step

        # Build the CVXPY problem (once).
        self._build_problem()

    # -----------------------------------------------------------------------
    # Public API
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
        ref = np.asarray(ref, dtype=float)   # ensure numpy

        # ------------------------------------------------------------------
        # Step 1: build the linearised A_k, B_k matrices along the reference.
        # See §2 of MPC_FORMULATION.md.
        # ------------------------------------------------------------------
        A_list, B_list = self._linearise_along_ref(ref)

        # ------------------------------------------------------------------
        # Step 2: load all CVXPY parameters for this solve.
        # ------------------------------------------------------------------
        self._load_parameters(ref, A_list, B_list, boundary_normals, half_width)

        # ------------------------------------------------------------------
        # Step 3: warm-start with the shifted previous solution.
        # ------------------------------------------------------------------
        self._warm_start()

        # ------------------------------------------------------------------
        # Step 4: solve.
        # ------------------------------------------------------------------
        try:
            self._problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                # OSQP settings: tolerance and iteration budget.
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                polish=True,           # extra accuracy on convergence
            )
        except cp.SolverError as exc:
            warnings.warn(f"[MPCController] Solver error: {exc}. Falling back.")
            return self._fallback(ref)

        status = self._problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            warnings.warn(f"[MPCController] Solver status: {status}. Falling back.")
            return self._fallback(ref)

        # ------------------------------------------------------------------
        # Step 5: extract results and update warm-start cache.
        # ------------------------------------------------------------------
        u_opt = self._u_var.value          # (N, 2)  [[a_0, δ_0], …]
        e_opt = self._e_var.value          # (N+1, 4) error states

        if u_opt is None or e_opt is None:
            return self._fallback(ref)

        # Save shifted solution for warm-start on the next call.
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        self._u_last_applied = u_opt[0]

        # Recover predicted absolute positions: p_ref + e_pos
        predicted_xy = ref[:, :2] + e_opt[:, :2]   # (N+1, 2)

        # The command is the predicted state at step 1 (after applying u_0).
        # This matches the mpc_placeholder convention that run.py expects.
        command = predicted_xy[1, 0], predicted_xy[1, 1], ref[1, 2], ref[1, 3]
        command = np.array(command, dtype=float)

        return command, predicted_xy

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
        # Linearisation matrices (one per step along the horizon).
        self._A_par = cp.Parameter((N * 4, 4), name="A")  # packed: row i*4:(i+1)*4
        self._B_par = cp.Parameter((N * 4, 2), name="B")  # same packing

        # Boundary normals and effective half-width.
        self._n_par  = cp.Parameter((N, 2), name="normals")
        self._w_par  = cp.Parameter(nonneg=True, name="half_width_eff")

        # Previous input (for Δu constraint at the first step).
        self._u_prev_par = cp.Parameter(2, name="u_prev")

        # --- Build cost and constraints ---
        cost        = 0
        constraints = []

        # Initial error state is zero: we are expanding around the reference,
        # so the current measured state equals the reference at k=0.
        # (In a real deployment you would set e_0 = x_measured − x_ref_0.)
        constraints += [self._e_var[0] == 0]

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
            # Tracking error  e_k^T Q e_k
            cost += cp.quad_form(self._e_var[k], self.Q)
            # Control effort  u_k^T R u_k
            cost += cp.quad_form(self._u_var[k], self.R)
            # Smoothness (Δu)^T R_d (Δu) — rate of change of input
            if k == 0:
                delta_u = self._u_var[k] - self._u_prev_par
            else:
                delta_u = self._u_var[k] - self._u_var[k - 1]
            cost += cp.quad_form(delta_u, self.Rd)

            # --- Input bound constraints ---
            constraints += [
                self._u_var[k, 0] >= A_MIN,          # a_min ≤ a_k
                self._u_var[k, 0] <= A_MAX,           # a_k ≤ a_max
                self._u_var[k, 1] >= -DELTA_MAX,      # −δ_max ≤ δ_k
                self._u_var[k, 1] <=  DELTA_MAX,      # δ_k ≤ δ_max
            ]

            # --- Input-rate (slew-rate) constraints ---
            if k == 0:
                du = self._u_var[k] - self._u_prev_par
            else:
                du = self._u_var[k] - self._u_var[k - 1]
            constraints += [
                du[0] >= -DA_MAX,
                du[0] <=  DA_MAX,
                du[1] >= -DDELTA_MAX,
                du[1] <=  DDELTA_MAX,
            ]

            # --- Wall / boundary constraints ---
            # | n_k^T (p_k − p_ref_k) | ≤ w_eff
            # where e_pos_k = e_var[k+1, 0:2]  (position error at next step)
            # Written as two linear inequalities (see §4 of MPC_FORMULATION.md).
            e_pos = self._e_var[k + 1, :2]
            n_k   = self._n_par[k, :]
            constraints += [
                 n_k @ e_pos <= self._w_par,
                -n_k @ e_pos <= self._w_par,
            ]

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
        ref: np.ndarray,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        boundary_normals: np.ndarray,
        half_width: float,
    ):
        """Pack all CVXPY parameter values from the current environment data."""
        N = self.N

        # Pack A and B into the flat (N*4, 4) and (N*4, 2) parameter shapes.
        A_packed = np.vstack(A_list)   # (N*4, 4)
        B_packed = np.vstack(B_list)   # (N*4, 2)
        self._A_par.value = A_packed
        self._B_par.value = B_packed

        # Boundary normals: take the first N normals (boundary_normals has M rows,
        # where M ≥ N since get_boundary_data is called with the same N).
        normals_N = np.asarray(boundary_normals[:N], dtype=float)
        # Safety: normalise in case the environment doesn't guarantee unit length.
        norms = np.linalg.norm(normals_N, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._n_par.value = normals_N / norms

        # Effective half-width after subtracting the safety margin.
        self._w_par.value = max(0.0, float(half_width) - WALL_MARGIN)

        # Previous applied input (for slew-rate constraint at k=0).
        self._u_prev_par.value = self._u_last_applied.copy()

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
        Return when the MPC solve fails.

        Falls back to the same behaviour as the original mpc_placeholder:
        command the next reference point and return the reference path as the
        'predicted' horizon.  This keeps the simulation running safely.
        """
        command = ref[1] if len(ref) > 1 else ref[0]
        predicted = ref[:, :2]
        return np.asarray(command, dtype=float), np.asarray(predicted, dtype=float)


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

    for step in range(5):
        ref                      = track.get_reference(state, N, dt)
        boundary_normals, half_w = track.get_boundary_data(track.last_index, N)
        command, predicted       = mpc.solve(ref, boundary_normals, half_w)

        assert command.shape   == (4,),          f"command shape {command.shape}"
        assert predicted.shape == (N + 1, 2),    f"predicted shape {predicted.shape}"
        print(f"  step {step+1}: command=[{command[0]:.2f}, {command[1]:.2f}, "
              f"{command[2]:.2f} m/s, {np.degrees(command[3]):.1f}°]  "
              f"status={mpc._problem.status}")

        # Advance the fake state toward the commanded point (same as run.py).
        tx, ty, tv = command[0], command[1], command[2]
        dx, dy = tx - state[0], ty - state[1]
        dist   = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx) if dist > 1e-9 else state[3]
        step_  = min(dist, tv * dt)
        state  = np.array([state[0] + step_ * np.cos(heading),
                           state[1] + step_ * np.sin(heading),
                           tv, heading])

    print("  All shape/type checks passed.")
