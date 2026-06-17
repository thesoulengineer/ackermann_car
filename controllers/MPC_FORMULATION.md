# MPC Formulation for the Ackermann Car Project

## Overview

Model Predictive Control (MPC) is a receding-horizon optimal control strategy.
At every timestep we solve a finite-horizon optimisation problem, apply only the
first control action, and re-solve at the next step using fresh measurements.
This "plan → act → replan" loop makes MPC robust to model mismatch and
disturbances — both very real on a physical car.

---

## 1. Vehicle Model (Kinematic Bicycle / Point-Mass)

We use a kinematic bicycle model. "Kinematic" means we ignore tyre forces and
treat the car as a rigid body that steers without slip. This is a good
approximation at low-to-moderate speeds (≤ 5 m/s), which matches our v_max.

### State vector  x = [X, Y, v, θ]

| Symbol | Meaning              | Unit  |
|--------|----------------------|-------|
| X      | World x-position     | m     |
| Y      | World y-position     | m     |
| v      | Longitudinal speed   | m/s   |
| θ      | Heading (from +x, CCW positive) | rad |

This is exactly the 4-element state vector defined in INTERFACE.md.

### Input vector  u = [a, δ]

| Symbol | Meaning              | Unit  |
|--------|----------------------|-------|
| a      | Longitudinal acceleration (throttle/brake) | m/s² |
| δ      | Front-wheel steering angle                 | rad  |

### Continuous-time equations of motion

    dX/dt  =  v · cos(θ)
    dY/dt  =  v · sin(θ)
    dv/dt  =  a
    dθ/dt  =  (v / L) · tan(δ)

where **L** is the wheelbase (distance between front and rear axle centres).

The turn-rate equation `dθ/dt = (v/L)·tan(δ)` comes from the no-slip
condition at each wheel: the instantaneous centre of rotation lies on the
rear axle extended, so `tan(δ) = L / R` where R is the turning radius, and
the angular rate is `v / R = v·tan(δ)/L`.

### Discrete-time (Euler, step dt)

    X_{k+1}  =  X_k  +  v_k · cos(θ_k) · dt
    Y_{k+1}  =  Y_k  +  v_k · sin(θ_k) · dt
    v_{k+1}  =  v_k  +  a_k · dt
    θ_{k+1}  =  θ_k  +  (v_k / L) · tan(δ_k) · dt

Euler is first-order accurate (error ∝ dt). For dt = 0.1 s this is fine.
If you switch to dt > 0.2 s, consider RK4.

---

## 2. Linearisation Around the Reference Trajectory

The bicycle model is nonlinear (cos/sin/tan). CVXPY can only solve convex
(quadratic) problems, so we **linearise** around the reference trajectory
provided by `Track.get_reference`.

Define the error state **e_k = x_k − x_ref_k** where x_ref comes from the
environment interface.

### Jacobian of f(x, u) w.r.t. x (at the reference operating point)

    A_k = I  +  dt · [
        0,  0,   cos(θ_ref_k),  −v_ref_k · sin(θ_ref_k),
        0,  0,   sin(θ_ref_k),   v_ref_k · cos(θ_ref_k),
        0,  0,   0,              0,
        0,  0,   tan(δ_ref) / L, 0
    ]

For the reference we take δ_ref ≈ 0 (small steering on the reference line),
so tan(δ_ref) ≈ 0 and the last row of the A matrix simplifies to all zeros.

### Jacobian of f(x, u) w.r.t. u (at the reference operating point)

    B_k = dt · [
        0,   0,
        0,   0,
        1,   0,
        0,   v_ref_k / L
    ]

The linearised error dynamics are then:

    e_{k+1} ≈ A_k · e_k  +  B_k · Δu_k

where **Δu_k = u_k − u_ref_k** is the deviation from the reference input.
On straights u_ref = [0, 0] (no acceleration, no steering), so Δu = u.

---

## 3. The Optimisation Problem

Over a horizon of N steps, we choose inputs u_0, …, u_{N-1} to minimise a
cost that penalises:

1. **Tracking error** — how far the state deviates from the reference.
2. **Control effort** — how large the inputs are (prevents aggressive moves).
3. **Input rate** — how fast the inputs change (prevents jerky behaviour).

### Cost function

    J = Σ_{k=0}^{N-1} [ e_k^T Q e_k  +  u_k^T R u_k  +  Δu_k^T R_d Δu_k ]
        + e_N^T P e_N

**Symbol key**

| Symbol | Meaning |
|--------|---------|
| Q      | (4×4) state-error weight matrix |
| R      | (2×2) control-effort weight matrix |
| R_d    | (2×2) input-rate (smoothness) weight matrix |
| P      | (4×4) terminal cost weight — rewards being close to reference at end of horizon |
| e_k    | State error at step k: [ΔX, ΔY, Δv, Δθ] |
| u_k    | Control input at step k: [a_k, δ_k] |
| Δu_k   | Change in input from previous step: u_k − u_{k-1} |

### Why each weight matters

- **Q**: penalises position and heading errors; a large Q forces the car to
  track the reference tightly. We typically weight X, Y errors more than v, θ.
- **R**: penalises large inputs; prevents the car from demanding extreme
  acceleration or steering.
- **R_d**: penalises rapid changes in inputs; results in smooth throttle and
  steering, which is important for physical actuators.
- **P**: the "terminal" weight stabilises the horizon; without it MPC can
  allow large errors to build up at the end of the horizon.

---

## 4. Constraints

### Input limits

    a_min ≤ a_k ≤ a_max         (acceleration bounds)
    −δ_max ≤ δ_k ≤ δ_max        (steering angle bounds)

### Input-rate limits (actuator slew rate)

    |a_k − a_{k-1}|  ≤  Δa_max
    |δ_k − δ_{k-1}|  ≤  Δδ_max

### Boundary / wall constraints

From `Track.get_boundary_data(index, N)` we get a unit normal **n_k** and
half-width **w** at each reference step. The wall constraint is:

    | n_k^T · (p_k − p_ref_k) |  ≤  w − d_margin

where p_k = [X_k, Y_k] is the car's predicted position and d_margin is a
small safety clearance. Rewritten as two linear inequalities:

     n_k^T · e_pos_k  ≤   w − d_margin
    −n_k^T · e_pos_k  ≤   w − d_margin

where e_pos_k = [ΔX_k, ΔY_k] are the first two components of e_k.
These are linear in the decision variable e_pos, so CVXPY handles them
directly.

---

## 5. CVXPY Problem Structure

CVXPY formulates the problem as a convex quadratic program (QP):

    minimise    (1/2) z^T H z  +  f^T z
    subject to  A_ineq z ≤ b_ineq

where z is the stacked vector of all error states and inputs over the horizon.

In practice we write the cost as a sum of `cp.quad_form(e, Q)` and
`cp.quad_form(u, R)` terms, and add constraints via `constraints += [...]`.
CVXPY automatically converts this to a QP and calls an appropriate solver
(OSQP by default — fast, reliable, warm-starts across timesteps).

---

## 6. Receding Horizon and Warm-Starting

After solving, we apply **only u_0** to the car. At the next timestep we:

1. Shift the previous solution by one step (drop u_0, copy u_1…u_{N-1}).
2. Use it as the initial guess for the new solve (warm-start).
3. Re-solve with the updated state measurement.

Warm-starting typically reduces solve time by 5–10× compared to cold starts,
which is critical for real-time performance.

---

## 7. Parameter Summary

| Parameter     | Symbol   | Typical value | Notes |
|---------------|----------|---------------|-------|
| Horizon       | N        | 15            | Passed in by controller at each call |
| Timestep      | dt       | 0.1 s         | Passed in by controller at each call |
| Wheelbase     | L        | 0.3 m         | Scale model car |
| v_max         | —        | 3.0 m/s       | From speed profile |
| a_max / a_min | —        | ±2.0 m/s²     | |
| δ_max         | —        | 0.5 rad (~29°)| |
| Δδ_max / Δa_max | —      | 0.3 / 0.5     | Per-step slew rate limits |
| Q (pos)       | —        | 10.0          | X, Y tracking weight |
| Q (heading)   | —        | 5.0           | θ tracking weight |
| Q (speed)     | —        | 1.0           | v tracking weight |
| R             | —        | [0.1, 0.5]    | a, δ effort weights |
| R_d           | —        | [0.05, 0.1]   | a, δ smoothness weights |

---

## 8. References

1. Rawlings, J. B., Mayne, D. Q., Diehl, M. M. — *Model Predictive Control:
   Theory, Computation, and Design* (2nd ed.), 2017.
2. Kong, J. et al. — "Kinematic and dynamic vehicle models for autonomous
   driving control design", IV 2015.
3. CVXPY documentation — https://www.cvxpy.org
4. OSQP solver — Stellato et al., "OSQP: An operator splitting solver for
   quadratic programs", Mathematical Programming Computation, 2020.
