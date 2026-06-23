# Model Predictive Control (MPC) Formulation

This document details the mathematical framework, dynamic equations, and optimization techniques used in the `HybridMPCController`.

---

## 1. Kinematic Bicycle Model

The vehicle's physical behavior is modeled using a 2D Kinematic Bicycle model, which assumes no lateral slip or tyre deformation. The state space is defined as $x = [x, y, v, \theta]^T$, and the control input space is $u = [a, \delta]^T$.

```
                 ▲ (Global Y)
                 │
                 │                 ▲ (Local Yaw/Heading θ)
                 │                /
                 │     Rear Axle /      Front Axle
                 │       (x, y) /──────/ (Steering δ)
                 │             /      /
                 │            /  L   /
                 │           /      /
                 │          /      /
─────────────────┴─────────/──────/─────────────────────► (Global X)
```

The continuous-time system equations are:
$$
\dot{x} = v \cos \theta
$$
$$
\dot{y} = v \sin \theta
$$
$$
\dot{v} = a
$$
$$
\dot{\theta} = \frac{v}{L} \tan \delta
$$

where $L$ represents the vehicle's wheelbase ($0.3\,\text{m}$).

---

## 2. Linearization and Discretization (LTV Model)

To solve the control problem in real time using convex Quadratic Programming (QP), we linearize the non-linear dynamics about a nominal state trajectory $\bar{x} = [\bar{x}, \bar{y}, \bar{v}, \bar{\theta}]^T$ and nominal control trajectory $\bar{u} = [\bar{a}, \bar{\delta}]^T$.

The continuous-time Jacobians $A_c$ and $B_c$ are calculated as:
$$
A_c = \frac{\partial f}{\partial x} = 
\begin{bmatrix}
0 & 0 & \cos\bar{\theta} & -\bar{v}\sin\bar{\theta} \\
0 & 0 & \sin\bar{\theta} & \bar{v}\cos\bar{\theta} \\
0 & 0 & 0 & 0 \\
0 & 0 & \frac{\tan\bar{\delta}}{L} & 0
\end{bmatrix}
$$

$$
B_c = \frac{\partial f}{\partial u} = 
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & \frac{\bar{v}}{L \cos^2\bar{\delta}}
\end{bmatrix}
$$

Using Euler integration with a step size $\Delta t$, we discretize the system to yield the Linear Time-Varying (LTV) dynamics:
$$
x_{k+1} = A_{d,k} x_k + B_{d,k} u_k + C_{d,k}
$$

where:
$$
A_{d,k} = I + \Delta t A_c
$$
$$
B_{d,k} = \Delta t B_c
$$
$$
C_{d,k} = \Delta t \left( f(\bar{x}_k, \bar{u}_k) - A_c \bar{x}_k - B_c \bar{u}_k \right)
$$

The affine term $C_{d,k}$ corrects for the offset introduced by the linearization point.

---

## 3. Local Ego-Frame Formulation

Linearizing global positions over sharp turns can lead to geometric errors. To prevent this, the optimization problem is formulated in the vehicle's local frame (Ego-Frame). 

At each time step:
1. The global vehicle state is mapped to the origin: $x_0 = [0, 0, v, 0]^T$.
2. The global reference path $\text{ref}$ and track boundary normals are rotated into this frame.
3. The LTV matrices are computed using local variables, avoiding angular wrapping issues around $\pm\pi$.
4. The resolved trajectory is transformed back to the global frame for visualization.

---

## 4. Cost Function Design

The objective function minimizes tracking errors, actuation effort, and control rate transitions:

$$
J = \sum_{k=0}^{N} \| e_k \|_Q^2 + \sum_{k=0}^{N-1} \left( \| u_k \|_R^2 + \| u_k - u_{k-1} \|_{R_d}^2 \right) + \sum_{k=0}^{N-1} \left( \rho S_k + \gamma S_k^2 \right)
$$

### Tracking Error $e_k$
Tracking errors are decomposed into longitudinal (along-track) and lateral (cross-track) components:
$$
e_k = 
\begin{bmatrix}
\cos(\theta_{\text{ref},k}) x_k + \sin(\theta_{\text{ref},k}) y_k - s_{\text{ref},k} \\
-\sin(\theta_{\text{ref},k}) x_k + \cos(\theta_{\text{ref},k}) y_k - n_{\text{ref},k} \\
v_k - v_{\text{ref},k} \\
\theta_k - \theta_{\text{ref},k}
\end{bmatrix}
$$

### Hard Constraints and Bounds
- **Velocity Limit:** $0.0 \le v_k \le 3.5\,\text{m/s}$
- **Acceleration Limit:** $-2.0 \le a_k \le 2.0\,\text{m/s}^2$
- **Steering Angle Limit:** $-0.5 \le \delta_k \le 0.5\,\text{rad}$
- **Slew Rate Limits:**
  $$
  | a_k - a_{k-1} | \le \dot{a}_{\max} \Delta t, \quad |\delta_k - \delta_{k-1}| \le \dot{\delta}_{\max} \Delta t
  $$

---

## 5. Collision Avoidance & Spatial Safety Constraints

### 5.1 Wall Boundary Constraints (Soft-Constrained)
To keep the vehicle within the track width $w_{\text{half}}$, we project the local position onto the local track normal $n = [n_x, n_y]^T$:
$$
n_x x_{k+1} + n_y y_{k+1} \le w_{\text{half}} - w_{\text{margin}} + S_k
$$
$$
- (n_x x_{k+1} + n_y y_{k+1}) \le w_{\text{half}} - w_{\text{margin}} + S_k
$$

where $S_k \ge 0$ is a slack variable heavily penalized in the objective function ($\rho = 10^4$) to prevent solver failure if the boundaries are breached.

### 5.2 Dynamic Obstacle Blended Normal Fields
For a circular obstacle located at $[o_x, o_y]^T$ with radius $r$:
1. A radial normal vector is defined: $n_{\text{rad}} = \frac{p_{\text{guess}} - o}{\|p_{\text{guess}} - o\|}$.
2. To avoid sudden steering corrections, we blend $n_{\text{rad}}$ with the track's lateral normal $n_{\text{perp}}$ using a weight $w$ based on the longitudinal distance $d_{\text{long}}$ to the obstacle:
   $$
   w = \max\left(0, 1 - \frac{d_{\text{long}}}{d_{\text{start}}}\right)
   $$
   $$
   n_{\text{blended}} = \text{Normalize}\left( (1-w)n_{\text{rad}} + w n_{\text{perp}} \right)
   $$
3. The collision avoidance constraint is then modeled as:
   $$
   n_{\text{blended}, x} x_{k+1} + n_{\text{blended}, y} y_{k+1} \ge \left( n_{\text{blended}, x} o_x + n_{\text{blended}, y} o_y + r + d_{\text{margin}} \right) - S_k
   $$

```
                           Track Wall (Left)
───────────────────────────────────────────────────────────────────────────
                       _.._
                     .'    `.
                    /  Obs   \   <--- (Bypass side latched dynamically)
                    | (ox,oy)|
                     \  r    /
                      `. __ .'
                                       __--- [Blended Normal Field]
                                   __--
                               __--
     Car Path              __--
    ─────────────►      _.-'
                      _.-'
                     /
───────────────────────────────────────────────────────────────────────────
                           Track Wall (Right)
```

### 5.3 Adaptive Reference Speed Scaling
When approaching large obstacles ($r > 1.2\,\text{m}$), the reference speed is scaled down to allow the controller to negotiate the obstacle safely:
$$
v_{\text{ref}, k} \leftarrow v_{\text{ref}, k} \times \text{clip}\left(\frac{d_{\text{edge}}}{5.0}, 0.5, 1.0\right)
$$
This ensures the vehicle brakes early before executing sudden lateral maneuvers.

### 5.4 Evasion Hysteresis Latching
To prevent lateral oscillations when an obstacle is centered, the bypass side is latched based on the obstacle's position relative to the local y-axis and the counter-clockwise flow of the racetrack:
- $\text{side} = 1.0$ (bypass on the left) if $y_{\text{obstacle}} < 0.1\,\text{m}$.
- $\text{side} = -1.0$ (bypass on the right) otherwise.
The latch remains active until the obstacle is passed ($x_{\text{obstacle}} < -1.0\,\text{m}$).

---

## 6. Solver Options and Performance Comparison

The controller supports two solvers: **OSQP** (an Operator Splitting QP solver based on ADMM) and **Clarabel** (an interior-point solver for conic optimization).

| Solver | Strengths | Weaknesses | Best Use Case |
| :--- | :--- | :--- | :--- |
| **OSQP** | Extremely fast iteration speeds; warm-starts effectively under small changes. | Can struggle with tight tolerances on highly constrained or ill-conditioned problems. | Real-world, high-frequency execution on standard tracks. |
| **Clarabel** | Robust convergence behavior; handles soft constraint slack margins reliably. | Higher computational cost per iteration; slower warm-start times. | Scenarios with massive blockages or complex safety limits. |

### Empirical Performance Comparison
Based on the unit tests and benchmarks:
- **Average Solve Time (Horizon $N=10$):**
  - OSQP: $\sim 3\text{ms}$ to $8\text{ms}$ per step.
  - Clarabel: $\sim 15\text{ms}$ to $25\text{ms}$ per step.
- **Robustness:** Clarabel achieves cleaner convergence when obstacles block the reference path entirely, activating the quadratic slack penalties without solver failure.
