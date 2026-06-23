# Environment-Controller Interface Specification

This document defines the interface and network contract between the **Simulator Process (ZMQ Client)** and the **Controller Process (ZMQ Server)**.

---

## 1. Network Protocol & Serialization

- **Socket Pattern:** Request-Reply (`ZMQ_REQ` on Simulator, `ZMQ_REP` on Controller) ensuring a strictly synchronized lockstep execution.
- **Port:** `5555` (Default, customizable via CLI parameters).
- **Format:** JSON-serialized strings.

---

## 2. Request Payload Schema (Simulator $\rightarrow$ Controller)

At every step of the simulation, the Simulator sends a single JSON object containing:

```json
{
  "state": [1.25, -0.52, 2.10, 0.45],
  "ref": [
    [1.25, -0.52, 2.10, 0.45],
    [1.46, -0.41, 2.15, 0.48],
    ...
  ],
  "normals": [
    [-0.23, 0.97],
    [-0.21, 0.98],
    ...
  ],
  "half_width": 5.0,
  "obstacles": [
    {"x": 0.0, "y": 30.0, "r": 2.5},
    {"x": -35.0, "y": -15.0, "r": 2.5}
  ]
}
```

### Field Definitions

| Field Name | JSON Type | Mathematical Shape | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `state` | List of floats | $4 \times 1$ | $[\text{m}, \text{m}, \text{m/s}, \text{rad}]$ | Current vehicle state: $[x, y, v, \theta]^T$. |
| `ref` | List of lists of floats | $(N+1) \times 4$ | $[\text{m}, \text{m}, \text{m/s}, \text{rad}]$ | Target reference path over the horizon. Elements are $[x_{ref}, y_{ref}, v_{ref}, \theta_{ref}]^T$. |
| `normals` | List of lists of floats | $(N+1) \times 2$ | Dimensional | Left-pointing unit normals $n = [n_x, n_y]^T$ mapped to each reference point. |
| `half_width` | Float | Scalar | $\text{m}$ | Track half-width (equal distance to left and right boundaries). |
| `obstacles` | List of dicts | Variable | $[\text{m}, \text{m}, \text{m}]$ | List of active circular obstacles. Each dict contains `"x"`, `"y"`, and `"r"` (radius). |

---

## 3. Reply Payload Schema (Controller $\rightarrow$ Simulator)

Upon resolving the mathematical optimization problem, the Controller returns a JSON object containing:

```json
{
  "action": [0.85, -0.12],
  "predicted_horizon": [
    [1.25, -0.52],
    [1.44, -0.42],
    ...
  ]
}
```

### Field Definitions

| Field Name | JSON Type | Mathematical Shape | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `action` | List of floats | $2 \times 1$ | $[\text{m/s}^2, \text{rad}]$ | Optimal control command $[a, \delta]^T$ (acceleration, steering angle) to be applied. |
| `predicted_horizon` | List of lists of floats | $(N+1) \times 2$ | $[\text{m}, \text{m}]$ | Predicted vehicle positions $[x_{pred}, y_{pred}]^T$ for visualization. |

---

## 4. Coordinate Reference Frame & Conversions

- **Global Frame:** Cartesian coordinates where the track's spatial properties (spline centerlines and boundary vertices) are defined. Angles are evaluated using the CCW convention starting from the $+x$ axis ($[-\pi, \pi]$).
- **Local Ego Frame:** Normalized coordinate space with the origin fixed at the vehicle's rear axle $[0, 0]^T$ and heading aligned along the current yaw angle $\theta_{\text{vehicle}}$.
- **Transformation Matrix:** Converting a global coordinate point $P_g = [X, Y]^T$ to the vehicle's Ego Frame $P_e = [x_e, y_e]^T$:
  $$
  \begin{bmatrix}
  x_e \\
  y_e
  \end{bmatrix}
  =
  \begin{bmatrix}
  \cos(-\theta) & -\sin(-\theta) \\
  \sin(-\theta) & \cos(-\theta)
  \end{bmatrix}
  \begin{bmatrix}
  X - X_{\text{car}} \\
  Y - Y_{\text{car}}
  \end{bmatrix}
  $$
- **Unwrapping Guarantee:** The heading reference $\theta_{\text{ref}}$ provided in the `ref` array is unwrapped locally relative to the vehicle's current heading. This prevents discontinuous jumps of $\pm2\pi$ across the horizon.