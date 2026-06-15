# Environment Interface Contract

This document defines the data formats that `sim/` exposes to the controller.
Implement against this spec — do not rely on implementation details.

---

## 1. State Vector

All state is represented as a 1-D array of length 4:

```
[x, y, v, theta]
```

| Index | Symbol | Unit | Description |
|-------|--------|------|-------------|
| 0 | `x` | m | Position, world X |
| 1 | `y` | m | Position, world Y |
| 2 | `v` | m/s | Longitudinal speed (scalar, ≥ 0) |
| 3 | `theta` | rad | Heading, measured from the +x axis, CCW positive (`atan2` convention) |

---

## 2. Reference Trajectory Array

`get_reference(state, N, dt)` returns a **numpy array of shape `(N+1, 4)`**.

- Row `0` is the current reference point (closest point on the centerline to `state`).
- Row `k` is the reference `k` steps ahead at timestep `dt`.
- Columns follow the state vector order: `[x_ref, y_ref, v_ref, theta_ref]`.
- `theta_ref` is **unwrapped** — no ±π discontinuities across the horizon.

The controller owns `N` and `dt` and passes them in on every call.
The environment never stores or assumes these values.

---

## 3. Boundary Data

Track boundaries are expressed as **left-pointing unit normals** on the centerline.

```python
normals    # numpy array, shape (M, 2) — one unit normal per centerline sample
half_width # float, metres — track half-width (same on both sides)
```

For a point `p` on the centerline with normal `n`, the left and right boundary
points are `p + half_width * n` and `p - half_width * n` respectively.

Wall constraints for the controller follow from: `|dot(car_pos - centerline_pos, n)| ≤ half_width`.

---

## 4. Ownership of N and dt

| Parameter | Owner | Rule |
|-----------|-------|------|
| `N` (horizon length) | Controller | Passed to `get_reference` each call |
| `dt` (timestep) | Controller | Passed to `get_reference` each call |

The environment never hard-codes `N` or `dt`. This keeps both sides in sync
without any shared configuration.

---

## 5. General Guarantees

- All outputs are **numpy arrays** (`numpy.ndarray`).
- All angles are in **radians**, always. No degrees, no mixed units.
- Return shapes are deterministic: the same query always returns the same shape.
- `normals` rows are unit vectors: `‖normals[i]‖ = 1` for all `i`.
