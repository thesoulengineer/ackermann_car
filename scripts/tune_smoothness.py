"""
tune_smoothness.py

Smoothness benchmark + avoidance-knob sweep for HybridMPCController.

The earlier weight sweep showed the controller is intrinsically smooth (obstacle-
free) and that the roughness lives in obstacle avoidance. So this focuses on the
avoidance shaping knobs (blend distance, activation range, obstacle margin, speed
scaling), measured on the oval WITH obstacles, NOISE-FREE.

Per run we record the applied controls + poses and report:
  * steering-rate RMS, jerk RMS      -> smoothness (lower = smoother)
  * cross-track RMS                  -> how far off-line it swings
  * min clearance to any obstacle    -> SAFETY guardrail (must stay >= 0)
  * fallback count, lap time

Run:  .venv/Scripts/python scripts/tune_smoothness.py
"""

from __future__ import annotations

import warnings

import numpy as np

from ackermann_car.controllers.hybrid_mpc import HybridMPCController
from ackermann_car.sim.car import KinematicBicycleModel
from ackermann_car.sim.lap_manager import LapManager
from ackermann_car.sim.obstacles import generate_obstacles
from ackermann_car.sim.track import Track

DT = 0.1
MAX_STEPS = 2000
SOLVER = "CLARABEL"  # matches run.py

HEADER = (
    f"{'config':34s} {'N':>2s} {'lap':>4s} {'t[s]':>6s} "
    f"{'steerRMS':>9s} {'jerkRMS':>8s} {'cteRMS':>7s} {'clr':>6s} {'fb':>3s}"
)


def run_lap(track, *, N=10, controller_kwargs=None, obstacles=None, max_steps=MAX_STEPS):
    """Drive one noise-free lap (obstacles optional); return a metrics dict."""
    controller_kwargs = controller_kwargs or {}
    model = KinematicBicycleModel(wheelbase=0.3)
    lap_mgr = LapManager(track)
    controller = HybridMPCController(
        N=N,
        dt=DT,
        enable_walls=True,
        enable_obstacles=obstacles is not None,
        max_iter=2,
        solver=SOLVER,
        **controller_kwargs,
    )
    state = np.array([track.cx[0], track.cy[0], track.v_ref[0], track.theta[0]])

    accels, deltas, xs, ys = [], [], [], []
    fallbacks = 0
    min_clearance = float("inf")
    sim_t = 0.0
    step = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the controller warns on each fallback; we count instead
        while lap_mgr.laps_completed < 1 and step < max_steps:
            ref = track.get_reference(state, N, DT)
            normals, half_width = track.get_boundary_data(track.last_index, N)
            action, _ = controller.solve_control(state, ref, normals, half_width, obstacles)

            if np.isclose(action[0], controller.a_min) and np.isclose(action[1], 0.0):
                fallbacks += 1

            accels.append(float(action[0]))
            deltas.append(float(action[1]))
            state = model.step(state, action, DT)  # NO process noise
            xs.append(float(state[0]))
            ys.append(float(state[1]))

            if obstacles:
                clr = min(
                    np.hypot(state[0] - o["x"], state[1] - o["y"]) - o["r"] for o in obstacles
                )
                min_clearance = min(min_clearance, clr)

            lap_mgr.update(track.s[track.last_index], sim_t)
            sim_t += DT
            step += 1

    a = np.asarray(accels)
    d = np.asarray(deltas)
    da = np.diff(a) / DT
    dd = np.diff(d) / DT
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cte = np.array([np.min(np.hypot(track.cx - x, track.cy - y)) for x, y in zip(xs, ys)])

    def rms(arr):
        return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")

    return {
        "steps": step,
        "lap": lap_mgr.laps_completed >= 1,
        "lap_time": sim_t,
        "steer_rate_rms": rms(dd),
        "jerk_rms": rms(da),
        "cte_rms": rms(cte),
        "min_clearance": min_clearance,
        "fallbacks": fallbacks,
    }


def print_row(m):
    clr = m["min_clearance"]
    clr_s = f"{clr:6.2f}" if np.isfinite(clr) else "   inf"
    print(
        f"{m['label']:34s} {m['N']:2d} {str(m['lap']):>4s} {m['lap_time']:6.1f} "
        f"{m['steer_rate_rms']:9.4f} {m['jerk_rms']:8.3f} "
        f"{m['cte_rms']:7.3f} {clr_s} {m['fallbacks']:3d}"
    )


# Avoidance-shaping sweep (row 0 = baseline / current defaults).
AVOID_SWEEP = [
    ("baseline (defaults)", {}),
    ("blend=15", {"blend_distance": 15.0}),
    ("blend=20", {"blend_distance": 20.0}),
    ("blend=20, act=12", {"blend_distance": 20.0, "obstacle_activation_range": 12.0}),
    (
        "blend=20, act=12, smin=0.7",
        {"blend_distance": 20.0, "obstacle_activation_range": 12.0, "speed_scale_min": 0.7},
    ),
    ("act=15", {"obstacle_activation_range": 15.0}),
    ("margin=0.4", {"obs_margin": 0.4}),
    ("smin=0.7, sdist=8", {"speed_scale_min": 0.7, "speed_scale_distance": 8.0}),
]

N_FIXED = 10  # horizon held at the production value; we're tuning avoidance, not N


def main():
    track = Track.oval(n=100, width=15.0)

    # Intrinsic baseline (no obstacles) for context.
    m0 = run_lap(track, N=N_FIXED)
    m0["label"], m0["N"] = "intrinsic (no obstacles)", N_FIXED
    print("=== Reference: intrinsic smoothness (noise-free, no obstacles) ===")
    print(HEADER)
    print("-" * len(HEADER))
    print_row(m0)

    # Avoidance sweep, with obstacles (noise-free).
    obstacles = generate_obstacles(
        track, radius=max(0.8, round(0.3 * track.half_width, 1)), rng=np.random.default_rng(0)
    )
    print(f"\n=== Avoidance sweep: oval with {len(obstacles)} obstacles (noise-free) ===")
    print(HEADER)
    print("-" * len(HEADER))
    rows = []
    for label, kw in AVOID_SWEEP:
        m = run_lap(track, N=N_FIXED, controller_kwargs=kw, obstacles=obstacles)
        m["label"], m["N"], m["kwargs"] = label, N_FIXED, kw
        rows.append(m)
        print_row(m)

    base = rows[0]
    # Safety guardrail: keep clearance no worse than baseline (and non-negative).
    safe_floor = min(0.0, base["min_clearance"])
    cands = [r for r in rows[1:] if r["lap"] and r["min_clearance"] >= max(0.0, safe_floor)]

    def score(r):
        return r["steer_rate_rms"] / base["steer_rate_rms"] + r["jerk_rms"] / base["jerk_rms"]

    print()
    if cands:
        best = min(cands, key=score)
        print(
            f"SMOOTHEST safe avoidance config: '{best['label']}'  kwargs={best['kwargs']}\n"
            f"  steer-rate RMS : {base['steer_rate_rms']:.4f} -> {best['steer_rate_rms']:.4f}\n"
            f"  jerk RMS       : {base['jerk_rms']:.3f} -> {best['jerk_rms']:.3f}\n"
            f"  min clearance  : {base['min_clearance']:.2f} m -> {best['min_clearance']:.2f} m"
        )
    else:
        print("No avoidance config improved smoothness without losing clearance.")


if __name__ == "__main__":
    main()
