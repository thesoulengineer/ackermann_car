"""
tests/test_obstacles.py

Unit tests + comparison benchmark for the Hybrid MPC Controller.
Runs every test scenario for both OSQP and CLARABEL solvers to compare metrics.
Horizon N=10 for faster execution.
"""

from __future__ import annotations

import os
import time

import cvxpy as cp
import numpy as np
import pytest

from ackermann_car.controllers.hybrid_mpc import HybridMPCController


# ---------- Diagnostic helper ----------
def _print_diagnostics(mpc, solve_time, label="MPC solve"):
    """Print solver diagnostics and key metrics."""
    status = mpc._problem.status
    print(f"\n{label} ({mpc.solver}) diagnostics:")
    print(f"  Solver status  : {status}")
    print(f"  Solve time     : {solve_time:.4f} s")

    obj = mpc._problem.objective.value
    if obj is not None:
        print(f"  Objective value: {obj:.6f}")

    try:
        iters = mpc._problem.solver_stats.num_iters
        if iters is not None:
            print(f"  Solver iter    : {iters}")
    except AttributeError:
        pass

    slack = mpc._S_var.value
    if slack is not None:
        print(f"  Slack (min/mean/max): {slack.min():.3e} / {slack.mean():.3e} / {slack.max():.3e}")
        print(f"  Active slacks  : {np.sum(slack > 1e-6)} / {len(slack)}")

    if solve_time > 0.5:
        print("  ⚠️ Warning: solve time > 0.5 s – consider reducing N or tightening tolerances.")


# ---------- Parametrized Unit Tests (Runs for both OSQP and CLARABEL) ----------


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_walls_only(solver_name):
    """Verify that wall constraints keep the car within track boundaries."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=True, enable_obstacles=False, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 2.0
    state = ref[0]

    t0 = time.perf_counter()
    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=None,
    )
    elapsed = time.perf_counter() - t0

    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    max_y = np.max(np.abs(predicted[:, 1]))
    assert max_y <= half_width - mpc.wall_margin + 0.05

    _print_diagnostics(mpc, elapsed, "Walls‑only")


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_obstacles_only(solver_name):
    """Verify that obstacle constraint steers the car to the left of the obstacle."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=False, enable_obstacles=True, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 2.0
    obstacles = [{"x": 0.6, "y": 0.0, "r": 0.2}]
    state = ref[0]

    t0 = time.perf_counter()
    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=obstacles,
    )
    elapsed = time.perf_counter() - t0

    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    obstacle_idx = int(np.argmin(np.abs(ref[:, 0] - 0.6)))
    # Safe avoidance to the left (Y positive)
    assert predicted[obstacle_idx, 1] > 0.15

    _print_diagnostics(mpc, elapsed, "Obstacles‑only")


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_walls_and_obstacles(solver_name):
    """Combine walls and obstacles; the car should avoid both."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=True, enable_obstacles=True, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 2.0
    obstacles = [{"x": 0.6, "y": 0.0, "r": 0.2}]
    state = ref[0]

    t0 = time.perf_counter()
    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=obstacles,
    )
    elapsed = time.perf_counter() - t0

    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    max_y = np.max(np.abs(predicted[:, 1]))
    assert max_y <= half_width - mpc.wall_margin + 0.05

    obstacle_idx = int(np.argmin(np.abs(ref[:, 0] - 0.6)))
    assert predicted[obstacle_idx, 1] > 0.15

    _print_diagnostics(mpc, elapsed, "Walls + obstacles")


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_no_constraints_fallback(solver_name):
    """With no walls and no obstacles, the controller should track the reference."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=False, enable_obstacles=False, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 2.0
    state = ref[0]

    t0 = time.perf_counter()
    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=None,
    )
    elapsed = time.perf_counter() - t0

    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert np.allclose(predicted[:, 1], 0.0, atol=1e-4)

    _print_diagnostics(mpc, elapsed, "No constraints")


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_unavoidable_obstacle(solver_name):
    """Very large obstacle – slack should be activated."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=True, enable_obstacles=True, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 0.2
    obstacles = [{"x": 0.6, "y": 0.0, "r": 5.0}]
    state = ref[0]

    t0 = time.perf_counter()
    action, predicted = mpc.solve_control(
        state=state,
        ref=ref,
        boundary_normals=boundary_normals,
        half_width=half_width,
        obstacles=obstacles,
    )
    elapsed = time.perf_counter() - t0

    assert mpc._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    slack = mpc._S_var.value
    assert slack is not None
    assert np.any(slack > 0.1)

    _print_diagnostics(mpc, elapsed, "Unavoidable obstacle")


@pytest.mark.parametrize("solver_name", ["OSQP", "CLARABEL"])
def test_performance_stress(solver_name):
    """Run multiple solves and report average time."""
    N = 10
    dt = 0.1
    mpc = HybridMPCController(
        N=N, dt=dt, enable_walls=True, enable_obstacles=True, max_iter=2, solver=solver_name
    )

    ref = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, N + 1)])
    boundary_normals = np.array([[0.0, 1.0] for _ in range(N + 1)])
    half_width = 2.0
    obstacles = [{"x": 0.6, "y": 0.0, "r": 0.2}]
    state = ref[0]

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        mpc.solve_control(state, ref, boundary_normals, half_width, obstacles)
        times.append(time.perf_counter() - t0)

    avg_time = np.mean(times)
    max_time = np.max(times)
    print(f"\nPerformance stress test (5 solves under {solver_name}):")
    print(f"  Average solve time: {avg_time:.4f} s")
    print(f"  Max solve time    : {max_time:.4f} s")
    if avg_time > 0.3:
        print("  ⚠️ Average time > 0.3 s – consider optimizing (reduce N, or tune tolerances).")
    else:
        print("  ✅ Average time is within acceptable range.")


# ---------- Comparative Benchmark Table ----------
@pytest.mark.skipif(
    os.environ.get("RUN_COMPARISON") != "1",
    reason="Set RUN_COMPARISON=1 to run this extended benchmark",
)
def test_parameter_comparison():
    """Benchmark different solver settings and print a comparison table."""
    N_values = [10, 15]
    rho_values = [1e4, 1e5]
    solver_values = ["OSQP", "CLARABEL"]
    polishing_values = [False, True]  # OSQP specific
    results = []

    ref_full = np.array([[x, 0.0, 2.0, 0.0] for x in np.linspace(0.0, 1.5, max(N_values) + 1)])
    boundary_normals_full = np.array([[0.0, 1.0] for _ in range(max(N_values) + 1)])
    half_width = 2.0
    obstacles = [{"x": 0.6, "y": 0.0, "r": 0.2}]
    state = ref_full[0]

    for N in N_values:
        for rho in rho_values:
            for solver in solver_values:
                if solver == "CLARABEL":
                    polish_opts = [None]
                else:
                    polish_opts = polishing_values

                for polish in polish_opts:
                    try:
                        mpc = HybridMPCController(
                            N=N,
                            dt=0.1,
                            enable_walls=True,
                            enable_obstacles=True,
                            max_iter=2,
                            solver=solver,
                            rho=rho,
                        )

                        if solver == "OSQP" and polish is not None:
                            mpc._osqp_polish = polish
                        else:
                            mpc._osqp_polish = False

                        ref = ref_full[: N + 1]
                        boundary_normals = boundary_normals_full[: N + 1]

                        t0 = time.perf_counter()
                        action, predicted = mpc.solve_control(
                            state=state,
                            ref=ref,
                            boundary_normals=boundary_normals,
                            half_width=half_width,
                            obstacles=obstacles,
                        )
                        elapsed = time.perf_counter() - t0

                        status = mpc._problem.status
                        iters = (
                            mpc._problem.solver_stats.num_iters
                            if hasattr(mpc._problem.solver_stats, "num_iters")
                            else None
                        )
                        obj = mpc._problem.objective.value
                        slack = mpc._S_var.value
                        slack_mean = np.mean(slack) if slack is not None else np.nan
                        max_cross = np.max(np.abs(predicted[:, 1]))

                        results.append(
                            {
                                "N": N,
                                "rho": rho,
                                "solver": solver,
                                "polish": "on" if polish else "off",
                                "time": elapsed,
                                "status": status,
                                "iters": iters,
                                "objective": obj,
                                "slack_mean": slack_mean,
                                "max_cross": max_cross,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "N": N,
                                "rho": rho,
                                "solver": solver,
                                "polish": "on" if polish else "off",
                                "time": np.nan,
                                "status": f"ERROR: {str(e)[:50]}",
                                "iters": None,
                                "objective": None,
                                "slack_mean": None,
                                "max_cross": None,
                            }
                        )

    # Print Table
    print("\n" + "=" * 120)
    print("Solver Comparison Benchmark (walls + obstacles)")
    print("=" * 120)
    header = (
        f"{'N':>3} {'rho':>8} {'solver':>10} {'polish':>6}"
        f" {'time (s)':>10} {'iters':>6} {'status':>16}"
        f" {'obj':>12} {'slack_mean':>12} {'max_cross':>10}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        time_str = f"{r['time']:.4f}" if not np.isnan(r["time"]) else "FAIL"
        iters_str = str(r["iters"]) if r["iters"] is not None else "-"
        obj_str = f"{r['objective']:.1f}" if r["objective"] is not None else "-"
        slack_str = f"{r['slack_mean']:.3f}" if r["slack_mean"] is not None else "-"
        cross_str = f"{r['max_cross']:.3f}" if r["max_cross"] is not None else "-"
        print(
            f"{r['N']:3d} {r['rho']:8.0e} {r['solver']:10} {r['polish']:6}"
            f" {time_str:>10} {iters_str:>6} {r['status']:>16}"
            f" {obj_str:>12} {slack_str:>12} {cross_str:>10}"
        )
    print("=" * 120)
