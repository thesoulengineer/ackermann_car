"""
render_demos.py

Headless batch renderer for the README showcase: runs the MPC simulation on each
built-in track (oval, circuit, figure-eight) and saves an animated GIF per track
to ``docs/media/``.

The controller runs in-process (no ZeroMQ) for reliable batch rendering -- the
dynamics, controller, and visuals are identical to ``scripts/run.py``; only the
transport differs, which is invisible in the output video.
"""

from __future__ import annotations

import logging
import os

import matplotlib

# Force a headless backend BEFORE importing pyplot: no windows pop up and GIF
# saving works on any platform (incl. Windows, where run.py would open a GUI).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ackermann_car.controllers.hybrid_mpc import HybridMPCController  # noqa: E402
from ackermann_car.sim.car import KinematicBicycleModel  # noqa: E402
from ackermann_car.sim.lap_manager import LapManager  # noqa: E402
from ackermann_car.sim.obstacles import generate_obstacles  # noqa: E402
from ackermann_car.sim.track import Track  # noqa: E402
from ackermann_car.sim.visualizer import LiveView  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("render_demos")

# Controller horizon / integration step (matches run.py).
N = 10
DT = 0.1
# Safety cap so a stuck or pathological run can never hang the batch render.
MAX_STEPS = 2500
# Keep every Nth simulation frame and play back at FPS -> compact, smooth GIF.
SAMPLE_EVERY = 4
FPS = 12
SAVE_DPI = 80
OUT_DIR = os.path.join("docs", "media")

# (file stem, display title, track factory, RNG seed)
TRACKS = [
    ("oval", "Oval", lambda: Track.oval(n=100, width=15.0), 0),
    ("circuit", "Circuit", lambda: Track.circuit(), 1),
    ("figure_eight", "Figure-Eight", lambda: Track.figure_eight(), 2),
]


def render_track(name, title, track, seed):
    """Simulate one lap on ``track`` and save it as ``docs/media/<name>.gif``.

    Returns True if a full lap completed (False if the step cap was hit first).
    """
    rng = np.random.default_rng(seed)

    # Scale the obstacle radius to the track so the narrow circuit/figure-eight
    # stay passable; generate_obstacles requires radius < half_width.
    radius = max(0.8, round(0.3 * track.half_width, 1))
    obstacles = generate_obstacles(track, radius=radius, rng=rng)

    model = KinematicBicycleModel(wheelbase=0.3)
    lap_mgr = LapManager(track)
    controller = HybridMPCController(
        N=N, dt=DT, enable_walls=True, enable_obstacles=True, max_iter=2, solver="CLARABEL"
    )

    # Start on the centerline at the start/finish line.
    state = np.array([track.cx[0], track.cy[0], track.v_ref[0], track.theta[0]])

    view = LiveView(track)
    view.ax.set_title(f"MPC Simulation — {title} ({len(obstacles)} obstacles)")
    for i, obs in enumerate(obstacles):
        view.ax.add_patch(
            plt.Circle(
                (obs["x"], obs["y"]),
                obs["r"],
                color="crimson",
                alpha=0.6,
                zorder=5,
                label="obstacle" if i == 0 else "",
            )
        )

    frames = []
    sim_t = 0.0
    step = 0
    while lap_mgr.laps_completed < 1 and step < MAX_STEPS:
        ref = track.get_reference(state, N, DT)
        normals, half_width = track.get_boundary_data(track.last_index, N)
        action, predicted = controller.solve_control(state, ref, normals, half_width, obstacles)

        # Integrate dynamics, then add small reproducible process noise (mirrors run.py).
        state = model.step(state, action, DT)
        state = state + rng.normal(0.0, [0.02, 0.02, 0.01, 0.003])
        state[3] = (state[3] + np.pi) % (2 * np.pi) - np.pi

        lap_mgr.update(track.s[track.last_index], sim_t)
        sim_t += DT
        if step % SAMPLE_EVERY == 0:
            frames.append((state.copy(), predicted.copy(), lap_mgr.hud_info()))
        step += 1

    completed = lap_mgr.laps_completed >= 1
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.gif")
    anim = view.animate(frames, interval=int(1000 / FPS))
    anim.save(out_path, writer="pillow", fps=FPS, dpi=SAVE_DPI)
    plt.close(view.fig)

    size_kb = os.path.getsize(out_path) / 1024.0
    status = "lap completed" if completed else f"!! hit step cap ({MAX_STEPS})"
    logger.info(
        f"[{name}] steps={step} frames={len(frames)} obstacles={len(obstacles)} "
        f"({status}) -> {out_path} ({size_kb:.0f} KB)"
    )
    return completed


def main():
    results = {}
    for name, title, factory, seed in TRACKS:
        logger.info(f"Rendering '{name}' ...")
        results[name] = render_track(name, title, factory(), seed)

    logger.info("All renders done.")
    for name, ok in results.items():
        logger.info(f"  {name:12s} : {'OK (lap completed)' if ok else 'INCOMPLETE (hit step cap)'}")
    if not all(results.values()):
        logger.warning("Some tracks did not complete a lap -- consider tuning radius/MAX_STEPS.")


if __name__ == "__main__":
    main()
