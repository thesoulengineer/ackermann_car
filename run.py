"""
run.py - minimal integration harness for the MPC racetrack project.

Purpose: prove the ENVIRONMENT interfaces work end to end, and give teammates a
working harness to drop their real components into WITHOUT changing the
environment interface (see docs/INTERFACE.md).

  [REAL]        modules I own - the environment, under sim/:
                  Track (geometry + get_reference + get_boundary_data),
                  speed_profile (used inside Track), LapManager, LiveView.
  [PLACEHOLDER] stand-ins for teammates' work, to be replaced in place:
                  - vehicle dynamics / integrator  (Teammate A)
                  - MPC controller                 (Teammate B)

The control loop only ever calls the environment through its public interface,
so swapping a placeholder for the real thing requires no change to sim/.

Run:  .\.venv\Scripts\python.exe run.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sim.track import Track            # [REAL] geometry + reference/boundary lookup
from sim.lap_manager import LapManager  # [REAL] lap detection + timing
from sim.visualizer import LiveView     # [REAL] animated live view


# ===========================================================================
# [PLACEHOLDER] Teammate B's MPC controller.
# ---------------------------------------------------------------------------
# Real version: solve a CVXPY optimization over the N-step horizon, using the
# reference (ref) as the tracking target and the boundary normals + half_width
# as wall constraints, and return the first control plus the predicted
# trajectory it rolls out.
#
# Stub: ignore the optimization entirely and just command the next reference
# point (pure pursuit of the lookahead). It still RECEIVES the boundary data so
# this call site already matches the real controller's signature.
# ===========================================================================
def mpc_placeholder(ref, boundary_normals, half_width):
    """Return (command, predicted_horizon).

    command          : [x, y, v, theta] target for the car this step
    predicted_horizon: (N+1, 2) positions to draw as the 'MPC prediction'
    """
    command = ref[1] if len(ref) > 1 else ref[0]   # next reference point
    predicted = ref[:, :2]                          # stand-in prediction = ref path
    return command, predicted


# ===========================================================================
# [PLACEHOLDER] Teammate A's vehicle dynamics / integrator.
# ---------------------------------------------------------------------------
# Real version: integrate the Ackermann/bicycle model one dt under the
# controller's (acceleration, steering) command.
#
# Stub: a kinematic point dragged toward the commanded target point at the
# commanded speed. The environment does not care HOW the state evolves, only
# that a state is [x, y, v, theta]; replace this function with real dynamics.
# ===========================================================================
def step_car_placeholder(state, command, dt):
    """Kinematic placeholder: move toward command point at its speed."""
    x, y = state[0], state[1]
    tx, ty, tv = command[0], command[1], command[2]
    dx, dy = tx - x, ty - y
    dist = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx) if dist > 1e-9 else state[3]
    step = min(dist, tv * dt)
    return np.array([x + step * np.cos(heading),
                     y + step * np.sin(heading),
                     tv, heading])


def simulate(track, N=15, dt=0.1, n_laps=2, max_steps=100_000):
    """Generator of (car_state, predicted_horizon, lap_info) frames.

    One iteration is one control step. The loop is interface-pure: every call
    into the environment goes through the public Track / LapManager API.
    """
    lap_mgr = LapManager(track)                                    # [REAL]

    # [PLACEHOLDER] initial car state: on the centerline at start/finish.
    state = np.array([track.cx[0], track.cy[0],
                      track.v_ref[0], track.theta[0]])

    sim_t = 0.0
    steps = 0
    prev_laps = 0
    while lap_mgr.laps_completed < n_laps and steps < max_steps:
        # ---- [REAL] environment queries (the interface under test) --------
        ref = track.get_reference(state, N, dt)                   # (N+1, 4)
        normals, half_width = track.get_boundary_data(track.last_index, N)

        # ---- [PLACEHOLDER] controller -> command + predicted trajectory ---
        command, predicted = mpc_placeholder(ref, normals, half_width)

        # ---- [PLACEHOLDER] vehicle dynamics -> next state -----------------
        state = step_car_placeholder(state, command, dt)

        # ---- [REAL] lap timing off the car's arc-length position ----------
        s_now = track.s[track.last_index]
        lap_mgr.update(s_now, sim_t)
        sim_t += dt
        steps += 1

        if lap_mgr.laps_completed > prev_laps:
            prev_laps = lap_mgr.laps_completed
            print(f"  lap {lap_mgr.laps_completed} complete: "
                  f"{lap_mgr.last_lap_time:.2f} s")

        yield state, predicted, lap_mgr.hud_info()


def main():
    # ---- [REAL] build the environment: track + speed profile --------------
    track = Track.oval()
    _ = track.v_ref  # force the speed profile to build now (my speed_profile)
    print(f"track: oval, length {track.length:.1f} m, "
          f"v_ref {track.v_ref.min():.1f}-{track.v_ref.max():.1f} m/s")

    # ---- [REAL] live visualizer, driven by the simulation -----------------
    view = LiveView(track)
    view.ax.set_title("Integration harness - placeholder car + MPC on the oval")
    # Keep a reference to the animation so it is not garbage-collected.
    _anim = view.animate(simulate(track, N=15, dt=0.1, n_laps=2), interval=20)
    plt.show()


if __name__ == "__main__":
    main()
