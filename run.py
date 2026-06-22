"""
run.py

Unified entry point for the Ackermann car MPC racetrack simulation.
Can run the simulator client, the controller server, or spawn both concurrently.

Usage:
  python run.py                     # Runs both simulator and controller (default)
  python run.py --mode simulator    # Runs only the simulator client
  python run.py --mode controller   # Runs only the controller server
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import subprocess
import os
import numpy as np

# Configure headless matplotlib backend for Docker/CI execution
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.track import Track
from sim.lap_manager import LapManager
from sim.visualizer import LiveView
from sim.car import KinematicBicycleModel
from controllers.mpc_controller import MPCController
from communication.network import SimulatorClient, ControllerServer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run")

# Define racetrack obstacles
# These are placed directly on the drivable area of the oval track.
OBSTACLES = [
    {"x": 35.0, "y": 15.0, "r": 1.5},    # First curve segment
    {"x": 0.0, "y": 30.0, "r": 1.5},     # Top straight/arc
    {"x": -35.0, "y": -15.0, "r": 2.0},   # Far turn
]


def run_simulator(host: str, port: int):
    """Run the simulator client process.

    Connects to the controller server, integrates dynamics, updates
    lap timing, and renders the visualizer.
    """
    logger.info("Starting Simulator Client...")
    track = Track.oval()
    model = KinematicBicycleModel(wheelbase=0.3)
    lap_mgr = LapManager(track)

    # Initial state: on the centerline at start/finish line.
    state = np.array([track.cx[0], track.cy[0], track.v_ref[0], track.theta[0]])

    # Initialize ZeroMQ client
    client = SimulatorClient(host=host, port=port)
    client.connect()

    # Simulation parameters
    N = 15
    dt = 0.1
    sim_t = 0.0

    def simulate_generator():
        nonlocal state, sim_t
        while lap_mgr.laps_completed < 2:
            # Query track environment
            ref = track.get_reference(state, N, dt)
            normals, half_width = track.get_boundary_data(track.last_index, N)

            # Send state to controller and get control actions and predicted horizon
            try:
                reply = client.send_state_and_wait(
                    state=state.tolist(),
                    ref=ref.tolist(),
                    normals=normals.tolist(),
                    half_width=half_width,
                    obstacles=OBSTACLES,
                )
                action = np.array(reply["action"])
                predicted = np.array(reply["predicted_horizon"])
            except Exception as e:
                logger.error(f"Communication error: {e}. Stopping simulation.")
                break

            # Integrate vehicle physical dynamics (using actual inputs [a, delta])
            state = model.step(state, action, dt)

            # Injection of perturbation/noise to simulate real-world drift and deviations
            noise = np.array([
                np.random.normal(0, 0.03),   # X deviation
                np.random.normal(0, 0.03),   # Y deviation
                np.random.normal(0, 0.01),   # Speed fluctuation
                np.random.normal(0, 0.005),  # Heading fluctuation
            ])
            state += noise
            state[3] = (state[3] + np.pi) % (2 * np.pi) - np.pi

            # Update lap timing
            s_now = track.s[track.last_index]
            lap_mgr.update(s_now, sim_t)
            sim_t += dt

            yield state, predicted, lap_mgr.hud_info()

    # Render Visualizer
    view = LiveView(track)
    view.ax.set_title("Decoupled MPC simulation - ZeroMQ TCP Loop")

    # Draw obstacles as red circle patches on the visualizer axis
    for i, obs in enumerate(OBSTACLES):
        circle = plt.Circle(
            (obs["x"], obs["y"]),
            obs["r"],
            color="crimson",
            alpha=0.6,
            zorder=5,
            label="Obstacle" if i == 0 else "",
        )
        view.ax.add_patch(circle)

    if matplotlib.get_backend().lower() == "agg":
        logger.info("Headless environment detected. Generating dynamic GIF animation...")

        # Store sampled frames to generate the GIF
        frame_samples = []
        step = 0
        for frame_data in simulate_generator():
            state, predicted, hud = frame_data
            # Capture every 5 steps to avoid bloating the GIF size (2 Hz)
            if step % 5 == 0:
                frame_samples.append((state.copy(), predicted.copy(), hud.copy()))
            step += 1
            if step % 20 == 0:
                logger.info(
                    f"Step {step:03d} | State: x={state[0]:.2f}, y={state[1]:.2f}, v={state[2]:.2f} | "
                    f"Lap: {hud.get('lap', '-')}"
                )

        os.makedirs("output", exist_ok=True)
        gif_path = "output/simulation.gif"

        try:
            # Recreate the animation in memory and save it to disk using Pillow
            anim = view.animate(frame_samples, interval=500)  # 500ms per frame for 2 Hz
            anim.save(gif_path, writer="pillow", fps=2)
            logger.info(f"Success: Dynamic animation saved to: {gif_path}")
        except Exception as e:
            logger.error(f"Error saving GIF ({e}). Falling back to static PNG trajectory.")
            # Fallback to classic PNG
            fig, ax = plt.subplots(figsize=(9, 7))
            from sim.visualizer import draw_track
            draw_track(track, ax=ax, title="Simulation Trajectory (Fallback)")
            for obs in OBSTACLES:
                circle = plt.Circle((obs["x"], obs["y"]), obs["r"], color="crimson", alpha=0.6, zorder=5)
                ax.add_patch(circle)
            xs = [f[0][0] for f in frame_samples]
            ys = [f[0][1] for f in frame_samples]
            ax.plot(xs, ys, "-", color="tab:blue", lw=2, label="Actual path")
            ax.legend()
            fig.savefig("output/trajectory.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved static fallback plot to output/trajectory.png")
    else:
        logger.info("Simulation initialized. Starting animation...")
        # Keep reference to the animation object so it is not garbage collected
        _anim = view.animate(simulate_generator(), interval=20)
        plt.show()

    # Clean up sockets on visualizer exit
    client.close()
    logger.info("Simulator client stopped.")


def run_controller(host: str, port: int):
    """Run the controller server process.

    Binds the socket and solves MPC problems as they are requested.
    """
    logger.info("Starting Controller Server...")
    controller = MPCController(N=15, dt=0.1)

    server = ControllerServer(host=host, port=port)
    server.bind()

    logger.info("Controller server is ready. Waiting for simulator requests...")
    req_count = 0
    try:
        while True:
            request = server.receive_request()

            # Unpack all fields including actual vehicle state
            state   = np.array(request["state"])
            ref     = np.array(request["ref"])
            normals = np.array(request["normals"])
            half_width = float(request["half_width"])
            obstacles  = request.get("obstacles", [])

            # Solve MPC QP with state feedback (closed-loop)
            action, predicted = controller.solve_control(
                state, ref, normals, half_width, obstacles
            )

            # Reply with control action and visual horizon
            server.send_reply(action.tolist(), predicted.tolist())

            req_count += 1
            if req_count % 50 == 0:
                logger.info(
                    f"[Controller] Solved {req_count} steps | "
                    f"last action: a={action[0]:.3f}, δ={action[1]:.3f} rad"
                )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as e:
        logger.error(f"Controller server error: {e}")
    finally:
        server.close()
        logger.info("Controller server stopped.")


def run_both(host: str, port: int):
    """Spawns the controller process and runs the simulator in the main thread."""
    logger.info("Spawning decoupled controller and simulator processes...")

    # Spawn controller server subprocess
    # Uses the current python interpreter to run this same script with --mode controller
    controller_process = subprocess.Popen(
        [sys.executable, __file__, "--mode", "controller", "--host", host, "--port", str(port)]
    )

    # Let the controller server bind to the socket first
    time.sleep(0.8)

    try:
        # Run simulator client in the main thread
        run_simulator(host, port)
    finally:
        logger.info("Terminating controller process...")
        controller_process.terminate()
        controller_process.wait()
        logger.info("Controller process successfully terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ackermann Car Receding Horizon MPC Simulation")
    parser.add_argument(
        "--mode",
        choices=["simulator", "controller", "both"],
        default="both",
        help="Run mode: 'simulator', 'controller', or 'both' (default: both)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="ZeroMQ connection IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZeroMQ connection port (default: 5555)",
    )

    args = parser.parse_args()

    if args.mode == "both":
        run_both(args.host, args.port)
    elif args.mode == "simulator":
        run_simulator(args.host, args.port)
    elif args.mode == "controller":
        run_controller(args.host, args.port)
