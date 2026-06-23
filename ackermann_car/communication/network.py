"""
communication/network.py

ZeroMQ socket communication wrappers for decoupled simulator-controller execution.
Uses a Request-Reply (REQ-REP) pattern to ensure synchronous step-by-step lockstep execution.
"""

from __future__ import annotations

import json
import logging

import zmq

logger = logging.getLogger("communication")


class SimulatorClient:
    """ZeroMQ REQ client running in the Simulator process.

    It sends the current state, reference trajectory, boundaries, and obstacles
    to the controller, and blocks until it receives the control command reply.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.address = f"tcp://{host}:{port}"

    def connect(self):
        """Connect to the controller REP server."""
        logger.info(f"Simulator connecting to controller at {self.address}...")
        self.socket.connect(self.address)

    def send_state_and_wait(
        self,
        state: list[float],
        ref: list[list[float]],
        normals: list[list[float]],
        half_width: float,
        obstacles: list[dict] | None = None,
    ) -> dict:
        """Send environment data and wait for the controller's control command.

        Parameters
        ----------
        state : list of 4 floats [x, y, v, theta]
        ref : list of shape (N+1, 4)
        normals : list of shape (N, 2)
        half_width : float
        obstacles : list of dicts, optional (each has 'x', 'y', 'r')

        Returns
        -------
        reply : dict containing:
            'action': [a, delta]
            'predicted_horizon': list of shape (N+1, 2)
        """
        payload = {
            "state": state,
            "ref": ref,
            "normals": normals,
            "half_width": half_width,
            "obstacles": obstacles or [],
        }
        # Send serialized JSON
        self.socket.send_string(json.dumps(payload))
        # Block until reply is received
        reply_str = self.socket.recv_string()
        return json.loads(reply_str)

    def close(self):
        """Close the socket and context."""
        self.socket.close()
        self.context.term()


class ControllerServer:
    """ZeroMQ REP server running in the Controller process.

    It waits for state requests from the simulator, solves the control problem,
    and replies immediately with the optimal control inputs and predicted horizon.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = f"tcp://{host}:{port}"

    def bind(self):
        """Bind to the address to listen for simulator clients."""
        logger.info(f"Controller server binding to {self.address}...")
        self.socket.bind(self.address)

    def receive_request(self) -> dict:
        """Receive environment data from the simulator.

        Returns
        -------
        request : dict containing state, ref, normals, half_width, obstacles
        """
        request_str = self.socket.recv_string()
        return json.loads(request_str)

    def send_reply(self, action: list[float], predicted_horizon: list[list[float]]):
        """Send the control command and predicted horizon back to the simulator.

        Parameters
        ----------
        action : list of 2 floats [a, delta]
        predicted_horizon : list of shape (N+1, 2)
        """
        payload = {"action": action, "predicted_horizon": predicted_horizon}
        self.socket.send_string(json.dumps(payload))

    def close(self):
        """Close the socket and context."""
        self.socket.close()
        self.context.term()
