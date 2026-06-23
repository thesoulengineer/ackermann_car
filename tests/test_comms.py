"""
tests/test_comms.py

Unit tests for the ZeroMQ socket communication layer and serialization.
"""

from __future__ import annotations

import threading
import time
import pytest
from ackermann_car.communication.network import SimulatorClient, ControllerServer


def test_zmq_communication_loop():
    """Verify that SimulatorClient and ControllerServer can establish a connection and exchange data correctly."""
    host = "127.0.0.1"
    port = 5557

    # Setup the controller server
    server = ControllerServer(host=host, port=port)
    server.bind()

    # Shared container to hold the request captured by the server
    received_request = {}

    def server_thread_run():
        try:
            req = server.receive_request()
            received_request.update(req)
            # Reply with mockup control action and horizon
            server.send_reply(
                action=[1.5, -0.2],
                predicted_horizon=[[1.0, 2.0], [2.0, 3.0]],
            )
        finally:
            server.close()

    # Start the server in a separate thread so it doesn't block the test
    t = threading.Thread(target=server_thread_run, daemon=True)
    t.start()

    # Allow ZMQ server socket a brief moment to bind
    time.sleep(0.2)

    # Initialize the client
    client = SimulatorClient(host=host, port=port)
    client.connect()

    # Test payload
    state = [1.0, 2.0, 3.0, 0.5]
    ref = [[1.0, 2.0, 3.0, 0.5], [1.1, 2.1, 3.0, 0.51]]
    normals = [[-0.5, 0.866], [-0.51, 0.86]]
    half_width = 4.0
    obstacles = [{"x": 10.0, "y": 20.0, "r": 1.5}]

    try:
        # Send data and wait for response
        reply = client.send_state_and_wait(
            state=state,
            ref=ref,
            normals=normals,
            half_width=half_width,
            obstacles=obstacles,
        )
    finally:
        # Ensure client socket is closed
        client.close()
        t.join(timeout=2.0)

    # Assertions on received data at the server side
    assert received_request["state"] == state
    assert received_request["ref"] == ref
    assert received_request["normals"] == normals
    assert received_request["half_width"] == half_width
    assert received_request["obstacles"] == obstacles

    # Assertions on received reply at the client side
    assert reply["action"] == [1.5, -0.2]
    assert reply["predicted_horizon"] == [[1.0, 2.0], [2.0, 3.0]]
