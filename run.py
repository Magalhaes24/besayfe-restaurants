#!/usr/bin/env python
"""Start the FastAPI server on an available port."""

import socket
import subprocess
import sys


def find_available_port(start_port: int = 8000) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while port < 65535:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise RuntimeError("No available ports found")


if __name__ == "__main__":
    port = find_available_port()
    host = "127.0.0.1"

    print(f"\n{'='*60}")
    print(f"Starting Restaurant Menu Sheet Normalizer")
    print(f"{'='*60}")
    print(f"[SERVER] Running at: http://{host}:{port}")
    print(f"[DOCS]   API Docs:   http://{host}:{port}/docs")
    print(f"[CHECK]  Health:     http://{host}:{port}/health")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--reload",
    ]

    subprocess.run(cmd)
