"""Build and run container entries (Docker, or Podman with ``ASTORIDS_DOCKER=podman``).

An entry is a directory with a ``Dockerfile`` whose image serves one agent over
WebSocket on port 8080 (``protocol.py``). The image must not need the
internet once built: with ``internal=True`` the container runs on a network
with no outside access, and the runner talks to it at its own address.
"""

from __future__ import annotations

import os
import subprocess
import time

DOCKER = os.environ.get("ASTORIDS_DOCKER", "docker")
INTERNAL_NETWORK = "astorids-internal"
PORT = 8080


def _run(*args: str, timeout: float = 120.0, capture: bool = True) -> str:
    out = subprocess.run([DOCKER, *args], check=True, timeout=timeout, text=True,
                         stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    return (out.stdout or "").strip()


def build(context: str, tag: str, timeout: float = 1800.0, quiet: bool = True) -> str:
    """``docker build`` the entry in ``context`` as ``tag``; returns the tag. Raises with the log on failure."""
    try:
        _run("build", "-t", tag, context, timeout=timeout, capture=quiet)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"docker build failed:\n{(e.stdout or '')[-6000:]}") from None
    return tag


def internal_network(name: str = INTERNAL_NETWORK) -> str:
    """A bridge network without outside access (created if missing)."""
    if subprocess.run([DOCKER, "network", "inspect", name], capture_output=True).returncode != 0:
        _run("network", "create", "--internal", name)
    return name


class Container:
    """One running entry. Use as a context manager; ``url`` is where its agent server listens."""

    def __init__(self, image: str, internal: bool = False, memory: str = "2g", cpus: str = "1"):
        self.image, self.internal, self.memory, self.cpus = image, internal, memory, cpus
        self.id: str | None = None
        self.url: str | None = None

    def start(self, ready_timeout: float = 90.0) -> "Container":
        limits = ["--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "512"]
        if self.internal:
            net = internal_network()
            self.id = _run("run", "-d", "--rm", "--network", net, *limits, self.image)
            ip = _run("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self.id)
            self.url = f"ws://{ip}:{PORT}/ws"
        else:
            self.id = _run("run", "-d", "--rm", "-p", f"127.0.0.1::{PORT}", *limits, self.image)
            host_port = _run("port", self.id, f"{PORT}/tcp").splitlines()[0].rsplit(":", 1)[1]
            self.url = f"ws://127.0.0.1:{host_port}/ws"
        try:
            self._wait_ready(ready_timeout)
        except BaseException:
            self.stop()  # don't leave a broken entry running
            raise
        return self

    def _wait_ready(self, timeout: float) -> None:
        from .protocol import RemoteAgent
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            try:
                agent = RemoteAgent(self.url, connect_timeout=5.0, timeout=10.0)
                agent.get_agent_type()
                agent.close()
                return
            except Exception as e:  # noqa: BLE001 - not up yet
                last = e
                time.sleep(1.0)
        raise RuntimeError(f"the entry's agent server didn't answer on port {PORT} within {timeout:.0f} s"
                           f" ({last!r})\n{self.logs()}")

    def logs(self, tail: int = 60) -> str:
        if not self.id:
            return ""
        r = subprocess.run([DOCKER, "logs", "--tail", str(tail), self.id], capture_output=True, text=True)
        return (r.stdout + r.stderr)[-6000:]

    def stop(self) -> None:
        if self.id:
            subprocess.run([DOCKER, "rm", "-f", self.id], capture_output=True)
            self.id = None

    def __enter__(self) -> "Container":
        return self.start() if self.id is None else self

    def __exit__(self, *exc) -> None:
        self.stop()
