"""Agents over WebSocket: the Planet Wars RTS JSON remote-invocation wire format.

An entry runs as a server (usually in a Docker container, on port 8080).
The match runner connects, creates an agent, calls its methods, and ends it.
Every message is one JSON text frame. Requests::

    {"requestType": "init",   "target": "agent", "className": "<anything>", "method": "<ignored>"}
    {"requestType": "invoke", "target": "agent", "objectId": "<id>", "method": "getAction", "args": [state]}
    {"requestType": "end",    "target": "agent", "objectId": "<id>", "method": ""}

and each gets exactly one response, in order::

    {"status": "ok", "result": {"objectId": "<id>"}}         # init
    {"status": "ok", "result": <the method's return value>}  # invoke
    {"status": "error", "error": "<message>"}

The methods are ``prepareToPlayAs(player, params, opponent)``,
``getAction(state)``, ``getAgentType()`` and ``processGameOver(finalState)``
(``agent.Agent``; ``state.py`` has the JSON). Any language with WebSocket and
JSON can implement it.

Needs the ``competition`` extra (``pip install -e ".[competition]"``, i.e. ``websockets``).
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Callable

from .agent import DEFAULT_OPPONENT, Agent

INIT, INVOKE, END = "init", "invoke", "end"
METHODS = {"getAction": "get_action", "getAgentType": "get_agent_type",
           "prepareToPlayAs": "prepare_to_play_as", "processGameOver": "process_game_over"}
DEFAULT_PORT = 8080
MAX_MESSAGE = 4 * 2**20  # bytes


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def request(request_type: str, method: str = "", object_id: str | None = None, args: list | None = None,
            class_name: str | None = None) -> dict:
    return {"requestType": request_type, "target": "agent", "className": class_name, "method": method,
            "objectId": object_id, "args": list(args or [])}


def _strip_type(arg):
    """Kotlin clients tag objects with a ``"type"`` class discriminator; agents don't need it."""
    if isinstance(arg, dict) and "type" in arg:
        return {k: v for k, v in arg.items() if k != "type"}
    return arg


# ---- server ---------------------------------------------------------------------------
class AgentServer:
    """Serves one agent class: every ``init`` creates a fresh agent from ``factory``.

    Each connection has its own agents and runs in its own thread, so a league
    can play several matches against one server at once.
    """

    def __init__(self, factory: Callable[[], Agent], host: str = "0.0.0.0", port: int = DEFAULT_PORT,
                 verbose: bool = False):
        self.factory, self.host, self.port, self.verbose = factory, host, port, verbose
        self._server = None

    def handle(self, message: str, agents: dict[str, Agent]) -> dict:
        """One request -> its response (never raises)."""
        try:
            req = json.loads(message)
            kind = req.get("requestType")
            if kind == INIT:
                agent_id = str(uuid.uuid4())
                agents[agent_id] = self.factory()
                return {"status": "ok", "result": {"objectId": agent_id}}
            if kind == INVOKE:
                agent = agents.get(req.get("objectId"))
                if agent is None:
                    raise ValueError(f"no such agent: {req.get('objectId')}")
                method = req.get("method", "")
                name = METHODS.get(method, camel_to_snake(method))
                if name not in METHODS.values():
                    raise ValueError(f"unknown method: {method}")
                result = getattr(agent, name)(*[_strip_type(a) for a in req.get("args") or []])
                return {"status": "ok", "result": result}
            if kind == END:
                removed = agents.pop(req.get("objectId"), None)
                return {"status": "ok", "result": {"message": "Agent removed" if removed else "No such agent"}}
            raise ValueError(f"unknown request type: {kind}")
        except Exception as e:  # report it to the caller, keep serving
            if self.verbose:
                print(f"error handling request: {e!r}", flush=True)
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    def _connection(self, ws) -> None:
        agents: dict[str, Agent] = {}
        for message in ws:
            ws.send(json.dumps(self.handle(message, agents)))

    def serve_forever(self) -> None:
        from websockets.sync.server import serve
        with serve(self._connection, self.host, self.port, compression=None, max_size=MAX_MESSAGE) as server:
            self._server = server
            print(f"agent server listening on ws://{self.host}:{self.port}", flush=True)
            server.serve_forever()

    def start(self) -> "AgentServer":
        """Serve from a background thread (``port=0`` picks a free port; read ``self.port`` after)."""
        from websockets.sync.server import serve
        self._server = serve(self._connection, self.host, self.port, compression=None, max_size=MAX_MESSAGE)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"ws://{host}:{self.port}/ws"


def serve(factory: Callable[[], Agent], host: str = "0.0.0.0", port: int = DEFAULT_PORT,
          verbose: bool = False) -> None:
    """Serve agents made by ``factory`` until killed (the entry point of a container)."""
    AgentServer(factory, host, port, verbose).serve_forever()


# ---- client ---------------------------------------------------------------------------
class RemoteError(RuntimeError):
    """The remote agent reported an error."""


def normalise_url(url: str) -> str:
    """``host:port`` or ``ws://host:port`` -> ``ws://host:port/ws`` (Planet Wars servers listen on ``/ws``)."""
    if "://" not in url:
        url = "ws://" + url
    scheme, rest = url.split("://", 1)
    return url if "/" in rest else f"{scheme}://{rest}/ws"


class RemoteAgent(Agent):
    """An agent on a server, called like a local one. Connects on first use.

    Calls block until the response arrives (``timeout`` seconds at most; the
    match runner enforces the real-time budget itself).
    """

    def __init__(self, url: str, class_name: str = "agent", timeout: float = 60.0, connect_timeout: float = 30.0):
        self.url = normalise_url(url)
        self.class_name, self.timeout, self.connect_timeout = class_name, timeout, connect_timeout
        self._ws = None
        self._id: str | None = None
        self._type: str | None = None

    def _call(self, req: dict):
        self._ws.send(json.dumps(req))
        resp = json.loads(self._ws.recv(timeout=self.timeout))
        if resp.get("status") != "ok":
            raise RemoteError(resp.get("error") or "remote error")
        return resp.get("result")

    def connect(self) -> "RemoteAgent":
        if self._ws is None:
            from websockets.sync.client import connect
            opts = dict(open_timeout=self.connect_timeout, compression=None, max_size=MAX_MESSAGE)
            try:  # agents are local: never go through an HTTP(S)_PROXY (websockets >= 15 would)
                self._ws = connect(self.url, proxy=None, **opts)
            except TypeError:  # websockets < 15 has no proxy support
                self._ws = connect(self.url, **opts)
            self._ws.__enter__()  # held open across calls (websockets >= 17.1 wants this said)
            self._id = self._call(request(INIT, "<ignored>", class_name=self.class_name))["objectId"]
        return self

    def invoke(self, method: str, *args):
        self.connect()
        return self._call(request(INVOKE, method, self._id, list(args)))

    def prepare_to_play_as(self, player: str, params: dict, opponent: str | None = DEFAULT_OPPONENT) -> str:
        self._type = str(self.invoke("prepareToPlayAs", player, params, opponent or DEFAULT_OPPONENT))
        return self._type

    def get_action(self, state: dict) -> dict:
        return self.invoke("getAction", state)

    def get_agent_type(self) -> str:
        if self._type is None:
            self._type = str(self.invoke("getAgentType"))
        return self._type

    def process_game_over(self, final_state: dict) -> None:
        self.invoke("processGameOver", final_state)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._call(request(END, "", self._id))
            except Exception:
                pass
            self._ws.close()
            self._ws = None
