"""Built-in competition agents, and :func:`make_agent` to build any agent from a spec string.

Specs (the same names the league, the gauntlet and the CLI use):

* ``bots`` or ``bots:<style>``: the scripted bot (``astroids_gym.bot``), any style
* ``idle``: does nothing; ``random``: random commands
* ``path/to/model.npz``: a trained team exported to NumPy (``league.export_policy``);
  an SB3 ``.zip`` works too when stable-baselines3 is installed
* ``ws://host:port`` (or ``host:port``): an agent served over WebSocket (``protocol.RemoteAgent``)
* ``package.module:ClassName``: any :class:`~astroids_gym.competition.agent.Agent` subclass
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

import numpy as np

from ..bot import scripted_action
from ..env import TURN_LEVELS
from ..obs import encode_battle
from .agent import Agent, Player
from .state import DO_NOTHING, ship_command


class DoNothingAgent(Player):
    def get_action(self, state: dict) -> dict:
        return DO_NOTHING

    def get_agent_type(self) -> str:
        return "idle"


class RandomAgent(Player):
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> dict:
        r = self.rng
        return {"ships": [ship_command(i, r.uniform(-1, 1), r.random() < 0.5, r.random() < 0.3)
                          for i in self.ship_ids]}

    def get_agent_type(self) -> str:
        return "random"


class BotAgent(Player):
    """The built-in scripted bot as a competition agent.

    The bot gives each ship a random personality when it spawns (how keen it is
    to close in, its preferred range) and lets it wander when no enemy is in
    sight. Those aren't part of the game state, so this agent keeps them itself,
    redrawing them whenever one of its ships respawns, and uses its own random
    generator. It flies like the in-game bots, though not shot for shot.
    """

    def __init__(self, style: str | None = None, seed: int | None = None):
        super().__init__()
        self.style = style
        self.rng = np.random.default_rng(seed)
        self.memory: dict[int, list] = {}  # ship id -> [deaths, wander, aggro, pref_dist, target id]

    def prepare_to_play_as(self, player, params, opponent=None) -> str:
        self.memory = {}
        return super().prepare_to_play_as(player, params, opponent)

    def get_action(self, state: dict) -> dict:
        sim = self.simulation(state)
        sim.rng = self.rng
        commands = []
        for i in self.ship_ids:
            s = sim.ships[i]
            if not s.alive:
                continue
            m = self.memory.get(i)
            if m is None or m[0] != s.deaths:  # new ship (or respawned): a new personality
                m = self.memory[i] = [s.deaths, sim.rand(-1, 1), sim.rand(0.6, 1.0), sim.rand(140, 260), None]
            s.wander, s.aggro, s.pref_dist = m[1], m[2], m[3]
            s.target = sim.ships[m[4]] if m[4] is not None else None
            turn, thrust, fire = scripted_action(sim, s, self.frame_skip, self.style)
            m[1], m[4] = s.wander, (s.target.id if s.target is not None else None)
            commands.append(ship_command(i, turn, thrust, fire))
        return {"ships": commands}

    def get_agent_type(self) -> str:
        return f"bots:{self.style}" if self.style else "bots"


class ModelAgent(Player):
    """A trained team (one shared network flies every ship from its own observation), greedy actions."""

    def __init__(self, path: str, name: str | None = None):
        super().__init__()
        from ..numpy_policy import NumpyPolicy
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.policy = NumpyPolicy(path)
        self.obs_config = self.policy.obs_config
        self._warm = False

    def prepare_to_play_as(self, player, params, opponent=None) -> str:
        out = super().prepare_to_play_as(player, params, opponent)
        if not self._warm:  # compile the numba encoder now rather than on the clock
            from ..sim import Simulation
            sim = Simulation(self.config, seed=0)
            self.policy(encode_battle(sim, list(self.ship_ids), self.obs_config))
            self._warm = True
        return out

    def get_action(self, state: dict) -> dict:
        sim = self.simulation(state)
        ids = [i for i in self.ship_ids if sim.ships[i].alive]
        if not ids:
            return DO_NOTHING
        acts = self.policy(encode_battle(sim, ids, self.obs_config))
        return {"ships": [ship_command(i, TURN_LEVELS[int(a[0])], a[1], a[2]) for i, a in zip(ids, acts)]}

    def get_agent_type(self) -> str:
        return self.name


def _export_zip(path: str) -> str:
    """An SB3 ``.zip`` -> a cached ``.npz`` export (needs stable-baselines3 and torch)."""
    st = os.stat(path)
    cache = os.path.join(tempfile.gettempdir(), "astorids-models")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, f"{os.path.splitext(os.path.basename(path))[0]}-{st.st_size}-{int(st.st_mtime)}.npz")
    if os.path.exists(out):
        return out
    from stable_baselines3 import PPO

    from ..numpy_policy import export_policy
    return export_policy(PPO.load(path, device="cpu"), out)


def make_agent(spec: str) -> Agent:
    """An agent from a spec string (see the module docstring)."""
    if spec == "bots" or spec.startswith("bots:"):
        return BotAgent(spec.split(":", 1)[1] if ":" in spec else None)
    if spec == "idle":
        return DoNothingAgent()
    if spec == "random":
        return RandomAgent()
    if spec.startswith(("ws://", "wss://")):
        from .protocol import RemoteAgent
        return RemoteAgent(spec)
    if spec.endswith(".npz"):
        return ModelAgent(spec)
    if spec.endswith(".zip"):
        return ModelAgent(_export_zip(spec), os.path.splitext(os.path.basename(spec))[0])
    if ":" in spec and not os.path.exists(spec):
        mod, _, name = spec.partition(":")
        if mod.replace(".", "").replace("_", "").isalnum() and name.isidentifier():
            try:
                module = importlib.import_module(mod)
            except ModuleNotFoundError:  # a module beside the caller, as ``python -m`` would find it
                if os.getcwd() in sys.path:
                    raise
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(mod)
            return getattr(module, name)()
        from .protocol import RemoteAgent  # host:port
        return RemoteAgent(spec)
    raise ValueError(f"unknown agent spec {spec!r}")
