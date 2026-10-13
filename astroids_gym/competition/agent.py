"""The competition agent interface, the same shape as Planet Wars RTS's ``PlanetWarsAgent``.

An agent flies a whole team. Before a match it is told which side it plays
(``"blue"`` or ``"red"``), the match's ``params`` and its opponent's name;
then, every decision step, it gets the full game ``state`` and returns one
action for its team; at the end it sees the final state. ``state.py``
describes the JSON of ``params``, ``state`` and actions.

Over the wire the methods are called ``prepareToPlayAs``, ``getAction``,
``getAgentType`` and ``processGameOver`` (see ``protocol.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..sim import Config, Simulation
from .state import TEAMS, config_from_params, restore, team_index

DEFAULT_OPPONENT = "Anon"


class Agent(ABC):
    """Fully observable competition agent."""

    @abstractmethod
    def get_action(self, state: dict) -> dict:
        """The team's action for this decision step: ``{"ships": [state.ship_command(...), ...]}``."""

    @abstractmethod
    def get_agent_type(self) -> str:
        """A short name for this agent."""

    def prepare_to_play_as(self, player: str, params: dict, opponent: str | None = DEFAULT_OPPONENT) -> str:
        """Called once before each match; returns the agent type."""
        return self.get_agent_type()

    def process_game_over(self, final_state: dict) -> None:
        """Called once after each match with the final state."""


class Player(Agent):
    """An :class:`Agent` that remembers its side and the match settings.

    Gives ``self.player`` (``"blue"`` or ``"red"``), ``self.team`` (0 or 1),
    ``self.opponent``, ``self.params``, ``self.config`` (the match's
    :class:`Config`), ``self.ship_ids`` (the ids of its own ships) and
    :meth:`simulation`, a forward model of any state.
    """

    def __init__(self):
        self.player = TEAMS[0]
        self.team = 0
        self.opponent = DEFAULT_OPPONENT
        self.params: dict = {}
        self.config = Config.competitive()

    def prepare_to_play_as(self, player: str, params: dict, opponent: str | None = DEFAULT_OPPONENT) -> str:
        self.player, self.team = player, team_index(player)
        self.opponent = opponent or DEFAULT_OPPONENT
        self.params, self.config = params, config_from_params(params)
        return self.get_agent_type()

    @property
    def ship_ids(self) -> range:
        ts = self.config.team_size
        return range(self.team * ts, (self.team + 1) * ts)

    @property
    def frame_skip(self) -> int:
        return int(self.params.get("frame_skip", 2))

    def simulation(self, state: dict, seed: int | None = None) -> Simulation:
        """``state`` as a :class:`Simulation` (see ``state.restore``)."""
        return restore(state, self.config, seed)
