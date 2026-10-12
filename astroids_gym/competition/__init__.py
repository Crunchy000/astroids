"""The astorids competition format, modelled on Planet Wars RTS.

* ``agent``: the agent interface (``prepareToPlayAs`` / ``getAction`` / ``processGameOver``)
* ``state``: match ``params``, game ``state`` and actions as JSON; ``restore`` a forward model
* ``agents``: built-in agents (scripted bots, trained models) and ``make_agent(spec)``
* ``params``: match settings drawn at random per match
* ``runner``: ``run_match``, real time or turn by turn
* ``protocol``: agents served over WebSocket (``AgentServer``, ``RemoteAgent``)
* ``league``: round-robin league with TrueSkill ratings and a Markdown leaderboard
* ``docker``: build and run container entries; ``submissions``: evaluate an entry from a GitHub issue

``python -m astroids_gym.competition --help`` runs agents, matches, leagues and evaluations.
"""

from .agent import Agent, Player
from .agents import BotAgent, DoNothingAgent, ModelAgent, RandomAgent, make_agent
from .params import VARIANTS, random_params
from .runner import MatchResult, run_match
from .state import (DO_NOTHING, TEAMS, config_from_params, make_params, parse_action, restore, ship_command,
                    snapshot)

__all__ = [
    "Agent", "Player", "BotAgent", "DoNothingAgent", "ModelAgent", "RandomAgent", "make_agent", "VARIANTS",
    "random_params", "MatchResult", "run_match", "DO_NOTHING", "TEAMS", "config_from_params", "make_params",
    "parse_action", "restore", "ship_command", "snapshot",
]
