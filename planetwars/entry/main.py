import asyncio
import os

from core.game_state import GameParams, Player
from core.game_state_factory import GameStateFactory
from client_server.game_agent_server import GameServerAgent


def make_agent():
    """The all-moves planner by default; PW_AGENT=neural serves the trained network instead."""
    if os.environ.get("PW_AGENT", "planner") == "neural":
        from agent import NeuralAgent, load
        load()  # fail fast if the model is missing
        return NeuralAgent()
    from planner import PlannerAgent
    return PlannerAgent()


if __name__ == "__main__":
    # warm up numpy so the first real move isn't slow
    warm = make_agent()
    warm.prepare_to_play_as(Player.Player1, GameParams(num_planets=30))
    for _ in range(3):
        warm.get_action(GameStateFactory(GameParams(num_planets=30)).create_game())
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=make_agent()).start())
