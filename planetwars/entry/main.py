import asyncio

from agent import NeuralAgent, load
from client_server.game_agent_server import GameServerAgent
from core.game_state import GameParams, Player
from core.game_state_factory import GameStateFactory

if __name__ == "__main__":
    load()  # fail fast if the model is missing
    # warm up numpy so the first real move isn't slow
    warm = NeuralAgent()
    warm.prepare_to_play_as(Player.Player1, GameParams(num_planets=30))
    for _ in range(3):
        warm.get_action(GameStateFactory(GameParams(num_planets=30)).create_game())
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=NeuralAgent()).start())
