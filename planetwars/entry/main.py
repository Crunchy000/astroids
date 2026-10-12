import asyncio

from agent import NeuralAgent, load
from client_server.game_agent_server import GameServerAgent

if __name__ == "__main__":
    load()  # fail fast if the model is missing
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=NeuralAgent()).start())
