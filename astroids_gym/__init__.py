"""Gymnasium / PettingZoo environments for the 5v5 Asteroids battle."""

from gymnasium.envs.registration import register

from .bot import scripted_action
from .env import (ACTION_SPACE, TURN_LEVELS, AsteroidsBattleEnv, AsteroidsBattleParallelEnv,
                  RewardConfig, decode_action)
from .obs import ObsConfig
from .sim import Config, Simulation

__all__ = [
    "ACTION_SPACE", "TURN_LEVELS", "AsteroidsBattleEnv", "AsteroidsBattleParallelEnv", "Config", "ObsConfig",
    "RewardConfig", "Simulation", "decode_action", "scripted_action", "parallel_env",
]


def parallel_env(**kwargs) -> AsteroidsBattleParallelEnv:
    return AsteroidsBattleParallelEnv(**kwargs)


register(id="AsteroidsBattle-v0", entry_point="astroids_gym.env:AsteroidsBattleEnv")
