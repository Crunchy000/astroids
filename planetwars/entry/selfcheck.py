"""Play the neural agent against the official sample agents in the official Python engine.

Mirrors the submission sanity check (20 planets, 1000 ticks, both sides):
    python selfcheck.py --games 3
"""
import argparse
import random
import time

from agent import NeuralAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.random_agents import CarefulRandomAgent
from core.forward_model import ForwardModel
from core.game_state import GameParams, Player
from core.game_state_factory import GameStateFactory


def play(me, other, params, side):
    """One game with `me` as Player `side`. Returns (won, slowest move of `me` in seconds)."""
    gs = GameStateFactory(params).create_game()
    mine = Player.Player1 if side == 1 else Player.Player2
    me.prepare_to_play_as(mine, params)
    other.prepare_to_play_as(mine.opponent(), params)
    model = ForwardModel(gs, params)
    worst = 0.0
    while not model.is_terminal():
        t = time.perf_counter()
        a = me.get_action(gs)
        worst = max(worst, time.perf_counter() - t)
        model.step({mine: a, mine.opponent(): other.get_action(gs)})
    return model.get_leader() == mine, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--planets", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    params = GameParams(num_planets=args.planets, max_ticks=args.ticks)
    for name, cls in (("greedy", GreedyHeuristicAgent), ("careful random", CarefulRandomAgent)):
        wins = games = 0
        worst = 0.0
        for g in range(args.games):
            for side in (1, 2):
                won, w = play(NeuralAgent(), cls(), params, side)
                wins += won
                worst = max(worst, w)
                games += 1
        print(f"vs {name}: won {wins}/{games}; slowest move {worst * 1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
