"""Evaluate (and optionally watch) a pilot in blue seat 0 against the scripted bots.

    python examples/evaluate.py                              # the built-in bot flies the seat (baseline)
    python examples/evaluate.py --model random
    python examples/evaluate.py --model models/ppo_pilot.zip
    python examples/evaluate.py --model models/ppo_pilot.zip --render   # needs pygame

Scores use the plain game reward (kills, deaths, hits; no aim shaping), so a
trained model and the bot are compared on equal terms over the same seeds.
"""

import argparse

import gymnasium as gym
import numpy as np

import astroids_gym  # noqa: F401  (registers AsteroidsBattle-v0)


def evaluate(act, episodes=10, seed=10_000, render_mode=None, verbose=False):
    """Run ``act(obs) -> action`` (``None`` = built-in bot) and return summary stats."""
    env = gym.make("AsteroidsBattle-v0", render_mode=render_mode)
    kills, deaths, returns = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += r
            done = term or trunc
        kills.append(info["kills"]); deaths.append(info["deaths"]); returns.append(ret)
        if verbose:
            print(f"episode {ep}: return {ret:6.2f}  kills {info['kills']}  deaths {info['deaths']}  "
                  f"team score {info['score'][0]}-{info['score'][1]}")
    env.close()
    return {"return": float(np.mean(returns)), "kills": float(np.mean(kills)),
            "deaths": float(np.mean(deaths)), "kd": sum(kills) / max(sum(deaths), 1)}


def fmt(stats):
    return (f"return {stats['return']:6.2f}  kills {stats['kills']:4.1f}  "
            f"deaths {stats['deaths']:4.1f}  K/D {stats['kd']:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="SB3 PPO .zip, or 'random'; omit for the built-in bot")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args()

    if args.model == "random":
        space = astroids_gym.ACTION_SPACE
        act = lambda obs: space.sample()
    elif args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        act = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        act = lambda obs: None  # the env lets the built-in bot fly the seat

    stats = evaluate(act, args.episodes, args.seed, "human" if args.render else None, verbose=True)
    print("mean", fmt(stats))


if __name__ == "__main__":
    main()
