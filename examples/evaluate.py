"""Evaluate (and optionally watch) a pilot against the scripted bots.

    python examples/evaluate.py                              # scripted bot as the pilot (baseline)
    python examples/evaluate.py --model models/ppo_pilot.zip
    python examples/evaluate.py --model models/ppo_pilot.zip --render   # needs pygame
"""

import argparse

import gymnasium as gym
import numpy as np

import astroids_gym
from astroids_gym import scripted_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="SB3 PPO .zip; omit to fly the scripted bot, 'random' for random")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    env = gym.make("AsteroidsBattle-v0", render_mode="human" if args.render else None)
    sim_env = env.unwrapped

    if args.model == "random":
        act = lambda obs: env.action_space.sample()
    elif args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        act = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        # Map the bot's continuous turn onto the discrete action space.
        def act(obs):
            turn, thrust, fire = scripted_action(sim_env.sim, sim_env.sim.ships[sim_env.controlled])
            return np.array([int(np.sign(round(turn))) + 1, int(thrust), int(fire)])

    kills, deaths, returns = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += r
            done = term or trunc
        kills.append(info["kills"]); deaths.append(info["deaths"]); returns.append(ret)
        print(f"episode {ep}: return {ret:6.2f}  kills {info['kills']}  deaths {info['deaths']}  "
              f"team score {info['score'][0]}-{info['score'][1]}")
    print(f"mean return {np.mean(returns):.2f}  kills {np.mean(kills):.1f}  deaths {np.mean(deaths):.1f}  "
          f"K/D {sum(kills) / max(sum(deaths), 1):.2f}")
    env.close()


if __name__ == "__main__":
    main()
