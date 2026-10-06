"""Record MP4 videos of a pilot in the 5v5 battle.

    pip install -e ".[video]"
    python examples/record_video.py --model models/ppo_pilot.zip --out videos/pilot.mp4
    python examples/record_video.py --model models/ppo_pilot.zip --team --out videos/team.mp4
    python examples/record_video.py --out videos/bot.mp4          # the built-in bot, for comparison

Default: the pilot flies blue seat 0 (circled) with scripted teammates and enemies.
``--team``: the pilot flies all five blue ships against the scripted red team.
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np

from astroids_gym import AsteroidsBattleEnv, AsteroidsBattleParallelEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="SB3 PPO .zip; omit to film the built-in bot")
    p.add_argument("--out", default="videos/battle.mp4")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--team", action="store_true", help="pilot flies the whole blue team")
    p.add_argument("--caption", help="text in the top-right corner (default: who is flying)")
    args = p.parse_args()

    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        who = os.path.splitext(os.path.basename(args.model))[0]
    else:
        policy, who = None, "built-in bot"

    steps = int(args.seconds * 30)  # agent steps run at 30 Hz with the default frame_skip
    if args.team:
        env = AsteroidsBattleParallelEnv(render_mode="rgb_array", learning_teams=("blue",),
                                         max_steps=steps)
        if policy is None:  # let the scripted bot fly blue too
            env = AsteroidsBattleParallelEnv(render_mode="rgb_array", learning_teams=(), max_steps=steps)
        obs, _ = env.reset(seed=args.seed)
        caption = args.caption or f"BLUE TEAM: {who}"

        def advance():
            nonlocal obs
            names = list(obs)
            if names:
                acts = policy(np.stack([obs[a] for a in names]))
                obs, *_ = env.step(dict(zip(names, acts)))
            else:
                obs, *_ = env.step({})
    else:
        env = AsteroidsBattleEnv(render_mode="rgb_array", max_steps=steps)
        obs, _ = env.reset(seed=args.seed)
        caption = args.caption or f"CIRCLED: {who}"

        def advance():
            nonlocal obs
            obs, *_ = env.step(policy(obs) if policy else None)

    env.render()  # creates the renderer
    env._renderer.caption = caption
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with imageio.get_writer(args.out, fps=env.metadata["render_fps"], codec="libx264",
                            quality=7, macro_block_size=8) as writer:
        writer.append_data(env.render())
        for _ in range(steps):
            advance()
            writer.append_data(env.render())
    print(f"wrote {args.out}  final score BLUE {env.sim.score[0]} - RED {env.sim.score[1]}")
    env.close()


if __name__ == "__main__":
    main()
