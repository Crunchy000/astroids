"""Train a PPO pilot for one blue ship with Stable-Baselines3.

    pip install -e ".[train]"
    python examples/train_ppo.py --timesteps 2000000 --n-envs 8

Self-play: pit a new learner against frozen copies of an earlier model
(red team, and optionally blue teammates):

    python examples/train_ppo.py --opponent-model models/ppo_pilot.zip --out models/ppo_gen2
"""

import argparse
import importlib.util

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import astroids_gym  # noqa: F401  (registers AsteroidsBattle-v0)


class FrozenPolicy:
    """A saved PPO pilot flying several ships at once (gets a batch of observations).

    Loaded lazily inside each env worker process, with one torch thread so the
    workers don't oversubscribe the CPU.
    """

    def __init__(self, path):
        self.path, self.model = path, None

    def __call__(self, obs):
        if self.model is None:
            torch.set_num_threads(1)
            self.model = PPO.load(self.path, device="cpu")
        return self.model.predict(obs, deterministic=False)[0]

    def __getstate__(self):  # ship only the path to subprocesses
        return {"path": self.path, "model": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--out", default="models/ppo_pilot")
    p.add_argument("--opponent-model", help="frozen PPO .zip to fly the red team (self-play)")
    p.add_argument("--teammate-model", help="frozen PPO .zip to fly the blue teammates")
    p.add_argument("--resume", help="continue training this .zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1,
                   help="keep low so the learner doesn't starve the env worker processes")
    args = p.parse_args()
    torch.set_num_threads(args.torch_threads)

    env_kwargs = {}
    if args.opponent_model:
        env_kwargs["opponents"] = FrozenPolicy(args.opponent_model)
    if args.teammate_model:
        env_kwargs["teammates"] = FrozenPolicy(args.teammate_model)

    venv = make_vec_env(
        "AsteroidsBattle-v0", n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        monitor_kwargs={"info_keywords": ("kills", "deaths")},
    )
    if args.resume:
        model = PPO.load(args.resume, env=venv)
    else:
        model = PPO(
            "MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=5, learning_rate=3e-4,
            gamma=0.995, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
            policy_kwargs={"net_arch": [256, 256]}, seed=args.seed, verbose=1,
            tensorboard_log="runs" if importlib.util.find_spec("tensorboard") else None,
        )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
