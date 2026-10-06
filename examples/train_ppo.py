"""Train a PPO pilot for one blue ship with Stable-Baselines3.

    pip install -e ".[train]"
    python examples/train_ppo.py --timesteps 6000000 --n-envs 8

Aim shaping (``--aim``) pays a small bonus each step for pointing where a shot
would hit the nearest enemy. It fades linearly to zero over ``--aim-anneal``
steps, so the finished pilot is optimised for the real game reward. Every
``--eval-every`` steps the pilot and the built-in bot each fly blue seat 0 on the
same seeds; the best model is kept and ``--stop-when-beats`` ends training once
the pilot out-scores the bot.

Self-play: pit a new learner against frozen copies of an earlier model
(red team, and optionally blue teammates):

    python examples/train_ppo.py --opponent-model models/ppo_pilot.zip --out models/ppo_gen2
"""

import argparse
import importlib.util
import json
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import astroids_gym  # noqa: F401  (registers AsteroidsBattle-v0)
from astroids_gym import RewardConfig
from evaluate import evaluate, fmt


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


class AimAnneal(BaseCallback):
    """Linearly fade the aim bonus from ``start`` to 0 over ``steps`` timesteps."""

    def __init__(self, start, steps):
        super().__init__()
        self.start, self.steps, self.current = start, steps, None

    def _on_step(self):
        aim = round(self.start * max(0.0, 1 - self.num_timesteps / self.steps), 5)
        if aim != self.current:
            self.training_env.env_method("set_reward_weights", aim=aim)
            self.current = aim
            self.logger.record("train/aim_bonus", aim)
        return True


class BeatTheBot(BaseCallback):
    """Periodically compare the pilot with the bot in the same seat; keep the best model."""

    def __init__(self, every, episodes, out, stop_when_beats):
        super().__init__()
        self.every, self.episodes, self.out, self.stop = every, episodes, out, stop_when_beats
        self.next_eval, self.best = every, -float("inf")
        print("evaluating the built-in bot in blue seat 0 ...")
        self.bot = evaluate(lambda obs: None, episodes)
        print("bot   ", fmt(self.bot))
        self.history = []

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval = self.num_timesteps + self.every
        stats = evaluate(lambda obs: self.model.predict(obs, deterministic=True)[0], self.episodes)
        beats = stats["return"] > self.bot["return"] and stats["kd"] > self.bot["kd"]
        print(f"[{self.num_timesteps:>9,}] pilot {fmt(stats)}  {'BEATS THE BOT' if beats else ''}")
        for k, v in stats.items():
            self.logger.record(f"eval/{k}", v)
        self.history.append({"timesteps": self.num_timesteps, **stats})
        with open(f"{self.out}_eval.json", "w") as f:
            json.dump({"bot": self.bot, "history": self.history}, f, indent=1)
        if stats["return"] > self.best:
            self.best = stats["return"]
            self.model.save(f"{self.out}_best")
        return not (beats and self.stop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=6_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--out", default="models/ppo_pilot")
    p.add_argument("--aim", type=float, default=0.01, help="initial aim bonus per step (0 = off)")
    p.add_argument("--aim-anneal", type=int, default=3_000_000, help="steps to fade the aim bonus to 0")
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--stop-when-beats", action="store_true")
    p.add_argument("--opponent-model", help="frozen PPO .zip to fly the red team (self-play)")
    p.add_argument("--teammate-model", help="frozen PPO .zip to fly the blue teammates")
    p.add_argument("--resume", help="continue training this .zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1,
                   help="keep low so the learner doesn't starve the env worker processes")
    args = p.parse_args()
    torch.set_num_threads(args.torch_threads)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env_kwargs = {"reward_config": RewardConfig(aim=args.aim)}
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
            gamma=0.995, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2,
            policy_kwargs={"net_arch": [256, 256]}, seed=args.seed, verbose=0,
            tensorboard_log="runs" if importlib.util.find_spec("tensorboard") else None,
        )
    callbacks = [BeatTheBot(args.eval_every, args.eval_episodes, args.out, args.stop_when_beats)]
    if args.aim:
        callbacks.append(AimAnneal(args.aim, args.aim_anneal))
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(args.out)
    print(f"saved {args.out}.zip (best eval: {args.out}_best.zip)")


if __name__ == "__main__":
    main()
