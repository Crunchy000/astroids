"""Train one shared PPO policy to fly the whole blue team against the scripted red team.

    pip install -e ".[train]"
    python examples/train_team.py --init results/ppo_pilot.zip --n-workers 4

Every blue ship is driven by the same network (parameter sharing), each acting on
its own ego-centric observation. Each worker process runs one 5v5 battle and
contributes five "env slots" (one per blue ship) to a single SB3 VecEnv, so the
policy sees 5 x n_workers observations per step in one batch.

``--team-spirit`` shares teammates' kill/death rewards so ships care about the
team result. Starting from a single-seat pilot (``--init``) skips relearning
how to fly and aim. Every ``--eval-every`` samples the team plays ``--eval-battles``
fixed battles; the score is blue's win rate and mean kill margin, compared with
an all-bot blue team on the same battles. History goes to ``<out>_eval.json``
and ``<out>_eval.png``; the best model to ``<out>_best.zip``.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from astroids_gym import ACTION_SPACE, AsteroidsBattleParallelEnv, RewardConfig
from astroids_gym.obs import ObsConfig, observation_space
from plot_progress import plot

TEAM_METRICS = [
    {"key": "win_rate", "title": "Blue win rate", "fmt": "{:.0%}"},
    {"key": "margin", "title": "Average kill margin (blue − red)", "fmt": "{:+.1f}"},
]


def _worker(remote, seed, env_kwargs):
    torch.set_num_threads(1)
    env = AsteroidsBattleParallelEnv(learning_teams=("blue",), **env_kwargs)
    names = env.possible_agents
    rng = np.random.default_rng(seed)

    def reset():
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        return np.stack([obs[a] for a in names])

    obs = reset()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, rew, term, trunc, _ = env.step(dict(zip(names, data)))
            obs = np.stack([obs[a] for a in names])
            rew = np.array([rew[a] for a in names], dtype=np.float32)
            done = all(trunc.values()) or all(term.values())
            final = obs
            if done:
                obs = reset()
            remote.send((obs, rew, done, final, tuple(env.sim.score)))
        elif cmd == "reset":
            obs = reset()
            remote.send(obs)
        elif cmd == "set_reward_weights":
            for k, v in data.items():
                setattr(env.reward_cfg, k, v)
            remote.send(None)
        elif cmd == "close":
            remote.close()
            return


class TeamVecEnv(VecEnv):
    """n_workers battles x 5 blue ships = 5 * n_workers SB3 env slots."""

    def __init__(self, n_workers, seed=0, **env_kwargs):
        self.team_size = AsteroidsBattleParallelEnv(learning_teams=("blue",)).cfg.team_size
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.remotes, self.procs = [], []
        for i in range(n_workers):
            local, remote = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(remote, seed + i, env_kwargs), daemon=True)
            p.start()
            remote.close()
            self.remotes.append(local)
            self.procs.append(p)
        super().__init__(n_workers * self.team_size, observation_space(ObsConfig()), ACTION_SPACE)
        self.ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_len = 0
        self.t0 = time.time()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        self.ep_ret[:] = 0
        self.ep_len = 0
        return np.concatenate([r.recv() for r in self.remotes])

    def step_async(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.team_size:(i + 1) * self.team_size]))

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        obs = np.concatenate([o for o, *_ in results])
        rew = np.concatenate([rw for _, rw, *_ in results])
        dones = np.repeat([d for _, _, d, *_ in results], self.team_size)
        self.ep_ret += rew
        self.ep_len += 1  # all battles run in lockstep with the same max_steps
        infos = [{} for _ in range(self.num_envs)]
        for w, (_, _, done, final, score) in enumerate(results):
            if not done:
                continue
            for j in range(self.team_size):
                k = w * self.team_size + j
                infos[k] = {"terminal_observation": final[j], "TimeLimit.truncated": True,
                            "episode": {"r": float(self.ep_ret[k]), "l": self.ep_len,
                                        "t": round(time.time() - self.t0, 2)},
                            "team_score": score}
                self.ep_ret[k] = 0
        if dones.any():
            self.ep_len = 0
        return obs, rew, dones, infos

    def env_method(self, method_name, *args, indices=None, **kwargs):
        assert method_name == "set_reward_weights"
        for r in self.remotes:
            r.send(("set_reward_weights", kwargs))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)

    # Unused by PPO; required by the VecEnv interface.
    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


def evaluate_team(act, battles=40, seed=70_000):
    """Blue flown by ``act(batch_obs)`` (``None`` = scripted bots) vs the scripted red team."""
    margins = []
    for b in range(battles):
        env = AsteroidsBattleParallelEnv(learning_teams=("blue",) if act else ())
        obs, _ = env.reset(seed=seed + b)
        while True:
            if act:
                names = list(obs)
                obs, *_ = env.step(dict(zip(names, act(np.stack([obs[a] for a in names])))))
            else:
                obs, *_ = env.step({})
            if env._steps >= env.max_steps:
                break
        margins.append(env.sim.score[0] - env.sim.score[1])
    m = np.array(margins)
    return {"win_rate": float((m > 0).mean()), "margin": float(m.mean()),
            "draws": int((m == 0).sum())}


def fmt(s):
    return f"win rate {s['win_rate']:4.0%}  kill margin {s['margin']:+5.1f}"


class TeamEval(BaseCallback):
    def __init__(self, every, battles, out):
        super().__init__()
        self.every, self.battles, self.out = every, battles, out
        self.next_eval, self.best, self.history = every, -float("inf"), []
        print(f"evaluating an all-bot blue team over {battles} battles ...")
        self.bot = evaluate_team(None, battles)
        print("bot team   ", fmt(self.bot))

    def _record(self):
        stats = evaluate_team(lambda o: self.model.predict(o, deterministic=True)[0], self.battles)
        beats = stats["win_rate"] > self.bot["win_rate"] and stats["margin"] > self.bot["margin"]
        print(f"[{self.num_timesteps:>10,}] agent team {fmt(stats)}  {'BEATS THE BOT TEAM' if beats else ''}",
              flush=True)
        self.history.append({"timesteps": self.num_timesteps, **stats})
        with open(f"{self.out}_eval.json", "w") as f:
            json.dump({"bot": self.bot, "metrics": TEAM_METRICS, "history": self.history,
                       "title": f"Shared PPO policy flying all of blue vs the bot team "
                                f"({self.battles} battles per point)",
                       "bot_label": "All-bot blue team", "agent_label": "PPO team",
                       "xlabel": "Training samples, ship-steps (millions)"}, f, indent=1)
        if importlib.util.find_spec("matplotlib"):
            plot(f"{self.out}_eval.json")
        score = stats["margin"] + 10 * stats["win_rate"]
        if score > self.best:
            self.best = score
            self.model.save(f"{self.out}_best")

    def _on_training_start(self):
        self._record()  # where the starting policy stands as a team

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.every
            self._record()
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=20_000_000, help="ship-steps (5 per battle step)")
    p.add_argument("--n-workers", type=int, default=4, help="parallel battles (one process each)")
    p.add_argument("--init", help="start from this PPO .zip (e.g. the single-seat pilot)")
    p.add_argument("--team-spirit", type=float, default=0.5)
    p.add_argument("--aim", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-battles", type=int, default=40)
    p.add_argument("--out", default="models/ppo_team")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    venv = TeamVecEnv(args.n_workers, seed=args.seed,
                      reward_config=RewardConfig(team_spirit=args.team_spirit, aim=args.aim))
    n_steps = 256  # per slot: 256 x 5 x n_workers samples per update
    hyper = {"n_steps": n_steps, "batch_size": n_steps * venv.num_envs // 4,
             "learning_rate": args.lr, "ent_coef": 0.003}
    if args.init:
        model = PPO.load(args.init, env=venv, custom_objects=hyper, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, n_epochs=5, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                    policy_kwargs={"net_arch": [256, 256]}, seed=args.seed, device="cpu", **hyper)
    model.learn(total_timesteps=args.timesteps,
                callback=TeamEval(args.eval_every, args.eval_battles, args.out))
    model.save(args.out)
    print(f"saved {args.out}.zip (best eval: {args.out}_best.zip)")
    venv.close()


if __name__ == "__main__":
    main()
