"""Train a league of team models with different personalities against each other.

    pip install -e ".[train]"
    python examples/train_league.py --init results/ppo_team.zip --rounds 10

Each member is a shared-policy team model (one network flies all five ships)
with its own reward "personality". Every round, each member trains in turn
against a mix of: the other members' latest versions (50%), past snapshots of
the whole league (30%) and the built-in bots (20%), playing a random side each
battle. After every round, all members, the starting champion and the bots
play a side-swapped round-robin tournament; ratings are Bradley-Terry Elo,
anchored so the built-in bots are 1000. Everything goes to ``--out``:
snapshots (``.zip`` + ``.npz``), ``league.json`` and charts.

``--rules competitive`` (the default) plays with the control zone, weapon heat,
limited lives and friendly fire, on the extended observations; the starting
model is widened to the extra inputs with zero weights, so it starts out
playing exactly as before.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

# One BLAS thread per process: every worker runs NumPy policy maths, and the
# default (one thread per core in every process) oversubscribes the CPU ~7x.
# Must be set before NumPy is imported; torch's own threads are set explicitly.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from astroids_gym import ACTION_SPACE, Config
from astroids_gym.obs import observation_space
from league import _worker, export_policy, fit_elo, make_rules, tournament, win_matrix

# Shared by every personality: zone points, the battle result and friendly fire
# (inert under classic rules, where there is no zone and no friendly fire).
_BASE = {"hit_dealt": 0.2, "hit_taken": -0.2, "zone": 0.15, "win": 3.0, "friendly_hit": -0.3}
PERSONALITIES = {
    # Values kills far above its own survival.
    "striker": {**_BASE, "kill": 1.5, "death": -0.5, "hit_dealt": 0.3, "hit_taken": -0.1,
                "team_spirit": 0.3, "zone": 0.1},
    # Values staying alive above everything.
    "guardian": {**_BASE, "kill": 0.8, "death": -1.5, "hit_taken": -0.4, "team_spirit": 0.5,
                 "friendly_hit": -0.5},
    # Every ship gets its teammates' kills and deaths in full, and it plays the objective.
    "pack": {**_BASE, "kill": 1.0, "death": -1.0, "team_spirit": 1.0, "zone": 0.25},
}


class LeagueVecEnv(VecEnv):
    """``n_workers`` processes x ``battles_per_worker`` battles x 5 learner ships = env slots."""

    def __init__(self, n_workers: int = 4, battles_per_worker: int = 4, seed: int = 0,
                 rules: str = "classic"):
        self.team_size = Config().team_size
        self.per_worker = battles_per_worker * self.team_size
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.remotes, self.procs = [], []
        for i in range(n_workers):
            local, remote = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(remote, seed + 1000 * i, battles_per_worker, rules),
                            daemon=True)
            p.start()
            remote.close()
            self.remotes.append(local)
            self.procs.append(p)
        super().__init__(n_workers * self.per_worker, observation_space(make_rules(rules)[1]), ACTION_SPACE)
        self.ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self.results: list[tuple[str, float, int]] = []  # (opponent, score margin, outcome) per battle
        self.t0 = time.time()

    def configure(self, **kwargs):
        """``reward=dict(...)`` (RewardConfig fields) and/or ``opponents=[(spec, weight), ...]``."""
        for r in self.remotes:
            r.send(("configure", kwargs))
        for r in self.remotes:
            r.recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        self.ep_ret[:] = 0
        self.ep_len[:] = 0
        return np.concatenate([r.recv() for r in self.remotes])

    def step_async(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.per_worker:(i + 1) * self.per_worker]))

    def step_wait(self):
        out = [r.recv() for r in self.remotes]
        obs = np.concatenate([o[0] for o in out])
        rew = np.concatenate([o[1] for o in out])
        battle_done = [d for o in out for d in o[2]]
        finals = [f for o in out for f in o[3]]
        for o in out:
            self.results.extend(o[4])
        dones = np.repeat([over for over, _ in battle_done], self.team_size)
        self.ep_ret += rew
        self.ep_len += 1
        infos = [{} for _ in range(self.num_envs)]
        for bi, (over, eliminated) in enumerate(battle_done):
            if not over:
                continue
            for j in range(self.team_size):
                k = bi * self.team_size + j
                # A wipe-out is a real end of the game; a time-out is a cut-off (bootstrap the value).
                infos[k] = {"terminal_observation": finals[bi][j], "TimeLimit.truncated": not eliminated,
                            "episode": {"r": float(self.ep_ret[k]), "l": int(self.ep_len[k]),
                                        "t": round(time.time() - self.t0, 2)}}
                self.ep_ret[k] = 0
                self.ep_len[k] = 0
        return obs, rew, dones, infos

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)

    # Required by the VecEnv interface; unused by PPO.
    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("use configure()")

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


class ThreadSwitch(BaseCallback):
    """One torch thread while collecting (workers need the cores), all cores for the PPO update."""

    def __init__(self, train_threads: int):
        super().__init__()
        self.train_threads = train_threads

    def _on_rollout_start(self):
        torch.set_num_threads(1)

    def _on_rollout_end(self):
        torch.set_num_threads(self.train_threads)

    def _on_step(self):
        return True


def warm_start(init_path: str, venv, hyper: dict, seed: int) -> PPO:
    """A new PPO for ``venv`` initialised from a saved model.

    If the env has more observation inputs than the saved model (extended
    competitive observations), the extra input columns of the first layers start
    at zero, so the new model initially plays exactly like the old one.
    """
    old = PPO.load(init_path, device="cpu")
    model = PPO("MlpPolicy", venv, n_epochs=old.n_epochs, gamma=old.gamma, gae_lambda=old.gae_lambda,
                clip_range=0.2, policy_kwargs={"net_arch": [256, 256]}, seed=seed, device="cpu",
                verbose=0, **hyper)
    old_sd, new_sd = old.policy.state_dict(), model.policy.state_dict()
    for k, v in new_sd.items():
        o = old_sd[k]
        if o.shape == v.shape:
            new_sd[k] = o.clone()
        else:
            assert o.dim() == 2 and o.shape[0] == v.shape[0] and o.shape[1] < v.shape[1], k
            w = torch.zeros_like(v)
            w[:, :o.shape[1]] = o
            new_sd[k] = w
    model.policy.load_state_dict(new_sd)
    return model


def opponent_mix(name, latest, snapshots, rng, bots_share=0.2, max_snapshots=6):
    """Other members' latest, past snapshots (5:3) and bots (``bots_share``); renormalised if a group is empty."""
    others = [latest[o] for o in latest if o != name]
    current = set(latest.values())
    past = [s for s in snapshots if s not in current]
    if len(past) > max_snapshots:
        past = list(rng.choice(past, max_snapshots, replace=False))
    rest = 1.0 - bots_share
    groups = [(others, rest * 0.625), (past, rest * 0.375), (["bots"], bots_share)]
    total = sum(w for g, w in groups if g)
    return [(spec, w / total / len(g)) for g, w in groups if g for spec in g]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="starting champion .zip (every member starts here)")
    p.add_argument("--rules", choices=["competitive", "classic"], default="competitive")
    p.add_argument("--members", default="striker,guardian,pack")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--samples-per-round", type=int, default=1_000_000, help="ship-steps per member per round")
    p.add_argument("--n-workers", type=int, default=os.cpu_count())
    p.add_argument("--battles-per-worker", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=128, help="PPO rollout length per env slot")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tournament-battles", type=int, default=10, help="battles per pair per round")
    p.add_argument("--bots-share", type=float, default=0.2, help="fraction of training battles against bots")
    p.add_argument("--out", default="league")
    p.add_argument("--resume", action="store_true", help="continue the run in --out from its last round")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(1)
    os.makedirs(f"{args.out}/models", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = args.members.split(",")
    venv = LeagueVecEnv(args.n_workers, args.battles_per_worker, seed=args.seed, rules=args.rules)
    hyper = {"n_steps": args.n_steps, "batch_size": args.n_steps * venv.num_envs // 4,
             "learning_rate": args.lr, "ent_coef": 0.003}
    print(f"{venv.num_envs} env slots ({args.n_workers} workers x {args.battles_per_worker} battles x 5 ships), "
          f"{hyper['n_steps'] * venv.num_envs:,} samples per PPO update", flush=True)

    champion = f"{args.out}/models/champion.npz"
    models, latest = {}, {}
    if args.resume:  # pick up each member's snapshot from the last finished round
        with open(f"{args.out}/league.json") as f:
            state = json.load(f)
        done = len(state["rounds"])
        snapshots = [champion] + [f"{args.out}/models/{n}_r{r}.npz" for r in range(1, done + 1) for n in names]
        for n in names:
            models[n] = PPO.load(f"{args.out}/models/{n}_r{done}.zip", env=venv, custom_objects=hyper, device="cpu")
            latest[n] = f"{args.out}/models/{n}_r{done}.npz"
        print(f"resuming after round {done}", flush=True)
    else:
        # The starting champion, widened to this ruleset's observations (same behaviour).
        export_policy(warm_start(args.init, venv, hyper, args.seed), champion)
        for i, n in enumerate(names):
            models[n] = warm_start(args.init, venv, hyper, args.seed + i)
            latest[n] = champion
        snapshots = [champion]
        state = {"rules": args.rules, "members": names,
                 "personalities": {n: PERSONALITIES[n] for n in names}, "rounds": []}
    switch = ThreadSwitch(os.cpu_count())

    for rnd in range(len(state["rounds"]) + 1, args.rounds + 1):
        round_log = {"round": rnd, "training": {}, "bots_share": args.bots_share}
        for n in names:
            opponents = opponent_mix(n, latest, snapshots, rng, args.bots_share)
            venv.configure(reward=PERSONALITIES[n], opponents=opponents)
            venv.results.clear()
            model = models[n]
            model._last_obs = None  # the env was stepped by other members: start fresh battles
            t = time.time()
            model.learn(args.samples_per_round, callback=switch, reset_num_timesteps=False)
            rate = args.samples_per_round / (time.time() - t)
            path = f"{args.out}/models/{n}_r{rnd}"
            model.save(path)
            export_policy(model, path + ".npz")
            # Win rate by opponent kind during this training phase.
            kinds = {}
            for spec, _, outcome in venv.results:
                kind = "bots" if spec == "bots" else "league" if spec in latest.values() else "snapshots"
                kinds.setdefault(kind, []).append((outcome + 1) / 2)
            summary = {k: round(float(np.mean(v)), 3) for k, v in kinds.items()}
            round_log["training"][n] = {"samples_per_s": round(rate), "win_rate_vs": summary}
            print(f"round {rnd} {n:<9} {rate:6.0f} samples/s  training win rate "
                  + "  ".join(f"vs {k} {v:.0%}" for k, v in sorted(summary.items())), flush=True)
            latest[n] = path + ".npz"
            snapshots.append(path + ".npz")

        # Tournament: current members + starting champion + bots.
        players = [latest[n] for n in names] + [champion, "bots"]
        t = time.time()
        results = tournament(players, args.tournament_battles, seed=10_000 * rnd, workers=args.n_workers,
                             rules=args.rules)
        elo = fit_elo(results, anchor="bots")
        disp = {latest[n]: n for n in names} | {champion: "champion", "bots": "bots"}
        round_log["elo"] = {disp[p]: round(e) for p, e in elo.items()}
        round_log["matrix"] = {f"{disp[a]}|{disp[b]}": [round(w, 3), round(m, 2)]
                               for (a, b), (w, m) in win_matrix(results).items()}
        state["rounds"].append(round_log)
        with open(f"{args.out}/league.json", "w") as f:
            json.dump(state, f, indent=1)
        print(f"round {rnd} tournament ({len(results)} battles, {time.time() - t:.0f}s): "
              + "  ".join(f"{k} {v}" for k, v in sorted(round_log["elo"].items(), key=lambda kv: -kv[1])),
              flush=True)
        try:
            from plot_league import plot_league
            plot_league(f"{args.out}/league.json")
        except ImportError:
            pass
    venv.close()


if __name__ == "__main__":
    main()
