"""League machinery: fast battles between teams flown by models or bots.

* ``NumpyPolicy`` - a PPO policy network exported to NumPy. Frozen opponents and
  tournaments use it instead of SB3's ``predict`` (~5x faster per call),
  so worker processes never import torch.
* ``_worker`` - the process behind ``train_league.LeagueVecEnv``. It runs
  several battles; in each, the learner flies one team (side picked at random
  per battle) against an opponent sampled from a weighted list (``"bots"`` or a
  ``.npz`` policy). Every learner ship is one env slot.
* ``play_match`` / ``tournament`` / ``fit_elo`` - side-swapped round-robin
  battles in a process pool, and Bradley-Terry Elo ratings from the results.

This module never imports torch or SB3, so worker and tournament processes stay light.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections import defaultdict
from itertools import combinations

import numpy as np

from astroids_gym import ACTION_SPACE, Config, RewardConfig, Simulation, decode_action, scripted_action
from astroids_gym.obs import ObsConfig, encode_battle
from astroids_gym.sim import Events

FRAME_SKIP = 2
MAX_STEPS = 1800  # 60 s battles
SPLITS = np.cumsum(ACTION_SPACE.nvec)[:-1]  # logits layout [5 | 2 | 2]

# ---- policies -------------------------------------------------------------------
def _hidden_keys(sd, net="policy_net") -> list[str]:
    """Weight keys of the Linear layers in an SB3 MLP extractor, in order (any depth)."""
    idx = sorted(int(k.split(".")[2]) for k in sd if k.startswith(f"mlp_extractor.{net}.") and k.endswith(".weight"))
    return [f"mlp_extractor.{net}.{i}" for i in idx]


def export_policy(model, path: str) -> str:
    """Save an SB3 PPO policy's actor network (any depth/width) as a ``.npz`` for ``NumpyPolicy``."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    arrays = {}
    for i, k in enumerate(_hidden_keys(sd)):
        arrays[f"w{i}"], arrays[f"b{i}"] = sd[k + ".weight"].T, sd[k + ".bias"]
    arrays["wo"], arrays["bo"] = sd["action_net.weight"].T, sd["action_net.bias"]
    np.savez(path, **arrays)
    return path


class NumpyPolicy:
    def __init__(self, path: str):
        d = np.load(path)
        if "wo" in d:  # any depth
            n = sum(1 for k in d.files if k.startswith("w") and k != "wo")
            self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
            self.out = (d["wo"], d["bo"])
        else:  # original two-hidden-layer files
            self.layers = [(d["w0"], d["b0"]), (d["w1"], d["b1"])]
            self.out = (d["w2"], d["b2"])

    @property
    def widths(self) -> list[int]:
        return [w.shape[1] for w, _ in self.layers]

    def __call__(self, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Actions for a batch of observations: argmax, or sampled when ``rng`` is given."""
        h = obs
        for w, b in self.layers:
            h = np.tanh(h @ w + b)
        logits = h @ self.out[0] + self.out[1]
        acts = []
        for part in np.split(logits, SPLITS, axis=1):
            if rng is not None:  # Gumbel-max sampling == categorical sampling
                part = part - np.log(-np.log(rng.random(part.shape)))
            acts.append(part.argmax(axis=1))
        return np.stack(acts, axis=1)


_policy_cache: dict[str, NumpyPolicy] = {}


def _policy(spec: str) -> NumpyPolicy:
    if spec not in _policy_cache:
        _policy_cache[spec] = NumpyPolicy(spec)
    return _policy_cache[spec]


# ---- one battle ---------------------------------------------------------------
class Battle:
    """A 5v5 battle stepped at the agent rate (FRAME_SKIP physics ticks per step)."""

    def __init__(self, seed: int, cfg: Config | None = None):
        self.sim = Simulation(cfg or Config(), seed=seed)
        ts = self.sim.cfg.team_size
        self.ids = [list(range(ts)), list(range(ts, 2 * ts))]
        self.steps = 0
        self._obs, self._obs_step = None, -1

    def obs(self, team: int, oc: ObsConfig) -> np.ndarray:
        # Both teams' observations come from one vectorised pass per step.
        if self._obs_step != self.steps:
            self._obs, self._obs_step = encode_battle(self.sim, range(self.sim.n_ships), oc), self.steps
        return self._obs[self.ids[team]]

    def team_actions(self, team: int, spec: str, oc: ObsConfig, rng=None, actions=None) -> dict:
        """Actions for one team flown by ``spec`` ("bots" or a .npz policy), or given ``actions``."""
        sim = self.sim
        if actions is None and spec.startswith("bots"):  # "bots" or "bots:<style>"
            style = spec.split(":", 1)[1] if ":" in spec else None
            return {i: scripted_action(sim, sim.ships[i], FRAME_SKIP, style) for i in self.ids[team]}
        if actions is None:
            actions = _policy(spec)(self.obs(team, oc), rng)
        return {i: decode_action(a) for i, a in zip(self.ids[team], actions)}

    def step(self, acts: dict) -> Events:
        total = Events(self.sim.n_ships)
        for _ in range(FRAME_SKIP):
            total.merge(self.sim.step(acts))
        self.steps += 1
        return total


# ---- training env -----------------------------------------------------------------
def make_rules(rules: str) -> tuple[Config, ObsConfig]:
    """``"classic"`` or ``"competitive"`` -> (game config, observation config)."""
    if rules == "competitive":
        return Config.competitive(), ObsConfig(extended=True)
    return Config(), ObsConfig()


def battle_over(b: Battle) -> tuple[bool, bool]:
    """(over, eliminated): a battle ends at the time limit or when a team is wiped out."""
    eliminated = b.sim.eliminated(0) or b.sim.eliminated(1)
    return eliminated or b.steps >= MAX_STEPS or b.sim.over, eliminated


def _worker(remote, seed: int, n_battles: int, rules: str = "classic"):
    rng = np.random.default_rng(seed)
    cfg, oc = make_rules(rules)
    reward = RewardConfig()
    opponents: list[tuple[str, float]] = [("bots", 1.0)]

    def new_battle():
        b = Battle(int(rng.integers(2**31)), cfg)
        b.side = int(rng.integers(2))
        specs, weights = zip(*opponents)
        b.opp = specs[rng.choice(len(specs), p=np.array(weights) / sum(weights))]
        return b

    battles = [new_battle() for _ in range(n_battles)]
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            acts_per = np.split(data, n_battles)
            # Opponent moves: one batched forward pass per distinct policy across battles.
            opp_acts: dict[int, dict] = {}
            by_spec = defaultdict(list)
            for k, b in enumerate(battles):
                by_spec[b.opp].append(k)
            for spec, ks in by_spec.items():
                if spec.startswith("bots"):
                    for k in ks:
                        opp_acts[k] = battles[k].team_actions(1 - battles[k].side, spec, oc)
                else:
                    obs = np.concatenate([battles[k].obs(1 - battles[k].side, oc) for k in ks])
                    acts = np.split(_policy(spec)(obs, rng), len(ks))
                    for k, a in zip(ks, acts):
                        opp_acts[k] = battles[k].team_actions(1 - battles[k].side, spec, oc, actions=a)
            obs_out, rew_out, done_out, final_out, results = [], [], [], [], []
            for k, b in enumerate(battles):
                acts = {**opp_acts[k], **b.team_actions(b.side, "", oc, actions=acts_per[k])}
                ev = b.step(acts)
                r = reward.rewards(b.sim, ev)
                over, eliminated = battle_over(b)
                if over:
                    end = reward.final(b.sim, b.side)
                    r = [x + end for x in r]
                rew_out.append(np.array([r[i] for i in b.ids[b.side]], dtype=np.float32))
                o = b.obs(b.side, oc)
                done_out.append((over, eliminated))
                final_out.append(o)
                if over:
                    sim, me = b.sim, b.side
                    w = sim.winner()
                    results.append((b.opp, sim.team_score(me) - sim.team_score(1 - me),
                                    0 if w < 0 else (1 if w == me else -1)))
                    battles[k] = b = new_battle()
                    o = b.obs(b.side, oc)
                obs_out.append(o)
            remote.send((np.concatenate(obs_out), np.concatenate(rew_out), done_out, final_out, results))
        elif cmd == "reset":
            battles = [new_battle() for _ in range(n_battles)]
            remote.send(np.concatenate([b.obs(b.side, oc) for b in battles]))
        elif cmd == "configure":
            if "reward" in data:
                reward = RewardConfig(**data["reward"])
            if "opponents" in data:
                opponents = data["opponents"]
            remote.send(None)
        elif cmd == "close":
            remote.close()
            return


# ---- tournaments & Elo --------------------------------------------------------------
def play_match(args) -> tuple[str, str, float, float, int]:
    """One battle, ``a`` flying blue and ``b`` red -> (a, b, a's score, b's score, outcome).

    Score is kills plus control-zone points; outcome is 1 if ``a`` won, -1 if
    ``b`` won, 0 for a draw (wiping out the other team wins outright).
    """
    a, b, seed, rules = args
    cfg, oc = make_rules(rules)
    battle = Battle(seed, cfg)
    while not battle_over(battle)[0]:
        battle.step({**battle.team_actions(0, a, oc), **battle.team_actions(1, b, oc)})
    sim = battle.sim
    w = sim.winner()
    return a, b, sim.team_score(0), sim.team_score(1), 0 if w < 0 else (1 if w == 0 else -1)


def tournament(players: list[str], battles_per_pair: int = 10, seed: int = 0, workers: int = 4,
               rules: str = "classic"):
    """Round-robin; each pair plays half its battles on each side, on the same seeds."""
    jobs = []
    for i, (a, b) in enumerate(combinations(players, 2)):
        for n in range(battles_per_pair):
            s = seed + 1000 * i + n // 2
            jobs.append((a, b, s, rules) if n % 2 == 0 else (b, a, s, rules))
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(play_match, jobs, chunksize=1)


def fit_elo(results, anchor: str, anchor_elo: float = 1000.0, iters: int = 500) -> dict[str, float]:
    """Bradley-Terry ratings (Elo scale) from ``play_match`` results.

    A win scores 1, a draw 0.5. Every pair gets one virtual draw so a team that
    wins every battle still gets a finite rating.
    """
    wins = defaultdict(float)
    games = defaultdict(float)
    players = sorted({p for r in results for p in r[:2]})
    for a, b, _, _, outcome in results:
        s = (outcome + 1) / 2
        wins[a, b] += s
        wins[b, a] += 1 - s
        games[a, b] += 1
        games[b, a] += 1
    for a, b in combinations(players, 2):
        if games[a, b]:
            for x, y in ((a, b), (b, a)):
                wins[x, y] += 0.5
                games[x, y] += 1
    strength = {p: 1.0 for p in players}
    for _ in range(iters):  # minorise-maximise updates for Bradley-Terry
        new = {}
        for p in players:
            w = sum(wins[p, q] for q in players if q != p)
            d = sum(games[p, q] / (strength[p] + strength[q]) for q in players if q != p and games[p, q])
            new[p] = w / d if d else strength[p]
        strength = new
    base = math.log10(strength[anchor])
    return {p: anchor_elo + 400 * (math.log10(s) - base) for p, s in strength.items()}


def win_matrix(results) -> dict[tuple[str, str], tuple[float, float]]:
    """(a, b) -> (a's win rate vs b, a's mean score margin vs b), from either side."""
    w = defaultdict(list)
    m = defaultdict(list)
    for a, b, sa, sb, outcome in results:
        w[a, b].append((outcome + 1) / 2)
        w[b, a].append((1 - outcome) / 2)
        m[a, b].append(sa - sb)
        m[b, a].append(sb - sa)
    return {k: (float(np.mean(w[k])), float(np.mean(m[k]))) for k in w}
