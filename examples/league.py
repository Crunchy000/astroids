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
from astroids_gym.entity import entity_features
from astroids_gym.obs import ObsConfig, encode_battle, global_state
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
    """Save an SB3 PPO policy's actor network (any depth/width) as a ``.npz`` for ``NumpyPolicy``.

    With a centralised critic the actor only reads the first ``n_actor_inputs``
    observation inputs (the rest is the global state, for the critic), so the
    NumPy policy takes the plain observation. An attention actor
    (``EntityPolicy``) also exports its entity extractor (keys ``fx.*``).
    """
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    n_in = getattr(model.policy, "n_actor_inputs", None)
    arrays = {}
    ent = getattr(model.policy, "entity_kwargs", None)
    if ent is not None:
        fx = model.policy.pi_features_extractor
        arrays["entity"] = np.array([fx.heads, int(fx.oc.extended)])
        arrays |= {"fx." + k[len("pi_features_extractor."):]: v for k, v in sd.items()
                   if k.startswith("pi_features_extractor.")}
        n_in = None
    for i, k in enumerate(_hidden_keys(sd)):
        w = sd[k + ".weight"].T
        if i == 0 and n_in is not None:
            w = w[:n_in]
        arrays[f"w{i}"], arrays[f"b{i}"] = w, sd[k + ".bias"]
    arrays["wo"], arrays["bo"] = sd["action_net.weight"].T, sd["action_net.bias"]
    np.savez(path, **arrays)
    return path


class NumpyPolicy:
    def __init__(self, path: str):
        d = np.load(path)
        self.entity = None
        if "entity" in d:  # attention actor: entity extractor, then the MLP head
            heads, extended = (int(x) for x in d["entity"])
            self.entity = ({k[3:]: d[k] for k in d.files if k.startswith("fx.")}, ObsConfig(extended=bool(extended)),
                           heads)
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

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """Action logits for a batch of observations, laid out [turn 5 | thrust 2 | fire 2]."""
        h = obs if self.entity is None else entity_features(self.entity[0], obs, self.entity[1], self.entity[2])
        for w, b in self.layers:
            h = np.tanh(h @ w + b)
        return h @ self.out[0] + self.out[1]

    def __call__(self, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Actions for a batch of observations: argmax, or sampled when ``rng`` is given."""
        acts = []
        for part in np.split(self.logits(obs), SPLITS, axis=1):
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


def _worker(remote, seed: int, n_battles: int, rules: str = "classic", central: bool = False,
            remote_policies: bool = False):
    """Training battles for ``LeagueVecEnv``.

    With ``remote_policies`` the worker never runs a network: each reply also
    carries the frozen opponents' observations ({spec: obs}), and the next
    "step" command brings their actions back, computed in the main process
    (``PolicyBank``, e.g. on a GPU). The scripted bots always run here.
    """
    rng = np.random.default_rng(seed)
    cfg, oc = make_rules(rules)

    def learner_obs(b):
        """Learner ships' observations; with ``central``, the global battle state is appended."""
        o = b.obs(b.side, oc)
        if not central:
            return o
        g = global_state(b.sim, b.side)
        return np.concatenate([o, np.broadcast_to(g, (len(o), len(g)))], axis=1)

    reward = RewardConfig()
    opponents: list[tuple[str, float]] = [("bots", 1.0)]

    def new_battle():
        b = Battle(int(rng.integers(2**31)), cfg)
        b.side = int(rng.integers(2))
        specs, weights = zip(*opponents)
        b.opp = specs[rng.choice(len(specs), p=np.array(weights) / sum(weights))]
        return b

    def by_spec():
        groups = defaultdict(list)
        for k, b in enumerate(battles):
            groups[b.opp].append(k)
        return groups

    def opponent_requests():
        """{spec: observations} of the network-flown opponents, in ``by_spec`` order."""
        return {spec: np.concatenate([battles[k].obs(1 - battles[k].side, oc) for k in ks])
                for spec, ks in by_spec().items() if not spec.startswith("bots")}

    battles = [new_battle() for _ in range(n_battles)]
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            if remote_policies:
                data, given = data
            acts_per = np.split(data, n_battles)
            # Opponent moves: one batched forward pass per distinct policy across battles.
            opp_acts: dict[int, dict] = {}
            for spec, ks in by_spec().items():
                if spec.startswith("bots"):
                    for k in ks:
                        opp_acts[k] = battles[k].team_actions(1 - battles[k].side, spec, oc)
                    continue
                if remote_policies:
                    acts = np.split(given[spec], len(ks))
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
                o = learner_obs(b)
                done_out.append((over, eliminated))
                final_out.append(o)
                if over:
                    sim, me = b.sim, b.side
                    w = sim.winner()
                    results.append((b.opp, sim.team_score(me) - sim.team_score(1 - me),
                                    0 if w < 0 else (1 if w == me else -1)))
                    battles[k] = b = new_battle()
                    o = learner_obs(b)
                obs_out.append(o)
            reply = (np.concatenate(obs_out), np.concatenate(rew_out), done_out, final_out, results)
            remote.send(reply + (opponent_requests(),) if remote_policies else reply)
        elif cmd == "reset":
            battles = [new_battle() for _ in range(n_battles)]
            obs = np.concatenate([learner_obs(b) for b in battles])
            remote.send((obs, opponent_requests()) if remote_policies else obs)
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


def _match_worker(remote, jobs: list[tuple[str, str, int, str]]):
    """Plays ``jobs`` side by side; the networks run in the main process (see ``play_matches``).

    Each round it sends ("need", {spec: obs}) for every network-flown team still
    playing, gets {spec: actions} back, and steps; finally ("done", results).
    """
    cfg, oc = make_rules(jobs[0][3])
    battles = [Battle(seed, cfg) for _, _, seed, _ in jobs]
    live = list(range(len(jobs)))
    while live:
        teams = defaultdict(list)  # spec -> [(battle, team)]
        for k in live:
            for team, spec in enumerate(jobs[k][:2]):
                if not spec.startswith("bots"):
                    teams[spec].append((k, team))
        remote.send(("need", {spec: np.concatenate([battles[k].obs(t, oc) for k, t in kt])
                              for spec, kt in teams.items()}))
        given = remote.recv()
        acts = {k: {} for k in live}
        for spec, kt in teams.items():
            for (k, t), a in zip(kt, np.split(given[spec], len(kt))):
                acts[k] |= battles[k].team_actions(t, spec, oc, actions=a)
        for k in live:
            for team, spec in enumerate(jobs[k][:2]):
                if spec.startswith("bots"):
                    acts[k] |= battles[k].team_actions(team, spec, oc)
            battles[k].step(acts[k])
        live = [k for k in live if not battle_over(battles[k])[0]]
    out = []
    for (a, b, _, _), battle in zip(jobs, battles):
        sim, w = battle.sim, battle.sim.winner()
        out.append((a, b, sim.team_score(0), sim.team_score(1), 0 if w < 0 else (1 if w == 0 else -1)))
    remote.send(("done", out))


def play_matches(jobs: list[tuple[str, str, int, str]], workers: int = 4, bank=None) -> list:
    """``play_match`` for every job, in order.

    Without ``bank``: a process pool, one battle per task, NumPy policies. With a
    ``policies.PolicyBank`` (e.g. on a GPU): ``workers`` processes each simulate
    their share of battles side by side, and every network runs in this process,
    batched across all battles. Both give the same results (greedy actions).
    """
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
    if bank is None:
        with ctx.Pool(workers) as pool:
            return pool.map(play_match, jobs, chunksize=1)
    shares = [list(range(len(jobs)))[w::workers] for w in range(workers)]
    shares = [s for s in shares if s]
    pipes, procs = [], []
    for share in shares:
        local, remote = ctx.Pipe()
        p = ctx.Process(target=_match_worker, args=(remote, [jobs[i] for i in share]), daemon=True)
        p.start()
        remote.close()
        pipes.append(local)
        procs.append(p)
    results = [None] * len(jobs)
    msgs = [pipe.recv() for pipe in pipes]
    active = list(range(len(pipes)))
    while active:
        need = [w for w in active if msgs[w][0] == "need"]
        for w in active:
            if msgs[w][0] == "done":
                for i, r in zip(shares[w], msgs[w][1]):
                    results[i] = r
        acts = bank.act_grouped([msgs[w][1] for w in need], sample=False)
        for w, a in zip(need, acts):
            pipes[w].send(a)
        active = need
        for w in active:
            msgs[w] = pipes[w].recv()
    for p in procs:
        p.join(timeout=5)
    return results


def tournament(players: list[str], battles_per_pair: int = 10, seed: int = 0, workers: int = 4,
               rules: str = "classic", bank=None):
    """Round-robin; each pair plays half its battles on each side, on the same seeds."""
    jobs = []
    for i, (a, b) in enumerate(combinations(players, 2)):
        for n in range(battles_per_pair):
            s = seed + 1000 * i + n // 2
            jobs.append((a, b, s, rules) if n % 2 == 0 else (b, a, s, rules))
    return play_matches(jobs, workers, bank)


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
