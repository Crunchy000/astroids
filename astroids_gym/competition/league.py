"""A round-robin league with TrueSkill ratings and a Markdown leaderboard, as Planet Wars RTS runs.

Every pair of entries plays ``games_per_pair`` matches: half with each side,
each pair of sides on the same seed, so both teams get the same random match
settings and the same map. Ratings use TrueSkill (each entry starts at
μ = 25, σ = 25/3) and the leaderboard ranks by the cautious estimate μ − 3σ,
as the Planet Wars leaderboard does.

Results are appended to a JSON-lines file as matches finish. Rerunning the
league with the same file only plays what's missing, so adding an entry
plays just its matches.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Callable

from .agents import make_agent
from .params import random_params

_SQRT2 = math.sqrt(2.0)


class _Normal:
    """Standard normal pdf / cdf, accurate far into the tails (``statistics.NormalDist.cdf`` isn't)."""

    @staticmethod
    def pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / _SQRT2)

    @staticmethod
    def inv_cdf(p: float) -> float:
        return NormalDist().inv_cdf(p)


_N = _Normal()


# ---- TrueSkill (two players, one team each) -----------------------------------------
@dataclass
class Rating:
    mu: float = 25.0
    sigma: float = 25.0 / 3

    @property
    def conservative(self) -> float:
        return self.mu - 3 * self.sigma


class TrueSkill:
    """The TrueSkill update for one game between two players (win, loss or draw)."""

    def __init__(self, mu: float = 25.0, sigma: float = 25.0 / 3, beta: float | None = None,
                 tau: float | None = None, draw_probability: float = 0.10):
        self.mu, self.sigma = mu, sigma
        self.beta = sigma / 2 if beta is None else beta
        self.tau = sigma / 100 if tau is None else tau
        self.margin = _N.inv_cdf((draw_probability + 1) / 2) * math.sqrt(2) * self.beta

    def new(self) -> Rating:
        return Rating(self.mu, self.sigma)

    @staticmethod
    def _v_win(t: float, e: float) -> float:
        d = _N.cdf(t - e)
        return _N.pdf(t - e) / d if d > 1e-300 else -(t - e)

    @staticmethod
    def _v_draw(t: float, e: float) -> float:
        d = _N.cdf(e - t) - _N.cdf(-e - t)
        if d < 1e-300:
            return -t - e if t > 0 else -t + e
        return (_N.pdf(-e - t) - _N.pdf(e - t)) / d

    @staticmethod
    def _w_draw(t: float, e: float) -> float:
        d = _N.cdf(e - t) - _N.cdf(-e - t)
        if d < 1e-300:
            return 1.0
        v = TrueSkill._v_draw(t, e)
        return v * v + ((e - t) * _N.pdf(e - t) + (e + t) * _N.pdf(e + t)) / d

    def rate(self, a: Rating, b: Rating, outcome: float) -> tuple[Rating, Rating]:
        """New ratings after ``a`` beats ``b`` (``outcome`` 1), draws (0.5) or loses (0)."""
        if outcome < 0.5:
            nb, na = self.rate(b, a, 1.0)
            return na, nb
        va, vb = a.sigma ** 2 + self.tau ** 2, b.sigma ** 2 + self.tau ** 2
        c2 = 2 * self.beta ** 2 + va + vb
        c = math.sqrt(c2)
        t, e = (a.mu - b.mu) / c, self.margin / c
        if outcome > 0.5:
            v = self._v_win(t, e)
            w = v * (v + t - e)
        else:
            v, w = self._v_draw(t, e), self._w_draw(t, e)
        w = min(max(w, 0.0), 1.0)
        return (Rating(a.mu + va / c * v, math.sqrt(va * (1 - va / c2 * w))),
                Rating(b.mu - vb / c * v, math.sqrt(vb * (1 - vb / c2 * w))))


# ---- schedule and play ----------------------------------------------------------------
def schedule(names: list[str], games_per_pair: int = 4, seed: int = 0) -> list[tuple[str, str, int]]:
    """(blue, red, match seed) for every match; each seed is played once from each side."""
    jobs = []
    for i, (a, b) in enumerate(combinations(names, 2)):
        for n in range(games_per_pair):
            s = seed + 1000 * i + n // 2
            jobs.append((a, b, s) if n % 2 == 0 else (b, a, s))
    return jobs


def match_key(blue: str, red: str, seed: int) -> str:
    return f"{blue}|{red}|{seed}"


def play_job(args) -> dict:
    """One scheduled match -> its result as JSON (agents built from their specs, then closed)."""
    blue, red, seed, specs, realtime, fixed = args
    agents = [make_agent(specs[blue]), make_agent(specs[red])]
    try:
        from .runner import run_match
        result = run_match(agents[0], agents[1], random_params(seed, fixed=fixed), seed=seed, realtime=realtime,
                           names=(blue, red)).to_json()
    finally:
        for a in agents:
            if hasattr(a, "close"):
                a.close()
    result.pop("params", None)
    result["key"] = match_key(blue, red, seed)
    return result


def load_results(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_league(entries: dict[str, str], games_per_pair: int = 4, seed: int = 0, realtime: bool = True,
               workers: int = 1, results_path: str | None = None, fixed: bool = False,
               progress: Callable[[str], None] | None = print) -> list[dict]:
    """Play every missing match of the league; returns all results for these entries."""
    names = list(entries)
    jobs = schedule(names, games_per_pair, seed)
    done = {r["key"]: r for r in load_results(results_path)}
    todo = [(b, r, s, entries, realtime, fixed) for b, r, s in jobs if match_key(b, r, s) not in done]
    if progress:
        progress(f"{len(jobs)} matches, {len(jobs) - len(todo)} already played, {len(todo)} to play")
    out = open(results_path, "a") if results_path else None

    def record(res: dict, k: int) -> None:
        done[res["key"]] = res
        if out:
            out.write(json.dumps(res) + "\n")
            out.flush()
        if progress:
            b, r = res["names"]["blue"], res["names"]["red"]
            sc = res["score"]
            progress(f"[{k}/{len(todo)}] {b} {sc['blue']:.1f} - {sc['red']:.1f} {r}: {res['winner']}"
                     f" ({res['wall_seconds']:.1f} s)")

    try:
        if workers <= 1:
            for k, job in enumerate(todo, 1):
                record(play_job(job), k)
        else:
            # One BLAS thread per process: several processes with a thread per core each
            # oversubscribe the CPU, and spinning BLAS threads slow small matrices badly.
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for k, res in enumerate(pool.imap_unordered(play_job, todo), 1):
                    record(res, k)
    finally:
        if out:
            out.close()
    return [done[match_key(b, r, s)] for b, r, s in jobs]


# ---- ratings and tables -----------------------------------------------------------------
def ratings(results: list[dict], env: TrueSkill | None = None) -> dict[str, Rating]:
    """TrueSkill ratings from results, applied in a fixed order (by seed, then sides)."""
    env = env or TrueSkill()
    table: dict[str, Rating] = {}
    for res in sorted(results, key=lambda r: (r["seed"], r["names"]["blue"], r["names"]["red"])):
        b, r = res["names"]["blue"], res["names"]["red"]
        rb, rr = table.setdefault(b, env.new()), table.setdefault(r, env.new())
        outcome = {"blue": 1.0, "red": 0.0}.get(res["winner"], 0.5)
        table[b], table[r] = env.rate(rb, rr, outcome)
    return table


def summarise(results: list[dict]) -> dict[str, dict]:
    """Per entry: matches, wins, draws, losses, win rate (draws half), mean margin, timing, forfeits."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for res in results:
        for me, other in (("blue", "red"), ("red", "blue")):
            a = acc[res["names"][me]]
            a["matches"] += 1
            a["won"] += res["winner"] == me
            a["drawn"] += res["winner"] == "draw"
            a["lost"] += res["winner"] == other
            a["margin"] += res["score"][me] - res["score"][other]
            st = res["stats"][me]
            a["actions"] += st["actions"]
            a["late"] += st["late"]
            a["ms"] += st["mean_ms"] * st["actions"]
            a["forfeits"] += st["forfeit"] is not None
    out = {}
    for name, a in acc.items():
        n = a["matches"]
        out[name] = {
            "matches": int(n), "won": int(a["won"]), "drawn": int(a["drawn"]), "lost": int(a["lost"]),
            "win_rate": (a["won"] + 0.5 * a["drawn"]) / n, "margin": a["margin"] / n,
            "decision_ms": a["ms"] / a["actions"] if a["actions"] else 0.0,
            "late": a["late"] / max(a["late"] + a["actions"], 1), "forfeits": int(a["forfeits"]),
        }
    return out


def head_to_head(results: list[dict]) -> dict[tuple[str, str], float]:
    """(a, b) -> a's win rate against b over both sides (draws half)."""
    score: dict[tuple[str, str], list[float]] = defaultdict(list)
    for res in results:
        b, r = res["names"]["blue"], res["names"]["red"]
        x = {"blue": 1.0, "red": 0.0}.get(res["winner"], 0.5)
        score[b, r].append(x)
        score[r, b].append(1 - x)
    return {k: sum(v) / len(v) for k, v in score.items()}


def leaderboard_markdown(results: list[dict], title: str = "Astorids league", note: str = "") -> str:
    rate, summ, h2h = ratings(results), summarise(results), head_to_head(results)
    order = sorted(rate, key=lambda n: -rate[n].conservative)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = [f"# {title}: leaderboard", "",
             f"{len(results)} matches, updated {stamp}. TrueSkill ratings (every entry starts at μ = 25, σ = 8.33), "
             "ranked by μ − 3σ. Win rate counts draws as half; margin is kills plus zone points per match; "
             "late is the share of decision steps an entry's team coasted waiting for its answer.", ""]
    if note:
        lines += [note, ""]
    lines += ["| # | Agent | μ | σ | μ − 3σ | Matches | W-D-L | Win rate | Margin | Decision ms | Late |",
              "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for k, n in enumerate(order, 1):
        r, s = rate[n], summ[n]
        forfeit = f" ({s['forfeits']} forfeited)" if s["forfeits"] else ""
        lines.append(f"| {k} | {n} | {r.mu:.2f} | {r.sigma:.2f} | {r.conservative:.2f} | {s['matches']} | "
                     f"{s['won']}-{s['drawn']}-{s['lost']}{forfeit} | {100 * s['win_rate']:.0f}% | "
                     f"{s['margin']:+.1f} | {s['decision_ms']:.1f} | {100 * s['late']:.1f}% |")
    lines += ["", "## Head to head", "", "Row's win rate against column.", "",
              "| | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for a in order:
        cells = ["" if a == b else (f"{100 * h2h[a, b]:.0f}%" if (a, b) in h2h else "n/a") for b in order]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
