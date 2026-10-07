"""Measure *how* a team plays: shooting, movement, spacing, focus fire, causes of death.

    python examples/analyze_tactics.py --model results/ppo_team.zip --battles 10

Runs the same battles with blue flown by the model and by the built-in bots
(red is always the bots) and prints blue's behaviour side by side.
"""

import argparse
import math
from collections import defaultdict

import numpy as np

from astroids_gym import AsteroidsBattleParallelEnv
from astroids_gym.sim import Simulation

# ---- instrument the simulation (records why ships take damage) ----------------
_cause = {"now": None}
_log: list = []


def _wrap(name, cause):
    orig = getattr(Simulation, name)

    def wrapped(self, *a, **k):
        prev, _cause["now"] = _cause["now"], cause
        try:
            return orig(self, *a, **k)
        finally:
            _cause["now"] = prev
    setattr(Simulation, name, wrapped)


_wrap("_ship_collisions", "ram")
_wrap("_bullets", "bullet")
_wrap("_asteroids", "asteroid")
_orig_hit, _orig_kill = Simulation._hit, Simulation._kill


def _hit(self, s, attacker, ev):
    if s.alive and s.invuln <= 0:
        _log.append(("hit", self.time, s.id, s.team, attacker.id if attacker else None, _cause["now"],
                     math.hypot(self.wdx(s.x, attacker.x), self.wdy(s.y, attacker.y)) if attacker else None))
    return _orig_hit(self, s, attacker, ev)


def _kill(self, s, attacker, ev):
    _log.append(("death", self.time, s.id, s.team, attacker.id if attacker else None, _cause["now"], None))
    return _orig_kill(self, s, attacker, ev)


Simulation._hit, Simulation._kill = _hit, _kill


def run(policy, battles, seed):
    stats = defaultdict(list)
    counts = defaultdict(float)
    for b in range(battles):
        _log.clear()
        env = AsteroidsBattleParallelEnv(learning_teams=("blue",) if policy else ())
        obs, _ = env.reset(seed=seed + b)
        sim = env.sim
        sim.sfx = []
        blue = [s for s in sim.ships if s.team == 0]
        red = [s for s in sim.ships if s.team == 1]
        while env._steps < env.max_steps:
            if policy:
                names = list(obs)
                obs, *_ = env.step(dict(zip(names, policy(np.stack([obs[a] for a in names])))))
            else:
                obs, *_ = env.step({})
            alive_b = [s for s in blue if s.alive]
            alive_r = [s for s in red if s.alive]
            for s in alive_b:
                stats["speed"].append(math.hypot(s.vx, s.vy))
                stats["thrusting"].append(float(s.thrusting))
                if alive_r:
                    stats["enemy_dist"].append(min(math.hypot(sim.wdx(s.x, o.x), sim.wdy(s.y, o.y)) for o in alive_r))
                mates = [o for o in alive_b if o is not s]
                if mates:
                    stats["ally_dist"].append(min(math.hypot(sim.wdx(s.x, o.x), sim.wdy(s.y, o.y)) for o in mates))
                stats["x"].append(s.x / sim.cfg.width)
                # How many enemy bullets are within 150 px and heading at this ship.
                stats["threats"].append(sum(
                    1 for bl in sim.bullets if bl.team != s.team
                    and (sim.wdx(bl.x, s.x) ** 2 + sim.wdy(bl.y, s.y) ** 2) < 150 ** 2
                    and sim.wdx(bl.x, s.x) * bl.vx + sim.wdy(bl.y, s.y) * bl.vy > 0))
            counts["blue_alive_frac"] += len(alive_b) / len(blue)
            counts["red_alive_frac"] += len(alive_r) / len(red)
            counts["steps"] += 1
        # Per-battle event tallies.
        shots = [e for e in sim.sfx if e[1] == "shot"]
        counts["blue_shots"] += sum(1 for e in shots if e[3] == 0)
        counts["red_shots"] += sum(1 for e in shots if e[3] == 1)
        hits = [e for e in _log if e[0] == "hit"]
        counts["blue_bullet_hits"] += sum(1 for e in hits if e[5] == "bullet" and e[3] == 1)
        counts["red_bullet_hits"] += sum(1 for e in hits if e[5] == "bullet" and e[3] == 0)
        for e in hits:
            if e[5] == "bullet" and e[3] == 1 and e[6] is not None:
                stats["blue_hit_range"].append(e[6])
        for e in _log:
            if e[0] == "death":
                side = "blue" if e[3] == 0 else "red"
                counts[f"{side}_deaths_{e[5]}"] += 1
        # Focus fire: distinct blue attackers that hit each red ship in the 3 s before it died.
        for d in (e for e in _log if e[0] == "death" and e[3] == 1 and e[5] == "bullet"):
            shooters = {e[4] for e in hits if e[2] == d[2] and d[1] - 3 <= e[1] <= d[1] and e[4] is not None}
            stats["shooters_per_kill"].append(len(shooters))
        counts["blue_score"] += sim.score[0]
        counts["red_score"] += sim.score[1]
        counts["wins"] += sim.score[0] > sim.score[1]
    return stats, counts


def summarize(stats, counts, battles):
    n = counts["steps"]
    m = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")
    kills_by_bullet = counts["red_deaths_bullet"] or 1
    return {
        "Wins": f"{counts['wins']:.0f}/{battles}",
        "Kills per battle (blue - red)": f"{counts['blue_score']/battles:.1f} - {counts['red_score']/battles:.1f}",
        "Blue shots per battle": f"{counts['blue_shots']/battles:.0f}",
        "Blue accuracy (hits / shots)": f"{counts['blue_bullet_hits']/max(counts['blue_shots'],1):.1%}",
        "Red accuracy vs blue": f"{counts['red_bullet_hits']/max(counts['red_shots'],1):.1%}",
        "Range of blue's hits (px)": f"{m('blue_hit_range'):.0f}",
        "Distance to nearest enemy (px)": f"{m('enemy_dist'):.0f}",
        "Distance to nearest teammate (px)": f"{m('ally_dist'):.0f}",
        "Speed (px/s, max 260)": f"{m('speed'):.0f}",
        "Time thrusting": f"{m('thrusting'):.0%}",
        "Enemy bullets incoming (avg)": f"{m('threats'):.2f}",
        "Avg x position (0 = blue home, 1 = red home)": f"{m('x'):.2f}",
        "Blue ships alive (avg)": f"{counts['blue_alive_frac']/n:.0%}",
        "Red ships alive (avg)": f"{counts['red_alive_frac']/n:.0%}",
        "Blue shooters per kill (focus fire)": f"{m('shooters_per_kill'):.2f}",
        "Blue deaths: bullets / asteroids / rams": (
            f"{counts['blue_deaths_bullet']/battles:.1f} / {counts['blue_deaths_asteroid']/battles:.1f} / "
            f"{counts['blue_deaths_ram']/battles:.1f}"),
        "Red deaths: bullets / asteroids / rams": (
            f"{counts['red_deaths_bullet']/battles:.1f} / {counts['red_deaths_asteroid']/battles:.1f} / "
            f"{counts['red_deaths_ram']/battles:.1f}"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--battles", type=int, default=10)
    p.add_argument("--seed", type=int, default=123_000)
    args = p.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    agent = summarize(*run(lambda o: model.predict(o, deterministic=True)[0], args.battles, args.seed), args.battles)
    bots = summarize(*run(None, args.battles, args.seed), args.battles)
    w = max(map(len, agent))
    print(f"{'Blue team flown by':<{w}}  {'PPO team':>14}  {'Bots':>14}")
    for k in agent:
        print(f"{k:<{w}}  {agent[k]:>14}  {bots[k]:>14}")


if __name__ == "__main__":
    main()
