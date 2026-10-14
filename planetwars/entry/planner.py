"""A planner that tries every launch: project each one forward with nobody else acting, keep the best.

Each tick every (source, target, amount) the rules allow is scored by where the game stands `HORIZON` ticks
later if nothing else is launched by either side. With no new launches, every planet's future depends only on
the fleets already heading for it, so one move changes just two planets (the source, now smaller, and the
target, with one more arrival). That lets all moves be projected at once as arrays, in a few milliseconds,
instead of stepping a copy of the game for each of them.

The score counts ships on planets, which is also how the game is decided: ship difference, ten times the
growth difference, half the difference in ships after another 20 ticks of growth, five per planet of lead,
a small penalty for planets with no friendly planet nearby, and a huge value for winning or losing outright.
"""
import os
import time

import numpy as np

from agent import OWNER, NeuralAgent, arrays, load, choose
from core.game_state import Action, Player

HORIZON = int(os.environ.get("PW_HORIZON", "90"))
AMOUNTS = os.environ.get("PW_AMOUNTS", "0.25,0.5,1.0")
FRACTIONS = tuple(float(x) for x in AMOUNTS.split(","))
USE_POLICY = os.environ.get("PW_POLICY_CANDIDATE", "0") == "1"  # also try the neural policy's own move


def arrival_step(dist, radius, speed):
    """Index of the tick (0 is the one being decided) in which a fleet `dist` from a planet's centre arrives."""
    return np.maximum(np.floor((dist - radius) / speed) + 1.0, 0.0).astype(np.int64)


def run(owner, ships, growth, a1, a2):
    """The rules for a planet given the ships each player lands on it at each tick. Shapes: [M] and [H, M]."""
    for k in range(a1.shape[0]):
        p1, p2 = a1[k], a2[k]
        pending = (p1 > 0) | (p2 > 0)
        if not pending.any():
            ships = np.where(owner == 0, ships, ships + growth)
            continue
        neutral = owner == 0
        net = p1 - p2
        n_ships = ships - np.abs(net)
        n_flip = pending & (n_ships < 0)
        n_owner = np.where(n_flip, np.where(net > 0, 1, 2), owner)
        n_ships = np.where(pending, np.where(n_flip, -n_ships, n_ships), ships)
        own_in = np.where(owner == 1, p1, p2)
        opp_in = np.where(owner == 1, p2, p1)
        p_ships = ships + growth + np.where(pending, own_in - opp_in, 0.0)
        p_flip = pending & (p_ships < 0)
        p_owner = np.where(p_flip, 3 - owner, owner)
        p_ships = np.where(p_flip, -p_ships, p_ships)
        owner = np.where(neutral, n_owner, p_owner)
        ships = np.where(neutral, n_ships, p_ships)
    return owner, ships


def arrivals(s, horizon):
    """Ships landing on each planet at each tick from the fleets in flight: two [horizon, n] arrays (player 1, 2)."""
    n = s.ships.shape[0]
    a = np.zeros((2, horizon, n))
    for f in np.nonzero(s.tr_on)[0]:
        d = s.tr_dest[f]
        k = int(arrival_step(np.linalg.norm(s.tr_pos[f] - s.pos[d]), s.radius[d], s.speed))
        if k < horizon:
            a[s.tr_owner[f] - 1, k, d] += s.tr_ships[f]
    return a[0], a[1]


def score(owner, ships, s, me, near):
    """Position score for `me` from final owners and ships, shaped [C, n] (planets only; fleets in flight do not count)."""
    mine, opp = owner == me, owner == 3 - me
    g = s.growth[None, :]
    sc = (ships * mine).sum(1) - (ships * opp).sum(1)
    sc += 10.0 * ((g * mine).sum(1) - (g * opp).sum(1))
    fut = ships + 20.0 * g
    sc += 0.5 * ((fut * mine).sum(1) - (fut * opp).sum(1))
    n_mine, n_opp = mine.sum(1), opp.sum(1)
    isolated = mine & ((mine.astype(float) @ near.T) == 0) & (n_mine[:, None] > 1)
    sc -= 2.0 * (g * isolated).sum(1)
    sc += 5.0 * (n_mine - n_opp)
    sc += np.where((n_opp == 0) & (n_mine > 0), 10000.0, 0.0) - np.where((n_mine == 0) & (n_opp > 0), 10000.0, 0.0)
    return sc


def project(s, me, horizon, ci, cj, cn, a1, a2, base_owner, base_ships, d):
    """Owners and ships of every planet, `horizon` ticks after each candidate launch (source ci, target cj, ships cn)."""
    c = ci.size
    k_arr = arrival_step(d[ci, cj], s.radius[cj], s.speed)
    # target planet: the existing arrivals plus this fleet
    t1, t2 = a1[:, cj].copy(), a2[:, cj].copy()
    land = k_arr < horizon
    (t1 if me == 1 else t2)[k_arr[land], np.nonzero(land)[0]] += cn[land]
    # source planet: launching leaves it with fewer ships to meet whatever arrives
    s1, s2 = a1[:, ci], a2[:, ci]
    owner = np.concatenate([s.owner[cj], s.owner[ci]])
    ships = np.concatenate([s.ships[cj], s.ships[ci] - cn])
    growth = np.concatenate([s.growth[cj], s.growth[ci]])
    f_owner, f_ships = run(owner, ships, growth, np.concatenate([t1, s1], axis=1), np.concatenate([t2, s2], axis=1))
    all_owner = np.tile(base_owner, (c, 1))
    all_ships = np.tile(base_ships, (c, 1))
    rows = np.arange(c)
    all_owner[rows, cj], all_ships[rows, cj] = f_owner[:c], f_ships[:c]
    all_owner[rows, ci], all_ships[rows, ci] = f_owner[c:], f_ships[c:]
    return all_owner, all_ships


def plan(s, me, horizon=HORIZON, fractions=FRACTIONS, extra=None):
    """Best launch as (src, dst, ships), or None to wait. `extra`: more (src, dst, ships) moves to consider."""
    n = s.ships.shape[0]
    horizon = max(1, min(horizon, int(s.max_ticks - s.tick)))
    a1, a2 = arrivals(s, horizon)
    d = np.sqrt(((s.pos[:, None, :] - s.pos[None, :, :]) ** 2).sum(-1))
    near = (d <= 160.0) & ~np.eye(n, dtype=bool)
    base_owner, base_ships = run(s.owner.copy(), s.ships.copy(), s.growth, a1, a2)
    base = score(base_owner[None, :], base_ships[None, :], s, me, near)[0]

    srcs = np.nonzero((s.owner == me) & ~s.tr_on & (s.ships >= 1.0))[0]
    if srcs.size == 0:
        return None
    ci, cj, cn = [], [], []
    for frac in fractions:
        amt = np.maximum(1.0, np.floor(s.ships[srcs] * frac))
        ok = amt <= s.ships[srcs]
        for i, a, good in zip(srcs, amt, ok):
            if not good:
                continue
            dst = np.array([j for j in range(n) if j != i])
            ci.append(np.full(dst.size, i)); cj.append(dst); cn.append(np.full(dst.size, a))
    if extra:
        for i, j, a in extra:
            ci.append(np.array([i])); cj.append(np.array([j])); cn.append(np.array([a]))
    if not ci:
        return None
    ci, cj, cn = np.concatenate(ci), np.concatenate(cj), np.concatenate(cn)
    all_owner, all_ships = project(s, me, horizon, ci, cj, cn, a1, a2, base_owner, base_ships, d)
    sc = score(all_owner, all_ships, s, me, near)
    best = int(np.argmax(sc))
    if sc[best] > base:
        return int(ci[best]), int(cj[best]), float(cn[best])
    return None


class PlannerAgent(NeuralAgent):
    """Tries every launch each tick (see the module docstring)."""

    def __init__(self, horizon=HORIZON, fractions=FRACTIONS, use_policy=USE_POLICY):
        super().__init__()
        self.horizon, self.fractions, self.use_policy = horizon, tuple(fractions), use_policy

    def get_action(self, game_state):
        try:
            if self.player not in (Player.Player1, Player.Player2) or not game_state.planets:
                return Action.do_nothing()
            s = arrays(game_state, self.params.transporter_speed, self.params.max_ticks)
            me = OWNER[self.player]
            extra = None
            if self.use_policy:
                params, heads = load()
                i, j, a = choose(params, heads, s, me)
                extra = [(i, j, a)] if i >= 0 else None
            best = plan(s, me, self.horizon, self.fractions, extra=extra)
            if best is None:
                return Action.do_nothing()
            src, dst, ships = best
            ships = min(ships, game_state.planets[src].n_ships)
            return Action(player_id=self.player, source_planet_id=src, destination_planet_id=dst, num_ships=ships)
        except Exception as e:  # never forfeit a tick over a bug
            print("PlannerAgent error:", repr(e), flush=True)
            return Action.do_nothing()

    def get_agent_type(self) -> str:
        return "pwjax planner agent"
