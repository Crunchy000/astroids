"""Short play-outs of the planner's best few launches, within the move's time (see `search`).

A slower "teacher" that plays each of the entry's best options forward on a copy of the game, both sides moved by the
planner (the network re-ranking its 16 best launches by hand score), and keeps the option that ends up best, beats the
entry by a wide margin (pwjax/teacher.py: 4 options played 15 ticks forward won 52 of 64). This is that search for
the entry itself: the game is stepped as arrays (`step`, the official engine's rules, as pwjax.env.step), and the
options are advanced a tick at a time together, so whenever time runs out they have all been played equally far.
"""
import time
from types import SimpleNamespace

import numpy as np

FIELDS = ("owner", "ships", "tr_on", "tr_owner", "tr_dest", "tr_ships", "tr_pos", "tr_vel")


def copy_state(s):
    return SimpleNamespace(**{k: (v.copy() if k in FIELDS else v) for k, v in vars(s).items()})


def launch(s, player, action):
    """The engine's launch rules for one player's action (src, dst, ships) or None, in place."""
    if action is None:
        return
    src, dst, num = int(action[0]), int(action[1]), float(action[2])
    n = s.ships.shape[0]
    if not (0 <= src < n and 0 <= dst < n) or not np.isfinite(num) or num <= 0:
        return
    if s.tr_on[src] or s.owner[src] != player or s.ships[src] < num:
        return
    d = s.pos[dst] - s.pos[src]
    mag = float(np.sqrt((d * d).sum()))
    s.ships[src] -= num
    s.tr_on[src], s.tr_owner[src], s.tr_dest[src], s.tr_ships[src] = True, player, dst, num
    s.tr_pos[src] = s.pos[src]
    s.tr_vel[src] = (d / mag if mag > 0 else d) * s.speed


def step(s, a1, a2):
    """One tick of the game (ForwardModel.step), returning a new state: launches, then fleets move or land, then the
    planets (landings fought out, owned planets grow)."""
    s = copy_state(s)
    launch(s, 1, a1)
    launch(s, 2, a2)
    n = s.ships.shape[0]
    dist = np.sqrt(((s.tr_pos - s.pos[s.tr_dest]) ** 2).sum(-1))
    arrive = s.tr_on & (dist < s.radius[s.tr_dest])
    p1_in, p2_in, pending = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for f in np.nonzero(arrive)[0]:
        j = s.tr_dest[f]
        pending[j] = True
        (p1_in if s.tr_owner[f] == 1 else p2_in)[j] += s.tr_ships[f]
    moving = s.tr_on & ~arrive
    s.tr_pos = np.where(moving[:, None], s.tr_pos + s.tr_vel, s.tr_pos)
    s.tr_on = moving
    neutral = s.owner == 0
    net = p1_in - p2_in
    n_ships = s.ships - np.abs(net)
    n_flip = pending & (n_ships < 0)
    n_owner = np.where(n_flip, np.where(net > 0, 1, 2), s.owner)
    n_ships = np.where(pending, np.where(n_flip, -n_ships, n_ships), s.ships)
    own_in = np.where(s.owner == 1, p1_in, p2_in)
    opp_in = np.where(s.owner == 1, p2_in, p1_in)
    p_ships = s.ships + s.growth + np.where(pending, own_in - opp_in, 0.0)
    p_flip = pending & (p_ships < 0)
    p_owner = np.where(p_flip, 3 - s.owner, s.owner)
    p_ships = np.where(p_flip, -p_ships, p_ships)
    s.owner = np.where(neutral, n_owner, p_owner)
    s.ships = np.where(neutral, n_ships, p_ships)
    s.tick = s.tick + 1
    return s


def over(s):
    return s.tick > s.max_ticks or not (s.owner == 1).any() or not (s.owner == 2).any()


def can_launch(s, player):
    return bool(((s.owner == player) & ~s.tr_on & (s.ships >= 1.0)).any())


def end_value(s, me, params, heads, horizon, mix):
    """What a play-out's last position is worth to `me`, as the entry ranks a projection: nobody launches for
    `horizon` ticks, then the network mixed with the hand score (10 more or less for a clear win or loss). A game
    that ended counts 10 for a win, -10 for a loss."""
    import planner as P
    if over(s):
        mine = s.ships[s.owner == me].sum()
        theirs = s.ships[s.owner == 3 - me].sum()
        return 10.0 * float(np.sign(mine - theirs))
    h = max(1, min(horizon, int(s.max_ticks - s.tick)))
    a1, a2 = P.arrivals(s, h)
    po, ps = P.base_states(s.owner.copy(), s.ships.copy(), s.growth, a1, a2)
    o, sh = P.run(po[-1], ps[-1], s.growth, a1[-1:], a2[-1:])
    d = np.sqrt(((s.pos[:, None, :] - s.pos[None, :, :]) ** 2).sum(-1))
    near = (d <= 160.0) & ~np.eye(s.ships.shape[0], dtype=bool)
    hand = P.score(o[None], sh[None], s, me, near)
    return float(P.totals_of(params, heads, s, me, [o], [sh], hand, s.tick + h, mix)[0])


COST = {"roll": 0.005, "value": 0.004}  # recent seconds per planner call in a play-out and per option valued


class OutOfTime(Exception):
    pass


def search(s, me, params, heads, options, ticks, deadline, roll_top=16, horizon=90, mix=0.5, extras=("spare",), hint=0.0):
    """Plays each of `options` (actions (src, dst, ships), or None to wait) forward up to `ticks` ticks, both sides
    moved by the planner re-ranking its `roll_top` best launches, and values where each ends up (`end_value`).

    All options advance one tick at a time. Before every planner call the time it will take (the slowest so far, or a
    recent average) and the time to value the options are checked against `deadline` (a time.perf_counter() value);
    if they would not fit, the tick in progress is dropped, so all options stay played equally far. Returns (index of
    the best option, values, ticks played), or None if not even the first tick fitted. `hint`: seconds a planner call
    is expected to take here at least (the entry's own candidates took about that long this move)."""
    import planner as P
    reserve = COST["value"] * 1.5 * len(options)
    slowest = [max(COST["roll"], hint)]

    def roll(st, side):
        if not can_launch(st, side):
            return None
        t = time.perf_counter()
        if t + 1.25 * max(slowest) + reserve > deadline:
            raise OutOfTime
        a = P.plan_ranked(st, side, params, heads, horizon, P.FRACTIONS, roll_top, mix, 1e9, extras, look="",
                          pairs=0, pair_kinds="")
        dt = time.perf_counter() - t
        slowest.append(dt)
        COST["roll"] = 0.9 * COST["roll"] + 0.1 * dt
        return a

    try:
        opp = roll(s, 3 - me)
    except OutOfTime:
        return None
    states = [step(s, a if me == 1 else opp, opp if me == 1 else a) for a in options]
    played = 1
    try:
        while played < ticks:
            nxt = [st if over(st) else step(st, roll(st, 1), roll(st, 2)) for st in states]
            states, played = nxt, played + 1
    except OutOfTime:
        pass
    t = time.perf_counter()
    vals = np.array([end_value(st, me, params, heads, horizon, mix) for st in states])
    COST["value"] = 0.9 * COST["value"] + 0.1 * (time.perf_counter() - t) / len(options)
    return int(np.argmax(vals)), vals, played
