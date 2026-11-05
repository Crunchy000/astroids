"""Short play-outs of the planner's best few launches, within the move's time (see `search`).

A slower "teacher" that plays each of the entry's best options forward on a copy of the game, both sides moved by the
planner (the network re-ranking its 16 best launches by hand score), and keeps the option that ends up best, beats the
entry by a wide margin (pwjax/teacher.py: 4 options played 15 ticks forward won 52 of 64). This is that search for
the entry itself: the game is stepped as arrays (`step`, the official engine's rules, as pwjax.env.step), and the
options are advanced a tick at a time together, so whenever time runs out they have all been played equally far.
"""
import os
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
PROCS = os.environ.get("PW_SEARCH_PROCS", "1")  # processes playing options at once: "auto" (one per core, at most one
                                                # per option) or a number; 1 plays them all in this process
LINK_S = 0.003                                  # time kept back for passing results between processes


class OutOfTime(Exception):
    pass


def advance(s, me, firsts, opp, ticks, deadline, params, heads, roll_top, horizon, mix, extras, hint, reserve):
    """Plays the options `firsts` forward together a tick at a time (the opponent's first move `opp` is given), both
    sides moved by the planner, until `ticks` ticks are played or a planner call would not fit before `deadline`
    less `reserve` seconds. Returns (ticks played, states after each tick: hist[k][i] is option i after k + 1)."""
    import planner as P
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

    states = [step(s, a if me == 1 else opp, opp if me == 1 else a) for a in firsts]
    hist = [states]
    try:
        while len(hist) < ticks:
            hist.append([st if over(st) else step(st, roll(st, 1), roll(st, 2)) for st in hist[-1]])
    except OutOfTime:
        pass
    return len(hist), hist


def values(states, me, params, heads, horizon, mix):
    t = time.perf_counter()
    v = [end_value(st, me, params, heads, horizon, mix) for st in states]
    COST["value"] = 0.9 * COST["value"] + 0.1 * (time.perf_counter() - t) / max(1, len(states))
    return v


def _worker(conn):
    """A helper process: plays the options it is sent (see `advance`), then values them at the depth it is told."""
    net, hist = None, None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "go":
            _, new_net, job = msg
            if new_net is not None:
                net = new_net
            try:
                played, hist = advance(**job, params=net[0], heads=net[1])
            except Exception as e:  # tell the main process, which then stops using helpers
                print("playout helper:", repr(e), flush=True)
                played, hist = None, None
            conn.send(played)
        elif msg[0] == "value":
            _, d, me, horizon, mix = msg
            try:
                conn.send(values(hist[d - 1], me, net[0], net[1], horizon, mix))
            except Exception as e:
                print("playout helper:", repr(e), flush=True)
                conn.send(None)
            hist = None


class Pool:
    """Helper processes for `search`, forked once, so they start with the code already loaded."""

    def __init__(self, n):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        self.conns, self.procs, self.nets = [], [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b,), daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)
            self.nets.append(None)

    def close(self):
        for pr in self.procs:
            pr.kill()
        for pr in self.procs:
            pr.join(0.1)


_POOL = {"pool": None, "fails": 0}
MAX_FAILS = 3  # helpers that miss their time this often are not used again


def helpers(k, procs=None):
    """How many helper processes to use for k options (0: play them all in this process); `procs` as PROCS."""
    procs = PROCS if procs is None else str(procs)
    if _POOL["fails"] >= MAX_FAILS or k < 2:
        return 0
    if procs == "auto":
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 1
    else:
        cores = int(float(procs))
    n = min(k, cores) - 1
    if n <= 0:
        return 0
    pool = _POOL["pool"]
    if pool is None or len(pool.conns) < n:
        try:
            if pool is not None:
                pool.close()
            _POOL["pool"] = Pool(n)
        except Exception as e:  # no helper processes here (say, inside a daemonic process): play everything here
            print("playout: no helper processes:", repr(e), flush=True)
            _POOL["pool"], _POOL["fails"] = None, MAX_FAILS
            return 0
    return n


def _drop_pool(why):
    print(f"playout: {why}; restarting the helpers", flush=True)
    _POOL["pool"].close()
    _POOL["pool"] = None
    _POOL["fails"] += 1


def _get(conn, wait):
    """A helper's answer within `wait` seconds, or None."""
    try:
        return conn.recv() if conn.poll(max(0.0, wait)) else None
    except (EOFError, OSError):
        return None


def start(k=4, procs=None):
    """Start the helper processes now (before serving), not during the first searched move."""
    return helpers(k, procs)


def search(s, me, params, heads, options, ticks, deadline, roll_top=16, horizon=90, mix=0.5, extras=("spare",), hint=0.0,
           procs=None):
    """Plays each of `options` (actions (src, dst, ships), or None to wait) forward up to `ticks` ticks, both sides
    moved by the planner re-ranking its `roll_top` best launches, and values where each ends up (`end_value`).

    Options advance one tick at a time. Before every planner call the time it will take (the slowest so far, or a
    recent average) and the time to value the options are checked against `deadline` (a time.perf_counter() value);
    if they would not fit, the tick in progress is dropped. With more than one core the options are shared out among
    helper processes (`PW_SEARCH_PROCS`), each playing its own as far as time allows; all are then valued at the depth
    the shortest got to, so they are always compared after the same number of ticks. Returns (index of the best
    option, values, ticks played), or None if not even the first tick fitted. `hint`: seconds a planner call is
    expected to take here at least (the entry's own candidates took about that long this move)."""
    import planner as P
    k = len(options)
    n = helpers(k, procs)
    groups = [list(range(i, k, n + 1)) for i in range(n + 1)]
    reserve = COST["value"] * 1.5 * len(groups[0]) + (LINK_S if n else 0.0)
    if time.perf_counter() + 1.25 * max(COST["roll"], hint) + reserve > deadline:
        return None
    opp = None
    if can_launch(s, 3 - me):
        t = time.perf_counter()
        opp = P.plan_ranked(s, 3 - me, params, heads, horizon, P.FRACTIONS, roll_top, mix, 1e9, extras, look="",
                            pairs=0, pair_kinds="")
        COST["roll"] = 0.9 * COST["roll"] + 0.1 * (time.perf_counter() - t)
    job = dict(s=s, me=me, opp=opp, ticks=ticks, deadline=deadline, roll_top=roll_top, horizon=horizon, mix=mix,
               extras=extras, hint=hint)
    pool, sent = _POOL["pool"], 0
    if n:
        try:
            for w, g in enumerate(groups[1:]):
                net = None if pool.nets[w] is params else (params, heads)
                pool.conns[w].send(("go", net, dict(job, firsts=[options[i] for i in g],
                                                    reserve=COST["value"] * 1.5 * len(g) + LINK_S)))
                pool.nets[w] = params
                sent += 1
        except Exception as e:
            pool.nets = [None] * len(pool.nets)
            _drop_pool(f"could not reach the helpers ({e!r})")
            sent = 0
    played, hist = advance(**dict(job, firsts=[options[i] for i in groups[0]], reserve=reserve), params=params,
                           heads=heads)
    depth = [played]
    for w in range(sent):
        got = _get(pool.conns[w], deadline - time.perf_counter() + 0.02)
        if got is not None:
            depth.append(got)
        else:  # compare only the options played here this move
            _drop_pool("a helper missed its time")
            sent, depth = 0, [played]
            break
    d = min(depth)
    try:
        for w in range(sent):
            pool.conns[w].send(("value", d, me, horizon, mix))
    except Exception as e:
        _drop_pool(f"could not reach the helpers ({e!r})")
        sent, d = 0, played
    vals = np.full(k, -np.inf)
    vals[groups[0]] = values(hist[d - 1], me, params, heads, horizon, mix)
    for w in range(sent):
        got = _get(pool.conns[w], deadline + 0.035 - time.perf_counter())
        if got is not None:
            vals[groups[w + 1]] = got
        else:
            _drop_pool("a helper missed its time valuing")
            break
    return int(np.argmax(vals)), vals, d
