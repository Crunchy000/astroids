"""A planner that tries every launch: project each one forward with nobody else acting, keep the best.
(With a value network on disk, the best few by this score are re-ranked by it; see `plan_ranked`.)

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

import pwnet
from agent import OWNER, NeuralAgent, arrays, load, choose
from core.game_state import Action, Player

HORIZON = int(os.environ.get("PW_HORIZON", "90"))
AMOUNTS = os.environ.get("PW_AMOUNTS", "0.25,0.5,1.0")
FRACTIONS = tuple(float(x) for x in AMOUNTS.split(","))
SIZES = tuple(x for x in os.environ.get("PW_SIZES", "spare").split("+") if x)  # the agent's launch sizes beyond the fractions: see `parse_size`
TOP = int(os.environ.get("PW_TOP", "32"))       # most launches the value network re-ranks (best hand score first)
BUDGET_MS = float(os.environ.get("PW_BUDGET_MS", "90"))  # stop adding launches to re-rank past this many ms into a move
CHUNK = 8                                       # launches per pass through the network
MIX = float(os.environ.get("PW_MIX", "0.5"))    # weight of the hand score against the value network's
VALUE = os.environ.get("PW_VALUE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "value.npz"))
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


def parse_size(spec):
    """'f:X' (a fraction of the source's ships), 'cap[:M[:C]]' (what takes the target when the fleet lands, times M, plus
    1 and C), 'spare[:X]' (a fraction of what the source can do without); 'capm' and 'spare2' are short for
    'cap:1.2:1' and 'spare:0.5'. The same sizes as pwjax.planner."""
    spec = {"capm": "cap:1.2:1", "spare2": "spare:0.5"}.get(spec, spec)
    kind, *args = spec.split(":")
    assert kind in ("f", "cap", "spare"), spec
    vals = [float(x) for x in args]
    if kind == "cap":
        vals = (vals + [1.0, 0.0])[:2] if len(vals) < 2 else vals
    elif kind == "spare":
        vals = vals or [1.0]
    return kind, tuple(vals)


def base_states(owner, ships, growth, a1, a2):
    """[H, n] owners and ships of every planet just before each tick of the nobody-launches projection."""
    po, ps = [], []
    for k in range(a1.shape[0]):
        po.append(owner)
        ps.append(ships)
        owner, ships = run(owner, ships, growth, a1[k:k + 1], a2[k:k + 1])
    return np.array(po), np.array(ps)


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


def candidates(s, me, horizon=HORIZON, fractions=FRACTIONS, extra=None, extras=()):
    """Every launch the rules allow and where it leaves the map `horizon` ticks on, with the hand score of that.

    Returns None when there is nothing to launch from, else a dict: ci, cj, cn (source, target, ships), owner and
    ships ([C, n] after the horizon), score ([C]), and the same for waiting: base_owner, base_ships, base.
    `fractions` are shares of the source's ships; `extras` more sizes (see `parse_size`), dropped where they repeat an
    earlier size for the same source and target.
    """
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
    sizes = [parse_size(e) for e in extras]
    if any(k != "f" for k, _ in sizes):  # sizes worked out from the projection
        po, ps = base_states(s.owner.copy(), s.ships.copy(), s.growth, a1, a2)
        k_all = arrival_step(d, s.radius[None, :], s.speed)                      # [n, n] tick a fleet from i lands on j
        kk = np.minimum(k_all, horizon - 1)
        cols = np.arange(n)[None, :]
        mine_in, opp_in = (a1 if me == 1 else a2)[kk, cols], (a2 if me == 1 else a1)[kk, cols]
        before_o, before_s = po[kk, cols], ps[kk, cols]
        need = before_s + np.where(before_o != 0, s.growth[None, :], 0.0) + opp_in - mine_in
        lands = (k_all < horizon) & (before_o != me)
        kept = (po == me).all(0)                                                  # the projection never loses planet i
        spare = np.where(kept, np.floor(ps.min(0)), 0.0)
    ci, cj, cn = [], [], []
    tried = {i: [] for i in srcs}                                                 # sizes already tried per source and target
    options = [("f", (f,)) for f in fractions] + sizes
    for f, (kind, v) in enumerate(options):
        for i in srcs:
            dst = np.array([j for j in range(n) if j != i])
            if f < len(fractions):
                x = np.full(dst.size, max(1.0, np.floor(s.ships[i] * v[0])))
            elif kind == "cap":
                x = np.where(lands[i, dst], np.floor(np.maximum(need[i, dst], 0.0) * v[0]) + 1.0 + v[1], 0.0)
            elif kind == "spare":
                x = np.full(dst.size, np.floor(spare[i] * v[0]))
            else:
                x = np.full(dst.size, max(1.0, np.floor(s.ships[i] * v[0])))
            ok = (x >= 1.0) & (x <= s.ships[i])
            if f >= len(fractions):  # not a size already tried for this pair
                for earlier in tried[i]:
                    ok &= x != earlier
            tried[i].append(x)
            ci.append(np.full(int(ok.sum()), i)); cj.append(dst[ok]); cn.append(x[ok])
    if extra:
        for i, j, a in extra:
            ci.append(np.array([i])); cj.append(np.array([j])); cn.append(np.array([a]))
    if not ci:
        return None
    ci, cj, cn = np.concatenate(ci), np.concatenate(cj), np.concatenate(cn)
    all_owner, all_ships = project(s, me, horizon, ci, cj, cn, a1, a2, base_owner, base_ships, d)
    return dict(ci=ci, cj=cj, cn=cn, owner=all_owner, ships=all_ships, score=score(all_owner, all_ships, s, me, near),
                base_owner=base_owner, base_ships=base_ships, base=base, horizon=horizon)


def plan(s, me, horizon=HORIZON, fractions=FRACTIONS, extra=None, extras=()):
    """Best launch as (src, dst, ships), or None to wait. `extra`: more (src, dst, ships) moves to consider."""
    c = candidates(s, me, horizon, fractions, extra, extras)
    if c is None:
        return None
    best = int(np.argmax(c["score"]))
    if c["score"][best] > c["base"]:
        return int(c["ci"][best]), int(c["cj"][best]), float(c["cn"][best])
    return None


def value_of(params, heads, s, me, owner, ships, tick):
    """The value network's view (a number in about [-1, 1]) of a position with no fleets in flight."""
    return float(value_many(params, heads, s, me, [owner], [ships], tick)[0])


def still_features(s, me, owners, shipss, tick):
    """pwnet.features for several positions of the same game with no fleets in flight, all at once: planet [B, N, F]
    and global [B, G] features (the same numbers as `pwnet.features` on each; checked in tests/test_planner.py)."""
    o, sh = np.asarray(owners), np.asarray(shipss, dtype=float)
    valid = s.valid[None, :]
    vf = np.broadcast_to(valid.astype(float), sh.shape)
    mine, opp, neu = valid & (o == me), valid & (o == 3 - me), valid & (o == 0)
    x = (pwnet.WIDTH - s.pos[:, 0]) if me == 2 else s.pos[:, 0]
    y = (pwnet.HEIGHT - s.pos[:, 1]) if me == 2 else s.pos[:, 1]
    zero = np.zeros_like(sh)
    rows = lambda v: np.broadcast_to(v[None, :], sh.shape)  # noqa: E731
    planet = np.stack([
        vf, mine.astype(float), opp.astype(float), neu.astype(float), sh / 50.0, np.log1p(sh) / 4.0,
        rows(s.growth * 10.0), rows(x / pwnet.WIDTH), rows(y / pwnet.HEIGHT),
        zero, zero, zero, zero, zero, zero, zero, zero, zero,   # fleets: none in flight
        np.where(mine, 1.0, -1.0) * sh / 50.0, (mine & (sh >= 1.0)).astype(float), rows(s.radius / 40.0),
    ], axis=-1) * vf[..., None]
    my_total, opp_total = np.where(mine, sh, 0.0).sum(1), np.where(opp, sh, 0.0).sum(1)
    my_growth, opp_growth = np.where(mine, s.growth, 0.0).sum(1), np.where(opp, s.growth, 0.0).sum(1)
    n_valid = max(float(s.valid.sum()), 1.0)
    b = sh.shape[0]
    tick, max_ticks = float(tick), float(s.max_ticks)
    glob = np.stack([
        np.full(b, tick / max_ticks), np.full(b, (max_ticks - tick) / 1000.0), np.full(b, s.speed / 5.0), np.full(b, n_valid / 30.0),
        my_total / (my_total + opp_total + 1.0), my_growth / (my_growth + opp_growth + 1e-6),
        mine.sum(1) / n_valid, opp.sum(1) / n_valid, np.log1p(my_total + opp_total) / 8.0,
    ], axis=-1)
    return planet, glob


def value_many(params, heads, s, me, owners, shipss, tick):
    """`value_of` for several positions of the same game at once (one batched pass through the network)."""
    planet, glob = still_features(s, me, owners, shipss, tick)
    dt = next(iter(params.values())).dtype
    valid = np.broadcast_to(s.valid, (len(owners), s.valid.shape[0]))
    return pwnet.value_batch(np, params, heads, planet.astype(dt), glob.astype(dt), valid)


def plan_ranked(s, me, params, heads, horizon=HORIZON, fractions=FRACTIONS, top=TOP, mix=MIX, budget_ms=BUDGET_MS, extras=()):
    """Like `plan`, but the best launches by hand score (and waiting) are re-ranked by the value network, mixed with
    the hand score as (1 - mix) * value + mix * tanh(hand / 200). Launches go through the network eight at a time,
    best hand score first, until `top` have been seen or `budget_ms` (counted from the start of this call) would be
    exceeded by another batch, so a slow machine looks at fewer. Returns (src, dst, ships) or None to wait."""
    started = time.perf_counter()
    c = candidates(s, me, horizon, fractions, None, extras)
    if c is None:
        return None
    order = np.argsort(-c["score"], kind="stable")[:top]
    tick = s.tick + c["horizon"]
    best_total, best_k, took = -np.inf, None, 0.0
    for lo in range(0, len(order), CHUNK):
        if lo and (time.perf_counter() - started) * 1000 + took > budget_ms:
            break
        t0 = time.perf_counter()
        idx = order[lo:lo + CHUNK]
        owners = [c["owner"][k] for k in idx] + ([c["base_owner"]] if lo == 0 else [])
        shipss = [c["ships"][k] for k in idx] + ([c["base_ships"]] if lo == 0 else [])
        hand = np.concatenate([c["score"][idx], [c["base"]] if lo == 0 else []])
        v = value_many(params, heads, s, me, owners, shipss, tick)
        for k, o in enumerate(owners):
            won, lost = not (o == 3 - me).any(), not (o == me).any()
            total = ((1.0 - mix) * float(v[k]) + mix * np.tanh(hand[k] / 200.0)
                     + (10.0 if won and not lost else 0.0) - (10.0 if lost and not won else 0.0))
            if total > best_total:
                best_total, best_k = total, (None if k == len(idx) else int(idx[k]))
        took = max(took, (time.perf_counter() - t0) * 1000)
    if best_k is None:
        return None
    return int(c["ci"][best_k]), int(c["cj"][best_k]), float(c["cn"][best_k])


class PlannerAgent(NeuralAgent):
    """Tries every launch each tick (see the module docstring)."""

    def __init__(self, horizon=HORIZON, fractions=FRACTIONS, use_policy=USE_POLICY, value=VALUE, top=TOP, mix=MIX, budget_ms=BUDGET_MS, extras=SIZES):
        super().__init__()
        self.horizon, self.fractions, self.use_policy, self.extras = horizon, tuple(fractions), use_policy, tuple(extras)
        self.value, self.top, self.mix, self.budget_ms = None, top, mix, budget_ms
        if value and value != "off" and os.path.exists(value):
            params, heads = load(value)  # ranks the planner's best launches (single precision: twice as fast, same picks)
            self.value = ({k: v.astype(np.float32) for k, v in params.items()}, heads)

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
            if self.value is not None and extra is None:
                try:
                    best = plan_ranked(s, me, *self.value, self.horizon, self.fractions, self.top, self.mix, self.budget_ms, self.extras)
                except Exception as e:  # fall back to the hand-written score
                    print("PlannerAgent value error:", repr(e), flush=True)
                    best = plan(s, me, self.horizon, self.fractions, extras=self.extras)
            else:
                best = plan(s, me, self.horizon, self.fractions, extra=extra, extras=self.extras)
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
