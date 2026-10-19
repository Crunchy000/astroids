"""A planner that tries every launch: project each one forward with nobody else acting, keep the best.
(With a value network on disk, the best few by this score are re-ranked by it, looking one opponent reply ahead; see
`plan_ranked`.)

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
TOP = int(os.environ.get("PW_TOP", "128"))      # most launches the value network re-ranks (best hand score first)
BUDGET_MS = float(os.environ.get("PW_BUDGET_MS", "90"))  # stop adding launches to re-rank past this many ms into a move
CHUNK = 8                                       # launches per pass through the network
LOOK = os.environ.get("PW_LOOK", "8:4")         # 'OURS:REPLIES[f]': look one reply deep (see `plan_ranked`); empty for none
LOOK_MS = float(os.environ.get("PW_LOOK_MS", "18"))  # time the look-ahead needs besides scoring its positions
LOOK_SOURCES = 12                               # the opponent's planets with the most free ships that answers come from
MIX = float(os.environ.get("PW_MIX", "0.5"))    # weight of the hand score against the value network's
MIX_OPEN = float(os.environ["PW_MIX_OPEN"]) if os.environ.get("PW_MIX_OPEN") else None  # that weight before tick OPEN_TICKS
OPEN_TICKS = 150
VALUE = os.environ.get("PW_VALUE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "value.npz"))
JUDGE = os.environ.get("PW_JUDGE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge.npz"))  # see plan_ranked
USE_POLICY = os.environ.get("PW_POLICY_CANDIDATE", "0") == "1"  # also try the neural policy's own move


def arrival_step(dist, radius, speed):
    """Index of the tick (0 is the one being decided) in which a fleet `dist` from a planet's centre arrives."""
    return np.maximum(np.floor((dist - radius) / speed) + 1.0, 0.0).astype(np.int64)


def tick_rule(owner, ships, growth, p1, p2):
    """One tick of the rules for planets, given the ships each player lands on them in it. Shapes: [M]."""
    pending = (p1 > 0) | (p2 > 0)
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
    return np.where(neutral, n_owner, p_owner), np.where(neutral, n_ships, p_ships)


def run(owner, ships, growth, a1, a2):
    """The rules for a planet given the ships each player lands on it at each tick. Shapes: [M] and [H, M]. Only ticks
    with landings are stepped: in between, an owned planet just grows."""
    cur = 0
    for k in np.nonzero(((a1 > 0) | (a2 > 0)).any(1))[0]:
        ships = np.where(owner == 0, ships, ships + growth * (k - cur))
        owner, ships = tick_rule(owner, ships, growth, a1[k], a2[k])
        cur = k + 1
    return owner, np.where(owner == 0, ships, ships + growth * (a1.shape[0] - cur))


def settle(owner, ships, growth, start, ticks, p1, p2, horizon):
    """`run` for planets that only see a few landings: the state just before tick `start` ([M]), the ticks of the
    landings ([M, E], ascending; `horizon` or more for none) and the ships each player lands then ([M, E]). Between
    landings an owned planet just grows, so only the landings are stepped. Returns owners and ships after `horizon`."""
    cur = np.asarray(start, dtype=float).copy()
    for e in range(ticks.shape[1]):
        t = ticks[:, e]
        on = t < horizon
        ships = np.where(on & (owner != 0), ships + growth * (t - cur), ships)
        o2, s2 = tick_rule(owner, ships, growth, p1[:, e], p2[:, e])
        owner, ships = np.where(on, o2, owner), np.where(on, s2, ships)
        cur = np.where(on, t + 1.0, cur)
    return owner, np.where(owner != 0, ships + growth * (horizon - cur), ships)


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
    h = a1.shape[0]
    po, ps = np.empty((h,) + owner.shape, owner.dtype), np.empty((h,) + ships.shape)
    cur = 0
    for k in [*np.nonzero(((a1 > 0) | (a2 > 0)).any(1))[0], h]:  # the ticks with landings, then the end
        end = min(k + 1, h)                                        # until then owners hold and owned planets grow
        po[cur:end] = owner
        ps[cur:end] = ships + np.arange(end - cur)[:, None] * np.where(owner == 0, 0.0, growth)
        if k < h:
            owner, ships = tick_rule(owner, ps[k], growth, a1[k], a2[k])
            cur = k + 1
    return po, ps


def landing_ticks(a1, a2):
    """[n, E] the ticks in which fleets in flight land on each planet, ascending, padded with the horizon H."""
    h = a1.shape[0]
    busy = (a1 + a2) > 0
    ticks = np.sort(np.where(busy, np.arange(h)[:, None], h).T, -1)
    return ticks[:, :max(int(busy.sum(0).max()), 1)]


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


def project(s, me, horizon, ci, cj, cn, a1, a2, po, ps, base_owner, base_ships, d):
    """Owners and ships of every planet, `horizon` ticks after each candidate launch (source ci, target cj, ships cn).
    Only those two planets differ from the nobody-launches projection (po, ps before each tick; base_owner and
    base_ships after), and each is stepped through just its own landings (see `settle`)."""
    h, c = horizon, ci.size
    ticks = landing_ticks(a1, a2)
    # target planet: as projected until this fleet lands, then through that landing and the later ones
    k_arr = arrival_step(d[ci, cj], s.radius[cj], s.speed)
    land = k_arr < h
    kc = np.minimum(k_arr, h - 1)
    ev = np.concatenate([kc[:, None], np.sort(np.where(ticks[cj] > kc[:, None], ticks[cj], h), -1)], 1)
    evc, live = np.minimum(ev, h - 1), ev < h
    p1 = np.where(live, a1[evc, cj[:, None]], 0.0)
    p2 = np.where(live, a2[evc, cj[:, None]], 0.0)
    (p1 if me == 1 else p2)[:, 0] += cn
    t_own, t_sh = settle(po[kc, cj], ps[kc, cj], s.growth[cj], kc, ev, p1, p2, h)
    # source planet: launching leaves it with fewer ships to meet whatever arrives
    sev = ticks[ci]
    sevc, slive = np.minimum(sev, h - 1), sev < h
    q1 = np.where(slive, a1[sevc, ci[:, None]], 0.0)
    q2 = np.where(slive, a2[sevc, ci[:, None]], 0.0)
    s_own, s_sh = settle(s.owner[ci].copy(), s.ships[ci] - cn, s.growth[ci], np.zeros(c), sev, q1, q2, h)
    all_owner = np.tile(base_owner, (c, 1))
    all_ships = np.tile(base_ships, (c, 1))
    rows = np.arange(c)
    all_owner[rows, cj] = np.where(land, t_own, base_owner[cj])
    all_ships[rows, cj] = np.where(land, t_sh, base_ships[cj])
    all_owner[rows, ci], all_ships[rows, ci] = s_own, s_sh
    return all_owner, all_ships


def candidates(s, me, horizon=HORIZON, fractions=FRACTIONS, extra=None, extras=(), keep=False):
    """Every launch the rules allow and where it leaves the map `horizon` ticks on, with the hand score of that.

    Returns None when there is nothing to launch from, else a dict: ci, cj, cn (source, target, ships), owner and
    ships ([C, n] after the horizon), score ([C]), and the same for waiting: base_owner, base_ships, base.
    `fractions` are shares of the source's ships; `extras` more sizes (see `parse_size`), dropped where they repeat an
    earlier size for the same source and target. With `keep`, the dict also holds what `replies` needs: the arrivals
    (a1, a2), distances (d), the score's neighbours (near) and the projection tick by tick (po, ps).
    """
    n = s.ships.shape[0]
    horizon = max(1, min(horizon, int(s.max_ticks - s.tick)))
    a1, a2 = arrivals(s, horizon)
    d = np.sqrt(((s.pos[:, None, :] - s.pos[None, :, :]) ** 2).sum(-1))
    near = (d <= 160.0) & ~np.eye(n, dtype=bool)
    po, ps = base_states(s.owner.copy(), s.ships.copy(), s.growth, a1, a2)
    base_owner, base_ships = run(po[-1], ps[-1], s.growth, a1[-1:], a2[-1:])
    base = score(base_owner[None, :], base_ships[None, :], s, me, near)[0]

    srcs = np.nonzero((s.owner == me) & ~s.tr_on & (s.ships >= 1.0))[0]
    if srcs.size == 0:
        return None
    sizes = [parse_size(e) for e in extras]
    if any(k != "f" for k, _ in sizes):  # sizes worked out from the projection
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
    all_owner, all_ships = project(s, me, horizon, ci, cj, cn, a1, a2, po, ps, base_owner, base_ships, d)
    out = dict(ci=ci, cj=cj, cn=cn, owner=all_owner, ships=all_ships, score=score(all_owner, all_ships, s, me, near),
               base_owner=base_owner, base_ships=base_ships, base=base, horizon=horizon)
    if keep:
        out.update(a1=a1, a2=a2, d=d, near=near, po=po, ps=ps)
    return out


def replies(s, me, c, opts, fractions=FRACTIONS, extras=(), top=4, sources=12, focus=False):
    """The opponent's best answers to each of our options, as pwjax.lvalue.make_reply_agent finds them.

    `c` is `candidates(s, me, ..., keep=True)` and `opts` our options as indices into it (-1 for waiting). For each
    option the opponent's planner runs on the game after our launch: launches from its `sources` planets with the most
    free ships, of sizes `fractions` and `extras` ('f' and 'spare' kinds), each projected `c["horizon"]` ticks with
    nobody else acting, ranked by the opponent's hand score. Returns the projected owners and ships after its `top` best
    answers, [len(opts), top, n], which of those exist ([len(opts), top] bool) and the answers ([len(opts), top, 3]:
    source, target, ships). With `focus` the first two answers to each option are instead the opponent's best launch at
    that option's target and at its source (by the same hand score), and the rest the best of the others.
    """
    opp, h, n, n_opt = 3 - me, c["horizon"], s.ships.shape[0], len(opts)
    a1, a2, d, po, ps = c["a1"], c["a2"], c["d"], c["po"], c["ps"]
    A = np.repeat(np.stack([a1, a2])[None], n_opt, 0)                             # [O, 2, H, n] arrivals after our launch
    ships0 = np.repeat(s.ships[None].astype(float), n_opt, 0)
    PO, PS = np.repeat(po[None], n_opt, 0), np.repeat(ps[None], n_opt, 0)       # [O, H, n] before each tick
    FO, FS = np.repeat(c["base_owner"][None], n_opt, 0), np.repeat(c["base_ships"][None], n_opt, 0)
    oo, pp = [], []
    for o, k in enumerate(opts):
        if k >= 0:
            i, j, x = int(c["ci"][k]), int(c["cj"][k]), float(c["cn"][k])
            ships0[o, i] -= x
            kk = int(arrival_step(d[i, j], s.radius[j], s.speed))
            if kk < h:
                A[o, me - 1, kk, j] += x
            oo += [o, o]
            pp += [i, j]
    if oo:  # only our source and target differ from the nobody-launches projection without our launch
        oo, pp = np.array(oo), np.array(pp)
        q1, q2 = A[oo, 0, :, pp].T, A[oo, 1, :, pp].T                                # [H, m]
        qo, qs = base_states(s.owner[pp].copy(), ships0[oo, pp], s.growth[pp], q1, q2)
        PO[oo, :, pp], PS[oo, :, pp] = qo.T, qs.T
        FO[oo, pp], FS[oo, pp] = run(qo[-1], qs[-1], s.growth[pp], q1[-1:], q2[-1:])
    free = s.valid & (s.owner == opp) & ~s.tr_on & (s.ships >= 1.0)
    src = np.argsort(-np.where(free, s.ships, -1.0), kind="stable")[:sources]
    src = src[free[src]]
    found, acts = np.zeros((n_opt, top), bool), np.zeros((n_opt, top, 3))
    out_o, out_s = np.repeat(FO[:, None], top, 1), np.repeat(FS[:, None], top, 1)
    if src.size == 0:
        return out_o, out_s, found, acts
    # launch sizes [O, S, F]
    sizes = [np.broadcast_to(np.maximum(1.0, np.floor(s.ships[src] * f)), (n_opt, src.size)) for f in fractions]
    kept = (PO[:, :, src] == opp).all(1)
    spare = np.where(kept, np.floor(PS[:, :, src].min(1)), 0.0)                  # [O, S]
    for kind, v in (parse_size(e) for e in extras):
        assert kind in ("f", "spare"), kind
        sizes.append(np.floor(spare * v[0]) if kind == "spare" else np.broadcast_to(np.maximum(1.0, np.floor(s.ships[src] * v[0])), (n_opt, src.size)))
    x = np.stack(sizes, -1)                                                       # [O, S, F]
    ok = (x >= 1.0) & (x <= s.ships[src][None, :, None])
    for f in range(len(fractions), x.shape[-1]):
        ok[..., f] &= ~(x[..., :f] == x[..., f:f + 1]).any(-1)
    cand = ok[..., None] & s.valid[None, None, None, :] & (np.arange(n)[None, None, None, :] != src[None, :, None, None])
    o_c, s_c, f_c, t_c = np.nonzero(cand)                                         # [C] option, source, size, target
    if o_c.size == 0:
        return out_o, out_s, found, acts
    i_c, x_c = src[s_c], x[o_c, s_c, f_c]
    # the target: as projected until the fleet lands, then stepped through its landings and the later ones
    k_c = arrival_step(d[i_c, t_c], s.radius[t_c], s.speed)
    lands = k_c < h
    kc = np.minimum(k_c, h - 1)
    busy = (A[:, 0] + A[:, 1]) > 0                                                # [O, H, n] ticks with landings
    e_max = max(int(busy.sum(1).max()), 1)
    ticks = np.where(busy, np.arange(h)[None, :, None], h).transpose(0, 2, 1)     # [O, n, H]
    ticks = np.sort(ticks, -1)[..., :e_max]                                       # [O, n, E] landing ticks, padded
    later = ticks[o_c, t_c]                                                       # [C, E]
    later = np.sort(np.where(later > kc[:, None], later, h), -1)
    ev = np.concatenate([kc[:, None], later], 1)                                  # [C, E + 1]
    evc = np.minimum(ev, h - 1)
    live = ev < h
    p1 = np.where(live, A[o_c[:, None], 0, evc, t_c[:, None]], 0.0)
    p2 = np.where(live, A[o_c[:, None], 1, evc, t_c[:, None]], 0.0)
    (p1 if opp == 1 else p2)[:, 0] += x_c
    t_own, t_sh = settle(PO[o_c, kc, t_c], PS[o_c, kc, t_c], s.growth[t_c], kc, ev, p1, p2, h)
    t_own, t_sh = np.where(lands, t_own, FO[o_c, t_c]), np.where(lands, t_sh, FS[o_c, t_c])
    # the source: fewer ships from the start, through the landings on it
    so, ss, sf = np.nonzero(ok)
    si = src[ss]
    sev = ticks[so, si]                                                           # [B, E]
    sevc = np.minimum(sev, h - 1)
    q1 = np.where(sev < h, A[so[:, None], 0, sevc, si[:, None]], 0.0)
    q2 = np.where(sev < h, A[so[:, None], 1, sevc, si[:, None]], 0.0)
    s_own, s_sh = settle(s.owner[si].copy(), ships0[so, si] - x[so, ss, sf], s.growth[si], np.zeros(so.size), sev, q1, q2, h)
    row = np.full(ok.shape, -1)
    row[so, ss, sf] = np.arange(so.size)
    r_c = row[o_c, s_c, f_c]
    # the opponent's hand score of every answer (see `score`), from the projection without it and the two planets it
    # changes: per planet the score counts +-(1.5 ships + 20 growth + 5), less 2 growth for each lone planet
    g, near = s.growth, c["near"]
    s_own, s_sh = s_own[r_c], s_sh[r_c]
    side = lambda o: np.where(o == opp, 1.0, np.where(o == me, -1.0, 0.0))  # noqa: E731
    per = side(FO) * (1.5 * FS + 20.0 * g + 5.0)                                  # [O, n]
    sc = (per.sum(1)[o_c] - per[o_c, t_c] - per[o_c, i_c] + side(t_own) * (1.5 * t_sh + 20.0 * g[t_c] + 5.0)
          + side(s_own) * (1.5 * s_sh + 20.0 * g[i_c] + 5.0))
    mine, theirs = FO == opp, FO == me
    d_t, d_s = (t_own == opp) * 1.0 - mine[o_c, t_c], (s_own == opp) * 1.0 - mine[o_c, i_c]
    n_mine = mine.sum(1)[o_c] + d_t + d_s
    n_theirs = theirs.sum(1)[o_c] + (t_own == me) * 1.0 - theirs[o_c, t_c] + (s_own == me) * 1.0 - theirs[o_c, i_c]
    rows = np.arange(o_c.size)
    m = mine[o_c]
    m[rows, t_c], m[rows, i_c] = t_own == opp, s_own == opp
    friends = (mine * 1.0 @ near.T)[o_c] + near[t_c] * d_t[:, None] + near[i_c] * d_s[:, None]
    lone = (m & (friends == 0)) @ g
    sc = (sc - 2.0 * np.where(n_mine > 1, lone, 0.0) + 5.0 * (n_mine - n_theirs)
          + np.where((n_theirs == 0) & (n_mine > 0), 10000.0, 0.0) - np.where((n_mine == 0) & (n_theirs > 0), 10000.0, 0.0))
    # the opponent's best `top` for each option, and where they leave the map
    order = np.lexsort((-sc, o_c))                                                # by option, then best first (stable)
    first = np.searchsorted(o_c[order], np.arange(n_opt))
    count = np.bincount(o_c, minlength=n_opt)
    for o in range(n_opt):
        mine_o = order[first[o]:first[o] + int(count[o])]                        # this option's answers, best first
        if focus:
            k_opt = opts[o]
            slots = []
            for f in ((int(c["cj"][k_opt]), int(c["ci"][k_opt])) if k_opt >= 0 else (-1, -1)):
                hit = mine_o[t_c[mine_o] == f]
                slots.append(hit[0] if hit.size else -1)
            taken = [x for x in slots if x >= 0]
            slots += list(mine_o[~np.isin(mine_o, taken)][:top - 2])
            slots = np.array(slots + [-1] * (top - len(slots)))
            r = np.nonzero(slots >= 0)[0]
            pick = slots[r]
        else:
            k = min(top, int(count[o]))
            pick, r = mine_o[:k], np.arange(k)
        out_o[o, r, t_c[pick]], out_s[o, r, t_c[pick]] = t_own[pick], t_sh[pick]
        out_o[o, r, i_c[pick]], out_s[o, r, i_c[pick]] = s_own[pick], s_sh[pick]
        found[o, r] = True
        acts[o, r] = np.stack([i_c[pick], t_c[pick], x_c[pick]], -1)
    return out_o, out_s, found, acts


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


def still_features(s, me, owners, shipss, tick, dtype=float):
    """pwnet.features for several positions of the same game with no fleets in flight, all at once: planet [B, N, F]
    and global [B, G] features (the same numbers as `pwnet.features` on each; checked in tests/test_planner.py)."""
    o, sh = np.asarray(owners), np.asarray(shipss, dtype=float)
    b, n = sh.shape
    valid = s.valid
    mine, opp = valid & (o == me), valid & (o == 3 - me)
    x = (pwnet.WIDTH - s.pos[:, 0]) if me == 2 else s.pos[:, 0]
    y = (pwnet.HEIGHT - s.pos[:, 1]) if me == 2 else s.pos[:, 1]
    planet = np.zeros((b, n, pwnet.N_PLANET_FEATS), dtype)  # columns 9 to 17 describe fleets: none in flight
    shv = sh * valid
    planet[:, :, 0] = valid
    planet[:, :, 1] = mine
    planet[:, :, 2] = opp
    planet[:, :, 3] = valid & (o == 0)
    planet[:, :, 4] = shv / 50.0
    planet[:, :, 5] = np.log1p(shv) / 4.0
    planet[:, :, 6] = s.growth * 10.0 * valid
    planet[:, :, 7] = x / pwnet.WIDTH * valid
    planet[:, :, 8] = y / pwnet.HEIGHT * valid
    planet[:, :, 18] = np.where(mine, shv, -shv) / 50.0
    planet[:, :, 19] = mine & (sh >= 1.0)
    planet[:, :, 20] = s.radius / 40.0 * valid
    my_total, opp_total = (sh * mine).sum(1), (sh * opp).sum(1)
    my_growth, opp_growth = mine @ s.growth, opp @ s.growth
    n_valid = max(float(valid.sum()), 1.0)
    tick, max_ticks = float(tick), float(s.max_ticks)
    glob = np.empty((b, pwnet.N_GLOBAL_FEATS), dtype)
    glob[:, 0], glob[:, 1], glob[:, 2], glob[:, 3] = tick / max_ticks, (max_ticks - tick) / 1000.0, s.speed / 5.0, n_valid / 30.0
    glob[:, 4] = my_total / (my_total + opp_total + 1.0)
    glob[:, 5] = my_growth / (my_growth + opp_growth + 1e-6)
    glob[:, 6], glob[:, 7] = mine.sum(1) / n_valid, opp.sum(1) / n_valid
    glob[:, 8] = np.log1p(my_total + opp_total) / 8.0
    return planet, glob


def value_many(params, heads, s, me, owners, shipss, tick):
    """`value_of` for several positions of the same game at once (one batched pass through the network)."""
    planet, glob = still_features(s, me, owners, shipss, tick, next(iter(params.values())).dtype)
    valid = np.broadcast_to(s.valid, (len(owners), s.valid.shape[0]))
    return pwnet.value_fast(pwnet.prepare(params, heads), planet, glob, valid)


def totals_of(params, heads, s, me, owners, shipss, hand, tick, mix):
    """The ranking score of projected positions: (1 - mix) * value + mix * tanh(hand / 200), and 10 more (less) where
    the opponent (we) has no planet left."""
    owners = np.asarray(owners)
    v = value_many(params, heads, s, me, list(owners), list(shipss), tick)
    won, lost = ~(owners == 3 - me).any(-1), ~(owners == me).any(-1)
    return ((1.0 - mix) * np.asarray(v, float) + mix * np.tanh(np.asarray(hand) / 200.0)
            + np.where(won & ~lost, 10.0, 0.0) - np.where(lost & ~won, 10.0, 0.0))


def plan_ranked(s, me, params, heads, horizon=HORIZON, fractions=FRACTIONS, top=TOP, mix=MIX, budget_ms=BUDGET_MS, extras=(),
                look=LOOK, look_ms=LOOK_MS, mix_open=MIX_OPEN, judge=None):
    """Like `plan`, but the best launches by hand score (and waiting) are re-ranked by the value network, mixed with
    the hand score as (1 - mix) * value + mix * tanh(hand / 200). Launches go through the network eight at a time,
    best hand score first, until `top` have been seen or `budget_ms` (counted from the start of this call) would be
    exceeded by another batch, so a slow machine looks at fewer. Returns (src, dst, ships) or None to wait.

    With `look` = 'OURS:REPLIES' the ranking looks one reply deep, as pwjax.lvalue.make_reply_agent does: the best OURS
    options so far are each worth the worst of their own score and the scores after the opponent's REPLIES best answers
    by its hand score (see `replies`). Re-ranking then stops early enough to leave time for that: `look_ms` plus, for
    scoring the OURS * REPLIES positions, the slowest batch of eight so far for every eight of them. If less time is
    left than that, fewer options look ahead (none if fewer than two would). With `mix_open`, that is the hand score's
    weight before tick OPEN_TICKS.

    With `judge` (params, heads, its time per position over the network's), a second, bigger network scores the
    look-ahead: the chosen options themselves and the positions after the answers, as pwjax.lvalue.make_reply_agent
    does with `judge`. The time it needs is reserved the same way, scaled by that ratio."""
    started = time.perf_counter()
    if mix_open is not None and s.tick < OPEN_TICKS:
        mix = mix_open
    ours, n_rep = (int(x.rstrip("f")) for x in look.split(":")) if look else (0, 0)
    focus = look.endswith("f")
    ratio, per_opt = (judge[2], n_rep + 1) if judge is not None else (1.0, n_rep)  # look-ahead cost per option, in positions
    c = candidates(s, me, horizon, fractions, None, extras, keep=ours > 0)
    if c is None:
        return None
    order = np.argsort(-c["score"], kind="stable")[:top]
    tick = s.tick + c["horizon"]
    # launches that leave the map the same (say, fleets that land after the horizon) are scored once, at the first
    _, first, same = np.unique(np.concatenate([c["owner"][order], c["ships"][order]], 1), axis=0, return_index=True,
                               return_inverse=True)
    rep = first[same.reshape(-1)]                                                 # each launch's first like it
    fresh = np.nonzero(rep == np.arange(order.size))[0]
    tot, done, took = np.zeros(order.size), np.zeros(order.size, bool), 0.0
    for lo in range(0, fresh.size, CHUNK):
        reserve = (look_ms + took * ratio * ours * per_opt / CHUNK) if ours else 0.0
        if lo and (time.perf_counter() - started) * 1000 + took + reserve > budget_ms:
            break
        t0 = time.perf_counter()
        idx = order[fresh[lo:lo + CHUNK]]
        owners = [c["owner"][k] for k in idx] + ([c["base_owner"]] if lo == 0 else [])
        shipss = [c["ships"][k] for k in idx] + ([c["base_ships"]] if lo == 0 else [])
        hand = np.concatenate([c["score"][idx], [c["base"]] if lo == 0 else []])
        t = totals_of(params, heads, s, me, owners, shipss, hand, tick, mix)
        tot[fresh[lo:lo + CHUNK]], done[fresh[lo:lo + CHUNK]] = t[:idx.size], True
        if lo == 0:
            wait = t[-1]
        took = max(took, (time.perf_counter() - t0) * 1000)
    seen = np.nonzero(done[rep])[0]                                               # launches scored, best hand first
    cut = int(np.searchsorted(seen, CHUNK))                                       # waiting goes after the first eight
    seen_k = np.concatenate([order[seen[:cut]], [-1], order[seen[cut:]]]).astype(int)
    seen_t = np.concatenate([tot[rep[seen[:cut]]], [wait], tot[rep[seen[cut:]]]])
    best = int(np.argmax(seen_t))  # the first best on a tie, as before
    if ours and took > 0:
        left = budget_ms - (time.perf_counter() - started) * 1000 - look_ms
        ours = min(ours, int(left / (took * ratio / CHUNK)) // per_opt)
    if ours >= 2:
        pick = np.argsort(-seen_t, kind="stable")[:ours]
        opts = seen_k[pick]
        r_own, r_sh, found, _ = replies(s, me, c, opts, fractions, extras, n_rep, LOOK_SOURCES, focus)
        if judge is None:
            worst = seen_t[pick].copy()
        else:  # the judge's own view of each option
            params, heads = judge[0], judge[1]
            worst = totals_of(params, heads, s, me, [c["owner"][k] if k >= 0 else c["base_owner"] for k in opts],
                              [c["ships"][k] if k >= 0 else c["base_ships"] for k in opts],
                              [c["score"][k] if k >= 0 else c["base"] for k in opts], tick, mix)
        if found.any():
            o_idx, r_idx = np.nonzero(found)
            ro, rs = r_own[o_idx, r_idx], r_sh[o_idx, r_idx]
            r_tot = totals_of(params, heads, s, me, ro, rs, score(ro, rs, s, me, c["near"]), tick, mix)
            np.minimum.at(worst, o_idx, r_tot)
        best = int(pick[int(np.argmax(worst))])
    k = int(seen_k[best])
    if k < 0:
        return None
    return int(c["ci"][k]), int(c["cj"][k]), float(c["cn"][k])


def speed_ratio(net, judge, n=30, reps=5):
    """How many times longer `judge` takes than `net` (each (params, heads)) per position, timed on a dummy batch."""
    def took(params, heads):
        g = pwnet.N_GLOBAL_FEATS
        f = params["in.w"].shape[0] - g
        planet, glob = np.zeros((CHUNK, n, f), np.float32), np.zeros((CHUNK, g), np.float32)
        valid = np.ones((CHUNK, n), bool)
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            pwnet.value_fast(pwnet.prepare(params, heads), planet, glob, valid)
            best = min(best, time.perf_counter() - t0)
        return best
    took(*net), took(*judge)  # warm up
    return max(1.0, took(*judge) / took(*net))


class PlannerAgent(NeuralAgent):
    """Tries every launch each tick (see the module docstring)."""

    def __init__(self, horizon=HORIZON, fractions=FRACTIONS, use_policy=USE_POLICY, value=VALUE, top=TOP, mix=MIX, budget_ms=BUDGET_MS, extras=SIZES,
                 look=LOOK, judge=JUDGE):
        super().__init__()
        self.horizon, self.fractions, self.use_policy, self.extras, self.look = horizon, tuple(fractions), use_policy, tuple(extras), look
        self.value, self.top, self.mix, self.budget_ms = None, top, mix, budget_ms
        if value and value != "off" and os.path.exists(value):
            params, heads = load(value)  # ranks the planner's best launches (single precision: twice as fast, same picks)
            self.value = ({k: v.astype(np.float32) for k, v in params.items()}, heads)
        self.judge = None
        if self.value is not None and look and judge and judge != "off" and os.path.exists(judge):
            jp, jh = load(judge)
            jp = {k: v.astype(np.float32) for k, v in jp.items()}
            self.judge = (jp, jh, speed_ratio(self.value, (jp, jh)))

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
                    best = plan_ranked(s, me, *self.value, self.horizon, self.fractions, self.top, self.mix, self.budget_ms, self.extras, self.look,
                                       judge=self.judge)
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
