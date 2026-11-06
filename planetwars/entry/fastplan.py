"""`planner.candidates` compiled with numba: the same numbers as the numpy code, several times faster.

With 12 to 30 planets the numpy version spends most of its time starting small array operations rather than doing
arithmetic, and the play-outs (see `playout.py`) call it for both sides on every tick of every option, so it limits how
far they get within a move's time. This is the same calculation written as loops: the projection with nobody
launching, every launch the rules allow and where it leaves the map, and the hand score. It adds and rounds in the same
order as numpy (row sums in numpy's pairwise order, fleet distances as `np.linalg.norm` works them out), so the
results are equal, not just close; `tests/test_fastplan.py` checks that on positions from real games.

`planner.candidates` uses it when numba is installed, unless PW_NUMBA=0. The first call compiles it (a few seconds,
cached on disk afterwards); `warm` does that up front.
"""
import numpy as np
import numba as nb
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

KIND = {"f": 3, "cap": 1, "spare": 2}  # launch sizes beyond the fractions (kind 0); see planner.parse_size


@intrinsic
def _fma(typingctx, a, b, c):
    """a * b + c rounded once, as numpy's dot of a 2-vector with itself works it out (so as `np.linalg.norm`)."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        dbl = ir.DoubleType()
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(dbl, [dbl, dbl, dbl]), "llvm.fma.f64")
        return builder.call(fn, args)
    return sig, codegen


@nb.njit(cache=True)
def _psum(a, n):
    """a[0] + ... + a[n - 1] in numpy's order (pairwise, eight running sums; n up to 128)."""
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[i]
        return r
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    i = 8
    while i < n - (n % 8):
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@nb.njit(cache=True)
def _arrival(dist, radius, speed):
    """planner.arrival_step for one fleet."""
    return int(max(np.floor((dist - radius) / speed) + 1.0, 0.0))


@nb.njit(cache=True)
def _tick(owner, ships, growth, p1, p2):
    """planner.tick_rule for one planet."""
    pending = p1 > 0.0 or p2 > 0.0
    if owner == 0:
        if not pending:
            return owner, ships
        net = p1 - p2
        ns = ships - abs(net)
        if ns < 0.0:
            return (1 if net > 0.0 else 2), -ns
        return owner, ns
    own_in = p1 if owner == 1 else p2
    opp_in = p2 if owner == 1 else p1
    ps = ships + growth + ((own_in - opp_in) if pending else 0.0)
    if pending and ps < 0.0:
        return 3 - owner, -ps
    return owner, ps


@nb.njit(cache=True)
def _settle(owner, ships, g, cur, ev, p1, p2, h):
    """planner.settle for one planet."""
    for e in range(ev.size):
        t = ev[e]
        if t < h:
            if owner != 0:
                ships = ships + g * (t - cur)
            owner, ships = _tick(owner, ships, g, p1[e], p2[e])
            cur = t + 1.0
    if owner != 0:
        ships = ships + g * (h - cur)
    return owner, ships


@nb.njit(cache=True)
def _score(owner, ships, growth, me, near, buf):
    """planner.score for one position."""
    n = owner.size
    n_mine, n_opp = 0, 0
    for j in range(n):
        n_mine += owner[j] == me
        n_opp += owner[j] == 3 - me
    sums = np.empty(7)
    for q in range(7):
        for j in range(n):
            m = 1.0 if owner[j] == me else 0.0
            o = 1.0 if owner[j] == 3 - me else 0.0
            if q == 0:
                buf[j] = ships[j] * m
            elif q == 1:
                buf[j] = ships[j] * o
            elif q == 2:
                buf[j] = growth[j] * m
            elif q == 3:
                buf[j] = growth[j] * o
            elif q == 4:
                buf[j] = (ships[j] + 20.0 * growth[j]) * m
            elif q == 5:
                buf[j] = (ships[j] + 20.0 * growth[j]) * o
            else:
                iso = owner[j] == me and n_mine > 1
                if iso:
                    for k in range(n):
                        if near[j, k] and owner[k] == me:
                            iso = False
                            break
                buf[j] = growth[j] * (1.0 if iso else 0.0)
        sums[q] = _psum(buf, n)
    sc = sums[0] - sums[1]
    sc += 10.0 * (sums[2] - sums[3])
    sc += 0.5 * (sums[4] - sums[5])
    sc -= 2.0 * sums[6]
    sc += 5.0 * (n_mine - n_opp)
    win = 10000.0 if (n_opp == 0 and n_mine > 0) else 0.0
    loss = 10000.0 if (n_mine == 0 and n_opp > 0) else 0.0
    sc += win - loss
    return sc


@nb.njit(cache=True)
def _arrivals(n, h, pos, radius, speed, tr_on, tr_owner, tr_dest, tr_ships, tr_pos):
    """planner.arrivals."""
    a1, a2 = np.zeros((h, n)), np.zeros((h, n))
    for f in range(n):
        if tr_on[f]:
            d = tr_dest[f]
            vx, vy = tr_pos[f, 0] - pos[d, 0], tr_pos[f, 1] - pos[d, 1]
            k = _arrival(np.sqrt(_fma(vy, vy, vx * vx)), radius[d], speed)
            if k < h:
                if tr_owner[f] == 1:
                    a1[k, d] += tr_ships[f]
                else:
                    a2[k, d] += tr_ships[f]
    return a1, a2


@nb.njit(cache=True)
def _base_states(owner0, ships0, growth, a1, a2):
    """planner.base_states."""
    h, n = a1.shape
    po, ps = np.empty((h, n), np.int64), np.empty((h, n))
    owner, ships = owner0.copy(), ships0.copy()
    cur, k = 0, 0
    while True:
        while k < h:
            hit = False
            for j in range(n):
                if a1[k, j] > 0.0 or a2[k, j] > 0.0:
                    hit = True
                    break
            if hit:
                break
            k += 1
        end = min(k + 1, h)
        for t in range(cur, end):
            for j in range(n):
                po[t, j] = owner[j]
                ps[t, j] = ships[j] + (t - cur) * (0.0 if owner[j] == 0 else growth[j])
        if k >= h:
            break
        for j in range(n):
            owner[j], ships[j] = _tick(owner[j], ps[k, j], growth[j], a1[k, j], a2[k, j])
        cur = k + 1
        k += 1
    return po, ps


@nb.njit(cache=True)
def _run_last(po, ps, growth, a1, a2):
    """planner.run over the projection's last tick."""
    n = po.size
    owner, ships = po.copy(), ps.copy()
    hit = False
    for j in range(n):
        if a1[j] > 0.0 or a2[j] > 0.0:
            hit = True
    cur = 0
    if hit:
        for j in range(n):
            if owner[j] != 0:
                ships[j] = ships[j] + growth[j] * 0
            owner[j], ships[j] = _tick(owner[j], ships[j], growth[j], a1[j], a2[j])
        cur = 1
    for j in range(n):
        if owner[j] != 0:
            ships[j] = ships[j] + growth[j] * (1 - cur)
    return owner, ships


@nb.njit(cache=True)
def _dist_near(pos):
    """Distances between planets and the hand score's neighbours (within 160), as planner.candidates has them."""
    n = pos.shape[0]
    d = np.empty((n, n))
    near = np.zeros((n, n), np.bool_)
    for i in range(n):
        for j in range(n):
            dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
            d[i, j] = np.sqrt(dx * dx + dy * dy)
            near[i, j] = d[i, j] <= 160.0 and i != j
    return d, near


@nb.njit(cache=True)
def _base(owner, ships, growth, pos, radius, speed, tr_on, tr_owner, tr_dest, tr_ships, tr_pos, me, h):
    """Where the map stands after `h` ticks with nobody launching, and its hand score."""
    n = owner.size
    a1, a2 = _arrivals(n, h, pos, radius, speed, tr_on, tr_owner, tr_dest, tr_ships, tr_pos)
    _, near = _dist_near(pos)
    po, ps = _base_states(owner, ships, growth, a1, a2)
    bo, bs = _run_last(po[h - 1], ps[h - 1], growth, a1[h - 1], a2[h - 1])
    return bo, bs, _score(bo, bs, growth, me, near, np.empty(n))


@nb.njit(cache=True)
def _cands(owner, ships, growth, pos, radius, speed, tr_on, tr_owner, tr_dest, tr_ships, tr_pos, me, h, is_src,
           kinds, v0, v1):
    n = owner.size
    a1, a2 = _arrivals(n, h, pos, radius, speed, tr_on, tr_owner, tr_dest, tr_ships, tr_pos)
    d, near = _dist_near(pos)
    po, ps = _base_states(owner, ships, growth, a1, a2)
    base_owner, base_ships = _run_last(po[h - 1], ps[h - 1], growth, a1[h - 1], a2[h - 1])
    buf = np.empty(n)
    base = _score(base_owner, base_ships, growth, me, near, buf)

    a_me, a_op = (a1, a2) if me == 1 else (a2, a1)
    need, lands, spare = np.zeros((n, n)), np.zeros((n, n), np.bool_), np.zeros(n)
    for i in range(n):
        kept, low = True, np.inf
        for t in range(h):
            kept = kept and po[t, i] == me
            low = min(low, ps[t, i])
        spare[i] = np.floor(low) if kept else 0.0
        for j in range(n):
            k_all = _arrival(d[i, j], radius[j], speed)
            kk = min(k_all, h - 1)
            bo = po[kk, j]
            need[i, j] = ps[kk, j] + (growth[j] if bo != 0 else 0.0) + a_op[kk, j] - a_me[kk, j]
            lands[i, j] = k_all < h and bo != me

    # every launch, in numpy's order: by size option, then source, then target
    n_opt = kinds.size
    n_frac = 0
    while n_frac < n_opt and kinds[n_frac] == 0:
        n_frac += 1
    xs = np.zeros((n_opt, n, n))
    m = 0
    for i in range(n):
        m += is_src[i]
    cap = n_opt * m * max(n - 1, 1)
    ci, cj, cn = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap)
    c = 0
    for f in range(n_opt):
        for i in range(n):
            if not is_src[i]:
                continue
            for j in range(n):
                if j == i:
                    continue
                kind = kinds[f]
                if kind == 0 or kind == 3:
                    x = max(1.0, np.floor(ships[i] * v0[f]))
                elif kind == 1:
                    x = (np.floor(max(need[i, j], 0.0) * v0[f]) + 1.0 + v1[f]) if lands[i, j] else 0.0
                else:
                    x = np.floor(spare[i] * v0[f])
                ok = x >= 1.0 and x <= ships[i]
                if f >= n_frac:
                    for e in range(f):
                        ok = ok and x != xs[e, i, j]
                xs[f, i, j] = x
                if ok:
                    ci[c], cj[c], cn[c] = i, j, x
                    c += 1
    ci, cj, cn = ci[:c], cj[:c], cn[:c]

    # where each leaves the map: only its source and target differ from the projection (see planner.project)
    cnt = np.zeros(n, np.int64)
    for j in range(n):
        for k in range(h):
            if a1[k, j] + a2[k, j] > 0.0:
                cnt[j] += 1
    e_max = max(cnt.max(), 1)
    ticks = np.full((n, e_max), h, np.int64)
    for j in range(n):
        q = 0
        for k in range(h):
            if a1[k, j] + a2[k, j] > 0.0:
                ticks[j, q] = k
                q += 1
    all_owner, all_ships, score = np.empty((c, n), np.int64), np.empty((c, n)), np.empty(c)
    ev, p1, p2 = np.empty(e_max + 1, np.int64), np.empty(e_max + 1), np.empty(e_max + 1)
    q1, q2 = np.empty(e_max), np.empty(e_max)
    for r in range(c):
        i, j, x = ci[r], cj[r], cn[r]
        k_arr = _arrival(d[i, j], radius[j], speed)
        kc = min(k_arr, h - 1)
        ev[0] = kc
        q = 1
        for e in range(e_max):
            if ticks[j, e] > kc:
                ev[q] = ticks[j, e]
                q += 1
        while q <= e_max:
            ev[q] = h
            q += 1
        for e in range(e_max + 1):
            live = ev[e] < h
            p1[e] = a1[ev[e], j] if live else 0.0
            p2[e] = a2[ev[e], j] if live else 0.0
        if me == 1:
            p1[0] += x
        else:
            p2[0] += x
        t_own, t_sh = _settle(po[kc, j], ps[kc, j], growth[j], float(kc), ev, p1, p2, h)
        for e in range(e_max):
            live = ticks[i, e] < h
            q1[e] = a1[ticks[i, e], i] if live else 0.0
            q2[e] = a2[ticks[i, e], i] if live else 0.0
        s_own, s_sh = _settle(owner[i], ships[i] - x, growth[i], 0.0, ticks[i], q1, q2, h)
        for k in range(n):
            all_owner[r, k], all_ships[r, k] = base_owner[k], base_ships[k]
        if k_arr < h:
            all_owner[r, j], all_ships[r, j] = t_own, t_sh
        all_owner[r, i], all_ships[r, i] = s_own, s_sh
        score[r] = _score(all_owner[r], all_ships[r], growth, me, near, buf)
    return (ci, cj, cn, all_owner, all_ships, score, base_owner, base_ships, base, a1, a2, d, near, po, ps)


def _state(s):
    f64, i64 = np.float64, np.int64
    return (np.ascontiguousarray(s.owner, i64), np.ascontiguousarray(s.ships, f64), np.ascontiguousarray(s.growth, f64),
            np.ascontiguousarray(s.pos, f64), np.ascontiguousarray(s.radius, f64), float(s.speed),
            np.ascontiguousarray(s.tr_on, np.bool_), np.ascontiguousarray(s.tr_owner, i64),
            np.ascontiguousarray(s.tr_dest, i64), np.ascontiguousarray(s.tr_ships, f64), np.ascontiguousarray(s.tr_pos, f64))


def base(s, me, h):
    """(owners, ships, hand score) `h` ticks on if nobody launches: candidates' base_owner, base_ships and base."""
    o, sh, sc = _base(*_state(s), int(me), int(h))
    return o, sh, np.float64(sc)


def candidates(s, me, horizon, fractions, extras, keep, reserved, parse_size):
    """planner.candidates (without `extra`), compiled; the same dict."""
    n = s.ships.shape[0]
    horizon = max(1, min(horizon, int(s.max_ticks - s.tick)))
    is_src = (s.owner == me) & ~s.tr_on & (s.ships >= 1.0)
    for r in reserved:
        is_src[r] = False
    if not is_src.any():
        return None
    kinds, v0, v1 = [0] * len(fractions), [float(f) for f in fractions], [0.0] * len(fractions)
    for kind, vals in (parse_size(e) for e in extras):
        kinds.append(KIND[kind])
        v0.append(float(vals[0]))
        v1.append(float(vals[1]) if kind == "cap" else 0.0)
    r = _cands(*_state(s), int(me), int(horizon), is_src, np.array(kinds, np.int64), np.array(v0, np.float64),
               np.array(v1, np.float64))
    ci, cj, cn, owner, ships, score, base_owner, base_ships, base, a1, a2, d, near, po, ps = r
    out = dict(ci=ci, cj=cj, cn=cn, owner=owner, ships=ships, score=score, base_owner=base_owner,
               base_ships=base_ships, base=np.float64(base), horizon=horizon)
    if keep:
        out.update(a1=a1, a2=a2, d=d, near=near, po=po, ps=ps)
    return out


def warm():
    """Compile (or load from the cache) before the first move."""
    from types import SimpleNamespace
    import planner
    n = 4
    s = SimpleNamespace(valid=np.ones(n, bool), owner=np.array([1, 2, 0, 0]), ships=np.array([20.0, 20.0, 5.0, 5.0]),
                        pos=np.array([[100.0, 100.0], [500.0, 300.0], [300.0, 200.0], [200.0, 400.0]]),
                        growth=np.full(n, 0.1), radius=np.full(n, 10.0), tr_on=np.array([False, True, False, False]),
                        tr_owner=np.array([0, 2, 0, 0]), tr_dest=np.array([0, 2, 0, 0]), tr_ships=np.array([0.0, 4.0, 0, 0]),
                        tr_pos=np.array([[0.0, 0.0], [450.0, 280.0], [0, 0], [0, 0]]), tr_vel=np.zeros((n, 2)),
                        tick=10.0, speed=3.5, max_ticks=2000.0)
    candidates(s, 1, 90, (0.25, 0.5, 1.0), ("spare", "cap"), True, (), planner.parse_size)
    base(s, 1, 90)
