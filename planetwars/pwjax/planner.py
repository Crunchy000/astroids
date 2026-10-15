"""The all-moves planner (entry/planner.py) in JAX, so thousands of planner games can be played on a GPU.

Same algorithm, same numbers: every launch is projected `horizon` ticks ahead with nobody else acting, and
the position then is scored. Because no new launches happen in a projection, a planet's future depends only on
the fleets already heading for it, and one launch changes just two planets (its source and its target). The
target's and the source's timelines are run for every candidate at once; everything else is the shared
"nobody does anything" projection of the whole map.

Fixed shapes: planets are padded to N, and the `sources` planets with the most free ships are the ones
tried as launch points (the numpy planner tries all of them; the two agree when `sources` is at least the
number of free planets).
"""
import jax
import jax.numpy as jnp

from . import env

FRACTIONS = (0.25, 0.5, 1.0)


def planet_step(owner, ships, growth, p1, p2):
    """One tick of the rules for planets, given the ships each player lands on them this tick."""
    pending = (p1 > 0) | (p2 > 0)
    net = p1 - p2
    n_ships = ships - jnp.abs(net)
    n_flip = pending & (n_ships < 0)
    n_owner = jnp.where(n_flip, jnp.where(net > 0, 1, 2), owner)
    n_out = jnp.where(pending, jnp.where(n_flip, -n_ships, n_ships), ships)
    own_in = jnp.where(owner == 1, p1, p2)
    opp_in = jnp.where(owner == 1, p2, p1)
    p_ships = ships + growth + jnp.where(pending, own_in - opp_in, 0.0)
    p_flip = pending & (p_ships < 0)
    p_owner = jnp.where(p_flip, 3 - owner, owner)
    p_out = jnp.where(p_flip, -p_ships, p_ships)
    neutral = owner == 0
    return jnp.where(neutral, n_owner, p_owner).astype(owner.dtype), jnp.where(neutral, n_out, p_out)


def arrivals(s: env.State, horizon):
    """[2, horizon, N]: ships each player lands on each planet at each tick from fleets already in flight."""
    d = s.tr_dest
    dist = jnp.sqrt(jnp.sum((s.tr_pos - s.pos[d]) ** 2, axis=-1))
    k = jnp.maximum(jnp.floor((dist - s.radius[d]) / s.speed) + 1.0, 0.0).astype(jnp.int32)
    ok = s.tr_on & (k < horizon)
    a = jnp.zeros((2, horizon, s.ships.shape[0]), s.ships.dtype)
    return a.at[jnp.clip(s.tr_owner - 1, 0, 1), jnp.clip(k, 0, horizon - 1), d].add(jnp.where(ok, s.tr_ships, 0.0))


def project(s: env.State, horizon, h_eff):
    """Owners and ships of every planet after h_eff <= horizon ticks with nobody launching anything."""
    a = jnp.moveaxis(arrivals(s, horizon), 1, 0)

    def step(c, x):
        owner, ships = c
        k, ak = x
        o2, s2 = planet_step(owner, ships, s.growth, ak[0], ak[1])
        on = k < h_eff
        return (jnp.where(on, o2, owner), jnp.where(on, s2, ships)), None

    (owner, ships), _ = jax.lax.scan(step, (s.owner, s.ships), (jnp.arange(horizon), a))
    return owner, ships


def near_matrix(s: env.State):
    """[N, N]: planets within 160 of each other (the score's idea of "has a friendly planet nearby")."""
    d = jnp.sqrt(jnp.sum((s.pos[:, None, :] - s.pos[None, :, :]) ** 2, axis=-1))
    return ((d <= 160.0) & ~jnp.eye(s.ships.shape[0], dtype=bool)).astype(s.ships.dtype)


def score_final(owner, ships, s: env.State, me, near):
    """TeamTitans-style position score for `me`: owner and ships are [..., N]."""
    mine, opp = owner == me, owner == 3 - me
    g = s.growth
    sc = jnp.sum(jnp.where(mine, ships, 0.0), -1) - jnp.sum(jnp.where(opp, ships, 0.0), -1)
    sc += 10.0 * (jnp.sum(jnp.where(mine, g, 0.0), -1) - jnp.sum(jnp.where(opp, g, 0.0), -1))
    fut = ships + 20.0 * g
    sc += 0.5 * (jnp.sum(jnp.where(mine, fut, 0.0), -1) - jnp.sum(jnp.where(opp, fut, 0.0), -1))
    n_mine, n_opp = jnp.sum(mine, -1), jnp.sum(opp, -1)
    friends = jnp.einsum("...n,mn->...m", mine.astype(g.dtype), near)
    isolated = mine & (friends == 0) & (n_mine[..., None] > 1)
    sc -= 2.0 * jnp.sum(jnp.where(isolated, g, 0.0), -1)
    sc += 5.0 * (n_mine - n_opp)
    return sc + jnp.where((n_opp == 0) & (n_mine > 0), 10000.0, 0.0) - jnp.where((n_mine == 0) & (n_opp > 0), 10000.0, 0.0)


ALIASES = {"capm": "cap:1.2:1", "spare2": "spare:0.5"}
KINDS = ("f", "cap", "spare")


def parse_size(spec):
    """'f:0.35' (a fraction of the source's ships), 'cap:M:C' (what takes the target when the fleet lands, times M,
    plus 1 and C), 'spare:X' (a fraction of what the source can do without); 'cap', 'spare', 'capm' and 'spare2' are
    short for 'cap:1:0', 'spare:1', 'cap:1.2:1' and 'spare:0.5'."""
    spec = ALIASES.get(spec, spec)
    kind, *args = spec.split(":")
    assert kind in KINDS, spec
    vals = [float(x) for x in args]
    if kind == "cap":
        vals = (vals + [1.0, 0.0])[:2] if len(vals) < 2 else vals
    elif kind == "spare":
        vals = vals or [1.0]
    return kind, tuple(vals)


def base_states(s: env.State, a, h_eff, horizon):
    """[H, N] owners and ships of every planet just before each tick of the nobody-launches projection."""
    def step(c, x):
        owner, ships = c
        k, ak = x
        o2, s2 = planet_step(owner, ships, s.growth, ak[0], ak[1])
        on = k < h_eff
        return (jnp.where(on, o2, owner), jnp.where(on, s2, ships)), (owner, ships)

    _, (po, ps) = jax.lax.scan(step, (s.owner, s.ships), (jnp.arange(horizon), a))
    return po, ps


def make_planner(sources=12, horizon=90, top=8, fractions=FRACTIONS, extras=()):
    """Returns plan(state, me, h_eff) -> dict with the chosen move and the `top` best candidates.

    h_eff <= horizon is how far to project this time (the scan runs `horizon` steps, later ones are skipped).
    Besides the fractions of the source's ships, `extras` adds launch sizes (see `parse_size`), most of them worked out
    from the projection:
      cap     just enough ships to take the target when they land (what is there then, plus one)
      cap:M:C the same with a margin: M times what is there, plus one and C
      spare   all the ships the source can do without: its smallest garrison over the projection, or none if the
              projection loses the planet
      spare:X a fraction X of that
      f:X     a fraction X of the source's ships
    Duplicates of an earlier size for the same source and target are dropped.
    Keys: action (src, dst, ships), best, base (scores of the best move and of waiting), cand_src/dst/ships/score [top],
    cand_owner/cand_ships [top, N] (projected positions), base_owner/base_ships [N].
    """
    extras = tuple(parse_size(e) for e in extras)
    nf = len(fractions) + len(extras)

    def plan(s: env.State, me, h_eff):
        n = s.ships.shape[0]
        dt = s.ships.dtype
        fr = jnp.asarray(fractions, dt)
        d = jnp.sqrt(jnp.sum((s.pos[:, None, :] - s.pos[None, :, :]) ** 2, axis=-1))
        near = near_matrix(s)
        free = s.valid & (s.owner == me) & ~s.tr_on & (s.ships >= 1.0)
        _, src = jax.lax.top_k(jnp.where(free, s.ships, -1.0), sources)           # [K]
        src_ok = free[src]
        a = jnp.moveaxis(arrivals(s, horizon), 1, 0)                               # [H, 2, N]
        k_c = jnp.maximum(jnp.floor((d[src] - s.radius[None, :]) / s.speed) + 1.0, 0.0).astype(jnp.int32)  # [K, N]
        ships_src = s.ships[src]
        amt = jnp.maximum(1.0, jnp.floor(ships_src[:, None] * fr[None, :]))        # [K, F]
        amts = [jnp.broadcast_to(amt[:, f, None], (sources, n)) for f in range(len(fractions))]
        if any(k != "f" for k, _ in extras):
            po, ps = base_states(s, a, h_eff, horizon)
            kk = jnp.clip(k_c, 0, horizon - 1)
            cols = jnp.arange(n)[None, :]
            mine_in, opp_in = a[:, me - 1, :][kk, cols], a[:, 2 - me, :][kk, cols]
            before_o, before_s = po[kk, cols], ps[kk, cols]
            need = before_s + jnp.where(before_o != 0, s.growth[None, :], 0.0) + opp_in - mine_in
            lands = (k_c < jnp.minimum(h_eff, horizon)) & (before_o != me)
            live = jnp.arange(horizon)[:, None] < h_eff
            kept = jnp.all(jnp.where(live, po[:, src] == me, True), 0)             # [K] the projection never loses the source
            low = jnp.min(jnp.where(live, ps[:, src], jnp.inf), 0)
            spare = jnp.where(kept, jnp.floor(low), 0.0)
        for kind, v in extras:
            if kind == "cap":
                x = jnp.where(lands, jnp.floor(jnp.maximum(need, 0.0) * v[0]) + 1.0 + v[1], 0.0)
            elif kind == "spare":
                x = jnp.broadcast_to(jnp.floor(spare * v[0])[:, None], (sources, n))
            else:
                x = jnp.broadcast_to(jnp.maximum(1.0, jnp.floor(ships_src * v[0]))[:, None], (sources, n))
            amts.append(x)
        cn = jnp.stack(amts, 1)                                                    # [K, F, N] launch size per source, option, target
        amt_ok = src_ok[:, None, None] & (cn >= 1.0) & (cn <= ships_src[:, None, None])
        for f in range(len(fractions), nf):                                        # drop extra sizes already tried for this pair
            amt_ok = amt_ok.at[:, f].set(amt_ok[:, f] & ~jnp.any(cn[:, :f] == cn[:, f:f + 1], 1))
        ci = jnp.broadcast_to(src[:, None, None], (sources, nf, n))
        cj = jnp.broadcast_to(jnp.arange(n)[None, None, :], (sources, nf, n))
        ok = amt_ok & s.valid[None, None, :] & (ci != cj)
        k_f = jnp.broadcast_to(k_c[:, None, :], (sources, nf, n))

        def step(c, x):
            (bo, bs), (do, ds), (so, ss) = c
            k, ak = x
            on = k < h_eff
            nbo, nbs = planet_step(bo, bs, s.growth, ak[0], ak[1])
            mine1, mine2 = (me == 1), (me == 2)
            land = (k_f == k) & ok
            p1 = ak[0][cj] + jnp.where(land & mine1, cn, 0.0)
            p2 = ak[1][cj] + jnp.where(land & mine2, cn, 0.0)
            ndo, nds = planet_step(do, ds, s.growth[cj], p1, p2)
            nso, nss = planet_step(so, ss, s.growth[src][:, None, None], ak[0][src][:, None, None], ak[1][src][:, None, None])
            sel = lambda new, old: jnp.where(on, new, old)  # noqa: E731
            return ((sel(nbo, bo), sel(nbs, bs)), (sel(ndo, do), sel(nds, ds)), (sel(nso, so), sel(nss, ss))), None

        init = ((s.owner, s.ships), (s.owner[cj], s.ships[cj]),
                (jnp.broadcast_to(s.owner[src][:, None, None], (sources, nf, n)), ships_src[:, None, None] - cn))
        ((bo, bs), (do, ds), (so, ss)), _ = jax.lax.scan(step, init, (jnp.arange(horizon), a))

        idx = jnp.arange(n)
        # final position for each candidate: the shared one, with its source and target planets replaced
        is_j = idx[None, None, None, :] == cj[..., None]                          # [K, F, target, planet]
        is_i = idx[None, None, None, :] == ci[..., None]
        f_owner = jnp.where(is_j, do[..., None], jnp.where(is_i, so[..., None], bo))
        f_ships = jnp.where(is_j, ds[..., None], jnp.where(is_i, ss[..., None], bs))
        sc = score_final(f_owner, f_ships, s, me, near)                            # [K, F, N]
        sc = jnp.where(ok, sc, -jnp.inf)
        base = score_final(bo, bs, s, me, near)
        flat = sc.reshape(-1)
        vals, pick = jax.lax.top_k(flat, top)
        flat_i, flat_j, flat_n = ci.reshape(-1), cj.reshape(-1), cn.reshape(-1)
        best = vals[0]
        go = best > base + 2e-3   # a launch that changes nothing is no reason to launch (float32 noise is ~1e-3)
        action = (jnp.where(go, flat_i[pick[0]], -1).astype(jnp.int32), jnp.where(go, flat_j[pick[0]], -1).astype(jnp.int32),
                  jnp.where(go, flat_n[pick[0]], 0.0).astype(dt))
        fo, fs = f_owner.reshape(-1, n), f_ships.reshape(-1, n)
        return dict(action=action, best=best, base=base, cand_src=flat_i[pick], cand_dst=flat_j[pick], cand_ships=flat_n[pick],
                    cand_score=vals, cand_owner=fo[pick], cand_ships_final=fs[pick], base_owner=bo, base_ships=bs)

    return plan
