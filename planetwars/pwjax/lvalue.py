"""A learned value for the all-moves planner.

The planner ranks launches by a hand-written score of the position that results when nobody does anything for
`horizon` ticks. This module asks whether a network trained on game results ranks them better.

    python -m pwjax.lvalue gen   --out data  --envs 2048 --batches 4          # planner-vs-planner games, recorded
    python -m pwjax.lvalue train --data data --init value_v1.npz --out vnet    # value of the projected position
    python -m pwjax.lvalue match vnet/latest.npz --games 512                   # re-rank the planner's top moves

What is recorded and learned: every `stride` ticks, for each side, the position the planner projects for the
move it plays (planets only, `horizon` ticks on). The label is the game's result from the mover's view.
Games are varied by giving each side its own horizon and some random moves, and some sides play the greedy bot.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import baselines, env, evaluate, maps, policy
from . import planner as jplanner
from .ppo import save_params, tree_where
from .vsearch import play_chunked

HORIZONS = np.array([60, 80, 90, 100])
EPS = np.array([0.0, 0.0, 0.03, 0.1])
PHASES = (0.0, 0.1, 0.25, 0.5, 1.01)


# ---------------------------------------------------------------- agents

def fresh(s, owner, ships, tick):
    """A planets-only position: the map of `s`, with the given owners and ships and no fleets in flight."""
    return s._replace(owner=owner.astype(jnp.int32), ships=ships, tr_on=jnp.zeros_like(s.tr_on), tick=tick)


def options(plan, s, me, h_eff, top, params=None, heads=None):
    """What the planner can do this tick: its `top` launches (by hand score) and waiting, as arrays [top + 1, ...].
    Returns (planner output, owners, ships, hand scores, usable, value-network values or None)."""
    out = plan(s, me, h_eff)
    owners = jnp.concatenate([out["cand_owner"], out["base_owner"][None]])
    ships = jnp.concatenate([out["cand_ships_final"], out["base_ships"][None]])
    hand = jnp.concatenate([out["cand_score"], out["base"][None]])
    ok = jnp.isfinite(hand)
    v = net_values(params, heads, s, me, owners, ships, h_eff) if params is not None else None
    return out, owners, ships, hand, ok, v


def net_values(params, heads, s, me, owners, ships, h_eff):
    """The value network's number for each projected position in `owners`, `ships` ([options, N])."""
    return jax.vmap(lambda o, sh: policy.apply(params, heads, fresh(s, o, sh, s.tick + h_eff), me)[2])(owners, ships)


def totals(s, me, owners, hand, v, mix, scale=200.0):
    """The value network's view of each option, mixed with the hand score: (1 - mix) * v + mix * tanh(hand / scale),
    and a clear win or loss counts for 10 either way."""
    mine, opp = s.valid & (owners == me), s.valid & (owners == 3 - me)
    won, lost = jnp.sum(opp, -1) == 0, jnp.sum(mine, -1) == 0
    total = (1 - mix) * v + mix * jnp.tanh(jnp.where(jnp.isfinite(hand), hand, 0.0) / scale)
    return total + jnp.where(won & ~lost, 10.0, 0.0) - jnp.where(lost & ~won, 10.0, 0.0)


def make_decider(plan, top, params=None, heads=None, has2=False, heads2=None, plan_r=None, ours=0):
    """decide(s, me, h, eps, kind, key, params2) -> (action, (afterstate owners, ships, ticks looked ahead, hand score)).

    kind 0 plays the planner's hand score, 1 the value network alone, 2 the network mixed half and half with the
    hand score, 3 a second (older) network, passed in as `params2`, mixed the same way. With `plan_r` (the opponent's
    planner) kind 2 also looks one reply deep over its `ours` best options (see look_ahead). With probability eps one
    of the usable options is picked at random instead."""
    def decide(s, me, h, eps, kind, key, params2=None):
        h_eff = jnp.maximum(jnp.minimum(h, s.max_ticks - s.tick), 1)
        out, owners, ships, hand, ok, v = options(plan, s, me, h_eff, top, params, heads)
        by_hand = jnp.where(ok, hand - jnp.where(jnp.arange(top + 1) == top, 2e-3, 0.0), -jnp.inf)  # waiting wins ties
        if params is None:
            score = by_hand
        else:
            by_v = jnp.where(ok, totals(s, me, owners, hand, v, 0.0), -jnp.inf)
            by_mix = jnp.where(ok, totals(s, me, owners, hand, v, 0.5), -jnp.inf)
            if plan_r is not None:
                cand, val = look_ahead(plan_r, params, heads, s, me, h_eff, out, by_mix, ours, 0.5)
                by_mix = jnp.full(top + 1, -jnp.inf).at[cand].set(val)
            score = jnp.where(kind == 0, by_hand, jnp.where(kind == 1, by_v, by_mix))
            if has2:
                v2 = net_values(params2, heads2, s, me, owners, ships, h_eff)
                score = jnp.where(kind == 3, jnp.where(ok, totals(s, me, owners, hand, v2, 0.5), -jnp.inf), score)
        k1, k2 = jax.random.split(key)
        rand = (jax.random.uniform(k1) < eps) & ok.any()
        pick = jnp.where(rand, jax.random.categorical(k2, jnp.where(ok, 0.0, -jnp.inf)), jnp.argmax(score))
        launch = (pick < top) & ok[pick]
        j = jnp.minimum(pick, top - 1)
        action = (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32),
                  jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
                  jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
        hand_score = jplanner.score_final(owners[pick], ships[pick], s, me, jplanner.near_matrix(s))
        return action, (owners[pick], ships[pick], h_eff, hand_score)
    return decide


def make_hand_agent(sources=12, horizon=90, top=8, fractions=jplanner.FRACTIONS, extras=()):
    """The planner as an agent function (state, me, key) -> action."""
    decide = make_decider(jplanner.make_planner(sources, horizon, top, fractions, extras), top)
    return lambda s, me, key: decide(s, me, horizon, 0.0, 0, key)[0]


def make_value_agent(params, heads, sources=12, horizon=90, top=8, mix=0.0, scale=200.0, wait=True,
                     fractions=jplanner.FRACTIONS, extras=()):
    """The planner's `top` launches (by hand score) and waiting, re-ranked by the value network
    (mixed with the hand score as `(1 - mix) * value + mix * tanh(hand / scale)`)."""
    plan = jplanner.make_planner(sources, horizon, top, fractions, extras)

    def act(s, me, key):
        del key
        h = jnp.maximum(jnp.minimum(horizon, s.max_ticks - s.tick), 1)
        out, owners, ships, hand, ok, v = options(plan, s, me, h, top, params, heads)
        ok = ok & jnp.concatenate([jnp.ones(top, bool), jnp.array([wait])])
        k = jnp.argmax(jnp.where(ok, totals(s, me, owners, hand, v, mix, scale), -jnp.inf))
        launch = (k < top) & ok[k]
        j = jnp.minimum(k, top - 1)
        return (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
    return act


def look_ahead(plan_r, params, heads, s, me, h, out, plain, ours, mix, scale=200.0, weight=1.0, owners=None, v=None):
    """One reply deep: the `ours` best options by `plain` ([top + 1], waiting last, -inf where unusable) are each played
    on a copy of the game, `plan_r` (the opponent's planner) finds its best answers by its own hand score, and the
    network scores the position after each. Returns those options' indices [ours], best plain score first, and what
    each is worth: weight * (the worst of its own score and those after the answers) + (1 - weight) * its own score.

    Given the options' projected `owners` and network values `v` ([top + 1]), also returns, for each, the network's
    value of the position where that worst case is reached (1 or -1 where that position is already won or lost)."""
    top = plain.shape[0] - 1
    first, cand = jax.lax.top_k(plain, ours)                                       # [ours]
    usable = jnp.isfinite(first)
    launch = (cand < top) & usable
    j = jnp.minimum(cand, top - 1)
    acts = (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32),
            jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
            jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
    after = jax.vmap(lambda a: env._launch(s, me, a))(acts)
    r = jax.vmap(lambda st: plan_r(st, 3 - me, h))(after)
    ro, rs = r["cand_owner"], r["cand_ships_final"]                                # [ours, replies, N]
    r_ok = jnp.isfinite(r["cand_score"])
    r_hand = jplanner.score_final(ro, rs, s, me, jplanner.near_matrix(s))
    r_v = jax.vmap(lambda o, sh: net_values(params, heads, s, me, o, sh, h))(ro, rs)
    r_tot = totals(s, me, ro, r_hand, r_v, mix, scale)
    worst = jnp.minimum(first, jnp.min(jnp.where(r_ok, r_tot, jnp.inf), 1))
    val = jnp.where(usable, weight * worst + (1.0 - weight) * first, -jnp.inf)
    if v is None:
        return cand, val
    # where the worst case is reached: the option itself (column 0) or one of the answers
    tots = jnp.concatenate([first[:, None], jnp.where(r_ok, r_tot, jnp.inf)], 1)
    leaf_o = jnp.concatenate([owners[cand][:, None], ro], 1)                     # [ours, 1 + replies, N]
    leaf_v = jnp.concatenate([v[cand][:, None], r_v], 1)
    mine, opp = s.valid & (leaf_o == me), s.valid & (leaf_o == 3 - me)
    won, lost = jnp.sum(opp, -1) == 0, jnp.sum(mine, -1) == 0
    leaf_v = jnp.where(won & ~lost, 1.0, jnp.where(lost & ~won, -1.0, leaf_v))
    return cand, val, jnp.take_along_axis(leaf_v, jnp.argmin(tots, 1)[:, None], 1)[:, 0]


def make_reply_agent(params, heads, sources=12, horizon=90, top=8, mix=0.0, scale=200.0, wait=True,
                     fractions=jplanner.FRACTIONS, extras=(), ours=8, replies=3, weight=1.0):
    """make_value_agent, looking one reply deep. The planner's projection assumes nobody launches anything more, so it
    cannot see a counter-attack on a planet a launch has just emptied. Here the `ours` best options by the mixed score
    (waiting included) are each played on a copy of the game, the opponent's planner finds its `replies` best answers
    to each by its own hand score, and the network scores the position after each answer. An option is then worth
    weight * (its worst case, the opponent not answering included) + (1 - weight) * (its plain score) (see look_ahead)."""
    plan = jplanner.make_planner(sources, horizon, top, fractions, extras)
    plan_r = jplanner.make_planner(sources, horizon, replies, fractions, extras)

    def act(s, me, key):
        del key
        h = jnp.maximum(jnp.minimum(horizon, s.max_ticks - s.tick), 1)
        out, owners, ships, hand, ok, v = options(plan, s, me, h, top, params, heads)
        ok = ok & jnp.concatenate([jnp.ones(top, bool), jnp.array([wait])])
        plain = jnp.where(ok, totals(s, me, owners, hand, v, mix, scale), -jnp.inf)
        cand, val = look_ahead(plan_r, params, heads, s, me, h, out, plain, ours, mix, scale, weight)
        k = cand[jnp.argmax(val)]
        launch = (k < top) & ok[k]
        j = jnp.minimum(k, top - 1)
        return (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
    return act


# ---------------------------------------------------------------- game generation

def make_gen(cfg, n_envs, params=None, heads=None, has2=False, heads2=None):
    """gen(state carry, per-side settings, key, params2) plays `cfg.chunk` ticks, recording every `stride`th."""
    plan = jplanner.make_planner(cfg.sources, cfg.horizon, cfg.top, parse_fractions(cfg.fractions), parse_extras(cfg.extras))
    plan_r, ours = None, 0
    if cfg.reply:  # OURS:REPLIES: kind 2 looks one reply deep
        ours, replies = (int(x) for x in cfg.reply.split(":"))
        plan_r = jplanner.make_planner(cfg.sources, cfg.horizon, replies, parse_fractions(cfg.fractions), parse_extras(cfg.extras))
    decide = make_decider(plan, cfg.top, params, heads, has2, heads2, plan_r, ours)
    v_decide = jax.vmap(decide, in_axes=(0, 0, 0, 0, 0, 0, None))
    v_step, v_term, v_greedy = jax.vmap(env.step), jax.vmap(env.is_terminal), jax.vmap(baselines.greedy)
    one, two = jnp.ones(n_envs, jnp.int32), jnp.full(n_envs, 2, jnp.int32)

    def now_scores(s):
        """The hand score of the position as it stands (planets only), for each player: what the dense targets are made of."""
        near = jplanner.near_matrix(s)
        return jnp.stack([jplanner.score_final(s.owner, s.ships, s, 1, near), jplanner.score_final(s.owner, s.ships, s, 2, near)])

    def tick(c, sides, k, p2):
        s, done = c
        h1, h2, e1, e2, g1, g2, q1, q2 = sides
        ka, kb = jax.random.split(k)
        a1, r1 = v_decide(s, one, h1, e1, q1, jax.random.split(ka, n_envs), p2)
        a2, r2 = v_decide(s, two, h2, e2, q2, jax.random.split(kb, n_envs), p2)
        gr1, gr2 = v_greedy(s, one), v_greedy(s, two)
        a1 = tuple(jnp.where(g1, g, p) for g, p in zip(gr1, a1))
        a2 = tuple(jnp.where(g2, g, p) for g, p in zip(gr2, a2))
        s2 = tree_where(done, s, v_step(s, a1, a2))
        return (s2, done | v_term(s2)), (r1, r2)

    def chunk(c, sides, key, p2):
        def one_tick(c, k):
            return tick(c, sides, k, p2)[0], None

        def snap_block(c, k):
            ks = jax.random.split(k, cfg.stride)
            c1, _ = jax.lax.scan(one_tick, c, ks[:-1])
            alive = ~c1[1]
            c2, (r1, r2) = tick(c1, sides, ks[-1], p2)
            rec = dict(alive=alive, tick=c1[0].tick,
                       owner=jnp.stack([r1[0], r2[0]], 1).astype(jnp.int8), ships=jnp.stack([r1[1], r2[1]], 1),
                       h=jnp.stack([r1[2], r2[2]], 1), hand=jnp.stack([r1[3], r2[3]], 1),
                       now=jax.vmap(now_scores)(c1[0]).astype(jnp.float32))
            return c2, rec

        return jax.lax.scan(snap_block, c, jax.random.split(key, cfg.chunk // cfg.stride))

    return jax.jit(chunk)


DENSE_OFFSETS = (3, 6, 12)  # snapshots ahead (stride ticks each): the score of the real position then, as extra targets


def play_batch(cfg, bank, map_ids, rng, key, gen, p2=None):
    """Play one batch of games from the given maps; returns the recorded positions and results."""
    n = len(map_ids)
    pick = lambda a: jnp.asarray(rng.choice(a, size=n).astype(a.dtype))  # noqa: E731
    greedy = lambda: jnp.asarray(rng.random(n) < cfg.p_greedy)  # noqa: E731
    kind_p = np.array([float(x) for x in cfg.kinds.split(",")])
    kinds = lambda: jnp.asarray(rng.choice(len(kind_p), size=n, p=kind_p / kind_p.sum()).astype(np.int32))  # noqa: E731
    sides = (pick(HORIZONS), pick(HORIZONS), pick(EPS), pick(EPS), greedy(), greedy(), kinds(), kinds())
    mi = jnp.asarray(map_ids)
    s = jax.vmap(lambda i: env.from_bank(bank, i))(mi)
    c = (s, jax.vmap(env.is_terminal)(s))
    recs = []
    for _ in range(-(-cfg.ticks // cfg.chunk)):
        if bool(c[1].all()):
            break
        key, k = jax.random.split(key)
        c, rec = gen(c, sides, k, p2)
        recs.append(rec)
    lead = jax.vmap(env.leader)(c[0])
    z1 = jnp.where(lead == 1, 1.0, jnp.where(lead == 2, -1.0, 0.0))
    flat = {k: jnp.concatenate([r[k] for r in recs]) for k in recs[0]}        # [snaps, E, ...]
    alive = np.asarray(flat["alive"]).reshape(-1)
    ids = np.nonzero(alive)[0]
    out = {}
    for k in ("tick", "owner", "ships", "h", "hand"):
        v = np.asarray(flat[k])
        out[k] = v.reshape((-1,) + v.shape[2:])[ids] if k != "tick" else v.reshape(-1)[ids]
    snaps, envs = flat["alive"].shape
    # dense targets: how the real position scores 24, 48 and 96 ticks later (the game's end if sooner), squashed
    alive_m = np.asarray(flat["alive"])
    last = np.where(alive_m.any(0), snaps - 1 - np.argmax(alive_m[::-1], 0), 0)
    now = np.asarray(flat["now"])                                              # [snaps, envs, 2]
    future = [np.take_along_axis(now, np.minimum(np.arange(snaps)[:, None] + k, last[None, :])[..., None], 0)
              for k in DENSE_OFFSETS]
    out["dense"] = np.tanh(np.stack(future, -1) / 200.0).astype(np.float32).reshape(snaps * envs, 2, len(DENSE_OFFSETS))[ids]
    out["kinds"] = np.stack([np.broadcast_to(np.asarray(sides[6])[None, :], (snaps, envs)).reshape(-1)[ids],
                             np.broadcast_to(np.asarray(sides[7])[None, :], (snaps, envs)).reshape(-1)[ids]], 1).astype(np.int8)
    out["map"] = np.broadcast_to(np.asarray(map_ids)[None, :], (snaps, envs)).reshape(-1)[ids]
    out["env"] = np.broadcast_to(np.arange(envs, dtype=np.int32)[None, :], (snaps, envs)).reshape(-1)[ids]  # which game and
    out["snap"] = np.broadcast_to(np.arange(snaps, dtype=np.int32)[:, None], (snaps, envs)).reshape(-1)[ids]  # how far into it
    out["z1"] = np.broadcast_to(np.asarray(z1)[None, :], (snaps, envs)).reshape(-1)[ids]
    out["greedy"] = np.stack([np.broadcast_to(np.asarray(sides[4])[None, :], (snaps, envs)).reshape(-1)[ids],
                              np.broadcast_to(np.asarray(sides[5])[None, :], (snaps, envs)).reshape(-1)[ids]], 1)
    return out, c, np.asarray(z1)


def cmd_gen(cfg):
    os.makedirs(cfg.out, exist_ok=True)
    if cfg.bank_from:  # the same maps as another run, so the two sets of positions can be trained on together
        bank_np = dict(np.load(os.path.join(cfg.bank_from, "maps.npz")))
        assert len(bank_np["valid"]) >= cfg.maps + cfg.val_maps
    else:
        bank_np = maps.make_bank(cfg.maps + cfg.val_maps, seed=cfg.seed, param_fn=maps.cog2026_params)
    np.savez(os.path.join(cfg.out, "maps.npz"), **bank_np)
    params, heads = evaluate.load(cfg.value) if cfg.value else (None, None)
    # older networks the current one plays against (kind 3 sides): --value2 and --pool, picked by how well we do against them
    pool = []
    for path in [x for x in [cfg.value2] + cfg.pool.split(",") if x]:
        pr, hd = evaluate.load(path)
        pool.append((path, {k: v for k, v in pr.items() if not k.startswith("aux.")}, hd))
    heads2 = pool[0][2] if pool else None
    winrate = np.full(len(pool), 0.5)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    for name, lo, hi, nb in (("val", cfg.maps, cfg.maps + cfg.val_maps, 1), ("train", 0, cfg.maps, cfg.batches)):
        parts, t0 = [], time.time()
        n = min(cfg.envs, hi - lo)
        gen = make_gen(cfg, n, params, heads, bool(pool), heads2)
        for b in range(nb):
            ids = rng.integers(lo, hi, size=n)
            key, k = jax.random.split(key)
            j = None
            if pool:  # prioritized: opponents we beat less often get more games
                w = (1.0 - winrate + 0.05) ** cfg.pfsp
                j = int(rng.choice(len(pool), p=w / w.sum()))
            part, _, z1 = play_batch(cfg, bank, ids, rng, k, gen, pool[j][1] if pool else None)
            part["game"] = part.pop("env") + b * n
            parts.append(part)
            row = dict(set=name, batch=b + 1, positions=len(part["z1"]), win1=round(float((z1 > 0).mean()), 3),
                       win2=round(float((z1 < 0).mean()), 3), seconds=round(time.time() - t0, 1))
            if pool:
                k1, k2, z = part["kinds"][:, 0], part["kinds"][:, 1], part["z1"]
                main = ((k1 == 3) & np.isin(k2, (1, 2))) | ((k2 == 3) & np.isin(k1, (1, 2)))
                won = np.where(k1 == 3, z < 0, z > 0)[main]
                if main.any():
                    winrate[j] = 0.5 * winrate[j] + 0.5 * float(won.mean())
                row.update(opponent=pool[j][0], main_wins_vs_it=round(float(won.mean()), 3) if main.any() else None)
            print(json.dumps(row), flush=True)
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(cfg.out, f"{name}.npz"), **data)


# ---------------------------------------------------------------- training

def load_set(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def make_values(bank, heads, aux=False):
    """vmapped network value of positions (and, with aux, the extra outputs trained on the dense targets)."""
    pwnet = policy.pwnet

    def one(params, i, owner, ships, tick, me):
        s = fresh(env.from_bank(bank, i), owner, ships, tick)
        if not aux:
            return policy.apply(params, heads, s, me)[2]
        planet, glob, _, _ = pwnet.features(jnp, s, me)
        _, g = pwnet.trunk(jnp, params, heads, planet, glob, s.valid)
        hid = jnp.maximum(g @ params["v1.w"] + params["v1.b"], 0.0)
        return (hid @ params["v2.w"] + params["v2.b"])[0], hid @ params["aux.w"] + params["aux.b"]
    return jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0))


def td_returns(v, z, game, me, snap, lam):
    """lambda-returns along each side's positions in each game: G = (1 - lam) * v(next position) + lam * G(next), and
    the game's result after the last. v, z, game, me, snap: [rows], z from the mover's view."""
    order = np.lexsort((snap, me, game))
    v_o, z_o, g_o, m_o = v[order], z[order], game[order], me[order]
    last = np.ones(len(order), bool)
    last[:-1] = (g_o[1:] != g_o[:-1]) | (m_o[1:] != m_o[:-1])  # the last recorded position of that side in that game
    ret = np.empty(len(order), np.float32)
    nxt = 0.0
    for i in range(len(order) - 1, -1, -1):  # numpy loop over ~10^6 rows: a few seconds
        nxt = z_o[i] if last[i] else (1.0 - lam) * v_o[i + 1] + lam * nxt
        ret[i] = nxt
    out = np.empty_like(ret)
    out[order] = ret
    return out


def cmd_train(cfg):
    bank = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "maps.npz")).items()}
    train, val = load_set(os.path.join(cfg.data, "train.npz")), load_set(os.path.join(cfg.data, "val.npz"))
    for extra in [d for d in cfg.extra.split(",") if d]:  # more games on the same maps
        other = np.load(os.path.join(extra, "maps.npz"))
        assert all(np.array_equal(other[k], np.asarray(bank[k])) for k in ("pos", "growth", "max_ticks")), "different maps"
        more_t, more_v = load_set(os.path.join(extra, "train.npz")), load_set(os.path.join(extra, "val.npz"))
        keep = [k for k in train if k in more_t and k in more_v]  # earlier runs have no dense targets: then none are used
        train = {k: np.concatenate([train[k], more_t[k]]) for k in keep}
        val = {k: np.concatenate([val[k], more_v[k]]) for k in keep}
    use_aux = "dense" in train and cfg.aux_weight > 0
    params, heads = evaluate.load(cfg.init)
    n_aux = train["dense"].shape[-1] if use_aux else 0
    if n_aux and "aux.w" not in params:  # extra outputs on the value head, used only while training
        params["aux.w"] = jax.random.normal(jax.random.PRNGKey(cfg.seed + 1), (params["v1.w"].shape[1], n_aux)) * 0.01
        params["aux.b"] = jnp.zeros(n_aux)
    values = make_values(bank, heads)
    values_aux = make_values(bank, heads, aux=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0, eps=1e-5))
    opt_state = opt.init(params)

    # a position belongs to the side that chose it: side 1's record is labelled from player 1's view and vice versa
    def mover_arrays(d):
        p = len(d["z1"])
        me = np.concatenate([np.ones(p, np.int32), np.full(p, 2, np.int32)])
        z = np.concatenate([d["z1"], -d["z1"]]).astype(np.float32)
        return dict(map=np.concatenate([d["map"], d["map"]]), owner=np.concatenate([d["owner"][:, 0], d["owner"][:, 1]]),
                    ships=np.concatenate([d["ships"][:, 0], d["ships"][:, 1]]),
                    tick=np.concatenate([d["tick"] + d["h"][:, 0], d["tick"] + d["h"][:, 1]]),
                    hand=np.concatenate([d["hand"][:, 0], d["hand"][:, 1]]), me=me, z=z,
                    **({"game": np.concatenate([d["game"], d["game"]]), "snap": np.concatenate([d["snap"], d["snap"]])}
                       if cfg.td_lambda else {}),
                    greedy=np.concatenate([d["greedy"][:, 0], d["greedy"][:, 1]]),
                    **({"dense": np.concatenate([d["dense"][:, 0], d["dense"][:, 1]])} if use_aux else {}))
    tr_all, va_all = mover_arrays(train), mover_arrays(val)
    # drop positions where the mover was the greedy bot: its "chosen" position is just the planner's idea
    keep_t, keep_v = ~tr_all["greedy"].astype(bool), ~va_all["greedy"].astype(bool)
    tr = {k: v[keep_t] for k, v in tr_all.items() if k not in ("game", "snap")}
    va = {k: v[keep_v] for k, v in va_all.items() if k not in ("game", "snap")}
    ev = jax.jit(values)

    def targets(params):
        """What each kept position is trained towards: the game's result, or (td_lambda > 0) lambda-returns of the
        network's own values of the side's later positions in the same game, the result coming in only at the end."""
        if not cfg.td_lambda:
            return tr["z"], va["z"]
        out = []
        for name, d, keep in (("train", tr_all, keep_t), ("held-out", va_all, keep_v)):
            v = np.concatenate([np.asarray(ev(params, *(jnp.asarray(d[k][i:i + 8192]) for k in ("map", "owner", "ships", "tick", "me"))))
                                for i in range(0, len(d["z"]), 8192)])
            ret = td_returns(v, d["z"], d["game"], d["me"], d["snap"], cfg.td_lambda)
            print(f"  {name}: lambda-returns, mean |return - value| {np.mean(np.abs(ret - v)):.4f}, "
                  f"mean |return - result| {np.mean(np.abs(ret - d['z'])):.4f}", flush=True)
            out.append(ret[keep])
        return out
    n_tr = len(tr["z"])
    print(f"train positions {n_tr}, held-out {len(va['z'])}", flush=True)
    dev = lambda d: {k: jnp.asarray(v) for k, v in d.items()}  # noqa: E731
    tr_d, va_d = dev(tr), dev(va)
    tr_tgt, va_tgt = targets(params)
    tgt_d = jnp.asarray(tr_tgt)
    max_ticks = np.asarray(bank["max_ticks"])

    def loss_fn(params, b, target, flip):
        me = jnp.where(flip, 3 - b["me"], b["me"])
        z = jnp.where(flip, -target, target)
        if not n_aux:
            v = values(params, b["map"], b["owner"], b["ships"], b["tick"], me)
            return jnp.mean((v - z) ** 2)
        v, aux = values_aux(params, b["map"], b["owner"], b["ships"], b["tick"], me)
        dense = jnp.where(flip[:, None], -b["dense"], b["dense"])
        return jnp.mean((v - z) ** 2) + cfg.aux_weight * jnp.mean((aux - dense) ** 2)

    @jax.jit
    def train_chunk(params, opt_state, key, lr, tgt):
        def step(c, k):
            params, opt_state = c
            k1, k2 = jax.random.split(k)
            ids = jax.random.randint(k1, (cfg.minibatch,), 0, n_tr)
            b = jax.tree.map(lambda x: x[ids], tr_d)
            flip = jax.random.uniform(k2, (cfg.minibatch,)) < cfg.flip
            loss, g = jax.value_and_grad(loss_fn)(params, b, tgt[ids], flip)
            upd, opt_state = opt.update(g, opt_state, params)
            upd = jax.tree.map(lambda u: lr * u, upd)
            return (optax.apply_updates(params, upd), opt_state), loss
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jax.random.split(key, cfg.chunk_steps))
        return params, opt_state, losses.mean()

    eval_values = jax.jit(values)

    def held_out(params):
        pred = []
        for i in range(0, len(va["z"]), 8192):
            b = {k: v[i:i + 8192] for k, v in va_d.items()}
            pred.append(np.asarray(eval_values(params, b["map"], b["owner"], b["ships"], b["tick"], b["me"])))
        return np.concatenate(pred)

    def metrics(pred, z, phase, name):
        decisive = z != 0
        row = dict(mse=round(float(np.mean((pred - z) ** 2)), 4), acc=round(float(np.mean(np.sign(pred[decisive]) == z[decisive])), 4),
                   by_phase=[round(float(np.mean((pred[(phase >= lo) & (phase < hi)] - z[(phase >= lo) & (phase < hi)]) ** 2)), 4)
                             for lo, hi in zip(PHASES[:-1], PHASES[1:])])
        print(f"  {name}: {json.dumps(row)}", flush=True)
        return row

    phase = va["tick"] / np.maximum(max_ticks[va["map"]], 1)
    # the hand score, rescaled to fit the results as well as it can be (best of several scales)
    best = None
    for scale in (20, 50, 100, 200, 500, 1000, 3000):
        x = np.tanh(va["hand"] / scale)
        a = np.vstack([x, np.ones_like(x)]).T
        coef = np.linalg.lstsq(a, va["z"], rcond=None)[0]
        mse = float(np.mean((a @ coef - va["z"]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, scale, a @ coef)
    print(f"hand score (tanh(score/{best[1]}) rescaled):", flush=True)
    base_row = metrics(best[2], va["z"], phase, "hand")
    print("start:", flush=True)
    pred = held_out(params)
    metrics(pred, va["z"], phase, "net")
    if cfg.td_lambda:
        print(f"  net vs lambda-returns: {json.dumps(dict(mse=round(float(np.mean((pred - va_tgt) ** 2)), 4)))}", flush=True)
    os.makedirs(cfg.out, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    t0 = time.time()
    n_chunks = cfg.steps // cfg.chunk_steps
    best_mse = float("inf")
    for c in range(n_chunks):
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * c / max(n_chunks - 1, 1)
        key, k = jax.random.split(key)
        params, opt_state, loss = train_chunk(params, opt_state, k, lr, tgt_d)
        if (c + 1) % cfg.report_every == 0 or c + 1 == n_chunks:
            print(f"step {(c + 1) * cfg.chunk_steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
            pred = held_out(params)
            row = metrics(pred, va["z"], phase, "net")
            if cfg.td_lambda:  # kept by how well it predicts its own targets on held-out games
                row = dict(mse=round(float(np.mean((pred - va_tgt) ** 2)), 4))
                print(f"  net vs lambda-returns: {json.dumps(row)}", flush=True)
            save_params(os.path.join(cfg.out, "latest.npz"), params, argparse.Namespace(heads=heads))
            if cfg.td_refresh:  # the targets move, so there is no best to keep: latest.npz is the result
                pass
            elif row["mse"] < best_mse:  # the network overfits the games it sees: keep the best on held-out games
                best_mse = row["mse"]
                save_params(os.path.join(cfg.out, "best.npz"), params, argparse.Namespace(heads=heads))
                print(f"  (best so far at step {(c + 1) * cfg.chunk_steps})", flush=True)
        if cfg.td_refresh and ((c + 1) * cfg.chunk_steps) % cfg.td_refresh == 0 and c + 1 < n_chunks:
            print(f"step {(c + 1) * cfg.chunk_steps}: new lambda-returns from the network as it is now", flush=True)
            tr_tgt, va_tgt = targets(params)
            tgt_d = jnp.asarray(tr_tgt)
    json.dump(dict(hand=base_row), open(os.path.join(cfg.out, "hand.json"), "w"))


# ---------------------------------------------------------------- head to head

def parse_fractions(text):
    return tuple(float(x) for x in text.split(",")) if text else jplanner.FRACTIONS


def parse_extras(text):
    return tuple(e for e in text.split("+") if e) if text else ()


def opponent(name):
    """An agent function for 'hand' (the hand-scored planner), 'value:PATH:MIX[:TOP]' (value-ranked planner), a sample bot,
    or a policy .npz played greedily. The planner kinds take options after semicolons: 'hand;extras=cap+spare:0.5;fractions=0.25,0.5,1'
    (launch sizes beyond the fractions of the source's ships, see planner.make_planner), e.g.
    'value:v4.npz:0.5:16;extras=cap'."""
    head, *kv = name.split(";")
    opts = dict(x.split("=", 1) for x in kv)
    fractions, extras = parse_fractions(opts.get("fractions", "")), parse_extras(opts.get("extras", ""))
    if head == "hand":
        return make_hand_agent(12, 90, 8, fractions, extras)
    if head.startswith("value:"):  # value:PATH:MIX[:TOP], a value-ranked planner; ';reply=OURS:REPLIES' looks one reply deep
        path, mix, *top = head.split(":")[1:]
        params, heads = evaluate.load(path)
        if opts.get("reply"):
            ours, replies = (int(x) for x in opts["reply"].split(":"))
            return make_reply_agent(params, heads, 12, 90, int(top[0]) if top else 8, float(mix), fractions=fractions, extras=extras,
                                    ours=ours, replies=replies)
        return make_value_agent(params, heads, 12, 90, int(top[0]) if top else 8, float(mix), fractions=fractions, extras=extras)
    if head.endswith(".npz"):
        pol, heads = evaluate.load(head)
        return evaluate.agent_fn("policy", pol, heads, False)
    return evaluate.agent_fn(head)


def score_of(agent, other, bank, seed, cfg):
    sc, ticks = play_chunked(agent, other, bank, jax.random.PRNGKey(seed), cfg.chunk, cfg.max_steps)
    sc = np.asarray(sc)
    return sc, float(np.asarray(ticks).mean())


def cmd_gauntlet(cfg):
    """Greedy forward selection of launch sizes: every size in the pool is added to the current set in turn and the
    result played against the current set (same maps, both sides); the best is kept while it wins clearly."""
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.games, seed=cfg.seed, param_fn=maps.cog2026_params).items()}
    pool = [e for e in cfg.pool.split(",") if e]
    cur = list(parse_extras(cfg.start))
    base = cfg.base
    fr = cfg.fractions

    def agent(extras):
        return opponent(f"{base};extras={'+'.join(extras)};fractions={fr}")

    n_games = 2 * cfg.games
    se = 0.5 / np.sqrt(n_games)
    for rnd in range(cfg.rounds):
        cur_agent = agent(cur)
        results = []
        for c in pool:
            if c in cur:
                continue
            t0 = time.time()
            sc, mt = score_of(agent(cur + [c]), cur_agent, bank, cfg.seed, cfg)
            results.append((float(sc.mean()), c))
            print(json.dumps(dict(round=rnd + 1, current=cur, add=c, score=round(float(sc.mean()), 4), wins=int((sc == 1).sum()),
                                  losses=int((sc == 0).sum()), games=len(sc), mean_ticks=round(mt, 1),
                                  seconds=round(time.time() - t0, 1))), flush=True)
        if not results:
            break
        best = max(results)
        verdict = best[0] > 0.5 + cfg.z * se
        print(json.dumps(dict(round=rnd + 1, best=best[1], score=round(best[0], 4), needed=round(0.5 + cfg.z * se, 4), added=bool(verdict))), flush=True)
        if not verdict:
            break
        cur.append(best[1])
    # which of the chosen sizes can go without losing: drop each in turn and play the full set against the rest
    for c in list(cur):
        rest = [x for x in cur if x != c]
        sc, _ = score_of(agent(rest), agent(cur), bank, cfg.seed, cfg)
        print(json.dumps(dict(drop=c, score_without=round(float(sc.mean()), 4), games=len(sc))), flush=True)
    print(json.dumps(dict(final=cur, fractions=fr)), flush=True)


def cmd_match(cfg):
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.games, seed=cfg.seed, param_fn=maps.cog2026_params).items()}
    fractions, extras = parse_fractions(cfg.fractions), parse_extras(cfg.extras)
    if cfg.value == "hand":
        runs = [("hand", make_hand_agent(cfg.sources, cfg.horizon, cfg.top, fractions, extras))]
    else:
        params, heads = evaluate.load(cfg.value)
        runs = [(f"mix={m}", make_value_agent(params, heads, cfg.sources, cfg.horizon, cfg.top, float(m), wait=not cfg.no_wait,
                                              fractions=fractions, extras=extras))
                for m in cfg.mix.split(",")]
        if cfg.reply:  # OURS:REPLIES[:WEIGHT], comma separated: look one opponent reply deep (see make_reply_agent)
            runs = []
            for spec in cfg.reply.split(","):
                o, rp, *w = spec.split(":")
                runs.append((f"mix={cfg.mix};reply={spec}", make_reply_agent(
                    params, heads, cfg.sources, cfg.horizon, cfg.top, float(cfg.mix), wait=not cfg.no_wait,
                    fractions=fractions, extras=extras, ours=int(o), replies=int(rp), weight=float(w[0]) if w else 1.0)))
    for opp in cfg.vs.split(","):
        fb = opponent(opp)
        for name, agent in runs:
            t0 = time.time()
            sc, ticks = play_chunked(agent, fb, bank, jax.random.PRNGKey(cfg.seed), cfg.chunk, cfg.max_steps)
            sc = np.asarray(sc)
            print(json.dumps(dict(agent=name, value=cfg.value, extras=cfg.extras, fractions=cfg.fractions, vs=opp, top=cfg.top, horizon=cfg.horizon, games=len(sc),
                                  score=round(float(sc.mean()), 4), wins=int((sc == 1).sum()), draws=int((sc == 0.5).sum()),
                                  losses=int((sc == 0).sum()), mean_ticks=round(float(np.asarray(ticks).mean()), 1),
                                  seconds=round(time.time() - t0, 1))), flush=True)


def main(argv=None):
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--envs", type=int, default=1024)
    g.add_argument("--batches", type=int, default=4)
    g.add_argument("--maps", type=int, default=4096)
    g.add_argument("--val-maps", type=int, default=256)
    g.add_argument("--ticks", type=int, default=2001)
    g.add_argument("--chunk", type=int, default=240)
    g.add_argument("--stride", type=int, default=8)
    g.add_argument("--sources", type=int, default=12)
    g.add_argument("--horizon", type=int, default=100)
    g.add_argument("--top", type=int, default=8)
    g.add_argument("--p-greedy", type=float, default=0.1)
    g.add_argument("--value", default=None, help="value network: some sides then play it (see --kinds)")
    g.add_argument("--value2", default=None, help="a second, older value network for kind 3")
    g.add_argument("--pool", default="", help="more older value networks, comma separated: each batch plays one of them as kind 3, "
                   "picked more often the less we beat it")
    g.add_argument("--pfsp", type=float, default=2.0, help="how strongly the pool favours opponents we beat less often")
    g.add_argument("--kinds", default="1,0,0", help="odds a side plays: hand-scored planner, value alone, value mixed with hand, "
                   "second network mixed with hand")
    g.add_argument("--extras", default="", help="launch sizes beyond the fractions, '+' separated (see planner.make_planner)")
    g.add_argument("--fractions", default="", help="fractions of the source's ships to try (default 0.25,0.5,1)")
    g.add_argument("--bank-from", default=None, help="directory of an earlier run whose maps.npz to reuse")
    g.add_argument("--reply", default="", help="OURS:REPLIES: sides of kind 2 look one opponent reply deep (see look_ahead)")
    g.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init", required=True)
    t.add_argument("--extra", default="", help="more game directories on the same maps, comma separated")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--chunk-steps", type=int, default=100)
    t.add_argument("--report-every", type=int, default=20)
    t.add_argument("--minibatch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--lr-end", type=float, default=3e-5)
    t.add_argument("--flip", type=float, default=0.25, help="share of samples shown from the other side's view")
    t.add_argument("--aux-weight", type=float, default=0.5, help="weight of the dense targets (how the real position scores "
                   "24, 48 and 96 ticks later) next to the game result; 0 turns them off")
    t.add_argument("--td-lambda", type=float, default=0.0,
                   help="train towards lambda-returns of the starting network's values along each game instead of the result")
    t.add_argument("--td-refresh", type=int, default=0, help="recompute the lambda-returns with the network as it is every this many steps")
    t.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("match")
    m.add_argument("value", help="value network .npz, or 'hand' for the hand-scored planner")
    m.add_argument("--games", type=int, default=256)
    m.add_argument("--vs", default="hand", help="comma separated: hand, greedy, rush, sniper, random, or a policy .npz")
    m.add_argument("--mix", default="0")
    m.add_argument("--top", type=int, default=8)
    m.add_argument("--sources", type=int, default=12)
    m.add_argument("--horizon", type=int, default=90)
    m.add_argument("--no-wait", action="store_true")
    m.add_argument("--reply", default="", help="OURS:REPLIES[:WEIGHT], comma separated: look one opponent reply deep "
                   "(see make_reply_agent); one run each, at the single --mix")
    m.add_argument("--extras", default="", help="launch sizes beyond the fractions for the tested agent, '+' separated")
    m.add_argument("--fractions", default="", help="fractions of the source's ships for the tested agent")
    m.add_argument("--chunk", type=int, default=250)
    m.add_argument("--max-steps", type=int, default=2001)
    m.add_argument("--seed", type=int, default=777)
    q = sub.add_parser("gauntlet")
    q.add_argument("--base", default="hand", help="the agent whose launch sizes are chosen: hand, or value:PATH:MIX[:TOP]")
    q.add_argument("--pool", required=True, help="sizes to try, comma separated: f:0.35, cap, cap:1.2:1, spare, spare:0.5, ...")
    q.add_argument("--start", default="", help="sizes the set starts with, '+' separated")
    q.add_argument("--fractions", default="0.25,0.5,1.0", help="the fractions every set includes")
    q.add_argument("--games", type=int, default=256, help="maps (each played from both sides)")
    q.add_argument("--rounds", type=int, default=4)
    q.add_argument("--z", type=float, default=2.0, help="standard errors above one half needed to add a size")
    q.add_argument("--chunk", type=int, default=250)
    q.add_argument("--max-steps", type=int, default=2001)
    q.add_argument("--seed", type=int, default=777)
    cfg = a.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    {"gen": cmd_gen, "train": cmd_train, "match": cmd_match, "gauntlet": cmd_gauntlet}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
