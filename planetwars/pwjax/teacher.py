"""A slower, stronger "teacher" for the entry, and teaching the entry's network what it finds.

The entry has 200 ms a move. It values each launch by the position it leads to when nobody launches anything for 90
ticks, and looks one opponent answer deep at its best eight. The teacher has no time limit. It starts the same way
(the entry's own ranking picks its `ours` best options), then plays each option forward `ticks` ticks on a copy of the
game, both sides played by a cheaper planner (`roll`), and values where each one ends up the way the entry values a
projection (the network mixed with the hand score). It plays the option that ends up best.

    python -m pwjax.teacher match value.npz --games 64 --ours 8 --ticks 60 --roll value:16    # teacher vs entry
    python -m pwjax.teacher gen   value.npz --out tq --batches 4                            # its verdicts, recorded
    python -m pwjax.teacher train --data tq --init value.npz --out vq                       # the entry's network learns them
    python -m pwjax.lvalue match vq/best.npz --top 128 --mix 0.5 --extras spare --reply 8:4 \\
        --vs 'value:value.npz:0.5:128;extras=spare;reply=8:4'

What is taught: for every option the teacher plays forward, the position the entry's network is asked about (the
planner's projection of that launch) and the network's value of where the play-out ended (or the result, if the game
ended). That covers the options the entry did not pick as well as the one it did, which the game's result never
labels. Training starts from the entry's network and keeps its shape, so the entry stays as fast.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env, evaluate, maps, policy
from . import planner as jplanner
from .lvalue import fresh, look_ahead, make_hand_agent, make_reply_agent, make_value_agent, options, totals
from .ppo import save_params, tree_where
from .vsearch import play_chunked

def make_roll(spec, params, heads, sources=12, horizon=90):
    """The planner both sides play during a play-out: 'hand' (hand score only, 8 launches) or 'value:R' (the network
    re-ranking the hand score's R best launches, no look-ahead), both with `spare` launch sizes."""
    if spec == "hand":
        return make_hand_agent(sources, horizon, 8, jplanner.FRACTIONS, ("spare",))
    kind, r = spec.split(":")
    assert kind == "value", spec
    return make_value_agent(params, heads, sources, horizon, int(r), 0.5, extras=("spare",))


def end_value(params, heads, s, done, me, horizon=90, mix=0.5):
    """What a play-out's last position is worth to `me`, the way the entry values a projection: nobody launches
    anything for `horizon` ticks, then the network mixed with the hand score (a clear win or loss counts 10). A game
    that ended counts 10 for a win, -10 for a loss. Returns (that total, the network's value or +-1 where the game
    ended)."""
    h = jnp.maximum(jnp.minimum(horizon, s.max_ticks - s.tick), 1)
    o, sh = jplanner.project(s, horizon, h)
    p = fresh(s, o, sh, s.tick + h)
    v = policy.apply(params, heads, p, me)[2]
    hand = jplanner.score_final(o, sh, s, me, jplanner.near_matrix(s))
    tot = totals(s, me, o[None], hand[None], v[None], mix)[0]
    lead = env.leader(s)
    res = jnp.where(lead == me, 1.0, jnp.where(lead == 0, 0.0, -1.0))
    return jnp.where(done, 10.0 * res, tot), jnp.where(done, res, v)


def make_teacher(params, heads, top=128, ours=8, replies=4, mix=0.5, ticks=60, roll="value:16", extra=0, blend=0.0,
                 sources=12, horizon=90, extras=("spare",)):
    """Returns (act, judge).

    judge(s, me, key) -> dict: the entry's `ours` best options (its look-ahead's choice set), plus `extra` more of the
    planner's `top` launches picked at random, each played forward `ticks` ticks; for each, the projected position the
    entry's network sees (owner, ships [K, N], at tick s.tick + h), the entry's own score of it (`entry`, -inf where
    unusable, only for the look-ahead's options), the network's value of that projection (`v0`), and what the play-out
    makes of it (`q`, a total as the entry ranks, and `qv`, the network's value at the end or the result).
    act(s, me, key) -> the action whose play-out ends best (plus `blend` times the entry's own score)."""
    plan = jplanner.make_planner(sources, horizon, top, jplanner.FRACTIONS, extras)
    plan_r = jplanner.make_planner(sources, horizon, replies, jplanner.FRACTIONS, extras)
    roll_fn = make_roll(roll, params, heads, sources, horizon)
    k_all = ours + extra

    def playout(s, me, a, opp_a):
        """Our action `a` and the opponent's `opp_a` this tick, then `ticks - 1` ticks of the roll planner each side."""
        a1 = tuple(jnp.where(me == 1, x, y) for x, y in zip(a, opp_a))
        a2 = tuple(jnp.where(me == 2, x, y) for x, y in zip(a, opp_a))
        s1 = env.step(s, a1, a2)
        key = jax.random.PRNGKey(0)

        def body(c, _):
            st, done = c
            st2 = env.step(st, roll_fn(st, jnp.int32(1), key), roll_fn(st, jnp.int32(2), key))
            st2 = tree_where(done, st, st2)
            return (st2, done | env.is_terminal(st2)), None

        (sT, done), _ = jax.lax.scan(body, (s1, env.is_terminal(s1)), None, length=ticks - 1)
        return sT, done

    def judge(s, me, key):
        h = jnp.maximum(jnp.minimum(horizon, s.max_ticks - s.tick), 1)
        out, owners, ships, hand, ok, v = options(plan, s, me, h, top, params, heads)
        plain = jnp.where(ok, totals(s, me, owners, hand, v, mix), -jnp.inf)
        cand, val = look_ahead(plan_r, params, heads, s, me, h, out, plain, ours, mix)
        entry = val
        if extra:  # some of the other launches, so the network also learns about the ones it ranks lower
            rest = ok & ~jnp.any(jnp.arange(top + 1)[:, None] == cand[None, :], 1)
            more = jax.random.choice(key, top + 1, (extra,), replace=False, p=rest / jnp.maximum(rest.sum(), 1))
            cand = jnp.concatenate([cand, more])
            entry = jnp.concatenate([val, jnp.where(rest[more], jnp.nan, -jnp.inf)])  # nan: usable, not in the look-ahead
        usable = ~jnp.isneginf(entry)
        launch = (cand < top) & usable
        j = jnp.minimum(cand, top - 1)
        acts = (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
        opp_a = roll_fn(s, 3 - me, key)
        sT, done = jax.vmap(lambda a: playout(s, me, a, opp_a))(acts)
        q, qv = jax.vmap(lambda st, d: end_value(params, heads, st, d, me, horizon, mix))(sT, done)
        q = jnp.where(usable, q, -jnp.inf)
        return dict(cand=cand, acts=acts, entry=entry, usable=usable, owner=owners[cand], ships=ships[cand], v0=v[cand],
                    q=q, qv=qv, h=h, ended=done)

    def act(s, me, key):
        r = judge(s, me, key)
        score = r["q"] + blend * jnp.where(jnp.isfinite(r["entry"]), r["entry"], 0.0)
        score = jnp.where(r["usable"], score, -jnp.inf)
        k = jnp.argmax(score)
        return tuple(x[k] for x in r["acts"])

    return act, judge


def entry_agent(params, heads, top=128):
    """The entry as the JAX code plays it: the network re-ranks the planner's `top` launches (with `spare` sizes), half
    and half with the hand score, and looks one reply deep at its best eight options (four answers each)."""
    return make_reply_agent(params, heads, 12, 90, top, 0.5, extras=("spare",), ours=8, replies=4)


# ---------------------------------------------------------------- teacher against the entry

def cmd_match(cfg):
    params, heads = evaluate.load(cfg.value)
    params = {k: v for k, v in params.items() if not k.startswith("aux.")}
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.games, seed=cfg.seed, param_fn=maps.cog2026_params).items()}
    act, _ = make_teacher(params, heads, cfg.top, cfg.ours, 4, 0.5, cfg.ticks, cfg.roll, 0, cfg.blend)
    for opp in cfg.vs.split(","):
        if opp == "entry":
            fb = entry_agent(params, heads, cfg.top)
        else:
            from .lvalue import opponent
            fb = opponent(opp)
        t0 = time.time()
        sc, ticks = play_chunked(act, fb, bank, jax.random.PRNGKey(cfg.seed), cfg.chunk, cfg.max_steps)
        sc = np.asarray(sc)
        print(json.dumps(dict(agent="teacher", value=cfg.value, ours=cfg.ours, ticks=cfg.ticks, roll=cfg.roll, blend=cfg.blend,
                              top=cfg.top, vs=opp, games=len(sc), score=round(float(sc.mean()), 4), wins=int((sc == 1).sum()),
                              draws=int((sc == 0.5).sum()), losses=int((sc == 0).sum()),
                              mean_ticks=round(float(np.asarray(ticks).mean()), 1), seconds=round(time.time() - t0, 1))),
              flush=True)


# ---------------------------------------------------------------- recording the teacher's verdicts

def cmd_gen(cfg):
    """Games of the entry (at `--play-top` launches, with look-ahead) against itself; every `stride` ticks one side,
    picked at random, is judged by the teacher, and its options' projected positions are recorded with what the
    play-outs make of them. With `--teacher-plays`, the judged side then plays the teacher's choice."""
    os.makedirs(cfg.out, exist_ok=True)
    params, heads = evaluate.load(cfg.value)
    params = {k: v for k, v in params.items() if not k.startswith("aux.")}
    bank_np = maps.make_bank(cfg.maps, seed=cfg.seed, param_fn=maps.cog2026_params)
    np.savez(os.path.join(cfg.out, "maps.npz"), **bank_np)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    play = entry_agent(params, heads, cfg.play_top)
    _, judge = make_teacher(params, heads, cfg.top, cfg.ours, 4, 0.5, cfg.ticks, cfg.roll, cfg.extra, cfg.blend)
    n = cfg.envs
    v_play, v_judge = jax.vmap(play), jax.vmap(judge)

    @jax.jit
    def chunk(c, eps, key):
        def one(c, k):
            s, done = c
            ka, kb, kr = jax.random.split(k, 3)
            a1 = v_play(s, jnp.ones(n, jnp.int32), jax.random.split(ka, n))
            a2 = v_play(s, jnp.full(n, 2, jnp.int32), jax.random.split(kb, n))
            r1, r2 = jax.random.uniform(kr, (2, n)) < eps  # now and then a side waits, so games leave the entry's own path
            nop = (jnp.full(n, -1, jnp.int32), jnp.full(n, -1, jnp.int32), jnp.zeros(n, s.ships.dtype))
            a1 = tuple(jnp.where(r1, x, y) for x, y in zip(nop, a1))
            a2 = tuple(jnp.where(r2, x, y) for x, y in zip(nop, a2))
            s2 = tree_where(done, s, jax.vmap(env.step)(s, a1, a2))
            return (s2, done | jax.vmap(env.is_terminal)(s2)), None

        def block(c, k):
            ks = jax.random.split(k, cfg.stride + 2)
            c, _ = jax.lax.scan(one, c, ks[:-2])
            s, done = c
            me = jnp.where(jax.random.bernoulli(ks[-2], 0.5, (n,)), 1, 2).astype(jnp.int32)
            r = v_judge(s, me, jax.random.split(ks[-1], n))
            # this tick: the judged side plays the teacher's choice (or the entry's), the other side the entry
            other = v_play(s, 3 - me, jax.random.split(ks[-1], n))
            if cfg.teacher_plays:
                score = jnp.where(r["usable"], r["q"], -jnp.inf)
            else:
                score = jnp.where(jnp.isfinite(r["entry"]), r["entry"], -jnp.inf)
            pick = jnp.argmax(score, 1)
            mine = tuple(jnp.take_along_axis(x, pick[:, None], 1)[:, 0] for x in r["acts"])
            a1 = tuple(jnp.where(me == 1, x, y) for x, y in zip(mine, other))
            a2 = tuple(jnp.where(me == 2, x, y) for x, y in zip(mine, other))
            s2 = tree_where(done, s, jax.vmap(env.step)(s, a1, a2))
            rec = dict(owner=r["owner"].astype(jnp.int8), ships=r["ships"], q=r["q"], qv=r["qv"], v0=r["v0"], entry=r["entry"],
                       use=r["usable"] & ~done[:, None], tick=s.tick + r["h"], now=s.tick, me=me, ended=r["ended"],
                       src=r["acts"][0].astype(jnp.int8), dst=r["acts"][1].astype(jnp.int8), sent=r["acts"][2],
                       owner_now=s.owner.astype(jnp.int8), ships_now=s.ships)
            return (s2, done | jax.vmap(env.is_terminal)(s2)), rec

        return jax.lax.scan(block, c, jax.random.split(key, cfg.chunk // (cfg.stride + 1)))

    rng = np.random.default_rng(cfg.seed + 7919 * cfg.game_seed)  # maps from --seed, games from both
    key = jax.random.PRNGKey(cfg.seed + 7919 * cfg.game_seed)
    game0 = 0
    for name, nb in (("val", cfg.val_batches), ("train", cfg.batches)):
        parts, t0 = [], time.time()
        for b in range(nb):
            lo, hi = (0, cfg.maps - cfg.val_maps) if name == "train" else (cfg.maps - cfg.val_maps, cfg.maps)
            ids = rng.integers(lo, hi, size=n)
            s = jax.vmap(lambda i: env.from_bank(bank, i))(jnp.asarray(ids))
            c = (s, jax.vmap(env.is_terminal)(s))
            eps = float(rng.choice([0.0, 0.02, 0.05]))
            recs = []
            for _ in range(-(-cfg.game_ticks // cfg.chunk)):
                if bool(c[1].all()):
                    break
                key, k = jax.random.split(key)
                c, rec = chunk(c, eps, k)
                recs.append(jax.tree.map(np.asarray, rec))
            r = {k: np.concatenate([x[k] for x in recs]) for k in recs[0]}            # [snaps, n, ...]
            lead = np.asarray(jax.vmap(env.leader)(c[0]))
            z1 = np.where(lead == 1, 1.0, np.where(lead == 2, -1.0, 0.0)).astype(np.float32)
            z = np.where(r["me"] == 1, z1[None], -z1[None])                          # [snaps, n] from the judged side's view
            snaps, _, k_all = r["q"].shape
            keep = r["use"].any(-1).reshape(-1)                                      # one row per judged position
            flat = lambda a: a.reshape((snaps * n,) + a.shape[2:])[keep]  # noqa: E731
            part = dict(owner=flat(r["owner"]), ships=flat(r["ships"]).astype(np.float32), q=flat(r["q"]), qv=flat(r["qv"]),
                        v0=flat(r["v0"]), entry=flat(r["entry"]), use=flat(r["use"]), ended=flat(r["ended"]),
                        src=flat(r["src"]), dst=flat(r["dst"]), sent=flat(r["sent"]),
                        owner_now=flat(r["owner_now"]), ships_now=flat(r["ships_now"]),
                        tick=flat(r["tick"]), now=flat(r["now"]), me=flat(r["me"]), z=flat(z),
                        map=flat(np.broadcast_to(ids[None], (snaps, n))),
                        game=flat(np.broadcast_to(np.arange(n)[None] + game0, (snaps, n))),
                        snap=flat(np.broadcast_to(np.arange(snaps)[:, None], (snaps, n))))
            game0 += n
            parts.append(part)
            u = part["use"]
            # how the teacher's verdicts compare with the entry's: does it pick a different option, and which predicts
            # the game's result better (the option the entry picks, valued by the network now and by the play-out)
            ent = np.where(np.isfinite(part["entry"]), part["entry"], -np.inf)
            e_pick = np.argmax(ent, 1)
            t_pick = np.argmax(np.where(u, part["q"], -np.inf), 1)
            rows = np.arange(len(e_pick))
            print(json.dumps(dict(
                set=name, batch=b + 1, positions=int(len(rows)), options=int(u.sum()), seconds=round(time.time() - t0, 1),
                win1=round(float((z1 > 0).mean()), 3), differ=round(float(np.mean(t_pick != e_pick)), 4),
                corr_result_v0=round(float(np.corrcoef(part["v0"][rows, e_pick], part["z"])[0, 1]), 4),
                corr_result_qv=round(float(np.corrcoef(part["qv"][rows, e_pick], part["z"])[0, 1]), 4),
                mean_abs_qv_v0=round(float(np.mean(np.abs(part["qv"] - part["v0"])[u])), 4))), flush=True)
            # saved after every batch, so a lost runtime keeps what was done
            np.savez(os.path.join(cfg.out, f"{name}.npz"), **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def cmd_record(cfg):
    """Like `gen`, but only positions with a choice in them, and in two passes. First the entry plays `--envs` games
    against itself and every `stride` ticks the game is kept if a side can launch (that side is judged; if both can,
    one at random); then the teacher judges the kept positions `--judge-batch` at a time. In `gen` about two judged
    positions in three had nothing to launch, and all of them cost as much to judge."""
    os.makedirs(cfg.out, exist_ok=True)
    params, heads = evaluate.load(cfg.value)
    params = {k: v for k, v in params.items() if not k.startswith("aux.")}
    bank_np = maps.make_bank(cfg.maps, seed=cfg.seed, param_fn=maps.cog2026_params)
    np.savez(os.path.join(cfg.out, "maps.npz"), **bank_np)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    play = entry_agent(params, heads, cfg.play_top)
    _, judge = make_teacher(params, heads, cfg.top, cfg.ours, 4, 0.5, cfg.ticks, cfg.roll, cfg.extra, cfg.blend)
    n, jb = cfg.envs, cfg.judge_batch
    v_play = jax.vmap(play)
    v_judge = jax.jit(jax.vmap(judge))
    blocks = max(cfg.chunk // cfg.stride, 1)

    def can_launch(s, p):
        return jnp.any(s.valid & (s.owner == p) & ~s.tr_on & (s.ships >= 2.0))

    @jax.jit
    def chunk(c, eps, key):
        def one(c, k):
            s, done = c
            ka, kb, kr = jax.random.split(k, 3)
            a1 = v_play(s, jnp.ones(n, jnp.int32), jax.random.split(ka, n))
            a2 = v_play(s, jnp.full(n, 2, jnp.int32), jax.random.split(kb, n))
            r1, r2 = jax.random.uniform(kr, (2, n)) < eps  # now and then a side waits, so games leave the entry's own path
            nop = (jnp.full(n, -1, jnp.int32), jnp.full(n, -1, jnp.int32), jnp.zeros(n, s.ships.dtype))
            a1 = tuple(jnp.where(r1, x, y) for x, y in zip(nop, a1))
            a2 = tuple(jnp.where(r2, x, y) for x, y in zip(nop, a2))
            s2 = tree_where(done, s, jax.vmap(env.step)(s, a1, a2))
            return (s2, done | jax.vmap(env.is_terminal)(s2)), None

        def block(c, k):
            ks = jax.random.split(k, cfg.stride + 1)
            c, _ = jax.lax.scan(one, c, ks[:-1])
            s, done = c
            d1 = jax.vmap(can_launch, (0, None))(s, 1)
            d2 = jax.vmap(can_launch, (0, None))(s, 2)
            coin = jax.random.bernoulli(ks[-1], 0.5, (n,))
            me = jnp.where(d1 & d2, jnp.where(coin, 1, 2), jnp.where(d1, 1, 2)).astype(jnp.int32)
            return c, dict(state=s, me=me, keep=(d1 | d2) & ~done)

        return jax.lax.scan(block, c, jax.random.split(key, blocks))

    rng = np.random.default_rng(cfg.seed + 7919 * cfg.game_seed)
    key = jax.random.PRNGKey(cfg.seed + 7919 * cfg.game_seed)
    game0 = 0
    for name, nb in (("val", cfg.val_batches), ("train", cfg.batches)):
        parts = []
        for b in range(nb):
            t0 = time.time()
            lo, hi = (0, cfg.maps - cfg.val_maps) if name == "train" else (cfg.maps - cfg.val_maps, cfg.maps)
            ids = rng.integers(lo, hi, size=n)
            s = jax.vmap(lambda i: env.from_bank(bank, i))(jnp.asarray(ids))
            c = (s, jax.vmap(env.is_terminal)(s))
            eps = float(rng.choice([0.0, 0.02, 0.05]))
            kept = []                                     # (states [m], me [m], env index [m], snapshot number [m])
            snap = 0
            for _ in range(-(-cfg.game_ticks // (blocks * cfg.stride))):
                if bool(c[1].all()):
                    break
                key, k = jax.random.split(key)
                c, rec = chunk(c, eps, k)
                rec = jax.tree.map(np.asarray, rec)
                for i in range(rec["keep"].shape[0]):
                    e = np.nonzero(rec["keep"][i])[0]
                    if e.size:
                        kept.append((jax.tree.map(lambda x: x[i][e], rec["state"]), rec["me"][i][e], e, np.full(e.size, snap + i)))
                snap += rec["keep"].shape[0]
            played = time.time() - t0
            lead = np.asarray(jax.vmap(env.leader)(c[0]))
            z1 = np.where(lead == 1, 1.0, np.where(lead == 2, -1.0, 0.0)).astype(np.float32)
            states = jax.tree.map(lambda *x: np.concatenate(x), *[k_[0] for k_ in kept])
            me = np.concatenate([k_[1] for k_ in kept])
            env_i = np.concatenate([k_[2] for k_ in kept])
            snaps = np.concatenate([k_[3] for k_ in kept])
            m = len(me)
            out = []
            for j0 in range(0, m, jb):                    # the teacher, `jb` positions at a time (the last one padded)
                idx = np.minimum(np.arange(j0, j0 + jb), m - 1)
                key, k = jax.random.split(key)
                r = v_judge(jax.tree.map(lambda x: jnp.asarray(x[idx]), states), jnp.asarray(me[idx]), jax.random.split(k, jb))
                r = jax.tree.map(np.asarray, r)
                take = slice(0, min(jb, m - j0))
                out.append(dict(owner=r["owner"][take].astype(np.int8), ships=r["ships"][take].astype(np.float32),
                                q=r["q"][take], qv=r["qv"][take], v0=r["v0"][take], entry=r["entry"][take],
                                use=r["usable"][take], ended=r["ended"][take], src=r["acts"][0][take].astype(np.int8),
                                dst=r["acts"][1][take].astype(np.int8), sent=r["acts"][2][take],
                                tick=(states.tick[idx] + r["h"])[take]))
            part = {k_: np.concatenate([o[k_] for o in out]) for k_ in out[0]}
            part.update(owner_now=states.owner.astype(np.int8), ships_now=states.ships.astype(np.float32), now=states.tick,
                        me=me.astype(np.int32), z=np.where(me == 1, z1[env_i], -z1[env_i]).astype(np.float32),
                        map=ids[env_i], game=env_i + game0, snap=snaps)
            keep = part["use"].sum(1) >= 2                # a launch to choose besides waiting
            part = {k_: v[keep] for k_, v in part.items()}
            game0 += n
            parts.append(part)
            u = part["use"]
            ent = np.where(np.isfinite(part["entry"]), part["entry"], -np.inf)
            e_pick, t_pick = np.argmax(ent, 1), np.argmax(np.where(u, part["q"], -np.inf), 1)
            rows = np.arange(len(e_pick))
            print(json.dumps(dict(
                set=name, batch=b + 1, kept=m, positions=int(keep.sum()), options=int(u.sum()),
                seconds_play=round(played, 1), seconds=round(time.time() - t0, 1), win1=round(float((z1 > 0).mean()), 3),
                differ=round(float(np.mean(t_pick != e_pick)), 4),
                q_lost_entry=round(float(np.mean(part["q"][rows, t_pick] - part["q"][rows, e_pick])), 4),
                corr_result_qv=round(float(np.corrcoef(part["qv"][rows, e_pick], part["z"])[0, 1]), 4))), flush=True)
            np.savez(os.path.join(cfg.out, f"{name}.npz"), **{k_: np.concatenate([p[k_] for p in parts]) for k_ in parts[0]})


# ---------------------------------------------------------------- teaching the entry's network

def cmd_train(cfg):
    """The entry's network learns, for each recorded option, `qv` (what the play-out made of it), mixed with `outcome`
    of the game's result where the option is the one played; `--rank` adds a loss that asks the network's ranking of a
    position's options (mixed with the hand score as the entry ranks) to match the play-outs'."""
    bank = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "maps.npz")).items()}
    load = lambda p: {k: np.asarray(v) for k, v in np.load(p).items()}  # noqa: E731
    tr, va = load(os.path.join(cfg.data, "train.npz")), load(os.path.join(cfg.data, "val.npz"))
    for extra in [d for d in cfg.extra.split(",") if d]:
        other = np.load(os.path.join(extra, "maps.npz"))
        assert all(np.array_equal(other[k], np.asarray(bank[k])) for k in ("pos", "growth", "max_ticks")), "different maps"
        t2, v2 = load(os.path.join(extra, "train.npz")), load(os.path.join(extra, "val.npz"))
        tr = {k: np.concatenate([tr[k], t2[k]]) for k in tr}
        va = {k: np.concatenate([va[k], v2[k]]) for k in va}
    params, heads = evaluate.load(cfg.init)
    params = {k: v for k, v in params.items() if not k.startswith("aux.")}

    def values(params, b):
        """[B, K] network values of the recorded options."""
        def one(i, owner, ships, tick, me):
            s = fresh(env.from_bank(bank, i), owner, ships, tick)
            return policy.apply(params, heads, s, me)[2]
        per_opt = jax.vmap(one, in_axes=(None, 0, 0, None, None))
        return jax.vmap(per_opt)(b["map"], b["owner"], b["ships"], b["tick"], b["me"])

    def hand_of(b):
        def one(i, owner, ships, me):
            s = env.from_bank(bank, i)
            return jplanner.score_final(owner.astype(jnp.int32), ships, s, me, jplanner.near_matrix(s))
        return jax.vmap(jax.vmap(one, in_axes=(None, 0, 0, None)))(b["map"], b["owner"], b["ships"], b["me"])

    def prep(d):
        d = {k: v for k, v in d.items() if k in ("owner", "ships", "qv", "use", "tick", "me", "map", "q")}
        d["use"] = d["use"].astype(np.float32)
        d["q"] = np.where(np.isfinite(d["q"]), d["q"], 0.0).astype(np.float32)
        return {k: jnp.asarray(v) for k, v in d.items()}

    tr_d, va_d = prep(tr), prep(va)
    tr_d["hand"] = jnp.concatenate([hand_of({k: v[i:i + 4096] for k, v in tr_d.items()}) for i in range(0, len(tr["me"]), 4096)])
    va_d["hand"] = jnp.concatenate([hand_of({k: v[i:i + 4096] for k, v in va_d.items()}) for i in range(0, len(va["me"]), 4096)])
    n_tr = len(tr["me"])
    print(f"train positions {n_tr} ({int(tr['use'].sum())} options), held-out {len(va['me'])} ({int(va['use'].sum())})", flush=True)

    def losses(params, b):
        v = values(params, b)
        u = b["use"]
        mse = jnp.sum(u * (v - b["qv"]) ** 2) / jnp.maximum(u.sum(), 1.0)
        if not cfg.rank:
            return mse, mse
        # the entry ranks options by 0.5 * v + 0.5 * tanh(hand / 200): ask that ranking to agree with the play-outs'
        tot = 0.5 * v + 0.5 * jnp.tanh(b["hand"] / 200.0)
        logit = jnp.where(u > 0, cfg.rank_temp * tot, -1e9)
        target = jax.nn.softmax(jnp.where(u > 0, cfg.rank_temp * b["q"], -1e9), -1)
        ce = -jnp.sum(target * jax.nn.log_softmax(logit, -1), -1)
        ent = -jnp.sum(target * jnp.log(jnp.maximum(target, 1e-12)), -1)
        rank = jnp.mean(ce - ent)  # KL divergence: zero when the rankings agree
        return mse + cfg.rank * rank, mse

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0, eps=1e-5))
    opt_state = opt.init(params)

    @jax.jit
    def train_chunk(params, opt_state, key, lr):
        def step(c, k):
            params, opt_state = c
            ids = jax.random.randint(k, (cfg.minibatch,), 0, n_tr)
            b = jax.tree.map(lambda x: x[ids], tr_d)
            (loss, _), g = jax.value_and_grad(losses, has_aux=True)(params, b)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, jax.tree.map(lambda u: lr * u, upd)), opt_state), loss
        (params, opt_state), ls = jax.lax.scan(step, (params, opt_state), jax.random.split(key, cfg.chunk_steps))
        return params, opt_state, ls.mean()

    eval_v = jax.jit(values)

    def held_out(params):
        pred = np.concatenate([np.asarray(eval_v(params, {k: v[i:i + 1024] for k, v in va_d.items()}))
                               for i in range(0, len(va["me"]), 1024)])
        u = va["use"].astype(bool)
        mse = float(np.mean((pred - va["qv"])[u] ** 2))
        # how often the network (mixed with the hand score) picks the option the play-outs like best
        tot = 0.5 * pred + 0.5 * np.tanh(np.asarray(va_d["hand"]) / 200.0)
        agree = float(np.mean(np.argmax(np.where(u, tot, -np.inf), 1) == np.argmax(np.where(u, va["q"], -np.inf), 1)))
        # as a judge of the look-ahead's options alone (the ones with an entry score): what its pick costs by the
        # play-outs, against the best of them, where there is a choice; `entry_lost` is the same for the entry's pick
        la = u & np.isfinite(va["entry"])
        dec = la.sum(1) >= 2
        q = np.where(la, va["q"], -np.inf)[dec]
        rows = np.arange(len(q))
        pick = np.argmax(np.where(la, tot, -np.inf)[dec], 1)
        e_pick = np.argmax(np.where(la, va["entry"], -np.inf)[dec], 1)
        lost = float(np.mean(q.max(1) - q[rows, pick]))
        return dict(mse=round(mse, 5), agree=round(agree, 4), lost=round(lost, 4),
                    entry_lost=round(float(np.mean(q.max(1) - q[rows, e_pick])), 4))

    os.makedirs(cfg.out, exist_ok=True)
    start = held_out(params)
    print("start:", json.dumps(start), flush=True)
    key, t0 = jax.random.PRNGKey(cfg.seed), time.time()
    crit = {"mse": lambda r: r["mse"], "agree": lambda r: -r["agree"], "lost": lambda r: r["lost"]}[cfg.keep]
    best = crit(start)
    n_chunks = cfg.steps // cfg.chunk_steps
    for c in range(n_chunks):
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * c / max(n_chunks - 1, 1)
        key, k = jax.random.split(key)
        params, opt_state, loss = train_chunk(params, opt_state, k, lr)
        if (c + 1) % cfg.report_every == 0 or c + 1 == n_chunks:
            row = held_out(params)
            print(json.dumps(dict(step=(c + 1) * cfg.chunk_steps, loss=round(float(loss), 5), **row,
                                  seconds=round(time.time() - t0))), flush=True)
            save_params(os.path.join(cfg.out, "latest.npz"), params, argparse.Namespace(heads=heads))
            m = crit(row)
            if m < best:
                best = m
                save_params(os.path.join(cfg.out, "best.npz"), params, argparse.Namespace(heads=heads))
                print(f"  (best so far at step {(c + 1) * cfg.chunk_steps})", flush=True)


def main(argv=None):
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)

    def teacher_args(p):
        p.add_argument("--top", type=int, default=128, help="launches the network re-ranks (as the entry)")
        p.add_argument("--ours", type=int, default=8, help="options played forward")
        p.add_argument("--ticks", type=int, default=60, help="how far each option is played forward")
        p.add_argument("--roll", default="value:16", help="who plays the play-outs: hand, or value:R")
        p.add_argument("--blend", type=float, default=0.0, help="weight of the entry's own score next to the play-out's")
        p.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("match")
    m.add_argument("value")
    teacher_args(m)
    m.add_argument("--games", type=int, default=64, help="maps, each played from both sides")
    m.add_argument("--vs", default="entry", help="entry, or opponents as lvalue.opponent names them (comma separated)")
    m.add_argument("--chunk", type=int, default=50)
    m.add_argument("--max-steps", type=int, default=2001)
    g = sub.add_parser("gen")
    g.add_argument("value")
    teacher_args(g)
    g.add_argument("--out", required=True)
    g.add_argument("--extra", type=int, default=4, help="more options played forward, picked at random from the rest")
    g.add_argument("--play-top", type=int, default=64, help="launches re-ranked by the entry playing the games")
    g.add_argument("--teacher-plays", action="store_true", help="the judged side plays the teacher's choice")
    g.add_argument("--envs", type=int, default=512)
    g.add_argument("--batches", type=int, default=4)
    g.add_argument("--val-batches", type=int, default=1)
    g.add_argument("--maps", type=int, default=8192)
    g.add_argument("--val-maps", type=int, default=512)
    g.add_argument("--stride", type=int, default=15, help="ticks of plain play between judged ticks")
    g.add_argument("--chunk", type=int, default=160)
    g.add_argument("--game-ticks", type=int, default=2001, help="stop the games after this many ticks (for quick tests)")
    g.add_argument("--game-seed", type=int, default=0, help="other games on the same maps (--seed picks the maps)")
    r = sub.add_parser("record")
    r.add_argument("value")
    teacher_args(r)
    r.add_argument("--out", required=True)
    r.add_argument("--extra", type=int, default=0, help="more options played forward, picked at random from the rest")
    r.add_argument("--play-top", type=int, default=64, help="launches re-ranked by the entry playing the games")
    r.add_argument("--envs", type=int, default=256, help="games played at once")
    r.add_argument("--judge-batch", type=int, default=512, help="positions the teacher judges at once")
    r.add_argument("--batches", type=int, default=4)
    r.add_argument("--val-batches", type=int, default=1)
    r.add_argument("--maps", type=int, default=8192)
    r.add_argument("--val-maps", type=int, default=512)
    r.add_argument("--stride", type=int, default=10, help="ticks between kept positions of a game")
    r.add_argument("--chunk", type=int, default=200)
    r.add_argument("--game-ticks", type=int, default=2001)
    r.add_argument("--game-seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--extra", default="", help="more recorded directories on the same maps, comma separated")
    t.add_argument("--init", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=float, default=0.0, help="weight of the ranking loss")
    t.add_argument("--rank-temp", type=float, default=10.0)
    t.add_argument("--keep", choices=("mse", "agree", "lost"), default="mse")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--chunk-steps", type=int, default=100)
    t.add_argument("--report-every", type=int, default=5)
    t.add_argument("--minibatch", type=int, default=256, help="positions per step (each with all its options)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-end", type=float, default=1e-5)
    t.add_argument("--seed", type=int, default=0)
    cfg = a.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    {"match": cmd_match, "gen": cmd_gen, "record": cmd_record, "train": cmd_train}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
