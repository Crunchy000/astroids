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
    v = None
    if params is not None:
        v = jax.vmap(lambda o, sh: policy.apply(params, heads, fresh(s, o, sh, s.tick + h_eff), me)[2])(owners, ships)
    return out, owners, ships, hand, ok, v


def totals(s, me, owners, hand, v, mix, scale=200.0):
    """The value network's view of each option, mixed with the hand score: (1 - mix) * v + mix * tanh(hand / scale),
    and a clear win or loss counts for 10 either way."""
    mine, opp = s.valid & (owners == me), s.valid & (owners == 3 - me)
    won, lost = jnp.sum(opp, -1) == 0, jnp.sum(mine, -1) == 0
    total = (1 - mix) * v + mix * jnp.tanh(jnp.where(jnp.isfinite(hand), hand, 0.0) / scale)
    return total + jnp.where(won & ~lost, 10.0, 0.0) - jnp.where(lost & ~won, 10.0, 0.0)


def make_decider(plan, top, params=None, heads=None):
    """decide(s, me, h, eps, kind, key) -> (action, (afterstate owners, ships, ticks looked ahead, hand score)).

    kind 0 plays the planner's hand score, 1 the value network alone, 2 the network mixed half and half with the
    hand score. With probability eps one of the usable options is picked at random instead."""
    def decide(s, me, h, eps, kind, key):
        h_eff = jnp.maximum(jnp.minimum(h, s.max_ticks - s.tick), 1)
        out, owners, ships, hand, ok, v = options(plan, s, me, h_eff, top, params, heads)
        by_hand = jnp.where(ok, hand - jnp.where(jnp.arange(top + 1) == top, 2e-3, 0.0), -jnp.inf)  # waiting wins ties
        if params is None:
            score = by_hand
        else:
            by_v = jnp.where(ok, totals(s, me, owners, hand, v, 0.0), -jnp.inf)
            by_mix = jnp.where(ok, totals(s, me, owners, hand, v, 0.5), -jnp.inf)
            score = jnp.where(kind == 0, by_hand, jnp.where(kind == 1, by_v, by_mix))
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


def make_hand_agent(sources=12, horizon=90, top=8):
    """The planner as an agent function (state, me, key) -> action."""
    decide = make_decider(jplanner.make_planner(sources, horizon, top), top)
    return lambda s, me, key: decide(s, me, horizon, 0.0, 0, key)[0]


def make_value_agent(params, heads, sources=12, horizon=90, top=8, mix=0.0, scale=200.0, wait=True):
    """The planner's `top` launches (by hand score) and waiting, re-ranked by the value network
    (mixed with the hand score as `(1 - mix) * value + mix * tanh(hand / scale)`)."""
    plan = jplanner.make_planner(sources, horizon, top)

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


# ---------------------------------------------------------------- game generation

def make_gen(cfg, n_envs, params=None, heads=None):
    """gen(state carry, per-side settings, key) plays `cfg.chunk` ticks, recording every `stride`th."""
    plan = jplanner.make_planner(cfg.sources, cfg.horizon, cfg.top)
    decide = make_decider(plan, cfg.top, params, heads)
    v_decide = jax.vmap(decide)
    v_step, v_term, v_greedy = jax.vmap(env.step), jax.vmap(env.is_terminal), jax.vmap(baselines.greedy)
    one, two = jnp.ones(n_envs, jnp.int32), jnp.full(n_envs, 2, jnp.int32)

    def tick(c, sides, k):
        s, done = c
        h1, h2, e1, e2, g1, g2, q1, q2 = sides
        ka, kb = jax.random.split(k)
        a1, r1 = v_decide(s, one, h1, e1, q1, jax.random.split(ka, n_envs))
        a2, r2 = v_decide(s, two, h2, e2, q2, jax.random.split(kb, n_envs))
        gr1, gr2 = v_greedy(s, one), v_greedy(s, two)
        a1 = tuple(jnp.where(g1, g, p) for g, p in zip(gr1, a1))
        a2 = tuple(jnp.where(g2, g, p) for g, p in zip(gr2, a2))
        s2 = tree_where(done, s, v_step(s, a1, a2))
        return (s2, done | v_term(s2)), (r1, r2)

    def chunk(c, sides, key):
        def one_tick(c, k):
            return tick(c, sides, k)[0], None

        def snap_block(c, k):
            ks = jax.random.split(k, cfg.stride)
            c1, _ = jax.lax.scan(one_tick, c, ks[:-1])
            alive = ~c1[1]
            c2, (r1, r2) = tick(c1, sides, ks[-1])
            rec = dict(alive=alive, tick=c1[0].tick,
                       owner=jnp.stack([r1[0], r2[0]], 1).astype(jnp.int8), ships=jnp.stack([r1[1], r2[1]], 1),
                       h=jnp.stack([r1[2], r2[2]], 1), hand=jnp.stack([r1[3], r2[3]], 1))
            return c2, rec

        return jax.lax.scan(snap_block, c, jax.random.split(key, cfg.chunk // cfg.stride))

    return jax.jit(chunk)


def play_batch(cfg, bank, map_ids, rng, key, gen):
    """Play one batch of games from the given maps; returns the recorded positions and results."""
    n = len(map_ids)
    pick = lambda a: jnp.asarray(rng.choice(a, size=n).astype(a.dtype))  # noqa: E731
    greedy = lambda: jnp.asarray(rng.random(n) < cfg.p_greedy)  # noqa: E731
    kind_p = np.array([float(x) for x in cfg.kinds.split(",")])
    kinds = lambda: jnp.asarray(rng.choice(3, size=n, p=kind_p[:3] / kind_p[:3].sum()).astype(np.int32))  # noqa: E731
    sides = (pick(HORIZONS), pick(HORIZONS), pick(EPS), pick(EPS), greedy(), greedy(), kinds(), kinds())
    mi = jnp.asarray(map_ids)
    s = jax.vmap(lambda i: env.from_bank(bank, i))(mi)
    c = (s, jax.vmap(env.is_terminal)(s))
    recs = []
    for _ in range(-(-cfg.ticks // cfg.chunk)):
        if bool(c[1].all()):
            break
        key, k = jax.random.split(key)
        c, rec = gen(c, sides, k)
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
    out["map"] = np.broadcast_to(np.asarray(map_ids)[None, :], (snaps, envs)).reshape(-1)[ids]
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
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    for name, lo, hi, nb in (("val", cfg.maps, cfg.maps + cfg.val_maps, 1), ("train", 0, cfg.maps, cfg.batches)):
        parts, t0 = [], time.time()
        n = min(cfg.envs, hi - lo)
        gen = make_gen(cfg, n, params, heads)
        for b in range(nb):
            ids = rng.integers(lo, hi, size=n)
            key, k = jax.random.split(key)
            part, _, z1 = play_batch(cfg, bank, ids, rng, k, gen)
            parts.append(part)
            print(json.dumps(dict(set=name, batch=b + 1, positions=len(part["z1"]), win1=round(float((z1 > 0).mean()), 3),
                                  win2=round(float((z1 < 0).mean()), 3), seconds=round(time.time() - t0, 1))), flush=True)
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(cfg.out, f"{name}.npz"), **data)


# ---------------------------------------------------------------- training

def load_set(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def make_values(bank, heads):
    def one(params, i, owner, ships, tick, me):
        s = fresh(env.from_bank(bank, i), owner, ships, tick)
        return policy.apply(params, heads, s, me)[2]
    return jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0))


def cmd_train(cfg):
    bank = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "maps.npz")).items()}
    train, val = load_set(os.path.join(cfg.data, "train.npz")), load_set(os.path.join(cfg.data, "val.npz"))
    for extra in [d for d in cfg.extra.split(",") if d]:  # more games on the same maps
        other = np.load(os.path.join(extra, "maps.npz"))
        assert all(np.array_equal(other[k], np.asarray(bank[k])) for k in ("pos", "growth", "max_ticks")), "different maps"
        train = {k: np.concatenate([train[k], v]) for k, v in load_set(os.path.join(extra, "train.npz")).items()}
        val = {k: np.concatenate([val[k], v]) for k, v in load_set(os.path.join(extra, "val.npz")).items()}
    params, heads = evaluate.load(cfg.init)
    values = make_values(bank, heads)
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
                    greedy=np.concatenate([d["greedy"][:, 0], d["greedy"][:, 1]]))
    tr, va = mover_arrays(train), mover_arrays(val)
    # drop positions where the mover was the greedy bot: its "chosen" position is just the planner's idea
    keep_t, keep_v = ~tr["greedy"].astype(bool), ~va["greedy"].astype(bool)
    tr = {k: v[keep_t] for k, v in tr.items()}
    va = {k: v[keep_v] for k, v in va.items()}
    n_tr = len(tr["z"])
    print(f"train positions {n_tr}, held-out {len(va['z'])}", flush=True)
    dev = lambda d: {k: jnp.asarray(v) for k, v in d.items()}  # noqa: E731
    tr_d, va_d = dev(tr), dev(va)
    max_ticks = np.asarray(bank["max_ticks"])

    def loss_fn(params, b, flip):
        me = jnp.where(flip, 3 - b["me"], b["me"])
        z = jnp.where(flip, -b["z"], b["z"])
        v = values(params, b["map"], b["owner"], b["ships"], b["tick"], me)
        return jnp.mean((v - z) ** 2)

    @jax.jit
    def train_chunk(params, opt_state, key, lr):
        def step(c, k):
            params, opt_state = c
            k1, k2 = jax.random.split(k)
            ids = jax.random.randint(k1, (cfg.minibatch,), 0, n_tr)
            b = jax.tree.map(lambda x: x[ids], tr_d)
            flip = jax.random.uniform(k2, (cfg.minibatch,)) < cfg.flip
            loss, g = jax.value_and_grad(loss_fn)(params, b, flip)
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
    metrics(held_out(params), va["z"], phase, "net")
    os.makedirs(cfg.out, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    t0 = time.time()
    n_chunks = cfg.steps // cfg.chunk_steps
    best_mse = float("inf")
    for c in range(n_chunks):
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * c / max(n_chunks - 1, 1)
        key, k = jax.random.split(key)
        params, opt_state, loss = train_chunk(params, opt_state, k, lr)
        if (c + 1) % cfg.report_every == 0 or c + 1 == n_chunks:
            print(f"step {(c + 1) * cfg.chunk_steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
            row = metrics(held_out(params), va["z"], phase, "net")
            save_params(os.path.join(cfg.out, "latest.npz"), params, argparse.Namespace(heads=heads))
            if row["mse"] < best_mse:  # the network overfits the games it sees: keep the best on held-out games
                best_mse = row["mse"]
                save_params(os.path.join(cfg.out, "best.npz"), params, argparse.Namespace(heads=heads))
                print(f"  (best so far at step {(c + 1) * cfg.chunk_steps})", flush=True)
    json.dump(dict(hand=base_row), open(os.path.join(cfg.out, "hand.json"), "w"))


# ---------------------------------------------------------------- head to head

def opponent(name):
    """An agent function for 'hand' (the hand-scored planner), a sample bot, or a policy .npz played greedily."""
    if name == "hand":
        return make_hand_agent(12, 90, 8)
    if name.endswith(".npz"):
        pol, heads = evaluate.load(name)
        return evaluate.agent_fn("policy", pol, heads, False)
    return evaluate.agent_fn(name)


def cmd_match(cfg):
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.games, seed=cfg.seed, param_fn=maps.cog2026_params).items()}
    if cfg.value == "hand":
        runs = [("hand", make_hand_agent(cfg.sources, cfg.horizon, cfg.top))]
    else:
        params, heads = evaluate.load(cfg.value)
        runs = [(f"mix={m}", make_value_agent(params, heads, cfg.sources, cfg.horizon, cfg.top, float(m), wait=not cfg.no_wait))
                for m in cfg.mix.split(",")]
    for opp in cfg.vs.split(","):
        fb = opponent(opp)
        for name, agent in runs:
            t0 = time.time()
            sc, ticks = play_chunked(agent, fb, bank, jax.random.PRNGKey(cfg.seed), cfg.chunk, cfg.max_steps)
            sc = np.asarray(sc)
            print(json.dumps(dict(agent=name, value=cfg.value, vs=opp, top=cfg.top, horizon=cfg.horizon, games=len(sc),
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
    g.add_argument("--kinds", default="1,0,0", help="odds a side plays: hand-scored planner, value alone, value mixed with hand")
    g.add_argument("--bank-from", default=None, help="directory of an earlier run whose maps.npz to reuse")
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
    m.add_argument("--chunk", type=int, default=250)
    m.add_argument("--max-steps", type=int, default=2001)
    m.add_argument("--seed", type=int, default=777)
    cfg = a.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    {"gen": cmd_gen, "train": cmd_train, "match": cmd_match}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
