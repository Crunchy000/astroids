"""A smaller, faster value network that copies a bigger one (distillation), so the entry can afford more launches.

The entry's time goes mostly on the value network, so a network that is twice as fast at the same rankings is worth
more launches and a fuller look-ahead on big maps. The student learns the teacher's number for the positions the
entry actually asks about: the planner's best launches (by hand score) in games the teacher plays, and the positions
after the opponent's answers. Its targets have none of a game result's noise, so far fewer games are needed.

    python -m pwjax.distill gen   --teacher r6.npz --out dd --batches 4
    python -m pwjax.distill train --teacher r6.npz --data dd --out small --d-model 64 --layers 3 --d-ff 128
    python -m pwjax.lvalue match small/best.npz --top 64 --reply 8:4 --vs 'value:r6.npz:0.5:64;extras=spare;reply=8:4'
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
from .lvalue import EPS, make_value_agent, net_values, options, totals
from .ppo import save_params, tree_where


def make_recorder(params, heads, cfg):
    """record(state, me, h, key) -> `cfg.per` positions the entry would score this tick: owners, ships [per, N],
    the teacher's values [per] and which are usable. Half come from the planner's best `cfg.top` launches and waiting,
    half from the opponent's answers to the best `ours` of those (the look-ahead's positions)."""
    plan = jplanner.make_planner(cfg.sources, cfg.horizon, cfg.top, jplanner.FRACTIONS, ("spare",))
    plan_r = jplanner.make_planner(cfg.sources, cfg.horizon, 4, jplanner.FRACTIONS, ("spare",))
    half = cfg.per // 2

    def record(s, me, h, key):
        out, owners, ships, hand, ok, v = options(plan, s, me, h, cfg.top, params, heads)
        k1, k2 = jax.random.split(key)
        pick = jax.random.choice(k1, cfg.top + 1, (half,), replace=False, p=ok / jnp.maximum(ok.sum(), 1))
        o1, s1, ok1 = owners[pick], ships[pick], ok[pick]
        plain = jnp.where(ok, totals(s, me, owners, hand, v, 0.5), -jnp.inf)
        # the answers to our 8 best options, as the look-ahead sees them
        first, cand = jax.lax.top_k(plain, 8)
        launch = (cand < cfg.top) & jnp.isfinite(first)
        j = jnp.minimum(cand, cfg.top - 1)
        acts = (jnp.where(launch, out["cand_src"][j], -1).astype(jnp.int32), jnp.where(launch, out["cand_dst"][j], -1).astype(jnp.int32),
                jnp.where(launch, out["cand_ships"][j], 0.0).astype(s.ships.dtype))
        after = jax.vmap(lambda a: env._launch(s, me, a))(acts)
        r = jax.vmap(lambda st: plan_r(st, 3 - me, h))(after)
        ro, rs = r["cand_owner"].reshape(32, -1), r["cand_ships_final"].reshape(32, -1)
        r_ok = (jnp.isfinite(r["cand_score"]) & jnp.isfinite(first)[:, None]).reshape(32)
        pick2 = jax.random.choice(k2, 32, (cfg.per - half,), replace=False, p=r_ok / jnp.maximum(r_ok.sum(), 1))
        o = jnp.concatenate([o1, ro[pick2]])
        sh = jnp.concatenate([s1, rs[pick2]])
        use = jnp.concatenate([ok1 & ok.any(), r_ok[pick2] & r_ok.any()])
        return o, sh, net_values(params, heads, s, me, o, sh, h), use
    return record


def cmd_gen(cfg):
    os.makedirs(cfg.out, exist_ok=True)
    params, heads = evaluate.load(cfg.teacher)
    params = {k: v for k, v in params.items() if not k.startswith("aux.")}
    bank_np = maps.make_bank(cfg.maps, seed=cfg.seed, param_fn=maps.cog2026_params)
    np.savez(os.path.join(cfg.out, "maps.npz"), **bank_np)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    play = make_value_agent(params, heads, cfg.sources, cfg.horizon, 16, 0.5, extras=("spare",))
    record = make_recorder(params, heads, cfg)
    n = cfg.envs
    v_play, v_rec = jax.vmap(play), jax.vmap(record)

    @jax.jit
    def chunk(c, hs, eps, key):
        def one(c, k):
            s, done = c
            ka, kb, kr = jax.random.split(k, 3)
            a1 = v_play(s, jnp.ones(n, jnp.int32), jax.random.split(ka, n))
            a2 = v_play(s, jnp.full(n, 2, jnp.int32), jax.random.split(kb, n))
            # now and then a side waits instead, so the games wander off the teacher's own path
            r1, r2 = jax.random.uniform(kr, (2, n)) < eps
            noop = (jnp.full(n, -1, jnp.int32), jnp.full(n, -1, jnp.int32), jnp.zeros(n, s.ships.dtype))
            a1 = tuple(jnp.where(r1, x, y) for x, y in zip(noop, a1))
            a2 = tuple(jnp.where(r2, x, y) for x, y in zip(noop, a2))
            s2 = tree_where(done, s, jax.vmap(env.step)(s, a1, a2))
            return (s2, done | jax.vmap(env.is_terminal)(s2)), None

        def block(c, k):
            ks = jax.random.split(k, cfg.stride + 1)
            c, _ = jax.lax.scan(one, c, ks[:-1])
            s, done = c
            me = jnp.where(jax.random.bernoulli(ks[-1], 0.5, (n,)), 1, 2).astype(jnp.int32)
            h = jnp.maximum(jnp.minimum(hs, s.max_ticks - s.tick), 1)
            o, sh, v, use = v_rec(s, me, h, jax.random.split(ks[-1], n))
            return c, dict(owner=o.astype(jnp.int8), ships=sh, v=v, use=use & ~done[:, None], tick=s.tick + h, me=me)

        return jax.lax.scan(block, c, jax.random.split(key, cfg.chunk // cfg.stride))

    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    for name, nb in (("val", 1), ("train", cfg.batches)):
        parts, t0 = [], time.time()
        for b in range(nb):
            ids = rng.integers(0, cfg.maps, size=n)
            s = jax.vmap(lambda i: env.from_bank(bank, i))(jnp.asarray(ids))
            c = (s, jax.vmap(env.is_terminal)(s))
            hs = jnp.full(n, cfg.horizon, jnp.int32)
            eps = float(rng.choice(EPS)) + 0.02
            recs = []
            for _ in range(-(-2001 // cfg.chunk)):
                if bool(c[1].all()):
                    break
                key, k = jax.random.split(key)
                c, rec = chunk(c, hs, eps, k)
                recs.append(jax.tree.map(np.asarray, rec))
            r = {k: np.concatenate([x[k] for x in recs]) for k in recs[0]}            # [snaps, n, ...]
            per = r["use"].shape[-1]
            use = r["use"].reshape(-1)
            rep = lambda a: np.broadcast_to(a[..., None], a.shape + (per,)).reshape(-1)  # noqa: E731
            part = dict(owner=r["owner"].reshape(-1, r["owner"].shape[-1])[use], ships=r["ships"].reshape(-1, r["ships"].shape[-1])[use].astype(np.float32),
                        v=r["v"].reshape(-1)[use], tick=rep(r["tick"])[use], me=rep(r["me"])[use],
                        map=rep(np.broadcast_to(ids[None], r["me"].shape))[use])
            parts.append(part)
            print(json.dumps(dict(set=name, batch=b + 1, positions=int(use.sum()), seconds=round(time.time() - t0, 1))), flush=True)
        np.savez(os.path.join(cfg.out, f"{name}.npz"), **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def cmd_train(cfg):
    bank = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "maps.npz")).items()}
    tr = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "train.npz")).items()}
    va = {k: jnp.asarray(v) for k, v in np.load(os.path.join(cfg.data, "val.npz")).items()}
    params = policy.init(jax.random.PRNGKey(cfg.seed), cfg.d_model, cfg.layers, cfg.heads, 32, cfg.d_ff)

    def values(params, b):
        def one(i, owner, ships, tick, me):
            s = env.from_bank(bank, i)
            s = s._replace(owner=owner.astype(jnp.int32), ships=ships, tr_on=jnp.zeros_like(s.tr_on), tick=tick)
            return policy.apply(params, cfg.heads, s, me)[2]
        return jax.vmap(one)(b["map"], b["owner"], b["ships"], b["tick"], b["me"])

    n_tr = len(tr["v"])
    print(f"train positions {n_tr}, held-out {len(va['v'])}", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0, eps=1e-5))
    opt_state = opt.init(params)

    @jax.jit
    def train_chunk(params, opt_state, key, lr):
        def step(c, k):
            params, opt_state = c
            ids = jax.random.randint(k, (cfg.minibatch,), 0, n_tr)
            b = jax.tree.map(lambda x: x[ids], tr)
            loss, g = jax.value_and_grad(lambda p: jnp.mean((values(p, b) - b["v"]) ** 2))(params)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, jax.tree.map(lambda u: lr * u, upd)), opt_state), loss
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jax.random.split(key, cfg.chunk_steps))
        return params, opt_state, losses.mean()

    eval_v = jax.jit(values)

    def held_out(params):
        pred = np.concatenate([np.asarray(eval_v(params, {k: v[i:i + 8192] for k, v in va.items()})) for i in range(0, len(va["v"]), 8192)])
        t = np.asarray(va["v"])
        return float(np.mean((pred - t) ** 2)), float(np.corrcoef(pred, t)[0, 1])

    os.makedirs(cfg.out, exist_ok=True)
    key, best, t0 = jax.random.PRNGKey(cfg.seed + 1), float("inf"), time.time()
    n_chunks = cfg.steps // cfg.chunk_steps
    for c in range(n_chunks):
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * c / max(n_chunks - 1, 1)
        key, k = jax.random.split(key)
        params, opt_state, loss = train_chunk(params, opt_state, k, lr)
        if (c + 1) % cfg.report_every == 0 or c + 1 == n_chunks:
            mse, corr = held_out(params)
            print(json.dumps(dict(step=(c + 1) * cfg.chunk_steps, loss=round(float(loss), 5), held_out_mse=round(mse, 5),
                                  corr=round(corr, 4), seconds=round(time.time() - t0))), flush=True)
            save_params(os.path.join(cfg.out, "latest.npz"), params, argparse.Namespace(heads=cfg.heads))
            if mse < best:
                best = mse
                save_params(os.path.join(cfg.out, "best.npz"), params, argparse.Namespace(heads=cfg.heads))


def main(argv=None):
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--teacher", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--envs", type=int, default=1024)
    g.add_argument("--batches", type=int, default=4)
    g.add_argument("--maps", type=int, default=8192)
    g.add_argument("--per", type=int, default=8, help="positions recorded per game and snapshot")
    g.add_argument("--stride", type=int, default=16)
    g.add_argument("--chunk", type=int, default=240)
    g.add_argument("--top", type=int, default=64)
    g.add_argument("--sources", type=int, default=12)
    g.add_argument("--horizon", type=int, default=90)
    g.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--d-ff", type=int, default=128)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--chunk-steps", type=int, default=500)
    t.add_argument("--report-every", type=int, default=2)
    t.add_argument("--minibatch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lr-end", type=float, default=5e-5)
    t.add_argument("--seed", type=int, default=0)
    cfg = a.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    {"gen": cmd_gen, "train": cmd_train}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
