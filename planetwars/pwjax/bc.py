"""Warm start: teach the network to copy the official greedy heuristic agent (behaviour cloning).

Every greedy move is one of the network's actions (it sends half the source's ships), so the copy can
be exact. Games are played by greedy with some random moves mixed in, so the network also sees states
greedy itself would not reach; both sides of every game are labelled with greedy's move.

    python -m pwjax.bc --out runs/bc --iters 300
    python -m pwjax.ppo --out runs/pw1 --init runs/bc/latest.npz
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import optax

from . import baselines, env, maps, policy
from .ppo import save_params

HALF = policy.pwnet.FRACTIONS.index(0.5)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--envs", type=int, default=512)
    a.add_argument("--steps", type=int, default=64, help="game ticks per iteration")
    a.add_argument("--iters", type=int, default=300)
    a.add_argument("--maps", type=int, default=20000)
    a.add_argument("--eps", type=float, default=0.25, help="chance of a careful-random move instead of greedy")
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--launch-weight", type=float, default=5.0)
    a.add_argument("--minibatch", type=int, default=4096)
    a.add_argument("--epochs", type=int, default=2)
    a.add_argument("--d-model", type=int, default=128)
    a.add_argument("--layers", type=int, default=3)
    a.add_argument("--heads", type=int, default=4)
    a.add_argument("--d-pair", type=int, default=32)
    a.add_argument("--d-ff", type=int, default=256)
    a.add_argument("--seed", type=int, default=0)
    cfg = a.parse_args(argv)
    os.makedirs(cfg.out, exist_ok=True)
    print("devices:", jax.devices(), flush=True)
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.maps, seed=cfg.seed + 1).items()}
    n_maps = bank["valid"].shape[0]
    E, heads = cfg.envs, cfg.heads
    key = jax.random.PRNGKey(cfg.seed)
    key, k = jax.random.split(key)
    params = policy.init(k, cfg.d_model, cfg.layers, cfg.heads, cfg.d_pair, cfg.d_ff)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    opt_state = opt.init(params)
    v_reset = jax.vmap(lambda i: env.from_bank(bank, i))
    n = bank["valid"].shape[1]

    def label(s, me):
        src, dst, _ = baselines.greedy(s, me)
        return jnp.where(src < 0, n * n, src * n + dst), jnp.int32(HALF)

    def act(s, me, k):
        k1, k2 = jax.random.split(k)
        g = baselines.greedy(s, me)
        r = baselines.careful_random(s, me, k1)
        use_r = jax.random.uniform(k2) < cfg.eps
        return tuple(jnp.where(use_r, y, x) for x, y in zip(g, r))

    @jax.jit
    def rollout(s, key):
        def one(s, k):
            k1, k2, k3 = jax.random.split(k, 3)
            a1 = jax.vmap(act)(s, jnp.full(E, 1), jax.random.split(k1, E))
            a2 = jax.vmap(act)(s, jnp.full(E, 2), jax.random.split(k2, E))
            l1 = jax.vmap(label)(s, jnp.full(E, 1))
            l2 = jax.vmap(label)(s, jnp.full(E, 2))
            s2 = jax.vmap(env.step)(s, a1, a2)
            fresh = v_reset(jax.random.randint(k3, (E,), 0, n_maps))
            done = jax.vmap(env.is_terminal)(s2)
            s2 = jax.tree.map(lambda x, y: jnp.where(done.reshape(done.shape + (1,) * (x.ndim - 1)), y, x), s2, fresh)
            return s2, (s, l1, l2)
        return jax.lax.scan(one, s, jax.random.split(key, cfg.steps))

    v_apply = jax.vmap(lambda p, s, me: policy.apply(p, heads, s, me)[:2], in_axes=(None, 0, 0))

    def loss_fn(p, s, me, idx, aidx):
        flat, amount_logits = v_apply(p, s, me)
        lp = jax.nn.log_softmax(flat)
        ce = -jnp.take_along_axis(lp, idx[:, None], axis=1)[:, 0]
        al = amount_logits.reshape(amount_logits.shape[0], n * n, -1)
        al = jax.nn.log_softmax(jnp.take_along_axis(al, jnp.clip(idx, 0, n * n - 1)[:, None, None], axis=1)[:, 0])
        noop = idx >= n * n
        ce_a = jnp.where(noop, 0.0, -jnp.take_along_axis(al, aidx[:, None], axis=1)[:, 0])
        hit = jnp.argmax(flat, axis=1) == idx
        launches = jnp.maximum((~noop).sum(), 1)
        # greedy does nothing on most ticks: weight its launches up so they aren't drowned out
        w = jnp.where(noop, 1.0, cfg.launch_weight)
        return ((ce + ce_a) * w).sum() / w.sum(), dict(
            ce=ce.mean(), ce_amount=ce_a.mean(), acc=hit.mean(), noop=noop.mean(),
            acc_launch=jnp.where(noop, False, hit).sum() / launches,
            acc_noop=jnp.where(noop, hit, False).sum() / jnp.maximum(noop.sum(), 1))

    @jax.jit
    def train(p, opt_state, traj, key):
        s, (i1, a1), (i2, a2) = traj
        s = jax.tree.map(lambda x: jnp.concatenate([x, x]).reshape((-1,) + x.shape[2:]), s)
        me = jnp.concatenate([jnp.ones((cfg.steps, E), jnp.int32), jnp.full((cfg.steps, E), 2, jnp.int32)]).reshape(-1)
        idx = jnp.concatenate([i1, i2]).reshape(-1)
        aidx = jnp.concatenate([a1, a2]).reshape(-1)
        total = idx.shape[0]
        n_mb = max(total // cfg.minibatch, 1)

        def epoch(c, k):
            perm = jax.random.permutation(k, total)[: n_mb * (total // n_mb)].reshape(n_mb, -1)

            def mb(c, ix):
                p, o = c
                (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(
                    p, jax.tree.map(lambda x: x[ix], s), me[ix], idx[ix], aidx[ix])
                u, o = opt.update(g, o, p)
                return (optax.apply_updates(p, u), o), m

            return jax.lax.scan(mb, c, perm)

        (p, opt_state), m = jax.lax.scan(epoch, (p, opt_state), jax.random.split(key, cfg.epochs))
        return p, opt_state, jax.tree.map(jnp.mean, m)

    key, k = jax.random.split(key)
    s = v_reset(jax.random.randint(k, (E,), 0, n_maps))
    t0 = time.time()
    for it in range(cfg.iters):
        key, k1, k2 = jax.random.split(key, 3)
        s, traj = rollout(s, k1)
        params, opt_state, m = train(params, opt_state, traj, k2)
        if (it + 1) % 10 == 0 or it == 0:
            row = dict(iter=it + 1, sec=round(time.time() - t0, 1), **{k: round(float(v), 4) for k, v in m.items()})
            print(json.dumps(row), flush=True)
            with open(os.path.join(cfg.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
        if (it + 1) % 50 == 0 or it + 1 == cfg.iters:
            save_params(os.path.join(cfg.out, "latest.npz"), params, cfg)


if __name__ == "__main__":
    main()
