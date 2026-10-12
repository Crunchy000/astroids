"""Score a trained policy against the official sample agents (or another policy) on fresh league maps.

    python -m pwjax.evaluate runs/pw1/latest.npz --games 512
    python -m pwjax.evaluate runs/pw1/latest.npz --vs runs/old/latest.npz

Each map is played twice, once from each side.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from . import baselines, env, maps, policy


def load(path):
    z = np.load(path)
    heads = int(z["cfg.heads"]) if "cfg.heads" in z.files else 4
    return {k: jnp.asarray(z[k]) for k in z.files if not k.startswith("cfg.")}, heads


def agent_fn(kind, params=None, heads=4, sample=False):
    """A function (state, player, key) -> action for vmapping."""
    if kind == "greedy":
        return baselines.greedy
    if kind == "random":
        return baselines.careful_random
    if kind == "nothing":
        return baselines.do_nothing
    if kind == "sniper":
        return baselines.sniper
    return lambda s, me, k: policy.sample(params, heads, s, me, k, greedy=not sample)[0]


def play(fa, fb, bank, key, max_steps=2001):
    """Agent a plays side 1 on every map and side 2 on a mirrored copy. Returns a's outcome per game."""
    n = bank["valid"].shape[0]
    s0 = jax.vmap(lambda i: env.from_bank(bank, i))(jnp.arange(n))
    s0 = jax.tree.map(lambda x: jnp.concatenate([x, x]), s0)
    side_a = jnp.concatenate([jnp.ones(n, jnp.int32), jnp.full(n, 2, jnp.int32)])

    def body(c, k):
        s, done = c
        ka, kb = jax.random.split(k)
        aa = jax.vmap(fa)(s, side_a, jax.random.split(ka, 2 * n))
        ab = jax.vmap(fb)(s, 3 - side_a, jax.random.split(kb, 2 * n))
        a1 = tuple(jnp.where(side_a == 1, x, y) for x, y in zip(aa, ab))
        a2 = tuple(jnp.where(side_a == 2, x, y) for x, y in zip(aa, ab))
        s2 = jax.vmap(env.step)(s, a1, a2)
        s2 = jax.tree.map(lambda new, old: jnp.where(done.reshape(done.shape + (1,) * (new.ndim - 1)), old, new),
                          s2, s)
        return (s2, done | jax.vmap(env.is_terminal)(s2)), None

    (s, done), _ = jax.lax.scan(body, (s0, jax.vmap(env.is_terminal)(s0)), jax.random.split(key, max_steps))
    lead = jax.vmap(env.leader)(s)
    return jnp.where(lead == side_a, 1.0, jnp.where(lead == 0, 0.5, 0.0)), s.tick


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("model")
    a.add_argument("--vs", default="greedy,random", help="greedy, random, sniper, nothing, or .npz paths (comma separated)")
    a.add_argument("--games", type=int, default=256, help="maps (each played from both sides)")
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--seed", type=int, default=12345)
    a.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    a.add_argument("--default-rules", action="store_true", help="GameParams defaults with 20 planets (the submission check)")
    args = a.parse_args(argv)
    params, heads = load(args.model)
    fa = agent_fn("policy", params, heads, args.sample)
    if args.default_rules:
        pfn = lambda rng: maps.Params(num_planets=20, max_ticks=1000)  # noqa: E731
    else:
        pfn = maps.cog2026_params
    results = {}
    for opp in args.vs.split(","):
        if opp.endswith(".npz"):
            p2, h2 = load(opp)
            fb = agent_fn("policy", p2, h2, args.sample)
        else:
            fb = agent_fn(opp)
        run = jax.jit(lambda bank, key: play(fa, fb, bank, key))
        scores, ticks = [], []
        for b in range(0, args.games, args.batch):
            bank = {k: jnp.asarray(v) for k, v in
                    maps.make_bank(min(args.batch, args.games - b), seed=args.seed + b, param_fn=pfn).items()}
            sc, tk = run(bank, jax.random.PRNGKey(args.seed + b))
            scores.append(np.asarray(sc))
            ticks.append(np.asarray(tk))
        sc = np.concatenate(scores)
        results[opp] = dict(score=round(float(sc.mean()), 4), wins=int((sc == 1).sum()), draws=int((sc == 0.5).sum()),
                            losses=int((sc == 0).sum()), mean_ticks=round(float(np.concatenate(ticks).mean()), 1))
        print(opp, json.dumps(results[opp]), flush=True)
    return results


if __name__ == "__main__":
    main()
