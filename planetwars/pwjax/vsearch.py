"""A one-move look-ahead agent in JAX: the policy proposes moves, a value network picks one.

Each turn the candidates are the policy's own choice, doing nothing, the best-aimed launch from each of
the `top` planets the policy likes launching from, and the greedy bot's move. Each candidate is played on
a copy of the game with the exact rules, both sides then follow `rollout` (the greedy bot or nothing) for
`horizon` ticks, and the value network scores the position that results. The best score wins; the
policy's own choice gets a small bonus `prior`.

    python -m pwjax.vsearch policy.npz --value v.npz --vs policy.npz --games 256

Used to ask whether a value network is any good at ranking moves: it plays the same policy without
search, so the difference is the value network's doing.
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import baselines, env, evaluate, maps, policy
from .ppo import tree_where


def candidates(pol, heads, s, me, top):
    """The moves worth comparing: the policy's choice, doing nothing, the best-aimed launch from each of the
    `top` planets the policy likes launching from, and the greedy bot's move. Returns (src, dst, ships), each [top + 3]."""
    flat, amount_logits, _, aux = policy.apply(pol, heads, s, me)
    n = amount_logits.shape[0]
    pair = flat[: n * n].reshape(n, n)
    none = (jnp.int32(-1), jnp.int32(-1), jnp.zeros((), s.ships.dtype))

    def move(i, j):
        legal = pair[i, j] > -1e8
        ships = aux["amounts"][i, j, jnp.argmax(amount_logits[i, j])]
        return (jnp.where(legal, i, -1).astype(jnp.int32), jnp.where(legal, j, -1).astype(jnp.int32),
                jnp.where(legal, ships, 0.0).astype(s.ships.dtype))

    best = jnp.argmax(flat)
    own = move(jnp.clip(best // n, 0, n - 1), best % n)
    own = tuple(jnp.where(best >= n * n, x, y) for x, y in zip(none, own))
    _, srcs = jax.lax.top_k(pair.max(axis=1), top)
    moves = jax.vmap(move)(srcs, jnp.argmax(pair[srcs], axis=1))
    g = baselines.greedy(s, me)
    return tuple(jnp.concatenate([jnp.stack([o, n_, gg]), m]) for o, n_, gg, m in zip(own, none, g, moves))


def outcome(s, me):
    lead = env.leader(s)
    return jnp.where(lead == me, 1.0, jnp.where(lead == 0, 0.0, -1.0))


def after(s, me, action, horizon, rollout="greedy", idle_opponent=False):
    """Play `action` for `me` (the other side plays the greedy bot's move, or nothing if `idle_opponent`), then
    `horizon - 1` more ticks of `rollout` (both sides greedy, or nobody does anything). Returns the position
    and whether the game is over."""
    o0 = (jnp.int32(-1), jnp.int32(-1), jnp.zeros((), s.ships.dtype)) if idle_opponent else baselines.greedy(s, 3 - me)
    a1 = tuple(jnp.where(me == 1, x, y) for x, y in zip(action, o0))
    a2 = tuple(jnp.where(me == 2, x, y) for x, y in zip(action, o0))
    s1 = env.step(s, a1, a2)
    nothing = (jnp.int32(-1), jnp.int32(-1), jnp.zeros((), s.ships.dtype))

    def body(c, _):
        st, done = c
        if rollout == "greedy":
            st2 = env.step(st, baselines.greedy(st, 1), baselines.greedy(st, 2))
        else:
            st2 = env.step(st, nothing, nothing)
        st2 = tree_where(done, st, st2)
        return (st2, done | env.is_terminal(st2)), None

    return jax.lax.scan(body, (s1, env.is_terminal(s1)), None, length=horizon - 1)[0]


def titans_score(s, me, width=640.0):
    """TeamTitansAgentV3's position score (planets only, fleets in flight do not count): ship difference,
    ten times the growth difference, half the difference in ships a further 20 ticks of growth would give,
    five per planet of lead, a penalty for planets far from all friendly ones, and a huge win or loss."""
    mine, opp = s.valid & (s.owner == me), s.valid & (s.owner == 3 - me)
    f = lambda x, m: jnp.sum(jnp.where(m, x, 0.0))  # noqa: E731
    score = f(s.ships, mine) - f(s.ships, opp)
    score += 10.0 * (f(s.growth, mine) - f(s.growth, opp))
    score += 0.5 * (f(s.ships + 20.0 * s.growth, mine) - f(s.ships + 20.0 * s.growth, opp))
    d = jnp.sqrt(jnp.sum((s.pos[:, None, :] - s.pos[None, :, :]) ** 2, axis=-1))
    n_mine, n_opp = jnp.sum(mine), jnp.sum(opp)
    friend = mine[None, :] & ~jnp.eye(s.ships.shape[0], dtype=bool)
    nearest = jnp.min(jnp.where(friend, d, 1e9), axis=1)
    score -= jnp.where(n_mine > 1, f(2.0 * s.growth, mine & (nearest > width / 4.0)), 0.0)
    score += 5.0 * (n_mine - n_opp)
    return score + jnp.where((n_opp == 0) & (n_mine > 0), 10000.0, 0.0) - jnp.where((n_mine == 0) & (n_opp > 0), 10000.0, 0.0)


def make_search(pol, heads, value=None, top=6, horizon=40, prior=0.02, rollout="greedy", v_heads=None):
    """Return act(state, me, key) -> (src, dst, ships), a search agent over `pol` scored by `value`."""
    v_params = pol if value is None else value
    v_heads = heads if v_heads is None else v_heads

    def act(s, me, key):
        del key
        src, dst, ships = candidates(pol, heads, s, me, top)

        def score(a, b, c):
            sh, done = after(s, me, (a, b, c), horizon, rollout)
            return jnp.where(done, outcome(sh, me), policy.apply(v_params, v_heads, sh, me)[2])

        vals = jax.vmap(score)(src, dst, ships) + prior * (jnp.arange(src.shape[0]) == 0)
        k = jnp.argmax(vals)
        return src[k], dst[k], ships[k]

    return act


def play_chunked(fa, fb, bank, key, chunk=250, max_steps=2001):
    """evaluate.play, in jitted chunks so that it stops once every game is over. Returns a's score per game."""
    n = bank["valid"].shape[0]
    s0 = jax.vmap(lambda i: env.from_bank(bank, i))(jnp.arange(n))
    s0 = jax.tree.map(lambda x: jnp.concatenate([x, x]), s0)
    side_a = jnp.concatenate([jnp.ones(n, jnp.int32), jnp.full(n, 2, jnp.int32)])

    @jax.jit
    def run(s, done, key):
        def body(c, k):
            s, done = c
            ka, kb = jax.random.split(k)
            aa = jax.vmap(fa)(s, side_a, jax.random.split(ka, 2 * n))
            ab = jax.vmap(fb)(s, 3 - side_a, jax.random.split(kb, 2 * n))
            a1 = tuple(jnp.where(side_a == 1, x, y) for x, y in zip(aa, ab))
            a2 = tuple(jnp.where(side_a == 2, x, y) for x, y in zip(aa, ab))
            s2 = tree_where(done, s, jax.vmap(env.step)(s, a1, a2))
            return (s2, done | jax.vmap(env.is_terminal)(s2)), None

        return jax.lax.scan(body, (s, done), jax.random.split(key, chunk))[0]

    s, done = s0, jax.vmap(env.is_terminal)(s0)
    for i in range(-(-max_steps // chunk)):
        if bool(done.all()):
            break
        key, k = jax.random.split(key)
        s, done = run(s, done, k)
    lead = jax.vmap(env.leader)(s)
    return jnp.where(lead == side_a, 1.0, jnp.where(lead == 0, 0.5, 0.0)), s.tick


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("policy")
    a.add_argument("--value", default=None, help="value network .npz (default: the policy's own value output)")
    a.add_argument("--vs", default="greedy", help="bots or .npz paths (comma separated), played without search")
    a.add_argument("--games", type=int, default=128, help="maps (each played from both sides)")
    a.add_argument("--top", type=int, default=6)
    a.add_argument("--horizon", type=int, default=40)
    a.add_argument("--prior", type=float, default=0.02)
    a.add_argument("--rollout", default="greedy", choices=["greedy", "noop"])
    a.add_argument("--seed", type=int, default=12345)
    a.add_argument("--chunk", type=int, default=250)
    a.add_argument("--max-steps", type=int, default=2001, help="ticks to play (the full game is 2001)")
    a.add_argument("--plain", action="store_true", help="no search (the baseline against the same opponents)")
    args = a.parse_args(argv)
    pol, heads = evaluate.load(args.policy)
    val, v_heads = evaluate.load(args.value) if args.value else (None, None)
    fa = evaluate.agent_fn("policy", pol, heads, False) if args.plain else \
        make_search(pol, heads, val, args.top, args.horizon, args.prior, args.rollout, v_heads)
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(args.games, seed=args.seed, param_fn=maps.cog2026_params).items()}
    for opp in args.vs.split(","):
        if opp.endswith(".npz"):
            p2, h2 = evaluate.load(opp)
            fb = evaluate.agent_fn("policy", p2, h2, False)
        else:
            fb = evaluate.agent_fn(opp)
        t0 = time.time()
        sc, ticks = play_chunked(fa, fb, bank, jax.random.PRNGKey(args.seed), args.chunk, args.max_steps)
        sc = np.asarray(sc)
        row = dict(opp=opp, score=round(float(sc.mean()), 4), wins=int((sc == 1).sum()), draws=int((sc == 0.5).sum()),
                   losses=int((sc == 0).sum()), mean_ticks=round(float(np.asarray(ticks).mean()), 1),
                   seconds=round(time.time() - t0, 1), search=not args.plain, top=args.top, horizon=args.horizon,
                   rollout=args.rollout, prior=args.prior, value=args.value or "policy head")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
