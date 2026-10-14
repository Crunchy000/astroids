"""How much does the choice of move matter, and can a value network see it?

Positions come from games between a mix of players. For each position the candidate moves of
`vsearch.candidates` are tried: each is played, then both sides keep following the policy (sampled, as in
training) to the end of the game, `rollouts` times. The mean result is that move's true worth, Q.

Reported:
  * how far apart the candidates' Q values are, and how much better than the policy's own choice the best
    of them is, judged on rollouts that were not used to pick it (so the estimate is not inflated);
  * for each value network, how well its look-ahead scores (the position `horizon` greedy ticks after the
    move, valued by the network) rank the candidates against Q, and the Q gained by playing its pick.

    python -m pwjax.qprobe policy.npz --values v1=v.npz --players policy.npz,other.npz
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import env, evaluate, maps, policy, vsearch, vtrain
from .ppo import tree_where


def make_mc(pol, heads, chunk):
    """Finish games from given positions with both sides sampling the policy. Returns (state, done) after `chunk` ticks."""

    @jax.jit
    def run(s, me, done, key):
        def one(s, me, done, key):
            def body(c, k):
                s, done = c
                ka, kb = jax.random.split(k)
                a_me = policy.sample(pol, heads, s, me, ka)[0]
                a_op = policy.sample(pol, heads, s, 3 - me, kb)[0]
                a1 = tuple(jnp.where(me == 1, x, y) for x, y in zip(a_me, a_op))
                a2 = tuple(jnp.where(me == 2, x, y) for x, y in zip(a_me, a_op))
                s2 = tree_where(done, s, env.step(s, a1, a2))
                return (s2, done | env.is_terminal(s2)), None

            return jax.lax.scan(body, (s, done), jax.random.split(key, chunk))[0]

        return jax.vmap(one)(s, me, done, key)

    return run


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("policy")
    a.add_argument("--values", default="", help="name=path.npz,... value networks to test (the policy's own head is always tested)")
    a.add_argument("--players", required=True, help="policies that play the games the positions come from")
    a.add_argument("--states", type=int, default=384)
    a.add_argument("--rollouts", type=int, default=12)
    a.add_argument("--top", type=int, default=6)
    a.add_argument("--horizons", default="40", help="look-ahead lengths to score with (comma separated)")
    a.add_argument("--rollout", default="greedy", choices=["greedy", "noop"])
    a.add_argument("--reuse", default=None, help="a --dump file from an earlier run on the same positions: skip the rollouts")
    a.add_argument("--prior", type=float, default=0.02)
    a.add_argument("--chunk", type=int, default=100)
    a.add_argument("--sub", type=int, default=12288, help="rollouts run at once")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--gen-envs", type=int, default=1024, help="games the positions are drawn from")
    a.add_argument("--gen-groups", type=int, default=16)
    a.add_argument("--gen-ticks", type=int, default=2001)
    a.add_argument("--max-chunks", type=int, default=100000, help="stop rollouts after this many chunks (testing)")
    a.add_argument("--dump", default=None, help="save the per-candidate numbers here (.npz)")
    args = a.parse_args(argv)
    pol, heads = evaluate.load(args.policy)
    nets = {"policy head": (pol, heads)}
    for item in [x for x in args.values.split(",") if x]:
        name, path = item.split("=")
        nets[name] = evaluate.load(path)

    # positions from games between the listed players (and the bots), or the ones saved by an earlier run
    if args.reuse:
        old = np.load(args.reuse)
        s = env.State(**{f: jnp.asarray(old["s_" + f]) for f in env.State._fields})
        me = jnp.asarray(old["me"])
        args.states = me.shape[0]
    else:
        players = [evaluate.load(p) for p in args.players.split(",")]
        g_cfg = vtrain.parse(["--out", "/tmp/qprobe", "--policies", args.players, "--envs", str(args.gen_envs),
                              "--groups", str(args.gen_groups), "--ticks", str(args.gen_ticks), "--seed", str(args.seed)])
        rng = np.random.default_rng(args.seed)
        bank = {k: jnp.asarray(v) for k, v in maps.make_bank(min(4096, 4 * args.gen_envs), seed=args.seed).items()}
        gen = vtrain.make_gen(g_cfg, bank, heads)
        pols = [p for p, _ in players]
        a_, b_ = vtrain.pick_players(g_cfg, len(pols), rng)

        def side(players_):
            kind, idx, temp, eps = players_
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[pols[i] for i in idx])
            return stacked, jnp.asarray(np.repeat(kind, g_cfg.envs // g_cfg.groups)), jnp.asarray(temp), jnp.asarray(eps)

        snaps, alive, z1 = gen(side(a_), side(b_), jax.random.PRNGKey(args.seed))
        flat, alive_f, _ = vtrain.flatten(snaps, alive, z1)
        idx = np.nonzero(np.asarray(alive_f))[0]
        idx = rng.choice(idx, size=args.states, replace=False)
        s = jax.tree.map(lambda x: x[idx], flat)
        me = jnp.asarray(rng.integers(1, 3, size=args.states), jnp.int32)
    S = args.states
    N = args.rollouts
    K = args.top + 3
    print(f"{S} positions, {K} candidate moves, {N} rollouts each; mean tick {float(np.asarray(s.tick).mean()):.0f}", flush=True)

    src, dst, ships = jax.jit(jax.vmap(lambda st, m: vsearch.candidates(pol, heads, st, m, args.top)))(s, me)

    # look-ahead scores of every candidate under every value network, for each look-ahead length
    scores = {}
    for horizon in [int(x) for x in args.horizons.split(",")]:
        after = jax.jit(jax.vmap(jax.vmap(lambda st, m, x, y, z, hz=horizon: vsearch.after(st, m, (x, y, z), hz, args.rollout),
                                          in_axes=(None, None, 0, 0, 0)), in_axes=(0, 0, 0, 0, 0)))
        sh, done = after(s, me, src, dst, ships)
        for name, (params, h) in nets.items():
            v = jax.jit(jax.vmap(jax.vmap(lambda st, m, d, params=params, h=h: jnp.where(
                d, vsearch.outcome(st, m), policy.apply(params, h, st, m)[2]), in_axes=(0, None, 0)),
                in_axes=(0, 0, 0)))(sh, me, done)
            scores[f"{name} @{horizon}"] = np.asarray(v)

    # true worth: finish the game from each candidate, N times, with the same random numbers for every candidate
    t0 = time.time()
    mc = make_mc(pol, heads, args.chunk)
    B = S * K * N
    rep = lambda x: jnp.repeat(x, K * N, axis=0)  # noqa: E731
    s_rep = jax.tree.map(rep, s)
    me_rep = rep(me)
    act = (src.reshape(S, K, 1).repeat(N, 2).reshape(B), dst.reshape(S, K, 1).repeat(N, 2).reshape(B),
           ships.reshape(S, K, 1).repeat(N, 2).reshape(B))
    base = jax.random.PRNGKey(args.seed + 1)
    # rollout (state i, n) shares its random stream across candidates
    keys = jax.vmap(lambda i, n: jax.random.fold_in(jax.random.fold_in(base, i), n))(
        jnp.repeat(jnp.arange(S), K * N), jnp.tile(jnp.arange(N), S * K))

    @jax.jit
    def first(st, m, x, y, z, k):
        def one(st, m, x, y, z, k):
            o = policy.sample(pol, heads, st, 3 - m, k)[0]
            a_me = (x, y, z)
            a1 = tuple(jnp.where(m == 1, p, q) for p, q in zip(a_me, o))
            a2 = tuple(jnp.where(m == 2, p, q) for p, q in zip(a_me, o))
            s2 = env.step(st, a1, a2)
            return s2, env.is_terminal(s2)
        return jax.vmap(one)(st, m, x, y, z, k)

    results = np.zeros(B, np.float32)
    if args.reuse:
        assert old["q"].shape == (S, K, N) and (old["src"] == np.asarray(src)).all(), "different positions or candidates"
        results = old["q"].reshape(B)
    for lo in range(0, 0 if args.reuse else B, args.sub):
        sl = slice(lo, min(lo + args.sub, B))
        take = lambda x: x[sl]  # noqa: E731
        st, done = first(jax.tree.map(take, s_rep), take(me_rep), take(act[0]), take(act[1]), take(act[2]),
                         jax.vmap(lambda k: jax.random.fold_in(k, 10**6))(keys[sl]))
        i = 0
        while not bool(done.all()) and i < args.max_chunks:
            st, done = mc(st, take(me_rep), done, jax.vmap(lambda k: jax.random.fold_in(k, i))(keys[sl]))
            i += 1
        results[sl] = np.asarray(jax.vmap(vsearch.outcome)(st, take(me_rep)))
        print(f"  rollouts {sl.stop}/{B} done in {time.time() - t0:.0f}s", flush=True)
    q = results.reshape(S, K, N)
    qm = q.mean(2)

    out = {"states": S, "candidates": K, "rollouts": N}
    distinct = np.array([len({(int(x), int(y), round(float(z), 3)) for x, y, z in zip(src[i], dst[i], ships[i])}) for i in range(S)])
    out["mean_distinct_candidates"] = round(float(distinct.mean()), 2)
    out["q_spread_mean_std_across_candidates"] = round(float(qm.std(1).mean()), 4)
    out["q_of_policy_choice_mean"] = round(float(qm[:, 0].mean()), 4)
    # oracle with honest evaluation: pick on half the rollouts, score on the other half
    h = N // 2
    pick = q[:, :, :h].mean(2).argmax(1)
    held = q[:, :, h:].mean(2)
    out["oracle_gain_over_policy_choice"] = round(float((held[np.arange(S), pick] - held[:, 0]).mean()), 4)
    out["oracle_pick_differs_from_policy_choice"] = round(float((pick != 0).mean()), 3)
    out["no_op_q_minus_policy_q"] = round(float((qm[:, 1] - qm[:, 0]).mean()), 4)
    sem = float((held[np.arange(S), pick] - held[:, 0]).std() / np.sqrt(S))
    out["oracle_gain_sem"] = round(sem, 4)
    for name, v in scores.items():
        picked = (v + args.prior * (np.arange(K) == 0)).argmax(1)
        gain = qm[np.arange(S), picked] - qm[:, 0]
        rho = [spearman(v[i], qm[i]) for i in range(S) if qm[i].std() > 0 and v[i].std() > 0]
        out[name] = dict(gain_over_policy_choice=round(float(gain.mean()), 4), gain_sem=round(float(gain.std() / np.sqrt(S)), 4),
                         pick_differs=round(float((picked != 0).mean()), 3), mean_spearman=round(float(np.nanmean(rho)), 3),
                         mean_score_spread=round(float(v.std(1).mean()), 4))
    print(json.dumps(out, indent=1), flush=True)
    if args.dump:
        np.savez(args.dump, q=q, src=np.asarray(src), dst=np.asarray(dst), ships=np.asarray(ships), me=np.asarray(me),
                 tick=np.asarray(s.tick), **{"s_" + f: np.asarray(getattr(s, f)) for f in env.State._fields},
                 **{"score_" + k.replace(" ", "_").replace("@", "h"): v for k, v in scores.items()})


if __name__ == "__main__":
    main()
