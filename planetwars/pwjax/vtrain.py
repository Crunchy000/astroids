"""Train a value function on game results: the critic half of an AlphaZero-style search agent.

Games run in lockstep on the accelerator between a mix of players: trained policies played at several
temperatures with some random moves, and the sample bots. Every `stride` ticks the position is
recorded; when the game ends each recorded position is labelled with the result (+1 win, 0 draw,
-1 loss, seen from the side to move). The value network (the policy network's own architecture, read
off its value output) regresses onto those labels, one fresh batch of games at a time.

    python -m pwjax.vtrain --out runs/v1 --policies it450.npz,it325.npz --minutes 30

Held-out games (fixed maps and players) report the squared error by game phase for the new network,
for the policy's own value head, and for plain ship share, each rescaled to fit the results.
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
from .ppo import potential, save_params, tree_where

KIND_POLICY, KIND_GREEDY, KIND_RUSH, KIND_RANDOM = 0, 1, 2, 3
BOT_KINDS = {"greedy": KIND_GREEDY, "rush": KIND_RUSH, "random": KIND_RANDOM}
PHASES = (0.0, 0.1, 0.25, 0.5, 1.01)


def parse(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--policies", required=True, help="comma separated .npz players (same architecture)")
    a.add_argument("--bots", default="greedy,rush,random")
    a.add_argument("--w-bot", type=float, default=0.15, help="weight of each bot against 1.0 for each policy")
    a.add_argument("--init", default=None, help="value network start (default: the first policy)")
    a.add_argument("--scratch", action="store_true", help="random initial weights")
    a.add_argument("--envs", type=int, default=2048)
    a.add_argument("--groups", type=int, default=16)
    a.add_argument("--stride", type=int, default=16, help="ticks between recorded positions")
    a.add_argument("--ticks", type=int, default=2001, help="ticks played per batch of games")
    a.add_argument("--batches", type=int, default=100)
    a.add_argument("--minutes", type=float, default=0, help="stop after this long (0: run all batches)")
    a.add_argument("--train-steps", type=int, default=64, help="optimizer steps per batch of games")
    a.add_argument("--minibatch", type=int, default=4096)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--lr-end", type=float, default=3e-5)
    a.add_argument("--maps", type=int, default=20000)
    a.add_argument("--val-games", type=int, default=1024)
    a.add_argument("--val-every", type=int, default=10)
    a.add_argument("--save-every", type=int, default=10)
    a.add_argument("--seed", type=int, default=0)
    return a.parse_args(argv)


def pick_players(cfg, n_policies, rng):
    """One player (kind, policy index, temperature, epsilon) per group and side, drawn from the mix."""
    bots = [b for b in cfg.bots.split(",") if b]
    kinds = [(KIND_POLICY, i) for i in range(n_policies)] + [(BOT_KINDS[b], 0) for b in bots]
    w = np.array([1.0] * n_policies + [cfg.w_bot] * len(bots))
    out = []
    for _ in range(2):
        pick = [kinds[i] for i in rng.choice(len(kinds), size=cfg.groups, p=w / w.sum())]
        kind = np.array([k for k, _ in pick])
        idx = np.array([i for _, i in pick])
        temp = np.where(kind == KIND_POLICY, rng.choice([0.7, 1.0, 1.0, 1.5], size=cfg.groups), 1.0)
        eps = np.where(kind == KIND_POLICY, rng.choice([0.0, 0.0, 0.02, 0.06], size=cfg.groups), 0.0)
        out.append((kind, idx, temp.astype(np.float32), eps.astype(np.float32)))
    return out


def make_gen(cfg, bank, heads):
    """Play a batch of games and return the recorded positions with their results."""
    E, G = cfg.envs, cfg.groups
    Eg = E // G
    n_maps = bank["valid"].shape[0]
    v_reset = jax.vmap(lambda i: env.from_bank(bank, i))
    v_step, v_term, v_leader = jax.vmap(env.step), jax.vmap(env.is_terminal), jax.vmap(env.leader)
    v_sample = jax.vmap(lambda p, s, me, k, t, e: policy.sample_t(p, heads, s, me, k, t, e),
                        in_axes=(None, 0, 0, 0, None, None))
    g_sample = jax.vmap(v_sample, in_axes=(0, 0, 0, 0, 0, 0))
    v_greedy, v_rush, v_random = jax.vmap(baselines.greedy), jax.vmap(baselines.rush), jax.vmap(baselines.careful_random)
    n_chunks = -(-cfg.ticks // cfg.stride)

    def act(params, kind, temp, eps, s, me, key):
        k1, k2 = jax.random.split(key)
        gs = jax.tree.map(lambda x: x.reshape((G, Eg) + x.shape[1:]), s)
        pa = g_sample(params, gs, me.reshape(G, Eg), jax.random.split(k1, E).reshape(G, Eg, -1), temp, eps)
        pa = jax.tree.map(lambda x: x.reshape((E,) + x.shape[2:]), pa)
        gr, ru, rn = v_greedy(s, me), v_rush(s, me), v_random(s, me, jax.random.split(k2, E))
        return tuple(jnp.where(kind == KIND_GREEDY, g, jnp.where(kind == KIND_RUSH, r, jnp.where(kind == KIND_RANDOM, q, p)))
                     for p, g, r, q in zip(pa, gr, ru, rn))

    def gen(side1, side2, key):
        (pr1, kd1, tp1, ep1), (pr2, kd2, tp2, ep2) = side1, side2
        k0, key = jax.random.split(key)
        s0 = v_reset(jax.random.randint(k0, (E,), 0, n_maps))
        one, two = jnp.ones(E, jnp.int32), jnp.full(E, 2, jnp.int32)

        def tick(c, k):
            s, done = c
            ka, kb = jax.random.split(k)
            s2 = v_step(s, act(pr1, kd1, tp1, ep1, s, one, ka), act(pr2, kd2, tp2, ep2, s, two, kb))
            s2 = tree_where(done, s, s2)
            return (s2, done | v_term(s2)), None

        def chunk(c, k):
            c2, _ = jax.lax.scan(tick, c, jax.random.split(k, cfg.stride))
            return c2, (c[0], ~c[1])

        (s_end, _), (snaps, alive) = jax.lax.scan(chunk, (s0, jnp.zeros(E, bool)), jax.random.split(key, n_chunks))
        lead = v_leader(s_end)
        z1 = jnp.where(lead == 1, 1.0, jnp.where(lead == 2, -1.0, 0.0))
        return snaps, alive, z1

    return jax.jit(gen)


def flatten(snaps, alive, z1):
    """[chunks, envs, ...] -> positions [chunks * envs, ...], whether recorded before the end, result for player 1."""
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), snaps)
    return flat, alive.reshape(-1), jnp.broadcast_to(z1[None, :], alive.shape).reshape(-1)


def make_train(cfg, heads):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0, eps=1e-5))

    def loss_fn(params, ss, me, z):
        v = jax.vmap(lambda s, m: policy.apply(params, heads, s, m)[2])(ss, me)
        return jnp.mean((v - z) ** 2)

    def train(params, opt_state, flat, alive, z1, key, lr):
        order = jnp.argsort(~alive)  # recorded positions first
        n_alive = jnp.maximum(alive.sum(), 1)

        def step(c, k):
            params, opt_state = c
            k1, k2 = jax.random.split(k)
            ids = order[jax.random.randint(k1, (cfg.minibatch,), 0, n_alive)]
            ss = jax.tree.map(lambda x: x[ids], flat)
            me = jax.random.randint(k2, (cfg.minibatch,), 1, 3)
            z = jnp.where(me == 1, z1[ids], -z1[ids])
            loss, g = jax.value_and_grad(loss_fn)(params, ss, me, z)
            upd, opt_state = opt.update(g, opt_state, params)
            upd = jax.tree.map(lambda u: lr * u, upd)
            return (optax.apply_updates(params, upd), opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jax.random.split(key, cfg.train_steps))
        return params, opt_state, losses.mean()

    return opt, jax.jit(train)


def make_values(heads):
    """Value of a batch of positions from each listed side's view, plus ship share for comparison."""
    @jax.jit
    def values(params, ss, me):
        return jax.vmap(lambda s, m: policy.apply(params, heads, s, m)[2])(ss, me)

    share = jax.jit(jax.vmap(potential))
    return values, share


def batched(fn, args, size=8192):
    n = jax.tree.leaves(args)[0].shape[0]
    outs = [np.asarray(fn(*jax.tree.map(lambda x: x[i:i + size], args))) for i in range(0, n, size)]
    return np.concatenate(outs)


def affine_mse(pred, z):
    """Squared error after the best rescaling a * pred + b (fit on the same positions: kind to the baselines)."""
    a = np.vstack([pred, np.ones_like(pred)]).T
    coef = np.linalg.lstsq(a, z, rcond=None)[0]
    return float(np.mean((a @ coef - z) ** 2)), coef


def report(val, params, ref_params, heads, values, share):
    """Squared error against the results by game phase, for the new network, the reference head and ship share."""
    ss, phase, z1 = val
    rows = {}
    me_all, z_all = [], []
    out = {"new": [], "ref": [], "share": []}
    for me_v in (1, 2):
        me = jnp.full(z1.shape, me_v, jnp.int32)
        out["new"].append(batched(lambda s, m: values(params, s, m), (ss, me)))
        out["ref"].append(batched(lambda s, m: values(ref_params, s, m), (ss, me)))
        out["share"].append(batched(lambda s, m: jax.vmap(potential)(s, m), (ss, me)))
        z_all.append(np.asarray(z1) * (1.0 if me_v == 1 else -1.0))
    z = np.concatenate(z_all)
    ph = np.concatenate([np.asarray(phase)] * 2)
    pred = {k: np.concatenate(v) for k, v in out.items()}
    for k in pred:
        mse, coef = affine_mse(pred[k], z) if k != "new" else (float(np.mean((pred[k] - z) ** 2)), None)
        fit = pred[k] if coef is None else coef[0] * pred[k] + coef[1]
        decisive = z != 0
        rows[k] = dict(mse=round(mse, 4), acc=round(float(np.mean(np.sign(fit[decisive]) == z[decisive])), 4),
                       by_phase=[round(float(np.mean((fit[(ph >= lo) & (ph < hi)] - z[(ph >= lo) & (ph < hi)]) ** 2)), 4)
                                 for lo, hi in zip(PHASES[:-1], PHASES[1:])])
    return rows


def main(argv=None):
    cfg = parse(argv)
    os.makedirs(cfg.out, exist_ok=True)
    print("devices:", jax.devices(), flush=True)
    paths = [p for p in cfg.policies.split(",") if p]
    loaded = [evaluate.load(p) for p in paths]
    heads = loaded[0][1]
    pols = [p for p, _ in loaded]
    assert all(h == heads for _, h in loaded)
    bank_path = os.path.join(cfg.out, "maps.npz")
    if os.path.exists(bank_path):
        bank_np = dict(np.load(bank_path))
    else:
        bank_np = maps.make_bank(cfg.maps, seed=cfg.seed)
        np.savez(bank_path, **bank_np)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    gen = make_gen(cfg, bank, heads)
    opt, train = make_train(cfg, heads)
    values, share = make_values(heads)
    ref = pols[0]
    if cfg.scratch:
        layers = sum(1 for k in ref if k.endswith(".wq"))
        params = policy.init(jax.random.PRNGKey(cfg.seed), ref["in.w"].shape[1], layers, heads,
                             ref["ps.w"].shape[1], ref["l0.w1"].shape[1])
    else:
        params = evaluate.load(cfg.init)[0] if cfg.init else ref
    opt_state = opt.init(params)
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    def batch_args(r, envs):
        def side(players):
            kind, idx, temp, eps = players
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[pols[i] for i in idx])
            return stacked, jnp.asarray(np.repeat(kind, envs // cfg.groups)), jnp.asarray(temp), jnp.asarray(eps)
        a, b = pick_players(cfg, len(pols), r)
        return side(a), side(b)

    # held-out games: other maps, a fixed set of players
    assert cfg.val_games % cfg.groups == 0 and cfg.envs % cfg.groups == 0
    val_cfg = argparse.Namespace(**{**vars(cfg), "envs": cfg.val_games})
    val_bank = {k: jnp.asarray(v) for k, v in maps.make_bank(cfg.val_games, seed=cfg.seed + 777).items()}
    val_gen = make_gen(val_cfg, val_bank, heads)
    s1, s2 = batch_args(np.random.default_rng(cfg.seed + 99), cfg.val_games)
    snaps, alive, z1 = val_gen(s1, s2, jax.random.PRNGKey(cfg.seed + 5))
    flat, alive_f, z_f = flatten(snaps, alive, z1)
    keep = np.asarray(alive_f)
    ticks_total = np.asarray(flat.max_ticks)[keep]
    val = (jax.tree.map(lambda x: jnp.asarray(np.asarray(x)[keep]), flat),
           np.asarray(flat.tick)[keep] / np.maximum(ticks_total, 1), np.asarray(z_f)[keep])
    print(f"held-out set: {keep.sum()} positions from {cfg.val_games} games", flush=True)
    print("before training:", json.dumps(report(val, params, ref, heads, values, share)), flush=True)

    t0 = time.time()
    for b in range(cfg.batches):
        frac = b / max(cfg.batches - 1, 1)
        lr = cfg.lr + (cfg.lr_end - cfg.lr) * frac
        key, kg, kt = jax.random.split(key, 3)
        t1 = time.time()
        snaps, alive, z1 = gen(*batch_args(rng, cfg.envs), kg)
        flat, alive_f, z_f = flatten(snaps, alive, z1)
        jax.block_until_ready(z_f)
        t2 = time.time()
        params, opt_state, loss = train(params, opt_state, flat, alive_f, z_f, kt, lr)
        loss = float(loss)
        row = dict(batch=b + 1, loss=round(loss, 4), positions=int(alive_f.sum()), draws=round(float((z1 == 0).mean()), 3),
                   gen_s=round(t2 - t1, 1), train_s=round(time.time() - t2, 1), lr=lr)
        print(json.dumps(row), flush=True)
        done = cfg.minutes and (time.time() - t0) / 60 >= cfg.minutes
        if (b + 1) % cfg.val_every == 0 or b + 1 == cfg.batches or done:
            rep = report(val, params, ref, heads, values, share)
            print("held-out:", json.dumps(rep), flush=True)
            with open(os.path.join(cfg.out, "val.jsonl"), "a") as f:
                f.write(json.dumps(dict(batch=b + 1, **rep)) + "\n")
        if (b + 1) % cfg.save_every == 0 or b + 1 == cfg.batches or done:
            save_params(os.path.join(cfg.out, "latest.npz"), params, argparse.Namespace(heads=heads))
        if done:
            break


if __name__ == "__main__":
    main()
