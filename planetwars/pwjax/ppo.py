"""PPO self-play for Planet Wars, fully on the accelerator.

Thousands of games run in lockstep under `jax.vmap`; a rollout is one `lax.scan`. The learner plays
one side of every game. The other side is, per group of games: the learner itself, an older snapshot
of it, the official greedy heuristic agent, or the official careful random agent. Opponents are
reassigned (and every game restarted) each "round", which is long enough for a full game.

    python -m pwjax.ppo --out runs/pw1 --envs 2048 --iters 3000
"""
import argparse
import json
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import baselines, env, maps, policy

OPP_SELF, OPP_POOL, OPP_GREEDY, OPP_RANDOM = 0, 1, 2, 3
OPP_NAMES = ("self", "pool", "greedy", "random")


def parse(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--envs", type=int, default=2048)
    a.add_argument("--groups", type=int, default=16)
    a.add_argument("--steps", type=int, default=128, help="rollout length")
    a.add_argument("--round", type=int, default=16, help="rollouts per opponent assignment")
    a.add_argument("--iters", type=int, default=3000)
    a.add_argument("--maps", type=int, default=50000)
    a.add_argument("--opp-probs", default="0.3,0.4,0.2,0.1", help="self,pool,greedy,random")
    a.add_argument("--pool-size", type=int, default=12)
    a.add_argument("--pool-every", type=int, default=32, help="iterations between snapshots")
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--gamma", type=float, default=0.999)
    a.add_argument("--lam", type=float, default=0.95)
    a.add_argument("--clip", type=float, default=0.2)
    a.add_argument("--ent", type=float, default=0.01)
    a.add_argument("--ent-end", type=float, default=0.002)
    a.add_argument("--vf", type=float, default=0.5)
    a.add_argument("--shaping", type=float, default=0.5, help="weight of the ship-share potential")
    a.add_argument("--epochs", type=int, default=3)
    a.add_argument("--minibatch", type=int, default=4096)
    a.add_argument("--d-model", type=int, default=128)
    a.add_argument("--layers", type=int, default=3)
    a.add_argument("--heads", type=int, default=4)
    a.add_argument("--d-pair", type=int, default=32)
    a.add_argument("--d-ff", type=int, default=256)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--save-every", type=int, default=50)
    a.add_argument("--init", default=None, help="start from this params .npz")
    a.add_argument("--resume", action="store_true")
    return a.parse_args(argv)


def potential(s, me):
    """Ship share in [-1, 1] (planets and fleets) from `me`'s side: the shaping potential."""
    def total(player):
        return (jnp.sum(jnp.where(s.valid & (s.owner == player), s.ships, 0.0))
                + jnp.sum(jnp.where(s.tr_on & (s.tr_owner == player), s.tr_ships, 0.0)))
    mine, theirs = total(me), total(3 - me)
    return (mine - theirs) / (mine + theirs + 1.0)


def tree_where(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c.reshape(c.shape + (1,) * (x.ndim - c.ndim)), x, y), a, b)


def make_fns(cfg, bank):
    heads = cfg.heads
    E, G = cfg.envs, cfg.groups
    Eg = E // G
    n_maps = bank["valid"].shape[0]
    sample = partial(policy.sample, heads=heads)
    v_sample = jax.vmap(lambda p, s, me, k: sample(p, s=s, me=me, key=k), in_axes=(None, 0, 0, 0))
    g_sample = jax.vmap(v_sample, in_axes=(0, 0, 0, 0))  # per group params
    v_step = jax.vmap(env.step)
    v_reset = jax.vmap(lambda i: env.from_bank(bank, i))
    v_terminal = jax.vmap(env.is_terminal)
    v_leader = jax.vmap(env.leader)
    v_greedy = jax.vmap(baselines.greedy)
    v_random = jax.vmap(baselines.careful_random)

    def reset(key):
        k1, k2 = jax.random.split(key)
        s = v_reset(jax.random.randint(k1, (E,), 0, n_maps))
        me = jax.random.randint(k2, (E,), 1, 3)
        return s, me

    def rollout(params, opp_params, opp_type, carry, key):
        """opp_params: params stacked per group [G, ...]; opp_type: [E]."""

        def one(c, k):
            s, me, ep_ret = c
            k1, k2, k3, k4 = jax.random.split(k, 4)
            act, idx, aidx, logp, value, _ = v_sample(params, s, me, jax.random.split(k1, E))
            pot = jax.vmap(potential)(s, me)
            opp_me = 3 - me
            gs = jax.tree.map(lambda x: x.reshape((G, Eg) + x.shape[1:]), s)
            o_act = g_sample(opp_params, gs, opp_me.reshape(G, Eg), jax.random.split(k2, E).reshape(G, Eg, -1))[0]
            o_act = jax.tree.map(lambda x: x.reshape((E,) + x.shape[2:]), o_act)
            gr = v_greedy(s, opp_me)
            rn = v_random(s, opp_me, jax.random.split(k3, E))
            o_act = tuple(jnp.where(opp_type == OPP_GREEDY, g, jnp.where(opp_type == OPP_RANDOM, r, o))
                          for o, g, r in zip(o_act, gr, rn))
            a1 = tuple(jnp.where(me == 1, x, y) for x, y in zip(act, o_act))
            a2 = tuple(jnp.where(me == 2, x, y) for x, y in zip(act, o_act))
            s2 = v_step(s, a1, a2)
            done = v_terminal(s2)
            lead = v_leader(s2)
            outcome = jnp.where(lead == me, 1.0, jnp.where(lead == 0, 0.0, -1.0))
            pot2 = jax.vmap(potential)(s2, me)
            reward = jnp.where(done, outcome, cfg.gamma * cfg.shaping * pot2) - cfg.shaping * pot
            s_new, me_new = reset(k4)
            s_next = tree_where(done, s_new, s2)
            me_next = jnp.where(done, me_new, me)
            ep_ret = ep_ret + reward
            out = dict(s=s, me=me, idx=idx, aidx=aidx, logp=logp, value=value, reward=reward, done=done,
                       outcome=outcome, ep_ret=ep_ret, ep_len=s2.tick)
            ep_ret = jnp.where(done, 0.0, ep_ret)
            return (s_next, me_next, ep_ret), out

        carry, traj = jax.lax.scan(one, carry, jax.random.split(key, cfg.steps))
        s, me, _ = carry
        last_value = v_sample(params, s, me, jax.random.split(key, E))[4]
        return carry, traj, last_value

    def gae(traj, last_value):
        def back(next_adv_v, x):
            next_adv, next_v = next_adv_v
            r, d, v = x
            nonterm = 1.0 - d.astype(jnp.float32)
            delta = r + cfg.gamma * next_v * nonterm - v
            adv = delta + cfg.gamma * cfg.lam * nonterm * next_adv
            return (adv, v), adv

        _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value),
                              (traj["reward"], traj["done"], traj["value"]), reverse=True)
        return adv, adv + traj["value"]

    v_eval = jax.vmap(lambda p, s, me, i, a: policy.evaluate(p, heads, s, me, i, a), in_axes=(None, 0, 0, 0, 0))

    def loss_fn(params, mb, ent_coef):
        logp, ent, value = v_eval(params, mb["s"], mb["me"], mb["idx"], mb["aidx"])
        ratio = jnp.exp(logp - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_clip = mb["value"] + jnp.clip(value - mb["value"], -cfg.clip, cfg.clip)
        vl = 0.5 * jnp.maximum((value - mb["ret"]) ** 2, (v_clip - mb["ret"]) ** 2).mean()
        loss = pg + cfg.vf * vl - ent_coef * ent.mean()
        return loss, dict(pg=pg, vf=vl, ent=ent.mean(), kl=((ratio - 1) - jnp.log(ratio)).mean(),
                          clipfrac=(jnp.abs(ratio - 1) > cfg.clip).mean())

    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr, eps=1e-5))

    def update(params, opt_state, traj, last_value, key, ent_coef):
        adv, ret = gae(traj, last_value)
        data = dict(s=traj["s"], me=traj["me"], idx=traj["idx"], aidx=traj["aidx"], logp=traj["logp"],
                    value=traj["value"], adv=adv, ret=ret)
        n = cfg.steps * E
        data = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), data)
        n_mb = max(n // cfg.minibatch, 1)

        def epoch(c, k):
            params, opt_state = c
            perm = jax.random.permutation(k, n)[: n_mb * (n // n_mb)].reshape(n_mb, -1)

            def mb_step(c, ix):
                params, opt_state = c
                mb = jax.tree.map(lambda x: x[ix], data)
                (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, ent_coef)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), m

            return jax.lax.scan(mb_step, (params, opt_state), perm)

        (params, opt_state), m = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, cfg.epochs))
        return params, opt_state, jax.tree.map(jnp.mean, m)

    return dict(reset=jax.jit(reset), rollout=jax.jit(rollout), update=jax.jit(update), opt=opt)


def save_params(path, params, cfg):
    arrays = policy.to_numpy(params)
    arrays["cfg.heads"] = np.array(cfg.heads)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_params(path):
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files if not k.startswith("cfg.")}


def main(argv=None):
    cfg = parse(argv)
    os.makedirs(cfg.out, exist_ok=True)
    print("devices:", jax.devices(), flush=True)
    assert cfg.envs % cfg.groups == 0
    t0 = time.time()
    bank_path = os.path.join(cfg.out, "maps.npz")
    if os.path.exists(bank_path):
        bank_np = dict(np.load(bank_path))
    else:
        bank_np = maps.make_bank(cfg.maps, seed=cfg.seed)
        np.savez(bank_path, **bank_np)
    bank = {k: jnp.asarray(v) for k, v in bank_np.items()}
    print(f"{cfg.maps} maps ready in {time.time() - t0:.0f}s", flush=True)
    fns = make_fns(cfg, bank)
    key = jax.random.PRNGKey(cfg.seed)
    key, k = jax.random.split(key)
    params = policy.init(k, cfg.d_model, cfg.layers, cfg.heads, cfg.d_pair, cfg.d_ff)
    if cfg.init:
        params = load_params(cfg.init)
    opt_state = fns["opt"].init(params)
    pool = [params]
    start = 0
    log_path = os.path.join(cfg.out, "log.jsonl")
    ckpt = os.path.join(cfg.out, "ckpt.pkl")
    if cfg.resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        params, opt_state, start, key = st["params"], st["opt_state"], st["iter"], st["key"]
        pool = st["pool"]
        print("resumed at iteration", start, flush=True)
    probs = np.array([float(x) for x in cfg.opp_probs.split(",")])
    probs = probs / probs.sum()
    host_rng = np.random.default_rng(cfg.seed + start)
    G, Eg = cfg.groups, cfg.envs // cfg.groups
    carry = None
    stats = {n: np.zeros(3) for n in OPP_NAMES}  # wins, draws, losses since the last log line
    t_last = time.time()
    for it in range(start, cfg.iters):
        if it % cfg.round == 0 or carry is None:
            types = host_rng.choice(4, size=G, p=probs)
            group_params = []
            for t in types:
                if t == OPP_POOL:
                    group_params.append(pool[host_rng.integers(len(pool))])
                else:
                    group_params.append(params)
            opp_params = jax.tree.map(lambda *xs: jnp.stack(xs), *group_params)
            opp_type = jnp.asarray(np.repeat(types, Eg))
            key, k = jax.random.split(key)
            s, me = fns["reset"](k)
            carry = (s, me, jnp.zeros(cfg.envs))
        key, k1, k2 = jax.random.split(key, 3)
        carry, traj, last_value = fns["rollout"](params, opp_params, opp_type, carry, k1)
        frac = it / max(cfg.iters - 1, 1)
        ent_coef = cfg.ent + (cfg.ent_end - cfg.ent) * frac
        params, opt_state, m = fns["update"](params, opt_state, traj, last_value, k2, ent_coef)
        done = np.asarray(traj["done"])
        outcome = np.asarray(traj["outcome"])
        ot = np.asarray(opp_type)
        for t, name in enumerate(OPP_NAMES):
            sel = done & (ot[None, :] == t)
            o = outcome[sel]
            stats[name] += [np.sum(o > 0), np.sum(o == 0), np.sum(o < 0)]
        if (it + 1) % cfg.pool_every == 0:
            pool.append(params)
            pool = pool[-cfg.pool_size:]
        if (it + 1) % 10 == 0 or it == start:
            dt = time.time() - t_last
            t_last = time.time()
            m = {k: float(v) for k, v in m.items()}
            row = dict(iter=it + 1, steps=(it + 1) * cfg.envs * cfg.steps, sec=round(dt, 1), **m)
            for name, (w, d, l) in stats.items():
                g = w + d + l
                row[f"win_{name}"] = round(float((w + 0.5 * d) / g), 3) if g else None
                row[f"games_{name}"] = int(g)
            ends = np.asarray(traj["ep_len"])[done]
            row["game_len"] = float(ends.mean()) if ends.size else None
            print(json.dumps(row), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            stats = {n: np.zeros(3) for n in OPP_NAMES}
        if (it + 1) % cfg.save_every == 0 or it + 1 == cfg.iters:
            save_params(os.path.join(cfg.out, "latest.npz"), params, cfg)
            with open(ckpt + ".tmp", "wb") as f:
                pickle.dump(dict(params=jax.device_get(params), opt_state=jax.device_get(opt_state),
                                 iter=it + 1, key=key, pool=jax.device_get(pool)), f)
            os.replace(ckpt + ".tmp", ckpt)


if __name__ == "__main__":
    main()
