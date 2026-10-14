"""JAX side of the policy: parameter init, sampling and log-probabilities (the network is entry/pwnet.py)."""
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "pwnet", pathlib.Path(__file__).resolve().parents[1] / "entry" / "pwnet.py")
pwnet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pwnet)

SMALL_INIT = ("p3.w", "pa.w", "n2.w")


def init(key, d_model=128, layers=3, heads=4, d_pair=32, d_ff=256) -> dict:
    shapes = pwnet.param_shapes(d_model=d_model, layers=layers, heads=heads, d_pair=d_pair, d_ff=d_ff)
    params = {}
    for (name, shape), k in zip(sorted(shapes.items()), jax.random.split(key, len(shapes))):
        if name.endswith(".g"):
            params[name] = jnp.ones(shape)
        elif len(shape) == 1:
            params[name] = jnp.zeros(shape)
        else:
            scale = (0.01 if name in SMALL_INIT else 1.0) / np.sqrt(shape[0])
            params[name] = jax.random.normal(k, shape) * scale
    return params


def apply(params, heads, s, me):
    """Network outputs for one game from player `me`'s point of view."""
    planet, glob, pair, aux = pwnet.features(jnp, s, me)
    flat, amount_logits, value = pwnet.forward(jnp, params, heads, planet, glob, pair, s.valid, aux["pair_ok"])
    return flat, amount_logits, value, aux


def _amount_dist(amount_logits, idx):
    n = amount_logits.shape[0]
    pi = jnp.clip(idx, 0, n * n - 1)
    return amount_logits.reshape(n * n, -1)[pi]


def sample(params, heads, s, me, key, greedy=False):
    """Sample an action. Returns (action tuple, pair index, amount index, log-prob, value, potential)."""
    flat, amount_logits, value, aux = apply(params, heads, s, me)
    k1, k2 = jax.random.split(key)
    n = amount_logits.shape[0]
    idx = jnp.argmax(flat) if greedy else jax.random.categorical(k1, flat)
    al = _amount_dist(amount_logits, idx)
    aidx = jnp.argmax(al) if greedy else jax.random.categorical(k2, al)
    noop = idx >= n * n
    logp = jax.nn.log_softmax(flat)[idx] + jnp.where(noop, 0.0, jax.nn.log_softmax(al)[aidx])
    src = jnp.where(noop, -1, idx // n).astype(jnp.int32)
    dst = jnp.where(noop, -1, idx % n).astype(jnp.int32)
    ships = jnp.where(noop, 0.0, aux["amounts"].reshape(n * n, -1)[jnp.clip(idx, 0, n * n - 1), aidx])
    return (src, dst, ships), idx, aidx, logp, value, aux["potential"]


def sample_t(params, heads, s, me, key, temp, eps):
    """Sample an action with the logits divided by `temp`; with probability `eps` play a uniformly random legal move.

    Used to make the games a value function learns from more varied than the policy's own play.
    """
    flat, amount_logits, _, aux = apply(params, heads, s, me)
    k1, k2, k3 = jax.random.split(key, 3)
    n = amount_logits.shape[0]
    rand = jax.random.uniform(k3) < eps
    logits = jnp.where(rand, jnp.where(flat > -1e8, 0.0, flat), flat / temp)
    idx = jax.random.categorical(k1, logits)
    al = _amount_dist(amount_logits, idx)
    aidx = jax.random.categorical(k2, jnp.where(rand, jnp.zeros_like(al), al / temp))
    noop = idx >= n * n
    src = jnp.where(noop, -1, idx // n).astype(jnp.int32)
    dst = jnp.where(noop, -1, idx % n).astype(jnp.int32)
    ships = jnp.where(noop, 0.0, aux["amounts"].reshape(n * n, -1)[jnp.clip(idx, 0, n * n - 1), aidx])
    return src, dst, ships


def evaluate(params, heads, s, me, idx, aidx):
    """Log-probability, entropy and value of a taken action (for the PPO update)."""
    flat, amount_logits, value, _ = apply(params, heads, s, me)
    n = amount_logits.shape[0]
    noop = idx >= n * n
    lp = jax.nn.log_softmax(flat)
    al = jax.nn.log_softmax(_amount_dist(amount_logits, idx))
    logp = lp[idx] + jnp.where(noop, 0.0, al[aidx])
    p = jnp.exp(lp)
    ent = -jnp.sum(jnp.where(p > 0, p * lp, 0.0)) - jnp.where(noop, 0.0, jnp.sum(jnp.exp(al) * al))
    return logp, ent, value


def to_numpy(params) -> dict:
    return {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
