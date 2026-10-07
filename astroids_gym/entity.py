"""Attention over entities: the observation as a set of tokens, and a NumPy forward pass.

An MLP sees the observation as one flat vector: the nearest enemy is always
inputs 8-17, the second nearest 18-27, and so on, and it has to learn the same
thing separately for every slot. The entity network instead splits the
observation into *tokens*: one for the ship itself (its own state plus the
rules block), and one per enemy, ally, bullet, asteroid and team bullet. Each
kind of token has its own embedding, shared across slots, and a transformer
layer lets every token attend to every other (absent slots are masked out).
The output is the ship's own token plus the mean over present entities, which
feeds the usual MLP head. The same weights handle any number of visible
entities, in any order.

The PyTorch module for training is ``policies.EntityExtractor``; this module
holds the layout and a NumPy forward pass (for frozen opponents and
tournaments, which never import torch). ``tests`` check the two agree.
"""

from __future__ import annotations

import math

import numpy as np

from .obs import ASTEROID_DIM, BULLET_DIM, ENEMY_DIM, RULES_DIM, SELF_DIM, SHIP_DIM, ObsConfig


def entity_groups(oc: ObsConfig) -> list[tuple[str, int, int, int]]:
    """(name, start, count, features) of each entity slot group in the observation."""
    groups, i = [], SELF_DIM
    for name, count, dim in (("enemy", oc.n_enemies, ENEMY_DIM), ("ally", oc.n_allies, SHIP_DIM),
                             ("bullet", oc.n_bullets, BULLET_DIM), ("asteroid", oc.n_asteroids, ASTEROID_DIM)):
        groups.append((name, i, count, dim))
        i += count * dim
    if oc.extended:
        groups.append(("team_bullet", oc.base_size + RULES_DIM, oc.n_team_bullets, BULLET_DIM))
    return groups


def context_slices(oc: ObsConfig) -> list[tuple[int, int]]:
    """The observation ranges that make up the ship's own token: its state, and the rules block."""
    out = [(0, SELF_DIM)]
    if oc.extended:
        out.append((oc.base_size, oc.base_size + RULES_DIM))
    return out


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def entity_features(p: dict, obs: np.ndarray, oc: ObsConfig, heads: int) -> np.ndarray:
    """NumPy twin of ``policies.EntityExtractor.forward``; ``p`` maps its state-dict keys to arrays."""
    obs = obs.astype(np.float32)
    n = obs.shape[0]

    def lin(name, x):  # one 2-D matmul over all tokens (much faster than a stack of small ones)
        w = p[name + ".weight"]
        return (x.reshape(-1, x.shape[-1]) @ w.T + p[name + ".bias"]).reshape(*x.shape[:-1], w.shape[0])

    ctx = np.concatenate([obs[:, a:b] for a, b in context_slices(oc)], axis=1)
    toks, masks = [lin("ctx", ctx)[:, None]], [np.ones((n, 1), bool)]
    for name, start, count, dim in entity_groups(oc):
        x = obs[:, start:start + count * dim].reshape(n, count, dim)
        toks.append(lin(f"emb.{name}", x))
        masks.append(x[:, :, 0] > 0.5)
    x, mask = np.concatenate(toks, axis=1), np.concatenate(masks, axis=1)
    t, d = x.shape[1], x.shape[2]
    dh = d // heads

    h = _layer_norm(x, p["ln1.weight"], p["ln1.bias"])
    q, k, v = np.split(lin("qkv", h), 3, axis=-1)
    q, k, v = (z.reshape(n, t, heads, dh).transpose(0, 2, 1, 3) for z in (q, k, v))
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)  # a Python float keeps float32
    scores = np.where(mask[:, None, None, :], scores, -1e9)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    att = scores / scores.sum(-1, keepdims=True)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(n, t, d)
    x = x + lin("proj", out)
    x = x + lin("ff2", np.maximum(lin("ff1", _layer_norm(x, p["ln2.weight"], p["ln2.bias"])), 0.0))

    m = mask[:, 1:, None]
    pooled = (x[:, 1:] * m).sum(1) / np.maximum(m.sum(1), 1).astype(np.float32)
    return np.concatenate([x[:, 0], pooled], axis=1)
