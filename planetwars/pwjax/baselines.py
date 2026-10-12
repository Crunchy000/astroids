"""The official sample agents in JAX (each returns an action `(src, dst, ships)` for one game)."""
import jax
import jax.numpy as jnp

from .env import State


def do_nothing(s: State, player, key=None):
    return jnp.int32(-1), jnp.int32(-1), jnp.zeros((), s.ships.dtype)


def greedy(s: State, player, key=None):
    """GreedyHeuristicAgent (competition_entry/GreedyHeuristicAgent.kt, agents/greedy_heuristic_agent.py)."""
    mine = s.valid & (s.owner == player) & ~s.tr_on & (s.ships > 10)
    cand = s.valid & (s.owner != player)
    src = jnp.argmax(jnp.where(mine, s.ships, -jnp.inf))
    dist = jnp.sqrt(jnp.sum((s.pos - s.pos[src]) ** 2, axis=-1))
    strength = jnp.where(s.owner == 0, s.ships, s.ships * 1.5)
    score = jnp.where(cand, strength + dist - 2 * s.growth, jnp.inf)
    dst = jnp.argmin(score)
    defense = s.ships[dst] + s.growth[dst] * (dist[dst] / s.speed)
    go = jnp.any(mine) & jnp.any(cand) & (s.ships[src] > defense)
    return (jnp.where(go, src, -1).astype(jnp.int32), jnp.where(go, dst, -1).astype(jnp.int32),
            jnp.where(go, s.ships[src] / 2, 0.0).astype(s.ships.dtype))


def careful_random(s: State, player, key):
    """CarefulRandomAgent: a random free planet of ours attacks a random enemy planet with half its ships."""
    k1, k2 = jax.random.split(key)
    mine = s.valid & (s.owner == player) & ~s.tr_on
    theirs = s.valid & (s.owner == 3 - player)
    src = jax.random.categorical(k1, jnp.where(mine, 0.0, -jnp.inf))
    dst = jax.random.categorical(k2, jnp.where(theirs, 0.0, -jnp.inf))
    go = jnp.any(mine) & jnp.any(theirs)
    return (jnp.where(go, src, -1).astype(jnp.int32), jnp.where(go, dst, -1).astype(jnp.int32),
            jnp.where(go, s.ships[src] / 2, 0.0).astype(s.ships.dtype))
