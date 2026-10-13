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


def rush(s: State, player, key=None):
    """The greedy agent's targeting, but every launch sends all of the source planet's ships."""
    src, dst, _ = greedy(s, player)
    go = src >= 0
    return src, dst, jnp.where(go, s.ships[jnp.maximum(src, 0)], 0.0).astype(s.ships.dtype)


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


def sniper(s: State, player, key=None):
    """A stronger hand-written agent (ours, not from the competition): sends just enough ships.

    Every free planet with spare ships (beyond what it needs to survive fleets already heading to
    it) may attack; targets are scored by growth gained per ship and tick spent, accounting for the
    target's growth on the way and every fleet already heading there.
    """
    n = s.ships.shape[0]
    opp = 3 - player
    mine = s.valid & (s.owner == player)
    to = (s.tr_dest[:, None] == jnp.arange(n)[None, :]) & s.tr_on[:, None]
    my_in = jnp.sum(jnp.where(to & (s.tr_owner == player)[:, None], s.tr_ships[:, None], 0.0), axis=0)
    opp_in = jnp.sum(jnp.where(to & (s.tr_owner == opp)[:, None], s.tr_ships[:, None], 0.0), axis=0)
    keep = jnp.maximum(opp_in - my_in, 0.0)
    avail = jnp.where(mine & ~s.tr_on, s.ships - keep, 0.0)
    d = jnp.sqrt(jnp.sum((s.pos[:, None, :] - s.pos[None, :, :]) ** 2, axis=-1))
    eta = d / s.speed
    is_opp = (s.owner == opp)[None, :]
    is_neu = (s.owner == 0)[None, :]
    defense = jnp.where(is_opp, s.ships[None, :] + s.growth[None, :] * eta, s.ships[None, :]) \
        + opp_in[None, :] - my_in[None, :]
    need = defense + 2.0
    target = s.valid[None, :] & (is_opp | is_neu)
    ok = target & (avail[:, None] > need) & (need > 0) & (avail[:, None] > 5)
    score = s.growth[None, :] * jnp.where(is_opp, 2.0, 1.0) / (need + 5.0 + 0.5 * eta)
    score = jnp.where(ok, score, -jnp.inf)
    flat = jnp.argmax(score.reshape(-1))
    i, j = flat // n, flat % n
    go = jnp.any(ok)
    return (jnp.where(go, i, -1).astype(jnp.int32), jnp.where(go, j, -1).astype(jnp.int32),
            jnp.where(go, need[i, j], 0.0).astype(s.ships.dtype))
