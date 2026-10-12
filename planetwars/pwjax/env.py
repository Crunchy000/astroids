"""The Planet Wars forward model in JAX, rule for rule the official ForwardModel (Kotlin and Python).

One game is a `State` of padded arrays; batch games with `jax.vmap`. Players are 1 and 2, neutral 0.
Every planet holds at most one transporter (the fleet it launched), so transporters are stored per
source planet. An action is `(src, dst, ships)`; `src < 0` means do nothing.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .maps import NEUTRAL, P1, P2

WIDTH, HEIGHT = 640.0, 480.0


class State(NamedTuple):
    valid: jax.Array      # [N] bool, real planet
    owner: jax.Array      # [N] int32
    ships: jax.Array      # [N] float
    pos: jax.Array        # [N, 2]
    growth: jax.Array     # [N]
    radius: jax.Array     # [N]
    tr_on: jax.Array      # [N] bool, planet i has a transporter in flight
    tr_owner: jax.Array   # [N] int32
    tr_dest: jax.Array    # [N] int32
    tr_ships: jax.Array   # [N]
    tr_pos: jax.Array     # [N, 2]
    tr_vel: jax.Array     # [N, 2]
    tick: jax.Array       # [] int32
    speed: jax.Array      # [] transporter speed
    max_ticks: jax.Array  # [] int32


def from_bank(bank: dict, i) -> State:
    """Initial state of map i of a bank (a dict of jnp arrays from maps.make_bank)."""
    n = bank["valid"].shape[1]
    f = bank["ships"].dtype
    return State(
        valid=bank["valid"][i], owner=bank["owner"][i], ships=bank["ships"][i], pos=bank["pos"][i],
        growth=bank["growth"][i], radius=bank["radius"][i],
        tr_on=jnp.zeros(n, bool), tr_owner=jnp.zeros(n, jnp.int32), tr_dest=jnp.zeros(n, jnp.int32),
        tr_ships=jnp.zeros(n, f), tr_pos=jnp.zeros((n, 2), f), tr_vel=jnp.zeros((n, 2), f),
        tick=jnp.int32(0), speed=bank["speed"][i], max_ticks=bank["max_ticks"][i],
    )


def _launch(s: State, player, action) -> State:
    src, dst, num = action
    n_planets = jnp.sum(s.valid)
    ok_ids = (src >= 0) & (src < n_planets) & (dst >= 0) & (dst < n_planets)
    num = jnp.asarray(num, s.ships.dtype)
    ok_num = jnp.isfinite(num) & (num > 0)
    si = jnp.clip(src, 0, s.ships.shape[0] - 1)
    di = jnp.clip(dst, 0, s.ships.shape[0] - 1)
    ok = ok_ids & ok_num & ~s.tr_on[si] & (s.owner[si] == player) & (s.ships[si] >= num)
    d = s.pos[di] - s.pos[si]
    mag = jnp.sqrt(jnp.sum(d * d))
    direction = jnp.where(mag > 0, d / jnp.where(mag > 0, mag, 1.0), d)
    at = jnp.arange(s.ships.shape[0]) == si
    sel = at & ok
    return s._replace(
        ships=jnp.where(sel, s.ships - num, s.ships),
        tr_on=s.tr_on | sel,
        tr_owner=jnp.where(sel, player, s.tr_owner),
        tr_dest=jnp.where(sel, di, s.tr_dest),
        tr_ships=jnp.where(sel, num, s.tr_ships),
        tr_pos=jnp.where(sel[:, None], s.pos[si][None], s.tr_pos),
        tr_vel=jnp.where(sel[:, None], (direction * s.speed)[None], s.tr_vel),
    )


def step(s: State, a1, a2) -> State:
    """ForwardModel.step({Player1: a1, Player2: a2})."""
    s = _launch(s, P1, a1)
    s = _launch(s, P2, a2)
    # updateTransporters
    dest_pos = s.pos[s.tr_dest]
    dist = jnp.sqrt(jnp.sum((s.tr_pos - dest_pos) ** 2, axis=-1))
    arrive = s.tr_on & (dist < s.radius[s.tr_dest])
    n = s.ships.shape[0]
    to_dest = (s.tr_dest[:, None] == jnp.arange(n)[None]) & arrive[:, None]           # [src, dst]
    p1_in = jnp.sum(jnp.where(to_dest & (s.tr_owner == P1)[:, None], s.tr_ships[:, None], 0.0), axis=0)
    p2_in = jnp.sum(jnp.where(to_dest & (s.tr_owner == P2)[:, None], s.tr_ships[:, None], 0.0), axis=0)
    has_pending = jnp.any(to_dest, axis=0)
    moving = s.tr_on & ~arrive
    tr_pos = jnp.where(moving[:, None], s.tr_pos + s.tr_vel, s.tr_pos)
    tr_on = moving
    # updatePlanets
    neutral = s.owner == NEUTRAL
    net = p1_in - p2_in
    n_ships = s.ships - jnp.abs(net)
    n_flip = has_pending & (n_ships < 0)
    n_owner = jnp.where(n_flip, jnp.where(net > 0, P1, P2), s.owner)
    n_ships = jnp.where(has_pending, jnp.where(n_flip, -n_ships, n_ships), s.ships)
    own_in = jnp.where(s.owner == P1, p1_in, p2_in)
    opp_in = jnp.where(s.owner == P1, p2_in, p1_in)
    p_ships = s.ships + s.growth + jnp.where(has_pending, own_in - opp_in, 0.0)
    p_flip = has_pending & (p_ships < 0)
    p_owner = jnp.where(p_flip, 3 - s.owner, s.owner)
    p_ships = jnp.where(p_flip, -p_ships, p_ships)
    real = s.valid
    owner = jnp.where(real, jnp.where(neutral, n_owner, p_owner), s.owner)
    ships = jnp.where(real, jnp.where(neutral, n_ships, p_ships), s.ships)
    return s._replace(owner=owner, ships=ships, tr_on=tr_on, tr_pos=tr_pos, tick=s.tick + 1)


def planet_ships(s: State, player):
    return jnp.sum(jnp.where(s.valid & (s.owner == player), s.ships, 0.0))


def is_terminal(s: State):
    has1 = jnp.any(s.valid & (s.owner == P1))
    has2 = jnp.any(s.valid & (s.owner == P2))
    return (s.tick > s.max_ticks) | ~has1 | ~has2


def leader(s: State):
    """getLeader(): the player with more ships on planets (transporters don't count); 0 on a tie."""
    s1, s2 = planet_ships(s, P1), planet_ships(s, P2)
    return jnp.where(s1 > s2, P1, jnp.where(s2 > s1, P2, NEUTRAL))
