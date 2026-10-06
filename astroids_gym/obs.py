"""Ego-centric observation encoding shared by the single- and multi-agent envs.

Everything is expressed in the observing ship's frame (x' = forward,
y' = right) and scaled into ``[-1, 1]``. Each entity slot starts with a
``present`` flag so absent entries (dead ships, fewer bullets) are zeros.

Layout (default sizes):

* self (8):        alive, vx', vy', sin(a), cos(a), hp, cooldown, invulnerable
* enemies 5 x 10:  present, dx', dy', dvx', dvy', sin(rel heading), cos(rel heading), hp,
                   cos/sin of the lead direction (where to point to hit it)
* allies 4 x 8:    same as enemies, without the lead direction
* bullets 8 x 5:   present, dx', dy', dvx', dvy'   (nearest enemy bullets)
* asteroids 5 x 6: present, dx', dy', dvx', dvy', radius
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .bot import lead
from .sim import Ship, Simulation

SELF_DIM = 8
SHIP_DIM = 8
ENEMY_DIM = SHIP_DIM + 2
BULLET_DIM = 5
ASTEROID_DIM = 6


@dataclass(frozen=True)
class ObsConfig:
    n_enemies: int = 5
    n_allies: int = 4
    n_bullets: int = 8
    n_asteroids: int = 5

    @property
    def size(self) -> int:
        return (SELF_DIM + self.n_enemies * ENEMY_DIM + self.n_allies * SHIP_DIM
                + self.n_bullets * BULLET_DIM + self.n_asteroids * ASTEROID_DIM)


def observation_space(oc: ObsConfig) -> spaces.Box:
    return spaces.Box(-1.0, 1.0, shape=(oc.size,), dtype=np.float32)


def encode(sim: Simulation, s: Ship, oc: ObsConfig) -> np.ndarray:
    cfg = sim.cfg
    out = np.zeros(oc.size, dtype=np.float32)
    ca, sa = math.cos(s.a), math.sin(s.a)
    pos_scale = 1.0 / (cfg.width / 2)
    ship_v_scale = 1.0 / (2 * cfg.max_speed)
    bullet_v_scale = 1.0 / (cfg.bullet_speed + 2 * cfg.max_speed)

    def rot(x: float, y: float) -> tuple[float, float]:
        return x * ca + y * sa, -x * sa + y * ca

    vx, vy = rot(s.vx, s.vy)
    out[0:SELF_DIM] = (
        float(s.alive), vx / cfg.max_speed, vy / cfg.max_speed, sa, ca,
        s.hp / cfg.ship_hp if s.alive else 0.0,
        max(s.cool, 0.0) / cfg.fire_cooldown if s.alive else 0.0,
        float(s.alive and s.invuln > 0),
    )
    i = SELF_DIM

    def nearest(items, k):
        scored = []
        for o in items:
            dx, dy = sim.wdx(s.x, o.x), sim.wdy(s.y, o.y)
            scored.append((dx * dx + dy * dy, dx, dy, o))
        scored.sort(key=lambda t: t[0])
        return scored[:k]

    for enemies, k, dim in ((True, oc.n_enemies, ENEMY_DIM), (False, oc.n_allies, SHIP_DIM)):
        others = [o for o in sim.ships
                  if o.alive and o is not s and (o.team != s.team) == enemies]
        for j, (_, dx, dy, o) in enumerate(nearest(others, k)):
            px, py = rot(dx, dy)
            rvx, rvy = rot(o.vx - s.vx, o.vy - s.vy)
            rel = o.a - s.a
            feats = [1.0, px * pos_scale, py * pos_scale, rvx * ship_v_scale, rvy * ship_v_scale,
                     math.sin(rel), math.cos(rel), o.hp / cfg.ship_hp]
            if enemies:
                aim = lead(sim, s, o)[0] - s.a
                feats += [math.cos(aim), math.sin(aim)]
            out[i + j * dim:i + (j + 1) * dim] = feats
        i += k * dim

    enemy_bullets = [b for b in sim.bullets if b.team != s.team]
    for j, (_, dx, dy, b) in enumerate(nearest(enemy_bullets, oc.n_bullets)):
        px, py = rot(dx, dy)
        rvx, rvy = rot(b.vx - s.vx, b.vy - s.vy)
        out[i + j * BULLET_DIM:i + (j + 1) * BULLET_DIM] = (
            1.0, px * pos_scale, py * pos_scale, rvx * bullet_v_scale, rvy * bullet_v_scale,
        )
    i += oc.n_bullets * BULLET_DIM

    for j, (_, dx, dy, a) in enumerate(nearest(sim.asteroids, oc.n_asteroids)):
        px, py = rot(dx, dy)
        rvx, rvy = rot(a.vx - s.vx, a.vy - s.vy)
        out[i + j * ASTEROID_DIM:i + (j + 1) * ASTEROID_DIM] = (
            1.0, px * pos_scale, py * pos_scale, rvx * ship_v_scale, rvy * ship_v_scale,
            a.r / 48.0,
        )

    np.clip(out, -1.0, 1.0, out=out)
    return out
