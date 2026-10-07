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

With ``ObsConfig(extended=True)`` (for the competitive rules) 32 more follow:

* rules (12):      heat, overheated, zone dx', zone dy', inside zone, zone control
                   (own - enemy ships inside) / 5, score lead (kills + zone points) / 30,
                   own respawns left, enemy respawns left, own ships alive / 5,
                   enemy ships alive / 5, time remaining
* team bullets 4 x 5: nearest teammates' bullets (only with friendly fire), as above

The extra inputs come last, so a classic 160-input model can be warm-started by
giving the new inputs zero weights.
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
RULES_DIM = 12


@dataclass(frozen=True)
class ObsConfig:
    n_enemies: int = 5
    n_allies: int = 4
    n_bullets: int = 8
    n_asteroids: int = 5
    extended: bool = False  # add the competitive-rules block (see module docstring)
    n_team_bullets: int = 4

    @property
    def base_size(self) -> int:
        return (SELF_DIM + self.n_enemies * ENEMY_DIM + self.n_allies * SHIP_DIM
                + self.n_bullets * BULLET_DIM + self.n_asteroids * ASTEROID_DIM)

    @property
    def size(self) -> int:
        extra = RULES_DIM + self.n_team_bullets * BULLET_DIM if self.extended else 0
        return self.base_size + extra


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

    if oc.extended:
        i = oc.base_size
        me, them = s.team, 1 - s.team
        alive_n = [sum(1 for o in sim.ships if o.alive and o.team == t) for t in (0, 1)]
        lives = [(sim.lives[t] / cfg.team_lives if cfg.team_lives else 1.0) for t in (0, 1)]
        zone = [0.0] * 5
        if cfg.zone:
            zx, zy = rot(sim.wdx(s.x, cfg.width / 2), sim.wdy(s.y, cfg.height / 2))
            inside = sim.zone_counts()
            zone = [zx * pos_scale, zy * pos_scale,
                    float(s.alive and math.hypot(zx, zy) < cfg.zone_radius),
                    (inside[me] - inside[them]) / cfg.team_size,
                    (sim.team_score(me) - sim.team_score(them)) / 30.0]
        out[i:i + RULES_DIM] = (
            s.heat if s.alive else 0.0, float(s.alive and s.overheated), *zone,
            lives[me], lives[them], alive_n[me] / cfg.team_size, alive_n[them] / cfg.team_size,
            max(0.0, 1.0 - sim.time / cfg.battle_seconds),
        )
        i += RULES_DIM
        if cfg.friendly_fire:
            mates = [b for b in sim.bullets if b.team == s.team and b.owner is not s]
            for j, (_, dx, dy, b) in enumerate(nearest(mates, oc.n_team_bullets)):
                px, py = rot(dx, dy)
                rvx, rvy = rot(b.vx - s.vx, b.vy - s.vy)
                out[i + j * BULLET_DIM:i + (j + 1) * BULLET_DIM] = (
                    1.0, px * pos_scale, py * pos_scale, rvx * bullet_v_scale, rvy * bullet_v_scale,
                )

    np.clip(out, -1.0, 1.0, out=out)
    return out


_fast = None


def encode_battle(sim: Simulation, ids, oc: ObsConfig) -> np.ndarray:
    """Observations for several ships of one battle, shape (len(ids), oc.size).

    Uses the numba-compiled encoder (``fastobs``, ~8x faster) when numba is
    installed, otherwise ``encode`` per ship. Both give identical results.
    """
    global _fast
    if _fast is None:
        try:
            from .fastobs import encode_fast
            _fast = encode_fast
        except ImportError:
            _fast = lambda sim, ids, oc: np.stack([encode(sim, sim.ships[i], oc) for i in ids])  # noqa: E731
    return _fast(sim, ids, oc)
