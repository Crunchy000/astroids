"""Headless 5v5 Asteroids battle simulation.

A Python port of the browser game in ``index.html``: same physics constants,
toroidal (wrapping) world, bullets, splitting asteroids, hit points and
respawns. Ships are driven by per-step actions so that learning agents and the
built-in scripted bot can be mixed freely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

TAU = 2 * math.pi
N_TEAMS = 2


@dataclass
class Config:
    width: float = 1280.0
    height: float = 720.0
    team_size: int = 5
    ship_r: float = 11.0
    ship_hp: int = 3
    turn_rate: float = 4.2  # rad/s
    thrust: float = 220.0  # px/s^2
    max_speed: float = 260.0
    drag: float = 0.35  # per second
    bullet_speed: float = 480.0
    bullet_life: float = 1.3
    fire_cooldown: float = 0.32
    respawn_time: float = 3.0
    spawn_invuln: float = 1.5
    ram_speed: float = 120.0  # closing speed above which enemy ships damage each other
    asteroids: bool = True
    asteroid_count: int = 7
    min_asteroids: int = 3
    dt: float = 1 / 60


def wrap_angle(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


class Ship:
    __slots__ = (
        "id", "team", "x", "y", "vx", "vy", "a", "hp", "cool", "invuln",
        "alive", "respawn", "thrusting", "kills", "deaths",
        # scripted-bot personality
        "wander", "aggro", "pref_dist", "target",
    )

    def __init__(self, ship_id: int, team: int):
        self.id = ship_id
        self.team = team
        self.kills = 0
        self.deaths = 0
        self.alive = False
        self.respawn = 0.0
        self.thrusting = False
        self.target: Ship | None = None


class Bullet:
    __slots__ = ("x", "y", "vx", "vy", "life", "team", "owner")

    def __init__(self, x, y, vx, vy, life, team, owner):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life, self.team, self.owner = life, team, owner


class Asteroid:
    __slots__ = ("x", "y", "vx", "vy", "r", "size", "a", "spin", "shape", "hp")


class Events:
    """Per-ship counters accumulated during one :meth:`Simulation.step`."""

    def __init__(self, n: int):
        self.hits_dealt = [0] * n
        self.hits_taken = [0] * n
        self.kills = [0] * n
        self.deaths = [0] * n
        self.shots = [0] * n

    def merge(self, other: "Events") -> None:
        for name in ("hits_dealt", "hits_taken", "kills", "deaths", "shots"):
            mine, theirs = getattr(self, name), getattr(other, name)
            for i, v in enumerate(theirs):
                mine[i] += v


class Simulation:
    """The battle world. Ship ``i`` is on team ``i // team_size`` (0 = blue, 1 = red)."""

    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    # ---- helpers -------------------------------------------------------
    def rand(self, a: float = 0.0, b: float = 1.0) -> float:
        return a + self.rng.random() * (b - a)

    def wdx(self, a: float, b: float) -> float:
        """Shortest x-delta from a to b on the wrapping world."""
        w = self.cfg.width
        d = b - a
        if d > w / 2:
            d -= w
        elif d < -w / 2:
            d += w
        return d

    def wdy(self, a: float, b: float) -> float:
        h = self.cfg.height
        d = b - a
        if d > h / 2:
            d -= h
        elif d < -h / 2:
            d += h
        return d

    def _wrap(self, o) -> None:
        o.x %= self.cfg.width
        o.y %= self.cfg.height

    @property
    def n_ships(self) -> int:
        return self.cfg.team_size * N_TEAMS

    # ---- lifecycle -----------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.time = 0.0
        self.score = [0, 0]
        self.ships = [Ship(i, i // cfg.team_size) for i in range(self.n_ships)]
        for s in self.ships:
            self._spawn_ship(s)
        self.bullets: list[Bullet] = []
        self.asteroids: list[Asteroid] = []
        if cfg.asteroids:
            for _ in range(cfg.asteroid_count):
                self.asteroids.append(self._make_asteroid(
                    self.rand(cfg.width * 0.3, cfg.width * 0.7),
                    self.rand(0, cfg.height),
                    3 if self.rng.random() < 0.5 else 2,
                ))

    def _spawn_ship(self, s: Ship) -> None:
        cfg = self.cfg
        if s.team == 0:
            s.x = self.rand(cfg.width * 0.05, cfg.width * 0.2)
        else:
            s.x = self.rand(cfg.width * 0.8, cfg.width * 0.95)
        s.y = self.rand(cfg.height * 0.15, cfg.height * 0.85)
        s.vx = s.vy = 0.0
        s.a = 0.0 if s.team == 0 else math.pi
        s.hp = cfg.ship_hp
        s.cool = self.rand(0, cfg.fire_cooldown)
        s.invuln = cfg.spawn_invuln
        s.alive = True
        s.thrusting = False
        s.target = None
        s.wander = self.rand(-1, 1)
        s.aggro = self.rand(0.6, 1.0)
        s.pref_dist = self.rand(140, 260)

    def _make_asteroid(self, x: float, y: float, size: int) -> Asteroid:
        a = Asteroid()
        a.x, a.y, a.size, a.hp = x, y, size, size
        a.r = size * 16.0
        a.shape = [self.rand(0.72, 1.15) for _ in range(9 + int(self.rng.integers(0, 4)))]
        sp = self.rand(20, 60) / size
        d = self.rand(0, TAU)
        a.vx, a.vy = math.cos(d) * sp, math.sin(d) * sp
        a.a = self.rand(0, TAU)
        a.spin = self.rand(-0.8, 0.8)
        return a

    # ---- combat --------------------------------------------------------
    def _hit(self, s: Ship, attacker: Ship | None, ev: Events) -> None:
        if s.invuln > 0 or not s.alive:
            return
        s.hp -= 1
        ev.hits_taken[s.id] += 1
        if attacker is not None and attacker.team != s.team:
            ev.hits_dealt[attacker.id] += 1
        if s.hp <= 0:
            self._kill(s, attacker, ev)

    def _kill(self, s: Ship, attacker: Ship | None, ev: Events) -> None:
        s.alive = False
        s.respawn = self.cfg.respawn_time
        s.deaths += 1
        ev.deaths[s.id] += 1
        if attacker is not None and attacker.team != s.team:
            self.score[attacker.team] += 1
            attacker.kills += 1
            ev.kills[attacker.id] += 1

    def _break_asteroid(self, idx: int) -> None:
        a = self.asteroids.pop(idx)
        if a.size > 1:
            for _ in range(2):
                self.asteroids.append(self._make_asteroid(a.x, a.y, a.size - 1))

    # ---- step ----------------------------------------------------------
    def step(self, actions: dict[int, tuple[float, bool, bool]] | Iterable) -> Events:
        """Advance one physics tick.

        ``actions`` maps ship id -> ``(turn, thrust, fire)`` where ``turn`` is in
        ``[-1, 1]`` (fraction of max turn rate, positive = clockwise on screen).
        Ships without an action coast. Dead ships ignore their action.
        """
        cfg = self.cfg
        dt = cfg.dt
        ev = Events(self.n_ships)
        if not isinstance(actions, dict):
            actions = dict(enumerate(actions))

        for s in self.ships:
            if not s.alive:
                s.respawn -= dt
                if s.respawn <= 0:
                    self._spawn_ship(s)
                continue
            s.cool -= dt
            s.invuln -= dt
            turn, thrust, fire = actions.get(s.id, (0.0, False, False))
            turn = max(-1.0, min(1.0, float(turn)))
            s.a = wrap_angle(s.a + turn * cfg.turn_rate * dt)
            s.thrusting = bool(thrust)
            if s.thrusting:
                s.vx += math.cos(s.a) * cfg.thrust * dt
                s.vy += math.sin(s.a) * cfg.thrust * dt
            drag = math.exp(-cfg.drag * dt)
            s.vx *= drag
            s.vy *= drag
            sp = math.hypot(s.vx, s.vy)
            if sp > cfg.max_speed:
                s.vx *= cfg.max_speed / sp
                s.vy *= cfg.max_speed / sp
            s.x += s.vx * dt
            s.y += s.vy * dt
            self._wrap(s)
            if fire and s.cool <= 0:
                s.cool = cfg.fire_cooldown
                cx, cy = math.cos(s.a), math.sin(s.a)
                self.bullets.append(Bullet(
                    s.x + cx * cfg.ship_r, s.y + cy * cfg.ship_r,
                    s.vx + cx * cfg.bullet_speed, s.vy + cy * cfg.bullet_speed,
                    cfg.bullet_life, s.team, s,
                ))
                ev.shots[s.id] += 1

        self._ship_collisions(ev)
        self._bullets(ev)
        self._asteroids(ev)
        self.time += dt
        return ev

    def _ship_collisions(self, ev: Events) -> None:
        r2 = self.cfg.ship_r * 1.6
        ships = self.ships
        for i, a in enumerate(ships):
            if not a.alive:
                continue
            for b in ships[i + 1:]:
                if not b.alive:
                    continue
                dx, dy = self.wdx(a.x, b.x), self.wdy(a.y, b.y)
                d = math.hypot(dx, dy)
                if 0 < d < r2:
                    nx, ny = dx / d, dy / d
                    push = (r2 - d) / 2
                    a.x -= nx * push; a.y -= ny * push
                    b.x += nx * push; b.y += ny * push
                    self._wrap(a); self._wrap(b)
                    rv = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                    if rv < 0:
                        a.vx += rv * nx; a.vy += rv * ny
                        b.vx -= rv * nx; b.vy -= rv * ny
                    if a.team != b.team and -rv > self.cfg.ram_speed:
                        self._hit(a, b, ev)
                        self._hit(b, a, ev)

    def _bullets(self, ev: Events) -> None:
        dt = self.cfg.dt
        r2 = self.cfg.ship_r ** 2
        keep = []
        for bl in self.bullets:
            bl.x += bl.vx * dt
            bl.y += bl.vy * dt
            bl.life -= dt
            self._wrap(bl)
            dead = bl.life <= 0
            if not dead:
                for s in self.ships:
                    if not s.alive or s.team == bl.team:
                        continue
                    dx, dy = self.wdx(bl.x, s.x), self.wdy(bl.y, s.y)
                    if dx * dx + dy * dy < r2:
                        self._hit(s, bl.owner, ev)
                        dead = True
                        break
            if not dead:
                for k in range(len(self.asteroids) - 1, -1, -1):
                    a = self.asteroids[k]
                    dx, dy = self.wdx(bl.x, a.x), self.wdy(bl.y, a.y)
                    if dx * dx + dy * dy < a.r * a.r:
                        dead = True
                        a.hp -= 1
                        if a.hp <= 0:
                            self._break_asteroid(k)
                        break
            if not dead:
                keep.append(bl)
        self.bullets = keep

    def _asteroids(self, ev: Events) -> None:
        cfg = self.cfg
        dt = cfg.dt
        for a in self.asteroids:
            a.x += a.vx * dt
            a.y += a.vy * dt
            a.a += a.spin * dt
            self._wrap(a)
        for s in self.ships:
            if not s.alive:
                continue
            for k in range(len(self.asteroids) - 1, -1, -1):
                a = self.asteroids[k]
                dx, dy = self.wdx(a.x, s.x), self.wdy(a.y, s.y)
                d = math.hypot(dx, dy)
                if d < a.r + cfg.ship_r * 0.7:
                    if s.invuln <= 0:
                        self._kill(s, None, ev)
                        self._break_asteroid(k)
                    else:  # bounce off while invulnerable
                        d = d or 1.0
                        s.vx, s.vy = dx / d * 120, dy / d * 120
                    break
        if cfg.asteroids and len(self.asteroids) < cfg.min_asteroids:
            if self.rng.random() < 0.5:
                x, y = 0.0, self.rand(0, cfg.height)
            else:
                x, y = self.rand(0, cfg.width), 0.0
            self.asteroids.append(self._make_asteroid(x, y, 3))
