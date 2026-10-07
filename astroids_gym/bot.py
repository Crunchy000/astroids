"""Scripted opponent: the same heuristic AI that drives the browser game."""

from __future__ import annotations

import math

from .sim import Ship, Simulation, wrap_angle

Action = tuple[float, bool, bool]


def lead(sim: Simulation, s: Ship, target: Ship) -> tuple[float, float, float, float]:
    """Where to point to hit ``target`` with a bullet fired now.

    Returns ``(world angle, bullet flight time, dx, dy)`` where (dx, dy) is the
    wrap-aware offset from ``s`` to the target.
    """
    cfg = sim.cfg
    dx, dy = sim.wdx(s.x, target.x), sim.wdy(s.y, target.y)
    rvx, rvy = target.vx - s.vx, target.vy - s.vy
    a = rvx * rvx + rvy * rvy - cfg.bullet_speed ** 2
    b = 2 * (dx * rvx + dy * rvy)
    c = dx * dx + dy * dy
    t = 0.0
    disc = b * b - 4 * a * c
    if disc >= 0 and a != 0:
        r = math.sqrt(disc)
        roots = [x for x in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if x > 0]
        t = min(roots) if roots else 0.0
    return math.atan2(dy + rvy * t, dx + rvx * t), t, dx, dy


def scripted_action(sim: Simulation, s: Ship, horizon: int = 1) -> Action:
    """Chase the nearest enemy, lead shots, dodge bullets and asteroids.

    ``horizon`` is how many physics ticks the action will be held for; the turn
    command is scaled so the ship doesn't overshoot its desired heading.
    """
    cfg = sim.cfg
    if not s.alive:
        return (0.0, False, False)

    best, best_d = None, math.inf
    for o in sim.ships:
        if not o.alive or o.team == s.team:
            continue
        d = math.hypot(sim.wdx(s.x, o.x), sim.wdy(s.y, o.y))
        if o is s.target:
            d *= 0.8  # mild target stickiness
        if d < best_d:
            best, best_d = o, d
    s.target = best

    desired, thrust, fire = s.a, False, False

    if best is not None:
        aim, t, dx, dy = lead(sim, s, best)
        dist = math.hypot(dx, dy)
        desired = aim

        err = abs(wrap_angle(aim - s.a))
        if err < 0.12 and dist < cfg.bullet_speed * cfg.bullet_life * 0.9 and t < cfg.bullet_life:
            fire = True

        facing = abs(wrap_angle(math.atan2(dy, dx) - s.a)) < 0.6
        if dist > s.pref_dist and facing and sim.rng.random() < s.aggro:
            thrust = True
        if dist < s.pref_dist * 0.5:
            # break off, then come back around
            desired = math.atan2(dy, dx) + math.pi * 0.6 * (1 if s.wander >= 0 else -1)
            thrust = abs(wrap_angle(desired - s.a)) < 0.8
    else:
        s.wander += sim.rand(-1, 1) * cfg.dt * horizon
        desired = s.a + s.wander

    # Control zone: with no enemy close by, go and hold the centre.
    if cfg.zone and (best is None or best_d > 380):
        zx, zy = sim.wdx(s.x, cfg.width / 2), sim.wdy(s.y, cfg.height / 2)
        zd = math.hypot(zx, zy)
        if zd > cfg.zone_radius * 0.5:
            desired = math.atan2(zy, zx)
            thrust = abs(wrap_angle(desired - s.a)) < 0.5 and math.hypot(s.vx, s.vy) < 150
        fire = fire and best is not None

    # Dodge enemy bullets heading our way and nearby asteroids.
    ax = ay = threat = 0.0
    ww, wh = cfg.width, cfg.height
    hw, hh = ww / 2, wh / 2
    for bl in sim.bullets:
        if bl.team == s.team:
            continue
        dx = (s.x - bl.x + hw) % ww - hw  # inlined sim.wdx / sim.wdy (hot loop)
        dy = (s.y - bl.y + hh) % wh - hh
        d2 = dx * dx + dy * dy
        if d2 > 140 * 140 or dx * bl.vx + dy * bl.vy <= 0:
            continue
        sgn = 1 if dx * bl.vy - dy * bl.vx > 0 else -1
        w = 1 - math.sqrt(d2) / 140
        ax += -bl.vy * sgn * w
        ay += bl.vx * sgn * w
        threat += w
    for ast in sim.asteroids:
        dx, dy = sim.wdx(ast.x, s.x), sim.wdy(ast.y, s.y)
        d = math.hypot(dx, dy) - ast.r - cfg.ship_r
        if d > 90:
            continue
        w = 1 - max(d, 0) / 90
        denom = max(d + ast.r, 1e-6)
        ax += dx / denom * 400 * w
        ay += dy / denom * 400 * w
        threat += w * 1.5
    if threat > 0.35:
        desired = math.atan2(ay, ax)
        thrust = abs(wrap_angle(desired - s.a)) < 1.0
        fire = fire and threat < 1

    diff = wrap_angle(desired - s.a)
    turn = max(-1.0, min(1.0, diff / (cfg.turn_rate * cfg.dt * horizon)))
    return (turn, thrust, fire)
