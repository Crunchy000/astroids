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


def _threats(sim: Simulation, s: Ship) -> tuple[float, float, float]:
    """Dodge direction (ax, ay) away from enemy bullets heading at ``s`` and nearby asteroids, and its urgency."""
    cfg = sim.cfg
    ww, wh = cfg.width, cfg.height
    hw, hh = ww / 2, wh / 2
    ax = ay = threat = 0.0
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
        dx, dy = (s.x - ast.x + hw) % ww - hw, (s.y - ast.y + hh) % wh - hh
        reach = 90 + ast.r + cfg.ship_r
        if dx * dx + dy * dy > reach * reach * 1.002:  # clearly out of range; hypot decides close calls
            continue
        d = math.hypot(dx, dy) - ast.r - cfg.ship_r
        if d > 90:
            continue
        w = 1 - max(d, 0) / 90
        denom = max(d + ast.r, 1e-6)
        ax += dx / denom * 400 * w
        ay += dy / denom * 400 * w
        threat += w * 1.5
    return ax, ay, threat


STYLES = ("rusher", "sniper", "zone")


def scripted_action(sim: Simulation, s: Ship, horizon: int = 1, style: str | None = None) -> Action:
    """Chase the nearest enemy, lead shots, dodge bullets and asteroids.

    ``horizon`` is how many physics ticks the action will be held for; the turn
    command is scaled so the ship doesn't overshoot its desired heading.

    ``style`` gives the bot a distinct way of fighting (used for evaluation
    gauntlets): ``"rusher"`` closes to point-blank range and never breaks off,
    ``"sniper"`` holds at 60-90% of gun range and backs away from anything
    closer, ``"zone"`` sits in the control zone and only fights what comes near.
    The ``FORMATIONS`` styles fly as a team: see ``formation_action``.
    """
    if style in FORMATIONS:
        return formation_action(sim, s, horizon, style)
    cfg = sim.cfg
    if not s.alive:
        return (0.0, False, False)
    ww, wh = cfg.width, cfg.height
    hw, hh = ww / 2, wh / 2

    best, best_d = None, math.inf
    for o in sim.ships:
        if not o.alive or o.team == s.team:
            continue
        d = math.hypot((o.x - s.x + hw) % ww - hw, (o.y - s.y + hh) % wh - hh)  # inlined wdx / wdy
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

        gun_range = cfg.bullet_speed * cfg.bullet_life
        if style == "rusher":  # straight at them, no breaking off
            desired = aim
            thrust = facing and dist > 60
        elif style == "sniper":  # keep them at long range
            if dist < 0.6 * gun_range:
                desired = math.atan2(-dy, -dx)
                thrust = abs(wrap_angle(desired - s.a)) < 0.8
            else:
                desired = aim
                thrust = facing and dist > 0.9 * gun_range
    else:
        s.wander += sim.rand(-1, 1) * cfg.dt * horizon
        desired = s.a + s.wander

    # Control zone: with no enemy close by, go and hold the centre.
    engage = 250 if style == "zone" else 380
    if cfg.zone and (best is None or best_d > engage):
        zx, zy = sim.wdx(s.x, cfg.width / 2), sim.wdy(s.y, cfg.height / 2)
        zd = math.hypot(zx, zy)
        if zd > cfg.zone_radius * 0.5:
            desired = math.atan2(zy, zx)
            thrust = abs(wrap_angle(desired - s.a)) < 0.5 and math.hypot(s.vx, s.vy) < 150
        elif style == "zone":  # inside: hold position, keep aiming at whoever is nearest
            thrust = False
            if best is not None:
                desired = aim
        fire = fire and best is not None

    # Dodge enemy bullets heading our way and nearby asteroids.
    ax, ay, threat = _threats(sim, s)
    if threat > 0.35:
        desired = math.atan2(ay, ax)
        thrust = abs(wrap_angle(desired - s.a)) < 1.0
        fire = fire and threat < 1

    diff = wrap_angle(desired - s.a)
    turn = max(-1.0, min(1.0, diff / (cfg.turn_rate * cfg.dt * horizon)))
    return (turn, thrust, fire)


# Team formations: each ship flies to a slot in a shape centred on its team and
# facing the objective (the enemy, or the control zone while the enemy is far).
FORMATIONS = ("wedge", "line", "falsegap", "echelon", "pincer", "hammer", "circle", "retreat")
SPACING = 70.0  # px between neighbouring slots


def _slot(style: str, k: int, n: int, r: float, t: float) -> tuple[float, float]:
    """(forward, left) offset of slot ``k`` of ``n`` from the formation anchor.

    ``r`` is the nearest enemy's distance from the team's centre in gun ranges and
    ``t`` the battle time, for the shapes that change as the fight develops.
    """
    c = k - (n - 1) / 2  # lateral index: -2..2 for five ships
    if style == "wedge":  # leader in front, the rest swept back on both sides
        return -abs(c) * SPACING * 0.9, c * SPACING * 0.8
    if style in ("line", "retreat"):  # shoulder to shoulder
        return 0.0, c * SPACING
    if style == "echelon":  # a diagonal: the left leads, the right is held back (a refused flank)
        return -c * SPACING * 0.7, c * SPACING * 0.8
    if style == "falsegap":
        # A line with a hole in the middle. When the enemy comes within range the
        # centre falls back to draw them into the hole, and the flanks push forward
        # and close in, catching them in a crossfire.
        lure = r < 1.1
        if abs(c) < 1.5:
            side = 0.0 if c == 0 else math.copysign(abs(c) + 0.5, c)
            return (-1.8 * SPACING if lure else -0.6 * SPACING), side * SPACING
        side = math.copysign(abs(c) + 0.5, c)
        return (1.3 * SPACING, side * SPACING * 0.7) if lure else (0.0, side * SPACING)
    if style == "pincer":
        # Double envelopment: the centre holds while two horns swing wide and close
        # round both sides of the enemy once they are in reach.
        if abs(c) < 0.5:
            return -0.5 * SPACING, 0.0
        reach = min(r, 1.6) * SPACING * 2.2 if r < 1.4 else 0.0
        return reach * abs(c) / 2, math.copysign(1.2 + abs(c) * 0.6, c) * SPACING
    if style == "hammer":
        # Hammer and anvil: three ships hold a line (the anvil) while the other two
        # (the hammer) swing out to the right and strike the enemy's flank.
        if k < n - 2 or n < 3:
            return 0.0, (k - (min(n, n - 2) - 1) / 2) * SPACING + SPACING * 0.5
        side = (2.6 + (k - (n - 2)) * 0.7) * SPACING
        return (min(r, 1.5) * SPACING * 2.4 if r < 1.6 else -SPACING), -side
    if style == "circle":
        # Cantabrian circle: the team rides a rotating ring, so every ship keeps
        # moving and passes the front in turn, firing as it goes.
        a = 2 * math.pi * k / n + 0.9 * t
        return 120.0 * math.cos(a), 120.0 * math.sin(a)
    raise ValueError(f"unknown formation {style!r}")


def formation_action(sim: Simulation, s: Ship, horizon: int = 1, style: str = "line") -> Action:
    """Hold a slot in the team's ``style`` formation, fire at whatever is in range, dodge threats."""
    cfg = sim.cfg
    if not s.alive:
        return (0.0, False, False)
    mates = [o for o in sim.ships if o.alive and o.team == s.team]
    enemies = [o for o in sim.ships if o.alive and o.team != s.team]
    ref = mates[0]  # every ship measures from the same teammate, so all agree on the shape
    cx = sum(sim.wdx(ref.x, o.x) for o in mates) / len(mates) + ref.x
    cy = sum(sim.wdy(ref.y, o.y) for o in mates) / len(mates) + ref.y
    gun_range = cfg.bullet_speed * cfg.bullet_life

    near = min((math.hypot(sim.wdx(cx, o.x), sim.wdy(cy, o.y)) for o in enemies), default=math.inf)
    if enemies and (not cfg.zone or near < 1.5 * gun_range):
        ox = sum(sim.wdx(cx, o.x) for o in enemies) / len(enemies)
        oy = sum(sim.wdy(cy, o.y) for o in enemies) / len(enemies)
    elif cfg.zone:
        ox, oy = sim.wdx(cx, cfg.width / 2), sim.wdy(cy, cfg.height / 2)
    else:
        ox, oy = math.cos(s.a), math.sin(s.a)
    heading = math.atan2(oy, ox)
    fx, fy = math.cos(heading), math.sin(heading)
    r = near / gun_range
    lure = style == "falsegap" and r < 1.1
    cruise = min(math.hypot(ox, oy), 90.0)  # the formation's own speed toward the objective
    # Feigned retreat: when the enemy comes close, turn and run to draw them out of
    # shape, then turn on them once they are inside 55% of gun range. The bots keep
    # no memory, so the feint runs on a clock: for 3 s in every 10 at most.
    fleeing = style == "retreat" and 0.55 < r < 1.1 and sim.time % 10.0 < 3.0
    if fleeing:
        cruise = -110.0

    # Slots go left to right in the order the ships already stand, so nobody crosses over.
    if style == "circle":  # ring slots go round in the order the ships stand on the ring
        order = sorted(mates, key=lambda o: (math.atan2(sim.wdx(cx, o.x) * -fy + sim.wdy(cy, o.y) * fx,
                                                        sim.wdx(cx, o.x) * fx + sim.wdy(cy, o.y) * fy)
                                             - 0.9 * sim.time) % (2 * math.pi))
    else:
        order = sorted(mates, key=lambda o: (-sim.wdx(cx, o.x) * fy + sim.wdy(cy, o.y) * fx, o.id))
    slots = [_slot(style, k, len(mates), r, sim.time) for k in range(len(mates))]
    mf = sum(f for f, _ in slots) / len(slots)  # centre the shape on the centre of mass
    fwd, left = slots[order.index(s)]
    fwd -= mf
    tx = cx + fx * fwd - fy * left
    ty = cy + fy * fwd + fx * left
    sx, sy = sim.wdx(s.x, tx), sim.wdy(s.y, ty)
    sd = math.hypot(sx, sy)
    # Station keeping: aim for the formation's cruise velocity plus a pull toward
    # the slot, and thrust along the velocity error (turning round to brake).
    px, py = 1.5 * sx, 1.5 * sy
    pn = math.hypot(px, py)
    if pn > 150.0:
        px, py = px * 150.0 / pn, py * 150.0 / pn
    vx, vy = fx * cruise + px, fy * cruise + py
    ex, ey = vx - s.vx, vy - s.vy
    to_slot = math.atan2(ey, ex)
    need = math.hypot(ex, ey) > 25

    best = min(enemies, key=lambda o: math.hypot(sim.wdx(s.x, o.x), sim.wdy(s.y, o.y)), default=None)
    fire = False
    if best is not None:
        aim, t, dx, dy = lead(sim, s, best)
        dist = math.hypot(dx, dy)
        fire = abs(wrap_angle(aim - s.a)) < 0.12 and dist < gun_range * 0.9 and t < cfg.bullet_life
    # A false-gap centre ship falls back while the enemy closes in; once they are
    # inside 60% of gun range the trap is sprung and everyone fights.
    feint = (lure and near > 0.6 * gun_range and fwd < -SPACING and sd > 60) or fleeing
    if best is not None and dist < gun_range and not feint:
        # A target in range: turn and shoot, thrusting back toward the slot when facing it.
        desired = aim
        thrust = need and abs(wrap_angle(to_slot - s.a)) < 0.6
    elif need:
        desired = to_slot
        thrust = abs(wrap_angle(desired - s.a)) < 0.5
    else:
        desired = aim if best is not None and dist < gun_range * 1.2 else heading
        thrust = False

    ax, ay, threat = _threats(sim, s)
    if threat > 0.35:
        desired = math.atan2(ay, ax)
        thrust = abs(wrap_angle(desired - s.a)) < 1.0
        fire = fire and threat < 1

    diff = wrap_angle(desired - s.a)
    turn = max(-1.0, min(1.0, diff / (cfg.turn_rate * cfg.dt * horizon)))
    return (turn, thrust, fire)
