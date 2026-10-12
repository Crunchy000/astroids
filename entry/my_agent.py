"""An example competition entry: start here.

Each ship turns toward where its shot would hit the nearest enemy, fires when
lined up, closes to gun range, and goes to hold the control zone when no enemy
is near. Then the agent uses the forward model: it plays the plan forward a
third of a second, and any ship that would die there tries the opposite
throttle instead.

Serve it with ``python -m astroids_gym.competition serve entry.my_agent:MyAgent``
(the repository's ``Dockerfile`` does), or try it in a match with
``python -m astroids_gym.competition match entry.my_agent:MyAgent bots``.
"""

from __future__ import annotations

import math

from astroids_gym.bot import lead
from astroids_gym.competition import Player, ship_command
from astroids_gym.sim import wrap_angle

LOOKAHEAD = 20  # physics ticks (1/3 s)


class MyAgent(Player):
    def get_agent_type(self) -> str:
        return "my-agent"

    def plan(self, sim, s) -> tuple[float, bool, bool]:
        """(turn, thrust, fire) for ship ``s``."""
        cfg = sim.cfg
        gun_range = cfg.bullet_speed * cfg.bullet_life
        enemies = [o for o in sim.ships if o.alive and o.team != s.team]
        target = min(enemies, key=lambda o: sim.wdx(s.x, o.x) ** 2 + sim.wdy(s.y, o.y) ** 2, default=None)
        dist = math.inf if target is None else math.hypot(sim.wdx(s.x, target.x), sim.wdy(s.y, target.y))
        if target is not None and dist < 1.6 * gun_range:
            aim, flight_time, _, _ = lead(sim, s, target)  # where to point so a shot fired now hits
            fire = abs(wrap_angle(aim - s.a)) < 0.1 and flight_time < cfg.bullet_life and not s.overheated
            thrust = dist > 0.7 * gun_range and abs(wrap_angle(aim - s.a)) < 0.5
        else:  # nobody near: hold the control zone
            aim = math.atan2(sim.wdy(s.y, cfg.height / 2), sim.wdx(s.x, cfg.width / 2))
            fire = False
            thrust = abs(wrap_angle(aim - s.a)) < 0.5 and math.hypot(s.vx, s.vy) < 120
        turn = wrap_angle(aim - s.a) / (cfg.turn_rate * cfg.dt * self.frame_skip)  # don't overshoot
        return max(-1.0, min(1.0, turn)), thrust, fire

    def get_action(self, state: dict) -> dict:
        sim = self.simulation(state)  # the state as a Simulation: geometry helpers and a forward model
        plan = {s.id: self.plan(sim, s) for s in sim.ships if s.alive and s.id in self.ship_ids}

        # Forward model: play the plan on, the enemy coasting, and see who dies.
        future = self.simulation(state, seed=0)
        dies = set()
        for _ in range(LOOKAHEAD):
            dies |= {i for i, n in enumerate(future.step(plan).deaths) if n and i in plan}
        for i in dies:
            turn, thrust, fire = plan[i]
            plan[i] = (turn, not thrust, fire)

        return {"ships": [ship_command(i, *cmd) for i, cmd in plan.items()]}
