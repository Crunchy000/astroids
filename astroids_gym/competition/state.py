"""Game parameters and game states as JSON, and back.

Every match is described by one ``params`` dict and, at each decision step,
one ``state`` dict. Both are plain JSON, so an agent can be written in any
language. Python agents can rebuild a :class:`~astroids_gym.sim.Simulation`
from a state with :func:`restore` and use it as a forward model: step it with
candidate actions to see what happens.

``params`` holds every :class:`~astroids_gym.sim.Config` field plus
``frame_skip`` (physics ticks per decision) and ``decision_ms`` (the time
budget per decision in real-time matches).

``state`` (positions in pixels, velocities in px/s, angles in radians,
``x`` right and ``y`` down, the world wraps at the edges)::

    {"tick": 120, "time": 2.0,
     "score": {"blue": 1, "red": 0},          # kills
     "points": {"blue": 0.5, "red": 0.0},     # control-zone points
     "lives": {"blue": 19, "red": 20},        # respawns left
     "reinforcements": {"blue": 0, "red": 0}, # respawns earned from zone points so far
     "ships": [{"id": 0, "team": "blue", "alive": true, "x": ..., "y": ..., "vx": ..., "vy": ...,
                "angle": ..., "hp": 3, "cooldown": 0.1, "invuln": 0.0, "respawn": 0.0,
                "heat": 0.25, "overheated": false, "thrusting": true, "kills": 1, "deaths": 0}, ...],
     "bullets": [{"x": ..., "y": ..., "vx": ..., "vy": ..., "life": 0.4, "team": "red", "owner": 7}, ...],
     "asteroids": [{"x": ..., "y": ..., "vx": ..., "vy": ..., "r": 48.0, "size": 3, "hp": 3,
                    "angle": ..., "spin": ...}, ...]}

Ship ``i`` is on blue for ``i < team_size`` and on red otherwise.
"""

from __future__ import annotations

import dataclasses
import math

from ..sim import Asteroid, Bullet, Config, Ship, Simulation

TEAMS = ("blue", "red")
FRAME_SKIP = 2  # physics ticks per decision: 60 Hz physics, 30 decisions per second
DECISION_MS = 33.0  # one decision's worth of game time at real-time speed

_CONFIG_FIELDS = tuple(f.name for f in dataclasses.fields(Config))


def team_index(player: str) -> int:
    """``"blue"`` -> 0, ``"red"`` -> 1."""
    return TEAMS.index(player)


def make_params(cfg: Config | None = None, frame_skip: int = FRAME_SKIP,
                decision_ms: float = DECISION_MS) -> dict:
    """The ``params`` dict of a match played with ``cfg`` (default: the competitive rules)."""
    cfg = cfg or Config.competitive()
    return {**dataclasses.asdict(cfg), "frame_skip": frame_skip, "decision_ms": decision_ms}


def config_from_params(params: dict) -> Config:
    """The :class:`Config` in ``params`` (unknown keys are ignored)."""
    return Config(**{k: params[k] for k in _CONFIG_FIELDS if k in params})


def max_steps(params: dict) -> int:
    """Decision steps in a full-length battle."""
    cfg = config_from_params(params)
    return int(round(cfg.battle_seconds / (cfg.dt * params.get("frame_skip", FRAME_SKIP))))


def _per_team(values) -> dict:
    return {TEAMS[0]: values[0], TEAMS[1]: values[1]}


def snapshot(sim: Simulation) -> dict:
    """The full game state as a JSON-ready dict."""
    cfg = sim.cfg
    return {
        "tick": int(round(sim.time / cfg.dt)),
        "time": sim.time,
        "score": _per_team(sim.score),
        "points": _per_team(sim.points),
        "lives": _per_team(sim.lives),
        "reinforcements": _per_team(sim.reinforcements),
        "ships": [{
            "id": s.id, "team": TEAMS[s.team], "alive": s.alive, "x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy,
            "angle": s.a, "hp": s.hp, "cooldown": s.cool, "invuln": s.invuln, "respawn": s.respawn,
            "heat": s.heat, "overheated": s.overheated, "thrusting": s.thrusting,
            "kills": s.kills, "deaths": s.deaths,
        } for s in sim.ships],
        "bullets": [{"x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy, "life": b.life, "team": TEAMS[b.team],
                     "owner": b.owner.id} for b in sim.bullets],
        "asteroids": [{"x": a.x, "y": a.y, "vx": a.vx, "vy": a.vy, "r": a.r, "size": a.size, "hp": a.hp,
                       "angle": a.a, "spin": a.spin} for a in sim.asteroids],
    }


def restore(state: dict, params: dict | Config, seed: int | None = None) -> Simulation:
    """A :class:`Simulation` in exactly ``state``, to read or to step as a forward model.

    The restored simulation has its own random generator (``seed``), so random
    events (respawn points, asteroid fragments, new asteroids) can differ from
    the real match. Everything else steps identically. The scripted bot's
    per-ship personality is not part of the state: restored ships get neutral
    values (see ``agents.BotAgent`` for a bot that keeps its own).
    """
    cfg = params if isinstance(params, Config) else config_from_params(params)
    sim = Simulation(cfg, seed=seed)
    sim.time = float(state["time"])
    sim.score = [int(state["score"][t]) for t in TEAMS]
    sim.points = [float(state["points"][t]) for t in TEAMS]
    sim.lives = [int(state["lives"][t]) for t in TEAMS]
    sim.reinforcements = [int(state["reinforcements"][t]) for t in TEAMS]
    ships = []
    for d in state["ships"]:
        s = Ship(int(d["id"]), team_index(d["team"]))
        s.alive, s.x, s.y, s.vx, s.vy = bool(d["alive"]), float(d["x"]), float(d["y"]), float(d["vx"]), float(d["vy"])
        s.a, s.hp, s.cool, s.invuln = float(d["angle"]), int(d["hp"]), float(d["cooldown"]), float(d["invuln"])
        s.respawn, s.heat, s.overheated = float(d["respawn"]), float(d["heat"]), bool(d["overheated"])
        s.thrusting, s.kills, s.deaths = bool(d["thrusting"]), int(d["kills"]), int(d["deaths"])
        s.wander, s.aggro, s.pref_dist, s.target = 0.0, 0.8, 200.0, None
        ships.append(s)
    sim.ships = ships
    sim.bullets = [Bullet(float(b["x"]), float(b["y"]), float(b["vx"]), float(b["vy"]), float(b["life"]),
                          team_index(b["team"]), ships[int(b["owner"])]) for b in state["bullets"]]
    asteroids = []
    for d in state["asteroids"]:
        a = Asteroid()
        a.x, a.y, a.vx, a.vy = float(d["x"]), float(d["y"]), float(d["vx"]), float(d["vy"])
        a.r, a.size, a.hp = float(d["r"]), int(d["size"]), int(d["hp"])
        a.a, a.spin = float(d["angle"]), float(d["spin"])
        a.shape = [1.0] * 9  # outline is cosmetic and not part of the state
        asteroids.append(a)
    sim.asteroids = asteroids
    return sim


# ---- actions ------------------------------------------------------------------------
DO_NOTHING: dict = {"ships": []}


def ship_command(ship_id: int, turn: float = 0.0, thrust: bool = False, fire: bool = False) -> dict:
    """One ship's part of an action: ``turn`` in [-1, 1] (fraction of the turn rate, positive =
    clockwise on screen), ``thrust`` and ``fire`` on or off, held for the whole decision step."""
    return {"id": int(ship_id), "turn": float(turn), "thrust": bool(thrust), "fire": bool(fire)}


def parse_action(action, team: int, cfg: Config) -> tuple[dict[int, tuple[float, bool, bool]], int]:
    """An agent's action -> ({ship id: (turn, thrust, fire)}, number of rejected commands).

    An action is ``{"ships": [ship_command, ...]}``. Commands for ships on the
    other team, unknown ids, repeated ids and non-finite turns are rejected;
    ships without a command coast. Anything that isn't an action is rejected
    as a whole (one failure) and all the team's ships coast.
    """
    if not isinstance(action, dict) or not isinstance(action.get("ships", None), list):
        return {}, 1
    lo, hi = team * cfg.team_size, (team + 1) * cfg.team_size
    out: dict[int, tuple[float, bool, bool]] = {}
    failed = 0
    for cmd in action["ships"]:
        try:
            sid = int(cmd["id"])
            turn = float(cmd.get("turn", 0.0))
            if not (lo <= sid < hi) or sid in out or not math.isfinite(turn):
                raise ValueError
            out[sid] = (max(-1.0, min(1.0, turn)), bool(cmd.get("thrust", False)), bool(cmd.get("fire", False)))
        except (TypeError, ValueError, KeyError, AttributeError):
            failed += 1
    return out, failed
