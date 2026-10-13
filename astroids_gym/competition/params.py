"""Match settings drawn at random, as Planet Wars RTS draws its game parameters.

Every match uses the competitive rules (``Config.competitive()``) except for
the settings below, each drawn uniformly at random per match. The point is to
reward teams that cope with any version of the game, not just the one they
trained on.

* ``asteroid_count``: 2 to 8 asteroids at the start (``min_asteroids`` is half of it, at least 1)
* ``team_lives``: 15 to 25 respawns per team
* ``zone_radius``: a control zone of 100 to 160 px
* ``bullet_life``: 0.7 to 0.9 s, so gun range is 336 to 432 px
* ``flight``: half the matches use the dogfight flight physics (``Config.dogfight()``:
  the velocity follows the nose, the turn rate depends on speed, turning bleeds speed)

Battles last 60 s (1,800 decisions at 30 per second) and teams are five ships.
"""

from __future__ import annotations

import numpy as np

from ..sim import Config
from .state import DECISION_MS, FRAME_SKIP, make_params

VARIANTS = {
    "asteroid_count": (2, 8),
    "team_lives": (15, 25),
    "zone_radius": (100.0, 160.0),
    "bullet_life": (0.7, 0.9),
    "flight": (False, True),
}


def random_params(seed: int | None = None, decision_ms: float = DECISION_MS, fixed: bool = False) -> dict:
    """Match ``params`` with the ``VARIANTS`` drawn from ``seed`` (``fixed``: the plain competitive rules)."""
    if fixed:
        return make_params(Config.competitive(), FRAME_SKIP, decision_ms)
    rng = np.random.default_rng(seed)
    count = int(rng.integers(VARIANTS["asteroid_count"][0], VARIANTS["asteroid_count"][1] + 1))
    over = dict(
        asteroid_count=count,
        min_asteroids=max(1, count // 2),
        team_lives=int(rng.integers(VARIANTS["team_lives"][0], VARIANTS["team_lives"][1] + 1)),
        zone_radius=round(float(rng.uniform(*VARIANTS["zone_radius"])), 1),
        bullet_life=round(float(rng.uniform(*VARIANTS["bullet_life"])), 3),
    )
    flight = bool(rng.random() < 0.5)
    cfg = Config.dogfight(**over) if flight else Config.competitive(**over)
    return make_params(cfg, FRAME_SKIP, decision_ms)
