"""Map generation, faithful to the official Kotlin GameStateFactory and GameParamGenerator.

Maps are generated on the host with numpy and stored in a padded "bank" of arrays, so the
jitted environment can reset any number of games by indexing into it.
"""
from dataclasses import dataclass

import numpy as np

N_MAX = 32  # padded planet count (CoG 2026 maps have 10 to 30 planets)

NEUTRAL, P1, P2 = 0, 1, 2


@dataclass
class Params:
    """Mirror of games.planetwars.core.GameParams (defaults included)."""
    width: int = 640
    height: int = 480
    edge_separation: float = 25.0
    radial_separation: float = 1.5
    growth_to_radius_factor: float = 200.0
    num_planets: int = 10
    initial_neutral_ratio: float = 0.5
    max_ticks: int = 2000
    min_initial_ships_per_planet: int = 2
    max_initial_ships_per_planet: int = 20
    min_growth_rate: float = 0.02
    max_growth_rate: float = 0.1
    transporter_speed: float = 3.0


def cog2026_params(rng: np.random.Generator) -> Params:
    """GameParamGenerator.randomParams(): what the live league plays."""
    raw = int(rng.integers(10, 31))
    return Params(
        num_planets=raw if raw % 2 == 0 else raw + 1,
        initial_neutral_ratio=float(rng.uniform(0.25, 0.35)),
        min_growth_rate=0.05,
        max_growth_rate=0.2,
        transporter_speed=float(rng.uniform(2.0, 5.0)),
    )


def training_params(rng: np.random.Generator) -> Params:
    """Mostly the league's distribution, plus some default-rule maps (the submission check uses those)."""
    u = rng.random()
    if u < 0.8:
        p = cog2026_params(rng)
    else:
        p = Params(num_planets=int(rng.choice([10, 12, 16, 20, 24, 30])),
                   transporter_speed=float(rng.choice([3.0, rng.uniform(2.0, 5.0)])))
    if rng.random() < 0.1:
        p.max_ticks = 1000
    return p


def create_game(params: Params, rng: np.random.Generator, max_attempts: int = 1_000_000) -> dict:
    """GameStateFactory.createGame(): rejection-sample the left half, then mirror it."""
    half = []
    n_neutral = int(params.num_planets * params.initial_neutral_ratio) // 2
    w2 = params.width / 2
    for _ in range(max_attempts):
        if len(half) >= params.num_planets // 2:
            break
        x = float(int(rng.random() * params.width / 2))
        y = float(int(rng.random() * params.height))
        ships = rng.random() * (params.max_initial_ships_per_planet - params.min_initial_ships_per_planet) \
            + params.min_initial_ships_per_planet
        growth = rng.uniform(params.min_growth_rate, params.max_growth_rate)
        r = growth * params.growth_to_radius_factor
        e = params.edge_separation
        if x - e < r or x + e > w2 - r or y - e < r or y + e > params.height - r:
            continue
        if any(np.hypot(px - x, py - y) < params.radial_separation * (pg * params.growth_to_radius_factor + r)
               for px, py, pg, _, _ in half):
            continue
        half.append((x, y, growth, ships, NEUTRAL if len(half) < n_neutral else P1))
    else:
        raise RuntimeError("map generation did not converge")
    planets = list(half) + [(params.width - x, params.height - y, g, s, P2 if o == P1 else o)
                            for x, y, g, s, o in half]
    arr = np.array(planets, dtype=np.float64).reshape(-1, 5)
    return {
        "x": arr[:, 0], "y": arr[:, 1], "growth": arr[:, 2], "ships": arr[:, 3],
        "owner": arr[:, 4].astype(np.int32), "radius": arr[:, 2] * params.growth_to_radius_factor,
    }


def make_bank(n_maps: int, seed: int = 0, param_fn=training_params) -> dict:
    """n_maps padded maps as numpy arrays (each with its own GameParams)."""
    rng = np.random.default_rng(seed)
    bank = {
        "valid": np.zeros((n_maps, N_MAX), bool), "owner": np.zeros((n_maps, N_MAX), np.int32),
        "ships": np.zeros((n_maps, N_MAX), np.float32), "pos": np.zeros((n_maps, N_MAX, 2), np.float32),
        "growth": np.zeros((n_maps, N_MAX), np.float32), "radius": np.zeros((n_maps, N_MAX), np.float32),
        "speed": np.zeros(n_maps, np.float32), "max_ticks": np.zeros(n_maps, np.int32),
    }
    for m in range(n_maps):
        p = param_fn(rng)
        g = create_game(p, rng)
        n = len(g["x"])
        bank["valid"][m, :n] = True
        bank["owner"][m, :n] = g["owner"]
        bank["ships"][m, :n] = g["ships"]
        bank["pos"][m, :n, 0] = g["x"]
        bank["pos"][m, :n, 1] = g["y"]
        bank["growth"][m, :n] = g["growth"]
        bank["radius"][m, :n] = g["radius"]
        bank["speed"][m] = p.transporter_speed
        bank["max_ticks"][m] = p.max_ticks
    return bank
