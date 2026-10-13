"""The JAX port must play exactly like the official Python forward model (vendored in entry/)."""
import math
import pathlib
import random
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entry"
sys.path.insert(0, str(ENTRY))
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import Action, GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402

from pwjax import baselines, env  # noqa: E402
from pwjax.maps import N_MAX  # noqa: E402

OWNER = {Player.Neutral: 0, Player.Player1: 1, Player.Player2: 2}


def to_state(gs, params) -> env.State:
    n = len(gs.planets)
    a = {k: np.zeros(N_MAX) for k in ("ships", "growth", "radius", "tr_ships")}
    i32 = {k: np.zeros(N_MAX, np.int32) for k in ("owner", "tr_owner", "tr_dest")}
    pos, tr_pos, tr_vel = np.zeros((N_MAX, 2)), np.zeros((N_MAX, 2)), np.zeros((N_MAX, 2))
    valid, tr_on = np.zeros(N_MAX, bool), np.zeros(N_MAX, bool)
    for i, p in enumerate(gs.planets):
        valid[i] = True
        i32["owner"][i] = OWNER[p.owner]
        a["ships"][i], a["growth"][i], a["radius"][i] = p.n_ships, p.growth_rate, p.radius
        pos[i] = p.position.x, p.position.y
        t = p.transporter
        if t is not None:
            tr_on[i] = True
            i32["tr_owner"][i], i32["tr_dest"][i], a["tr_ships"][i] = OWNER[t.owner], t.destination_index, t.n_ships
            tr_pos[i], tr_vel[i] = (t.s.x, t.s.y), (t.v.x, t.v.y)
    assert n <= N_MAX
    return env.State(valid=jnp.array(valid), owner=jnp.array(i32["owner"]), ships=jnp.array(a["ships"]),
                     pos=jnp.array(pos), growth=jnp.array(a["growth"]), radius=jnp.array(a["radius"]),
                     tr_on=jnp.array(tr_on), tr_owner=jnp.array(i32["tr_owner"]), tr_dest=jnp.array(i32["tr_dest"]),
                     tr_ships=jnp.array(a["tr_ships"]), tr_pos=jnp.array(tr_pos), tr_vel=jnp.array(tr_vel),
                     tick=jnp.int32(gs.game_tick), speed=jnp.float64(params.transporter_speed),
                     max_ticks=jnp.int32(params.max_ticks))


def fuzz_action(rng: random.Random, gs, player) -> Action:
    """Mostly sensible launches, plus no-ops and every kind of illegal action."""
    n = len(gs.planets)
    u = rng.random()
    if u < 0.25:
        return Action.do_nothing()
    if u < 0.75:
        free = [p for p in gs.planets if p.owner == player and p.transporter is None]
        if not free:
            return Action.do_nothing()
        src = rng.choice(free)
        num = src.n_ships * rng.choice([0.2, 0.5, 0.9, 1.0, rng.random()])
        return Action(player_id=player, source_planet_id=src.id, destination_planet_id=rng.randrange(n), num_ships=num)
    src, dst = rng.randint(-2, n + 1), rng.randint(-2, n + 1)
    ships = gs.planets[src].n_ships if 0 <= src < n else 5.0
    num = rng.choice([0.0, -1.0, math.nan, math.inf, ships * 1.5, ships, ships * 0.5])
    return Action(player_id=player, source_planet_id=src, destination_planet_id=dst, num_ships=num)


def as_tuple(a: Action):
    if a is Action.DO_NOTHING:
        return jnp.int32(-1), jnp.int32(-1), jnp.float64(0.0)
    return jnp.int32(a.source_planet_id), jnp.int32(a.destination_planet_id), jnp.float64(a.num_ships)


def check_same(gs, s: env.State):
    ref = to_state(gs, GameParams())
    np.testing.assert_array_equal(np.asarray(s.owner), np.asarray(ref.owner))
    np.testing.assert_allclose(np.asarray(s.ships), np.asarray(ref.ships), rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(s.tr_on), np.asarray(ref.tr_on))
    on = np.asarray(ref.tr_on)
    for k in ("tr_owner", "tr_dest", "tr_ships", "tr_pos", "tr_vel"):
        np.testing.assert_allclose(np.asarray(getattr(s, k))[on], np.asarray(getattr(ref, k))[on], atol=1e-9)
    assert int(s.tick) == gs.game_tick


step = jax.jit(env.step)
terminal = jax.jit(env.is_terminal)
leader = jax.jit(env.leader)
greedy = jax.jit(baselines.greedy)


@pytest.mark.parametrize("seed", range(6))
def test_forward_model_and_greedy_match_official(seed):
    rng = random.Random(seed)
    random.seed(seed)
    n_planets = rng.choice([10, 16, 20, 30])
    params = GameParams(num_planets=n_planets, initial_neutral_ratio=rng.uniform(0.25, 0.5),
                        min_growth_rate=0.05, max_growth_rate=0.2,
                        transporter_speed=rng.uniform(2.0, 5.0), max_ticks=rng.choice([300, 600]))
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    s = to_state(gs, params)
    agents = {}
    for p in (Player.Player1, Player.Player2):
        agents[p] = GreedyHeuristicAgent()
        agents[p].prepare_to_play_as(p, params)
    while True:
        check_same(gs, s)
        assert bool(terminal(s)) == model.is_terminal()
        if model.is_terminal():
            break
        actions = {}
        for p in (Player.Player1, Player.Player2):
            want = agents[p].get_action(gs)
            got = greedy(s, OWNER[p])
            assert tuple(np.asarray(x).item() for x in got) == pytest.approx(
                tuple(float(x) for x in as_tuple(want)))
            # mix the greedy agent with fuzzed actions so every rule gets exercised
            actions[p] = want if rng.random() < 0.3 else fuzz_action(rng, gs, p)
        s = step(s, as_tuple(actions[Player.Player1]), as_tuple(actions[Player.Player2]))
        model.step(actions)
    assert int(leader(s)) == OWNER[model.get_leader()]


def test_rush_is_greedy_that_sends_everything():
    random.seed(3)
    params = GameParams(num_planets=16, max_ticks=300)
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    s = to_state(gs, params)
    agent = {p: GreedyHeuristicAgent() for p in (Player.Player1, Player.Player2)}
    for p, a in agent.items():
        a.prepare_to_play_as(p, params)
    launched = 0
    rush = jax.jit(baselines.rush)
    while not model.is_terminal():
        for p in agent:
            gsrc, gdst, _ = (np.asarray(x).item() for x in greedy(s, OWNER[p]))
            src, dst, num = (np.asarray(x).item() for x in rush(s, OWNER[p]))
            assert (src, dst) == (gsrc, gdst)
            assert num == (float(s.ships[src]) if src >= 0 else 0.0)
            launched += src >= 0
        actions = {p: agent[p].get_action(gs) for p in agent}
        s = step(s, as_tuple(actions[Player.Player1]), as_tuple(actions[Player.Player2]))
        model.step(actions)
    assert launched > 0
