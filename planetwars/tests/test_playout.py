"""The entry's array game step (playout.step) must follow the official engine tick for tick, and the play-out search
must keep to its deadline."""
import pathlib
import random
import sys
import time

import numpy as np

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entry"
sys.path.insert(0, str(ENTRY))
import agent as entry_agent  # noqa: E402
import planner  # noqa: E402
import playout  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402


def as_action(a):
    if a.source_planet_id < 0 or a.num_ships <= 0:
        return None
    return a.source_planet_id, a.destination_planet_id, a.num_ships


def test_step_matches_the_engine():
    for seed in range(3):
        random.seed(seed)
        params = GameParams(num_planets=20 + 2 * seed, max_ticks=500, min_growth_rate=0.05, max_growth_rate=0.2,
                            transporter_speed=2.5 + seed, initial_neutral_ratio=0.3)
        gs = GameStateFactory(params).create_game()
        model = ForwardModel(gs, params)
        bots = {p: GreedyHeuristicAgent() for p in (Player.Player1, Player.Player2)}
        for p, a in bots.items():
            a.prepare_to_play_as(p, params)
        s = entry_agent.arrays(gs, params.transporter_speed, params.max_ticks)
        while not model.is_terminal() and gs.game_tick < 400:
            acts = {p: bots[p].get_action(gs) for p in bots}
            s = playout.step(s, as_action(acts[Player.Player1]), as_action(acts[Player.Player2]))
            model.step(acts)
            got = entry_agent.arrays(gs, params.transporter_speed, params.max_ticks)
            np.testing.assert_array_equal(s.owner, got.owner)
            np.testing.assert_allclose(s.ships, got.ships, rtol=1e-9, atol=1e-9)
            np.testing.assert_array_equal(s.tr_on, got.tr_on)
            on = got.tr_on
            np.testing.assert_allclose(s.tr_pos[on], got.tr_pos[on], rtol=1e-9, atol=1e-9)
            assert s.tick == got.tick


def test_search_keeps_to_its_deadline():
    random.seed(7)
    params = GameParams(num_planets=30, max_ticks=2000, min_growth_rate=0.05, max_growth_rate=0.2, transporter_speed=3.0,
                        initial_neutral_ratio=0.3)
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    agents = {p: planner.PlannerAgent() for p in (Player.Player1, Player.Player2)}
    for p, a in agents.items():
        a.prepare_to_play_as(p, params)
    params_v, heads = agents[Player.Player1].value
    c = None
    while c is None or gs.game_tick < 150:  # mid-game, with something for player 1 to launch
        model.step({p: agents[p].get_action(gs) for p in agents})
        s = entry_agent.arrays(gs, params.transporter_speed, params.max_ticks)
        t0 = time.perf_counter()
        c = planner.candidates(s, 1, extras=("spare",))
        hint = time.perf_counter() - t0  # as plan_ranked passes it: a planner call costs at least this
    assert not model.is_terminal()
    order = np.argsort(-c["score"])[:4]
    opts = [(int(c["ci"][k]), int(c["cj"][k]), float(c["cn"][k])) for k in order] + [None]
    t0 = time.perf_counter()  # what one planner call of a play-out costs on this machine (busy or not)
    planner.plan_ranked(s, 1, params_v, heads, 90, planner.FRACTIONS, 16, 0.5, 1e9, ("spare",), look="", pairs=0, pair_kinds="")
    call = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = playout.search(s, 1, params_v, heads, opts, 200, t0 + 0.08, hint=max(hint, call))
    took = time.perf_counter() - t0
    assert r is not None and 0 <= r[0] < len(opts) and r[2] >= 1
    assert took < 0.08 + 0.5 * call + 0.03, (took, call)


def test_helper_processes_play_the_same_as_one():
    """Shared out among helper processes, the options are played and valued exactly as in one process."""
    random.seed(11)
    params = GameParams(num_planets=16, max_ticks=2000, min_growth_rate=0.05, max_growth_rate=0.2, transporter_speed=3.0,
                        initial_neutral_ratio=0.3)
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    agents = {p: planner.PlannerAgent(search="") for p in (Player.Player1, Player.Player2)}
    for p, a in agents.items():
        a.prepare_to_play_as(p, params)
    c = None
    while c is None or gs.game_tick < 60:  # with something for player 1 to launch
        model.step({p: agents[p].get_action(gs) for p in agents})
        s = entry_agent.arrays(gs, params.transporter_speed, params.max_ticks)
        c = planner.candidates(s, 1, extras=("spare",))
    params_v, heads = agents[Player.Player1].value
    order = np.argsort(-c["score"])[:3]
    opts = [(int(c["ci"][k]), int(c["cj"][k]), float(c["cn"][k])) for k in order] + [None]
    far = time.perf_counter() + 60
    one = playout.search(s, 1, params_v, heads, opts, 6, far, procs=1)
    many = playout.search(s, 1, params_v, heads, opts, 6, far, procs=3)
    assert one[2] == many[2] == 6
    np.testing.assert_allclose(one[1], many[1], rtol=1e-6)
    assert one[0] == many[0]
