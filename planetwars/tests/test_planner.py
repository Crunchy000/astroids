"""The all-moves planner must predict, for every launch, exactly what the official engine does when nobody else acts."""
import copy
import pathlib
import random
import sys

import numpy as np

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entry"
sys.path.insert(0, str(ENTRY))
import agent as entry_agent  # noqa: E402
import planner  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import Action, GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402


def test_projection_matches_the_engine():
    random.seed(4)
    params = GameParams(num_planets=22, max_ticks=600, min_growth_rate=0.05, max_growth_rate=0.2,
                        transporter_speed=3.3, initial_neutral_ratio=0.3)
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    bots = {p: GreedyHeuristicAgent() for p in (Player.Player1, Player.Player2)}
    for p, a in bots.items():
        a.prepare_to_play_as(p, params)
    horizon, checked = 60, 0
    rng = np.random.default_rng(0)
    while not model.is_terminal() and gs.game_tick < 300:
        if gs.game_tick % 40 == 5:
            for me in (1, 2):
                s = entry_agent.arrays(gs, params.transporter_speed, params.max_ticks)
                n = len(gs.planets)
                a1, a2 = planner.arrivals(s, horizon)
                d = np.sqrt(((s.pos[:, None, :] - s.pos[None, :, :]) ** 2).sum(-1))
                bo, bs = planner.run(s.owner.copy(), s.ships.copy(), s.growth, a1, a2)
                srcs = np.nonzero((s.owner == me) & ~s.tr_on & (s.ships >= 1))[0]
                if srcs.size == 0:
                    continue
                ci = rng.choice(srcs, 8)
                cj = np.array([rng.choice([j for j in range(n) if j != i]) for i in ci])
                cn = np.maximum(1.0, np.floor(s.ships[ci] * rng.choice([0.25, 0.5, 1.0], 8)))
                ao, ash = planner.project(s, me, horizon, ci, cj, cn, a1, a2, bo, bs, d)
                for k in range(8):
                    sim = ForwardModel(copy.deepcopy(gs), params)
                    player = Player.Player1 if me == 1 else Player.Player2
                    sim.step({player: Action(player_id=player, source_planet_id=int(ci[k]), destination_planet_id=int(cj[k]),
                                             num_ships=float(cn[k])), player.opponent(): Action.do_nothing()})
                    for _ in range(horizon - 1):
                        sim.step({Player.Player1: Action.do_nothing(), Player.Player2: Action.do_nothing()})
                    got_owner = np.array([entry_agent.OWNER[p.owner] for p in sim.state.planets])
                    got_ships = np.array([p.n_ships for p in sim.state.planets])
                    np.testing.assert_array_equal(ao[k], got_owner)
                    np.testing.assert_allclose(ash[k], got_ships, rtol=1e-6, atol=1e-6)
                    checked += 1
        model.step({p: bots[p].get_action(gs) for p in bots})
    assert checked >= 40


def test_planner_returns_a_legal_move_or_waits():
    random.seed(1)
    params = GameParams(num_planets=16, max_ticks=400)
    gs = GameStateFactory(params).create_game()
    agent = planner.PlannerAgent()
    agent.prepare_to_play_as(Player.Player1, params)
    a = agent.get_action(gs)
    assert a.source_planet_id < 0 or (gs.planets[a.source_planet_id].owner == Player.Player1
                                      and 0 < a.num_ships <= gs.planets[a.source_planet_id].n_ships)


def test_entry_serves_the_planner_unless_told_otherwise(monkeypatch):
    import main
    monkeypatch.delenv("PW_AGENT", raising=False)
    assert isinstance(main.make_agent(), planner.PlannerAgent)
    monkeypatch.setenv("PW_AGENT", "neural")
    assert type(main.make_agent()) is entry_agent.NeuralAgent
