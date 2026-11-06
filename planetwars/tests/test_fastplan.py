"""The compiled launch scoring (entry/fastplan.py) must give exactly what the numpy code gives, and the play-outs'
batched planner (planner.plan_many) must pick what plan_ranked picks one position at a time."""
import pathlib
import random
import sys

import numpy as np
import pytest

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entry"
sys.path.insert(0, str(ENTRY))
import agent as entry_agent  # noqa: E402
import planner  # noqa: E402
import playout  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402

pytestmark = pytest.mark.skipif(planner.fastplan is None, reason="numba not installed")
KEYS = ("ci", "cj", "cn", "owner", "ships", "score", "base_owner", "base_ships", "base", "a1", "a2", "d", "near", "po", "ps")


def positions(sizes=(10, 18, 30), every=9, until=500, seed=100):
    """Positions from planner-against-planner games (no play-outs, so they are quick)."""
    out = []
    for g, n in enumerate(sizes):
        random.seed(seed + g)
        params = GameParams(num_planets=n, max_ticks=2000, min_growth_rate=0.05, max_growth_rate=0.2,
                            transporter_speed=(2.0, 3.5, 5.0)[g % 3], initial_neutral_ratio=0.3)
        model = ForwardModel(GameStateFactory(params).create_game(), params)
        agents = {p: planner.PlannerAgent(search="", look="") for p in (Player.Player1, Player.Player2)}
        for p, a in agents.items():
            a.prepare_to_play_as(p, params)
        while not model.is_terminal() and model.state.game_tick < until:
            if model.state.game_tick % every == 0:
                out.append(entry_agent.arrays(model.state, params.transporter_speed, params.max_ticks))
            model.step({p: a.get_action(model.state) for p, a in agents.items()})
    return out, agents[Player.Player1].value


def numpy_candidates(*args, **kw):
    fast = planner.fastplan
    planner.fastplan = None
    try:
        return planner.candidates(*args, **kw)
    finally:
        planner.fastplan = fast


def test_compiled_candidates_equal_numpy():
    states, _ = positions()
    checked = 0
    for s in states:
        for me in (1, 2):
            for extras in (("spare",), ("spare", "cap", "capm", "f:0.1", "spare2")):
                for horizon in (90, 25):
                    want = numpy_candidates(s, me, horizon, planner.FRACTIONS, None, extras, keep=True)
                    got = planner.candidates(s, me, horizon, planner.FRACTIONS, None, extras, keep=True)
                    assert (want is None) == (got is None)
                    if want is None:
                        continue
                    for k in KEYS:
                        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(want[k]), err_msg=k)
                    checked += 1
    assert checked > 100


def test_compiled_base_equals_candidates_base():
    states, _ = positions(sizes=(14,), every=25)
    for s in states:
        c = numpy_candidates(s, 1, 90, planner.FRACTIONS, None, ("spare",))
        if c is None:
            continue
        o, sh, sc = planner.fastplan.base(s, 1, c["horizon"])
        np.testing.assert_array_equal(o, c["base_owner"])
        np.testing.assert_array_equal(sh, c["base_ships"])
        assert sc == c["base"]


def test_batched_planner_picks_as_one_at_a_time():
    states, (params, heads) = positions(sizes=(12, 24), every=4, until=300)
    same = total = 0
    for k in range(0, len(states) - 3, 3):
        if not states[k].tick < states[k + 1].tick < states[k + 2].tick:  # positions from two games
            continue
        items = [(states[k + q], side) for q in range(3) for side in (1, 2)]
        many = planner.plan_many(items, params, heads, 90, planner.FRACTIONS, 16, 0.5, ("spare",))
        for (s, me), got in zip(items, many):
            want = planner.plan_ranked(s, me, params, heads, 90, planner.FRACTIONS, 16, 0.5, 1e9, ("spare",), look="",
                                       pairs=0, pair_kinds="")
            same += got == want
            total += 1
    assert total > 50 and same == total


def test_batched_end_values_match_one_at_a_time():
    states, (params, heads) = positions(sizes=(16,), every=40)
    for s in states[1:6]:
        after = [playout.step(s, None, None)] + [playout.step(s, a, None) for a in
                                                 planner.plan_many([(s, 1), (s, 1)], params, heads, 90,
                                                                   planner.FRACTIONS, 16, 0.5, ("spare",)) if a]
        together = playout.end_values(after, 1, params, heads, 90, 0.5)
        alone = [playout.end_value(x, 1, params, heads, 90, 0.5) for x in after]
        np.testing.assert_allclose(together, alone, rtol=0, atol=1e-5)
