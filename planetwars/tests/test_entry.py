"""The numpy entry must compute exactly what the JAX trainer computes."""
import pathlib
import random
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entry"
sys.path.insert(0, str(ENTRY))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agent as entry_agent  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402
from test_parity import fuzz_action, to_state  # noqa: E402

from pwjax import policy  # noqa: E402
from pwjax.maps import N_MAX  # noqa: E402


def test_numpy_entry_matches_jax(tmp_path):
    params = policy.init(jax.random.PRNGKey(3), d_model=32, layers=2, heads=4, d_pair=16, d_ff=64)
    # make the heads opinionated so argmax ties are unlikely
    params = {k: (v * 300 if k in policy.SMALL_INIT else v) for k, v in params.items()}
    path = tmp_path / "m.npz"
    np.savez(path, **policy.to_numpy(params), **{"cfg.heads": np.array(4)})
    np_params, heads = entry_agent.load(str(path))
    jparams = {k: jnp.asarray(v, jnp.float64) for k, v in np_params.items()}
    rng = random.Random(0)
    random.seed(0)
    params_g = GameParams(num_planets=24, max_ticks=400, min_growth_rate=0.05, max_growth_rate=0.2,
                          transporter_speed=3.7, initial_neutral_ratio=0.3)
    gs = GameStateFactory(params_g).create_game()
    model = ForwardModel(gs, params_g)
    greedy = {p: GreedyHeuristicAgent() for p in (Player.Player1, Player.Player2)}
    for p, a in greedy.items():
        a.prepare_to_play_as(p, params_g)
    checked = 0
    while not model.is_terminal():
        if gs.game_tick % 37 == 0:
            n = len(gs.planets)
            s_np = entry_agent.arrays(gs, params_g.transporter_speed, params_g.max_ticks)
            s_j = to_state(gs, params_g)
            for me in (1, 2):
                flat_j, amt_j, val_j, aux_j = policy.apply(jparams, heads, s_j, me)
                planet, glob, pair, aux = entry_agent.pwnet.features(np, s_np, me)
                flat, amt, val = entry_agent.pwnet.forward(np, np_params, heads, planet, glob, pair, s_np.valid,
                                                           aux["pair_ok"])
                pj = np.asarray(flat_j[:-1]).reshape(N_MAX, N_MAX)[:n, :n]
                np.testing.assert_allclose(flat[:-1].reshape(n, n), pj, rtol=1e-9, atol=1e-6)
                np.testing.assert_allclose(flat[-1], float(flat_j[-1]), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(val, float(val_j), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(amt, np.asarray(amt_j)[:n, :n], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(aux["amounts"], np.asarray(aux_j["amounts"])[:n, :n], atol=1e-9)
                src, dst, ships = entry_agent.choose(np_params, heads, s_np, me)
                (js, jd, jsh), *_ = policy.sample(jparams, heads, s_j, me, jax.random.PRNGKey(0), greedy=True)
                assert (src, dst) == (int(js), int(jd))
                assert abs(ships - float(jsh)) < 1e-9
                checked += 1
        actions = {p: (greedy[p].get_action(gs) if rng.random() < 0.5 else fuzz_action(rng, gs, p))
                   for p in (Player.Player1, Player.Player2)}
        model.step(actions)
    assert checked >= 10


def test_agent_plays_in_official_engine(tmp_path, monkeypatch):
    params = policy.init(jax.random.PRNGKey(1), d_model=16, layers=1, heads=2, d_pair=8, d_ff=32)
    params["n2.b"] = params["n2.b"] - 10.0  # never "do nothing", so every launch path gets used
    path = tmp_path / "m.npz"
    np.savez(path, **policy.to_numpy(params), **{"cfg.heads": np.array(2)})
    monkeypatch.setattr(entry_agent, "MODEL", str(path))
    monkeypatch.setattr(entry_agent.load, "__defaults__", (str(path),))
    params_g = GameParams(num_planets=20, max_ticks=150)
    gs = GameStateFactory(params_g).create_game()
    model = ForwardModel(gs, params_g)
    me = entry_agent.NeuralAgent()
    me.prepare_to_play_as(Player.Player2, params_g)
    other = GreedyHeuristicAgent()
    other.prepare_to_play_as(Player.Player1, params_g)
    launched = 0
    failed_before = ForwardModel.n_failed_actions
    while not model.is_terminal():
        a = me.get_action(gs)
        launched += a.source_planet_id >= 0
        model.step({Player.Player1: other.get_action(gs), Player.Player2: a})
    assert launched > 5
    assert ForwardModel.n_failed_actions == failed_before  # the greedy agent never fails, so neither did we
