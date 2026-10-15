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


def _numpy_view(s):
    """The JAX state, trimmed to its real planets, as the entry's numpy code sees it."""
    import types
    n = int(np.asarray(s.valid).sum())
    assert np.asarray(s.valid)[:n].all()
    d = {k: np.asarray(v) for k, v in s._asdict().items()}
    for k in ("valid", "owner", "ships", "pos", "growth", "radius", "tr_on", "tr_owner", "tr_dest", "tr_ships", "tr_pos", "tr_vel"):
        d[k] = d[k][:n]
    d["ships"], d["growth"], d["radius"] = (d[k].astype(np.float64) for k in ("ships", "growth", "radius"))
    d["tr_ships"], d["pos"], d["tr_pos"] = (d[k].astype(np.float64) for k in ("tr_ships", "pos", "tr_pos"))
    d["speed"], d["tick"], d["max_ticks"] = float(d["speed"]), int(d["tick"]), int(d["max_ticks"])
    return types.SimpleNamespace(**d)


def test_jax_planner_scores_like_the_numpy_one():
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from pwjax import env, maps, policy
    from pwjax import planner as jplanner

    plan = jax.jit(jplanner.make_planner(sources=32, horizon=90, top=4))
    params = policy.init(jax.random.PRNGKey(0))
    checked = 0
    for seed in range(3):
        bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=seed, param_fn=maps.cog2026_params).items()}
        s = env.from_bank(bank, 0)
        key = jax.random.PRNGKey(seed + 7)
        for t in range(200):
            key, k1, k2 = jax.random.split(key, 3)
            if t % 20 == 10:
                for me in (1, 2):
                    ns = _numpy_view(s)
                    horizon = max(1, min(90, int(ns.max_ticks - ns.tick)))
                    out = plan(s, me, horizon)
                    n = ns.ships.shape[0]
                    a1, a2 = planner.arrivals(ns, horizon)
                    d = np.sqrt(((ns.pos[:, None, :] - ns.pos[None, :, :]) ** 2).sum(-1))
                    near = (d <= 160.0) & ~np.eye(n, dtype=bool)
                    bo, bs = planner.run(ns.owner.copy(), ns.ships.copy(), ns.growth, a1, a2)
                    np.testing.assert_allclose(float(out["base"]), planner.score(bo[None], bs[None], ns, me, near)[0], rtol=1e-4, atol=1e-2)
                    np.testing.assert_array_equal(np.asarray(out["base_owner"])[:n], bo)
                    want = planner.plan(ns, me, horizon)
                    src, dst, ships = (np.asarray(x) for x in out["action"])
                    if (want is None) != (src < 0):  # only where launching gains nothing (a tie, within float32 noise)
                        assert abs(float(out["best"]) - float(out["base"])) < 0.05
                        continue
                    if want is None:
                        continue
                    # same best score, even where two moves tie
                    def value(i, j, c):
                        ao, ash = planner.project(ns, me, horizon, np.array([i]), np.array([j]), np.array([c], float), a1, a2, bo, bs, d)
                        return planner.score(ao, ash, ns, me, near)[0]
                    np.testing.assert_allclose(value(int(src), int(dst), float(ships)), value(*want), rtol=1e-4, atol=1e-2)
                    np.testing.assert_allclose(float(out["best"]), value(*want), rtol=1e-4, atol=1e-2)
                    checked += 1
            s = env.step(s, policy.sample_t(params, 4, s, 1, k1, 1.0, 0.5), policy.sample_t(params, 4, s, 2, k2, 1.0, 0.5))
    assert checked >= 8


def test_value_agent_with_the_hand_score_alone_plays_the_planner_move():
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from pwjax import env, lvalue, maps, policy

    params = policy.init(jax.random.PRNGKey(0))
    hand = jax.jit(lvalue.make_hand_agent(12, 90, 8))
    val = jax.jit(lvalue.make_value_agent(params, 4, 12, 90, 8, mix=1.0, scale=1e4))
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=3, param_fn=maps.cog2026_params).items()}
    s = env.from_bank(bank, 0)
    key = jax.random.PRNGKey(5)
    same = total = 0
    for t in range(120):
        key, k1, k2 = jax.random.split(key, 3)
        if t % 10 == 5:
            for me in (1, 2):
                a, b = hand(s, me, k1), val(s, me, k1)
                total += 1
                same += int(all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)))
        s = env.step(s, policy.sample_t(params, 4, s, 1, k1, 1.0, 0.5), policy.sample_t(params, 4, s, 2, k2, 1.0, 0.5))
    assert total >= 20 and same >= total - 2


def test_numpy_value_ranking_matches_the_jax_agent():
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from pwjax import env, lvalue, maps, policy

    params = policy.init(jax.random.PRNGKey(2))
    np_params = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    agent = jax.jit(lvalue.make_value_agent(params, 4, 32, 90, 8, mix=0.5))
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=9, param_fn=maps.cog2026_params).items()}
    s = env.from_bank(bank, 0)
    key = jax.random.PRNGKey(1)
    same = total = 0
    for t in range(150):
        key, k1, k2 = jax.random.split(key, 3)
        if t % 10 == 5:
            for me in (1, 2):
                ns = _numpy_view(s)
                # the value of a position is the same number in both
                v_np = planner.value_of(np_params, 4, ns, me, ns.owner, ns.ships, 77.0)
                v_jx = float(policy.apply(params, 4, lvalue.fresh(s, s.owner, s.ships, jnp.int32(77)), me)[2])
                assert abs(v_np - v_jx) < 1e-3, (v_np, v_jx)
                want = planner.plan_ranked(ns, me, np_params, 4, 90, planner.FRACTIONS, 8, 0.5)
                got = tuple(np.asarray(x) for x in agent(s, me, k1))
                total += 1
                same += int((want is None and got[0] < 0) or (want is not None and (int(got[0]), int(got[1])) == want[:2]))
        s = env.step(s, policy.sample_t(params, 4, s, 1, k1, 1.0, 0.5), policy.sample_t(params, 4, s, 2, k2, 1.0, 0.5))
    assert total >= 20 and same >= 0.85 * total, (same, total)


def test_projected_launch_sizes():
    """'cap' takes the target when the fleet lands, 'spare' leaves the source in our hands, every size is legal."""
    import jax
    import jax.numpy as jnp
    from pwjax import baselines, env, maps
    from pwjax import planner as jplanner

    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=5, param_fn=maps.cog2026_params).items()}
    s = env.from_bank(bank, 0)
    step, greedy = jax.jit(env.step), jax.jit(baselines.greedy)
    for _ in range(120):  # fleets in flight, planets changing hands
        s = step(s, greedy(s, 1), greedy(s, 2))
    for me in (1, 2):
        only = lambda *e: jax.jit(jplanner.make_planner(sources=32, horizon=90, top=24, fractions=(), extras=e))  # noqa: E731
        out = only("cap")(s, me, 90)
        ok = np.isfinite(np.asarray(out["cand_score"]))
        assert ok.any()
        src, dst, n = np.asarray(out["cand_src"]), np.asarray(out["cand_dst"]), np.asarray(out["cand_ships"])
        ships = np.asarray(s.ships)
        owner_after = np.asarray(out["cand_owner"])
        for i in np.nonzero(ok)[0]:
            assert 1 <= n[i] <= ships[src[i]]
            assert owner_after[i][dst[i]] == me, (me, i)  # nobody else acts in the projection, so the capture sticks
        out = only("spare")(s, me, 90)
        ok = np.isfinite(np.asarray(out["cand_score"]))
        src, n = np.asarray(out["cand_src"]), np.asarray(out["cand_ships"])
        for i in np.nonzero(ok)[0]:
            assert 1 <= n[i] <= ships[src[i]]
            assert np.asarray(out["cand_owner"])[i][src[i]] == me
        out = only("f:0.3", "cap:1.5:3", "spare:0.5")(s, me, 90)
        ok = np.isfinite(np.asarray(out["cand_score"]))
        src, n = np.asarray(out["cand_src"]), np.asarray(out["cand_ships"])
        assert ok.any() and all(1 <= n[i] <= ships[src[i]] for i in np.nonzero(ok)[0])


def test_numpy_launch_sizes_match_the_jax_ones():
    """The entry's numpy planner offers the same (source, target, ships) launches, and scores them the same, as the JAX one."""
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from pwjax import baselines, env, maps
    from pwjax import planner as jplanner

    extras = ("cap", "spare", "f:0.9", "cap:1.2:1", "spare:0.5", "f:0.5")
    fractions = (0.25, 0.5, 1.0)
    big = 32 * (len(fractions) + len(extras)) * 30
    plan = jax.jit(jplanner.make_planner(sources=32, horizon=90, top=big, fractions=fractions, extras=extras))
    step, greedy = jax.jit(env.step), jax.jit(baselines.greedy)
    compared = 0
    for seed in range(2):
        bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=seed + 20, param_fn=maps.cog2026_params).items()}
        s = env.from_bank(bank, 0)
        for t in range(240):
            if t % 60 == 30:
                for me in (1, 2):
                    out = plan(s, me, 90)
                    ok = np.isfinite(np.asarray(out["cand_score"]))
                    got = {(int(i), int(j), float(c)): float(v) for i, j, c, v in zip(
                        np.asarray(out["cand_src"])[ok], np.asarray(out["cand_dst"])[ok], np.asarray(out["cand_ships"])[ok],
                        np.asarray(out["cand_score"])[ok])}
                    ns = _numpy_view(s)
                    c = planner.candidates(ns, me, 90, fractions, None, extras)
                    want = {}
                    for i, j, n, v in zip(c["ci"], c["cj"], c["cn"], c["score"]):
                        want.setdefault((int(i), int(j), float(n)), float(v))
                    assert set(got) == set(want), (len(got), len(want), sorted(set(got) ^ set(want))[:5])
                    for k in got:
                        assert abs(got[k] - want[k]) < 0.05 + 1e-4 * abs(want[k]), (k, got[k], want[k])
                    compared += len(got)
            s = step(s, greedy(s, 1), greedy(s, 2))
    assert compared > 500
