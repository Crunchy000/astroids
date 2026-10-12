"""The competition format: JSON states, agents, matches, the WebSocket protocol, the league."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from astroids_gym import Config, ObsConfig, Simulation
from astroids_gym.bot import scripted_action
from astroids_gym.competition import (BotAgent, DoNothingAgent, ModelAgent, Player, RandomAgent, make_agent,
                                      make_params, parse_action, random_params, restore, run_match, ship_command,
                                      snapshot)
from astroids_gym.competition import league as league_mod
from astroids_gym.competition.league import Rating, TrueSkill, leaderboard_markdown, run_league, schedule
from astroids_gym.competition.submissions import evaluate, parse_issue, report_markdown
from astroids_gym.obs import encode_battle

MODELS = Path(__file__).resolve().parents[1] / "competition" / "models"


def short(seconds=3.0, **kw):
    return make_params(Config.competitive(battle_seconds=seconds, **kw))


def played(seed=3, ticks=400):
    sim = Simulation(Config.competitive(), seed=seed)
    for _ in range(ticks):
        sim.step({s.id: scripted_action(sim, s, 1) for s in sim.ships})
    return sim


def test_state_round_trips_through_json():
    sim = played()
    state = json.loads(json.dumps(snapshot(sim)))
    back = restore(state, make_params(sim.cfg))
    assert snapshot(back) == snapshot(sim)
    oc = ObsConfig(extended=True)
    assert np.array_equal(encode_battle(back, range(10), oc), encode_battle(sim, range(10), oc))


def test_restored_state_steps_like_the_match():
    cfg = Config.competitive(asteroids=False)
    sim = Simulation(cfg, seed=4)
    fm = restore(snapshot(sim), cfg)
    for _ in range(30):  # no deaths this early, so nothing random happens
        acts = {s.id: scripted_action(sim, s, 1) for s in sim.ships}
        sim.step(acts)
        fm.step(acts)
    assert snapshot(fm) == snapshot(sim)


def test_params_round_trip_and_vary():
    p = random_params(7)
    assert p == random_params(7) and json.loads(json.dumps(p)) == p
    seen = {(q["asteroid_count"], q["flight"], q["team_lives"]) for q in map(random_params, range(40))}
    assert len(seen) > 20
    assert {random_params(s)["flight"] for s in range(20)} == {False, True}
    assert random_params(7, fixed=True)["asteroid_count"] == Config.competitive().asteroid_count


def test_parse_action_rejects_bad_commands():
    cfg = Config.competitive()
    acts, bad = parse_action({"ships": [ship_command(0, 2.0, True, False), ship_command(7, 0.5),
                                        ship_command(0, -1.0), {"id": 1, "turn": math.nan}, {"turn": 1.0},
                                        {"id": 2}]}, 0, cfg)
    assert acts == {0: (1.0, True, False), 2: (0.0, False, False)} and bad == 4
    assert parse_action({"ships": [ship_command(7, 0.5)]}, 1, cfg) == ({7: (0.5, False, False)}, 0)
    assert parse_action(None, 0, cfg) == ({}, 1)
    assert parse_action({"ships": "all"}, 0, cfg) == ({}, 1)


def test_turn_based_match_is_reproducible():
    def untimed(res):
        d = res.to_json()
        d.pop("wall_seconds")
        for st in d["stats"].values():
            del st["total_ms"], st["max_ms"], st["mean_ms"]
        return d

    a = run_match(BotAgent(seed=1), BotAgent("pincer", seed=2), short(), seed=5)
    b = run_match(BotAgent(seed=1), BotAgent("pincer", seed=2), short(), seed=5)
    assert untimed(a) == untimed(b)
    assert a.steps == 90 and a.names == {"blue": "bots", "red": "bots:pincer"}
    assert a.stats["blue"].actions == 90 and a.stats["blue"].late == 0


def test_bots_beat_doing_nothing():
    res = run_match(BotAgent(seed=0), DoNothingAgent(), short(20.0, asteroids=False), seed=1)
    assert res.winner == "blue" and res.kills["blue"] > 0


class Slow(Player):
    def get_agent_type(self):
        return "slow"

    def get_action(self, state):
        time.sleep(0.08)
        return {"ships": [ship_command(i, 1.0, True, False) for i in self.ship_ids]}


def test_slow_agent_acts_late_in_real_time():
    res = run_match(Slow(), RandomAgent(0), short(1.0), seed=2, realtime=True)
    slow = res.stats["blue"]
    assert slow.late > 0 and 0 < slow.actions < res.steps and slow.mean_ms >= 80
    assert res.stats["red"].late == 0


class Broken(Player):
    def get_agent_type(self):
        return "broken"

    def prepare_to_play_as(self, player, params, opponent=None):
        raise RuntimeError("no model file")

    def get_action(self, state):
        raise AssertionError("never called")


def test_failing_to_get_ready_forfeits():
    res = run_match(RandomAgent(0), Broken(), short(), seed=0)
    assert res.winner == "blue" and res.steps == 0 and "no model file" in res.stats["red"].forfeit


def test_model_agent_plays():
    agent = make_agent(str(MODELS / "league_striker.npz"))
    assert isinstance(agent, ModelAgent) and agent.obs_config.extended
    res = run_match(agent, BotAgent(seed=0), short(2.0), seed=1)
    assert res.stats["blue"].actions == res.steps and res.stats["blue"].rejected == 0


def test_agent_over_websocket():
    pytest.importorskip("websockets")
    from astroids_gym.competition.protocol import AgentServer, RemoteAgent, RemoteError
    server = AgentServer(lambda: BotAgent(seed=0), host="127.0.0.1", port=0).start()
    try:
        remote = RemoteAgent(f"127.0.0.1:{server.port}")
        assert remote.get_agent_type() == "bots"
        res = run_match(remote, RandomAgent(0), short(2.0), seed=1, realtime=True)
        assert res.stats["blue"].actions > 0 and res.stats["blue"].errors == 0
        with pytest.raises(RemoteError, match="unknown method"):
            remote.invoke("__init__")
        # A Kotlin client tags objects with a "type" field; the server ignores it.
        sim = Simulation(Config.competitive(), seed=0)
        params = {**make_params(sim.cfg), "type": "astorids.GameParams"}
        assert remote.invoke("prepareToPlayAs", "red", params, "someone") == "bots"
        action = remote.invoke("getAction", {**snapshot(sim), "type": "astorids.GameState"})
        assert parse_action(action, 1, sim.cfg)[1] == 0 and len(action["ships"]) == 5
        remote.close()
    finally:
        server.stop()


def test_trueskill_matches_reference_values():
    env = TrueSkill()
    a, b = env.rate(Rating(), Rating(), 1.0)
    assert (round(a.mu, 3), round(b.mu, 3), round(a.sigma, 3)) == (29.396, 20.604, 7.171)
    a, b = env.rate(Rating(), Rating(), 0.5)
    assert a.mu == pytest.approx(25.0) and a.sigma == pytest.approx(6.457, abs=2e-3)
    a, b = env.rate(Rating(50, 1), Rating(0, 1), 0.0)  # a huge upset stays finite
    assert (round(a.mu, 2), round(b.mu, 2)) == (48.59, 1.41)


def test_league_plays_each_pair_from_both_sides(tmp_path, monkeypatch):
    monkeypatch.setattr(league_mod, "random_params", lambda seed, fixed=False: short(2.0, asteroids=False))
    entries = {"idle": "idle", "random": "random", "bots": "bots"}
    path = str(tmp_path / "results.jsonl")
    results = run_league(entries, games_per_pair=2, results_path=path, realtime=False, progress=None)
    assert len(results) == 6 == len(schedule(list(entries), 2))
    sides = {(r["names"]["blue"], r["names"]["red"]) for r in results}
    assert ("idle", "bots") in sides and ("bots", "idle") in sides
    again = run_league(entries, games_per_pair=2, results_path=path, realtime=False, progress=None)
    assert again == results and len(open(path).readlines()) == 6  # nothing replayed
    md = leaderboard_markdown(results)
    assert md.index("| 1 | bots |") < md.index("| idle |")


def test_parse_issue():
    body = "Here is my entry\n```yaml\nid: My-Team  # comment\nrepo_url: https://github.com/someone/agent/commit/" \
           "9c1133cd88217abf99a892e89d95bae6fd0ed66b\n```\n"
    assert parse_issue(body) == {"id": "my-team", "repo": "https://github.com/someone/agent",
                                 "commit": "9c1133cd88217abf99a892e89d95bae6fd0ed66b"}
    body = "```yaml\nid: x\nrepo_url: https://github.com/a/b\ncommit: abc1234\n```"
    assert parse_issue(body)["commit"] == "abc1234"
    for bad in ("no block", "```yaml\nid: x\nrepo_url: https://github.com/a/b\n```",
                "```yaml\nid: x\nrepo_url: https://gitlab.com/a/b/commit/abc1234\n```",
                "```yaml\nid: ../x\nrepo_url: https://github.com/a/b/commit/abc1234\n```"):
        with pytest.raises(ValueError):
            parse_issue(bad)


def test_evaluate_report(monkeypatch):
    monkeypatch.setattr(league_mod, "random_params", lambda seed, fixed=False: short(1.0))
    results = evaluate("random", "mine", {"bots": "bots", "idle": "idle"}, games=2, realtime=False, progress=None)
    md = report_markdown(results, "mine")
    assert len(results) == 4 and "| bots |" in md and "| idle |" in md and "Overall" in md
