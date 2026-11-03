"""Record what the Python entry does, for tools/parity.js to check the browser port (pw.js) against.

Writes one JSON file with
  engine    whole games in the official engine (entry/core): the start, both sides' actions every tick and the state
            every 50 ticks, so the JavaScript rules can replay them;
  positions positions from those games with the entry's choices for both sides: `plan` (hand score only), `plan_ranked`
            (value network and look-ahead, no time limit) and the network's value of the nobody-launches projection;
  agent     the entry (no time limit) against their greedy agent from the start of a game, its actions every tick.

    python planetwars/play/tools/dump_parity.py OUT.json [--games 4] [--agent-ticks 300]
"""
import argparse
import copy
import json
import pathlib
import random
import sys

import numpy as np

ENTRY = pathlib.Path(__file__).resolve().parents[2] / "entry"
sys.path.insert(0, str(ENTRY))
import planner  # noqa: E402
from agent import OWNER, arrays  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import Action, GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402

PLAYERS = {1: Player.Player1, 2: Player.Player2}


def league_params(rng):
    raw = rng.randint(10, 30)
    return GameParams(num_planets=raw if raw % 2 == 0 else raw + 1, initial_neutral_ratio=rng.uniform(0.25, 0.35),
                      min_growth_rate=0.05, max_growth_rate=0.2, transporter_speed=rng.uniform(2.0, 5.0))


def state_json(gs, params):
    planets = []
    for p in gs.planets:
        t = p.transporter
        planets.append(dict(id=p.id, owner=OWNER[p.owner], ships=p.n_ships, x=p.position.x, y=p.position.y,
                            growth=p.growth_rate, radius=p.radius,
                            tr=None if t is None else dict(x=t.s.x, y=t.s.y, vx=t.v.x, vy=t.v.y, owner=OWNER[t.owner],
                                                           src=t.source_index, dst=t.destination_index, ships=t.n_ships)))
    return dict(tick=gs.game_tick, speed=params.transporter_speed, maxTicks=params.max_ticks, planets=planets)


def action_json(a):
    if a is None or a == Action.DO_NOTHING or a.source_planet_id < 0:
        return None
    return dict(src=a.source_planet_id, dst=a.destination_planet_id, ships=a.num_ships)


def move_json(m):
    return None if m is None else dict(src=int(m[0]), dst=int(m[1]), ships=float(m[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--agent-ticks", type=int, default=300)
    ap.add_argument("--every", type=int, default=40)
    args = ap.parse_args()

    agent = planner.PlannerAgent()  # the entry as served (value.npz in single precision)
    vparams, vheads = agent.value
    out = dict(engine=[], positions=[], agent=[])
    for g in range(args.games):
        rng = random.Random(100 + g)
        random.seed(100 + g)
        params = league_params(rng)
        gs = GameStateFactory(params).create_game()
        start = state_json(gs, params)
        model = ForwardModel(gs, params)
        greedy = GreedyHeuristicAgent()
        greedy.prepare_to_play_as(Player.Player2, params)
        actions, checkpoints = [], {}
        while not model.is_terminal():
            s = arrays(gs, params.transporter_speed, params.max_ticks)
            if gs.game_tick % args.every == 5:
                pos = dict(state=state_json(gs, params), sides={})
                for me in (1, 2):
                    o = {}
                    ranked = planner.plan_ranked(s, me, vparams, vheads, budget_ms=1e9, extras=("spare",), out=o)
                    c = planner.candidates(s, me, extras=("spare",))
                    base_v = None if c is None else planner.value_of(vparams, vheads, s, me, c["base_owner"], c["base_ships"], s.tick + c["horizon"])
                    pos["sides"][me] = dict(plan=move_json(planner.plan(s, me, extras=("spare",))), ranked=move_json(ranked),
                                            follow=None if "follow" not in o else dict(wait=o["follow"][0], move=move_json(o["follow"][1])),
                                            base_value=base_v, base_score=None if c is None else float(c["base"]),
                                            n_candidates=0 if c is None else int(c["ci"].size))
                out["positions"].append(pos)
            # player 1 plays the hand-scored planner, player 2 their greedy agent: quick, and plenty of fleets in flight
            m = planner.plan(s, 1, extras=("spare",))
            a1 = Action(player_id=Player.Player1, source_planet_id=int(m[0]), destination_planet_id=int(m[1]),
                        num_ships=float(min(m[2], gs.planets[int(m[0])].n_ships))) if m else Action.DO_NOTHING
            a2 = greedy.get_action(gs)
            actions.append([action_json(a1), action_json(a2)])
            model.step({Player.Player1: a1, Player.Player2: a2})
            if gs.game_tick % 50 == 0 or model.is_terminal():
                checkpoints[gs.game_tick] = state_json(gs, params)
        out["engine"].append(dict(start=start, actions=actions, checkpoints=checkpoints))
        print(f"game {g}: {len(gs.planets)} planets, {gs.game_tick} ticks, {len(out['positions'])} positions so far", flush=True)

    for g in range(2):  # the entry itself, as it plays (two-planet captures and all), against their greedy agent
        rng = random.Random(200 + g)
        random.seed(200 + g)
        params = league_params(rng)
        gs = GameStateFactory(params).create_game()
        start = state_json(gs, params)
        model = ForwardModel(gs, params)
        me = 1 + g % 2
        entry = planner.PlannerAgent(budget_ms=1e9)
        entry.prepare_to_play_as(PLAYERS[me], params)
        greedy = GreedyHeuristicAgent()
        greedy.prepare_to_play_as(PLAYERS[3 - me], params)
        actions = []
        while not model.is_terminal() and gs.game_tick < args.agent_ticks:
            a = {PLAYERS[me]: entry.get_action(copy.deepcopy(gs)), PLAYERS[3 - me]: greedy.get_action(gs)}
            actions.append([action_json(a[Player.Player1]), action_json(a[Player.Player2])])
            model.step(a)
        out["agent"].append(dict(start=start, entry=me, actions=actions))
        print(f"agent game {g}: entry is player {me}, {len(actions)} ticks", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
