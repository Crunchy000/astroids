"""Play a running entry (Docker container on a port) over the league's WebSocket protocol.

Sends exactly what the Kotlin RemoteAgent sends (JSON-RMI with "type" discriminators), runs the games
in the official Python engine, and the opponent is an official sample agent:

    docker run -d -p 49875:8080 my-entry
    python tools/remote_match.py --port 49875 --games 4
"""
import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "entry"))
from websockets.sync.client import connect  # noqa: E402

from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402
from agents.random_agents import CarefulRandomAgent  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_state import Action, GameParams, Player  # noqa: E402
from core.game_state_factory import GameStateFactory  # noqa: E402


class Remote:
    def __init__(self, port):
        try:
            self.ws = connect(f"ws://localhost:{port}/ws", proxy=None, max_size=None).__enter__()
        except TypeError:  # websockets < 15 has no proxy argument
            self.ws = connect(f"ws://localhost:{port}/ws", max_size=None).__enter__()
        r = self.call({"requestType": "init", "target": "agent", "className": "remote", "method": "<ignored>",
                       "objectId": None, "args": []})
        self.object_id = r["result"]["objectId"]

    def call(self, req):
        self.ws.send(json.dumps(req))
        r = json.loads(self.ws.recv())
        if r.get("status") != "ok":
            raise RuntimeError(r)
        return r

    def invoke(self, method, *args):
        return self.call({"requestType": "invoke", "target": "agent", "className": None, "method": method,
                          "objectId": self.object_id, "args": list(args)})["result"]

    def close(self):
        self.call({"requestType": "end", "target": "agent", "className": None, "method": "",
                   "objectId": self.object_id, "args": []})
        self.ws.close()


def params_json(p: GameParams):
    d = p.model_dump(by_alias=True)
    d["type"] = "GameParams"
    return d


def state_json(gs):
    d = gs.model_dump(by_alias=True, mode="json")
    d["type"] = "GameState"
    return d


def play(port, opponent, params, remote_side):
    gs = GameStateFactory(params).create_game()
    model = ForwardModel(gs, params)
    remote = Remote(port)
    me = Player.Player1 if remote_side == 1 else Player.Player2
    name = remote.invoke("prepareToPlayAs", me.value, params_json(params), "Anonymous")
    opponent.prepare_to_play_as(me.opponent(), params)
    times, failed0 = [], ForwardModel.n_failed_actions
    while not model.is_terminal():
        t = time.perf_counter()
        res = remote.invoke("getAction", state_json(gs))
        times.append(time.perf_counter() - t)
        res.pop("type", None)
        a = Action.model_validate(res) if res.get("sourcePlanetId", -1) >= 0 else Action.do_nothing()
        model.step({me: a, me.opponent(): opponent.get_action(gs)})
    remote.invoke("processGameOver", state_json(gs))
    remote.close()
    return model.get_leader() == me, name, times, ForwardModel.n_failed_actions - failed0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=49875)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    params = GameParams(num_planets=20, max_ticks=1000)  # the submission check's settings
    for opp_name, cls in (("greedy", GreedyHeuristicAgent), ("careful random", CarefulRandomAgent)):
        wins, times, failed = 0, [], 0
        for g in range(args.games):
            for side in (1, 2):
                won, name, t, f = play(args.port, cls(), params, side)
                wins += won
                times += t
                failed += f
        t = sorted(times)
        print(f"{name} vs {opp_name}: won {wins}/{2 * args.games}; move time median {t[len(t) // 2] * 1000:.1f} ms,"
              f" p99 {t[int(len(t) * 0.99)] * 1000:.1f} ms, max {t[-1] * 1000:.1f} ms; illegal actions {failed}",
              flush=True)


if __name__ == "__main__":
    main()
