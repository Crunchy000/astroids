"""Two settings of the entry's planner against each other in the official Python engine, on league maps.

Each map is played from both sides. Settings are PlannerAgent keyword arguments, comma separated, e.g.

    python tools/entry_match.py --a "" --b "pairs=2000" --maps 300 --procs 12 --out runs/pairs.jsonl

prints one JSON line per game (and appends it to --out), then a summary of A's score. With --b-dir, side B is the entry
in that directory instead (say, an older commit's `entry/`), run in a process of its own so it keeps its own code. Maps follow the live league
(GameParamGenerator.randomParams(): 10 to 30 planets, neutral ratio 0.25 to 0.35, growth 0.05 to 0.2, speed 2 to 5,
2000 ticks), seeded by map number, so two runs with the same --seed play the same maps. Each side keeps the entry's
own time budget; the engine does not enforce the league's limit, but moves slower than --slow ms are counted.
"""
import argparse
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "entry"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per game: no thread pile-ups
    os.environ.setdefault(_v, "1")


def parse(spec):
    """'pairs=2000,look=8:4' -> {'pairs': 2000, 'look': '8:4'} (numbers where they parse)."""
    out = {}
    for part in [p for p in spec.split(",") if p.strip()]:
        k, v = part.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def league_params(rng):
    from core.game_state import GameParams
    raw = rng.randint(10, 30)
    return GameParams(num_planets=raw if raw % 2 == 0 else raw + 1, initial_neutral_ratio=rng.uniform(0.25, 0.35),
                      min_growth_rate=0.05, max_growth_rate=0.2, transporter_speed=rng.uniform(2.0, 5.0), max_ticks=2000)


def _serve(conn, path, kw):
    """Side B from another entry directory: answers prepare_to_play_as and get_action sent over `conn`."""
    sys.path.insert(0, path)
    from planner import PlannerAgent
    agent = PlannerAgent(**kw)
    while True:
        msg = conn.recv()
        if msg[0] == "prep":
            agent.prepare_to_play_as(msg[1], msg[2])
            conn.send(None)
        elif msg[0] == "act":
            conn.send(agent.get_action(msg[1]))
        else:
            return


class Elsewhere:
    """An entry from another directory, in its own process (so its modules are its own), as a player."""

    def __init__(self, path, kw):
        self.conn, child = mp.get_context("spawn").Pipe()
        self.proc = mp.get_context("spawn").Process(target=_serve, args=(child, path, kw))
        self.proc.start()

    def prepare_to_play_as(self, player, params):
        self.conn.send(("prep", player, params))
        self.conn.recv()

    def get_action(self, state):
        self.conn.send(("act", state))
        return self.conn.recv()

    def close(self):
        self.conn.send(("stop",))
        self.proc.join(10)


def play(job):
    """One game: (map number, side A plays, settings A, settings B, seed, slow ms) -> result dict."""
    from core.forward_model import ForwardModel
    from core.game_state import Player
    from core.game_state_factory import GameStateFactory
    from planner import PlannerAgent
    m, a_side, a_kw, b_kw, seed, slow, b_dir = job
    rng = random.Random(seed * 100003 + m)
    params = league_params(rng)
    random.seed(seed * 100003 + m)
    gs = GameStateFactory(params).create_game()
    pa = Player.Player1 if a_side == 1 else Player.Player2
    agents = {pa: PlannerAgent(**a_kw), pa.opponent(): Elsewhere(b_dir, b_kw) if b_dir else PlannerAgent(**b_kw)}
    for p, ag in agents.items():
        ag.prepare_to_play_as(p, params)
    model = ForwardModel(copy.deepcopy(gs), params)
    took = {p: [] for p in agents}
    while not model.is_terminal():
        acts = {}
        for p, ag in agents.items():
            t0 = time.perf_counter()
            acts[p] = ag.get_action(model.state)
            took[p].append((time.perf_counter() - t0) * 1000)
        model.step(acts)
    if b_dir:
        agents[pa.opponent()].close()
    lead = model.get_leader()
    score = 1.0 if lead == pa else 0.0 if lead == pa.opponent() else 0.5
    ta, tb = took[pa], took[pa.opponent()]
    return dict(map=m, a_side=a_side, a_score=score, ticks=model.state.game_tick, planets=params.num_planets,
                a_ms=round(sum(ta) / len(ta), 1), b_ms=round(sum(tb) / len(tb), 1),
                a_slow=sum(t > slow for t in ta), b_slow=sum(t > slow for t in tb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="", help="PlannerAgent settings of the side being tested")
    ap.add_argument("--b", default="", help="settings of its opponent (default: the entry as it is)")
    ap.add_argument("--b-dir", default="", help="the opponent is the entry in this directory (with --b's settings)")
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--first", type=int, default=0, help="first map number")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow", type=float, default=200.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    a_kw, b_kw = parse(args.a), parse(args.b)
    jobs = [(m, side, a_kw, b_kw, args.seed, args.slow, args.b_dir) for m in range(args.first, args.first + args.maps) for side in (1, 2)]
    pts = n = 0
    t0 = time.time()
    f = open(args.out, "a") if args.out else None
    # not mp.Pool: its workers are daemonic and could not start the play-outs' helper processes
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as pool:
        for r in (fut.result() for fut in as_completed([pool.submit(play, j) for j in jobs])):
            pts, n = pts + r["a_score"], n + 1
            line = json.dumps(dict(a=args.a, b=args.b + (f" @{args.b_dir}" if args.b_dir else ""), seed=args.seed, **r))
            print(line, flush=True)
            if f:
                f.write(line + "\n")
                f.flush()
    se = (0.25 / max(n, 1)) ** 0.5
    print(json.dumps(dict(a=args.a, b=args.b, games=n, a_points=pts, score=round(pts / max(n, 1), 4), se=round(se, 4),
                          seconds=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
