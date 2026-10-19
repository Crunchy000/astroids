"""Ask the entry's planner about one position of a TracePair game and show how it ranks launches.
usage: replay.py TRACE.jsonl.gz TICK [TARGET ...]   (TARGET: planet ids whose launches to show as well)"""
import gzip
import json
import os
import sys

sys.path.insert(0, "/home/user/astorids/planetwars/entry")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import planner as P  # noqa: E402
from agent import OWNER, arrays  # noqa: E402
from client_server.util import deserialize_args  # noqa: E402
from core.game_state import GameParams, Player  # noqa: E402


def position(path, tick):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())
        for line in f:
            r = json.loads(line)
            if r["t"] == tick:
                return head, r
    raise SystemExit("no such tick")


def main():
    path, tick = sys.argv[1], int(sys.argv[2])
    show = [int(x) for x in sys.argv[3:]]
    head, r = position(path, tick)
    gs = deserialize_args("get_action", [r["state"]])[0]
    params = GameParams(num_planets=head["planets"], transporter_speed=head["speed"])
    me = head["a_side"]
    agent = P.PlannerAgent()
    agent.prepare_to_play_as(Player.Player1 if me == 1 else Player.Player2, params)
    s = arrays(gs, params.transporter_speed, params.max_ticks)
    params_v, heads = agent.value
    c = P.candidates(s, me, P.HORIZON, P.FRACTIONS, None, agent.extras, keep=True)
    order = np.argsort(-c["score"], kind="stable")
    tick_h = s.tick + c["horizon"]
    tot = P.totals_of(params_v, heads, s, me, list(c["owner"]), list(c["ships"]), c["score"], tick_h, P.MIX)
    base_t = P.totals_of(params_v, heads, s, me, [c["base_owner"]], [c["base_ships"]], [c["base"]], tick_h, P.MIX)[0]
    rank_hand = np.empty(len(order), int)
    rank_hand[order] = np.arange(len(order))
    print(f"tick {tick}: {len(order)} launches; wait: hand {c['base']:.1f} total {base_t:.3f}")
    by_tot = np.argsort(-tot, kind="stable")
    print(" best by total (hand rank in brackets):")
    for k in by_tot[:12]:
        print(f"  {c['ci'][k]:2d}->{c['cj'][k]:2d} n={c['cn'][k]:4.0f} hand {c['score'][k]:7.1f} [#{rank_hand[k]}] total {tot[k]:.3f}")
    for j in show:
        print(f" launches to {j}:")
        for k in np.nonzero(c["cj"] == j)[0]:
            print(f"  {c['ci'][k]:2d}->{j:2d} n={c['cn'][k]:4.0f} hand {c['score'][k]:7.1f} [#{rank_hand[k]}] total {tot[k]:.3f}")
    print(" planner picks:", agent.get_action(gs))
    m = r[f"m{me}"]
    print(" played:", m.get("sourcePlanetId"), "->", m.get("destinationPlanetId"), m.get("numShips"))




def look(path, tick, ours=8, n_rep=4):
    """The look-ahead as plan_ranked does it, over the best `ours` options by total (all launches scored, no budget)."""
    head, r = position(path, tick)
    gs = deserialize_args("get_action", [r["state"]])[0]
    params = GameParams(num_planets=head["planets"], transporter_speed=head["speed"])
    me = head["a_side"]
    agent = P.PlannerAgent()
    agent.prepare_to_play_as(Player.Player1 if me == 1 else Player.Player2, params)
    s = arrays(gs, params.transporter_speed, params.max_ticks)
    pv, heads = agent.value
    c = P.candidates(s, me, P.HORIZON, P.FRACTIONS, None, agent.extras, keep=True)
    tick_h = s.tick + c["horizon"]
    order = np.argsort(-c["score"], kind="stable")[:P.TOP]
    owners = [c["owner"][k] for k in order] + [c["base_owner"]]
    shipss = [c["ships"][k] for k in order] + [c["base_ships"]]
    hand = np.concatenate([c["score"][order], [c["base"]]])
    tot = P.totals_of(pv, heads, s, me, owners, shipss, hand, tick_h, P.MIX)
    keys = np.concatenate([order, [-1]])
    pick = np.argsort(-tot, kind="stable")[:ours]
    opts = keys[pick]
    r_own, r_sh, found, acts = P.replies(s, me, c, opts, P.FRACTIONS, agent.extras, n_rep, P.LOOK_SOURCES)
    for o, k in enumerate(opts):
        name = "wait" if k < 0 else f"{c['ci'][k]}->{c['cj'][k]} n={c['cn'][k]:.0f}"
        ro, rs = r_own[o][found[o]], r_sh[o][found[o]]
        rt = P.totals_of(pv, heads, s, me, ro, rs, P.score(ro, rs, s, me, c["near"]), tick_h, P.MIX) if len(ro) else []
        reps = ", ".join(f"{int(a[0])}->{int(a[1])} n={a[2]:.0f}: {t:.3f}" for a, t in zip(acts[o][found[o]], rt))
        print(f" {name:16s} total {tot[pick[o]]:.3f} worst {min([tot[pick[o]], *rt]):.3f} | {reps}")


if "--look" in sys.argv:
    look(sys.argv[1], int(sys.argv[2]))
    sys.exit(0)

if __name__ == "__main__":
    main()
