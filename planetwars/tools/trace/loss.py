"""Summarise a TracePair game from side A's point of view.
usage: loss.py TRACE.jsonl.gz [--every 25] [--launches T0:T1]"""
import gzip
import json
import math
import sys


def load(path):
    with gzip.open(path, "rt") as f:
        head = json.loads(f.readline())
        ticks = [json.loads(line) for line in f]
    return head, ticks


def side_of(owner):
    return {"Player1": 1, "Player2": 2}.get(owner, 0)


def totals(st, p):
    pl = st["planets"]
    mine = [x for x in pl if side_of(x["owner"]) == p]
    fleet = sum(x["transporter"]["nShips"] for x in pl if x.get("transporter") and side_of(x["transporter"]["owner"]) == p)
    return len(mine), sum(x["growthRate"] for x in mine), sum(x["nShips"] for x in mine), fleet


def main():
    path = sys.argv[1]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 25
    rng = sys.argv[sys.argv.index("--launches") + 1] if "--launches" in sys.argv else None
    head, ticks = load(path)
    a = head["a_side"]
    b = 3 - a
    fin = ticks[-1]
    print(f"seed {head['seed']} A is P{a}; {head['planets']} planets, speed {head['speed']:.2f}; ends t={fin['t']} A points {fin['a_points']}")
    st0 = ticks[0]["state"]["planets"]
    for x in st0:
        if side_of(x["owner"]) in (1, 2):
            print(f"  start P{side_of(x['owner'])} planet {x['id']} at ({x['position']['x']:.0f},{x['position']['y']:.0f}) g={x['growthRate']:.3f} ships={x['nShips']:.0f}")
    print("   t | A planets growth ships fleet | B planets growth ships fleet")
    for r in ticks:
        if r["t"] % every == 0 or r.get("final"):
            ta, tb = totals(r["state"], a), totals(r["state"], b)
            print(f"{r['t']:4d} | {ta[0]:3d} {ta[1]:.3f} {ta[2]:6.0f} {ta[3]:6.0f} | {tb[0]:3d} {tb[1]:.3f} {tb[2]:6.0f} {tb[3]:6.0f}")
    # ownership changes of planets A held
    prev = {x["id"]: side_of(x["owner"]) for x in st0}
    for r in ticks[1:]:
        for x in r["state"]["planets"]:
            o = side_of(x["owner"])
            if o != prev[x["id"]]:
                tag = "A" if o == a else ("B" if o == b else "N")
                was = "A" if prev[x["id"]] == a else ("B" if prev[x["id"]] == b else "N")
                print(f"  t={r['t']:4d} planet {x['id']:2d} (g={x['growthRate']:.3f}) {was} -> {tag}, {x['nShips']:.0f} ships left")
                prev[x["id"]] = o
    if rng:
        t0, t1 = (int(v) for v in rng.split(":"))
        for r in ticks:
            if t0 <= r["t"] <= t1 and not r.get("final"):
                pl = {x["id"]: x for x in r["state"]["planets"]}
                for side in (1, 2):
                    m = r[f"m{side}"]
                    if m.get("sourcePlanetId", -1) >= 0 and m.get("numShips", 0) > 0:
                        s, d = pl[m["sourcePlanetId"]], pl[m["destinationPlanetId"]]
                        dist = math.hypot(s["position"]["x"] - d["position"]["x"], s["position"]["y"] - d["position"]["y"])
                        who = "A" if side == a else "B"
                        dw = {a: "A", b: "B"}.get(side_of(d["owner"]), "N")
                        print(f"  t={r['t']:4d} {who}: {s['id']:2d}({s['nShips']:.0f}) -> {d['id']:2d} [{dw} {d['nShips']:.0f} g={d['growthRate']:.3f}] "
                              f"n={m['numShips']:.0f} dist={dist:.0f} eta={dist / head['speed']:.0f}")


if __name__ == "__main__":
    main()
