"""What became of every launch in TracePair games: by side, target kind and flight time, how many captured their target
and how many ships went into launches that did not. usage: fleets.py DIR [losses|wins|all]"""
import glob, gzip, json, math, re, sys
from collections import defaultdict
from loss import side_of

games = {}
for log in glob.glob(sys.argv[1] + "/../hunt_h?.log"):
    for line in open(log):
        m = re.match(r"GAME seed=(\d+) a_side=(\d) ticks=(\d+) .* a_points=([\d.]+)", line)
        if m: games[(int(m[1]), int(m[2]))] = float(m[4])
which = sys.argv[2] if len(sys.argv) > 2 else "all"
stats = defaultdict(lambda: [0, 0, 0.0, 0.0])  # launches, captured, ships sent, ships sent in failed ones
for (seed, aside), pts in games.items():
    if which == "losses" and pts > 0.5 or which == "wins" and pts < 0.5:
        continue
    with gzip.open(f"{sys.argv[1]}/s{seed}_a{aside}.jsonl.gz", "rt") as f:
        head = json.loads(f.readline())
        rows = [json.loads(l) for l in f]
    pending = {}  # source planet -> (who, kind, eta bucket, n, dst, tick)
    for r in rows:
        pl = {x["id"]: x for x in r["state"]["planets"]}
        for src, (who, kind, bucket, n, dst, t0) in list(pending.items()):
            if pl[src].get("transporter") is None and r["t"] > t0 + 1:
                ok = side_of(pl[dst]["owner"]) == who
                k = ("A" if who == aside else "B", kind, bucket)
                stats[k][0] += 1; stats[k][1] += ok; stats[k][2] += n; stats[k][3] += 0 if ok else n
                del pending[src]
        if r.get("final"):
            break
        for side in (1, 2):
            m = r[f"m{side}"]
            if m.get("sourcePlanetId", -1) >= 0 and m.get("numShips", 0) > 0:
                s, d = pl[m["sourcePlanetId"]], pl[m["destinationPlanetId"]]
                if side_of(s["owner"]) != side or s.get("transporter") is not None or s["nShips"] < m["numShips"]:
                    continue
                o = side_of(d["owner"])
                kind = "own" if o == side else ("neutral" if o == 0 else "enemy")
                eta = math.hypot(s["position"]["x"] - d["position"]["x"], s["position"]["y"] - d["position"]["y"]) / head["speed"]
                bucket = "<25" if eta < 25 else "25-50" if eta < 50 else "50-75" if eta < 75 else "75+"
                pending[s["id"]] = (side, kind, bucket, m["numShips"], d["id"], r["t"])
n_games = sum(1 for p in games.values() if not (which == "losses" and p > 0.5 or which == "wins" and p < 0.5))
print(f"{which}: {n_games} games (A = our entry, B = EgoCentric); 'took' = target ours when the fleet lands")
print("side kind     eta    launches  took  ships/game  lost ships/game")
for k in sorted(stats):
    if k[1] == "own":
        continue
    n, ok, sent, lost = stats[k]
    print(f"{k[0]:4s} {k[1]:8s} {k[2]:6s} {n/n_games:8.1f} {ok/max(n,1):5.2f} {sent/n_games:10.1f} {lost/n_games:10.1f}")
for side in ("A", "B"):
    n = sum(v[0] for k, v in stats.items() if k[0] == side and k[1] == "own"); sh = sum(v[2] for k, v in stats.items() if k[0] == side and k[1] == "own")
    print(f"{side} to own planets: {n/n_games:.1f} launches, {sh/n_games:.0f} ships per game")
