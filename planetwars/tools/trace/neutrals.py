"""Neutral planets each side took in the first T ticks (growth summed), losses against wins."""
import glob, gzip, json, re, sys
from loss import side_of
T = int(sys.argv[2]) if len(sys.argv) > 2 else 150
games = {}
for log in glob.glob(sys.argv[1] + "/../hunt_h?.log"):
    for line in open(log):
        m = re.match(r"GAME seed=(\d+) a_side=(\d) ticks=(\d+) .* a_points=([\d.]+)", line)
        if m: games[(int(m[1]), int(m[2]))] = float(m[4])
agg = {"loss": [0, 0, 0.0, 0.0, 0.0, 0], "win": [0, 0, 0.0, 0.0, 0.0, 0]}
for (seed, aside), pts in games.items():
    with gzip.open(f"{sys.argv[1]}/s{seed}_a{aside}.jsonl.gz", "rt") as f:
        f.readline()
        r0 = json.loads(f.readline())
        neutral = {p["id"]: p for p in r0["state"]["planets"] if side_of(p["owner"]) == 0}
        first = {}
        for line in f:
            r = json.loads(line)
            if r["t"] > T: break
            for p in r["state"]["planets"]:
                if p["id"] in neutral and p["id"] not in first and side_of(p["owner"]) != 0:
                    first[p["id"]] = side_of(p["owner"])
    a = agg["loss" if pts < 0.5 else "win"]
    a[0] += sum(1 for v in first.values() if v == aside); a[1] += sum(1 for v in first.values() if v != aside)
    a[2] += sum(neutral[k]["growthRate"] for k, v in first.items() if v == aside)
    a[3] += sum(neutral[k]["growthRate"] for k, v in first.items() if v != aside)
    a[4] += sum(p["growthRate"] for k, p in neutral.items() if k not in first)
    a[5] += 1
for k, a in agg.items():
    n = a[5]
    print(f"{k}: {n} games; first {T} ticks, per game: we took {a[0]/n:.2f} neutrals (growth {a[2]/n:.3f}), "
          f"EgoCentric {a[1]/n:.2f} (growth {a[3]/n:.3f}), left neutral growth {a[4]/n:.3f}")
