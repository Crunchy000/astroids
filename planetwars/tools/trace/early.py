"""Ship and growth balance (A minus B) at a few ticks for every finished TracePair game."""
import glob, gzip, json, re, sys
from loss import totals
games = {}
for log in glob.glob(sys.argv[1] + "/../hunt_h?.log"):
    for line in open(log):
        m = re.match(r"GAME seed=(\d+) a_side=(\d) ticks=(\d+) .* a_points=([\d.]+)", line)
        if m: games[(int(m[1]), int(m[2]))] = (int(m[3]), float(m[4]))
marks = [50, 100, 200, 300]
print("seed side ticks pts | " + " ".join(f"dS{t:<4d} dG{t:<4d}" for t in marks))
for (seed, side), (ticks, pts) in sorted(games.items(), key=lambda x: (x[1][1], x[0])):
    with gzip.open(f"{sys.argv[1]}/s{seed}_a{side}.jsonl.gz", "rt") as f:
        f.readline(); out = {}
        for line in f:
            r = json.loads(line)
            if r["t"] in marks:
                ta, tb = totals(r["state"], side), totals(r["state"], 3 - side)
                out[r["t"]] = (ta[2] + ta[3] - tb[2] - tb[3], ta[1] - tb[1])
            if r["t"] > max(marks): break
    print(f"{seed} {side} {ticks:5d} {pts:.1f} | " + " ".join(f"{out.get(t,(0,0))[0]:6.0f} {out.get(t,(0,0))[1]:+.3f}" for t in marks))
