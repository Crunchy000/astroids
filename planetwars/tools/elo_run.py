"""Play a set of head-to-head pairings with tools/PairEval.kt and append the results to a CSV for tools/elo.py.

Agents are either sample agents of their repo run inside the Kotlin process (a key such as "evo") or servers that
are already listening on a local port (remoteNNNN). Pairs run a few at a time; a server plays one game at a
time, so two pairs that need the same server never overlap.

    python tools/elo_run.py --classpath "$CP" --agents agents.json --out results/elo_games.csv \
        --sets 2 --per-side 2 --source stage1 --workers 3

agents.json maps a name to an agent key ("evo", "titans", ...) or to "remote<PORT>". With --pairs a.vs.b ...
only those pairs are played; otherwise every pair. Pairs already in the CSV under the same --source are skipped.
"""
import argparse
import csv
import itertools
import json
import pathlib
import re
import subprocess
import threading
import time

RESULT = re.compile(r"RESULT a=(\S+) b=(\S+) a_wins=([\d.]+) b_wins=([\d.]+) games=(\d+) a_timeouts=(\d+) b_timeouts=(\d+) a_ms=([\d.]+) b_ms=([\d.]+)")
FIELDS = ["a", "b", "a_wins", "b_wins", "draws", "source", "a_timeouts", "b_timeouts", "a_ms", "b_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classpath", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--timeout-ms", type=int, default=200)
    ap.add_argument("--source", default="run")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pairs", nargs="*", default=None, help="a.vs.b pairs; default all pairs")
    ap.add_argument("--games-for", nargs="*", default=[], help="a.vs.b=SETS to override the number of sets for a pair")
    args = ap.parse_args()

    agents = json.load(open(args.agents))
    spec = lambda n: agents[n]  # noqa: E731
    pairs = [tuple(p.split(".vs.")) for p in args.pairs] if args.pairs else list(itertools.combinations(sorted(agents), 2))
    override = {tuple(x.split("=")[0].split(".vs.")): int(x.split("=")[1]) for x in args.games_for}
    out = pathlib.Path(args.out)
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            if r["source"] == args.source:
                done.add((r["a"], r["b"]))
    todo = [p for p in pairs if p not in done]
    print(f"{len(todo)} pairs to play, {len(done)} already done", flush=True)

    lock = threading.Lock()
    busy = set()
    write_lock = threading.Lock()
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    def remote_of(n):
        return n if spec(n).startswith("remote") else None

    def worker():
        while True:
            with lock:
                pick = None
                for p in todo:
                    need = {x for x in map(remote_of, p) if x}
                    if not (need & busy):
                        pick = p
                        busy.update(need)
                        todo.remove(p)
                        break
                if pick is None and not todo:
                    return
            if pick is None:
                time.sleep(2)
                continue
            a, b = pick
            sets = override.get(pick, args.sets)
            t0 = time.time()
            cmd = ["java", "-cp", args.classpath, "games.planetwars.runners.PairEvalKt",
                   f"{spec(a)},{spec(b)},{sets},{args.per_side},{args.timeout_ms}"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5400).stdout
                m = RESULT.search(res)
            except subprocess.TimeoutExpired:
                m = None
            with lock:
                busy.difference_update({x for x in map(remote_of, pick) if x})
            if not m:
                print(f"FAILED {a} vs {b}", flush=True)
                continue
            games = int(m.group(5))
            aw, bw = float(m.group(3)), float(m.group(4))
            row = [a, b, aw, bw, games - aw - bw, args.source, m.group(6), m.group(7), m.group(8), m.group(9)]
            with write_lock:
                with open(out, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            print(f"{a} vs {b}: {aw:.0f}-{bw:.0f} of {games} ({time.time() - t0:.0f} s)", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
