"""Command line for the competition: serve an agent, play a match, run a league, evaluate an entry.

    python -m astroids_gym.competition serve competition/models/m10.npz --port 8080
    python -m astroids_gym.competition match bots competition/models/m10.npz --seed 3 --realtime
    python -m astroids_gym.competition league bots m10=competition/models/m10.npz rival=ws://localhost:8080
    python -m astroids_gym.competition evaluate ws://localhost:8080 --games 10
    python -m astroids_gym.competition submission --issue-file issue.md --out report.md

Agents are spec strings (``agents.make_agent``): ``bots[:style]``, ``idle``, ``random``,
a ``.npz``/``.zip`` model, ``ws://host:port``, ``package.module:Class``. In the
league, ``docker:<dir>`` builds and runs the entry in ``<dir>`` and ``image:<name>``
runs a built image. Name an entry with ``name=spec``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _named(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, spec = item.partition("=")
        if not sep:
            spec = item
            base = os.path.basename(item.rstrip("/"))
            name = os.path.splitext(base)[0] if item.endswith((".npz", ".zip")) else item
        if name in out:
            raise SystemExit(f"two entries named {name!r}: name them with name=spec")
        out[name] = spec
    return out


def cmd_serve(a) -> None:
    from .agents import make_agent
    from .protocol import serve
    make_agent(a.agent)  # fail now, not on the first match
    print(f"serving {a.agent}", flush=True)
    serve(lambda: make_agent(a.agent), a.host, a.port, verbose=True)


def cmd_match(a) -> None:
    from .agents import make_agent
    from .params import random_params
    from .runner import run_match
    params = random_params(a.seed, fixed=a.fixed)
    res = run_match(make_agent(a.blue), make_agent(a.red), params, seed=a.seed, realtime=a.realtime,
                    names=(a.blue, a.red))
    flight = "flight physics" if params["flight"] else "classic physics"
    print(f"settings: {params['asteroid_count']} asteroids, {params['team_lives']} respawns, zone "
          f"{params['zone_radius']:.0f} px, gun range {params['bullet_speed'] * params['bullet_life']:.0f} px, {flight}")
    for t in ("blue", "red"):
        s = res.stats[t]
        print(f"{t:4} {res.names[t]}: {res.score[t]:.1f} ({res.kills[t]} kills + {res.points[t]:.1f} zone)"
              f"  decisions {s.actions}, late {s.late}, errors {s.errors}, rejected {s.rejected},"
              f" mean {s.mean_ms:.1f} ms, max {s.max_ms:.1f} ms" + (f", FORFEIT {s.forfeit}" if s.forfeit else ""))
    print(f"winner: {res.winner}{' (wiped out)' if res.eliminated else ''} after {res.steps} steps,"
          f" {res.wall_seconds:.1f} s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res.to_json(), f, indent=1)


def cmd_league(a) -> None:
    from . import docker
    from .league import leaderboard_markdown, run_league
    entries = _named(a.entries)
    if a.entries_file:
        with open(a.entries_file) as f:
            entries |= {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    containers = []
    try:
        for name, spec in list(entries.items()):  # containers live for the whole league
            if spec.startswith(("docker:", "image:")):
                kind, _, target = spec.partition(":")
                image = docker.build(target, f"astorids-entry-{name.lower()}") if kind == "docker" else target
                c = docker.Container(image, internal=a.internal).start()
                containers.append(c)
                entries[name] = c.url
                print(f"{name}: {image} at {c.url}", flush=True)
        results = run_league(entries, a.games_per_pair, a.seed, realtime=not a.turn_based, workers=a.workers,
                             results_path=a.results, fixed=a.fixed)
    finally:
        for c in containers:
            c.stop()
    md = leaderboard_markdown(results, a.title)
    if a.leaderboard:
        with open(a.leaderboard, "w") as f:
            f.write(md)
    print(md)


def cmd_evaluate(a) -> None:
    from .submissions import baselines, evaluate, report_markdown
    opponents = _named(a.opponents) if a.opponents else baselines()
    results = evaluate(a.agent, a.name, opponents, games=a.games, realtime=not a.turn_based)
    md = report_markdown(results, a.name)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    print(md)


def cmd_submission(a) -> None:
    from .submissions import evaluate_submission
    if a.issue_body_env:
        body = os.environ.get(a.issue_body_env, "")
    else:
        with open(a.issue_file) as f:
            body = f.read()
    md, results = evaluate_submission(body, games=a.games)
    with open(a.out, "w") as f:
        f.write(md)
    if a.results:
        with open(a.results, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in results))
    print(md)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m astroids_gym.competition", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve one agent over WebSocket (a container's entry point)")
    s.add_argument("agent")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("match", help="play one match")
    s.add_argument("blue")
    s.add_argument("red")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--realtime", action="store_true", help="enforce the decision time budget")
    s.add_argument("--fixed", action="store_true", help="plain competitive rules instead of random settings")
    s.add_argument("--json", help="write the result here")
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("league", help="round-robin league with TrueSkill ratings")
    s.add_argument("entries", nargs="*", help="name=spec (or just spec)")
    s.add_argument("--entries-file", help='JSON {"name": "spec", ...} added to the entries')
    s.add_argument("--games-per-pair", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--turn-based", action="store_true", help="wait for every answer instead of real time")
    s.add_argument("--fixed", action="store_true", help="plain competitive rules instead of random settings")
    s.add_argument("--workers", type=int, default=1, help="matches at once (keep well below the core count)")
    s.add_argument("--results", default="league_results.jsonl", help="JSON lines; rerunning only plays what's missing")
    s.add_argument("--leaderboard", default="leaderboard.md")
    s.add_argument("--title", default="Astorids league")
    s.add_argument("--internal", action="store_true", help="run containers without internet access")
    s.set_defaults(fn=cmd_league)

    s = sub.add_parser("evaluate", help="score an agent against the baseline teams")
    s.add_argument("agent")
    s.add_argument("--name", default="entry")
    s.add_argument("--opponents", nargs="*", help="name=spec (default: the baselines)")
    s.add_argument("--games", type=int, default=10, help="per opponent, half on each side")
    s.add_argument("--turn-based", action="store_true")
    s.add_argument("--out", help="write the Markdown report here")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("submission", help="evaluate the entry described in a submission issue")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--issue-file")
    src.add_argument("--issue-body-env", help="read the issue body from this environment variable")
    s.add_argument("--games", type=int, default=10)
    s.add_argument("--out", default="report.md")
    s.add_argument("--results", help="write the match results here (JSON lines)")
    s.set_defaults(fn=cmd_submission)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
