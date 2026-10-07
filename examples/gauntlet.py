"""Score teams against a gauntlet of different opponents.

    python examples/gauntlet.py results/population_best.zip results/league_striker.zip --battles 20

The gauntlet is the default bots, three bot styles that punish different
weaknesses (``bots:rusher`` charges to point-blank range, ``bots:sniper`` holds
at long range, ``bots:zone`` sits in the control zone) and a hall of fame of
earlier champions. Every opponent is played on the same seeds from both sides.
The gauntlet score is the mean win rate over all opponents: one number that
only goes up if a team is better against *everything*, not just the bots.
"""

import argparse
import json
import os
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from league import play_matches  # noqa: E402

BOT_STYLES = ["bots", "bots:rusher", "bots:sniper", "bots:zone"]
HALL_OF_FAME = {
    "league striker": "results/league_striker.zip",
    "population m16": "results/population_best.zip",
}


def as_spec(path: str, tmp: str) -> str:
    """bots[:style] and .npz pass through; an SB3 .zip is exported to .npz."""
    if path.startswith("bots") or path.endswith(".npz"):
        return path
    from stable_baselines3 import PPO
    from league import export_policy
    return export_policy(PPO.load(path, device="cpu"), os.path.join(tmp, os.path.basename(path) + ".npz"))


def gauntlet(team: str, opponents: dict[str, str], battles: int = 20, seed: int = 500_000,
             workers: int = 4, rules: str = "competitive", bank=None) -> dict[str, tuple[float, float]]:
    """{opponent name: (team's win rate, mean score margin)}; draws count half.

    With ``bank`` (``policies.PolicyBank``) the networks run batched in this process, e.g. on a GPU.
    """
    jobs, names = [], []
    for k, (name, spec) in enumerate(opponents.items()):
        if spec == team:
            continue
        for n in range(battles):
            s = seed + 1000 * k + n // 2
            jobs.append((team, spec, s, rules) if n % 2 == 0 else (spec, team, s, rules))
            names.append(name)
    results = play_matches(jobs, workers, bank)
    out = {}
    for name in dict.fromkeys(names):
        wins, margins = [], []
        for nm, (a, b, sa, sb, outcome) in zip(names, results):
            if nm != name:
                continue
            mine = 1 if a == team else -1
            wins.append((outcome * mine + 1) / 2)
            margins.append((sa - sb) * mine)
        out[name] = (float(np.mean(wins)), float(np.mean(margins)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teams", nargs="+", help="team models (.zip or .npz) to score")
    p.add_argument("--hof", nargs="*", default=None, help="hall-of-fame models (default: league striker, m16)")
    p.add_argument("--battles", type=int, default=20, help="battles per opponent (half on each side)")
    p.add_argument("--seed", type=int, default=500_000)
    p.add_argument("--names", help="comma-separated display names for the teams (for --json)")
    p.add_argument("--json", help="also write the results here (for plot_gauntlet.py)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--device", help="run the networks batched in this process on this torch device "
                                    "(e.g. cuda); default: NumPy in worker processes")
    args = p.parse_args()
    bank = None
    if args.device:
        from astroids_gym.policies import PolicyBank
        bank = PolicyBank(args.device)
    names = args.names.split(",") if args.names else [os.path.basename(t) for t in args.teams]
    report = {"battles": args.battles, "teams": {}}

    with tempfile.TemporaryDirectory() as tmp:
        hof = ({os.path.basename(h): h for h in args.hof} if args.hof is not None else HALL_OF_FAME)
        opponents = {s: s for s in BOT_STYLES} | {n: as_spec(h, tmp) for n, h in hof.items() if os.path.exists(h)}
        for team, display in zip(args.teams, names):
            spec = as_spec(team, tmp)
            res = gauntlet(spec, opponents, args.battles, args.seed, args.workers, bank=bank)
            score = np.mean([w for w, _ in res.values()])
            print(f"\n{team}: gauntlet score {score:.0%} ({args.battles} battles per opponent)")
            for name, (w, m) in res.items():
                print(f"  vs {name:16} wins {w:4.0%}  score margin {m:+5.1f}")
            report["teams"][display] = {"score": float(score), "vs": {n: list(v) for n, v in res.items()}}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
