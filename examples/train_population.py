"""Population-based training (PBT): a population of team models that evolves.

    pip install -e ".[train]"
    python examples/train_population.py --founders results/league_striker.zip:striker,results/league_pack.zip:pack,results/league_guardian.zip:guardian --size 8

Each member is a shared-policy team model with a *genome*: its reward weights,
learning rate, entropy bonus and network width. Founders are trained models
(e.g. the league finalists); the population is filled with mutated copies.

Every round, each member trains in turn (against the other members, past
snapshots and the bots, on a random side), then everyone plays a side-swapped
round-robin tournament; Elo has the built-in bots at 1000. Every
``--exploit-every`` rounds, the bottom quarter of the population is replaced
by copies of the top quarter (weights, optimiser state and genome), and each
copy mutates: reward weights, learning rate and entropy are scaled by 0.8 or
1.25, and with probability ``--p-grow`` the network widens by 64 units per
layer. Widening is function-preserving (see ``train_league.grow_model``), so a
grown network plays exactly as before and selection decides whether the extra
capacity pays off. Lineage (who descends from which founder) is recorded.

Everything goes to ``--out``: snapshots, ``population.json`` and charts.
"""

import argparse
import copy
import json
import os
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # see train_league.py

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from league import export_policy, fit_elo, tournament, win_matrix  # noqa: E402
from train_league import (PERSONALITIES, LeagueVecEnv, ThreadSwitch, grow_model,  # noqa: E402
                          net_widths, opponent_mix)


def mutate(genome: dict, rng, p_grow: float, max_width: int) -> tuple[dict, list[str]]:
    """Scale each reward weight (p = 0.5), the learning rate and entropy by 0.8 or 1.25; maybe widen."""
    g = copy.deepcopy(genome)
    changes = []
    for k, v in g["reward"].items():
        if v and rng.random() < 0.5:
            f = float(rng.choice([0.8, 1.25]))
            g["reward"][k] = round(v * f, 4)
            changes.append(f"{k}×{f}")
    g["reward"]["team_spirit"] = min(g["reward"].get("team_spirit", 0.0), 1.0)
    for k, lo, hi in (("lr", 2e-5, 5e-4), ("ent_coef", 5e-4, 2e-2)):
        f = float(rng.choice([0.8, 1.25]))
        g[k] = float(np.clip(g[k] * f, lo, hi))
        changes.append(f"{k}×{f}")
    if rng.random() < p_grow and max(g["widths"]) + 64 <= max_width:
        g["widths"] = [w + 64 for w in g["widths"]]
        changes.append(f"widen→{g['widths'][0]}")
    return g, changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--founders", required=True, help="comma-separated path.zip:personality")
    p.add_argument("--size", type=int, default=8, help="population size")
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--samples-per-round", type=int, default=400_000, help="ship-steps per member per round")
    p.add_argument("--exploit-every", type=int, default=2)
    p.add_argument("--p-grow", type=float, default=0.3, help="chance a mutated copy widens its network")
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--bots-share", type=float, default=0.4)
    p.add_argument("--n-workers", type=int, default=os.cpu_count())
    p.add_argument("--battles-per-worker", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--tournament-battles", type=int, default=16, help="battles per pair per round")
    p.add_argument("--rules", choices=["competitive", "classic"], default="competitive")
    p.add_argument("--out", default="population")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    os.makedirs(f"{args.out}/models", exist_ok=True)
    venv = LeagueVecEnv(args.n_workers, args.battles_per_worker, seed=args.seed, rules=args.rules)
    base_hyper = {"n_steps": args.n_steps, "batch_size": args.n_steps * venv.num_envs // 4}

    def hyper(g):
        return {**base_hyper, "learning_rate": g["lr"], "ent_coef": g["ent_coef"]}

    def load(zip_path, g):
        return PPO.load(zip_path, env=venv, custom_objects=hyper(g), device="cpu")

    def save(member, rnd):
        path = f"{args.out}/models/{member['id']}_r{rnd}"
        member["model"].save(path)
        export_policy(member["model"], path + ".npz")
        member["latest"] = path + ".npz"
        member["latest_zip"] = path + ".zip"

    if args.resume:
        with open(f"{args.out}/population.json") as f:
            state = json.load(f)
        done = len(state["rounds"])
        members = []
        for m in state["population"]:
            m = dict(m)
            m["model"] = load(m["latest_zip"], m["genome"])
            members.append(m)
        snapshots = state["snapshots"]
        next_id = state["next_id"]
        print(f"resuming after round {done}", flush=True)
    else:
        founders = [spec.split(":") for spec in args.founders.split(",")]
        members, next_id = [], 0
        for i in range(args.size):
            path, personality = founders[i % len(founders)]
            founder_model = PPO.load(path, device="cpu")
            genome = {"reward": dict(PERSONALITIES[personality]), "lr": 1e-4, "ent_coef": 0.003,
                      "widths": net_widths(founder_model)}
            changes = []
            if i >= len(founders):  # founders themselves stay unmutated
                genome, changes = mutate(genome, rng, args.p_grow, args.max_width)
            model = grow_model(founder_model, venv, hyper(genome), args.seed + i, genome["widths"])
            members.append({"id": f"m{next_id:02d}", "founder": personality, "parent": None, "born": 0,
                            "genome": genome, "mutations": changes, "model": model})
            next_id += 1
        for m in members:
            save(m, 0)
        snapshots = [m["latest"] for m in members]
        state = {"rules": args.rules, "size": args.size, "founders": [f[1] for f in founders],
                 "rounds": [], "events": []}
        done = 0
    switch = ThreadSwitch(os.cpu_count())

    def persist():
        state["population"] = [{k: v for k, v in m.items() if k != "model"} for m in members]
        state["snapshots"] = snapshots
        state["next_id"] = next_id
        with open(f"{args.out}/population.json", "w") as f:
            json.dump(state, f, indent=1)

    persist()
    for rnd in range(done + 1, args.rounds + 1):
        log = {"round": rnd, "training": {}}
        t_round = time.time()
        latest = {m["id"]: m["latest"] for m in members}
        for m in members:
            venv.configure(reward=m["genome"]["reward"],
                           opponents=opponent_mix(m["id"], latest, snapshots, rng, args.bots_share))
            venv.results.clear()
            m["model"]._last_obs = None
            t = time.time()
            m["model"].learn(args.samples_per_round, callback=switch, reset_num_timesteps=False)
            vs_bots = [(o + 1) / 2 for spec, _, o in venv.results if spec == "bots"]
            log["training"][m["id"]] = {"samples_per_s": round(args.samples_per_round / (time.time() - t)),
                                        "win_vs_bots": round(float(np.mean(vs_bots)), 3) if vs_bots else None}
            save(m, rnd)
            snapshots.append(m["latest"])

        players = [m["latest"] for m in members] + ["bots"]
        results = tournament(players, args.tournament_battles, seed=10_000 * rnd, workers=args.n_workers,
                             rules=args.rules)
        elo = fit_elo(results, anchor="bots")
        name = {m["latest"]: m["id"] for m in members} | {"bots": "bots"}
        log["elo"] = {name[p]: round(e) for p, e in elo.items()}
        log["matrix"] = {f"{name[a]}|{name[b]}": [round(w, 3), round(mm, 2)]
                         for (a, b), (w, mm) in win_matrix(results).items()}
        log["members"] = [{"id": m["id"], "founder": m["founder"], "widths": m["genome"]["widths"],
                           "parent": m["parent"]} for m in members]
        ranked = sorted(members, key=lambda m: -log["elo"][m["id"]])
        print(f"round {rnd} ({time.time() - t_round:.0f}s) elo: "
              + "  ".join(f"{m['id']}[{m['founder'][:3]},{m['genome']['widths'][0]}] {log['elo'][m['id']]}"
                          for m in ranked), flush=True)

        # Exploit and explore: the bottom quarter becomes mutated copies of the top quarter.
        if rnd % args.exploit_every == 0 and rnd < args.rounds:
            k = max(1, len(members) // 4)
            winners, losers = ranked[:k], ranked[-k:]
            for loser in losers:
                parent = winners[int(rng.integers(len(winners)))]
                genome, changes = mutate(parent["genome"], rng, args.p_grow, args.max_width)
                model = load(parent["latest_zip"], genome)
                if genome["widths"] != parent["genome"]["widths"]:
                    model = grow_model(model, venv, hyper(genome), args.seed + next_id, genome["widths"])
                child = {"id": f"m{next_id:02d}", "founder": parent["founder"], "parent": parent["id"],
                         "born": rnd, "genome": genome, "mutations": changes, "model": model}
                next_id += 1
                save(child, rnd)
                members[members.index(loser)] = child
                state["events"].append({"round": rnd, "replaced": loser["id"], "replaced_elo": log["elo"][loser["id"]],
                                        "child": child["id"], "parent": parent["id"],
                                        "parent_elo": log["elo"][parent["id"]], "mutations": changes})
                print(f"  {loser['id']} ({log['elo'][loser['id']]}) replaced by {child['id']} <- {parent['id']} "
                      f"({log['elo'][parent['id']]}): {', '.join(changes)}", flush=True)
        state["rounds"].append(log)
        persist()
        try:
            from plot_population import plot_population
            plot_population(f"{args.out}/population.json")
        except ImportError:
            pass
    venv.close()


if __name__ == "__main__":
    main()
