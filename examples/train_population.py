"""Population-based training (PBT): a population of team models that evolves.

    pip install -e ".[train]"
    python examples/train_population.py --size 10 --exploiters 2 \
        --founders results/population_best.zip:evolved,results/population_best.zip:pack,results/population_best.zip:guardian

Each member is a shared-policy team model with a *genome*: its reward weights,
learning rate, entropy bonus and network width. Founders are trained models
plus a reward personality; each founder starts a *line*, and the population
is filled with mutated copies of the founders.

Every round, each member trains in turn on a random side, then the members,
the starting model and the bots play a side-swapped round-robin tournament;
Elo has the built-in bots at 1000. Every ``--exploit-every`` rounds, the
bottom quarter of the population is replaced by copies of the top quarter
(weights, optimiser state and genome), and each copy mutates: reward weights,
learning rate and entropy are scaled by 0.8 or 1.25, and with probability
``--p-grow`` the network widens by 64 units per layer. Widening is
function-preserving (see ``train_league.grow_model``), so a grown network plays
exactly as before and selection decides whether the extra capacity pays off.

Upgrades over plain PBT (each can be switched off to compare):

* **Centralised critic** (``--critic central``): the value network also sees
  the whole battle (``astroids_gym.policies.CentralCriticPolicy``); the actor
  still flies from its own ship's view.
* **Prioritised opponents** (``--sampling pfsp``): every member keeps a record
  of its results against each opponent (training battles and tournaments), and
  picks opponents with weight (1 - p)^2, p = its win rate against them: hard
  opponents come up often, ones it always beats rarely. The bots' share is split
  over the bot styles (``bot.STYLES``) the same way.
* **Exploiters** (``--exploiters``): extra models outside the population that
  train only against the current best member. When one beats it (``--exploit-win``
  in a 40-battle check) its snapshot joins every member's opponent pool, so the
  population has to fix the weakness it found; then it restarts as a copy of
  the new best. After ``--exploiter-patience`` rounds without a win it restarts
  anyway.
* **Line protection**: selection never removes the last member of a founder
  line, so different reward personalities stay in the population (and in each
  other's opponent pools) instead of one line taking over.

Lineage (who descends from which founder) is recorded. Everything goes to
``--out``: snapshots, ``population.json`` and charts.
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

from astroids_gym.bot import FORMATIONS, MANEUVERS, PAIRS, STYLES  # noqa: E402
from league import FRAME_SKIP, MAX_STEPS, export_policy, fit_elo, make_rules, tournament, win_matrix  # noqa: E402
from train_league import (PERSONALITIES, LeagueVecEnv, ThreadSwitch, grow_model,  # noqa: E402
                          net_widths, opponent_mix)

BOT_SPECS = ["bots"] + [f"bots:{s}" for s in STYLES + FORMATIONS + PAIRS + MANEUVERS]
# Exploiters only care about beating their target.
EXPLOITER_REWARD = {**PERSONALITIES["evolved"], "win": 6.0}


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


def win_prob(record: dict, spec: str) -> float:
    """Win rate against ``spec`` with one virtual win and one virtual loss (0.5 if never met)."""
    w, n = record.get(spec, (0.0, 0))
    return (w + 1.0) / (n + 2.0)


def pfsp_mix(record, pool, bots, rng, bots_share, max_past=8):
    """Prioritised fictitious self-play: weight (1 - p)^2 within the league pool and within the bots.

    ``pool`` is [(spec, group)], group "latest" (always kept) or "past"/"exploiter"
    (at most ``max_past`` of each are drawn, favouring hard ones).
    """
    def weight(spec):
        return max((1.0 - win_prob(record, spec)) ** 2, 0.02)  # never drop an opponent entirely

    chosen = [spec for spec, g in pool if g == "latest"]
    for group in ("past", "exploiter"):
        specs = [spec for spec, g in pool if g == group]
        if len(specs) > max_past:
            w = np.array([weight(x) for x in specs])
            specs = list(rng.choice(specs, max_past, replace=False, p=w / w.sum()))
        chosen += specs
    out = []
    for specs, share in ((chosen, 1.0 - bots_share), (bots, bots_share)):
        if not specs:
            continue
        w = np.array([weight(x) for x in specs])
        out += [(x, float(share * wi / w.sum())) for x, wi in zip(specs, w)]
    total = sum(w for _, w in out)
    return [(x, w / total) for x, w in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--founders", required=True, help="comma-separated path.zip:personality")
    p.add_argument("--size", type=int, default=8, help="population size")
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--samples-per-round", type=int, default=400_000, help="ship-steps per member per round")
    p.add_argument("--exploit-every", type=int, default=2)
    p.add_argument("--p-grow", type=float, default=0.3, help="chance a mutated copy widens its network")
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--bots-share", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4, help="founders' learning rate (mutations scale it from there)")
    p.add_argument("--critic", choices=["central", "local"], default="central")
    p.add_argument("--sampling", choices=["pfsp", "uniform"], default="pfsp")
    p.add_argument("--exploiters", type=int, default=2)
    p.add_argument("--exploit-win", type=float, default=0.65, help="win rate an exploiter needs vs its target")
    p.add_argument("--exploiter-patience", type=int, default=4, help="rounds before an exploiter restarts")
    p.add_argument("--no-line-protection", action="store_true")
    p.add_argument("--n-workers", type=int, default=os.cpu_count())
    p.add_argument("--battles-per-worker", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--tournament-battles", type=int, default=12, help="battles per pair per round")
    p.add_argument("--rules", choices=["competitive", "classic", "dogfight"], default="competitive")
    p.add_argument("--out", default="population")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="torch device for training: auto (a GPU if there is one), cuda, cpu")
    p.add_argument("--inference", choices=["auto", "main", "workers"], default="auto",
                   help="where frozen opponents and tournament players run: in this process on --device "
                        "(best with a GPU), or as NumPy in the worker processes (best on CPU only); auto picks")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    inference = args.inference if args.inference != "auto" else ("main" if device != "cpu" else "workers")
    bank = None
    if inference == "main":
        from astroids_gym.policies import PolicyBank
        bank = PolicyBank(device)
    print(f"training on {device}; opponents and tournaments run in the "
          f"{'main process' if bank else 'worker processes'}; {args.n_workers} workers", flush=True)

    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    os.makedirs(f"{args.out}/models", exist_ok=True)
    venv = LeagueVecEnv(args.n_workers, args.battles_per_worker, seed=args.seed, rules=args.rules,
                        central=args.critic == "central", bank=bank)
    # Each member's turn resets every battle, so a turn shorter than one battle never
    # trains on the late game (zone fights, the result). It needs samples-per-round
    # >= battle steps x env slots (n_workers x battles_per_worker x 5).
    battle_steps = min(MAX_STEPS, round(make_rules(args.rules)[0].battle_seconds * 60 / FRAME_SKIP))
    if args.samples_per_round < battle_steps * venv.num_envs:
        print(f"WARNING: {args.samples_per_round:,} samples per round over {venv.num_envs} ship slots is "
              f"{args.samples_per_round // venv.num_envs} steps each, less than one {battle_steps}-step battle: "
              f"no battle will finish. Lower --battles-per-worker or raise --samples-per-round.", flush=True)
    base_hyper = {"n_steps": args.n_steps, "batch_size": args.n_steps * venv.num_envs // 4}
    bots = BOT_SPECS if args.sampling == "pfsp" else ["bots"]

    def hyper(g):
        return {**base_hyper, "learning_rate": g["lr"], "ent_coef": g["ent_coef"]}

    def load(zip_path, g):
        return PPO.load(zip_path, env=venv, custom_objects=hyper(g), device=device)

    def save(member, rnd):
        path = f"{args.out}/models/{member['id']}_r{rnd}"
        member["model"].save(path)
        export_policy(member["model"], path + ".npz")
        member["latest"] = path + ".npz"
        member["latest_zip"] = path + ".zip"

    def new_exploiter(source, rnd):
        """An exploiter restarting as a copy of ``source`` (a member), with the exploiter reward."""
        nonlocal next_id
        genome = {**copy.deepcopy(source["genome"]), "reward": dict(EXPLOITER_REWARD)}
        x = {"id": f"x{next_id:02d}", "founder": source["founder"], "parent": source["id"], "born": rnd,
             "genome": genome, "target": None, "tries": 0, "model": load(source["latest_zip"], genome)}
        next_id += 1
        save(x, rnd)
        return x

    if args.resume:
        with open(f"{args.out}/population.json") as f:
            state = json.load(f)
        done = len(state["rounds"])
        members, exploiters = [], []
        for group, src in ((members, state["population"]), (exploiters, state.get("exploiter_models", []))):
            for m in src:
                m = dict(m)
                m["model"] = load(m["latest_zip"], m["genome"])
                group.append(m)
        snapshots, exploit_snaps = state["snapshots"], state.get("exploit_snapshots", [])
        records, next_id, start = state.get("records", {}), state["next_id"], state["start"]
        print(f"resuming after round {done}", flush=True)
    else:
        founders = [spec.split(":") for spec in args.founders.split(",")]
        members, exploiters, next_id = [], [], 0
        for i in range(args.size):
            path, personality = founders[i % len(founders)]
            founder_model = PPO.load(path, device="cpu")
            genome = {"reward": dict(PERSONALITIES[personality]), "lr": args.lr, "ent_coef": 0.003,
                      "widths": net_widths(founder_model)}
            changes = []
            if i >= len(founders):  # founders themselves stay unmutated
                genome, changes = mutate(genome, rng, args.p_grow, args.max_width)
            model = grow_model(founder_model, venv, hyper(genome), args.seed + i, genome["widths"], device)
            members.append({"id": f"m{next_id:02d}", "founder": personality, "parent": None, "born": 0,
                            "genome": genome, "mutations": changes, "model": model})
            next_id += 1
        for m in members:
            save(m, 0)
        # The first founder's model, unchanged, is the fixed reference every round.
        start = members[0]["latest"].replace("_r0.npz", "_start.npz")
        export_policy(members[0]["model"], start)
        snapshots = [m["latest"] for m in members]
        exploit_snaps, records = [], {}
        exploiters = [new_exploiter(members[0], 0) for _ in range(args.exploiters)]
        state = {"rules": args.rules, "size": args.size, "founders": [f[1] for f in founders],
                 "start": start, "rounds": [], "events": [], "exploiter_events": [],
                 "settings": {"critic": args.critic, "sampling": args.sampling, "exploiters": args.exploiters,
                              "line_protection": not args.no_line_protection, "bots_share": args.bots_share,
                              "samples_per_round": args.samples_per_round}}
        done = 0
    switch = ThreadSwitch(os.cpu_count())

    def persist():
        state["population"] = [{k: v for k, v in m.items() if k != "model"} for m in members]
        state["exploiter_models"] = [{k: v for k, v in x.items() if k != "model"} for x in exploiters]
        state["snapshots"], state["exploit_snapshots"] = snapshots, exploit_snaps
        state["records"], state["next_id"] = records, next_id
        with open(f"{args.out}/population.json", "w") as f:
            json.dump(state, f, indent=1)

    def remember(member_id, results):
        rec = records.setdefault(member_id, {})
        for spec, outcome in results:
            w, n = rec.get(spec, (0.0, 0))
            rec[spec] = (w + (outcome + 1) / 2, n + 1)

    def train(m, opponents):
        venv.configure(reward=m["genome"]["reward"], opponents=opponents)
        venv.results.clear()
        m["model"]._last_obs = None
        t = time.time()
        m["model"].learn(args.samples_per_round, callback=switch, reset_num_timesteps=False)
        remember(m["id"], [(spec, o) for spec, _, o in venv.results])
        return args.samples_per_round / (time.time() - t)

    persist()
    best = members[0]
    for rnd in range(done + 1, args.rounds + 1):
        log = {"round": rnd, "training": {}, "exploiters": {}}
        t_round = time.time()
        if state["rounds"]:
            prev = state["rounds"][-1]["elo"]
            best = max(members, key=lambda m: prev.get(m["id"], -1e9))
        latest = {m["id"]: m["latest"] for m in members}
        current = set(latest.values())
        for m in members:
            if args.sampling == "pfsp":
                pool = ([(latest[o], "latest") for o in latest if o != m["id"]]
                        + [(s, "past") for s in snapshots if s not in current]
                        + [(s, "exploiter") for s in exploit_snaps])
                opponents = pfsp_mix(records.get(m["id"], {}), pool, bots, rng, args.bots_share)
            else:
                opponents = opponent_mix(m["id"], latest, snapshots + exploit_snaps, rng, args.bots_share)
            rate = train(m, opponents)
            kinds = {}
            for spec, _, o in venv.results:
                kind = (spec if spec.startswith("bots") else "members" if spec in current
                        else "exploiters" if spec in exploit_snaps else "snapshots")
                kinds.setdefault(kind, []).append((o + 1) / 2)
            log["training"][m["id"]] = {"samples_per_s": round(rate),
                                        "win_rate_vs": {k: round(float(np.mean(v)), 3) for k, v in kinds.items()}}
            wins = [w for v in kinds.values() for w in v]
            print(f"  {m['id']} trained: {rate:,.0f} samples/s, won {np.mean(wins) if wins else 0:.0%} "
                  f"of {len(wins)} training battles", flush=True)
            save(m, rnd)
            snapshots.append(m["latest"])

        # Exploiters train only against the best member (by last round's Elo).
        for i, x in enumerate(exploiters):
            if x["target"] != best["id"]:
                x["target"], x["tries"] = best["id"], 0
            rate = train(x, [(best["latest"], 1.0)])
            save(x, rnd)
            x["tries"] += 1
            check = tournament([x["latest"], best["latest"]], 40, seed=900_000 + 1000 * rnd + 100 * i,
                               workers=args.n_workers, rules=args.rules, bank=bank)
            win = float(np.mean([(o * (1 if a == x["latest"] else -1) + 1) / 2 for a, _, _, _, o in check]))
            remember(best["id"], [(x["latest"], o * (1 if b == x["latest"] else -1)) for a, b, _, _, o in check])
            event = {"round": rnd, "exploiter": x["id"], "target": best["id"], "win": round(win, 3),
                     "samples_per_s": round(rate)}
            if win >= args.exploit_win:
                exploit_snaps.append(x["latest"])
                event["result"] = "added to pools"
            elif x["tries"] >= args.exploiter_patience:
                event["result"] = "gave up"
            if "result" in event:  # restart from the newest version of the best member
                exploiters[i] = new_exploiter(best, rnd)
                event["restarted_as"] = exploiters[i]["id"]
            state["exploiter_events"].append(event)
            log["exploiters"][x["id"]] = event
            print(f"  exploiter {x['id']} vs {best['id']}: {win:.0%} {event.get('result', '')}", flush=True)

        players = [m["latest"] for m in members] + [start, "bots"]
        results = tournament(players, args.tournament_battles, seed=10_000 * rnd, workers=args.n_workers,
                             rules=args.rules, bank=bank)
        name = {m["latest"]: m["id"] for m in members} | {start: "start", "bots": "bots"}
        for m in members:  # tournament results go into each member's record too
            remember(m["id"], [(b if a == m["latest"] else a, o if a == m["latest"] else -o)
                               for a, b, _, _, o in results if m["latest"] in (a, b)])
        elo = fit_elo(results, anchor="bots")
        log["elo"] = {name[p]: round(e) for p, e in elo.items()}
        log["matrix"] = {f"{name[a]}|{name[b]}": [round(w, 3), round(mm, 2)]
                         for (a, b), (w, mm) in win_matrix(results).items()}
        log["members"] = [{"id": m["id"], "founder": m["founder"], "widths": m["genome"]["widths"],
                           "parent": m["parent"]} for m in members]
        ranked = sorted(members, key=lambda m: -log["elo"][m["id"]])
        print(f"round {rnd} ({time.time() - t_round:.0f}s) elo: start {log['elo']['start']}  "
              + "  ".join(f"{m['id']}[{m['founder'][:3]},{m['genome']['widths'][0]}] {log['elo'][m['id']]}"
                          for m in ranked), flush=True)

        # Exploit and explore: the bottom quarter becomes mutated copies of the top quarter.
        if rnd % args.exploit_every == 0 and rnd < args.rounds:
            k = max(1, len(members) // 4)
            winners, losers = ranked[:k], []
            for m in reversed(ranked):
                if len(losers) == k or m in winners:
                    break
                line = [o for o in members if o["founder"] == m["founder"] and o not in losers]
                if args.no_line_protection or len(line) > 1:  # never remove a line's last member
                    losers.append(m)
            for loser in losers:
                parent = winners[int(rng.integers(len(winners)))]
                genome, changes = mutate(parent["genome"], rng, args.p_grow, args.max_width)
                model = load(parent["latest_zip"], genome)
                if genome["widths"] != parent["genome"]["widths"]:
                    model = grow_model(model, venv, hyper(genome), args.seed + next_id, genome["widths"], device)
                child = {"id": f"m{next_id:02d}", "founder": parent["founder"], "parent": parent["id"],
                         "born": rnd, "genome": genome, "mutations": changes, "model": model}
                next_id += 1
                records[child["id"]] = copy.deepcopy(records.get(parent["id"], {}))
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
