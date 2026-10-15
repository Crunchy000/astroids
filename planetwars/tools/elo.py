"""Elo ratings from head-to-head results (Bradley-Terry fit), with one agent fixed at 1000.

    python tools/elo.py results/elo_games.csv results/elo_earlier.csv --anchor evo --md results/elo.md

The CSV has the columns a, b, a_wins, b_wins, draws, source: one row per batch of games between a and b
(draws count as half a win for each side). P(a beats b) = 1 / (1 + 10 ** ((R_b - R_a) / 400)).

An agent that wins every game has no finite rating, so every pairing gets half a win and half a loss added before
the fit (8 wins out of 8 count as 8.5 to 0.5, about +480 Elo), and the ratings carry a weak prior toward the mean
(a normal with standard deviation `--prior` Elo). The 95% interval is a normal approximation around the fitted ratings (the inverse Hessian of the same penalised
likelihood, with the anchor held fixed); it is the spread of the ratings given the games, not a statement that a
saturated result is exact. A rating printed with a "+" belongs to an agent that played fewer than `--min-close`
games against opponents within 300 Elo of it: the games only say "at least this much".
"""
import argparse
import csv
import math
import pathlib

import numpy as np
from scipy.optimize import minimize

SCALE = 400.0 / math.log(10.0)  # Elo points per natural-log odds


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["a"], r["b"], float(r["a_wins"]), float(r["b_wins"]), float(r.get("draws") or 0), r.get("source", "")))
    return rows


def pooled(rows):
    """Games per ordered pair (a, b) summed over batches: points for a, points for b."""
    d = {}
    for a, b, wa, wb, dr, _ in rows:
        k = (a, b) if a < b else (b, a)
        pa, pb = (wa + dr / 2, wb + dr / 2) if a < b else (wb + dr / 2, wa + dr / 2)
        p = d.setdefault(k, [0.0, 0.0])
        p[0] += pa
        p[1] += pb
    return d


def fit(names, pairs, prior, pseudo=0.5):
    idx = {n: i for i, n in enumerate(names)}
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    pa = np.array([pairs[k][0] for k in pairs]) + pseudo
    pb = np.array([pairs[k][1] for k in pairs]) + pseudo
    tau = prior / SCALE

    def nll(th):
        d = th[ia] - th[ib]
        # -(pa * log sigmoid(d) + pb * log sigmoid(-d))
        v = pa * np.logaddexp(0.0, -d) + pb * np.logaddexp(0.0, d)
        return v.sum() + 0.5 * np.sum((th - th.mean()) ** 2) / tau**2

    def grad(th):
        d = th[ia] - th[ib]
        s = 1.0 / (1.0 + np.exp(-d))
        g = -(pa * (1 - s) - pb * s)
        out = np.zeros_like(th)
        np.add.at(out, ia, g)
        np.add.at(out, ib, -g)
        return out + (th - th.mean()) / tau**2

    res = minimize(nll, np.zeros(len(names)), jac=grad, method="L-BFGS-B")
    d = res.x[ia] - res.x[ib]
    sg = 1.0 / (1.0 + np.exp(-d))
    w = sg * (1.0 - sg) * (pa + pb)
    hess = np.zeros((len(names), len(names)))
    np.add.at(hess, (ia, ia), w)
    np.add.at(hess, (ib, ib), w)
    np.add.at(hess, (ia, ib), -w)
    np.add.at(hess, (ib, ia), -w)
    hess += (np.eye(len(names)) - 1.0 / len(names)) / tau**2
    return res.x * SCALE, hess / SCALE**2  # ratings in Elo, the Hessian with respect to them


def ratings(rows, anchor, anchor_value=1000.0, prior=1000.0, boot=4000, seed=0, return_samples=False):
    """Fitted ratings with the anchor fixed, their 95% interval, and `boot` draws from the normal approximation
    around the fit (for predictions that carry the uncertainty along)."""
    names = sorted({r[0] for r in rows} | {r[1] for r in rows})
    pairs = pooled(rows)
    fitted, hess = fit(names, pairs, prior)
    k = names.index(anchor)
    base = fitted - fitted[k] + anchor_value
    keep = [i for i in range(len(names)) if i != k]
    cov = np.zeros_like(hess)  # of the ratings relative to the anchor's, which is held fixed
    cov[np.ix_(keep, keep)] = np.linalg.inv(hess[np.ix_(keep, keep)])
    sd = np.sqrt(np.diag(cov))
    lo, hi = base - 1.96 * sd, base + 1.96 * sd
    if return_samples:
        draws = np.random.default_rng(seed).multivariate_normal(np.zeros(len(names)), cov, size=boot, method="eigh")
        return names, base, lo, hi, pairs, base + draws
    return names, base, lo, hi, pairs


def to_markdown(names, r, lo, hi, pairs, anchor, labels, notes, min_close=24):
    games = {n: 0.0 for n in names}
    close = {n: 0.0 for n in names}
    rating = dict(zip(names, r))
    for (a, b), (x, y) in pairs.items():
        games[a] += x + y
        games[b] += x + y
        if abs(rating[a] - rating[b]) <= 300:
            close[a] += x + y
            close[b] += x + y
    order = np.argsort(-r)
    out = ["| Rank | Agent | Elo | 95% interval | Games | Games against close opponents |", "|---:|---|---:|---|---:|---:|"]
    for rank, i in enumerate(order, 1):
        n = names[i]
        shown = f"{r[i]:.0f}{'+' if close[n] < min_close and n != anchor else ''}"
        tag = " (fixed)" if n == anchor else ""
        out.append(f"| {rank} | {labels.get(n, n)}{tag} | {shown} | {lo[i]:.0f} to {hi[i]:.0f} | {games[n]:.0f} | {close[n]:.0f} |")
    out.append("")
    out.append(notes)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--anchor", default="evo")
    ap.add_argument("--value", type=float, default=1000.0)
    ap.add_argument("--prior", type=float, default=1000.0, help="standard deviation of the prior on ratings, in Elo")
    ap.add_argument("--boot", type=int, default=4000, help="draws used for the predictions")
    ap.add_argument("--md", default=None)
    ap.add_argument("--labels", default=None, help="CSV of key,label for the printed table")
    ap.add_argument("--predict", nargs=2, metavar=("A", "B"), action="append", default=[],
                    help="print the chance that A beats B (median and 90%% interval over the draws); repeatable")
    args = ap.parse_args()
    rows = [r for path in args.csv for r in load(path)]
    names, r, lo, hi, pairs, samples = ratings(rows, args.anchor, args.value, args.prior, args.boot, return_samples=True)
    labels = {}
    if args.labels:
        with open(args.labels, newline="") as f:
            labels = {k: v for k, v in csv.reader(f)}
    table = to_markdown(names, r, lo, hi, pairs, args.anchor, labels, "")
    print(table)
    for a, b in args.predict:
        d = (samples[:, names.index(b)] - samples[:, names.index(a)]) / 400.0
        p = 1.0 / (1.0 + 10.0**d)
        point = 1.0 / (1.0 + 10.0 ** ((r[names.index(b)] - r[names.index(a)]) / 400.0))
        print(f"P({a} beats {b}) = {point:.2f} (draws: median {np.median(p):.2f}, 90% interval {np.percentile(p, 5):.2f} to {np.percentile(p, 95):.2f})")
    if args.md:
        pathlib.Path(args.md).write_text(table + "\n")


if __name__ == "__main__":
    main()
