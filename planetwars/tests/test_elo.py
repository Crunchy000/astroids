"""tools/elo.py recovers known ratings from simulated results and keeps the fixed agent at its value."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import elo  # noqa: E402


def simulate(true, games, seed=0):
    rng = np.random.default_rng(seed)
    names = sorted(true)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = 1.0 / (1.0 + 10 ** ((true[b] - true[a]) / 400.0))
            w = rng.binomial(games, p)
            rows.append((a, b, float(w), float(games - w), 0.0, "sim"))
    return rows


def test_recovers_ratings_and_anchor():
    true = {"a": 1000.0, "b": 1150.0, "c": 1300.0, "d": 1500.0, "e": 700.0}
    names, r, lo, hi, _ = elo.ratings(simulate(true, 400), anchor="a", boot=30)
    got = dict(zip(names, r))
    assert got["a"] == 1000.0
    for n in true:
        assert abs(got[n] - true[n]) < 60, (n, got[n], true[n])
    i = names.index("d")
    assert lo[i] < got["d"] < hi[i]


def test_perfect_record_stays_finite():
    rows = [("x", "y", 8.0, 0.0, 0.0, "sim"), ("y", "z", 8.0, 0.0, 0.0, "sim"), ("x", "z", 8.0, 0.0, 0.0, "sim")]
    names, r, _, _, _ = elo.ratings(rows, anchor="z", boot=10)
    got = dict(zip(names, r))
    assert np.isfinite(r).all() and got["x"] > got["y"] > got["z"] == 1000.0


def test_draws_count_half():
    rows = [("a", "b", 5.0, 5.0, 0.0, "sim"), ("a", "c", 0.0, 0.0, 10.0, "sim")]
    names, r, _, _, _ = elo.ratings(rows, anchor="a", boot=5)
    got = dict(zip(names, r))
    assert abs(got["b"] - 1000) < 1 and abs(got["c"] - 1000) < 1
