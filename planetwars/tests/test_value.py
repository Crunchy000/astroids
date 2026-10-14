"""The value-network tooling: varied play stays legal, the candidate moves are launchable, one-tick look-ahead is the real step."""
import sys
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pwjax import baselines, env, maps, policy, vsearch  # noqa: E402


def _midgame(seed=0, ticks=30):
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(1, seed=seed).items()}
    s = env.from_bank(bank, 0)
    params = policy.init(jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(ticks):
        key, k1, k2 = jax.random.split(key, 3)
        s = env.step(s, policy.sample_t(params, 4, s, 1, k1, 1.0, 0.5), policy.sample_t(params, 4, s, 2, k2, 1.0, 0.5))
    return params, s


def _legal(s, me, action):
    src, dst, ships = (np.asarray(x) for x in action)
    if src < 0:
        return True
    return bool(s.valid[src] and s.valid[dst] and src != dst and s.owner[src] == me and not s.tr_on[src]
                and 0 < ships <= float(s.ships[src]) + 1e-4)


def test_random_moves_are_legal():
    params, s = _midgame()
    for me in (1, 2):
        for i in range(40):
            a = policy.sample_t(params, 4, s, me, jax.random.PRNGKey(i), 1.0, 1.0)
            assert _legal(s, me, a)


def test_candidates_are_launchable_and_include_the_policy_choice():
    params, s = _midgame(seed=2)
    for me in (1, 2):
        src, dst, ships = vsearch.candidates(params, 4, s, me, top=4)
        assert src.shape == (7,)
        for k in range(7):
            assert _legal(s, me, (src[k], dst[k], ships[k]))
        assert int(src[1]) == -1  # candidate 1 is doing nothing
        best = policy.sample(params, 4, s, me, jax.random.PRNGKey(0), greedy=True)[0]
        assert (int(src[0]), int(dst[0])) == (int(best[0]), int(best[1]))


def test_one_tick_lookahead_is_the_real_step():
    params, s = _midgame(seed=3)
    me = 1
    action = vsearch.candidates(params, 4, s, me, top=3)
    a = tuple(x[2] for x in action)
    after, done = vsearch.after(s, me, a, horizon=1)
    expected = env.step(s, a, baselines.greedy(s, 2))
    for x, y in zip(jax.tree.leaves(after), jax.tree.leaves(expected)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y))
    assert bool(done) == bool(env.is_terminal(expected))
