"""A policy grown to more layers must play exactly like the original."""
import sys
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pwjax import env, grow, maps, policy  # noqa: E402


def test_grown_policy_gives_the_same_outputs():
    params = {k: np.asarray(v) for k, v in policy.init(jax.random.PRNGKey(0), layers=2).items()}
    # make the layers non-trivial so a mistake in the new ones would show
    rng = np.random.default_rng(1)
    params = {k: v + 0.05 * rng.normal(size=v.shape).astype(v.dtype) for k, v in params.items()}
    bigger = grow.grow(params, layers=5, seed=3)
    assert policy.pwnet.n_layers(bigger) == 5 and policy.pwnet.n_layers(params) == 2
    bank = {k: jnp.asarray(v) for k, v in maps.make_bank(3, seed=5).items()}
    for i in range(3):
        s = env.from_bank(bank, i)
        a = policy.apply({k: jnp.asarray(v) for k, v in params.items()}, 4, s, 1)[:3]
        b = policy.apply({k: jnp.asarray(v) for k, v in bigger.items()}, 4, s, 1)[:3]
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-5)
