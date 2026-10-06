import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import astroids_gym  # noqa: F401  (registers AsteroidsBattle-v0)
from astroids_gym import AsteroidsBattleEnv, AsteroidsBattleParallelEnv, Config, Simulation, scripted_action


def test_gymnasium_env_checker():
    check_env(gym.make("AsteroidsBattle-v0").unwrapped, skip_render_check=True)


def test_registered_env_runs_and_truncates():
    env = gym.make("AsteroidsBattle-v0", max_steps=50)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    for t in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert not term
    assert trunc


def test_seeding_is_deterministic():
    def rollout(seed):
        env = AsteroidsBattleEnv(max_steps=200)
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        traj = [obs]
        for _ in range(200):
            obs, r, *_ = env.step(env.action_space.sample())
            traj.append(obs)
        return np.stack(traj)

    np.testing.assert_array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_scripted_bots_fight():
    sim = Simulation(Config(), seed=0)
    for _ in range(60 * 60):  # one simulated minute
        sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
    assert sum(sim.score) > 5


def test_terminate_on_death():
    env = AsteroidsBattleEnv(terminate_on_death=True, opponents="scripted", teammates="idle")
    env.reset(seed=0)
    for _ in range(5000):
        *_, term, trunc, info = env.step(np.array([1, 0, 0]))  # sit still
        if term or trunc:
            break
    assert term and info["deaths"] == 1


def test_parallel_env_api():
    pz = pytest.importorskip("pettingzoo.test")
    pz.parallel_api_test(AsteroidsBattleParallelEnv(max_steps=100), num_cycles=100)


def test_parallel_env_vs_scripted():
    env = AsteroidsBattleParallelEnv(learning_teams=("blue",), max_steps=30)
    obs, _ = env.reset(seed=3)
    assert sorted(obs) == [f"blue_{i}" for i in range(5)]
    while env.agents:
        obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert all(trunc.values())
    assert env.state().shape == (80,)


def test_rgb_render():
    pytest.importorskip("pygame")
    env = AsteroidsBattleEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8 and frame.any()
    env.close()
