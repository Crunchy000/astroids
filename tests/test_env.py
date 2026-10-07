import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import astroids_gym  # noqa: F401  (registers AsteroidsBattle-v0)
from astroids_gym import AsteroidsBattleEnv, RewardConfig, AsteroidsBattleParallelEnv, Config, Simulation, scripted_action


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
        *_, term, trunc, info = env.step(np.array([2, 0, 0]))  # sit still
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


def test_callable_policies_get_one_batched_call_per_step():
    calls = []

    def policy(obs):
        calls.append(obs.shape)
        return np.tile([2, 1, 1], (len(obs), 1))  # no turn, thrust, fire

    env = AsteroidsBattleEnv(opponents=policy, teammates="idle")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert calls == [(5, env.observation_space.shape[0])]
    assert any(s.thrusting for s in env.sim.ships[5:])


def test_observation_layout():
    env = AsteroidsBattleEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (160,)
    assert obs[0] == 1.0  # alive
    assert obs[8] == 1.0  # first enemy slot present


def test_aim_bonus_rewards_pointing_at_the_lead_point():
    from astroids_gym.bot import lead

    env = AsteroidsBattleEnv(reward_config=RewardConfig(aim=1.0), opponents="idle", teammates="idle")
    env.reset(seed=0)
    sim = env.sim
    me, foe = sim.ships[0], sim.ships[5]
    for s in sim.ships[1:]:
        if s is not foe:
            s.alive, s.respawn = False, 1e9
    me.x, me.y, foe.x, foe.y = 300.0, 300.0, 600.0, 300.0
    me.a = lead(sim, me, foe)[0]
    on_target = env.reward_cfg.aim_bonus(sim, me)
    me.a += 1.0
    assert on_target > 0.99 and env.reward_cfg.aim_bonus(sim, me) < 0.01
    env.set_reward_weights(aim=0.0)
    assert env.reward_cfg.rewards(sim, astroids_gym.sim.Events(sim.n_ships))[0] == 0.0


def test_step_none_lets_the_bot_fly():
    env = AsteroidsBattleEnv(max_steps=1800)
    env.reset(seed=0)
    done = False
    while not done:
        *_, term, trunc, info = env.step(None)
        done = term or trunc
    assert info["kills"] > 0


def test_sound_events_and_soundtrack():
    from astroids_gym.audio import RATE, render_soundtrack
    from astroids_gym.bot import scripted_action

    sim = Simulation(Config(), seed=0)
    sim.sfx = []
    for _ in range(60 * 20):
        sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
    kinds = {e[1] for e in sim.sfx}
    assert {"shot", "boom"} <= kinds
    track = render_soundtrack(sim.sfx, sim.time)
    assert track.shape == (int(sim.time * RATE), 2)
    assert 0 < np.abs(track).max() <= 0.8 + 1e-6


def test_fast_encoder_matches_reference():
    pytest.importorskip("numba")
    from astroids_gym.bot import scripted_action
    from astroids_gym.fastobs import encode_fast
    from astroids_gym.obs import ObsConfig, encode

    oc = ObsConfig()
    sim = Simulation(Config(), seed=5)
    for step in range(600):
        sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
        if step % 20 == 0:
            ref = np.stack([encode(sim, s, oc) for s in sim.ships])
            np.testing.assert_array_equal(encode_fast(sim, range(10), oc), ref)
            np.testing.assert_array_equal(encode_fast(sim, [7, 2], oc), ref[[7, 2]])


def test_competitive_rules():
    from astroids_gym.bot import scripted_action

    sim = Simulation(Config.competitive(), seed=2)
    shots = team_hits = 0
    saw_overheat = saw_zone = False
    while not sim.over:
        ev = sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
        shots += sum(ev.shots)
        team_hits += sum(ev.team_hits)
        saw_overheat |= any(s.overheated for s in sim.ships)
        saw_zone |= sum(ev.zone_points) > 0
    assert saw_overheat and saw_zone and team_hits > 0
    assert sim.eliminated(0) or sim.eliminated(1) or sim.time >= 60 - 1e-6
    assert sim.winner() in (-1, 0, 1)


def test_competitive_parallel_env_ends_and_pays_win_reward():
    from astroids_gym import ObsConfig

    env = AsteroidsBattleParallelEnv(config=Config.competitive(team_lives=1),
                                     obs_config=ObsConfig(extended=True),
                                     reward_config=RewardConfig(win=5.0, zone=0.1, friendly_hit=-0.3),
                                     learning_teams=("blue",))
    obs, _ = env.reset(seed=0)
    assert next(iter(obs.values())).shape == (192,)
    steps = 0
    while env.agents:
        obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
        steps += 1
    assert all(term.values()) and steps < 1800  # random blue, 1 respawn: wiped out early
    assert env.sim.winner() == 1 and all(r <= -5.0 + 1.0 for r in rew.values())


def test_competitive_spawns_are_in_own_half_and_apart():
    import math

    for seed in range(10):
        sim = Simulation(Config.competitive(), seed=seed)
        w = sim.cfg.width
        assert all(0.15 * w <= s.x <= 0.35 * w for s in sim.ships[:5])
        assert all(0.65 * w <= s.x <= 0.85 * w for s in sim.ships[5:])
        gap = min(math.hypot(sim.wdx(a.x, b.x), sim.wdy(a.y, b.y)) for a in sim.ships[:5] for b in sim.ships[5:])
        assert gap >= 0.3 * w - 1e-6


def test_zone_reinforcements():
    sim = Simulation(Config.competitive(), seed=0)
    for s in sim.ships[1:]:
        s.alive, s.respawn = False, 1e9  # only blue ship 0 is left, parked in the zone
    me = sim.ships[0]
    me.x, me.y, me.vx, me.vy = sim.cfg.width / 2, sim.cfg.height / 2, 0.0, 0.0
    lives = sim.lives[0]
    for _ in range(int(3.5 / sim.cfg.dt)):  # 3.5 s in the zone = 3.5 points
        me.vx = me.vy = 0.0
        sim.step({})
    assert 3.4 < sim.points[0] < 3.6 and sim.points[1] == 0
    assert sim.lives[0] == lives + 1 and sim.reinforcements[0] == 1
