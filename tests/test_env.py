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


def test_global_state_is_mirrored_for_red():
    from astroids_gym.obs import SHIP_STATE_DIM, global_state, global_state_dim

    sim = Simulation(Config.competitive(), seed=3)
    for _ in range(30):
        sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
    blue, red = global_state(sim, 0), global_state(sim, 1)
    assert blue.shape == red.shape == (global_state_dim(sim.cfg),)
    assert np.all(np.abs(blue) <= 1.0)
    n = sim.cfg.team_size * SHIP_STATE_DIM
    w, h = sim.cfg.width, sim.cfg.height
    r0 = sim.ships[5]
    assert red[1] == pytest.approx(1 - r0.x / w, abs=1e-5) and red[2] == pytest.approx(1 - r0.y / h, abs=1e-5)
    # Each team's own block comes first; alive flags, hp and heat do not depend on the viewpoint.
    for k in (0, 7, 8):
        assert np.allclose(blue[k:n:SHIP_STATE_DIM], red[n + k:2 * n:SHIP_STATE_DIM])


def test_central_critic_policy_matches_mlp_actor():
    pytest.importorskip("stable_baselines3")
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy

    from astroids_gym import ACTION_SPACE
    from astroids_gym.policies import CentralCriticPolicy

    n, g = 12, 5
    pol = CentralCriticPolicy(spaces.Box(-1, 1, (n + g,), np.float32), ACTION_SPACE, lambda _: 1e-3,
                              net_arch=[16, 16], n_actor_inputs=n)
    ref = ActorCriticPolicy(spaces.Box(-1, 1, (n,), np.float32), ACTION_SPACE, lambda _: 1e-3, net_arch=[16, 16])
    ref.mlp_extractor.policy_net.load_state_dict(pol.mlp_extractor.policy_net.state_dict())
    ref.action_net.load_state_dict(pol.action_net.state_dict())
    x = torch.rand(4, n + g) * 2 - 1
    with torch.no_grad():
        a = torch.cat([d.logits for d in pol.get_distribution(x).distribution], 1)
        b = torch.cat([d.logits for d in ref.get_distribution(x[:, :n]).distribution], 1)
        assert torch.allclose(a, b)
        y = x.clone()
        y[:, n:] = 0  # the critic reads the global inputs, the actor does not
        assert not torch.allclose(pol.predict_values(x), pol.predict_values(y))
        assert torch.allclose(pol.get_distribution(y).distribution[0].logits, pol.get_distribution(x).distribution[0].logits)
    assert pol.mlp_extractor.value_net[0].in_features == n + g


def test_entity_policy_numpy_export_matches_torch(tmp_path):
    pytest.importorskip("stable_baselines3")
    import sys
    import torch
    from gymnasium import spaces

    from astroids_gym import ACTION_SPACE
    from astroids_gym.obs import ObsConfig, encode_battle
    from astroids_gym.policies import EntityPolicy

    sys.path.insert(0, "examples")
    from league import NumpyPolicy, export_policy

    oc = ObsConfig(extended=True)
    pol = EntityPolicy(spaces.Box(-1, 1, (oc.size + 7,), np.float32), ACTION_SPACE, lambda _: 1e-3,
                       net_arch=[32, 32], n_actor_inputs=oc.size, entity_kwargs={"d": 16, "heads": 2})

    class M:  # just what export_policy reads
        policy = pol

    sim = Simulation(Config.competitive(), seed=1)
    for _ in range(60):
        sim.step({s.id: scripted_action(sim, s) for s in sim.ships})
    sim.ships[3].alive = False
    obs = np.concatenate([encode_battle(sim, range(10), oc), np.random.default_rng(0).random((10, 7))], 1)
    obs = obs.astype(np.float32)
    with torch.no_grad():
        ref = torch.cat([d.logits for d in pol.get_distribution(torch.as_tensor(obs)).distribution], 1).numpy()
    npol = NumpyPolicy(export_policy(M, str(tmp_path / "e.npz")))
    out = npol.logits(obs[:, :oc.size])
    # Each head's logits are defined up to a constant (torch normalises them): compare softmaxes.
    for a, b in zip(np.split(ref, [5, 7], 1), np.split(out, [5, 7], 1)):
        pa, pb = np.exp(a - a.max(1, keepdims=True)), np.exp(b - b.max(1, keepdims=True))
        assert np.allclose(pa / pa.sum(1, keepdims=True), pb / pb.sum(1, keepdims=True), atol=1e-5)


def test_policy_bank_matches_numpy_and_tournaments_agree(tmp_path):
    pytest.importorskip("stable_baselines3")
    import sys
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy

    from astroids_gym import ACTION_SPACE
    from astroids_gym.obs import ObsConfig
    from astroids_gym.policies import EntityPolicy, PolicyBank

    sys.path.insert(0, "examples")
    from league import NumpyPolicy, export_policy, play_matches

    oc = ObsConfig(extended=True)
    box = spaces.Box(-1, 1, (oc.size,), np.float32)

    class M:
        pass

    specs = []
    for name, pol in (("mlp", ActorCriticPolicy(box, ACTION_SPACE, lambda _: 1e-3, net_arch=[32, 32])),
                      ("ent", EntityPolicy(box, ACTION_SPACE, lambda _: 1e-3, net_arch=[32, 32], n_actor_inputs=oc.size,
                                           entity_kwargs={"d": 16, "heads": 2}))):
        M.policy = pol
        specs.append(export_policy(M, str(tmp_path / f"{name}.npz")))
    bank = PolicyBank("cpu")
    rng = np.random.default_rng(0)
    x = (rng.random((64, oc.size), dtype=np.float32) * 2 - 1)
    x[:, 8:58:10] = rng.random((64, 5)) > 0.3  # enemy "present" flags
    for spec in specs:
        assert np.allclose(NumpyPolicy(spec).logits(x), bank.logits(spec, x), atol=1e-4)
        assert (NumpyPolicy(spec)(x) == bank.act(spec, x, sample=False)).all()
    jobs = [(specs[0], "bots", 11, "competitive"), ("bots:rusher", specs[0], 12, "competitive")]
    assert play_matches(jobs, workers=2) == play_matches(jobs, workers=2, bank=bank)


def test_formation_bots_hold_their_shape():
    """Before contact, a line is wide and shallow and a wedge is swept back."""
    from astroids_gym.bot import scripted_action

    def shape(style):
        sim = Simulation(Config.competitive(asteroids=False, width=3000, height=1800), seed=1)
        for _ in range(60 * 3):  # red sits still, far away
            sim.step({s.id: scripted_action(sim, s, 1, style) if s.team == 0 else (0.0, False, False)
                      for s in sim.ships})
        team = [s for s in sim.ships if s.team == 0 and s.alive]
        ref = team[0]
        pts = np.array([(sim.wdx(ref.x, s.x), sim.wdy(ref.y, s.y)) for s in team])
        pts -= pts.mean(axis=0)
        to_red = np.mean([(sim.wdx(ref.x, s.x), sim.wdy(ref.y, s.y)) for s in sim.ships if s.team == 1], axis=0)
        f = to_red / np.linalg.norm(to_red)
        fwd, lat = pts @ f, pts @ np.array([-f[1], f[0]])
        return fwd, lat

    fwd, lat = shape("line")
    assert np.ptp(lat) > 250 and np.ptp(fwd) < 60
    fwd, lat = shape("wedge")
    lead = np.argmax(fwd)  # the leader flies in the middle, the wings behind it
    assert np.ptp(fwd) > 80 and abs(lat[lead]) < 50


def test_tactics_bots_fight():
    from astroids_gym.bot import FORMATIONS, MANEUVERS, PAIRS, scripted_action

    for k, style in enumerate(FORMATIONS + PAIRS + MANEUVERS):
        sim = Simulation(Config.competitive(), seed=30 + k)
        shots = [0, 0]
        while not sim.over:
            ev = sim.step({s.id: scripted_action(sim, s, 1, style if s.team == 0 else None) for s in sim.ships})
            for s in sim.ships:
                shots[s.team] += ev.shots[s.id]
        assert shots[0] > 50 and sim.team_score(0) > 0, style


def test_crashes_are_counted_and_rewarded():
    from astroids_gym.env import RewardConfig

    sim = Simulation(Config(), seed=3)
    s = sim.ships[0]
    s.invuln = 0.0
    a = sim.asteroids[0]
    s.x, s.y, s.vx, s.vy = a.x, a.y, 0.0, 0.0  # inside an asteroid
    ev = sim.step({})
    assert ev.deaths[0] == 1 and ev.crashes[0] == 1
    assert RewardConfig(death=-1.0, crash=-1.5).individual(ev, 0) == -2.5


def test_dogfight_ships_fly_like_aircraft():
    import math
    cfg = Config.dogfight()
    assert cfg.asteroid_count < Config.competitive().asteroid_count
    # The turn rate peaks at corner speed and is slower when stopped or flat out.
    assert cfg.turn_factor(cfg.corner_speed) == 1.0
    assert cfg.turn_factor(0.0) < 1.0 and cfg.turn_factor(cfg.max_speed) < 1.0
    assert Config.competitive().turn_factor(0.0) == 1.0
    sim = Simulation(cfg, seed=1)
    s = sim.ships[0]
    s.vx, s.vy, s.a = 150.0, 0.0, 0.0
    for _ in range(30):  # half a second of hard turn: the velocity follows the nose and speed bleeds
        sim.step({s.id: (1.0, False, False)})
    heading_of_travel = math.atan2(s.vy, s.vx)
    assert abs(math.remainder(heading_of_travel - s.a, math.tau)) < 0.2
    turned = Simulation(cfg, seed=1)
    straight = turned.ships[0]
    straight.vx, straight.vy, straight.a = 150.0, 0.0, 0.0
    for _ in range(30):
        turned.step({straight.id: (0.0, False, False)})
    assert math.hypot(s.vx, s.vy) < math.hypot(straight.vx, straight.vy) - 5


def test_bots_fight_under_dogfight_rules():
    sim = Simulation(Config.dogfight(), seed=5)
    for _ in range(900):
        sim.step({x.id: scripted_action(sim, x, 1, "wedge" if x.team == 0 else None) for x in sim.ships})
    assert sim.team_score(0) + sim.team_score(1) > 0
