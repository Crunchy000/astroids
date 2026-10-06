"""Gymnasium (single-agent) and PettingZoo-style (multi-agent) environments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bot import lead, scripted_action
from .obs import ObsConfig, encode, observation_space
from .sim import Config, Events, Simulation, wrap_angle

# Policy for ships the learner doesn't control: "scripted" (built-in bot),
# "idle", "random", or a callable mapping a batch of observations, shape
# (n_ships, obs_dim), to a batch of MultiDiscrete actions, shape (n_ships, 3).
# An SB3 model works directly: ``lambda obs: model.predict(obs)[0]``.
Policy = Union[str, Callable[[np.ndarray], Any], None]

# Turn as a fraction of max turn rate (negative = left). The slow settings give
# the fine aim control that a full-rate turn per step (~8 degrees) lacks.
TURN_LEVELS = (-1.0, -0.25, 0.0, 0.25, 1.0)
ACTION_SPACE = spaces.MultiDiscrete([len(TURN_LEVELS), 2, 2])  # turn, thrust, fire


def decode_action(action) -> tuple[float, bool, bool]:
    a = np.asarray(action).reshape(-1)
    return (TURN_LEVELS[int(a[0])], bool(a[1]), bool(a[2]))


@dataclass
class RewardConfig:
    kill: float = 1.0
    death: float = -1.0
    hit_dealt: float = 0.2
    hit_taken: float = -0.2
    shot: float = 0.0  # set slightly negative to discourage spraying
    # Fraction of each teammate's (kill + death) reward also given to this ship.
    team_spirit: float = 0.0
    # Shaping: per agent step, up to ``aim`` for pointing where a bullet fired now
    # would hit the nearest enemy in range (Gaussian in the aim error, width
    # ``aim_width`` radians). Leave at 0 for the true game score.
    aim: float = 0.0
    aim_width: float = 0.15

    def aim_bonus(self, sim: Simulation, s) -> float:
        if not s.alive:
            return 0.0
        rng2 = (sim.cfg.bullet_speed * sim.cfg.bullet_life * 0.9) ** 2
        best, best_d2 = None, rng2
        for o in sim.ships:
            if o.alive and o.team != s.team:
                d2 = sim.wdx(s.x, o.x) ** 2 + sim.wdy(s.y, o.y) ** 2
                if d2 < best_d2:
                    best, best_d2 = o, d2
        if best is None:
            return 0.0
        err = wrap_angle(lead(sim, s, best)[0] - s.a)
        return self.aim * math.exp(-(err / self.aim_width) ** 2)

    def individual(self, ev: Events, i: int) -> float:
        return (self.kill * ev.kills[i] + self.death * ev.deaths[i]
                + self.hit_dealt * ev.hits_dealt[i] + self.hit_taken * ev.hits_taken[i]
                + self.shot * ev.shots[i])

    def rewards(self, sim: Simulation, ev: Events) -> list[float]:
        own = [self.individual(ev, i) for i in range(sim.n_ships)]
        if self.aim:
            own = [r + self.aim_bonus(sim, s) for r, s in zip(own, sim.ships)]
        if not self.team_spirit:
            return own
        team_kd = [0.0, 0.0]
        for s in sim.ships:
            team_kd[s.team] += self.kill * ev.kills[s.id] + self.death * ev.deaths[s.id]
        return [own[s.id] + self.team_spirit * (team_kd[s.team] - self.kill * ev.kills[s.id]
                                                - self.death * ev.deaths[s.id])
                for s in sim.ships]


class _Match:
    """Shared stepping logic: frame-skip, non-learning policies, rewards."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Config | None, obs_config: ObsConfig | None,
                 reward_config: RewardConfig | None, frame_skip: int, max_steps: int,
                 render_mode: str | None):
        self.cfg = config or Config()
        self.obs_cfg = obs_config or ObsConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.sim = Simulation(self.cfg)
        self._renderer = None
        self._steps = 0
        self.metadata = {**self.metadata, "render_fps": round(1 / (self.cfg.dt * frame_skip))}

    def _obs(self, ship_id: int) -> np.ndarray:
        return encode(self.sim, self.sim.ships[ship_id], self.obs_cfg)

    def _policy_actions(self, policies: dict[int, Policy], rng: np.random.Generator):
        """Decide actions for ships with a non-scripted fixed policy (once per agent step)."""
        out = {}
        batches: dict[int, tuple[Callable, list[int]]] = {}
        for sid, pol in policies.items():
            if pol in (None, "scripted"):
                continue
            if pol == "idle":
                out[sid] = (0.0, False, False)
            elif pol == "random":
                out[sid] = decode_action(rng.integers(0, ACTION_SPACE.nvec))
            else:
                batches.setdefault(id(pol), (pol, []))[1].append(sid)
        # One call per callable with all its ships' observations stacked.
        for pol, sids in batches.values():
            acts = np.asarray(pol(np.stack([self._obs(sid) for sid in sids])))
            for sid, a in zip(sids, acts.reshape(len(sids), -1)):
                out[sid] = decode_action(a)
        return out

    def _advance(self, learner_actions: dict[int, tuple], policies: dict[int, Policy],
                 rng: np.random.Generator) -> Events:
        acts = self._policy_actions(policies, rng)
        for sid, pol in policies.items():
            # Like every other pilot, the bot decides once per agent step.
            if pol in (None, "scripted"):
                acts[sid] = scripted_action(self.sim, self.sim.ships[sid], self.frame_skip)
        acts.update(learner_actions)
        total = Events(self.sim.n_ships)
        for _ in range(self.frame_skip):
            total.merge(self.sim.step(acts))
        self._steps += 1
        return total

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode")
            return None
        if self._renderer is None:
            from .render import Renderer
            self._renderer = Renderer(self.sim, self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class AsteroidsBattleEnv(_Match, gym.Env):
    """Single-agent view: the learner flies one blue ship in the 5v5 battle.

    The other nine ships are driven by ``teammates`` / ``opponents`` policies
    (default: the built-in scripted bot). Ships respawn, so an episode only ends
    by time limit (``max_steps``) unless ``terminate_on_death`` is set.

    Observation: ``Box(-1, 1, (150,))`` ego-centric features (see ``obs.py``).
    Action: ``MultiDiscrete([5, 2, 2])`` = (turn, thrust, fire); turn indexes
    ``TURN_LEVELS`` (full left, slow left, none, slow right, full right).
    ``step(None)`` lets the built-in bot fly your ship (a baseline to compare against).
    """

    def __init__(self, render_mode: str | None = None, *, config: Config | None = None,
                 obs_config: ObsConfig | None = None, reward_config: RewardConfig | None = None,
                 frame_skip: int = 2, max_steps: int = 1800, controlled_ship: int = 0,
                 teammates: Policy = "scripted", opponents: Policy = "scripted",
                 terminate_on_death: bool = False):
        super().__init__(config, obs_config, reward_config, frame_skip, max_steps, render_mode)
        self.controlled = controlled_ship
        me = self.sim.ships[controlled_ship]
        self.policies: dict[int, Policy] = {
            s.id: (teammates if s.team == me.team else opponents)
            for s in self.sim.ships if s.id != controlled_ship
        }
        self.terminate_on_death = terminate_on_death
        self.observation_space = observation_space(self.obs_cfg)
        self.action_space = ACTION_SPACE

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super(_Match, self).reset(seed=seed)  # seeds self.np_random
        self.sim.reset(int(self.np_random.integers(2**31)))
        self._steps = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(self.controlled), self._info()

    def step(self, action):
        if action is None:  # baseline: the scripted bot flies this ship
            act = scripted_action(self.sim, self.sim.ships[self.controlled], self.frame_skip)
        else:
            act = decode_action(action)
        ev = self._advance({self.controlled: act}, self.policies, self.np_random)
        reward = self.reward_cfg.rewards(self.sim, ev)[self.controlled]
        terminated = self.terminate_on_death and ev.deaths[self.controlled] > 0
        truncated = self._steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(self.controlled), float(reward), terminated, truncated, self._info()

    def set_reward_weights(self, **weights) -> None:
        """Change ``RewardConfig`` fields mid-training (e.g. anneal ``aim``) via ``env_method``."""
        for k, v in weights.items():
            setattr(self.reward_cfg, k, v)

    def _info(self) -> dict:
        me = self.sim.ships[self.controlled]
        return {"kills": me.kills, "deaths": me.deaths, "alive": me.alive,
                "score": tuple(self.sim.score), "time": self.sim.time}


try:  # Subclass PettingZoo's ParallelEnv when it's installed, for its tooling/wrappers.
    from pettingzoo import ParallelEnv as _ParallelBase
except ImportError:  # pragma: no cover
    _ParallelBase = object


class AsteroidsBattleParallelEnv(_Match, _ParallelBase):
    """Multi-agent (PettingZoo Parallel API) view for self-play / team training.

    Agents are named ``blue_0..blue_4`` and ``red_0..red_4``. By default every
    ship is a learning agent; pass ``learning_teams=("blue",)`` to train blue
    against the scripted red team. Ships respawn, so agents are never removed
    and the episode ends for everyone at ``max_steps`` (truncation).
    """

    TEAM_NAMES = ("blue", "red")

    def __init__(self, render_mode: str | None = None, *, config: Config | None = None,
                 obs_config: ObsConfig | None = None, reward_config: RewardConfig | None = None,
                 frame_skip: int = 2, max_steps: int = 1800,
                 learning_teams: tuple[str, ...] = ("blue", "red"), others: Policy = "scripted"):
        super().__init__(config, obs_config, reward_config, frame_skip, max_steps, render_mode)
        ts = self.cfg.team_size
        self._ids = {}
        for s in self.sim.ships:
            if self.TEAM_NAMES[s.team] in learning_teams:
                self._ids[f"{self.TEAM_NAMES[s.team]}_{s.id % ts}"] = s.id
        self.possible_agents = list(self._ids)
        self.agents: list[str] = []
        self.policies = {s.id: others for s in self.sim.ships if s.id not in self._ids.values()}
        self._obs_space = observation_space(self.obs_cfg)
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: ACTION_SPACE for a in self.possible_agents}
        self._rng = np.random.default_rng()

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        return ACTION_SPACE

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            for i, a in enumerate(self.possible_agents):
                self.action_spaces[a].seed(seed + i)
        self.sim.reset(int(self._rng.integers(2**31)))
        self._steps = 0
        self.agents = list(self.possible_agents)
        if self.render_mode == "human":
            self.render()
        return ({a: self._obs(i) for a, i in self._ids.items()},
                {a: self._info(i) for a, i in self._ids.items()})

    def step(self, actions: dict[str, Any]):
        learner = {self._ids[a]: decode_action(act) for a, act in actions.items()}
        ev = self._advance(learner, self.policies, self._rng)
        rewards = self.reward_cfg.rewards(self.sim, ev)
        done = self._steps >= self.max_steps
        obs = {a: self._obs(i) for a, i in self._ids.items()}
        rew = {a: float(rewards[i]) for a, i in self._ids.items()}
        term = {a: False for a in self.agents}
        trunc = {a: done for a in self.agents}
        infos = {a: self._info(i) for a, i in self._ids.items()}
        if done:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return obs, rew, term, trunc, infos

    def state(self) -> np.ndarray:
        """Global state: every ship's (alive, x, y, vx, vy, sin a, cos a, hp), normalised."""
        c = self.cfg
        return np.array([[s.alive, s.x / c.width, s.y / c.height, s.vx / c.max_speed,
                          s.vy / c.max_speed, np.sin(s.a), np.cos(s.a), s.hp / c.ship_hp]
                         for s in self.sim.ships], dtype=np.float32).ravel()

    def _info(self, ship_id: int) -> dict:
        s = self.sim.ships[ship_id]
        return {"kills": s.kills, "deaths": s.deaths, "alive": s.alive,
                "score": tuple(self.sim.score)}
