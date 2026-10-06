# Asteroids 5v5

Two fleets of five triangle ships fight it out in an Asteroids-style arena.

- **Watch it in the browser:** https://crunchy000.github.io/astorids/ (`index.html`, scripted AI on both sides)
- **Train agents in Python:** `astroids_gym`, a [Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym) and [PettingZoo](https://pettingzoo.farama.org/) port of the same game.

## Install

```bash
pip install -e .                  # core env: gymnasium + numpy
pip install -e ".[render]"        # + pygame for render_mode="human" / "rgb_array"
pip install -e ".[multiagent]"    # + pettingzoo for the multi-agent env
pip install -e ".[train]"         # + stable-baselines3 + torch for the example trainer
```

## Single-agent env (Gymnasium)

You fly one blue ship. The other nine are flown by the built-in scripted bot, the same AI the web page uses, unless you swap in other policies.

```python
import gymnasium as gym
import astroids_gym  # registers the env

env = gym.make("AsteroidsBattle-v0", render_mode="human")
obs, info = env.reset(seed=0)
for _ in range(1800):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
```

| | |
|---|---|
| **Action** | `MultiDiscrete([3, 2, 2])` = (turn left / none / right, thrust off/on, fire off/on) |
| **Observation** | `Box(-1, 1, (150,))`, relative to your ship: your own velocity, heading, HP, cooldown; the 5 nearest enemies and 4 nearest allies (position, relative velocity, relative heading, HP); the 8 nearest enemy bullets; the 5 nearest asteroids. Empty slots are zero, and each slot has a "present" flag. Details are in `astroids_gym/obs.py`. |
| **Reward** | +1 kill, −1 death, +0.2 hit dealt, −0.2 hit taken (set with `RewardConfig`) |
| **Episode** | Ships respawn, so an episode ends only at `max_steps` (default 1800 agent steps = 60 s of game time). `terminate_on_death=True` gives shorter episodes. |
| **Timing** | Physics runs at 60 Hz; the agent acts every `frame_skip=2` ticks (30 Hz). |

Useful kwargs: `teammates=` / `opponents=` take `"scripted"`, `"idle"`, `"random"` or a callable, so you can do self-play against frozen policies. A callable gets one batch of observations for all the ships it flies, shape `(n, 150)`, and returns `(n, 3)` actions, so an SB3 model plugs straight in: `opponents=lambda obs: model.predict(obs)[0]`. You can also pass `config=Config(asteroids=False, team_size=3, ...)`, `reward_config=RewardConfig(...)` and `obs_config=ObsConfig(...)`.

## Multi-agent env (PettingZoo Parallel API)

```python
from astroids_gym import parallel_env

env = parallel_env(learning_teams=("blue", "red"))   # all 10 ships are agents (self-play)
# env = parallel_env(learning_teams=("blue",))       # 5 blue agents vs the scripted red team
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

Agents are named `blue_0..4` / `red_0..4`. `env.state()` returns the global state for centralised critics (MAPPO, QMIX and similar). `RewardConfig(team_spirit=0.5)` mixes teammates' kills and deaths into each ship's reward.

## Training

```bash
python examples/evaluate.py --model random       # baseline: random pilot
python examples/evaluate.py                      # baseline: scripted bot as the pilot
python examples/train_ppo.py --timesteps 2000000 --n-envs 8
python examples/evaluate.py --model models/ppo_pilot.zip --render
# self-play: the next generation trains against the previous one flying the red team
python examples/train_ppo.py --opponent-model models/ppo_pilot.zip --out models/ppo_gen2
```

Baselines over 3 seeded 60 s episodes: random pilot ≈ −9.9 return (K/D 0.09); scripted bot ≈ +1.0 return (K/D 1.10). A trained agent should beat the scripted bot.

Speed: the bare env runs about 2,800 agent steps/s per process (pure Python, nine scripted bots). With PPO in the loop, a 4-core machine manages about 900 steps/s end to end, so set `--n-envs` to your core count. The trainer keeps PyTorch to one thread (`--torch-threads`) so it doesn't starve the env workers.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests cover Gymnasium's `check_env`, PettingZoo's `parallel_api_test`, seeded determinism, truncation and termination, and rendering.

Note: the Python simulation reimplements the browser game with the same constants and rules. It isn't a byte-for-byte replay of the JavaScript.
