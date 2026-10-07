# Asteroids 5v5

Two fleets of five triangle ships fight it out in an Asteroids-style arena.

- **Watch it in the browser:** https://crunchy000.github.io/astorids/ (`index.html`, scripted AI on both sides). It has synthesised sound effects (lasers, hits, explosions, asteroid thuds, panned by position): press any key to start audio, and M to mute.
- **Train agents in Python:** `astroids_gym`, a [Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym) and [PettingZoo](https://pettingzoo.farama.org/) port of the same game.

## Rules

Two rule sets, the same in the browser (press **C** to switch, or `?rules=classic`) and in Python (`Config()` / `Config.competitive()`):

- **Classic:** endless battle; ships respawn after 3 s; score = kills.
- **Competitive** (the browser default): 60-second battles with
  - **Control zone:** the team with more living ships inside the centre circle gains 1 point per second, and every 3 zone points earn it an extra respawn (reinforcements). Score = kills + zone points.
  - **Weapon heat:** each shot adds 0.25 heat; at 1.0 the gun locks until it cools to 0.5 (cooling 0.35/s), so bursts are fine but spraying isn't.
  - **Shorter guns:** 384 px range (bullets live 0.8 s instead of 1.3 s), so the spawns are out of each other's range and teams have to move to fight.
  - **Limited respawns:** 20 per team (plus zone reinforcements), 4 s timer. A team wiped out with none left loses outright; otherwise the higher score wins at the buzzer.
  - **Friendly fire:** bullets hit teammates (never the shooter).
  - **Fair spawns:** each team spawns in the middle of its own half (at least 0.3 of the map width from the enemy spawn either way, instead of ~0.2 across the wrap seam), facing the zone; a respawning ship takes the safest of 8 candidate points, and spawn protection lasts 2 s. With bots, ships dying within 3 s of respawning fell from 40% to 28%, and zone points per battle rose from ~6 to ~22.

The bots contest the zone when no enemy is near. For agents, `ObsConfig(extended=True)` adds 32 observation inputs for the rules (heat, zone position and control, score lead, respawns and ships left, time remaining, nearby teammates' bullets), and `RewardConfig` has `zone`, `friendly_hit` and `win` terms.

## Install

```bash
pip install -e .                  # core env: gymnasium + numpy
pip install -e ".[render]"        # + pygame for render_mode="human" / "rgb_array"
pip install -e ".[multiagent]"    # + pettingzoo for the multi-agent env
pip install -e ".[train]"         # + stable-baselines3 + torch for the example trainer
pip install -e ".[video]"         # + pygame + imageio/ffmpeg for MP4 recording
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
| **Action** | `MultiDiscrete([5, 2, 2])` = (turn: full left / slow left / none / slow right / full right, thrust off/on, fire off/on). The slow turns give the fine aim control the bot gets from steering continuously. |
| **Observation** | `Box(-1, 1, (160,))`, relative to your ship: your own velocity, heading, HP, cooldown; the 5 nearest enemies (position, relative velocity, relative heading, HP, and the *lead direction*, i.e. where to point so a shot fired now hits); the 4 nearest allies (the same without lead); the 8 nearest enemy bullets; the 5 nearest asteroids. Empty slots are zero, and each slot has a "present" flag. Details are in `astroids_gym/obs.py`. |
| **Reward** | +1 kill, −1 death, +0.2 hit dealt, −0.2 hit taken, plus optional aim shaping (below). All set with `RewardConfig`. |
| **Episode** | Ships respawn, so an episode ends only at `max_steps` (default 1800 agent steps = 60 s of game time). `terminate_on_death=True` gives shorter episodes. |
| **Timing** | Physics runs at 60 Hz; the agent acts every `frame_skip=2` ticks (30 Hz). |

Useful kwargs: `teammates=` / `opponents=` take `"scripted"`, `"idle"`, `"random"` or a callable, so you can do self-play against frozen policies. A callable gets one batch of observations for all the ships it flies, shape `(n, 160)`, and returns `(n, 3)` actions, so an SB3 model plugs straight in: `opponents=lambda obs: model.predict(obs)[0]`. You can also pass `config=Config(asteroids=False, team_size=3, ...)`, `reward_config=RewardConfig(...)` and `obs_config=ObsConfig(...)`. `env.step(None)` lets the built-in bot fly your ship, which gives a baseline in exactly the same seat.

### Aim shaping

`RewardConfig(aim=0.01)` adds up to `aim` per step for pointing where a shot fired now would hit the nearest enemy in range. The bonus is a Gaussian in the aim error (`aim_width`, default 0.15 rad). Kills are rare early in training, so this gives the agent a signal long before it lands one. Fade it out before the end of training so the final pilot optimises the real score: `train_ppo.py` does this for you, and `env.set_reward_weights(aim=...)` changes it mid-run.

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
python examples/evaluate.py                      # baseline: the built-in bot flies blue seat 0
python examples/evaluate.py --model random       # baseline: random pilot
python examples/train_ppo.py --n-envs 8 --stop-when-beats
python examples/evaluate.py --model models/ppo_pilot_best.zip --render
# self-play: the next generation trains against the previous one flying the red team
python examples/train_ppo.py --opponent-model models/ppo_pilot_best.zip --out models/ppo_gen2
```

`train_ppo.py` runs PPO with aim shaping (`--aim 0.01`, faded to 0 over `--aim-anneal` steps). Every `--eval-every` steps it plays the pilot and the built-in bot in the same seat on the same 20 seeds, scored with the plain game reward. It keeps the best model as `<out>_best.zip`, logs every evaluation to `<out>_eval.json`, and redraws the progress chart `<out>_eval.png` each time (`examples/plot_progress.py` draws it for any run). `--stop-when-beats` stops training once the pilot beats the bot on both mean return and K/D.

Speed: the bare env runs about 2,800 agent steps/s per process (pure Python, nine scripted bots). With PPO in the loop, a 4-core machine manages about 900 steps/s end to end, so set `--n-envs` to your core count. The trainer keeps PyTorch to one thread (`--torch-threads`) so it doesn't starve the env workers.

## Results: a PPO agent that beats the bot

`results/ppo_pilot.zip` was trained with the default `train_ppo.py` settings on 4 CPU cores, about 1.5 hours and 3.5M steps. The aim bonus started at 0.01 and faded to zero by 4M steps. To confirm, it was then evaluated on 50 battles that training never saw, against the built-in bot flying the same seat in the same battles:

| Blue seat 0 flown by | Avg reward | Kills | Deaths | Kills per death |
|---|---|---|---|---|
| **PPO agent** | **+2.03** | 6.1 | **5.2** | **1.16** |
| Built-in bot | −1.17 | 7.3 | 8.1 | 0.90 |

The agent wins mostly by staying alive: it dies about a third less often and still trades kills better than the bot. Training progress (20 battles per point):

![Training progress](results/progress.png)

Watch them on the website: https://crunchy000.github.io/astorids/videos.html (also linked from the game page). Videos (60 s with sound, same battle seed): [agent in seat 0](results/agent_seat0.mp4) · [bot in seat 0](results/bot_seat0.mp4).

On its own this agent doesn't help a whole team: when it flies all five blue ships, the team does about as well as an all-bot blue team (8 vs 10 wins out of 20 battles). Team training (below) fixes that.

## Team training

```bash
python examples/train_team.py --init results/ppo_pilot.zip --n-workers 4
```

One shared policy flies all five blue ships against the scripted red team (parameter sharing: each ship acts on its own observation, and all their experience trains the same network). Each worker process runs one battle, and every blue ship is one slot of the SB3 VecEnv. `--team-spirit 0.5` gives each ship half of its teammates' kill and death rewards, so ships care about the team result. Every `--eval-every` samples the team plays 40 fixed battles, scored by win rate and average kill margin against an all-bot blue team on the same battles. Progress goes to `<out>_eval.json` / `<out>_eval.png`, and the best model to `<out>_best.zip`.

### Team results

Starting from `results/ppo_pilot.zip`, `train_team.py` reached a 90% win rate (+9.8 average kill margin) over 40 evaluation battles after 2M ship-steps, and won every evaluation battle from 3M on (+18.8 at 4M), in about 35 minutes on 4 cores. An all-bot blue team wins 50% (−0.8). On 50 battles it had never seen, the 3M model won 98% (+15.4) where the all-bot team won 40% (−1.6). The 4M model is `results/ppo_team.zip`, with the progress chart `results/team_progress.png` and videos of the same battle: [team-trained blue wins 47–27](results/team_trained.mp4) · [all-bot blue loses 34–44](results/team_bots.mp4).

## League: teams of agents vs teams of agents

```bash
python examples/train_league.py --init results/ppo_team.zip --rounds 12 --bots-share 0.4
python examples/plot_league.py league/league.json          # Elo chart + who-beats-whom table
python examples/record_match.py --blue league/models/striker_r12.npz --red bots
```

`train_league.py` trains several shared-policy team models, each with its own reward personality (`striker`: kills first; `guardian`: survival first; `pack`: fully shared team rewards and the zone). Each round, every member trains against the other members' latest versions, past snapshots and the bots (`--bots-share`), on a random side each battle. A side-swapped round-robin tournament after every round gives Bradley-Terry Elo ratings with the built-in bots fixed at 1000. `--resume` continues a run from its last finished round. It runs at about 11,000 samples/s on 4 cores: battles step in multi-battle worker processes, opponents use NumPy copies of the policy, observations come from the numba encoder, and BLAS is pinned to one thread per process.

Results on the competitive rules (`results/league_*`): 12 rounds of 1M ship-steps per member, about 1.5 hours. All three started from the classic-rules team champion, which the bots beat 98% of the time under the new rules. Its spray-and-camp habit kept its guns overheated 46% of the time, so the league rewards include an overheat penalty and a small shot cost from round 4. After that, the **striker** broke away and finished level with the bots:

| Team (round 12) | Elo (bots = 1000) | vs starting champion |
|---|---|---|
| Striker | 988 | +410 |
| Pack | 617 | +39 |
| Starting champion | 578 | — |
| Guardian | 466 | −112 |

On 60 fresh battles the striker won 42% (± 6%) against the bots and 87% against the starting champion. The survival-first guardian ended below where it started. Charts: `results/league_elo.png`, `results/league_matrix.png`; models: `results/league_{striker,pack,guardian}.zip`; videos: [striker vs bots](results/league_striker_vs_bots.mp4) · [striker vs pack](results/league_striker_vs_pack.mp4).

## Videos

```bash
python examples/record_video.py --model models/ppo_pilot_best.zip --out videos/pilot.mp4         # pilot flies blue ship 0
python examples/record_video.py --model models/ppo_pilot_best.zip --team --out videos/team.mp4   # pilot flies all of blue
python examples/record_video.py --out videos/bot.mp4                                             # built-in bot, for comparison
```

`--seconds` sets the length (default 60) and `--seed` picks the battle. Videos are 1280×720 at 30 fps with a synthesised stereo soundtrack matching the browser game's sound effects (`--no-sound` to skip it; the synth is in `astroids_gym/audio.py`). Inside your own code, `render_mode="rgb_array"` returns frames from either env.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests cover Gymnasium's `check_env`, PettingZoo's `parallel_api_test`, seeded determinism, truncation and termination, aim shaping, batched policies, the bot baseline, and rendering.

Note: the Python simulation reimplements the browser game with the same constants and rules. It isn't a byte-for-byte replay of the JavaScript.
