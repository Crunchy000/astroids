# Planet Wars RTS entry, trained with JAX

An entry for Simon Lucas's [Planet Wars RTS](https://simonlucas.github.io/planet-wars-rts/) league,
trained by self-play on a GPU. The game itself is untouched: we port its rules to JAX so thousands of
games run in parallel, then ship a numpy copy of the network in their Python entry format.

```
planetwars/
  pwjax/           training code (JAX)
    maps.py        map generator, same as their Kotlin GameStateFactory + GameParamGenerator
    env.py         the forward model (ForwardModel.kt / forward_model.py), rule for rule
    baselines.py   their Greedy Heuristic and Careful Random sample agents
    policy.py      sampling and log-probabilities for PPO
    ppo.py         self-play PPO against itself, older snapshots and the sample agents
    evaluate.py    win rates against the sample agents or another model
  entry/           the competition entry (copy to the root of a repo to submit)
    Dockerfile     their Python template: python:3.11-slim, port 8080
    main.py        GameServerAgent(host="0.0.0.0", port=8080, agent=NeuralAgent())
    agent.py       the neural agent (numpy only)
    pwnet.py       features + network, shared verbatim with training (numpy or jax.numpy)
    model.npz      trained weights (self-play iteration 450)
    selfcheck.py   play the official sample agents in the official Python engine
    core/ agents/ client_server/   vendored from planet-wars-rts (MIT, see LICENSE-planet-wars-rts)
  tests/           parity tests against the official engine
  tools/           record_game.py (render a game as an MP4), remote_match.py
  media/           a recorded game against their greedy agent
```

## How the agent works

Each tick the network looks at every planet (owner, ships, growth, position, fleets in flight and
fleets heading to it) through a small transformer, then scores every legal (source, destination)
launch plus "do nothing", and picks a fleet size: a quarter, half, three quarters or all of the
source's ships, or just enough to take the target (with or without a margin). Player 2 sees the
map mirrored, so the network always plays from the left.

## Results

Training ran on Colab L4 GPUs and stopped when the iteration-550 snapshot won only 37% of its games
against iteration 450, so iteration 450 is the model in `entry/model.npz`. Win rates come from
`pwjax.evaluate` on fresh league maps, each played from both sides (128 to 256 games per row);
"stronger bot" is our own hand-written agent that sends just enough ships.

| Iteration | vs their greedy agent | vs our stronger bot | vs the iteration before |
|---:|---:|---:|---:|
| 0 (copies greedy) | 50% | 39% | |
| 125 | 84% | 89% | 83% (vs 0) |
| 150 | 92% | 93% | 81% (vs 125) |
| 225 | 98% | 95% | 80% (vs 150) |
| 275 | 99% | 96% | 76% (vs 225) |
| 325 | 100% | 96% | 73% (vs 275) |
| **450 (shipped)** | 99% | 98% | 86% (vs 325) |
| 550 | 98% | 98% | 37% (vs 450) |

Iteration 450 also passes their own submission check (`runEvaluation` in their Kotlin repo) 12 of 12
games against their sample agents, with no timeouts and a 9 ms average move.

![Our model (blue) against their greedy agent (orange)](media/game_vs_greedy.png)

[Watch the whole game](media/game_vs_greedy.mp4): our model wins on a 30-planet league map at tick 722.
Record another with `python tools/record_game.py --seed 7 --out game.mp4`.

## Checks

```bash
cd planetwars
python -m pytest -q tests          # JAX rules == official Python engine; numpy entry == JAX policy
python -m pwjax.evaluate entry/model.npz --games 256 --vs greedy,random
cd entry && python selfcheck.py    # official engine, submission-check settings
```

The parity tests replay fuzzed games (legal and illegal actions) in both engines and compare every
planet and fleet each tick; their greedy agent's moves are compared too.

## Training

```bash
pip install "jax[cuda12]" optax
python -m pwjax.ppo --out runs/pw1 --envs 2048 --iters 3000
python -m pwjax.evaluate runs/pw1/latest.npz --games 512
cp runs/pw1/latest.npz entry/model.npz
```

Maps follow the league's `GameParamGenerator.randomParams()` (10 to 30 planets, transporter speed
2 to 5, growth 0.05 to 0.2), with some default-rule maps mixed in because the submission check
plays 20-planet default maps for 1000 ticks.

## Submitting

Their bot builds the `Dockerfile` at the root of a repository, so the entry goes in a repo of its
own (the `entry/` folder as its root), then a submission issue on
`SimonLucas/planet-wars-rts-submissions` names that repo at a specific commit.
