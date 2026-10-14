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

### Past iteration 450

Every attempt to carry on from the iteration 450 weights (about 30 minutes each on an A100, 58M to
136M steps) drifted below it when played head-to-head, and none produced a clearly stronger model.
"Gate" is the new model's score against 450 during training (64 to 128 maps, both sides, argmax play);
the final columns are the end-of-run model on 256 maps.

| Run | Best gate vs 450 | Final vs 450 | Final vs rush bot |
|---|---:|---:|---:|
| Old settings: learning rate 1e-4, entropy 0.003 to 0.001 | 57% (iteration 80) | 18% | 80% |
| Rush bot and anchors, learning rate 2e-4, entropy 0.005 to 0.002 | 27% | 30% | 88% |
| Rush bot and anchors, learning rate 1e-4 decayed to 5e-6, entropy 0.003 to 0.001 | 55% (iteration 80) | 30% | 93% |
| The same with one epoch per batch and twice the batch | 48% | 36% | 90% |
| The same, grown to 6 layers | 46% | 18% (gate, iteration 140) | not run |
| The same, grown to 6 layers, learning rate 3e-4 decayed to 1e-5 | 37% | 27% (gate, iteration 140) | not run |

Iteration 450 itself scores 84% against the rush bot (the greedy agent sending every ship), 99%
against greedy and 99% against our stronger bot. The snapshot at iteration 80 of the decayed-rate
run played level with 450 (52%) and beat the rush bot 90% of the time; it is not shipped.

Options added for these runs, all off by default: `--opp-probs` takes a fifth value for the rush
bot, `--anchors a.npz,b.npz` keeps frozen models in the opponent pool, `--gate-every N` plays the
learner against the best model so far and keeps the winner in `best.npz`, `--lr-end` decays the
learning rate, and `python -m pwjax.grow in.npz out.npz --layers 6` adds layers that start as
no-ops, so the deeper model plays exactly like the original.

### Look-ahead search (not used)

`entry/search.py` puts a short search on top of the model: candidate moves are played on a copy of the
game with the official rules, both sides then follow a cheap scripted player for 40 to 50 ticks, and
the model's value estimate scores the result. It runs with `PW_SEARCH=1` (see `serve_port` style
launch in the notes) and is off by default. It did not make the agent stronger:

| Version | Result |
|---|---|
| Network's top 8 moves, greedy fast-forward | 6 of 6 against greedy; 2 of 24 against TeamTitansAgentV3 (plain model: 1 of 72) |
| Best target per free planet, greedy's move and no-op, greedy fast-forward | 37 of 80 (46%) against the plain model |
| The same, no further launches during the fast-forward | 34 of 80 (43%) against the plain model |

The model puts about 99% of its probability on one move, usually doing nothing, so its next-best
moves are not meaningful candidates, and the value estimates of candidate moves differ by about
0.02, which is too small to choose between them reliably. A search would need a value function
trained to rank moves, not only to predict the final result.

## Submitting

Their bot builds the `Dockerfile` at the root of a repository, so the entry goes in a repo of its
own (the `entry/` folder as its root), then a submission issue on
`SimonLucas/planet-wars-rts-submissions` names that repo at a specific commit.
