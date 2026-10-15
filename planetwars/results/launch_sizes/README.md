Raw lines from the A100 runs behind the "Dense targets, and launch sizes" section of the main README (one JSON object per
line; `score` counts a draw as half; `games` is plays, two per map).

- `gauntlet_hand`, `gauntlet_value`: size gauntlets for the hand-scored planner and the round 3 network (12 launches).
- `gauntlet_dA`, `gauntlet_dB`, `gauntlet_dA32`: gauntlets for round 5 (16 launches), round 5b (16) and round 5 (32).
- `gen_dA`, `gen_dB`: game generation for round 5 (default sizes) and round 5b (`f:0.9+spare+cap:1.2:1` in the games).
- `match_dA`, `match_dB`: the first head-to-head matches after training (16 launches).
- `cross_1`..`cross_7`: the follow-up combinations (net against net, sizes on or off, `pair`, 32 launches). Note
  `cross_3` is `extras=pair` (one size per source and target) and `cross_7` is round 5 with `spare` against round 3, 32 launches.
- `gen_d6`, `train_d6`, `match_d6`: round 6 (games played by round 5 with `spare`; held-out error every 100 steps; round 6 against round 5 at 32 launches).
