# Astorids competition

The 5v5 battle run as an AI competition in the format of [Planet Wars RTS](https://simonlucas.github.io/planet-wars-rts/): each entry is a Docker container that flies a whole team over a WebSocket, matches are played in real time with match settings drawn at random, and a TrueSkill league ranks the entries. Current standings: [leaderboard.md](leaderboard.md).

| | Planet Wars RTS | Astorids |
|---|---|---|
| **Game** | Capture planets by sending ships between them | Five triangle ships a side, the competitive rules: kills plus control-zone points, limited respawns, weapon heat, friendly fire |
| **Agent** | `prepareToPlayAs`, `getAction`, `processGameOver` | The same methods. One agent flies all five ships |
| **What it sees** | The full game state every tick | The full game state every decision step, with a forward model in Python |
| **Time** | 50 ms per decision at 20 Hz | 33 ms per decision at 30 Hz |
| **Late answers** | The agent acts on stale information | The same: its team coasts until the answer comes, then the answer is used |
| **Settings drawn per game** | Number of planets, neutral share, growth rate, transporter speed | Asteroids, respawns, zone size, gun range, flight physics |
| **Entry** | Docker image, WebSocket server on port 8080, JSON | The same, with the same wire format |
| **Submission** | A GitHub issue with a YAML block pointing at a commit | The same, in this repository |
| **Ranking** | TrueSkill league, ranked by μ − 3σ | The same |

## Matches

A match is one 60-second battle: 1,800 decision steps, 30 per second of game time. Every step the runner sends both agents the state and waits up to 33 ms for their actions. A team whose agent hasn't answered coasts for that step (no turn, thrust or fire). When the answer arrives it is used on the next step, so a slow agent still plays, on stale information, and gets the new state once it has answered. If both agents answer quickly, the game runs faster than real time.

The higher score (kills plus control-zone points) wins at the buzzer, and a team wiped out with no respawns left loses at once. An entry that fails to get ready (`prepareToPlayAs` raises, or takes over 30 s) forfeits the match. A command for a ship on the other team, an unknown ship or a non-finite turn is rejected and counted; the rest of the action still applies.

### Settings drawn per match

The competitive rules (`Config.competitive()`), except for these, drawn uniformly per match (`astroids_gym/competition/params.py`). Both sides of a pairing play the same seed, so they get the same settings and map.

| Setting | Range | Default |
|---|---|---|
| Asteroids at the start (`asteroid_count`) | 2 to 8 (the field never drops below half of that) | 7 |
| Respawns per team (`team_lives`) | 15 to 25 | 20 |
| Control zone radius (`zone_radius`) | 100 to 160 px | 130 px |
| Bullet life (`bullet_life`) | 0.7 to 0.9 s, a gun range of 336 to 432 px | 0.8 s |
| Flight physics (`flight`) | Off or on, half the matches each. On, ships fly like aircraft (`Config.dogfight()`): the velocity follows the nose, the turn rate depends on speed and turning bleeds speed | Off |

An agent learns the settings of each match from the `params` passed to `prepareToPlayAs`.

## The agent interface

```python
from astroids_gym.competition import Player, ship_command

class MyAgent(Player):            # Player keeps self.player ("blue"/"red"), self.team, self.params, self.config
    def get_agent_type(self) -> str:
        return "my-agent"

    def get_action(self, state: dict) -> dict:
        sim = self.simulation(state)       # the state as a Simulation: helpers, and a forward model
        commands = []
        for i in self.ship_ids:            # this team's ship ids
            s = sim.ships[i]
            if s.alive:
                commands.append(ship_command(i, turn=0.5, thrust=True, fire=False))
        return {"ships": commands}
```

`prepare_to_play_as(player, params, opponent)` is called before each match and `process_game_over(final_state)` after it; `Player` implements both. [`entry/my_agent.py`](../entry/my_agent.py) is a complete example: each ship leads its shots on the nearest enemy and holds the zone when nobody is near, then the agent plays its plan forward a third of a second on the forward model and changes the throttle of any ship that would die. `self.simulation(state)` rebuilds the battle exactly. Only random events (respawn points, asteroid fragments) can differ from the real match, and the opponent's moves are up to you to guess.

### The JSON

Agents in other languages work with plain JSON (`astroids_gym/competition/state.py` has the full description).

* **`params`**: every `Config` field (`width`, `height`, `team_size`, `turn_rate`, `thrust`, `max_speed`, `bullet_speed`, `bullet_life`, `zone_radius`, `team_lives`, `flight`, and so on) plus `frame_skip` (2 physics ticks per decision) and `decision_ms` (33).
* **`state`**: `tick`, `time`, and per team `score` (kills), `points` (zone), `lives` (respawns left) and `reinforcements`; then `ships` (`id`, `team`, `alive`, `x`, `y`, `vx`, `vy`, `angle`, `hp`, `cooldown`, `invuln`, `respawn`, `heat`, `overheated`, `thrusting`, `kills`, `deaths`), `bullets` (`x`, `y`, `vx`, `vy`, `life`, `team`, `owner`) and `asteroids` (`x`, `y`, `vx`, `vy`, `r`, `size`, `hp`, `angle`, `spin`). Pixels, px/s and radians; x right, y down; the world wraps at the edges. Ships 0 to 4 are blue and 5 to 9 red.
* **action**: `{"ships": [{"id": 3, "turn": -0.25, "thrust": true, "fire": false}, ...]}`, where `turn` is in [-1, 1] (a fraction of the turn rate, positive clockwise on screen) and the command is held for the whole step. Ships without a command coast.

### The wire format

The Planet Wars RTS JSON remote-invocation protocol (`protocol.py`), on any WebSocket path (`/ws` by convention). Each request gets one response, in order:

```
→ {"requestType": "init", "target": "agent", "className": "x", "method": "<ignored>"}
← {"status": "ok", "result": {"objectId": "a1b2"}}
→ {"requestType": "invoke", "target": "agent", "objectId": "a1b2", "method": "prepareToPlayAs", "args": ["red", {params}, "bots"]}
← {"status": "ok", "result": "my-agent"}
→ {"requestType": "invoke", "target": "agent", "objectId": "a1b2", "method": "getAction", "args": [{state}]}
← {"status": "ok", "result": {"ships": [...]}}
→ {"requestType": "invoke", "target": "agent", "objectId": "a1b2", "method": "processGameOver", "args": [{state}]}
→ {"requestType": "end", "target": "agent", "objectId": "a1b2", "method": ""}
```

Errors come back as `{"status": "error", "error": "..."}`. Each `init` makes a fresh agent, and a server should accept several connections at once, as the league may play several matches against it in parallel.

## Run it locally

```bash
pip install -e ".[competition,fast]"
python -m astroids_gym.competition match entry.my_agent:MyAgent bots --realtime        # one match
python -m astroids_gym.competition evaluate entry.my_agent:MyAgent --games 10        # vs the baselines
python -m astroids_gym.competition serve competition/models/m10.npz --port 8080      # an agent server
python -m astroids_gym.competition league bots m10=competition/models/m10.npz mine=ws://localhost:8080
```

An agent is a spec string everywhere: `bots` or `bots:<style>` (any of the bot styles), `idle`, `random`, a model (`.npz`, or an SB3 `.zip` with stable-baselines3 installed), `ws://host:port`, or `package.module:Class`. In a league, `docker:<dir>` builds and runs the entry in `<dir>`, and `image:<name>` runs a built image. `--turn-based` waits for every answer instead of playing in real time, so results depend only on the agents and the seed.

`competition/models/` holds the trained teams from the main README, exported to NumPy so they play without torch: `m10` (Colab run 5a, the strongest), `run3_m16`, `colab_m08`, `attention`, `population_m16`, `league_striker` and `dogfight_m17` (trained with flight physics).

## Build an entry

1. Fork this repository and edit [`entry/my_agent.py`](../entry/my_agent.py), or add your own module.
2. The top-level [`Dockerfile`](../Dockerfile) serves `entry.my_agent:MyAgent` on port 8080. Point it at your agent with `--build-arg AGENT=...` or by changing the `AGENT` default. A trained model works too: `AGENT=competition/models/m10.npz`.
3. Test it the way the evaluation will:
   ```bash
   docker build -t my-entry .
   docker run --rm -p 8080:8080 my-entry
   python -m astroids_gym.competition evaluate ws://localhost:8080 --games 10
   ```

Any language works: the image only has to serve the wire format on port 8080. It must not need the internet once it is built (it runs on a network without outside access), and it gets 1 CPU and 2 GB of memory.

## Submit

Open an issue in this repository with the **Competition entry** template, or with a YAML block like this:

````
```yaml
id: my-team
repo_url: https://github.com/<you>/astorids/commit/<full commit hash>
```
````

`id` is 1 to 48 lower-case letters, digits, `-` or `_`; `repo_url` must link to a specific commit of a public GitHub repository with a `Dockerfile` at the top. A maintainer adds the `evaluate` label, and a workflow (`.github/workflows/evaluate-entry.yml`) builds the image, runs it with no internet access, plays 10 real-time matches against each baseline (half on each side) and posts the results on the issue. Entries that build and play join the league.

The baselines are the default bots, the finger-four and pincer formation bots, the league striker, population m16 and m10.

## The league

```bash
python -m astroids_gym.competition league --entries-file competition/entries.json --games-per-pair 10 \
    --workers 3 --results competition/league_results.jsonl --leaderboard competition/leaderboard.md
```

Every pair plays `--games-per-pair` matches, half on each side, each pair of sides on the same seed. Ratings are TrueSkill (every entry starts at μ = 25, σ = 8.33; tested against the reference implementation), and the leaderboard ranks by μ − 3σ, with each entry's record, average margin, decision time and share of late steps, and a head-to-head table. Results go to a JSON-lines file as matches finish, so rerunning only plays what's missing: adding an entry plays just its matches. Keep `--workers` below the number of cores, or the real-time budget starts to bite.
