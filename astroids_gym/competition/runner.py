"""Play one match between two agents, real time or turn by turn.

Every decision step (30 per second of game time) the runner sends both agents
the current state and waits for their actions:

* **Real time** (``realtime=True``, how the league plays): each agent has
  ``params["decision_ms"]`` (33 ms) to answer. A team whose agent hasn't
  answered coasts for that step (no turn, thrust or fire). Its answer, when it
  comes, is used on the next step, so a slow agent acts on stale information,
  and it gets the new state once it has answered. If both answer quickly the
  game runs faster than real time. This follows Planet Wars RTS.
* **Turn by turn** (the default): the runner waits for both answers every step,
  so results depend only on the agents and the seed. Good for testing.

An agent that fails to get ready (``prepareToPlayAs`` raises or takes longer
than ``setup_timeout``) forfeits the match. Errors and malformed actions later
on only cost that step.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, wait
from dataclasses import asdict, dataclass, field

from ..sim import Simulation
from .agent import DEFAULT_OPPONENT, Agent
from .params import random_params
from .state import DECISION_MS, FRAME_SKIP, TEAMS, config_from_params, max_steps, parse_action, snapshot


class _Worker:
    """A daemon thread that runs one agent's calls in order (an agent never sees two calls at once)."""

    def __init__(self):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, fn, *args) -> Future:
        f: Future = Future()
        f.sent = time.perf_counter()
        self._q.put((f, fn, args))
        return f

    def _run(self) -> None:
        while (item := self._q.get()) is not None:
            f, fn, args = item
            if not f.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args)
            except BaseException as e:  # noqa: BLE001 - reported to the runner
                f.took = time.perf_counter() - f.sent
                f.set_exception(e)
            else:
                f.took = time.perf_counter() - f.sent
                f.set_result(result)

    def close(self) -> None:
        self._q.put(None)


@dataclass
class SeatStats:
    agent_type: str = ""
    actions: int = 0  # answers used
    late: int = 0  # steps the team coasted waiting for an answer
    errors: int = 0  # calls that raised
    rejected: int = 0  # ship commands rejected as invalid
    total_ms: float = 0.0
    max_ms: float = 0.0
    forfeit: str | None = None

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.actions if self.actions else 0.0


@dataclass
class MatchResult:
    names: dict  # {"blue": name, "red": name}
    winner: str  # "blue", "red" or "draw"
    score: dict  # kills + zone points per team
    kills: dict
    points: dict
    eliminated: bool  # a team was wiped out with no respawns left
    steps: int
    seed: int | None
    realtime: bool
    params: dict
    stats: dict = field(default_factory=dict)  # {"blue": SeatStats, "red": SeatStats}
    wall_seconds: float = 0.0

    def outcome(self, name: str) -> float:
        """1 if ``name`` won, 0.5 for a draw, 0 if it lost."""
        if self.winner == "draw":
            return 0.5
        return 1.0 if self.names[self.winner] == name else 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        for t in TEAMS:
            d["stats"][t]["mean_ms"] = round(self.stats[t].mean_ms, 3)
        return d


def run_match(blue: Agent, red: Agent, params: dict | None = None, seed: int | None = None,
              realtime: bool = False, names: tuple[str, str] | None = None, setup_timeout: float = 30.0,
              step_timeout: float = 60.0) -> MatchResult:
    """One battle, ``blue`` against ``red``. ``params`` default to ``random_params(seed)``.

    ``names`` are what each agent is told its opponent is called (and what the
    result records); by default the agents' own types.
    """
    params = params if params is not None else random_params(seed)
    cfg = config_from_params(params)
    frame_skip = int(params.get("frame_skip", FRAME_SKIP))
    budget = float(params.get("decision_ms", DECISION_MS)) / 1000.0 if realtime else step_timeout
    sim = Simulation(cfg, seed=seed)
    agents = (blue, red)
    workers = [_Worker(), _Worker()]
    stats = {t: SeatStats() for t in TEAMS}
    start = time.perf_counter()
    try:
        # Get ready, both at once.
        given = list(names) if names else [None, None]
        setup = [w.submit(a.prepare_to_play_as, TEAMS[t], params, given[1 - t] or DEFAULT_OPPONENT)
                 for t, (w, a) in enumerate(zip(workers, agents))]
        wait(setup, timeout=setup_timeout)
        for t, f in enumerate(setup):
            st = stats[TEAMS[t]]
            try:
                st.agent_type = str(f.result(timeout=0))
            except Exception as e:  # noqa: BLE001 - timeout or error: forfeit
                st.forfeit = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        names = tuple(given[t] or stats[TEAMS[t]].agent_type or TEAMS[t] for t in range(2))
        forfeits = [stats[t].forfeit is not None for t in TEAMS]
        if any(forfeits):
            winner = "draw" if all(forfeits) else TEAMS[forfeits.index(False)]
            zero = {t: 0 for t in TEAMS}
            return MatchResult(dict(zip(TEAMS, names)), winner, zero, zero, zero, False, 0, seed, realtime, params,
                               stats, time.perf_counter() - start)

        pending: list[Future | None] = [None, None]
        steps, limit = 0, max_steps(params)
        while not sim.over and steps < limit:
            for t in range(2):
                if pending[t] is None:  # an agent still thinking keeps its old request
                    pending[t] = workers[t].submit(agents[t].get_action, snapshot(sim))
            wait(pending, timeout=budget)
            acts = {}
            for t, f in enumerate(pending):
                st = stats[TEAMS[t]]
                if not f.done():
                    st.late += 1
                    continue
                pending[t] = None
                try:
                    cmds, bad = parse_action(f.result(), t, cfg)
                except Exception:  # noqa: BLE001 - the agent raised: coast this step
                    st.errors += 1
                    continue
                acts.update(cmds)
                st.rejected += bad
                st.actions += 1
                ms = f.took * 1000.0
                st.total_ms += ms
                st.max_ms = max(st.max_ms, ms)
            for _ in range(frame_skip):
                sim.step(acts)
            steps += 1

        final = snapshot(sim)
        done = [w.submit(a.process_game_over, final) for w, a in zip(workers, agents)]
        wait(done, timeout=5.0)
        w = sim.winner()
        return MatchResult(
            names=dict(zip(TEAMS, names)), winner="draw" if w < 0 else TEAMS[w],
            score={t: sim.team_score(i) for i, t in enumerate(TEAMS)},
            kills={t: sim.score[i] for i, t in enumerate(TEAMS)},
            points={t: sim.points[i] for i, t in enumerate(TEAMS)},
            eliminated=sim.eliminated(0) or sim.eliminated(1), steps=steps, seed=seed, realtime=realtime,
            params=params, stats=stats, wall_seconds=time.perf_counter() - start)
    finally:
        for w in workers:
            w.close()
