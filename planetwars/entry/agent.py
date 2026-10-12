"""The neural Planet Wars agent: a policy trained with JAX self-play, run here with numpy."""
import os
from types import SimpleNamespace

import numpy as np

import pwnet
from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import Action, GameState, Player

MODEL = os.environ.get("PW_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz"))
OWNER = {Player.Neutral: 0, Player.Player1: 1, Player.Player2: 2}
_cache = {}


def load(path=MODEL):
    if path not in _cache:
        z = np.load(path)
        heads = int(z["cfg.heads"]) if "cfg.heads" in z.files else 4
        _cache[path] = ({k: z[k].astype(np.float64) for k in z.files if not k.startswith("cfg.")}, heads)
    return _cache[path]


def arrays(gs: GameState, speed: float, max_ticks: int) -> SimpleNamespace:
    """The game as the arrays pwnet expects (no padding needed here)."""
    n = len(gs.planets)
    s = SimpleNamespace(
        valid=np.ones(n, bool), owner=np.zeros(n, np.int64), ships=np.zeros(n), pos=np.zeros((n, 2)),
        growth=np.zeros(n), radius=np.zeros(n), tr_on=np.zeros(n, bool), tr_owner=np.zeros(n, np.int64),
        tr_dest=np.zeros(n, np.int64), tr_ships=np.zeros(n), tr_pos=np.zeros((n, 2)), tr_vel=np.zeros((n, 2)),
        tick=float(gs.game_tick), speed=float(speed), max_ticks=float(max_ticks))
    for i, p in enumerate(gs.planets):
        s.owner[i] = OWNER[p.owner]
        s.ships[i], s.growth[i], s.radius[i] = p.n_ships, p.growth_rate, p.radius
        s.pos[i] = p.position.x, p.position.y
        t = p.transporter
        if t is not None:
            s.tr_on[i] = True
            s.tr_owner[i] = OWNER[t.owner]
            s.tr_dest[i] = min(max(t.destination_index, 0), n - 1)
            s.tr_ships[i] = t.n_ships
            s.tr_pos[i] = t.s.x, t.s.y
            s.tr_vel[i] = t.v.x, t.v.y
    return s


def choose(params, heads, s, me):
    """Greedy (argmax) action as (src, dst, ships); src = -1 means do nothing."""
    planet, glob, pair, aux = pwnet.features(np, s, me)
    flat, amount_logits, _ = pwnet.forward(np, params, heads, planet, glob, pair, s.valid, aux["pair_ok"])
    idx = int(np.argmax(flat))
    n = s.ships.shape[0]
    if idx >= n * n:
        return -1, -1, 0.0
    i, j = divmod(idx, n)
    a = int(np.argmax(amount_logits[i, j]))
    return i, j, float(aux["amounts"][i, j, a])


class NeuralAgent(PlanetWarsPlayer):
    def get_action(self, game_state: GameState) -> Action:
        try:
            if self.player not in (Player.Player1, Player.Player2) or not game_state.planets:
                return Action.do_nothing()
            params, heads = load()
            s = arrays(game_state, self.params.transporter_speed, self.params.max_ticks)
            src, dst, ships = choose(params, heads, s, OWNER[self.player])
            if src < 0:
                return Action.do_nothing()
            ships = min(ships, game_state.planets[src].n_ships)
            if not ships > 0:
                return Action.do_nothing()
            return Action(player_id=self.player, source_planet_id=src, destination_planet_id=dst, num_ships=ships)
        except Exception as e:  # never forfeit a tick over a bug
            print("NeuralAgent error:", e, flush=True)
            return Action.do_nothing()

    def get_agent_type(self) -> str:
        return "pwjax neural agent"
