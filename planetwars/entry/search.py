"""The neural agent with a short look-ahead on top.

Each turn the network proposes its top few moves. Every proposal is played on a copy of the game with
the official rules, both sides then follow the greedy sample agent for a few dozen ticks (cheap, a
fraction of a millisecond per tick), and the network's value estimate scores the position that
results. The best-scoring move is played. If time runs short the search stops and plays the best move
found so far, which is the network's own first choice when nothing else has been scored yet.
"""
import copy
import os
import time

import numpy as np

import pwnet
from agent import OWNER, NeuralAgent, arrays, load
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from core.forward_model import ForwardModel
from core.game_state import Action, Player

TOP_MOVES = int(os.environ.get("PW_TOP", "8"))
HORIZON = int(os.environ.get("PW_HORIZON", "40"))
BUDGET_MS = float(os.environ.get("PW_BUDGET_MS", "110"))
ROLLOUT = os.environ.get("PW_ROLLOUT", "greedy")  # what both sides do after the first move: greedy or noop
PRIOR = float(os.environ.get("PW_PRIOR", "0.02"))  # bonus for the network's own choice


class SearchAgent(NeuralAgent):
    def prepare_to_play_as(self, player, params, opponent=None):
        out = super().prepare_to_play_as(player, params, opponent)
        self.rollout = {p: GreedyHeuristicAgent() for p in (Player.Player1, Player.Player2)}
        for p, agent in self.rollout.items():
            agent.prepare_to_play_as(p, params)
        return out

    def get_action(self, game_state):
        start = time.perf_counter()
        try:
            if self.player not in (Player.Player1, Player.Player2) or not game_state.planets:
                return Action.do_nothing()
            params, heads = load()
            me = OWNER[self.player]
            s = arrays(game_state, self.params.transporter_speed, self.params.max_ticks)
            planet, glob, pair, aux = pwnet.features(np, s, me)
            flat, amount_logits, _ = pwnet.forward(np, params, heads, planet, glob, pair, s.valid, aux["pair_ok"])
            n = s.ships.shape[0]
            top = int(np.argmax(flat))
            pair_logits = flat[:n * n].reshape(n, n)

            def move(i, j):
                ships = min(float(aux["amounts"][i, j, int(np.argmax(amount_logits[i, j]))]),
                            game_state.planets[i].n_ships)
                if not ships > 0:
                    return None
                return Action(player_id=self.player, source_planet_id=i, destination_planet_id=j, num_ships=ships)

            # candidates: the network's own choice first, then each free planet of ours aimed at the
            # target the network likes best for it, the greedy agent's move, and doing nothing
            moves = []
            if top >= n * n:
                moves.append((Action.do_nothing(), True))
            else:
                m = move(*divmod(top, n))
                moves.append((m if m is not None else Action.do_nothing(), True))
            sources = [i for i in range(n) if s.owner[i] == me and not s.tr_on[i] and s.ships[i] > 0]
            sources.sort(key=lambda i: -pair_logits[i].max())
            for i in sources[:TOP_MOVES]:
                m = move(i, int(np.argmax(pair_logits[i])))
                if m is not None:
                    moves.append((m, False))
            g = self.rollout[self.player].get_action(game_state)
            if g is not Action.DO_NOTHING:
                moves.append((g, False))
            moves.append((Action.do_nothing(), False))
            seen, unique = set(), []
            for action, is_top in moves:
                key = (action.source_planet_id, action.destination_planet_id, round(float(action.num_ships), 3)) \
                    if action is not Action.DO_NOTHING else None
                if key not in seen:
                    seen.add(key)
                    unique.append((action, is_top))
            moves = unique
            best, best_score = None, -np.inf
            for action, is_top in moves:
                if best is not None and (time.perf_counter() - start) * 1000 > BUDGET_MS:
                    break
                score = self.evaluate(game_state, action, params, heads) + (PRIOR if is_top else 0.0)
                if score > best_score:
                    best, best_score = action, score
            return best if best is not None else Action.do_nothing()
        except Exception as e:  # never forfeit a tick over a bug
            print("SearchAgent error:", e, flush=True)
            return super().get_action(game_state)

    def evaluate(self, game_state, action, params, heads):
        """Value for us after playing `action` and then HORIZON ticks of greedy play on both sides."""
        gs = copy.deepcopy(game_state)
        model = ForwardModel(gs, self.params)
        me, opp = self.player, self.player.opponent()
        first = {me: action, opp: self.rollout[opp].get_action(gs)}
        model.step(first)
        for _ in range(HORIZON - 1):
            if model.is_terminal():
                break
            if ROLLOUT == "noop":
                model.step({})
            else:
                model.step({me: self.rollout[me].get_action(gs), opp: self.rollout[opp].get_action(gs)})
        if model.is_terminal():
            leader = model.get_leader()
            return 1.0 if leader == me else (-1.0 if leader == opp else 0.0)
        s = arrays(gs, self.params.transporter_speed, self.params.max_ticks)
        planet, glob, pair, aux = pwnet.features(np, s, OWNER[me])
        return float(pwnet.forward(np, params, heads, planet, glob, pair, s.valid, aux["pair_ok"])[2])

    def get_agent_type(self) -> str:
        return "pwjax neural agent with look-ahead"
