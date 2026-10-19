"""The entry, except that its first launch decision is forced to the K-th best distinct option by the value ranking
(PW_FORCE_RANK=K; 0 is the best, waiting counts as an option). Usage: serve_force.py PORT"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "entry"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import planner as P  # noqa: E402
from agent import OWNER, arrays  # noqa: E402
from client_server.game_agent_server import GameServerAgent  # noqa: E402
from core.game_state import Action  # noqa: E402

K = int(os.environ.get("PW_FORCE_RANK", "0"))


class Forced(P.PlannerAgent):
    def prepare_to_play_as(self, player, params, opponent=None):
        self.forced = False
        return super().prepare_to_play_as(player, params, opponent) if opponent is not None else super().prepare_to_play_as(player, params)

    def get_action(self, game_state):
        if not self.forced and self.player is not None and game_state.planets:
            s = arrays(game_state, self.params.transporter_speed, self.params.max_ticks)
            me = OWNER[self.player]
            c = P.candidates(s, me, self.horizon, self.fractions, None, self.extras)
            if c is not None:
                self.forced = True
                order = np.argsort(-c["score"], kind="stable")[:self.top]
                owners = [c["owner"][k] for k in order] + [c["base_owner"]]
                shipss = [c["ships"][k] for k in order] + [c["base_ships"]]
                hand = np.concatenate([c["score"][order], [c["base"]]])
                tot = P.totals_of(*self.value, s, me, owners, shipss, hand, s.tick + c["horizon"], self.mix)
                keys = np.concatenate([order, [-1]])
                X = np.concatenate([np.array(owners), np.array(shipss)], 1)
                _, first = np.unique(X, axis=0, return_index=True)
                ranked = [i for i in np.argsort(-tot, kind="stable") if i in set(first)]
                k = int(keys[ranked[min(K, len(ranked) - 1)]])
                print(f"FORCE tick={game_state.game_tick} rank={K} move={'wait' if k < 0 else (int(c['ci'][k]), int(c['cj'][k]), float(c['cn'][k]))}", flush=True)
                if k < 0:
                    return Action.do_nothing()
                return Action(player_id=self.player, source_planet_id=int(c["ci"][k]), destination_planet_id=int(c["cj"][k]),
                              num_ships=float(c["cn"][k]))
        return super().get_action(game_state)


if __name__ == "__main__":
    asyncio.run(GameServerAgent(host="127.0.0.1", port=int(sys.argv[1]), agent=Forced()).start())
