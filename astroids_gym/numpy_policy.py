"""A PPO policy network exported to NumPy (``.npz``), so it plays without torch.

``export_policy`` writes the files from a trained SB3 model: an MLP actor of
any depth, or an attention actor (entity extractor, then the MLP head).
Frozen opponents, tournaments and competition agents load them with
``NumpyPolicy`` instead of SB3's ``predict``. Nothing here imports torch.
"""

from __future__ import annotations

import numpy as np

from .entity import entity_features
from .env import ACTION_SPACE
from .obs import ObsConfig

SPLITS = np.cumsum(ACTION_SPACE.nvec)[:-1]  # logits layout [5 | 2 | 2]


def _hidden_keys(sd, net="policy_net") -> list[str]:
    """Weight keys of the Linear layers in an SB3 MLP extractor, in order (any depth)."""
    idx = sorted(int(k.split(".")[2]) for k in sd if k.startswith(f"mlp_extractor.{net}.") and k.endswith(".weight"))
    return [f"mlp_extractor.{net}.{i}" for i in idx]


def export_policy(model, path: str) -> str:
    """Save an SB3 PPO policy's actor network (any depth/width) as a ``.npz`` for ``NumpyPolicy``.

    With a centralised critic the actor only reads the first ``n_actor_inputs``
    observation inputs (the rest is the global state, for the critic), so the
    NumPy policy takes the plain observation. An attention actor
    (``EntityPolicy``) also exports its entity extractor (keys ``fx.*``).
    """
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    n_in = getattr(model.policy, "n_actor_inputs", None)
    arrays = {}
    ent = getattr(model.policy, "entity_kwargs", None)
    if ent is not None:
        fx = model.policy.pi_features_extractor
        arrays["entity"] = np.array([fx.heads, int(fx.oc.extended)])
        arrays |= {"fx." + k[len("pi_features_extractor."):]: v for k, v in sd.items()
                   if k.startswith("pi_features_extractor.")}
        n_in = None
    for i, k in enumerate(_hidden_keys(sd)):
        w = sd[k + ".weight"].T
        if i == 0 and n_in is not None:
            w = w[:n_in]
        arrays[f"w{i}"], arrays[f"b{i}"] = w, sd[k + ".bias"]
    arrays["wo"], arrays["bo"] = sd["action_net.weight"].T, sd["action_net.bias"]
    np.savez(path, **arrays)
    return path


class NumpyPolicy:
    def __init__(self, path: str):
        d = np.load(path)
        self.entity = None
        if "entity" in d:  # attention actor: entity extractor, then the MLP head
            heads, extended = (int(x) for x in d["entity"])
            self.entity = ({k[3:]: d[k] for k in d.files if k.startswith("fx.")}, ObsConfig(extended=bool(extended)),
                           heads)
        if "wo" in d:  # any depth
            n = sum(1 for k in d.files if k.startswith("w") and k != "wo")
            self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
            self.out = (d["wo"], d["bo"])
        else:  # original two-hidden-layer files
            self.layers = [(d["w0"], d["b0"]), (d["w1"], d["b1"])]
            self.out = (d["w2"], d["b2"])

    @property
    def obs_config(self) -> ObsConfig:
        """The observation layout this network reads (competitive models use the extended one)."""
        if self.entity is not None:
            return self.entity[1]
        return ObsConfig(extended=self.layers[0][0].shape[0] == ObsConfig(extended=True).size)

    @property
    def widths(self) -> list[int]:
        return [w.shape[1] for w, _ in self.layers]

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """Action logits for a batch of observations, laid out [turn 5 | thrust 2 | fire 2]."""
        h = obs if self.entity is None else entity_features(self.entity[0], obs, self.entity[1], self.entity[2])
        for w, b in self.layers:
            h = np.tanh(h @ w + b)
        return h @ self.out[0] + self.out[1]

    def __call__(self, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Actions for a batch of observations: argmax, or sampled when ``rng`` is given."""
        acts = []
        for part in np.split(self.logits(obs), SPLITS, axis=1):
            if rng is not None:  # Gumbel-max sampling == categorical sampling
                part = part - np.log(-np.log(rng.random(part.shape)))
            acts.append(part.argmax(axis=1))
        return np.stack(acts, axis=1)
