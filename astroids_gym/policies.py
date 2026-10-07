"""Stable-Baselines3 policies (needs the ``train`` extra: ``pip install -e ".[train]"``).

``CentralCriticPolicy`` is a MAPPO-style actor-critic. The observation is a
ship's own observation followed by the global battle state
(``obs.global_state``). The actor reads only the ship's own observation,
exactly like an ``MlpPolicy`` would, so it still flies from what one ship can
see and exports to NumPy unchanged. The critic, which is only used during
training, also reads the global state: every ship, both teams' respawns and
zone points, the clock. A critic that sees the whole battle gives better value
estimates, so less noisy advantages for the actor to learn from.

``EntityPolicy`` is the same with an attention-over-entities actor
(``EntityExtractor``, see ``astroids_gym.entity``) instead of a flat MLP input.

``PolicyBank`` runs frozen (exported) policies on a GPU for opponents and tournaments.
"""

from __future__ import annotations

import math

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from torch import nn

from .entity import context_slices, entity_groups
from .obs import ObsConfig


class FirstInputs(BaseFeaturesExtractor):
    """The first ``n`` observation inputs (a ship's own observation)."""

    def __init__(self, observation_space: spaces.Box, n: int):
        super().__init__(observation_space, n)
        self.n = n

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[:, :self.n]


class CentralCriticPolicy(ActorCriticPolicy):
    """Actor on the first ``n_actor_inputs`` inputs, critic on all of them."""

    def __init__(self, *args, n_actor_inputs: int, **kwargs):
        self.n_actor_inputs = n_actor_inputs
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.pi_features_extractor = FirstInputs(self.observation_space, self.n_actor_inputs)
        self.mlp_extractor = MlpExtractor(self.n_actor_inputs, net_arch=self.net_arch,
                                          activation_fn=self.activation_fn, device=self.device)
        full = MlpExtractor(self.vf_features_extractor.features_dim, net_arch=self.net_arch,
                            activation_fn=self.activation_fn, device=self.device)
        self.mlp_extractor.value_net = full.value_net

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["n_actor_inputs"] = self.n_actor_inputs
        data.pop("share_features_extractor", None)
        return data


class EntityExtractor(BaseFeaturesExtractor):
    """Tokens for the ship itself and every entity slot, one transformer layer, then
    [own token, mean of present entity tokens] (``2 * d`` features).

    Mirrored in NumPy by ``entity.entity_features`` (keep the two in step).
    """

    def __init__(self, observation_space: spaces.Box, d: int = 64, heads: int = 4, extended: bool = True):
        super().__init__(observation_space, 2 * d)
        self.oc = ObsConfig(extended=extended)
        self.groups = entity_groups(self.oc)
        self.ctx_slices = context_slices(self.oc)
        self.heads = heads
        self.ctx = nn.Linear(sum(b - a for a, b in self.ctx_slices), d)
        self.emb = nn.ModuleDict({name: nn.Linear(dim, d) for name, _, _, dim in self.groups})
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        n = obs.shape[0]
        ctx = th.cat([obs[:, a:b] for a, b in self.ctx_slices], dim=1)
        toks, masks = [self.ctx(ctx).unsqueeze(1)], [th.ones(n, 1, dtype=th.bool, device=obs.device)]
        for name, start, count, dim in self.groups:
            x = obs[:, start:start + count * dim].reshape(n, count, dim)
            toks.append(self.emb[name](x))
            masks.append(x[:, :, 0] > 0.5)
        x, mask = th.cat(toks, dim=1), th.cat(masks, dim=1)
        t, d = x.shape[1], x.shape[2]
        dh = d // self.heads

        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.reshape(n, t, self.heads, dh).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(n, t, d)
        x = x + self.proj(out)
        x = x + self.ff2(th.relu(self.ff1(self.ln2(x))))

        m = mask[:, 1:, None].float()
        pooled = (x[:, 1:] * m).sum(1) / m.sum(1).clamp(min=1.0)
        return th.cat([x[:, 0], pooled], dim=1)


class EntityPolicy(CentralCriticPolicy):
    """Attention-over-entities actor (on the first ``n_actor_inputs`` inputs); MLP critic on all inputs."""

    def __init__(self, *args, entity_kwargs: dict | None = None, **kwargs):
        self.entity_kwargs = dict(entity_kwargs or {})
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        actor_space = spaces.Box(-1.0, 1.0, (self.n_actor_inputs,), self.observation_space.dtype)
        self.pi_features_extractor = EntityExtractor(actor_space, **self.entity_kwargs)
        self.mlp_extractor = MlpExtractor(self.pi_features_extractor.features_dim, net_arch=self.net_arch,
                                          activation_fn=self.activation_fn, device=self.device)
        full = MlpExtractor(self.vf_features_extractor.features_dim, net_arch=self.net_arch,
                            activation_fn=self.activation_fn, device=self.device)
        self.mlp_extractor.value_net = full.value_net

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["entity_kwargs"] = self.entity_kwargs
        return data


class FrozenPolicy(nn.Module):
    """A NumPy export (``league.export_policy``) as a torch module, for batched inference on a GPU."""

    def __init__(self, path: str):
        super().__init__()
        d = np.load(path)
        self.fx = None
        if "entity" in d:
            heads, extended = (int(x) for x in d["entity"])
            oc = ObsConfig(extended=bool(extended))
            self.fx = EntityExtractor(spaces.Box(-1.0, 1.0, (oc.size,), np.float32), d=d["fx.ctx.weight"].shape[0],
                                      heads=heads, extended=bool(extended))
            self.fx.load_state_dict({k[3:]: th.as_tensor(d[k]) for k in d.files if k.startswith("fx.")})
        if "wo" in d:
            n = sum(1 for k in d.files if k.startswith("w") and k != "wo")
            layers, out = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)], (d["wo"], d["bo"])
        else:  # original two-hidden-layer files
            layers, out = [(d["w0"], d["b0"]), (d["w1"], d["b1"])], (d["w2"], d["b2"])
        self.hidden = nn.ModuleList([self._linear(w, b) for w, b in layers])
        self.out = self._linear(*out)

    @staticmethod
    def _linear(w, b) -> nn.Linear:  # exports store weights as (in, out)
        lin = nn.Linear(w.shape[0], w.shape[1])
        lin.weight.data = th.as_tensor(np.ascontiguousarray(w.T))
        lin.bias.data = th.as_tensor(b)
        return lin

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = obs if self.fx is None else self.fx(obs)
        for lin in self.hidden:
            h = th.tanh(lin(h))
        return self.out(h)


class PolicyBank:
    """Frozen opponents and tournament players on one device (e.g. a GPU), loaded on first use.

    With a GPU, running every network in the main process is much faster than
    NumPy in the CPU worker processes (an attention opponent takes ~8 ms per
    40-ship batch in NumPy): workers then only simulate battles. Used by
    ``LeagueVecEnv(bank=...)`` and ``league.play_matches(..., bank=...)``.
    """

    def __init__(self, device: str = "cuda", max_cached: int = 256):
        self.device = th.device(device)
        self.max_cached = max_cached
        self._cache: dict[str, FrozenPolicy] = {}

    def _get(self, spec: str) -> FrozenPolicy:
        if spec not in self._cache:
            if len(self._cache) >= self.max_cached:
                self._cache.pop(next(iter(self._cache)))
            self._cache[spec] = FrozenPolicy(spec).to(self.device).eval()
        else:  # keep recently used policies at the end
            self._cache[spec] = self._cache.pop(spec)
        return self._cache[spec]

    @th.no_grad()
    def logits(self, spec: str, obs: np.ndarray) -> np.ndarray:
        return self._get(spec)(th.as_tensor(obs, dtype=th.float32, device=self.device)).cpu().numpy()

    @th.no_grad()
    def act(self, spec: str, obs: np.ndarray, sample: bool) -> np.ndarray:
        """Actions [turn, thrust, fire] per row: sampled, or the most likely (as ``NumpyPolicy`` without rng)."""
        from . import ACTION_SPACE
        logits = self._get(spec)(th.as_tensor(obs, dtype=th.float32, device=self.device))
        parts = logits.split([int(n) for n in ACTION_SPACE.nvec], dim=1)
        if sample:
            acts = [th.distributions.Categorical(logits=p).sample() for p in parts]
        else:
            acts = [p.argmax(dim=1) for p in parts]
        return th.stack(acts, dim=1).cpu().numpy()

    def act_grouped(self, requests: list[dict[str, np.ndarray]], sample: bool) -> list[dict[str, np.ndarray]]:
        """``requests[w]`` maps spec -> observations from worker ``w``; one batched pass per spec over all workers."""
        out: list[dict[str, np.ndarray]] = [{} for _ in requests]
        specs = {s for req in requests for s in req}
        for spec in specs:
            parts = [(w, req[spec]) for w, req in enumerate(requests) if spec in req]
            acts = self.act(spec, np.concatenate([o for _, o in parts]), sample)
            i = 0
            for w, o in parts:
                out[w][spec] = acts[i:i + len(o)]
                i += len(o)
        return out
