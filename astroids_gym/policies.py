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
"""

from __future__ import annotations

import math

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
