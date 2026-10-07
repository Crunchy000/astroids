"""Stable-Baselines3 policies (needs the ``train`` extra: ``pip install -e ".[train]"``).

``CentralCriticPolicy`` is a MAPPO-style actor-critic. The observation is a
ship's own observation followed by the global battle state
(``obs.global_state``). The actor reads only the ship's own observation,
exactly like an ``MlpPolicy`` would, so it still flies from what one ship can
see and exports to NumPy unchanged. The critic, which is only used during
training, also reads the global state: every ship, both teams' respawns and
zone points, the clock. A critic that sees the whole battle gives better value
estimates, so less noisy advantages for the actor to learn from.
"""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor


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
