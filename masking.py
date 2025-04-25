from stable_baselines3.ppo.policies import MlpPolicy
from torch import nn
import torch

class MaskedMlpPolicy(MlpPolicy):
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(*self.mlp_extractor(features))

        if hasattr(obs, 'info') and 'action_mask' in obs.info:
            mask = obs.info['aciton_mask']
            logits = distribution.distribution.logits
            masked_logits = logits + (1 - torch.tensor(mask)) * -1e9
            distribution.distribution = torch.distributions.Categorical(logits=masked_logits)

        return distribution.get_actions(deterministic=deterministic)