from stable_baselines3.ppo.policies import ActorCriticPolicy
import torch

class MaskedMlpPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        mask = obs["action_mask"]
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        logits = distribution.distribution.logits
        masked_logits = logits + (1 - mask) * -1e9
        distribution.distribution = torch.distributions.Categorical(logits=masked_logits)

        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_probs