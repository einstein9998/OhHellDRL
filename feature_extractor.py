from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym

class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        self.net = nn.Sequential(
            nn.Linear(observation_space['obs'].shape[0], 128),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations["obs"])